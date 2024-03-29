# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import numpy as np
from tqdm import tqdm
import utool as ut
import vtool as vt
import logging
import os

import torch
import torchvision.transforms as T

from wbia import dtool as dt
from wbia.control import controller_inject
from wbia.constants import ANNOTATION_TABLE

from wbia_segmentation.utils.utils import merge_from_file, load_pretrained_weights, load_hf_model, overlay_seg_mask, apply_seg_mask
from wbia_segmentation.default_config import get_default_config
from wbia_segmentation.models import get_model
from wbia_segmentation.data.dataset import InferenceSegDataset

(print, rrr, profile) = ut.inject2(__name__)

_, register_ibs_method = controller_inject.make_ibs_register_decorator(__name__)

register_api = controller_inject.get_wbia_flask_api(__name__)
register_route = controller_inject.get_wbia_flask_route(__name__)

register_preproc_image = controller_inject.register_preprocs['image']
register_preproc_annot = controller_inject.register_preprocs['annot']

DEMOS = {}

CONFIGS = {
    "snowleopard": "https://wildbookiarepository.azureedge.net/models/segmentation.snow_leopard.20230311.yaml"
}

MODELS = {
    "snowleopard": "https://wildbookiarepository.azureedge.net/models/segmentation.segformerb2.snow_leopard.model.20230311.zip"
}


class SegmentationConfig(dt.Config):  # NOQA
    _param_info_list = [
        ut.ParamInfo('config_path', default=None),
    ]


@register_ibs_method
def register_segmentations(ibs, aid_list, config=None):
    r"""
    Predict binary segmentation mask for a specified species (0 background, 1 foreground)
    Args:
        ibs (WBIAController):  wbia controller object
        aid_list (int): annot ids specifying the input
        config (SegmentationConfig): Config object specifying params file to run segmentation model
    Returns:
        list of 2D binary numpy arrays: list of binary segmentation masks
    Example:
        >>> # ENABLE_DOCTEST
        >>> aid_list = ibs.get_valid_aids()
        >>> seg_mask_list = ibs.register_segmentations(aid_list, "snowleopard")
    """
    seg_mask_list = ibs._compute_segmentations(aid_list, config)

    return seg_mask_list

"""
Schema:
    'segmentationmask_rowid': 'integer primary key'
    'annotations_rowid':  'integer not null'
    'config_rowid': 'integer default 0'
    'seg_mask': 'ndarray'
"""
SegImgType = dt.ExternType(
    ut.partial(vt.imread, grayscale=True), vt.imwrite, extern_ext='.png'
)

@register_preproc_annot(
    tablename='SegmentationMask',
    parents=[ANNOTATION_TABLE],
    colnames=['seg_mask'],
    coltypes=[SegImgType],
    configclass=SegmentationConfig,
    fname='segmentation',
    chunksize=128,
)
@register_ibs_method
def register_segmentations_depc(depc, aid_list, config=None):
    r"""
    Predict binary segmentation mask for a specified species (0 background, 1 foreground)
    Args:
        ibs (WBIAController):  wbia controller object
        aid_list (int): annot ids specifying the input
        config (SegmentationConfig): Config object specifying params file to run segmentation model
    Returns:
        list of 2D binary numpy arrays: list of binary segmentation masks
    Example:
        >>> # ENABLE_DOCTEST
        >>> aid_list = ibs.get_valid_aids()
        >>> seg_mask_list = ibs.depc_annot.get('SegmentationMask', aid_list, 'seg_mask', {"config_path": 'snowleopard'})
    """
    ibs = depc.controller
    seg_masks = _compute_segmentations(ibs, aid_list, config['config_path'])
    for aid, mask in zip(aid_list, seg_masks):
        yield (np.array(mask),)


@register_ibs_method
def _compute_segmentations(ibs, aid_list, config=None, multithread=False):
    r"""
    Load config, data, model and predict segmentation masks
    Args:
        ibs (WBIAController):  wbia controller object
        aid_list (int): annot ids specifying the input
        config (str): URL to config file
    Returns:
        list of 2D binary numpy arrays: list of binary segmentation masks
    """
    # Get species from the first annotation
    #species = ibs.get_annot_species_texts(aid_list[0])

    # Load config
    if config is not None and config in CONFIGS:
        config_url = CONFIGS[config]
        cfg = _load_config(config_url)
    else:
        cfg = _load_config()

    # Load model
    if config is not None and config in MODELS:
        model_url = MODELS[config]
    else:
        model_url = None
    model = _load_model(cfg, model_url)

    # Create data loader with proper transformations
    test_loader, _ = _load_data(ibs, aid_list, cfg, multithread)

    # Compute segmentation masks
    model.eval()
    seg_mask_list = []

    with torch.no_grad():
        for images, names, image_sizes in tqdm(test_loader):
            images = images.to(cfg.device)
            output = model.predict(images.float())
            seg_masks = output.argmax(dim=1).detach().cpu()
            
            images = images.cpu()
            for i in range(images.shape[0]):
                im = images[0]
                im = im.permute(1, 2, 0)
                mask = seg_masks[0]
                
                # Apply mask to image (background is assigned value 0)
                overlayed_im = apply_seg_mask(im, mask)
                seg_mask_list.append(mask.numpy())
    
    return seg_mask_list


@register_ibs_method
def _render_segmentations(ibs, aid_list, config=None, multithread=False):
    r"""
    Load config, data, model, predict segmentation masks and save
    mask overlayed with original image for visualization purposes
    into local home folder as calculated in `masks_savedir` variable below.
    Args:
        ibs (WBIAController):  wbia controller object
        aid_list (int): annot ids specifying the input
        config (str): URL to config file
    Returns:
        gpath_list (list): list of paths to saved images
        names_list (list): list of names of saved images
        seg_mask_list (list): list of binary segmentation masks as numpy arrays
    """
    # Get species from the first annotation
    #species = ibs.get_annot_species_texts(aid_list[0])

    # Load config
    if config is not None and config in CONFIGS:
        config_url = CONFIGS[config]
        cfg = _load_config(config_url)
    else:
        cfg = _load_config()

    # Load model
    if config is not None and config in MODELS:
        model_url = MODELS[config]
    else:
        model_url = None
    model = _load_model(cfg, model_url)

    # Create data loader with proper transformations
    test_loader, _ = _load_data(ibs, aid_list, cfg, multithread)

    # Compute segmentation masks
    model.eval()
    gpath_list = []
    names_list = []
    seg_mask_list = []
    masks_savedir = os.path.join(os.getenv('HOME_FOLDER'), cfg.data.inference_mask_dir)
    print(f"Saving masks to {masks_savedir}")
    os.makedirs(masks_savedir, exist_ok=True)

    with torch.no_grad():
        for images, names, image_sizes in tqdm(test_loader):
            images = images.to(cfg.device)
            output = model.predict(images.float())
            seg_masks = output.argmax(dim=1).detach().cpu()
            
            images = images.cpu()
            for i in range(images.shape[0]):
                im = images[0]
                im = im.permute(1, 2, 0)
                mask = seg_masks[0]

                # Apply mask to image (mask is overlayed keeping original image)
                overlayed_im = overlay_seg_mask(im, mask)
                image_uuid_name = names[i].split("/")[-1]
                mask_fp = os.path.join(masks_savedir, image_uuid_name)+cfg.data.mask_suffix
                overlayed_im.save(mask_fp)

                gpath_list.append(mask_fp)
                names_list.append(f"{image_uuid_name}{cfg.data.mask_suffix}")
                seg_mask_list.append(mask.numpy())
    
    return gpath_list, names_list, seg_mask_list


def _load_config(config_url=None):
    r"""
    Load a configuration file
    """
    args = get_default_config()

    if config_url:
        config_fname = config_url.split('/')[-1]
        config_file = ut.grab_file_url(
            config_url, appname='wbia_segmentation', check_hash=True, fname=config_fname
        )
        args = merge_from_file(args, config_file)
    
    return args


def _load_model(cfg, model_url=None):
    r"""
    Load a model based on config file
    """
    if cfg.test.use_cuda:
        cfg.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        cfg.device = 'cpu'
    print('Building model: {}'.format(cfg.model.name))
    model = get_model(cfg)

    # Download the model weights

    if model_url:
        model_fname = model_url.split('/')[-1]
        model_path = ut.grab_file_url(
            model_url, appname='wbia_segmentation', check_hash=True, fname=model_fname
        )
    else:
        model_path = cfg.test.path_to_model

    if cfg.model.name == "hf":
        is_local = model_url is None
        model.model = load_hf_model(model, model_path, is_local)
    else:
        load_pretrained_weights(model, model_path)

    model = model.to(cfg.device)

    return model


def _load_data(ibs, aid_list, cfg, multithread=False):
    r"""
    Load data, preprocess and create data loaders
    """
    target_imsize = (cfg.data.img_height, cfg.data.img_width)

    test_transform = T.Compose(
        [
            T.Resize((target_imsize[0], target_imsize[1])),
        ]
    )

    image_paths = ibs.get_annot_image_paths(aid_list) # ['/data/db/_ibsdb/images/05ccb87c-dcfe-468b-3be4-d89018d4aa73.jpg', ... ]
    names = ibs.get_annot_name_rowids(aid_list) # [-1, -2, ...]

    dataset = InferenceSegDataset(image_paths, cfg, test_transform)

    if multithread:
        num_workers = cfg.data.num_workers
    else:
        num_workers = 0

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.test.batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    logging.info('Loaded {} images for model evaluation'.format(len(dataset)))

    return dataloader, dataset


def wbia_segmentation_test_ibs(demo_db_url, species, subset):
    pass


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m wbia_segmentation._plugin --allexamples
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
