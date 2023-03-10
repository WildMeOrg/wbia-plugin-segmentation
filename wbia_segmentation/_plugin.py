# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
import numpy as np
from tqdm import tqdm
import utool as ut
import logging
import os

import torch
import torchvision.transforms as T

from wbia import dtool as dt
from wbia.control import controller_inject
from wbia.constants import ANNOTATION_TABLE

from wbia_segmentation.utils.utils import merge_from_file, load_pretrained_weights, get_seg_overlay
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

CONFIGS = {}

MODELS = {}

class SegmentationConfig(dt.Config):  # NOQA
    _param_info_list = [
        ut.ParamInfo('config_path', default=None),
    ]


@register_ibs_method
def register_segmentations(ibs, aid_list, config_url=None):

    gpath_list, names, seg_masks = ibs._compute_segmentations(aid_list, config_url)

    seg_mask_gids = ibs.add_images(gpath_list, as_annots=True)
    metadata_dict_list = [{"mask_name": n} for n in gpath_list]
    ibs.set_annot_metadata(aid_list, metadata_dict_list)

    return seg_mask_gids, names, seg_masks


@register_preproc_annot(
    tablename='SegmentationMask',
    parents=[ANNOTATION_TABLE],
    colnames=['seg_mask'],
    coltypes=[np.ndarray],
    configclass=SegmentationConfig,
    fname='segmentation',
    chunksize=128,
)
@register_ibs_method
def register_segmentations_depc(depc, aid_list, config_url=None):
    ibs = depc.controller
    seg_mask_gids, seg_names, seg_masks = register_segmentations(
        ibs,
        aid_list,
        config_path=config_url,
    )
    for aid, mask in zip(aid_list, seg_masks):
        yield (mask,)


@register_ibs_method
def _compute_segmentations(ibs, aid_list, config_url=None, multithread=False):
    # Get species from the first annotation
    species = ibs.get_annot_species_texts(aid_list[0])

    # Load config
    if config_url is None:
        cfg = _load_config()
    else:
        cfg = _load_config(config_url)

    # Load model
    if species in MODELS:
        model_url = MODELS[species]
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
    os.makedirs(cfg.data.inference_mask_dir, exist_ok=True)

    with torch.no_grad():
        for images, names, image_sizes in tqdm(test_loader):
            images = images.to(cfg.train.device)
            output = model.predict(images.float())
            seg_masks = output.argmax(dim=1).detach().cpu()
            
            images = images.cpu()
            for i in range(images.shape[0]):
                im = images[0]
                im = im.permute(1, 2, 0)
                mask = seg_masks[0]

                overlayed_im = get_seg_overlay(im, mask)
                image_uuid_name = names[i].split("/")[-1]
                mask_fp = os.path.join(cfg.data.inference_mask_dir, image_uuid_name)+cfg.data.mask_suffix
                overlayed_im.save(mask_fp)

                gpath_list.append(mask_fp)
                names_list.append(f"{image_uuid_name}{cfg.data.mask_suffix}")
                seg_mask_list.append(mask)
    
    return gpath_list, names_list, seg_mask_list


def _load_config(config_name=None):
    r"""
    Load a configuration file
    """
    args = get_default_config()

    if config_name:
        """
        config_fname = config_url.split('/')[-1]
        config_file = ut.grab_file_url(
            config_url, appname='wbia_segmentation', check_hash=True, fname=config_fname
        )
        """
        config_file = f"/wbia/wbia-plugin-segmentation/wbia_segmentation/configs/{config_name}"

        args = merge_from_file(args, config_file)
    
    return args


def _load_model(cfg, model_url=None):
    r"""
    Load a model based on config file
    """
    cfg.train.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        model.model = model.model.from_pretrained(model_path)
    else:
        load_pretrained_weights(model, model_path)

    model = model.to(cfg.train.device)

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
