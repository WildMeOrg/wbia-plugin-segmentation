# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from wbia.control import controller_inject
import numpy as np
import utool as ut
import torch
import torchvision.transforms as transforms

from wbia_segmentation.utils.utils import merge_from_file, load_pretrained_weights
from wbia_segmentation.default_config import get_default_config
from wbia_segmentation.models import get_model
from wbia_segmentation.data.dataset import SegDataset

(print, rrr, profile) = ut.inject2(__name__)

_, register_ibs_method = controller_inject.make_ibs_register_decorator(__name__)

register_api = controller_inject.get_wbia_flask_api(__name__)
register_route = controller_inject.get_wbia_flask_route(__name__)

register_preproc_image = controller_inject.register_preprocs['image']
register_preproc_annot = controller_inject.register_preprocs['annot']

DEMOS = {}

CONFIGS = {}

MODELS = {}


@register_ibs_method
def segmentation_compute_mask(ibs, aid_list, config=None, multithread=False):
    # Get species from the first annotation
    species = ibs.get_annot_species_texts(aid_list[0])

    # Load config
    if config is None:
        config = CONFIGS[species]
    cfg = _load_config(config)

    # Load model
    model = _load_model(cfg, MODELS[species])

    # Preprocess images to model input
    test_loader, test_dataset = _load_data(ibs, aid_list, cfg, multithread)

    # Compute segmentation masks
    seg_masks = []
    model.eval()
    with torch.no_grad():
        for images, names in test_loader:
            if cfg.use_gpu:
                images = images.cuda(non_blocking=True)

            output = model(images.float())
            seg_masks.append(output.detach().cpu().numpy())

    seg_masks = np.concatenate(seg_masks)
    return seg_masks


def _load_config(config_url):
    r"""
    Load a configuration file
    """
    config_fname = config_url.split('/')[-1]
    config_file = ut.grab_file_url(
        config_url, appname='wbia_segmentation', check_hash=True, fname=config_fname
    )

    args = get_default_config()
    args = merge_from_file(args, config_file)
    
    return args


def _load_model(cfg, model_url):
    r"""
    Load a model based on config file
    """
    print('Building model: {}'.format(cfg.model.name))
    model = get_model(cfg)

    # Download the model weights
    model_fname = model_url.split('/')[-1]
    model_path = ut.grab_file_url(
        model_url, appname='wbia_segmentation', check_hash=True, fname=model_fname
    )

    if cfg.model.name == "hf":
        model.model = model.model.from_pretrained(model_path)
    else:
        load_pretrained_weights(model, model_path)

    if cfg.use_gpu:
        model = model.cuda()
    return model


def _load_data(ibs, aid_list, cfg, multithread=False):
    r"""
    Load data, preprocess and create data loaders
    """
    # TODO: Define transforms
    test_transform = None

    image_paths = ibs.get_annot_image_paths(aid_list)
    names = ibs.get_annot_name_rowids(aid_list)
    target_imsize = (cfg.data.height, cfg.data.width)

    dataset = SegDataset(cfg.data.train_dir, cfg, test_transform)

    if multithread:
        num_workers = cfg.data.workers
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
    print('Loaded {} images for model evaluation'.format(len(dataset)))

    return dataloader, dataset


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m wbia_segmentation._plugin --allexamples
    """
    import multiprocessing

    multiprocessing.freeze_support()  # for win32
    import utool as ut  # NOQA

    ut.doctest_funcs()
