import argparse
import numpy as np
from PIL import Image
import zipfile
import os
import yaml
from yaml.loader import SafeLoader
from argparse import Namespace
import pickle
import os.path as osp
import warnings
from functools import partial
from collections import OrderedDict
from typing import Dict
import wandb

import evaluate
import transformers

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


metric = evaluate.load("mean_iou")

class_labels = {
    0: "background",
    1: "foreground",
}


def mean_iou(preds: torch.Tensor, mask: torch.Tensor, id2label={0:'background', 1:'foreground'}) -> Dict:
    r"""
    Computes the mean IoU for a batch of predictions and masks.
    Args:
        preds (torch.Tensor): The predictions.
        mask (torch.Tensor): The ground truth masks.
        id2label (dict): A dictionary mapping class IDs to class names.
    Returns:
        The mean IoU.
    """
    metrics = metric._compute(
                predictions=preds,
                references=mask,
                num_labels=len(id2label),
                ignore_index=255,
                reduce_labels=False,
            )
        
    # add per category metrics as individual key-value pairs
    per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
    per_category_iou = metrics.pop("per_category_iou").tolist()

    metrics.update({f"accuracy_{id2label[i]}": v for i, v in enumerate(per_category_accuracy)})
    metrics.update({f"iou_{id2label[i]}": v for i, v in enumerate(per_category_iou)})

    return metrics

def display_results(net: torch.nn.Module, dset: torch.utils.data.Dataset, args: argparse.Namespace, wandb: wandb) -> None:
    r"""
    Displays and saves the results of the model on the validation set to wandb.
    Args:
        net (torch.nn.Module): The model.
        dset (torch.utils.data.Dataset): The validation set.
        args (argparse.Namespace): The arguments.
        wandb (wandb): The wandb object.
    """
    
    # Create a dataloader for the validation set and a wandb table to store the images
    batch_size = 5
    loader_args = dict(batch_size=batch_size, num_workers=2, pin_memory=True)
    ds_loader = DataLoader(dset, shuffle=False, drop_last=True, **loader_args)
    table = wandb.Table(columns=['ID', 'Image'])

    net.eval()

    # Iterate over the validation set and get segmentation masks for each image
    for images, masks, names in ds_loader:
        images = images.to(args.device)

        if args.model.name == 'hf':
            logits, _ = net(images, masks)
        else:
            logits = net(images)
        
        softmax = torch.nn.Softmax(dim=1)
        preds = softmax(logits)

        preds = torch.max(preds, dim=1).indices

        # Iterate over the batch, transforming the images and masks into wandb images

        for i in range(preds.shape[0]):
            im = images[i, ...].detach().cpu().numpy()
            im = im.transpose(1, 2, 0)
            mask = masks[i, ...].numpy()
            pred = preds[i, ...].detach().cpu().numpy()
            
            mask_img = wandb.Image(im, masks={
                "prediction" : {"mask_data" : pred, "class_labels" : class_labels},
                "ground truth" : {"mask_data" : mask, "class_labels" : class_labels}}
            )

            table.add_data(names[i], mask_img)
    
    # Log the table to wandb
    wandb.log({"Validation results" : table})


def overlay_seg_mask(image: torch.Tensor , seg: torch.Tensor) -> Image:
    r"""
    Overlays the segmentation mask on the image.
    Args:
        image (torch.Tensor): The image.
        seg (torch.Tensor): The segmentation mask.
    Returns:
        The image with the segmentation mask overlaid as a PIL image.
    """
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8) # height, width, 3
    color_seg[seg == 1, :] = [216, 82, 24]

    # Show image + mask
    img = np.array(image) * 0.5 + color_seg * 0.5
    img = img.astype(np.uint8)
    img = Image.fromarray(img).convert('RGB')
    return img


def apply_seg_mask(image: torch.Tensor, seg: torch.Tensor) -> Image:
    r"""
    Applies the segmentation mask to the image. Background pixels are set to 0.
    Args:
        image (torch.Tensor): The image.
        seg (torch.Tensor): The segmentation mask.
    Returns:
        The image with the segmentation mask applied as a PIL image.
    """
    seg_with_channels = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    seg_with_channels[seg == 1, :] = [1, 1, 1]

    # Show image without background
    img = np.array(image) * seg_with_channels
    img = img.astype(np.uint8)
    img = Image.fromarray(img).convert('RGB')
    return img


def merge_from_file(args: argparse.Namespace, cfg_path: str) -> argparse.Namespace:
    r"""
    Merges config from yaml file into args. Overwrites default args with config file values.
    Args:
        args (argparse.Namespace): The arguments.
        cfg_path (str): The path to the config file.
    Returns:
        argparse.Namespace: The merged arguments.
    """
    with open(cfg_path) as f:
        cfg = yaml.load(f, Loader=SafeLoader)
    
    args = vars(args)
    
    for group_key, group_value in cfg.items():
        group_args = vars(args[group_key])
        for key, value in group_value.items():
            group_args[key] = value
        args[group_key] = Namespace(**group_args)
    
    return Namespace(**args)


def load_checkpoint(fpath: str):
    r"""
    Loads checkpoint.
    ``UnicodeDecodeError`` can be well handled, which means
    python2-saved files can be read from python3.
    Args:
        fpath (str): path to checkpoint.
    Returns:
        dict
    """
    if fpath is None:
        raise ValueError('File path is None')
    if not osp.exists(fpath):
        raise FileNotFoundError('File is not found at "{}"'.format(fpath))
    map_location = None if torch.cuda.is_available() else 'cpu'
    try:
        checkpoint = torch.load(fpath, map_location=map_location)
    except UnicodeDecodeError:
        pickle.load = partial(pickle.load, encoding='latin1')
        pickle.Unpickler = partial(pickle.Unpickler, encoding='latin1')
        checkpoint = torch.load(fpath, pickle_module=pickle, map_location=map_location)
    except Exception:
        print('Unable to load checkpoint from "{}"'.format(fpath))
        raise
    return checkpoint


def load_hf_model(model: transformers.PreTrainedModel, compressed_model_path: str, is_local: bool=False) -> transformers.PreTrainedModel:
    r"""
    Loads HuggingFace pretrained weights to model.
    Features::
        - Model assumed to be a zip file. HF expects two files: 'config.json' and 'pytorch_model.bin'.
    Args:
        model (transformers.PreTrainedModel): HF network model object.
        weight_path (str): path to zipped pretrained weights and config file.
        is_local (bool): whether the model is local or not.
    Returns:
        transformers.PreTrainedModel: HF network model object.
    """
    # If model is local, load it directly (assumes model is not zipped)
    if is_local:
        return model.model.from_pretrained(compressed_model_path)
    
    # If model is not local, unzip it and load it
    end_idx_path = compressed_model_path.rindex("/")
    unzip_path = compressed_model_path[:end_idx_path]

    zipped_model = zipfile.ZipFile(compressed_model_path)
    unzip_folder_name = zipped_model.namelist()[0].split("/")[0]
    zipped_model.extractall(unzip_path)
    zipped_model.close()
    
    return model.model.from_pretrained(os.path.join(unzip_path, unzip_folder_name))


def load_pretrained_weights(model: nn.Module, weight_path: str) -> None:
    r"""Loads pretrained weights to model.
    Features::
        - Incompatible layers (unmatched in name or size) will be ignored.
        - Can automatically deal with keys containing "module.".
    Args:
        model (nn.Module): network model.
        weight_path (str): path to pretrained weights.
    """
    
    checkpoint = load_checkpoint(weight_path)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []

    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]  # discard module.

        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)

    if len(matched_layers) == 0:
        warnings.warn(
            'The pretrained weights "{}" cannot be loaded, '
            'please check the key names manually '
            '(** ignored and continue **)'.format(weight_path)
        )
    else:
        print('Successfully loaded pretrained weights from "{}"'.format(weight_path))
        if len(discarded_layers) > 0:
            print(
                '** The following layers are discarded '
                'due to unmatched keys or layer size: {}'.format(discarded_layers)
            )