import matplotlib.pyplot as plt
import yaml
from yaml.loader import SafeLoader
from argparse import Namespace
import pickle
import os.path as osp
import warnings
from functools import partial
from collections import OrderedDict

import evaluate

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

metric = evaluate.load("mean_iou")

class_labels = {
    0: "background",
    1: "foreground",
}


def mean_iou(preds, mask, id2label={0:'background', 1:'foreground'}):
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

def display_results(net, dset, args, wandb):
    '''
    Get for each of the first num_to_show validation images,
    1. Form into batches
    2. Use the data loader to load images, masks, alphas and names
    3. Run through the net to produce logits
    4. Use argmax with dim=1 to get categorical label predictions
    5. Display the batch's images, manual segmentations (from the mask(),
       predictions, and names.
    '''
    
    batch_size = 5
    loader_args = dict(batch_size=batch_size, num_workers=2, pin_memory=True)
    ds_loader = DataLoader(dset, shuffle=False, drop_last=True, **loader_args)
    table = wandb.Table(columns=['ID', 'Image'])

    net.eval()

    for images, masks, names in ds_loader:
        images = images.to(args.train.device)

        if args.model.name == 'hf':
            logits, _ = net(images, masks)
        else:
            logits = net(images)
        
        softmax = torch.nn.Softmax(dim=1)
        preds = softmax(logits)

        preds = torch.max(preds, dim=1).indices

        '''
        At this point, probs, preds, masks should all be the same
        shape, specifically (num_to_show, width, height)
        And, each should be binary. For the decisions and masks, 0 indicates
        foreground, 1 indicates background.
        '''

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
    
    wandb.log({"Validation results" : table})


def merge_from_file(args, cfg_path):
    with open(cfg_path) as f:
        cfg = yaml.load(f, Loader=SafeLoader)
    
    args = vars(args)
    
    for group_key, group_value in cfg.items():
        group_args = vars(args[group_key])
        for key, value in group_value.items():
            group_args[key] = value
        args[group_key] = Namespace(**group_args)
    
    return Namespace(**args)


def load_checkpoint(fpath):
    r"""Loads checkpoint.
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


def load_pretrained_weights(model, weight_path):
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