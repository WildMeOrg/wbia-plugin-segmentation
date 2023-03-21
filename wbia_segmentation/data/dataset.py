import torchvision.transforms as T
from torchvision.io import read_image
from torch.utils.data import Dataset
import torch

import os
from os import listdir
from os.path import join
from pathlib import Path
from typing import List
import logging
import random
import argparse

def select_random_elements(index_list: List, num_elements: int, seed: int=None) -> List:
    """
    Selects a specified number of elements from a list at random while maintaining their original order.
    Args:
        lst: A list of dataset indices
        num_elements: An integer specifying the number of elements to select
        seed: An optional integer seed value for reproducibility
    Returns:
        A list of randomly selected indices from the input list in their original order
    """
    if num_elements > len(index_list):
        raise ValueError("num_elements must be less than or equal to the length of the input list")
    
    # Set the random seed if provided
    if seed is not None:
        random.seed(seed)
    
    # Make a copy of the list and shuffle its indices
    indices = list(range(len(index_list)))
    random.shuffle(indices)
    
    # Select the specified number of indices and sort them
    sample_indices = sorted(indices[:num_elements])
    
    # Extract the elements at the selected indices and return them in their original order
    return sorted(sample_indices)


class SegDataset(Dataset):
    def __init__(self,
                 images_dir: str,
                 args:  argparse.Namespace,
                 transform: T.Compose = None,
                 mask_suffix: str = '_mask.png'):
        '''
        Record the names, the image file names and the mask filenames (derived
        from the image file names).  This allows the original images (chips, actually)
        and binary masks to co-exist in the same folder, provided the mask
        images end with the given mask_suffix. It also, for historical reasons only,
        allows there to be images with the name 'blend' in them.  These are ignored.
        Finally, there is a one-to-one correspondence between image file and mask image
        files.
        Args:
            images_dir: directory containing the images and masks
            args: command line arguments
            transform: Torchvision transformation function
            mask_suffix: suffix of the mask images
        '''
        self.training_percent = args.data.training_percent
        self.model_name = args.model.name
        self.images_dir = Path(images_dir)
        self.transform = transform
        file_names = listdir(images_dir)
        file_exts = [os.path.splitext(fn)[1].lower() for fn in file_names]
        allowed_exts = ['.jpg', '.jpeg', '.png']
        file_names = [fn
                      for fn, fe in zip(file_names, file_exts)
                      if fe in allowed_exts]
        self.image_fns = [fn 
                          for fn in file_names
                          if 'blend' not in fn and not fn.lower().endswith(mask_suffix)]
        self.image_fns.sort()
        self.names = [os.path.splitext(im_fn)[0] for im_fn in self.image_fns]
        self.mask_fns = [fn for fn in file_names if fn.lower().endswith(mask_suffix)]
        self.mask_fns.sort()

        # Subsample the training data if requested (for experimentation purposes only)
        if "train" in images_dir and self.training_percent:
            sample_size = int(len(self.image_fns)*self.training_percent)
            selected_idxs = select_random_elements(self.image_fns, sample_size, seed=42)

            self.image_fns = [self.image_fns[i] for i in selected_idxs]
            self.mask_fns = [self.mask_fns[i] for i in selected_idxs]
            self.names = [self.names[i] for i in selected_idxs]

        if not self.image_fns:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        if len(self.names) != len(self.mask_fns):
            raise RuntimeError(f'Should have a mask for each example image in {images_dir}')
        logging.info(f'Creating dataset with {len(self.image_fns)} examples')

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        r"""
        For the given idx, return the image, the mask and the image name.
        The mask contains values 0, 1, 2. 0 is don't care, 1 is background, 2 is foreground.
        Values 1 are changed to 0 (background as well). Foreground is mapped to 1.
        The end result is: 0 -> background, 1 -> foreground.

        For images from the training set, random rotations and crops are applied.
        For images from other sets, only centered cropping is applied.
        """
        im_fn = join(self.images_dir, self.image_fns[idx])

        if self.model_name == "hf":
            im = read_image(im_fn)
        else:
            im = read_image(im_fn) / 255
        
        mask_fn = join(self.images_dir, self.mask_fns[idx])
        mask0 = read_image(mask_fn)
        mask = mask0.clone().detach()
        mask[mask == 1] = 0  # background/1 is changed to 0; don't care is also 0
        mask[mask == 2] = 1  # foreground/2 is changed to 1
        mask = torch.cat([mask, mask, mask], dim=0) # add channels to mask to match image channels
        
        # image and mask are concatenated to apply transforms to both at the same time
        # https://discuss.pytorch.org/t/how-to-apply-same-transform-on-a-pair-of-picture/14914/4?u=ssgosh
        if self.transform is not None:
            both_images = torch.cat((im.unsqueeze(0), mask.unsqueeze(0)),0)
            transformed = self.transform(both_images)
            im, mask = transformed[0], transformed[1][0]
      
        return im, mask, self.names[idx]


class InferenceSegDataset(Dataset):
    def __init__(self, image_fns_or_path, cfg, transform=None):
        self.model_name = cfg.model.name
        self.transform = transform

        if isinstance(image_fns_or_path, str):
            image_fns = listdir(image_fns_or_path)
            self.image_fns = [join(image_fns_or_path, fn) for fn in image_fns]
        else:
            self.image_fns = image_fns
        self.image_fns.sort()
        self.names = [os.path.splitext(im_fn)[0].split("/")[-1] for im_fn in self.image_fns]
        if not self.image_fns or len(self.image_fns) == 0:
            raise RuntimeError(f'No image filenames were specified or the list was empty, make sure you the db has images')
        logging.info(f'Creating inference dataset with {len(self.image_fns)} annotations')

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        if self.model_name == "hf":
            im = read_image(self.image_fns[idx])
        else:
            im = read_image(self.image_fns[idx]) / 255

        if self.transform is not None:  # MAKE SURE THIS HAS PROPER RESIZE
            im = self.transform(im)
      
        return im, self.names[idx], (im.size()[-2], im.size()[-1])
