from torchvision.io import read_image
from torch.utils.data import Dataset
import torch

import os
from os import listdir
from os.path import join
from pathlib import Path
import logging

import random

def select_random_elements(lst, num_elements, seed=None):
    """
    Selects a specified number of elements from a list at random while maintaining their original order.
    
    Args:
    lst: A list of elements
    num_elements: An integer specifying the number of elements to select
    seed: An optional integer seed value for reproducibility
    
    Returns:
    A list of randomly selected elements from the input list in their original order
    """
    if num_elements > len(lst):
        raise ValueError("num_elements must be less than or equal to the length of the input list")
    
    # Set the random seed if provided
    if seed is not None:
        random.seed(seed)
    
    # Make a copy of the list and shuffle its indices
    indices = list(range(len(lst)))
    random.shuffle(indices)
    
    # Select the specified number of indices and sort them
    sample_indices = sorted(indices[:num_elements])
    
    # Extract the elements at the selected indices and return them in their original order
    return sorted(sample_indices)


class SegDataset(Dataset):
    def __init__(self,
                 images_dir,
                 args,
                 transform = None,
                 mask_suffix = '_mask.png'):
        '''
        Record the names, the image file names and the mask filenames (derived
        from the image file names).  This allows the original images (chips, actually)
        and binary masks to co-exist in the same folder, provided the mask
        images end with the given mask_suffix. It also, for historical reasons only,
        allows there to be images with the name 'blend' in them.  These are ignored.
        Finally, there is a one-to-one correspondence between image file and mask image
        files.
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
        '''
        FIX ME.  THese comments are out of date.

        For the given idx, return the image, the mask, the alpha mask and the
        image name. The mask and the alpha mask require an explanation. The values
        in the mask are the reflection of the values in the input mask image:
        0 -> 2, 1->1 and 2->0. In other words, the regions outside the annotation
        now have value 2 and the foreground now has value 0. This makes computing
        the loss values easier. The alpha mask is a binary version of the original
        mask mapping 1 and 2 to 1, so that pixels that we must ignore are 0 and
        pixels to pay attention to are 1.

        For images from the training set, random rotations and crops are applied.
        For images from other sets, only centered cropping is applied.

        '''
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
    def __init__(self, image_fns, cfg, transform=None):
        self.model_name = cfg.model.name
        self.transform = transform
        self.image_fns = image_fns
        self.image_fns.sort()
        self.names = [os.path.splitext(im_fn)[0] for im_fn in self.image_fns]
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
