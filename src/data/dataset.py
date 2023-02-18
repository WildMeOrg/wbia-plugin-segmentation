from torchvision.io import read_image
from torch.utils.data import Dataset
import torch

from os import listdir
from os.path import splitext
from os.path import join
from pathlib import Path
import logging


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
        self.model_name = args.model_name
        self.images_dir = Path(images_dir)
        self.transform = transform
        file_names = listdir(images_dir)
        file_exts = [splitext(fn)[1].lower() for fn in file_names]
        allowed_exts = ['.jpg', '.jpeg', '.png']
        file_names = [fn
                      for fn, fe in zip(file_names, file_ext)
                      if fe in allowed_exts]
        self.image_fns = [fn 
                          for fn in file_names
                          if 'blend' not in fn and not fn.lower().endswith(mask_suffix)]
        self.image_fns.sort()
        self.names = [splitext(im_fn)[0] for im_fn in self.image_fns]
        self.mask_fns = [fn for fn in file_names if fn.lower().endswith(mask_suffix)]
        self.mask_fns.sort()
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



class InferenceDataset(Dataset):
    def __init__(self, images_dir, transform=None):
        self.images_dir = Path(images_dir)
        self.transform = transform
        self.image_fns = [fn for fn in listdir(images_dir)   # NEED TO BE MORE GENERAL than just jpg
                          if fn.lower().endswith('jpg')]
        self.image_fns.sort()
        if not self.image_fns or len(self.image_fns) == 0:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating inference dataset with {len(self.image_fns)} annotations')

    def __len__(self):
        return len(self.names)

    def __get_item__(self, idx):
        im_fn = join(self.images_dir, self.image_fns[idx])

        if self.model_name == "hf":
            im = read_image(im_fn)
        else:
            im = read_image(im_fn) / 255

        if self.transform is not None:  # MAKE SURE THIS HAS PROPER RESIZE
            im = self.transform(im)
      
        return im, self.names[idx], (im.size()[-2], im.size()[-1])



