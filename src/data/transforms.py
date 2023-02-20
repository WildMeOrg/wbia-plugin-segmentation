import torchvision.transforms as T
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode


def calc_padding(h, w):
    '''
    Calculate the padding needed to make the image square (and centered).
    Padding is only added to smaller dimension and it is equal on top/bottom, 
    left/right if the image dimensions are even and shifted up/left by one
    otherwise.
    NOT TESTED!
    '''
    if w < h:
        p = w - h
        left = p // 2
        right = p - left
        padding = (left, 0, right, 0)
    else:
        p = h - w
        top = p // 2
        bottom = p - top
        padding = (0, top, 0, bottom)
    return padding

  
class SquarePad:
    '''
    Transform to make an input image (train, val, test) square
    If the image is square already nothing is done to it.
    NOT TESTED!
    '''
    def __call__(self, im):
        s = im.size()
        h, w = s[-2], s[-1]
        if h == w:
            return im
        padding = calc_padding(h, w)
        return F.pad(im, padding, 0, 'constant')


def size_and_crop_to_original(bin_im, orig_height, orig_width):
    '''
    Return the binary image segmentation mask to the original
    image dimension by resizing and cropping.
    NOT TESTED!
    '''
    max_hw = max(orig_height, orig_width)
    left, top, _, _ = calc_padding(orig_height, orig_width)
    bin_im = F.resize(bin_im, max_hw, interpolation=InterpolationMode.NEAREST)
    bin_im = F.crop(bin_im, top, left, orig_height, orig_width)
    return bin_im


def build_train_val_transforms(args):
    """Build train and test transformation functions.
    Args:
        height (int): target image height.
        width (int): target image width.
        transforms_train (str or list of str, optional): transformations
            applied to training data. Default is 'random_flip'.
        transforms_test (str or list of str, optional): transformations
            applied to test data. Default is 'resize'.
        norm_mean (list or None, optional): normalization mean values.
            Default is ImageNet means.
        norm_std (list or None, optional): normalization standard deviation
            values. Default is ImageNet standard deviation values.
    Returns:
        transform_tr: transformation function for training
        transform_te: transformation function for testing
    """
    transform_tr = build_transforms(
        img_height=args.img_height, img_width=args.img_width, transforms=args.transforms_train, norm_mean=args.norm_mean, norm_std=args.norm_std
    )

    transform_te = build_transforms(
        img_height=args.img_height, img_width=args.img_width, transforms=args.transforms_test, norm_mean=args.norm_mean, norm_std=args.norm_std
    )
    return transform_tr, transform_te


def build_transforms(
    img_height,
    img_width,
    transforms='center_crop',
    norm_mean=None,
    norm_std=None,
    **kwargs
):
    """Build transformation functions.
    Args:
        height (int): target image height.
        width (int): target image width.
        transforms (str or list of str, optional): transformations applied to
            input data. Default is 'random_flip'.
        norm_mean (list or None, optional): normalization mean values.
            Default is ImageNet means.
        norm_std (list or None, optional): normalization standard deviation
            values. Default is ImageNet standard deviation values.
    Returns:
        transform_tr: transformation function
    """
    if transforms is None:
        transforms = []

    if isinstance(transforms, str):
        transforms = [transforms]

    if not isinstance(transforms, list):
        raise ValueError(
            'transforms must be a list of strings, but found to be {}'.format(
                type(transforms)
            )
        )

    if len(transforms) > 0:
        transforms = [t.lower() for t in transforms]

    if norm_mean is None or norm_std is None:
        normalize = None
    else:
        normalize = T.Normalize(mean=norm_mean, std=norm_std)

    print('Building transforms ...')
    transform_list = [SquarePad()]

    for tfm in transforms:
        if tfm == "affine":
            transform_list += [
                T.RandomAffine(
                    degrees=10,
                    translate=(0.05, 0.05),
                    scale=(0.9, 1.1),
                    shear=(5, 5),
                    fill=0,
                )
            ]

        elif tfm == "center_crop":
            transform_list += [T.CenterCrop(max(img_height, img_width))]

        elif tfm == "color_jitter":
            transform_list += [
                T.ColorJitter(
                    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1
                )
            ]

        elif tfm == "flip":
            transform_list += [T.RandomHorizontalFlip()]

        elif tfm == "grayscale":
            transform_list += [T.RandomGrayscale(p=0.2)]

        elif tfm == "perspective":
            transform_list += [T.RandomPerspective()]
        
        elif tfm == "random_crop":
            transform_list += [T.RandomCrop(max(img_height, img_width))]

        elif tfm == "random_resized_crop":
            transform_list += [T.RandomResizedCrop(max(img_height, img_width), scale=(0.2, 1))]
        
        elif tfm == "random_rotation":
            transform_list += [T.RandomRotation(30)]

        elif tfm == "resize":
            transform_list += [T.Resize(max(img_height, img_width))]

        elif tfm == "normalize":
            continue

        else:
            raise ValueError
    
    #transform_list += [T.ToTensor()]

    if normalize:
        transform_list += [normalize]

    return T.Compose(transform_list)
