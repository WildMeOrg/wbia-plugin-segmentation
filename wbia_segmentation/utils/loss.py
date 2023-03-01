from torch import Tensor
import torch


# dice loss without alpha
def dice_loss(preds: Tensor, mask: Tensor, epsilon=1e-6):
    """
    Compute the dice_loss across all classes (0 / 1 for foreground/background)
    preds should be nbatch, nclass, width, height of float
    mask should be nbatch, width, height of long (in range [0, nclass+1))
    alpha should be nbatch, width, height of long (in (0, 1))

    For the mask, the value nclasses indicates a pixel that is not of interest.
    The mask is converted to one-hot to make the multiplication easier.
    """
    num_classes = preds.size()[1]
    mask_one = torch.nn.functional.one_hot(mask, num_classes+1).permute(0, 3, 1, 2)
    dice_sum = 0
    for c in range(num_classes):
        pred_c = preds[:, c, ...]
        mask_c = mask_one[:, c, ...]
        inter = torch.sum(pred_c * mask_c)
        union_pred = torch.sum(pred_c)
        union_mask = torch.sum(mask_c)
        dice_coeff = 2 * inter / (union_pred + union_mask)
        dice_sum += dice_coeff
    dice_avg = dice_sum / num_classes
    return 1 - dice_avg    # reverse so the lower values are better 
