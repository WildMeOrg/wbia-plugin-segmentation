import matplotlib.pyplot as plt

import evaluate

import torch
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
                ignore_index=0,
                reduce_labels=False,
            )
        
    # add per category metrics as individual key-value pairs
    per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
    per_category_iou = metrics.pop("per_category_iou").tolist()

    metrics.update({f"accuracy_{id2label[i]}": v for i, v in enumerate(per_category_accuracy)})
    metrics.update({f"iou_{id2label[i]}": v for i, v in enumerate(per_category_iou)})

    return metrics

def display_results(net, dset, device, num_to_show, wandb):
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
    num_shown = 0
    for images, masks, names in ds_loader:
        images = images.to(device)
        net.eval()
        logits = net(images)
        m = torch.nn.Softmax(dim=1)
        probs = m(logits) # batch x
        preds = torch.max(probs, dim=1).indices

        '''
        Calculate how many images to show here. The default is the number
        per batch, but in the last iteration there may be fewer, which will
        occur if batch_size is not a multiple of num_to_show
        '''
        im_length = images.shape[0]
        if num_shown + im_length <= num_to_show:
            display_length = im_length
        else:
            display_length = num_shown + im_length - num_to_show
        num_shown += display_length

        '''
        At this point, probs, preds, masks and alphas should all be the same
        shape, specifically (num_to_show, width, height)
        And, each should be binary. For the decisions and masks, 0 indicates
        foreground, 1 indicates background. For alphas, 1 indicates that the pixel
        is of interest.
        '''

        for i in range(display_length):
            im = images[i, ...].detach().cpu().numpy()
            im = im.transpose(1, 2, 0)
            mask = masks[i, ...].numpy()
            pred = preds[i, ...].detach().cpu().numpy()
            
            wandb.Image(im, masks={
                "prediction" : {"mask_data" : pred, "class_labels" : class_labels},
                "ground truth" : {"mask_data" : mask, "class_labels" : class_labels}}
            )
            
        if num_shown >= num_to_show:
            break
