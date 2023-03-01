import matplotlib.pyplot as plt

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
