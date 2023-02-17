from pathlib import Path
import os
import logging
import math as m
from argparse import Namespace
from tqdm import tqdm
import wandb

import torch.nn as nn
import torch
from torchvision.utils import save_image


from models import get_model
from utils.optimization import (
    get_criterion,
    get_optimizer,
    get_scheduler
)
from data.helpers import get_data_loaders, get_test_data_loader
from utils.utils import display_results, mean_iou
from data.transforms import size_and_crop_to_original


def evaluate(net, dataloader, args, device, loss):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    iou_metrics_avg = {}

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true, name = batch
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)

        with torch.no_grad():
            softmax = torch.nn.Softmax(dim=1)
            if args.model_name == 'hf':
                logits, _ = net(image, mask_true)
            else:
                logits = net(image)
            
            preds = softmax(logits)
            dice_score += loss(preds, mask_true)
            
            pred_labels = preds.argmax(dim=1).detach().cpu().numpy()
            mask = mask_true.detach().cpu().numpy()
            iou_metrics = mean_iou(pred_labels, mask)

            if not iou_metrics_avg:
                iou_metrics_avg = iou_metrics
            else:
                for name, value in iou_metrics_avg.items():
                    if not m.isnan(value):
                        iou_metrics_avg[name] += iou_metrics[name]

    val_iou_metrics_avg = {}
    for name, value in iou_metrics_avg.items():
        if not m.isnan(value):
            val_iou_metrics_avg[f"val/{name}"] = value/num_val_batches

    net.train()
    return dice_score/num_val_batches, val_iou_metrics_avg


def train_net_coco(net, args):
    train_loader, n_train, val_loader, val_set = get_data_loaders(args)

    dice_loss, criterion = get_criterion(args)
    optimizer = get_optimizer(net, args)
    scheduler = get_scheduler(optimizer, args)
   
    grad_scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    sm = torch.nn.Softmax(dim=1)

    # 4. Set up to recall the best model and show results
    best_val = 1e6
    max_val_to_save = 0.4   # was 0.25
    path_to_best_model = None
    num_to_show = 5   # when visualizing validation errors

    # 5. Set up weights and biases tracking
    # wandb.init(project="test-project", entity="wildme")  <<== cause errors
    wandb.init(
        project=args.wandb_project_name,
        config=vars(args)
    )
    # wandb.watch(net)
    image_ct = 0         # Counts the number of images examined
    global_step_ct = 0   # Counts the number of training steps
    n_steps_per_epoch = m.ceil(n_train / args.batch_size)

    # 5. Begin training
    for epoch in range(1, args.epochs+1):
        net.train()

        with tqdm(total=n_train, desc=f'Epoch {epoch}/{args.epochs}', unit='img') as pbar:
            for images, masks, names in train_loader:
                assert images.shape[1] == args.n_channels, \
                    f'Network has been defined with {args.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                with torch.cuda.amp.autocast(enabled=args.amp):
                    net = net.float()
                    
                    if args.model_name == 'hf':
                        logits, masks = net(images, masks)
                    else:
                        images = images.to(device=args.device, dtype=torch.float32)
                        masks = masks.to(device=args.device, dtype=torch.long)
                        logits = net(images)
                    loss = criterion(logits, masks) \
                            + dice_loss(sm(logits), masks)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                im_length = images.shape[0]
                pbar.update(im_length)
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                global_step_ct += 1
                image_ct += im_length
 
                metrics = {"train/train_loss": loss.item(),
                           "train/epoch": 1 + global_step_ct / n_steps_per_epoch,
                           "train/image_count": image_ct}

                # Evaluation round
                # division_step = (n_train // (5 * batch_size)) # was 10 * batch_size
                division_step = (n_train // (1 * args.batch_size))
                if division_step > 0 and global_step_ct % division_step == 0:
                    '''
                    histograms = {}
                    for tag, value in net.named_parameters():
                        tag = tag.replace('/', '.')
                        histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                        histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())
                    '''
                    val_score, iou_metrics = evaluate(net, val_loader, args, args.device, dice_loss)

                    if val_score >= best_val:
                        print(f'Validation score {val_score}')
                    else:
                        print(f'Validation score {val_score} ... new best')
                        best_val = val_score

                        if best_val < max_val_to_save:
                            p = Path(args.dir_checkpoint)
                            p.mkdir(parents=True, exist_ok=True)
                            # save_name = f'checkpoint_step_{global_step_ct}_epoch_{epoch}.pth'
                            save_name = f'checkpoint_step_{global_step_ct}_epoch_{epoch}_valscore_{val_score:.4f}'
                            path_to_best_model = str(p / save_name)

                            if args.model_name == 'hf':
                                net.model.save_pretrained(path_to_best_model, from_pt=True)
                            else:
                                torch.save(net.state_dict(), path_to_best_model+'.pth')
                            logging.info('Saved new best model to', path_to_best_model)

                    # Output to wandb
                    val_metrics = {"val/val_dice": val_score}
                    wandb.log({**metrics, **val_metrics, **iou_metrics})
                    scheduler.step(val_score)
                else:
                    wandb.log(metrics)
    
    if args.model_name == 'hf':
        net.model.from_pretrained(path_to_best_model)
    else:
        net.load_state_dict(torch.load(path_to_best_model))
    net.to(args.device)
    net.eval()
    display_results(net, val_set, args, wandb)

    return path_to_best_model


def test(args):
    net_best = get_model(args)
    net_best.load_state_dict(torch.load(args.path_to_best))
    net_best.to(args.device)
    net_best.eval()

    dice_loss, _ = get_criterion(args)
    test_loader = get_test_data_loader(args)
    test_metric, iou_metrics = evaluate(net_best, test_loader, args.device, dice_loss)

    print(test_metric)
    print(iou_metrics)


def segmentation_output(args, names, labels, sizes):
    # NOT TESTED
    #  OOPS.  JUST REMEMEBERED THAT THIS NEEDS INFORMATION ABOUT THE ORIGINAL DIMENSIONS.
    num_images = len(names)
    assert num_images == labels.size()[0]
    os.makedirs(args.inference_mask_dir, exist_ok=True)

    for name, label, size in zip(names, labels, sizes):
        bin_im = size_and_crop_to_original(bin_im, size[0], size[1])
        fp = os.path.join(arg.inference_mask_dir, name, args.mask_suffix)
        save_image(bin_im, fp)


def inference(args):
    '''
    Apply inference to a folder of images.
    NOT TESTED
    '''
    net_best = get_model(args)
    net_best.load_state_dict(torch.load(args.path_to_best))
    net_best.to(args.device)
    net_best.eval()
    inference_loader = get_inference_data_loader(args)

    num_val_batches = len(inference_loader)

    # iterate over the validation set
    for batch in tqdm(inference_loader, total=num_val_batches, desc='Inference time', unit='batch', leave=False):
        image, name, im_size = batch
        image = image.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            logits = net(image)
            softmax = torch.nn.Softmax(dim=1)
            preds = softmax(logits)
            pred_labels = preds.argmax(dim=1).detach().cpu().numpy()
            segmentation_output(args, name, pred_labels, im_size)



def main():
    args = Namespace()
    
    # Management
    args.wandb_project_name = "test-project"
    args.dir_checkpoint = "./checkpoints"
    args.save_checkpoint = False
    args.path_to_best = ''
    args.processing_stage= 'Train' # OR 'Test' OR 'Inference' 
    
    # Data
    args.train_dir = 'snowleopard_v2/train'
    args.val_dir = 'snowleopard_v2/val'
    args.test_dir = 'snowleopard_v2/test'
    args.inference_dir = './inference'  # NEW
    args.mask_suffix = '_mask.png'         # NEW
    args.inference_mask_dir = './mask_results'

    args.num_workers = 2
    args.img_height = 400
    args.img_width = 400
    args.transforms_train = ["random_rotation", "random_crop"]
    args.transforms_test = 'center_crop'
    args.transforms_inference = 'resize'
    args.norm_mean = None
    args.norm_std = None

    # Model
    args.model_name = "hf"
    args.model_path = "nvidia/mit-b0"
    args.n_channels = 3
    args.n_classes = 2
    args.bilinear = False
    args.id2label = {0: "background", 1: "foreground"}
    args.label2id = {"background": 0, "foreground": 1}

    # Training
    args.epochs = 25
    args.batch_size = 2
    args.optim = "rms"
    args.scheduler = "plateau"
    args.scheduler_patience = 2
    args.lr = 1e-5
    args.amp = False

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(args.device)

    model = get_model(args)
    model = model.to(args.device)

    if args.processing_stage == 'Train':
        path_to_best = train_net_coco(model, args)
        print(path_to_best)
    elif args.processing_stage == 'Test':
        test(args)
    elif args.processing_stage == 'Inference':
        inference(args)


if __name__ == "__main__":
    main()
