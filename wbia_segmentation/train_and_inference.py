from pathlib import Path
import os
import logging
import math as m
import argparse
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
from data.helpers import get_data_loaders, get_test_data_loader, get_inference_data_loader
from utils.utils import display_results, mean_iou, merge_from_file, apply_seg_mask
from data.transforms import size_and_crop_to_original
from default_config import get_default_config


def evaluate(net, dataloader, args, loss):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0
    iou_metrics_avg = {}

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true, name = batch
        image = image.to(device=args.device, dtype=torch.float32)
        mask_true = mask_true.to(device=args.device, dtype=torch.long)

        with torch.no_grad():
            softmax = torch.nn.Softmax(dim=1)
            if args.model.name == 'hf':
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
    args.train.num_training_steps = args.train.epochs * len(train_loader)

    dice_loss, criterion = get_criterion(args)
    optimizer = get_optimizer(net, args)
    scheduler = get_scheduler(optimizer, args)
   
    grad_scaler = torch.cuda.amp.GradScaler(enabled=args.train.amp)
    sm = torch.nn.Softmax(dim=1)

    # 4. Set up to recall the best model and show results
    val_score = 1e6
    best_val = 1e6
    max_val_to_save = 0.4   # was 0.25
    path_to_best_model = None

    # 5. Set up weights and biases tracking
    # wandb.init(project="test-project", entity="wildme")  <<== cause errors
    wandb.init(
        project=args.management.wandb_project_name,
        config=vars(args)
    )
    # wandb.watch(net)
    image_ct = 0         # Counts the number of images examined
    global_step_ct = 0   # Counts the number of training steps
    n_steps_per_epoch = m.ceil(n_train / args.train.batch_size)

    # 5. Begin training
    for epoch in range(1, args.train.epochs+1):
        net.train()

        with tqdm(total=n_train, desc=f'Epoch {epoch}/{args.train.epochs}', unit='img') as pbar:
            for images, masks, names in train_loader:
                assert images.shape[1] == args.data.n_channels, \
                    f'Network has been defined with {args.data.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                with torch.cuda.amp.autocast(enabled=args.train.amp):
                    net = net.float()
                    
                    if args.model.name == 'hf':
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
                if args.train.scheduler and args.train.scheduler in ["plateau"]:
                    scheduler.step(val_score)
                elif args.train.scheduler:
                    scheduler.step()
                else:
                    pass
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
                division_step = (n_train // (1 * args.train.batch_size))
                if division_step > 0 and global_step_ct % division_step == 0:
                    '''
                    histograms = {}
                    for tag, value in net.named_parameters():
                        tag = tag.replace('/', '.')
                        histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                        histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())
                    '''
                    val_score, iou_metrics = evaluate(net, val_loader, args, dice_loss)

                    if val_score >= best_val:
                        print(f'Validation score {val_score}')
                    else:
                        print(f'Validation score {val_score} ... new best')
                        best_val = val_score

                        if best_val < max_val_to_save:
                            p = Path(args.management.dir_checkpoint)
                            p.mkdir(parents=True, exist_ok=True)
                            # save_name = f'checkpoint_step_{global_step_ct}_epoch_{epoch}.pth'
                            save_name = f'checkpoint_step_{global_step_ct}_epoch_{epoch}_valscore_{val_score:.4f}'
                            path_to_best_model = str(p / save_name)

                            if args.model.name == 'hf':
                                net.model.save_pretrained(path_to_best_model, from_pt=True)
                            else:
                                path_to_best_model = path_to_best_model+'.pth'
                                torch.save(net.state_dict(), path_to_best_model)
                            logging.info('Saved new best model to', path_to_best_model)

                    # Output to wandb
                    val_metrics = {"val/val_dice": val_score}
                    wandb.log({**metrics, **val_metrics, **iou_metrics})
                else:
                    wandb.log(metrics)
    
    if args.model.name == 'hf':
        net.model = net.model.from_pretrained(path_to_best_model)
    else:
        net.load_state_dict(torch.load(path_to_best_model))
    net.to(args.device)
    net.eval()
    display_results(net, val_set, args, wandb)

    return path_to_best_model


def test(args):
    net_best = get_model(args)
    if args.model.name == 'hf':
        net_best.model = net_best.model.from_pretrained(args.test.path_to_model)
    else:
        net_best.load_state_dict(torch.load(args.test.path_to_model))
    net_best.to(args.device)
    net_best.eval()

    dice_loss, _ = get_criterion(args)
    test_loader = get_test_data_loader(args)
    test_metric, iou_metrics = evaluate(net_best, test_loader, args, dice_loss)

    print(test_metric)
    print(iou_metrics)


def segmentation_output(args, names, labels, sizes):
    # NOT TESTED
    #  OOPS.  JUST REMEMBERED THAT THIS NEEDS INFORMATION ABOUT THE ORIGINAL DIMENSIONS.
    num_images = len(names)
    assert num_images == labels.size()[0]
    os.makedirs(args.data.inference_mask_dir, exist_ok=True)

    for name, label, size in zip(names, labels, sizes):
        bin_im = size_and_crop_to_original(label, size[0], size[1])
        fp = os.path.join(args.data.inference_mask_dir, f"{name}{args.data.mask_suffix}")
        save_image(bin_im, fp)


def apply_segmentation(args, names, images, labels, sizes):
    num_images = len(names)
    assert num_images == labels.shape[0]
    os.makedirs(args.data.inference_mask_dir, exist_ok=True)

    for name, image, bin_im, size in zip(names, images, labels, sizes):
        fp = os.path.join(args.data.inference_mask_dir, f"{name}{args.data.mask_suffix}")
        image = image.permute(1, 2, 0)
        overlayed_im = apply_seg_mask(image, bin_im)
        overlayed_im.save(fp)


def inference(args):
    '''
    Apply inference to a folder of images.
    '''
    net_best = get_model(args)
    if args.model.name == 'hf':
        net_best.model = net_best.model.from_pretrained(args.test.path_to_model)
    else:
        net_best.load_state_dict(torch.load(args.test.path_to_model))
    net_best.to(args.device)
    net_best.eval()
    inference_loader = get_inference_data_loader(args)
    num_val_batches = len(inference_loader)
    
    # iterate over the validation set
    for batch in tqdm(inference_loader, total=num_val_batches, desc='Inference time', unit='batch', leave=False):
        image, name, im_size = batch
        image = image.to(device=args.device, dtype=torch.float32)

        with torch.no_grad():
            if args.model.name == 'hf':
                output = net_best.predict(image)
                pred_labels = output.argmax(dim=1).detach().cpu()
            else:
                logits = net_best(image)
                softmax = torch.nn.Softmax(dim=1)
                preds = softmax(logits)
                pred_labels = preds.argmax(dim=1).detach().cpu()
            image = image.cpu()
            apply_segmentation(args, name, image, pred_labels, im_size)



def main(params):
    args = get_default_config()

    if params.cfg:
        args = merge_from_file(args, params.cfg)
    
    args.data.train_dir = f'{args.data.source}/train'
    args.data.val_dir = f'{args.data.source}/val'
    args.data.test_dir = f'{args.data.source}/test'
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(args.device)

    if args.management.processing_stage == 'Train':
        model = get_model(args)
        model = model.to(args.device)
        path_to_best = train_net_coco(model, args)
        print(f"Best model saved in {path_to_best}")
    elif args.management.processing_stage == 'Test':
        test(args)
    elif args.management.processing_stage == 'Inference':
        inference(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--cfg', type=str, default='', help='path to config file')

    parser.add_argument(
        'opts',
        default=None,
        nargs=argparse.REMAINDER,
        help='Modify config options using the command-line',
    )
    args = parser.parse_args()

    main(args)