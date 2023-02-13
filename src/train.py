from pathlib import Path
import logging
import math as m
from argparse import Namespace
from tqdm import tqdm
import wandb

import torch

from models import get_model
from utils.optimization import (
    get_criterion,
    get_optimizer,
    get_scheduler
)
from data.helpers import get_data_loaders, get_test_data_loader
from utils.utils import display_results


def evaluate(net, dataloader, device, loss):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    # iterate over the validation set
    for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
        image, mask_true, name = batch
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)

        with torch.no_grad():
            logits = net(image)
            m = torch.nn.Softmax(dim=1)
            preds = m(logits)
            dice_score += loss(preds, mask_true)

    net.train()
    return dice_score / num_val_batches


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
    wandb.init(project=args.wandb_project_name)
    wandb.config = vars(args)
    # wandb.watch(net)
    image_ct = 0         # Counts the number of images examined
    global_step_ct = 0   # Counts the number of training steps
    n_steps_per_epoch = m.ceil(n_train / args.batch_size)

    # 5. Begin training
    for epoch in range(1, args.epochs+1):
        net.train()

        with tqdm(total=n_train, desc=f'Epoch {epoch}/{args.epochs}', unit='img') as pbar:
            for images, masks, names in train_loader:
                assert images.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'
                images = images.to(device=args.device, dtype=torch.float32)
                masks = masks.to(device=args.device, dtype=torch.long)

                with torch.cuda.amp.autocast(enabled=args.amp):
                    net = net.float()
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
                    val_score = evaluate(net, val_loader, args.device, dice_loss)

                    if val_score >= best_val:
                        print(f'Validation score {val_score}')
                    else:
                        print(f'Validation score {val_score} ... new best')
                        best_val = val_score

                        if best_val < max_val_to_save:
                            p = Path(args.dir_checkpoint)
                            p.mkdir(parents=True, exist_ok=True)
                            # save_name = f'checkpoint_step_{global_step_ct}_epoch_{epoch}.pth'
                            save_name = f'checkpoint_step_{global_step_ct}_epoch_{epoch}_valscore_{val_score:.4f}.pth'
                            path_to_best_model = str(p / save_name)
                            torch.save(net.state_dict(), path_to_best_model)
                            logging.info('Saved new best model to', path_to_best_model)
                            display_results(net, val_set, args.device, num_to_show)

                    # Output to wandb
                    val_metrics = {"val/val_accuracy": val_score}
                    wandb.log({**metrics, **val_metrics})
                else:
                    wandb.log(metrics)

    return path_to_best_model


def test(args):
    net_best = get_model(args)
    net_best.load_state_dict(torch.load(args.path_to_best))
    net_best.to(args.device)
    net_best.eval()

    dice_loss, _ = get_criterion(args)
    test_loader = get_test_data_loader(args)
    test_metric = evaluate(net_best, test_loader, args.device, dice_loss)

    print(test_metric)

def main():
    args = Namespace()
    
    # Management
    args.wandb_project_name = "test-project"
    args.dir_checkpoint = "./checkpoints"
    args.save_checkpoint = False
    args.path_to_best = ''
    args.is_train = True
    
    # Data
    args.train_dir = './train'
    args.val_dir = './val'
    args.test_dir = './test'
    args.num_workers = 2
    args.img_height = 400
    args.img_width = 400
    args.transforms_train = ["random_rotation", "random_crop"]
    args.transforms_test = 'center_crop'
    args.norm_mean = None
    args.norm_std = None

    # Model
    args.model_name = "unet"
    args.n_channels = 3
    args.n_classes = 2
    args.bilinear = False

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

    if args.is_train:
        path_to_best = train_net_coco(model, args)
        print(path_to_best)
    else:
        test(args)

if __name__ == "__main__":
    main()
