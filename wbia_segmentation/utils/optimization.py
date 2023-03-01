import torch.optim as optim
import torch.nn as nn
from transformers import get_scheduler as get_hf_scheduler

from utils.loss import dice_loss

def get_criterion(args):
    return dice_loss, nn.CrossEntropyLoss(ignore_index=2)

def get_optimizer(model, args):
    if args.train.optim == "adam":
        optimizer = optim.Adam(
            model.parameters(), lr=args.train.lr, weight_decay=args.train.wd
        )
    elif args.train.optim == "adamw":
        optimizer = optim.AdamW(
            model.parameters(), lr=args.train.lr
        )
    elif args.train.optim == "rms":
        optimizer = optim.RMSprop(model.parameters(), lr=args.train.lr, weight_decay=1e-8, momentum=0.9)
    else:
        raise ValueError
    
    return optimizer


def get_scheduler(optimizer, args):
    if args.train.scheduler == "step":
        scheduler = optim.StepLR(optimizer, args.train.scheduler_step_size)
    elif args.train.scheduler == "linear":
        scheduler = get_hf_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=args.train.num_training_steps,
        )
    elif args.train.scheduler == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=args.train.scheduler_patience)
    else:
        raise ValueError

    return scheduler