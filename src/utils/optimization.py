import torch.optim as optim
import torch.nn as nn
from transformers import get_scheduler

from utils.loss import dice_loss

def get_criterion(args):
    return dice_loss, nn.CrossEntropyLoss(ignore_index=2)

def get_optimizer(model, args):
    if args.optim == "adam":
        optimizer = optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.wd
        )
    elif args.optim == "adamw":
        optimizer = optim.AdamW(
            model.parameters(), lr=args.lr
        )
    elif args.optim == "rms":
        optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=1e-8, momentum=0.9)
    else:
        raise ValueError
    
    return optimizer


def get_scheduler(optimizer, args):
    if args.scheduler == "step":
        scheduler = optim.StepLR(optimizer, args.scheduler_step_size)
    elif args.scheduler == "linear":
        scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=args.num_training_steps,
        )
    elif args.scheduler == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=args.scheduler_patience)
    else:
        raise ValueError

    return scheduler