import argparse
from typing import Callable, Tuple

import torch.optim as optim
import torch.nn as nn
from transformers import get_scheduler as get_hf_scheduler

from utils.loss import dice_loss

def get_criterion(args: argparse.Namespace) -> Tuple[Callable, Callable]:
    r"""
    Returns the loss functions to be used for training.
    Args:
        args (argparse.Namespace): The parsed arguments.
    Returns:
        The loss function.
    """
    return dice_loss, nn.CrossEntropyLoss(ignore_index=2)

def get_optimizer(model: nn.Module, args: argparse.Namespace) -> optim.Optimizer:
    r"""
    Returns the optimizer to be used for training.
    Args:
        model (torch.nn.Module): The model to be trained.
        args (argparse.Namespace): The parsed arguments.
    Returns:
        The optimizer.
    """
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
    r"""
    Returns the scheduler to be used for training.
    Args:
        optimizer (torch.optim.Optimizer): The optimizer to be used for training.
        args (argparse.Namespace): The parsed arguments.
    Returns:
        The scheduler.
    """
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