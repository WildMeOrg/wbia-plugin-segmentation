# -*- coding: utf-8 -*-
from argparse import Namespace


def get_default_config():
    cfg = Namespace()

    # management
    cfg.management = Namespace()
    cfg.management.wandb_project_name = "segmentation"
    cfg.management.dir_checkpoint = "./checkpoints"
    cfg.management.save_checkpoint = False
    cfg.management.processing_stage = 'Train' # OR 'Test' OR 'Inference'

    # model
    cfg.model = Namespace()
    cfg.model.name = 'hf'
    cfg.model.hf_path = 'nvidia/mit-b2'
    cfg.model.n_classes = 2
    cfg.model.bilinear = False
    cfg.model.id2label = {0: "background", 1: "foreground"}
    cfg.model.label2id = {"background": 0, "foreground": 1}

    # data
    cfg.data = Namespace()
    cfg.data.source = 'snowleopard_v2'
    cfg.data.inference_dir = './inference'
    cfg.data.mask_suffix = '_mask.png'
    cfg.data.inference_mask_dir = './mask_results'
    cfg.data.training_percent = None
    cfg.data.num_workers = 2
    cfg.data.n_channels = 3
    cfg.data.img_height = 400
    cfg.data.img_width = 400
    cfg.data.transforms_train = ["random_rotation", "random_crop"]
    cfg.data.transforms_test = 'center_crop'
    cfg.data.transforms_inference = 'resize'
    cfg.data.norm_mean = None
    cfg.data.norm_std = None

    # train
    cfg.train = Namespace()
    cfg.train.epochs = 25
    cfg.train.batch_size = 2
    cfg.train.optim = "adamw"
    cfg.train.scheduler = "linear"
    cfg.train.scheduler_patience = 2
    cfg.train.lr = 1e-5
    cfg.train.amp = False

    # test
    cfg.test = Namespace()
    cfg.test.path_to_best = ''
    cfg.test.batch_size = 2

    return cfg
