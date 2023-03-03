from .unet import UNet
from .hf import HfTransformer


MODELS = {
    "unet": UNet,
    "hf": HfTransformer,
}


def get_model(args):
    # Initialize all other models
    return MODELS[args.model.name](args)
