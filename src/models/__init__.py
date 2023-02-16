from models.unet import UNet
from models.hf import HfTransformer


MODELS = {
    "unet": UNet,
    "hf": HfTransformer,
}


def get_model(args):
    # Initialize all other models
    return MODELS[args.model_name](args)
