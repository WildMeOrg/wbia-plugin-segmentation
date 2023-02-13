from models.unet import UNet


MODELS = {
    "unet": UNet,
}


def get_model(args):
    # Initialize all other models
    return MODELS[args.model_name](args)
