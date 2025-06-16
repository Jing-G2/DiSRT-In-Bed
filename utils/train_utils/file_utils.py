import os
import blobfile as bf
from utils import logger


# ######################################
# find file functions
# ######################################
def parse_resume_step_from_filename(filename):
    """Parse filenames of the form path/to/model_{steps}.pt"""
    split = filename.split("_")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_model_path_in_directory(directory, epoch=-1):
    """
    Returns the path to the model file in the given directory.
    If epoch is -1, it returns the latest model file.
    """
    model_files = [file for file in os.listdir(directory) if file.endswith(".pt")]
    if len(model_files) == 0:
        raise FileNotFoundError()

    if epoch == -1:
        model_epochs = sorted(
            [int(file.split(".")[0].split("_")[-1]) for file in model_files]
        )
        epoch = model_epochs[-1]

    model_path = os.path.join(directory, f"model_{epoch}.pt")
    return model_path, epoch


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
