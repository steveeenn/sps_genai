import torch
import numpy as np


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")