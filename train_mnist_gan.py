import os
import torch

from helper_lib.data_loader import get_data_loader
from helper_lib.model import get_model
from helper_lib.trainer import train_gan
from helper_lib.utils import get_device, set_seed


set_seed(42)
device = get_device()
print("Using device:", device)

train_loader = get_data_loader(
    data_dir="./data",
    batch_size=128,
    train=True,
    dataset_name="mnist",
    download=True,
)

gan = get_model("GAN")

trained_gan = train_gan(
    model=gan,
    data_loader=train_loader,
    device=device,
    epochs=5,
)

os.makedirs("checkpoints", exist_ok=True)
torch.save(
    {"model_state_dict": trained_gan.state_dict()},
    "checkpoints/mnist_gan.pth",
)

print("Saved MNIST GAN checkpoint to checkpoints/mnist_gan.pth")