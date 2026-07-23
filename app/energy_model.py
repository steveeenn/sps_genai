from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image


def swish(x):
    return x * torch.sigmoid(x)


class EnergyModel(nn.Module):
    def __init__(self):
        super(EnergyModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 2 * 2, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = swish(self.conv1(x))
        x = swish(self.conv2(x))
        x = swish(self.conv3(x))
        x = swish(self.conv4(x))
        x = self.flatten(x)
        x = swish(self.fc1(x))
        return self.fc2(x)


def generate_samples(
    nn_energy_model,
    inp_imgs,
    steps,
    step_size,
    noise_std,
    generator=None,
):
    nn_energy_model.eval()

    for _ in range(steps):
        with torch.no_grad():
            noise = torch.randn(
                inp_imgs.shape,
                device=inp_imgs.device,
                generator=generator,
            ) * noise_std
            inp_imgs = (inp_imgs + noise).clamp(-1.0, 1.0)

        inp_imgs.requires_grad_(True)

        energy = nn_energy_model(inp_imgs)
        grads, = torch.autograd.grad(
            energy,
            inp_imgs,
            grad_outputs=torch.ones_like(energy),
        )

        with torch.no_grad():
            grads = grads.clamp(-0.03, 0.03)
            inp_imgs = (inp_imgs - step_size * grads).clamp(-1.0, 1.0)

    return inp_imgs.detach()


def load_energy_checkpoint(model, checkpoint_path, device):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        return False

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return True


def generate_energy_image(
    model,
    device,
    steps=256,
    step_size=10.0,
    noise_std=0.01,
    seed=None,
):
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)
    else:
        generator.seed()

    image = torch.rand(
        (1, 1, 32, 32),
        device=device,
        generator=generator,
    ) * 2 - 1

    image = generate_samples(
        model,
        image,
        steps=steps,
        step_size=step_size,
        noise_std=noise_std,
        generator=generator,
    )

    image = torch.clamp((image[0, 0] + 1) / 2, 0, 1)
    image = (image * 255).to(torch.uint8).cpu().numpy()
    return Image.fromarray(image)
