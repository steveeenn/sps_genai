import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from app.diffusion_model import (
    create_diffusion_model,
    offset_cosine_diffusion_schedule,
)
from app.energy_model import EnergyModel, generate_samples


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class SampleBuffer:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.examples = [
            torch.rand((1, 1, 32, 32), device=device) * 2 - 1
            for _ in range(128)
        ]

    def sample(self, batch_size, steps, step_size, noise_std):
        number_new = np.random.binomial(batch_size, 0.05)
        new_images = (
            torch.rand(
                (number_new, 1, 32, 32),
                device=self.device,
            ) * 2 - 1
        )

        number_old = batch_size - number_new
        if number_old > 0:
            old_images = torch.cat(
                random.choices(self.examples, k=number_old),
                dim=0,
            )
            input_images = torch.cat(
                [new_images, old_images],
                dim=0,
            )
        else:
            input_images = new_images

        generated_images = generate_samples(
            self.model,
            input_images,
            steps=steps,
            step_size=step_size,
            noise_std=noise_std,
        )

        self.examples = (
            list(torch.split(generated_images, 1, dim=0))
            + self.examples
        )[:8192]
        return generated_images


def train_energy_model(device, epochs):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        transforms.Pad(2, fill=-1),
    ])
    dataset = torchvision.datasets.MNIST(
        root=DATA_DIR,
        train=True,
        download=True,
        transform=transform,
    )
    train_loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
    )

    model = EnergyModel().to(device)
    buffer = SampleBuffer(model, device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0001,
        betas=(0.0, 0.999),
    )

    for epoch in range(epochs):
        total_loss = 0.0

        for real_images, _ in train_loader:
            real_images = real_images.to(device)
            real_images = (
                real_images
                + torch.randn_like(real_images) * 0.005
            ).clamp(-1.0, 1.0)

            fake_images = buffer.sample(
                batch_size=real_images.size(0),
                steps=60,
                step_size=10.0,
                noise_std=0.005,
            )

            model.train()
            scores = model(
                torch.cat([real_images, fake_images], dim=0)
            )
            real_scores, fake_scores = torch.split(
                scores,
                [real_images.size(0), fake_images.size(0)],
                dim=0,
            )

            contrastive_loss = (
                real_scores.mean() - fake_scores.mean()
            )
            regularization_loss = 0.1 * (
                real_scores.pow(2).mean()
                + fake_scores.pow(2).mean()
            )
            loss = contrastive_loss + regularization_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=0.1,
            )
            optimizer.step()
            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)
        print(
            f"Energy epoch {epoch + 1}/{epochs} "
            f"- loss: {average_loss:.4f}"
        )

    checkpoint_path = CHECKPOINT_DIR / "energy_model.pth"
    torch.save(
        {"model_state_dict": model.state_dict()},
        checkpoint_path,
    )
    print(f"Saved energy model to {checkpoint_path}")


def calculate_normalizer(dataset):
    loader = DataLoader(dataset, batch_size=64, shuffle=False)
    channel_sum = torch.zeros(3)
    channel_squared_sum = torch.zeros(3)
    pixel_count = 0

    for images, _ in loader:
        channel_sum += images.sum(dim=(0, 2, 3))
        channel_squared_sum += images.pow(2).sum(
            dim=(0, 2, 3)
        )
        pixel_count += (
            images.size(0) * images.size(2) * images.size(3)
        )

    mean = channel_sum / pixel_count
    variance = channel_squared_sum / pixel_count - mean.pow(2)
    std = torch.sqrt(variance.clamp_min(1e-6))
    return (
        mean.reshape(1, 3, 1, 1),
        std.reshape(1, 3, 1, 1),
    )


def update_ema(model, decay=0.8):
    with torch.no_grad():
        for ema_parameter, parameter in zip(
            model.ema_network.parameters(),
            model.network.parameters(),
        ):
            ema_parameter.mul_(decay).add_(
                parameter,
                alpha=1.0 - decay,
            )

        for ema_buffer, network_buffer in zip(
            model.ema_network.buffers(),
            model.network.buffers(),
        ):
            ema_buffer.copy_(network_buffer)


def diffusion_loss(model, images, loss_function, training):
    images = (
        images - model.normalizer_mean
    ) / model.normalizer_std
    noises = torch.randn_like(images)
    diffusion_times = torch.rand(
        (images.size(0), 1, 1, 1),
        device=images.device,
    )
    noise_rates, signal_rates = (
        offset_cosine_diffusion_schedule(diffusion_times)
    )
    noisy_images = (
        signal_rates * images + noise_rates * noises
    )

    network = (
        model.network if training else model.ema_network
    )
    predicted_noises = network(
        noisy_images,
        noise_rates ** 2,
    )
    return loss_function(predicted_noises, noises)


def train_diffusion_model(device, epochs):
    transform = transforms.Compose([
        transforms.CenterCrop(340),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
    ])
    train_dataset = torchvision.datasets.Flowers102(
        root=DATA_DIR,
        split="train",
        download=True,
        transform=transform,
    )
    validation_dataset = torchvision.datasets.Flowers102(
        root=DATA_DIR,
        split="val",
        download=True,
        transform=transform,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=64,
        shuffle=False,
    )

    mean, std = calculate_normalizer(train_dataset)
    model = create_diffusion_model().to(device)
    model.set_normalizer(
        mean.to(device),
        std.to(device),
    )

    optimizer = torch.optim.AdamW(
        model.network.parameters(),
        lr=1e-3,
        weight_decay=1e-4,
    )
    loss_function = nn.L1Loss()
    best_validation_loss = float("inf")

    for epoch in range(epochs):
        model.network.train()
        total_train_loss = 0.0

        for images, _ in train_loader:
            images = images.to(device)
            loss = diffusion_loss(
                model,
                images,
                loss_function,
                training=True,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema(model)
            total_train_loss += loss.item()

        model.ema_network.eval()
        total_validation_loss = 0.0
        with torch.no_grad():
            for images, _ in validation_loader:
                images = images.to(device)
                loss = diffusion_loss(
                    model,
                    images,
                    loss_function,
                    training=False,
                )
                total_validation_loss += loss.item()

        average_train_loss = (
            total_train_loss / len(train_loader)
        )
        average_validation_loss = (
            total_validation_loss / len(validation_loader)
        )
        print(
            f"Diffusion epoch {epoch + 1}/{epochs} "
            f"- train loss: {average_train_loss:.4f} "
            f"- validation loss: "
            f"{average_validation_loss:.4f}"
        )

        if average_validation_loss < best_validation_loss:
            best_validation_loss = average_validation_loss
            checkpoint_path = (
                CHECKPOINT_DIR / "diffusion_model.pth"
            )
            torch.save(
                {
                    "model_state_dict":
                        model.ema_network.state_dict(),
                    "normalizer_mean": mean,
                    "normalizer_std": std,
                },
                checkpoint_path,
            )
            print(
                f"Saved diffusion model to "
                f"{checkpoint_path}"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "model",
        choices=["energy", "diffusion", "all"],
        nargs="?",
        default="all",
    )
    parser.add_argument("--energy-epochs", type=int, default=10)
    parser.add_argument("--diffusion-epochs", type=int, default=50)
    args = parser.parse_args()

    set_seed()
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"Using device: {device}")

    if args.model in ("energy", "all"):
        train_energy_model(device, args.energy_epochs)

    if args.model in ("diffusion", "all"):
        train_diffusion_model(device, args.diffusion_epochs)


if __name__ == "__main__":
    main()
