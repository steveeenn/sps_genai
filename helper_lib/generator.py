import torchvision.transforms as transforms
from torchvision.utils import make_grid
import torch
import matplotlib.pyplot as plt


def generate_samples(model, device, num_samples=10):
    model.to(device)
    model.eval()

    with torch.no_grad():
        if hasattr(model, "generator"):
            z_dim = getattr(model, "z_dim", 100)
            noise = torch.randn(num_samples, z_dim, 1, 1).to(device)
            samples = model.generator(noise).detach().cpu()
            grid = make_grid(samples, normalize=True)

            plt.imshow(grid.permute(1, 2, 0))
            plt.axis("off")
            plt.show()
            return

        z = torch.randn(num_samples, 2).to(device)
        samples = model.decoder(z)
        samples = samples.cpu().numpy()

    rows = 2
    cols = (num_samples + 1) // 2

    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 4))
    axes = axes.flatten()

    for i in range(num_samples):
        axes[i].imshow(samples[i].squeeze(), cmap="gray")
        axes[i].axis("off")

    for i in range(num_samples, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()


def generate_sample_image(model, device):
    model.to(device)
    model.eval()

    with torch.no_grad():
        z_dim = getattr(model, "z_dim", 100)
        noise = torch.randn(1, z_dim, 1, 1).to(device)
        sample = model.generator(noise).detach().cpu().squeeze(0)

    sample = (sample * 0.5) + 0.5
    sample = sample.clamp(0, 1)

    return transforms.ToPILImage()(sample)