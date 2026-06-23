import torch
import matplotlib.pyplot as plt


def generate_samples(model, device, num_samples=10):
    model.to(device)
    model.eval()

    with torch.no_grad():
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