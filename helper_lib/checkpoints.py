import torch
import os


def save_checkpoint(model, optimizer, epoch, loss, accuracy, checkpoint_dir="checkpoints"):
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "accuracy": accuracy
    }

    checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch:03d}.pth")
    torch.save(checkpoint, checkpoint_path)

    return checkpoint_path


def load_checkpoint(model, optimizer, checkpoint_path, device="cpu"):
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    accuracy = checkpoint["accuracy"]

    print(f"Loaded checkpoint from epoch {epoch}")
    print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.2f}%")

    return epoch, loss, accuracy