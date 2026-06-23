import torch
import torch.nn.functional as F
from tqdm import tqdm
from checkpoints import save_checkpoint
from evaluator import evaluate_model


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device="cpu",
    epochs=10,
    checkpoint_dir="checkpoints"
):
    model.to(device)

    datalogs = []
    best_accuracy = 0.0

    for epoch in range(epochs):
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        model.train()

        train_loader_with_progress = tqdm(
            iterable=train_loader,
            ncols=120,
            desc=f"Epoch {epoch + 1}/{epochs}"
        )

        for batch_number, (inputs, labels) in enumerate(train_loader_with_progress):
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_correct += (predicted == labels).sum().item()
            running_total += labels.size(0)
            running_loss += loss.item()

            if batch_number % 100 == 99:
                train_loader_with_progress.set_postfix({
                    "avg accuracy": f"{running_correct / running_total:.3f}",
                    "avg loss": f"{running_loss / (batch_number + 1):.4f}"
                })

        train_loss = running_loss / len(train_loader)
        train_accuracy = 100 * running_correct / running_total

        val_loss, val_accuracy = evaluate_model(
            model,
            val_loader,
            criterion,
            device=device
        )

        datalogs.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        })

        checkpoint_path = save_checkpoint(
            model,
            optimizer,
            epoch + 1,
            val_loss,
            val_accuracy,
            checkpoint_dir=checkpoint_dir
        )

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            save_checkpoint(
                model,
                optimizer,
                epoch + 1,
                val_loss,
                val_accuracy,
                checkpoint_dir=f"{checkpoint_dir}/best"
            )

        print(
            f"Epoch {epoch + 1}: "
            f"Train Loss={train_loss:.4f}, Train Accuracy={train_accuracy:.2f}%, "
            f"Val Loss={val_loss:.4f}, Val Accuracy={val_accuracy:.2f}%"
        )

        print(f"Checkpoint saved: {checkpoint_path}")

    print("Finished Training")

    return model


def vae_loss_function(recon_x, x, mu, logvar):
    beta = 500
    bce = F.binary_cross_entropy(recon_x, x, reduction="sum")
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return beta * bce + kld


def train_vae_model(
    model,
    data_loader,
    criterion,
    optimizer,
    device="cpu",
    epochs=10
):
    model.to(device)
    model.train()

    for epoch in range(epochs):
        running_loss = 0.0

        data_loader_with_progress = tqdm(
            iterable=data_loader,
            ncols=120,
            desc=f"Epoch {epoch + 1}/{epochs}"
        )

        for data in data_loader_with_progress:
            inputs = data[0].to(device)

            optimizer.zero_grad()

            recon, mu, logvar = model(inputs)
            loss = criterion(recon, inputs, mu, logvar)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            data_loader_with_progress.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = running_loss / len(data_loader)
        print(f"Epoch {epoch + 1}: Loss={avg_loss:.4f}")

    print("Finished Training")

    return model