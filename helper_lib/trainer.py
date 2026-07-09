import torch
import torch.nn.functional as F
from tqdm import tqdm
from .checkpoints import save_checkpoint
from .evaluator import evaluate_model


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


def train_gan(model, data_loader, criterion=None, optimizer=None, device="cpu", epochs=10):
    model.to(device)
    model.train()

    z_dim = getattr(model, "z_dim", 100)
    lr = 5e-5
    n_critic = 5
    clip_value = 0.01

    if optimizer is None:
        opt_gen = torch.optim.RMSprop(model.generator.parameters(), lr=lr)
        opt_critic = torch.optim.RMSprop(model.critic.parameters(), lr=lr)
    else:
        opt_gen, opt_critic = optimizer

    datalogs = []

    for epoch in range(epochs):
        train_loader_with_progress = tqdm(
            iterable=data_loader,
            ncols=120,
            desc=f"Epoch {epoch + 1}/{epochs}"
        )

        for batch_number, (real, _) in enumerate(train_loader_with_progress):
            real = real.to(device)
            batch_size = real.size(0)

            for _ in range(n_critic):
                noise = torch.randn(batch_size, z_dim, 1, 1).to(device)
                fake = model.generator(noise).detach()
                critic_real = model.critic(real).mean()
                critic_fake = model.critic(fake).mean()
                loss_critic = -(critic_real - critic_fake)

                model.critic.zero_grad()
                loss_critic.backward()
                opt_critic.step()

                for p in model.critic.parameters():
                    p.data.clamp_(-clip_value, clip_value)

            noise = torch.randn(batch_size, z_dim, 1, 1).to(device)
            fake = model.generator(noise)
            loss_gen = -model.critic(fake).mean()

            model.generator.zero_grad()
            loss_gen.backward()
            opt_gen.step()

            if batch_number % 100 == 0:
                train_loader_with_progress.set_postfix({
                    "Batch": f"{batch_number}/{len(data_loader)}",
                    "D loss": f"{loss_critic.item():.4f}",
                    "G loss": f"{loss_gen.item():.4f}",
                })
                datalogs.append({
                    "epoch": epoch + batch_number / len(data_loader),
                    "Batch": batch_number / len(data_loader),
                    "D loss": loss_critic.item(),
                    "G loss": loss_gen.item(),
                })

    return model