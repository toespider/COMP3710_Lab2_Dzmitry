import argparse
import random
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Dataset
# ============================================================

class BrainSliceDataset(Dataset):
    """
    Loads the preprocessed OASIS PNG brain slices.

    Images are:
        256 x 256
        grayscale
        8-bit

    They are converted to:
        float32
        range [0, 1]
        shape [1, 256, 256]
    """

    def __init__(self, root_dir):
        self.root_dir = Path(root_dir)
        self.files = sorted(self.root_dir.glob("*.png"))

        if len(self.files) == 0:
            raise RuntimeError(f"No PNG files found in {self.root_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        path = self.files[index]

        image = Image.open(path).convert("L")
        image = np.asarray(image, dtype=np.float32) / 255.0

        tensor = torch.from_numpy(image).unsqueeze(0)

        return tensor


# ============================================================
# Variational Autoencoder
# ============================================================

class VAE(nn.Module):

    def __init__(self, latent_dim=2):
        super().__init__()

        self.latent_dim = latent_dim

        # ----------------------------------------------------
        # Encoder
        #
        # 256 -> 128 -> 64 -> 32 -> 16
        # ----------------------------------------------------

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

        # 256 channels x 16 x 16
        self.flatten_dim = 256 * 16 * 16

        # Encoder outputs distribution parameters.
        self.fc_mu = nn.Linear(self.flatten_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_dim, latent_dim)

        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------

        self.decoder_input = nn.Linear(
            latent_dim,
            self.flatten_dim
        )

        # 16 -> 32 -> 64 -> 128 -> 256
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                256, 128,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                128, 64,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                64, 32,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(
                32, 1,
                kernel_size=4,
                stride=2,
                padding=1
            ),

            # Pixel values are in [0, 1]
            nn.Sigmoid()
        )

    def encode(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Reparameterisation trick:

            z = mu + sigma * epsilon

        where:

            epsilon ~ N(0, I)
        """

        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)

        return mu + std * epsilon

    def decode(self, z):
        x = self.decoder_input(z)

        x = x.view(
            -1,
            256,
            16,
            16
        )

        return self.decoder(x)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)

        return reconstruction, mu, logvar


# ============================================================
# VAE loss
# ============================================================

def vae_loss(
    reconstruction,
    target,
    mu,
    logvar,
    beta=0.001
):
    """
    Total VAE loss:

        L = reconstruction loss + beta * KL divergence
    """

    # MSE is appropriate for continuous grayscale MRI
    reconstruction_loss = F.mse_loss(
        reconstruction,
        target,
        reduction="mean"
    )

    # KL[q(z|x) || N(0,I)]
    kl_loss = -0.5 * torch.mean(
        1
        + logvar
        - mu.pow(2)
        - logvar.exp()
    )

    total_loss = reconstruction_loss + beta * kl_loss

    return total_loss, reconstruction_loss, kl_loss


# ============================================================
# Training
# ============================================================

def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    beta
):
    model.train()

    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0

    for images in loader:

        images = images.to(
            device,
            non_blocking=True
        )

        optimizer.zero_grad(set_to_none=True)

        reconstruction, mu, logvar = model(images)

        loss, recon_loss, kl_loss = vae_loss(
            reconstruction,
            images,
            mu,
            logvar,
            beta
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()

    n = len(loader)

    return (
        total_loss / n,
        total_recon / n,
        total_kl / n
    )


# ============================================================
# Validation
# ============================================================

@torch.no_grad()
def validate(
    model,
    loader,
    device,
    beta
):
    model.eval()

    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0

    for images in loader:

        images = images.to(
            device,
            non_blocking=True
        )

        # For validation use the mean of q(z|x)
        # rather than random sampling.
        mu, logvar = model.encode(images)
        reconstruction = model.decode(mu)

        loss, recon_loss, kl_loss = vae_loss(
            reconstruction,
            images,
            mu,
            logvar,
            beta
        )

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()

    n = len(loader)

    return (
        total_loss / n,
        total_recon / n,
        total_kl / n
    )


# ============================================================
# Reconstruction visualisation
# ============================================================

@torch.no_grad()
def save_reconstructions(
    model,
    loader,
    device,
    output_path,
    n_images=8
):
    model.eval()

    images = next(iter(loader))
    images = images[:n_images].to(device)

    mu, _ = model.encode(images)
    reconstructions = model.decode(mu)

    images = images.cpu().numpy()
    reconstructions = reconstructions.cpu().numpy()

    fig, axes = plt.subplots(
        2,
        n_images,
        figsize=(16, 5)
    )

    for i in range(n_images):

        axes[0, i].imshow(
            images[i, 0],
            cmap="gray"
        )
        axes[0, i].axis("off")

        axes[1, i].imshow(
            reconstructions[i, 0],
            cmap="gray"
        )
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel(
        "Original",
        fontsize=12
    )

    axes[1, 0].set_ylabel(
        "Reconstruction",
        fontsize=12
    )

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )
    plt.close()


# ============================================================
# Latent-space visualisation
# ============================================================

@torch.no_grad()
def save_latent_space(
    model,
    loader,
    device,
    output_path
):
    model.eval()

    all_mu = []

    for images in loader:

        images = images.to(
            device,
            non_blocking=True
        )

        mu, _ = model.encode(images)

        all_mu.append(mu.cpu())

    latent = torch.cat(
        all_mu,
        dim=0
    ).numpy()

    if latent.shape[1] != 2:
        print(
            "Latent space plot skipped because "
            f"latent dimension = {latent.shape[1]}"
        )
        return

    plt.figure(figsize=(8, 7))

    plt.scatter(
        latent[:, 0],
        latent[:, 1],
        s=8,
        alpha=0.5
    )

    plt.xlabel("Latent dimension 1")
    plt.ylabel("Latent dimension 2")
    plt.title("VAE Latent Space")

    plt.grid(alpha=0.2)

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# Sample the learned manifold
# ============================================================

@torch.no_grad()
def save_manifold(
    model,
    device,
    output_path,
    grid_size=10,
    min_value=-3.0,
    max_value=3.0
):
    model.eval()

    values = torch.linspace(
        min_value,
        max_value,
        grid_size,
        device=device
    )

    latent_vectors = []

    for y in reversed(values):
        for x in values:
            latent_vectors.append(
                torch.tensor(
                    [x.item(), y.item()],
                    device=device
                )
            )

    latent_vectors = torch.stack(
        latent_vectors
    )

    generated = model.decode(
        latent_vectors
    )

    generated = generated.cpu().numpy()

    fig, axes = plt.subplots(
        grid_size,
        grid_size,
        figsize=(14, 14)
    )

    for i in range(grid_size):
        for j in range(grid_size):

            index = i * grid_size + j

            axes[i, j].imshow(
                generated[index, 0],
                cmap="gray"
            )

            axes[i, j].axis("off")

    plt.suptitle(
        "VAE Learned 2D Manifold",
        fontsize=18
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# Loss plot
# ============================================================

def save_loss_plot(
    history,
    output_path
):
    epochs = range(
        1,
        len(history["train_loss"]) + 1
    )

    plt.figure(figsize=(10, 6))

    plt.plot(
        epochs,
        history["train_loss"],
        label="Train"
    )

    plt.plot(
        epochs,
        history["val_loss"],
        label="Validation"
    )

    plt.xlabel("Epoch")
    plt.ylabel("VAE loss")
    plt.title("Training and Validation Loss")

    plt.legend()
    plt.grid(alpha=0.2)

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
        default="/home/groups/comp3710/OASIS"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64
    )

    parser.add_argument(
        "--latent-dim",
        type=int,
        default=2
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3
    )

    parser.add_argument(
        "--beta",
        type=float,
        default=0.001
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=6
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Setup
    # --------------------------------------------------------

    set_seed(args.seed)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 60)
    print("VAE training")
    print("=" * 60)
    print(f"Device: {device}")

    if device.type == "cuda":
        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    print(f"Latent dimension: {args.latent_dim}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Beta: {args.beta}")

    # --------------------------------------------------------
    # Directories
    # --------------------------------------------------------

    output_dir = Path("outputs")
    checkpoint_dir = Path("checkpoints")

    output_dir.mkdir(
        exist_ok=True
    )

    checkpoint_dir.mkdir(
        exist_ok=True
    )

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------

    train_dataset = BrainSliceDataset(
        Path(args.data_root)
        / "keras_png_slices_train"
    )

    val_dataset = BrainSliceDataset(
        Path(args.data_root)
        / "keras_png_slices_validate"
    )

    test_dataset = BrainSliceDataset(
        Path(args.data_root)
        / "keras_png_slices_test"
    )

    print()
    print(f"Training images:   {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")
    print(f"Test images:       {len(test_dataset)}")

    # --------------------------------------------------------
    # Data loaders
    # --------------------------------------------------------

    use_cuda = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=use_cuda,
        persistent_workers=args.workers > 0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=use_cuda,
        persistent_workers=args.workers > 0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=use_cuda,
        persistent_workers=args.workers > 0
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = VAE(
        latent_dim=args.latent_dim
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_recon": [],
        "val_recon": [],
        "train_kl": [],
        "val_kl": []
    }

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):

        train_loss, train_recon, train_kl = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.beta
        )

        val_loss, val_recon, val_kl = validate(
            model,
            val_loader,
            device,
            args.beta
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        history["train_recon"].append(train_recon)
        history["val_recon"].append(val_recon)

        history["train_kl"].append(train_kl)
        history["val_kl"].append(val_kl)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"Train {train_loss:.6f} | "
            f"Val {val_loss:.6f} | "
            f"Recon {val_recon:.6f} | "
            f"KL {val_kl:.6f}"
        )

        # Save best checkpoint
        if val_loss < best_val_loss:

            best_val_loss = val_loss

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
                "latent_dim": args.latent_dim
            }

            torch.save(
                checkpoint,
                checkpoint_dir / "vae_best.pt"
            )

    # --------------------------------------------------------
    # Save loss plot
    # --------------------------------------------------------

    save_loss_plot(
        history,
        output_dir / "loss.png"
    )

    # --------------------------------------------------------
    # Load best model
    # --------------------------------------------------------

    checkpoint = torch.load(
        checkpoint_dir / "vae_best.pt",
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print()
    print(
        f"Best model came from epoch "
        f"{checkpoint['epoch']}"
    )

    # --------------------------------------------------------
    # Final visualisations
    # --------------------------------------------------------

    save_reconstructions(
        model,
        test_loader,
        device,
        output_dir / "reconstructions.png"
    )

    save_latent_space(
        model,
        test_loader,
        device,
        output_dir / "latent_space.png"
    )

    if args.latent_dim == 2:

        save_manifold(
            model,
            device,
            output_dir / "manifold.png"
        )

    print()
    print("Finished.")
    print("Outputs:")
    print("  outputs/loss.png")
    print("  outputs/reconstructions.png")
    print("  outputs/latent_space.png")

    if args.latent_dim == 2:
        print("  outputs/manifold.png")

    print("Checkpoint:")
    print("  checkpoints/vae_best.pt")


if __name__ == "__main__":
    main()