import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_dataset
from torchvision.transforms import Compose, Resize, ToTensor
import torchvision


def get_datasets_and_loaders(dataset, batch_size, num_workers):
    if dataset == "flowers":
        transforms = Resize((256, 256))
        tensor_transforms = ToTensor()

        def preprocess_function(examples):
            examples["image"] = [
                transforms(image.convert("RGB")) for image in examples["image"]
            ]
            return examples

        def final_transform(batch):
            batch["image"] = [tensor_transforms(img) for img in batch["image"]]
            return batch

        train_dataset = load_dataset("dpdl-benchmark/oxford_flowers102", split="train")
        train_dataset = train_dataset.map(
            preprocess_function, batched=True, batch_size=64
        )
        val_dataset = load_dataset(
            "dpdl-benchmark/oxford_flowers102", split="validation"
        )
        val_dataset = val_dataset.map(preprocess_function, batched=True, batch_size=64)

        train_dataset.set_transform(final_transform)
        val_dataset.set_transform(final_transform)

        n_channels = 3
        img_size = 256

    elif dataset == "mnist":
        transforms = ToTensor()

        def preprocess_function(examples):
            examples["image"] = [transforms(image) for image in examples["image"]]
            return examples

        train_dataset = load_dataset("ylecun/mnist", split="train")
        train_dataset = train_dataset.map(
            preprocess_function, batched=True, batch_size=64
        )
        train_dataset.set_format("torch")
        val_dataset = load_dataset("ylecun/mnist", split="test")
        val_dataset = val_dataset.map(preprocess_function, batched=True, batch_size=64)
        val_dataset.set_format("torch")

        n_channels = 1
        img_size = 28
    elif dataset == "birds":

        pil_transforms = Compose(
            [
                Resize((256, 256)),
            ]
        )

        def preprocess_pil(examples):
            examples["image"] = [
                pil_transforms(image.convert("RGB")) for image in examples["image"]
            ]
            return examples

        train_dataset = load_dataset(
            "bentrevett/caltech-ucsd-birds-200-2011",
            split="train",
            columns=["image", "label"],
        )
        train_dataset = train_dataset.map(
            preprocess_pil, batched=True, num_proc=os.cpu_count()
        )

        val_dataset = load_dataset(
            "bentrevett/caltech-ucsd-birds-200-2011",
            split="test",
            columns=["image", "label"],
        )
        val_dataset = val_dataset.map(
            preprocess_pil, batched=True, num_proc=os.cpu_count()
        )

        tensor_transforms = ToTensor()

        def final_transform(batch):
            batch["image"] = [tensor_transforms(img) for img in batch["image"]]
            return batch

        train_dataset.set_transform(final_transform)
        val_dataset.set_transform(final_transform)

        n_channels = 3
        img_size = 256

    elif dataset == "animals":

        transforms = Compose([Resize((64, 64)), ToTensor()])

        def preprocess_function(examples):
            examples["image"] = [
                transforms(image.convert("RGB")) for image in examples["image"]
            ]
            return examples

        full_dataset = load_dataset("dgrnd4/animals-10", split="train")
        full_dataset.set_transform(preprocess_function)
        train_val_split = full_dataset.train_test_split(test_size=0.1)
        train_dataset = train_val_split["train"]
        val_dataset = train_val_split["test"]
        n_channels = 3
        img_size = 64

    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    print(f"Dataset: {dataset.upper()}")
    print(f"Train data shape: {train_dataset.data.shape}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, n_channels, img_size


def show_reconstructions(
    model, dataloader, n_images, n_channels, device, current_epoch, save_plots_dir
):
    model.eval()
    data_iter = iter(dataloader)
    try:
        data = next(data_iter)
    except StopIteration:
        print("Dataloader exhausted for reconstructions.")
        return

    images = data["image"][:n_images].to(device)

    with torch.no_grad():
        reconstructions, _, _, _ = model(images)
    reconstructions = reconstructions.cpu()
    images = images.cpu()

    fig, axes = plt.subplots(2, n_images, figsize=(n_images * 1.5, 3))
    for i in range(min(n_images, images.shape[0])):
        ax_orig = axes[0, i]
        img_orig = (
            images[i].squeeze(0) if n_channels == 1 else images[i].permute(1, 2, 0)
        )
        cmap_orig = "gray" if n_channels == 1 else None
        ax_orig.imshow(img_orig.numpy(), cmap=cmap_orig)
        ax_orig.axis("off")
        if i == 0:
            ax_orig.set_title("Original")

        ax_recon = axes[1, i]
        img_recon = (
            reconstructions[i].squeeze(0)
            if n_channels == 1
            else reconstructions[i].permute(1, 2, 0)
        )
        cmap_recon = "gray" if n_channels == 1 else None
        ax_recon.imshow(img_recon.numpy(), cmap=cmap_recon)
        ax_recon.axis("off")
        if i == 0:
            ax_recon.set_title("Reconstructed")

    plt.tight_layout()
    plt.savefig(f"{save_plots_dir}/reconstructions_epoch{current_epoch}.png")
    plt.close(fig)


def show_latent_codes(
    model,
    dataloader,
    num_samples,
    num_embeddings,
    device,
    save_plots_dir,
    current_epoch,
):
    model.eval()
    all_indices = []
    count = 0
    with torch.no_grad():
        for data in dataloader:
            if count >= num_samples:
                break
            img = data["image"].to(device)
            latent_indices = model.get_latent_representation(
                img
            )  # (B, H_latent, W_latent)
            all_indices.append(latent_indices.cpu().numpy().flatten())
            count += img.size(0)
            if (
                len(all_indices)
                * img.size(0)
                * latent_indices.shape[1]
                * latent_indices.shape[2]
                > 2e7
            ):  # Avoid memory issues
                print("Limiting latent code histogram due to large number of codes.")
                break

    if not all_indices:
        print("No latent codes collected for histogram.")
        return

    all_indices_flat = np.concatenate(all_indices)
    plt.figure(figsize=(10, 4))
    plt.hist(
        all_indices_flat,
        bins=num_embeddings,
        range=(-0.5, num_embeddings - 0.5),
        rwidth=0.8,
    )
    plt.title(f"Histogram of Latent Codebook Usage (K={num_embeddings})")
    plt.xlabel("Codebook Index")
    plt.ylabel("Frequency")
    plt.savefig(f"{save_plots_dir}/latent_codes_epoch{current_epoch}.png")
    plt.close()


def generate_random_samples(
    model,
    n_channels,
    latent_h,
    latent_w,
    num_samples,
    device,
    save_plots_dir,
    current_epoch,
):
    model.eval()
    random_indices = torch.randint(
        0,
        model.quantizer.num_embeddings,
        (num_samples, latent_h, latent_w),
        device=device,
    )
    with torch.no_grad():
        generated_images = model.reconstruct_from_indices(random_indices)
    generated_images = generated_images.cpu()

    fig, axes = plt.subplots(1, num_samples, figsize=(num_samples * 1.5, 1.5))
    if num_samples == 1:
        axes = [axes]  # type: ignore

    for i in range(num_samples):
        ax = axes[i]  # type: ignore

        img_gen = (
            generated_images[i].squeeze(0)
            if n_channels == 1
            else generated_images[i].permute(1, 2, 0)
        )
        cmap_gen = "gray" if n_channels == 1 else None
        ax.imshow(img_gen.numpy(), cmap=cmap_gen)
        ax.axis("off")
    plt.suptitle("Generated from Random Sample in Latent Space")
    plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))
    plt.savefig(f"{save_plots_dir}/random_samples_epoch{current_epoch}.png")
    plt.close(fig)


def save_training_loss(
    train_vq_loss,
    train_recon_loss,
    train_perplexity_loss,
    val_vq_loss,
    val_recon_loss,
    val_perplexity_loss,
    train_gan_g_loss,
    train_gan_d_loss,
    val_gan_g_loss,
    val_gan_d_loss,
    save_plots_dir,
):
    num_epochs = len(train_recon_loss)
    epochs_range = range(1, num_epochs + 1)

    plt.figure(figsize=(18, 10))

    # 1. Reconstruction Loss
    plt.subplot(2, 3, 1)
    plt.plot(epochs_range, train_recon_loss, label="Train Recon Loss")
    plt.plot(epochs_range, val_recon_loss, "--", label="Validation Recon Loss")
    plt.title("Reconstruction Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)

    # 2. VQ Loss
    plt.subplot(2, 3, 2)
    plt.plot(epochs_range, train_vq_loss, label="Train VQ Loss")
    plt.plot(epochs_range, val_vq_loss, "--", label="Validation VQ Loss")
    plt.title("VQ Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)

    # 3. Perplexity
    plt.subplot(2, 3, 3)
    plt.plot(epochs_range, train_perplexity_loss, label="Train Perplexity")
    plt.plot(epochs_range, val_perplexity_loss, "--", label="Validation Perplexity")
    plt.title("Codebook Perplexity")
    plt.xlabel("Epochs")
    plt.ylabel("Perplexity")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)

    # 4. Generator's Adversarial Loss
    plt.subplot(2, 3, 4)
    plt.plot(epochs_range, train_gan_g_loss, label="Train G Adversarial Loss")
    plt.plot(epochs_range, val_gan_g_loss, "--", label="Validation G Adversarial Loss")
    plt.title("Generator Adversarial Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)

    # 5. Discriminator's Loss
    plt.subplot(2, 3, 5)
    plt.xlim(num_epochs - len(train_gan_d_loss), num_epochs)
    plt.plot(epochs_range, train_gan_d_loss, label="Train D Loss")
    plt.plot(epochs_range, val_gan_d_loss, "--", label="Validation D Loss")
    plt.title("Discriminator Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()
    plot_path = f"{save_plots_dir}/loss_curves.png"
    plt.savefig(plot_path)
    print(f"Saved loss curves to {plot_path}")
    plt.close()


@torch.no_grad()
def generate_text_conditioned(
    model,
    tokens_ids,
    class_ids,
    generation_length,
    temperature,
    top_k,
):
    for _ in range(generation_length):
        logits = model(tokens_ids[:, -model.block_size :], class_ids)
        next_token_logits = logits[:, -1]
        probs = F.softmax(next_token_logits / temperature, dim=-1)

        if top_k is not None:
            values, _ = torch.topk(probs, k=top_k)
            probs[probs < values[:, -1, None]] = -float("Inf")
            sum_probs = torch.sum(probs, dim=1, keepdim=True)
            sum_probs[sum_probs <= 0] = 1.0
            probs = probs / sum_probs

        cur_tokens = torch.multinomial(probs, num_samples=1)
        tokens_ids = torch.cat([tokens_ids, cur_tokens], dim=-1)
    return tokens_ids



def visualize_codebook_tiled(model, latent_h, latent_w, save_path, device):
    """
    Visualizes the VQ-GAN codebook by decoding a full grid of each embedding vector.

    Args:
        model (nn.Module): The trained VQ-GAN model.
        latent_h (int): The height of the latent grid produced by the encoder.
        latent_w (int): The width of the latent grid produced by the encoder.
        save_path (str): Path to save the output grid image.
        device (torch.device): The device to run the model on.
    """
    model.eval()

    # 1. Get the codebook embedding matrix
    codebook_vectors = model.quantizer.embedding.weight.data
    num_embeddings, embedding_dim = codebook_vectors.shape

    print(
        f"Visualizing {num_embeddings} codebook vectors with latent grid size {latent_h}x{latent_w}..."
    )

    all_decoded_images = []

    # 2. Iterate through each codebook vector
    for i in range(num_embeddings):
        # Get the i-th vector
        vector = codebook_vectors[i]

        # 3. Create a full latent grid filled with this single vector
        # The decoder expects shape (B, C, H, W)
        # We start with shape (C) -> (1, C, 1, 1) -> (1, C, H, W)
        latent_grid = vector.view(1, embedding_dim, 1, 1).repeat(
            1, 1, latent_h, latent_w
        )

        # 4. Decode the grid
        with torch.no_grad():
            decoded_image = model.decoder(latent_grid.to(device))

        # Add the single image (removing the batch dimension) to our list
        all_decoded_images.append(decoded_image.squeeze(0))

    # 5. Arrange all the generated images into a single grid
    # Clamp the output to [0, 1] range for visualization
    decoded_images_tensor = torch.stack(all_decoded_images)

    nrow = int(np.ceil(num_embeddings**0.5))
    grid = torchvision.utils.make_grid(
        decoded_images_tensor, nrow=nrow, padding=2, normalize=False
    )

    # 6. Plot and save
    # Calculate a reasonable figure size
    fig_w = max(12, nrow * 1.5)
    fig_h = max(12, (num_embeddings // nrow) * 1.5)

    plt.figure(figsize=(fig_w, fig_h))
    plt.imshow(grid.permute(1, 2, 0).cpu().numpy())
    plt.title(f"VQ-GAN Tiled Codebook Visualization (K={num_embeddings})")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"Tiled codebook visualization saved to {save_path}")


def plot_gpt_loss(train_losses, val_losses, save_dir):
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, "--", label="Validation Loss")
    plt.title("Loss Curves")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "cond_prior_loss.png"))
    plt.close()
