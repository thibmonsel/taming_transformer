"""
Adapted from https://github.com/thibmonsel/vqGAN/blob/master/train_conditioned_prior.py
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from taming_transformer.misc import generate_text_conditioned, get_datasets_and_loaders, visualize_codebook_tiled, plot_gpt_loss
from taming_transformer.models.transformer import ConditionnedGPT
from taming_transformer.models.vq_gan import VectorQuantizedGAN


def train(model, data_loader, prior, optimizer, scheduler, device):
    train_loss = 0.0
    prior.train()
    for data in data_loader:
        with torch.no_grad():
            images = data["image"].to(device)

            _, _, indices, _ = model(images)
            indices = indices.detach()
            flat_indices = indices.view(images.shape[0], -1)
            prior_input_indices = flat_indices[:, :-1].contiguous()
            prior_target_indices = flat_indices[:, 1:].contiguous()

        labels = data["label"].to(device)
        logits = prior(prior_input_indices, labels)
        optimizer.zero_grad()

        loss = F.cross_entropy(
            logits.reshape(-1, model.quantizer.num_embeddings),
            prior_target_indices.reshape(-1),
        )

        loss.backward()
        train_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(prior.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    train_loss /= len(data_loader)
    return train_loss


def test(model, data_loader, prior, device):
    test_loss = 0.0
    prior.eval()
    with torch.no_grad():
        for data in data_loader:
            images = data["image"].to(device)
            _, _, indices, _ = model(images)
            indices = indices.detach()
            flat_indices = indices.view(images.shape[0], -1)
            prior_input_indices = flat_indices[:, :-1].contiguous()
            prior_target_indices = flat_indices[:, 1:].contiguous()

            labels = data["label"].to(device)
            logits = prior(prior_input_indices, labels)  # type: ignore
            loss = F.cross_entropy(
                logits.view(-1, model.quantizer.num_embeddings),
                prior_target_indices.view(-1),  # type: ignore
            )
            test_loss += loss.item()

    test_loss /= len(data_loader)
    return test_loss


def sample_from_prior(
    prior,
    model,
    device,
    latent_h_vq,
    latent_w_vq,
    n_channels,
    epoch,
    save_dir,
    n_samples=5,
):
    random_token_ids = torch.randint(
        0,
        model.quantizer.num_embeddings,
        (prior.n_classes, 1),
        device=device,
    )
    random_class_ids = torch.arange(
        0,
        prior.n_classes,
        device=device,
    )
    generated_indices = generate_text_conditioned(
        prior,
        random_token_ids,
        random_class_ids,
        generation_length=latent_h_vq * latent_w_vq - 1,
        temperature=1.0,
        top_k=None,
    )

    generated_indices = generated_indices.reshape(-1, latent_h_vq, latent_w_vq)
    generated_images = model.reconstruct_from_indices(generated_indices)

    cmap_gen = "gray" if n_channels == 1 else None
    for i in range(n_samples):
        plt.subplot(1, n_samples, i + 1)
        img = generated_images[i].detach().cpu()
        img = img.squeeze(dim=0) if n_channels == 1 else img.permute(1, 2, 0)
        plt.imshow(img.numpy(), cmap=cmap_gen)
        plt.title(f"{random_class_ids[i].item()}")
        plt.axis("off")
    plt.suptitle("Classes sampled from the conditional prior")
    plt.tight_layout()
    plt.savefig(
        os.path.join(save_dir, f"generated_samples_from_cond_prior_epoch_{epoch}.png")
    )
    plt.close()


def main(args):
    if args.deterministic:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        np.random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device(
        args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    )
    print(f"Using device: {device}")

    save_dir = os.path.join(args.dataset, args.save_plots_dir)
    os.makedirs(save_dir, exist_ok=True)

    train_loader, test_loader, n_channels, img_size = get_datasets_and_loaders(
        args.dataset, args.batch_size, args.num_workers
    )

    model = VectorQuantizedGAN(
        n_channels,
        img_size,
        args.base_channels,
        args.n_blocks,
        args.latent_channels,
        args.num_embeddings,
        args.commitment_cost,
        args.channel_multipliers,
    ).to(device)

    model.load_state_dict(
        torch.load(
            os.path.join(args.dataset, args.trained_vqgan_path) + "/best_vqvae.pt",
            map_location=device,
        )
    )
    model = model.to(device)
    model.eval()

    print("VQ-GAN model loaded and codebook visualized.")

    dataset_obj = train_loader.dataset
    num_classes = len(dataset_obj["label"].unique())

    # Determine latent dimensions
    dummy_input = torch.randn(1, n_channels, img_size, img_size).to(device)
    with torch.no_grad():
        encoder_output_shape = model.encoder(dummy_input).shape
    latent_h_vq, latent_w_vq = encoder_output_shape[2], encoder_output_shape[3]
    print(f"Latent grid dimensions after encoder: {latent_h_vq}x{latent_w_vq}")

    assert (
        args.block_size >= latent_h_vq * latent_w_vq
    ), "Block size must be greater than or equal to the product of latent dimensions."

    visualize_codebook_tiled(
        model,
        latent_h_vq,
        latent_w_vq,
        os.path.join(save_dir, "vqgan_codebook_tiled.png"),
        device,
    )
    print("Codebook visualization complete.")
    prior = ConditionnedGPT(
        args.num_embeddings,
        num_classes,
        args.block_size,
        args.gpt_embedding_dim,
        args.gpt_embedding_dim,
        args.n_heads,
        args.n_layers,
        args.dropout,
        True,
    )
    prior = prior.to(device)

    optimizer = optim.AdamW(prior.parameters(), lr=args.lr, weight_decay=1e-8)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, args.lr, epochs=args.max_epochs, steps_per_epoch=len(train_loader)
    )

    nb_params = sum(p.numel() for p in prior.parameters() if p.requires_grad)
    print(f"Prior Model Parameters: {nb_params} ")

    best_val_loss = float("inf")
    train_losses, val_losses = [], []
    for epoch in range(args.max_epochs):
        print(f"Epoch {epoch + 1}/{args.max_epochs}")

        # Training
        epoch_train_loss = train(
            model, train_loader, prior, optimizer, scheduler, device
        )
        train_losses.append(epoch_train_loss)
        print(
            "====> Epoch: {} Average Train Loss: {:.4f}".format(
                epoch + 1,
                train_losses[-1],
            )
        )

        # Validation
        epoch_val_loss = test(model, test_loader, prior, device)
        val_losses.append(epoch_val_loss)

        # Plotting loss and saving some samples during training
        if epoch % args.save_every == 0:
            sample_from_prior(
                prior,
                model,
                device,
                latent_h_vq,
                latent_w_vq,
                n_channels,
                epoch,
                save_dir,
                n_samples=num_classes,
            )
            plot_gpt_loss(train_losses, val_losses, save_dir)

        # Saving the best model
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            print(
                "====> Best test set reconstruction loss",
                f"seen so far: {best_val_loss:.4f}",
            )
            if epoch > args.max_epochs // 2:
                print("Saving model...")
                torch.save(prior.state_dict(), os.path.join(save_dir, "best_prior.pt"))
                print("Model saved.")

    print("Training complete.")
    print("Finalizing...")

    torch.save(
        prior.state_dict(),
        os.path.join(save_dir, f"last_cond_prior_epoch_{args.max_epochs}.pt"),
    )
    plot_gpt_loss(train_losses, val_losses, save_dir)

    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VQ-GAN training and visualization script"
    )
    # Data Handling arguments
    parser.add_argument(
        "--dataset",
        type=str,
        default="mnist",
        choices=["mnist", "fashion_mnist", "cifar10"],
        help="Dataset to use",
    )
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Input batch size for training"
    )
    parser.add_argument(
        "--num-workers", type=int, default=2, help="Number of workers for data loading"
    )

    # VQ-GAN Model arguments
    parser.add_argument(
        "--latent-channels",
        type=int,
        default=64,
        help="Dimensionality of VQ embedding vectors",
    )
    parser.add_argument(
        "--num-embeddings",
        type=int,
        default=128,
        help="Number of VQ embedding vectors (codebook size K)",
    )

    parser.add_argument(
        "--commitment-cost",
        type=float,
        default=1.0,
        help="Commitment cost (beta) for VQ loss",
    )
    parser.add_argument(
        "--base-channels",
        type=int,
        default=64,
        help="Base filter in Encoder/Decoder CNNs",
    )
    parser.add_argument(
        "--n-blocks",
        type=int,
        default=4,
        help="Number of blocks in Encoder/Decoder CNNs",
    )
    parser.add_argument(
        "--channel-multipliers", type=int, nargs="+", help="base filter multipliers"
    )
    # Transformer Model arguments
    parser.add_argument(
        "--block-size",
        type=int,
        default=256,
        help="Block size for transformer",
    )

    parser.add_argument(
        "--n-heads",
        type=int,
        default=8,
        help="Number of attention heads in transformer",
    )
    parser.add_argument(
        "--n-layers",
        type=int,
        default=6,
        help="Number of transformer layers",
    )
    parser.add_argument(
        "--gpt-embedding-dim",
        type=int,
        default=64,
        help="Dimensionality of VQ embedding vectors",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate for transformer",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use (cuda or cpu)",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic mode for reproducibility",
    )

    parser.add_argument(
        "--trained-vqgan-path",
        type=str,
        required=True,
        help="Trained VQ-GAN path",
    )

    parser.add_argument(
        "--save-plots-dir",
        type=str,
        required=True,
        help="Directory to save plots. If None, plots are not saved.",
    )

    parser.add_argument(
        "--max-epochs", type=int, default=10, help="Number of epochs to train"
    )

    parser.add_argument(
        "--save-every",
        type=int,
        default=5,
        help="Saving intermediate visualizations/results every N epochs",
    )
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate")

    args = parser.parse_args()

    main(args)
