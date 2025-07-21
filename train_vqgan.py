import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from taming_transformer.models import VectorQuantizedGAN
from taming_transformer.misc import (
    generate_random_samples,
    get_datasets_and_loaders,
    show_latent_codes,
    show_reconstructions,
    save_training_loss,
)
from taming_transformer.models.perceptual_loss import PerceptualLoss


def hinge_d_loss(logits_real, logits_fake):
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss

def train(
    model,
    perceptual_loss,
    train_loader,
    optimizer_enc_dec,
    optimizer_disc,
    device,
    epoch,
    args,
):
    model.train()

    epoch_percep_loss_avg = 0.0
    epoch_q_loss_avg = 0.0
    epoch_gan_g_loss_avg = 0.0
    epoch_gan_d_loss_avg = 0.0
    epoch_perplexity_avg = 0.0

    for batch_idx, data in enumerate(train_loader):
        img_real = data["image"].to(device)

        # Train VQ-GAN Generator (Encoder/Decoder/Quantizer)
        optimizer_enc_dec.zero_grad()
        recon_batch_g, q_loss, _, perplexity = model(img_real)

        # Perceptual and L1 loss for reconstruction
        recon_loss = F.l1_loss(recon_batch_g, img_real)
        percep_loss = perceptual_loss(recon_batch_g, img_real)

        # The base autoencoder loss
        loss_ae = recon_loss + percep_loss + q_loss

        if epoch >= args.disc_start_epoch:
            logits_fake_g = model.discriminator(recon_batch_g)
            loss_g_adv = -torch.mean(logits_fake_g)
            total_loss_g = loss_ae + args.gan_weight * loss_g_adv
        else:
            # During warm-up, only train the autoencoder
            loss_g_adv = torch.tensor(0.0).to(device)  
            total_loss_g = loss_ae

        total_loss_g.backward()
        optimizer_enc_dec.step()

        # Train Discriminator
        optimizer_disc.zero_grad()

        if epoch >= args.disc_start_epoch:
            with torch.no_grad():
                recon_batch_detached, _, _, _ = model(img_real)

            logits_real = model.discriminator(img_real.contiguous().detach())
            logits_fake_d = model.discriminator(recon_batch_detached)

            loss_d = hinge_d_loss(logits_real, logits_fake_d)
            loss_d.backward()
            optimizer_disc.step()
        else:
            loss_d = torch.tensor(0.0).to(device)

        epoch_percep_loss_avg += percep_loss.item()
        epoch_q_loss_avg += q_loss.item()
        epoch_gan_g_loss_avg += loss_g_adv.item()
        epoch_gan_d_loss_avg += loss_d.item()
        epoch_perplexity_avg += perplexity.item()

        if batch_idx % 10 == 0:
            print(
                f"Iter [{batch_idx * len(img_real)}/{len(train_loader.dataset)}] | "
                f"PercepLoss: {percep_loss.item():.4f} | QL: {q_loss.item():.4f} | "
                f"G_AdvL: {loss_g_adv.item():.4f} | D_L: {loss_d.item():.4f} | "
                f"Perplexity: {perplexity.item():.4f} | GAN W: {args.gan_weight if epoch >= args.disc_start_epoch else 0}"
            )

    num_batches = len(train_loader)
    epoch_percep_loss_avg /= num_batches
    epoch_q_loss_avg /= num_batches
    epoch_gan_g_loss_avg /= num_batches
    epoch_gan_d_loss_avg /= num_batches
    epoch_perplexity_avg /= num_batches

    return (
        epoch_percep_loss_avg,
        epoch_q_loss_avg,
        epoch_gan_g_loss_avg,
        epoch_gan_d_loss_avg,
        epoch_perplexity_avg,
    )


def test(model, perceptual_loss, test_loader, device):
    model.eval()

    avg_percep_loss = 0.0
    avg_q_loss = 0.0
    avg_perplexity = 0.0
    avg_discriminator_loss = 0.0
    avg_generator_adv_loss = 0.0

    with torch.no_grad():
        for data_batch in test_loader:
            img_real = data_batch["image"].to(device)

            recon_batch, q_loss, _, perplexity = model(img_real)

            percep_loss = perceptual_loss(img_real, recon_batch)

            # Use consistent loss functions with training
            logits_real_d = model.discriminator(img_real)
            logits_fake_d = model.discriminator(recon_batch)

            discriminator_loss = hinge_d_loss(logits_real_d, logits_fake_d)
            generator_adv_loss = -torch.mean(logits_fake_d)

            avg_percep_loss += percep_loss.item()
            avg_q_loss += q_loss.item()
            avg_perplexity += perplexity.item()
            avg_discriminator_loss += discriminator_loss.item()
            avg_generator_adv_loss += generator_adv_loss.item()

    num_batches = len(test_loader)
    avg_percep_loss /= num_batches
    avg_q_loss /= num_batches
    avg_perplexity /= num_batches
    avg_discriminator_loss /= num_batches
    avg_generator_adv_loss /= num_batches

    return (
        avg_percep_loss,
        avg_q_loss,
        avg_generator_adv_loss,
        avg_discriminator_loss,
        avg_perplexity,
    )


def main(args):
    torch.multiprocessing.set_sharing_strategy("file_system")
    print(
        f"--- PyTorch sharing strategy is now: {torch.multiprocessing.get_sharing_strategy()} ---"
    )

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
    )
    model = model.to(device)
    # Perceptual Loss layer indices for VGG16 features
    feature_layers = [3, 8, 15, 22]
    perceptual_loss = PerceptualLoss(feature_layers)
    perceptual_loss.to(device)
    perceptual_loss.eval()

    opt_enc_dec = torch.optim.Adam(
        list(model.encoder.parameters())
        + list(model.decoder.parameters())
        + list(model.quantizer.parameters()),
        lr=args.lr,
        betas=(0.5, 0.9),
    )
    opt_disc = torch.optim.Adam(
        model.discriminator.parameters(), lr=args.lr/4, betas=(0.5, 0.9)
    )

    print(
        "Model Parameters: "
        f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}"
    )

    # Determine latent dimensions
    dummy_input = torch.randn(1, n_channels, img_size, img_size).to(device)
    with torch.no_grad():
        encoder_output_shape = model.encoder(dummy_input).shape
    latent_h_vq, latent_w_vq = encoder_output_shape[2], encoder_output_shape[3]
    print(f"Latent grid dimensions after encoder: {latent_h_vq}x{latent_w_vq}")
    print("Starting Training...")
    (
        train_vq_loss,
        train_percep_loss,
        train_gan_g_loss,
        train_gan_d_loss,
        train_perplexity_loss,
    ) = ([], [], [], [], [])
    (
        val_vq_loss,
        val_percep_loss,
        val_gan_g_loss,
        val_gan_d_loss,
        val_perplexity_loss,
    ) = (
        [],
        [],
        [],
        [],
        [],
    )

    best_val_loss = float("inf")
    for epoch in range(args.max_epochs):
        print(f"Epoch {epoch + 1}/{args.max_epochs}")

        # Training
        (
            epoch_percep_loss,
            epoch_train_vq_loss,
            epoch_gan_g_train_loss,
            epoch_gan_d_train_loss,
            epoch_train_perplexity,
        ) = train(
            model,
            perceptual_loss,
            train_loader,
            opt_enc_dec,
            opt_disc,
            device,
            epoch,
            args,
        )
        train_vq_loss.append(epoch_train_vq_loss)
        train_percep_loss.append(epoch_percep_loss)
        train_gan_d_loss.append(epoch_gan_d_train_loss)
        train_gan_g_loss.append(epoch_gan_g_train_loss)
        train_perplexity_loss.append(epoch_train_perplexity)

        print(
            "====> Epoch: {} Train Avg Losses | PercepLoss: {:.4f} | VQ: {:.4f} | G_Adv: {:.4f} | D_Loss: {:.4f} | Perplexity: {:.4f}".format(
                epoch + 1,
                epoch_percep_loss,
                epoch_train_vq_loss,
                epoch_gan_g_train_loss,
                epoch_gan_d_train_loss,
                epoch_train_perplexity,
            )
        )

        # Validation
        (
            epoch_val_percep_loss,
            epoch_val_vq_loss,
            epoch_val_gan_g_loss,
            epoch_val_gan_d_loss,
            epoch_val_perplexity,
        ) = test(model, perceptual_loss, test_loader, device)
        val_vq_loss.append(epoch_val_vq_loss)
        val_percep_loss.append(epoch_val_percep_loss)
        val_gan_g_loss.append(epoch_val_gan_g_loss)
        val_gan_d_loss.append(epoch_val_gan_d_loss)
        val_perplexity_loss.append(epoch_val_perplexity)

        # Saving the best model
        if epoch_val_percep_loss + epoch_val_vq_loss < best_val_loss:
            best_val_loss = epoch_val_percep_loss + epoch_val_vq_loss
            print(
                "====> Best test set reconstruction"
                f"loss seen so far: {best_val_loss:.4f}"
            )
            print("Saving model...")
            torch.save(model.state_dict(), os.path.join(save_dir, "best_vqvae.pt"))
            print("Model saved.")

        if epoch % args.save_every == 0:
            show_reconstructions(
                model, test_loader, 5, n_channels, device, epoch, save_dir
            )
            show_latent_codes(
                model,
                test_loader,
                num_samples=5000,
                num_embeddings=args.num_embeddings,
                device=device,
                save_plots_dir=save_dir,
                current_epoch=epoch,
            )
            generate_random_samples(
                model,
                n_channels,
                latent_h_vq,
                latent_w_vq,
                num_samples=5,
                device=device,
                save_plots_dir=save_dir,
                current_epoch=epoch,
            )

            save_training_loss(
                train_vq_loss=train_vq_loss,
                train_recon_loss=train_percep_loss,
                train_perplexity_loss=train_perplexity_loss,
                train_gan_g_loss=train_gan_g_loss,
                train_gan_d_loss=train_gan_d_loss,
                val_vq_loss=val_vq_loss,
                val_recon_loss=val_percep_loss,
                val_perplexity_loss=val_perplexity_loss,
                val_gan_g_loss=val_gan_g_loss,
                val_gan_d_loss=val_gan_d_loss,
                save_plots_dir=save_dir,
            )

    print("Training complete.")
    print("Finalizing...")
    torch.save(
        model.state_dict(),
        os.path.join(save_dir, f"last_vqvae_epoch_{args.max_epochs}.pt"),
    )

    if not os.path.exists(os.path.join(save_dir, "best_vqvae.pt")):
        torch.save(
            model.state_dict(),
            os.path.join(save_dir, "best_vqvae.pt"),
        )
        print("Best model saved as best_vqvae.pt")

    save_training_loss(
        train_vq_loss=train_vq_loss,
        train_recon_loss=train_percep_loss,
        train_perplexity_loss=train_perplexity_loss,
        train_gan_g_loss=train_gan_g_loss,
        train_gan_d_loss=train_gan_d_loss,
        val_vq_loss=val_vq_loss,
        val_recon_loss=val_percep_loss,
        val_perplexity_loss=val_perplexity_loss,
        val_gan_g_loss=val_gan_g_loss,
        val_gan_d_loss=val_gan_d_loss,
        save_plots_dir=save_dir,
    )
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="VQ-VAE training and visualization script"
    )
    # Data Handling arguments
    parser.add_argument(
        "--dataset",
        type=str,
        default="birds",
        choices=["flowers", "birds", "animals", "mnist"],
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

    parser.add_argument(
        "--disc-start-epoch",
        type=int,
        default=5,
        help="Epoch to start training the discriminator and using GAN loss.",
    )
    parser.add_argument(
        "--gan-weight",
        type=float,
        default=0.8,
        help="Weight for the adversarial (GAN) loss term.",
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
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")

    args = parser.parse_args()

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version PyTorch was compiled with: {torch.version.cuda}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"Current GPU name: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is NOT available to PyTorch.")
    main(args)
