import torch
from torch import nn
import torch.nn.functional as F

# Decoder, Encoder together make up the GAN's Generator.
from taming_transformer.models.encoder import Encoder
from taming_transformer.models.discriminator import Discriminator
from taming_transformer.models.quantizer import VectorQuantizer
from taming_transformer.models.decoder import Decoder


class VectorQuantizedGAN(nn.Module):

    def __init__(
        self,
        in_channels,
        input_resolution,
        base_channels,
        n_blocks,
        latent_channels,
        num_embeddings,
        commitment_cost,
        channel_multipliers=(1, 2, 4, 4),
        attention_resolutions=(16,),
        dropout=0.0,
    ):
        super(VectorQuantizedGAN, self).__init__()

        self.encoder = Encoder(
            in_channels,
            input_resolution,
            base_channels,
            latent_channels,
            channel_multipliers,
            n_blocks,
            attention_resolutions,
            dropout,
        )
        self.quantizer = VectorQuantizer(
            num_embeddings, latent_channels, commitment_cost
        )
        self.decoder = Decoder(
            in_channels,
            latent_channels,
            base_channels,
            input_resolution,
            channel_multipliers,
            n_blocks - 1,
            attention_resolutions,
            dropout,
        )

        self.discriminator = Discriminator(in_channels, base_channels, n_blocks)

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, vq_loss, encoding_indices, perplexity = self.quantizer(z_e)
        x_hat = self.decoder(z_q)
        return x_hat, vq_loss, encoding_indices, perplexity

    @torch.no_grad()
    def get_latent_representation(self, x):
        """Helper to get discrete latent codes (indices) for an input."""
        z_e = self.encoder(x)
        _, _, encoding_indices, _ = self.quantizer(z_e)
        return encoding_indices

    @torch.no_grad()
    def reconstruct_from_indices(self, indices):
        """
        Correct reconstruction using synthetic quantizer forward path.
        """
        quantized = F.embedding(
            indices, self.quantizer.embedding.weight
        )  # (B, H, W, D)
        quantized = quantized.reshape(
            indices.shape[0], -1, indices.shape[1], indices.shape[2]
        )
        x_hat = self.decoder(quantized)
        return x_hat

    def compute_adaptive_lambda(self, l_rec, l_gan, delta=1e-6):
        last_layer_decoder = self.decoder.final_conv.weight
        l_rec_grads = torch.autograd.grad(
            l_rec,
            last_layer_decoder,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )[0]
        l_gan_grads = torch.autograd.grad(
            l_gan,
            last_layer_decoder,
            retain_graph=True,
            create_graph=False,
            allow_unused=True,
        )[0]
        if l_rec_grads is None or l_gan_grads is None:
            return 1.0
        else:
            λ = torch.norm(l_rec_grads) / (torch.norm(l_gan_grads) + delta)
            λ = torch.clamp(λ, 0, 1e4).detach()
            return λ
