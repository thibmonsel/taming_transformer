import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import ResidualBlock, NonLocalBlock, UpSampleBlock
class Decoder(nn.Module):
    """
    The Decoder part of a VQGAN-style model.

    This module reconstructs a high-resolution image from a low-resolution latent
    representation. It is designed to be a symmetric counterpart to the Encoder,
    progressively upsampling the feature maps while reducing channel depth.
    It is composed of a series of residual blocks, optional self-attention
    blocks, and upsampling layers.

    Args:
        output_channels (int):
            The number of channels in the final, reconstructed output tensor
            (e.g., 3 for an RGB image).

        latent_channels (int):
            The number of channels in the input latent representation `z`. This
            must match the `latent_channels` of the corresponding `Encoder`.

        base_channels (int):
            The base number of channels for calculating feature map sizes at
            different levels. This should be consistent with the `Encoder` to
            ensure architectural symmetry.

        final_resolution (int):
            The spatial resolution (height and width) of the final output image.
            This is the target size for the reconstruction.

        channel_multipliers (tuple[int], optional):
            A tuple of integers defining channel multipliers for each resolution
            level. This should be the same tuple as used in the `Encoder`. The
            decoder works backwards from the largest multiplier to the smallest.
            The length of this tuple determines the number of upsampling levels.
            Defaults to (1, 2, 4, 8).

        num_residual_blocks (int, optional):
            The number of `ResidualBlock` modules to use at each resolution level.
            Note: The implementation uses `num_residual_blocks + 1` blocks per
            level in the upsampling path to maintain architectural balance with
            common U-Net designs. Defaults to 2.

        attention_resolutions (tuple[int], optional):
            A tuple of spatial resolutions at which to insert a `NonLocalBlock`
            (self-attention). If the feature map resolution during upsampling
            matches a value in this tuple, an attention block is added.
            This should generally match the `Encoder`'s settings.
            Defaults to (16,).

        dropout (float, optional):
            The dropout rate to be used within the `ResidualBlock`s.
            Defaults to 0.0.
    """
    def __init__(
        self,
        output_channels: int,
        latent_channels: int,
        base_channels: int,
        final_resolution: int,
        channel_multipliers: tuple[int] = (1, 2, 4, 8),
        num_residual_blocks: int = 2,
        attention_resolutions: tuple[int] = (16,),
        dropout: float = 0.0,
    ):
        super().__init__()

        # The number of channels at the bottleneck (the last level of the encoder)
        bottleneck_channels = base_channels * channel_multipliers[-1]

        # The resolution at the bottleneck (the decoder's starting resolution)
        self.num_resolutions = len(channel_multipliers)
        current_resolution = final_resolution // (2 ** (self.num_resolutions - 1))

        # Initial convolution to map latent channels to the bottleneck channel size
        self.initial_conv = nn.Conv2d(latent_channels, bottleneck_channels, 3, 1, 1)

        # Middle block at the lowest resolution
        self.middle_block = nn.Sequential(
            ResidualBlock(bottleneck_channels, bottleneck_channels, dropout),
            NonLocalBlock(bottleneck_channels),
            ResidualBlock(bottleneck_channels, bottleneck_channels, dropout),
        )

        # Build the upsampling blocks
        self.up_blocks = nn.ModuleList()
        in_channels = bottleneck_channels
        for level_idx in reversed(range(self.num_resolutions)):
            out_channels = base_channels * channel_multipliers[level_idx]

            level_modules = nn.ModuleList()
            # More residual blocks in the decoder path
            for _ in range(num_residual_blocks + 1):
                level_modules.append(ResidualBlock(in_channels, out_channels, dropout))
                in_channels = out_channels

                # Add attention if resolution matches
                if current_resolution in attention_resolutions:
                    level_modules.append(NonLocalBlock(in_channels))

            # Add an upsampling block for all but the last (highest-res) level
            if level_idx != 0:
                level_modules.append(UpSampleBlock(in_channels))
                current_resolution *= 2

            self.up_blocks.append(level_modules)

        # Final layers to produce the output image
        self.final_norm = nn.GroupNorm(32, in_channels)
        self.final_conv = nn.Conv2d(in_channels, output_channels, 3, 1, 1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Processes the input latent tensor through the decoder to reconstruct an image.

        Args:
            z (torch.Tensor): The input latent tensor with shape
                              (B, C, H', W'), where C is `latent_channels` and
                              H', W' are the latent spatial dimensions.

        Returns:
            torch.Tensor: The reconstructed output tensor (e.g., an image) with
                          shape (B, `output_channels`, `final_resolution`, `final_resolution`).
        """
        # Prepare latent for the main decoder blocks
        x = self.initial_conv(z)
        x = self.middle_block(x)

        # Pass through upsampling blocks
        for level_modules in self.up_blocks:
            for module in level_modules:
                x = module(x)

        # Final processing
        x = self.final_norm(x)
        x = F.silu(x)
        x = self.final_conv(x)

        return x