import torch
import torch.nn as nn
import torch.nn.functional as F
from taming_transformer.models.modules import (
    ResidualBlock,
    DownSampleBlock,
    NonLocalBlock,
)

class Encoder(nn.Module):
    """
    The Encoder part of a VQGAN-style model.

    This module takes an image as input and progressively downsamples it into a
    lower-resolution latent representation. It is composed of a series of
    residual blocks, optional self-attention blocks, and downsampling layers.

    Args:
        input_channels (int):
            The number of channels in the input tensor (e.g., 3 for an RGB image).

        input_resolution (int):
            The spatial resolution (height and width) of the input image. This is
            used to determine where to place attention blocks based on the
            `attention_resolutions` parameter. Assumes square images.

        base_channels (int):
            The number of channels after the initial convolution. This serves as
            the base for calculating the number of channels at subsequent
            downsampling levels. It's a key parameter for controlling the model's capacity.

        latent_channels (int):
            The number of channels in the final output latent representation. This
            is the channel dimension of the tensor returned by the forward pass.

        channel_multipliers (tuple[int], optional):
            A tuple of integers that defines the channel multiplier for each
            resolution level. The number of channels at level `i` will be
            `base_channels * channel_multipliers[i]`. The length of this tuple
            determines the number of downsampling levels.
            Defaults to (1, 2, 4, 8).
            Example: With `base_channels=64` and `channel_multipliers=(1, 2, 4)`,
            the channels at each level will be 64, 128, and 256.

        num_residual_blocks (int, optional):
            The number of `ResidualBlock` modules to use at each resolution level.
            More blocks increase the depth and expressive power of the network.
            Defaults to 2.

        attention_resolutions (tuple[int], optional):
            A tuple of spatial resolutions at which to insert a `NonLocalBlock`
            (self-attention) after a residual block. The encoder downsamples the
            feature map at each level; if the resulting resolution matches a value
            in this tuple, an attention block is added.
            Defaults to (16,).
            Example: If `input_resolution=256`, downsampling happens at 128, 64, 32, etc.
            If `attention_resolutions=(16,)`, an attention block will be added when
            the feature map has a spatial size of 16x16.

        dropout (float, optional):
            The dropout rate to be used within the `ResidualBlock`s.
            Defaults to 0.0.
    """

    def __init__(
        self,
        input_channels: int,
        input_resolution: int,
        base_channels: int,
        latent_channels: int,
        channel_multipliers: tuple[int] = (1, 2, 4, 8),
        num_residual_blocks: int = 2,
        attention_resolutions: tuple[int] = (16,),
        dropout: float = 0.0,
    ):
        super().__init__()

        self.num_resolutions = len(channel_multipliers)
        self.initial_conv = nn.Conv2d(input_channels, base_channels, 3, 1, 1)

        self.down_blocks = nn.ModuleList()

        current_resolution = input_resolution
        in_channels = base_channels

        for level_idx in range(self.num_resolutions):
            out_channels = base_channels * channel_multipliers[level_idx]

            level_modules = nn.ModuleList()
            for _ in range(num_residual_blocks):
                level_modules.append(
                    ResidualBlock(
                        in_channels=in_channels,
                        out_channels=out_channels,
                        dropout=dropout,
                    )
                )
                in_channels = out_channels

                if current_resolution in attention_resolutions:
                    level_modules.append(NonLocalBlock(in_channels))

            # Add downsampling block at the end of each level, except the last one
            if level_idx != self.num_resolutions - 1:
                level_modules.append(DownSampleBlock(in_channels))
                current_resolution //= 2

            self.down_blocks.append(level_modules)

        self.middle_block = nn.Sequential(
            ResidualBlock(in_channels, in_channels, dropout),
            NonLocalBlock(in_channels),
            ResidualBlock(in_channels, in_channels, dropout),
        )

        self.final_norm = nn.GroupNorm(32, in_channels)
        self.final_conv = nn.Conv2d(in_channels, latent_channels, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Processes the input tensor through the encoder.

        Args:
            x (torch.Tensor): The input image tensor with shape
                              (B, C, H, W), where C is `input_channels` and
                              H, W are `input_resolution`.

        Returns:
            torch.Tensor: The encoded latent representation with shape
                          (B, `latent_channels`, H', W'), where H' and W'
                          are the final downsampled spatial dimensions.
        """
        x = self.initial_conv(x)

        for level_modules in self.down_blocks:
            for module in level_modules:
                x = module(x)

        x = self.middle_block(x)

        x = self.final_norm(x)
        x = F.silu(x)
        x = self.final_conv(x)

        return x