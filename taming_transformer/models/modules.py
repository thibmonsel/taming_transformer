import torch.nn as nn
import torch.nn.functional as F
import torch


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.GroupNorm(32, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )

        if in_channels != out_channels:
            self.channel_up = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        if self.in_channels != self.out_channels:
            return self.channel_up(x) + self.block(x)
        else:
            return x + self.block(x)


class UpSampleBlock(nn.Module):
    def __init__(self, channels):
        super(UpSampleBlock, self).__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0)
        return self.conv(x)


class DownSampleBlock(nn.Module):
    def __init__(self, channels):
        super(DownSampleBlock, self).__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 2, 0)

    def forward(self, x):
        pad = (0, 1, 0, 1)
        x = F.pad(x, pad, mode="constant", value=0)
        return self.conv(x)


class NonLocalBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.in_channels = in_channels

        # normalization layer
        self.norm = nn.GroupNorm(32, in_channels)

        # query, key and value layers
        self.q = nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        self.k = nn.Conv2d(in_channels, in_channels, 1, 1, 0)
        self.v = nn.Conv2d(in_channels, in_channels, 1, 1, 0)

        self.project_out = nn.Conv2d(in_channels, in_channels, 1, 1, 0)

        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):

        batch, _, height, width = x.size()

        x = self.norm(x)

        # query, key and value layers
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        # resizing the output from 4D to 3D to generate attention map
        q = q.reshape(batch, self.in_channels, height * width)
        k = k.reshape(batch, self.in_channels, height * width)
        v = v.reshape(batch, self.in_channels, height * width)

        # transpose the query tensor for dot product
        q = q.permute(0, 2, 1)

        # main attention formula
        scores = torch.bmm(q, k) * (self.in_channels**-0.5)
        weights = self.softmax(scores)
        weights = weights.permute(0, 2, 1)

        attention = torch.bmm(v, weights)

        # resizing the output from 3D to 4D to match the input
        attention = attention.reshape(batch, self.in_channels, height, width)
        attention = self.project_out(attention)

        # adding the identity to the output
        return x + attention
