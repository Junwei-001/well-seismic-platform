"""Compact 3D U-Net used for seismic fault segmentation."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class FaultSegNet(nn.Module):
    """Compact 3D U-Net returning one fault logit per voxel.

    Args:
        bottleneck_channels: Width of the deepest block. Official checkpoints
            use either 128 (recommended) or 512 channels.
    """

    def __init__(self, bottleneck_channels: int = 128) -> None:
        super().__init__()
        if bottleneck_channels not in (128, 512):
            raise ValueError("bottleneck_channels must be 128 or 512")

        self.bottleneck_channels = bottleneck_channels
        self.conv1a = nn.Conv3d(1, 16, 3, padding=1)
        self.conv1b = nn.Conv3d(16, 16, 3, padding=1)
        self.conv2a = nn.Conv3d(16, 32, 3, padding=1)
        self.conv2b = nn.Conv3d(32, 32, 3, padding=1)
        self.conv3a = nn.Conv3d(32, 64, 3, padding=1)
        self.conv3b = nn.Conv3d(64, 64, 3, padding=1)
        self.conv4a = nn.Conv3d(64, bottleneck_channels, 3, padding=1)
        self.conv4b = nn.Conv3d(
            bottleneck_channels, bottleneck_channels, 3, padding=1
        )
        self.conv5a = nn.Conv3d(bottleneck_channels + 64, 64, 3, padding=1)
        self.conv5b = nn.Conv3d(64, 64, 3, padding=1)
        self.conv6a = nn.Conv3d(64 + 32, 32, 3, padding=1)
        self.conv6b = nn.Conv3d(32, 32, 3, padding=1)
        self.conv7a = nn.Conv3d(32 + 16, 16, 3, padding=1)
        self.conv7b = nn.Conv3d(16, 16, 3, padding=1)
        self.classifier = nn.Conv3d(16, 1, 1)
        self.pool = nn.MaxPool3d(2)

    @staticmethod
    def _up(x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, scale_factor=2.0, mode="nearest")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c1 = F.relu(self.conv1a(x), inplace=True)
        c1 = F.relu(self.conv1b(c1), inplace=True)
        c2 = F.relu(self.conv2a(self.pool(c1)), inplace=True)
        c2 = F.relu(self.conv2b(c2), inplace=True)
        c3 = F.relu(self.conv3a(self.pool(c2)), inplace=True)
        c3 = F.relu(self.conv3b(c3), inplace=True)
        c4 = F.relu(self.conv4a(self.pool(c3)), inplace=True)
        c4 = F.relu(self.conv4b(c4), inplace=True)

        c5 = torch.cat((self._up(c4), c3), dim=1)
        c5 = F.relu(self.conv5a(c5), inplace=True)
        c5 = F.relu(self.conv5b(c5), inplace=True)
        c6 = torch.cat((self._up(c5), c2), dim=1)
        c6 = F.relu(self.conv6a(c6), inplace=True)
        c6 = F.relu(self.conv6b(c6), inplace=True)
        c7 = torch.cat((self._up(c6), c1), dim=1)
        c7 = F.relu(self.conv7a(c7), inplace=True)
        c7 = F.relu(self.conv7b(c7), inplace=True)
        return self.classifier(c7)
