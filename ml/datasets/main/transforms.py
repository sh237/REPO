import torch
import torch.nn as nn
import torchvision.transforms.v2 as transforms
from torch.nn.functional import interpolate
import random
import numpy as np


class SolarTransforms(nn.Module):
    def __init__(self, p=0.4):
        super().__init__()
        self.p = p
        # Register as buffer, but actual device movement will be done later
        self.register_buffer("channel_mask", torch.arange(10), persistent=False)

    @torch.cuda.amp.autocast()
    def forward(self, x):
        if random.random() > self.p:
            return x

        device = x.device
        history, channels, H, W = x.shape

        # Move channel_mask to the same device as the input tensor
        self.channel_mask = self.channel_mask.to(device)

        # Perform all transformations in batch
        x = x.reshape(-1, 1, H, W)  # [history * channels, 1, H, W]

        # 1. Apply rotation and scaling simultaneously
        if random.random() > self.p:
            angle = random.uniform(-30, 30)
            scale = random.uniform(0.9, 1.1)

            # Create affine transformation matrix
            theta = torch.tensor(
                [
                    [
                        scale * torch.cos(torch.tensor(angle * np.pi / 180)),
                        -scale * torch.sin(torch.tensor(angle * np.pi / 180)),
                        0,
                    ],
                    [
                        scale * torch.sin(torch.tensor(angle * np.pi / 180)),
                        scale * torch.cos(torch.tensor(angle * np.pi / 180)),
                        0,
                    ],
                ],
                device=device,
            ).float()

            # Generate grid
            grid = torch.nn.functional.affine_grid(
                theta.unsqueeze(0).repeat(x.size(0), 1, 1),
                x.size(),
                align_corners=False,
            )

            # Apply rotation and scaling in one operation
            x = torch.nn.functional.grid_sample(
                x, grid, mode="bilinear", align_corners=False
            )

        # Random brightness and contrast adjustment
        if random.random() > self.p:
            brightness = random.uniform(0.8, 1.2)
            contrast = random.uniform(0.8, 1.2)
            x = contrast * (x - x.mean()) + x.mean()  # Contrast adjustment
            x = brightness * x  # Brightness adjustment

        # New extension: Gaussian blur
        if random.random() > self.p:
            kernel_size = random.choice([3, 5])
            sigma = random.uniform(0.1, 2.0)
            blur = transforms.GaussianBlur(kernel_size, sigma)
            x = blur(x)

        # Return to original shape
        x = x.reshape(history, channels, H, W)

        # 3. Generate channel-specific noise in bulk
        if random.random() > self.p:
            # Calculate noise levels for each channel in bulk
            noise_levels = torch.where(
                self.channel_mask < 7,
                torch.empty(channels, device=device).uniform_(0.01, 0.03),
                torch.empty(channels, device=device).uniform_(0.005, 0.015),
            ).view(1, -1, 1, 1)

            # Calculate standard deviation in bulk
            std_per_channel = x.std(dim=(-2, -1), keepdim=True)

            # Generate noise in bulk and apply
            noise = torch.randn_like(x) * noise_levels * std_per_channel

            # Apply noise
            x = x + noise

        # 5. Apply brightness variations in bulk
        if random.random() > self.p:
            variations = torch.empty(1, channels, 1, 1, device=device).uniform_(
                0.85, 1.15
            )
            x = x * variations

        return x.contiguous()

    def __call__(self, x):
        return self.forward(x)
