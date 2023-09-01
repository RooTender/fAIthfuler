"""
This module defines a pair of neural network models for an image-to-image translation task.

The module includes:
- A Generator: Takes an input image and produces an output image.
- A Discriminator: Distinguishes between real and fake images.

Note: The Generator and Discriminator classes in this module are based on Pix2Pix architecture.
"""

import torch
from torch import nn


class Generator(nn.Module):
    """
    A convolutional neural network (CNN) generator for an image-to-image translation task.

    The generator takes an input image and produces an output image.

    Attributes:
        encoder1 (nn.Conv2d): The first convolutional layer in the encoder.
        encoder2 (nn.Sequential): The second convolutional layer in the encoder.
        encoder3 (nn.Sequential): ...
        decoder1 (nn.Sequential): The first convolutional layer in the decoder.
        decoder2 (nn.Sequential): The second convolutional layer in the decoder.
        decoder3 (nn.Sequential): ...
    """

    def __init__(self):
        super(Generator, self).__init__()

        # Encoder
        self.encoder1 = nn.Conv2d(
            4, 64, kernel_size=4, stride=2, padding=1, bias=False)

        self.encoder2 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )

        self.encoder3 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=4,
                      stride=2, padding=1, bias=False),
        )

        # Decoder
        self.decoder1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=4,
                               stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128)
        )

        self.decoder2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128*2, 64, kernel_size=4,
                               stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64)
        )

        self.decoder3 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64*2, 4, kernel_size=4,
                               stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, image):
        """
        Forward pass through the generator.

        Args:
            image (torch.Tensor): The input image tensor.

        Returns:
            torch.Tensor: The output image tensor.
        """
        e1 = self.encoder1(image)
        e2 = self.encoder2(e1)
        latent_space = self.encoder3(e2)

        d1 = torch.cat([self.decoder1(latent_space), e2], dim=1)
        d2 = torch.cat([self.decoder2(d1), e1], dim=1)
        out = self.decoder3(d2)

        return out


class Critic(nn.Module):
    """
    A convolutional neural network (CNN) critic for Wasserstein GANs.

    Attributes:
        structure (nn.Sequential): The sequential layers of the critic.
    """

    def __init__(self):
        super(Critic, self).__init__()

        self.structure = nn.Sequential(
            # 16 x 16
            nn.Conv2d(4, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(64, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # 8 x 8
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(128, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # 4 x 4
            nn.Conv2d(128, 256, kernel_size=4,
                      stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(256, affine=True),
            nn.LeakyReLU(0.2, inplace=True),

            # 2 x 2
            nn.Conv2d(256, 4, kernel_size=2, stride=1, padding=0, bias=False),
        )

    def forward(self, image):
        """
        Forward pass through the discriminator.

        Args:
            image (torch.Tensor): The input image tensor.

        Returns:
            torch.Tensor: The discriminator's output tensor (real vs. fake probability).
        """
        return self.structure(image)
