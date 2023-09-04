"""
This module defines a pair of neural network models for an image-to-image translation task.
However, instead of being based on U-Net, it also includes ResNet.
The 'flexy' suffix means that some settings are customizable.

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

    def __init__(self, layer_multiplier: int = 1, relu_factor: float = 0.2):
        super(Generator, self).__init__()

        # Encoder
        self.encoder1 = nn.Conv2d(
            4, 64 * layer_multiplier, kernel_size=4, stride=2, padding=1, bias=False)

        self.encoder2 = nn.Sequential(
            nn.LeakyReLU(relu_factor, inplace=True),
            nn.Conv2d(64 * layer_multiplier, 128 * layer_multiplier, kernel_size=4,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128 * layer_multiplier)
        )

        # Residual connection layers for encoder
        self.res_enc_block = nn.Sequential(
            nn.Conv2d(128 * layer_multiplier, 128 * layer_multiplier,
                      kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(relu_factor, inplace=True),
            nn.Conv2d(128 * layer_multiplier, 128 * layer_multiplier,
                      kernel_size=3, stride=1, padding=1)
        )

        self.encoder3 = nn.Sequential(
            nn.LeakyReLU(relu_factor, inplace=True),
            nn.Conv2d(128 * layer_multiplier, 256 * layer_multiplier,
                      kernel_size=4, stride=2, padding=1, bias=False),
        )

        # Decoder
        self.decoder1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256 * layer_multiplier, 128 * layer_multiplier,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128 * layer_multiplier)
        )

        self.decoder2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128*2 * layer_multiplier, 64 * layer_multiplier,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64 * layer_multiplier)
        )

        # Residual connection layers for decoder
        self.res_dec_block = nn.Sequential(
            nn.Conv2d(64*2 * layer_multiplier, 64*2 * layer_multiplier,
                      kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(relu_factor, inplace=True),
            nn.Conv2d(128*2 * layer_multiplier, 128*2 * layer_multiplier,
                      kernel_size=3, stride=1, padding=1)
        )

        self.decoder3 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64*2 * layer_multiplier, 4, kernel_size=4,
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
        res_blocks = 5

        e1 = self.encoder1(image)
        e2 = self.encoder2(e1)
        for _ in range(res_blocks):
            e2 += self.res_enc_block(e2)
        latent_space = self.encoder3(e2)

        d1 = torch.cat([self.decoder1(latent_space), e2], dim=1)
        d2 = torch.cat([self.decoder2(d1), e1], dim=1)
        for _ in range(res_blocks):
            d2 += self.res_enc_block(d2)
        out = self.decoder3(d2)

        return out


class Discriminator(nn.Module):
    """
    A convolutional neural network (CNN) discriminator for distinguishing real and fake images.

    Attributes:
        structure (nn.Sequential): The sequential layers of the discriminator.
    """

    def __init__(self):
        super(Discriminator, self).__init__()

        self.structure = nn.Sequential(
            nn.Conv2d(4*2, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=4,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 1, kernel_size=4, stride=1, padding=1, bias=False),
            nn.Sigmoid()
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
