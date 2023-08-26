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
        decoder1 (nn.Sequential): The first convolutional layer in the decoder.
        decoder2 (nn.Sequential): The second convolutional layer in the decoder.
    """
    def __init__(self):
        super(Generator, self).__init__()

        # Encoder
        self.encoder1 = nn.Conv2d(4, 64, kernel_size=4, stride=2, padding=1, bias=False)

        self.encoder2 = nn.Sequential(
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False)
        )

        # Decoder
        self.decoder1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64)
        )

        self.decoder2 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64*2, 4, kernel_size=4, stride=2, padding=1, bias=False),
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
        encoded_1 = self.encoder1(image)

        latent_space = self.encoder2(encoded_1)

        decoded_1 = torch.cat([self.decoder1(latent_space), encoded_1], dim=1)
        out = self.decoder2(decoded_1)

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

            nn.Conv2d(128, 1, kernel_size=4, stride=1, padding=1, bias=False),
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
