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

        # == Residual block ==
        self.encoder2_leaky = nn.LeakyReLU(relu_factor, inplace=True)
        self.encoder2_conv = nn.Conv2d(64 * layer_multiplier, 128 * layer_multiplier,
                                       kernel_size=4, stride=2, padding=1, bias=False)
        self.encoder2_norm = nn.BatchNorm2d(128 * layer_multiplier)

        self.res_enc1 = nn.Conv2d(128 * layer_multiplier, 128 * layer_multiplier,
                                  kernel_size=3, stride=1, padding=1)
        # ====================

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
            nn.BatchNorm2d(128 * layer_multiplier),
            nn.Dropout(0.5)
        )

        # == Residual block ==
        self.decoder2_relu = nn.ReLU(inplace=True)
        self.decoder2_conv = nn.Conv2d(128*2 * layer_multiplier, 64 * layer_multiplier,
                                       kernel_size=4, stride=2, padding=1, bias=False)
        self.decoder2_norm = nn.BatchNorm2d(64 * layer_multiplier)

        self.res_dec1 = nn.ConvTranspose2d(
            64*2 * layer_multiplier,
            64*2 * layer_multiplier,
            kernel_size=3, stride=1, padding=1
        )
        # ====================

        self.decoder3 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64*2 * layer_multiplier, 4,
                               kernel_size=4, stride=2, padding=1, bias=False),
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

        # Residual block
        e1_activated = self.encoder2_leaky(e1)
        e1_conv = self.encoder2_conv(e1_activated)
        e2 = e1_conv + self.res_enc1(e1_conv)
        e2 = self.encoder2_norm(e2)

        latent_space = self.encoder3(e2)


        d1 = torch.cat([self.decoder1(latent_space), e2], dim=1)

        # Residual block
        d1_activated = self.decoder2_relu(d1)
        d1_conv = torch.cat([self.decoder2_conv(d1_activated), e1], dim=1)
        d2 = d1_conv + self.res_dec1(d1_conv)
        d2 = self.decoder2_norm(d2)

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
