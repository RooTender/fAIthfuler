"""
This module defines a pair of neural network models for an image-to-image translation task.
However, instead of being based on U-Net, it also includes ResNet.
The difference from other file is that it also uses Neural Style Transfer for style learning.

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
        self.stored_latent_style = None

        # Encoder
        self.encoder1 = nn.Conv2d(
            4, 64 * layer_multiplier, kernel_size=4, stride=2, padding=1, bias=False)

        self.encoder2 = nn.Sequential(
            nn.LeakyReLU(relu_factor, inplace=True),
            nn.Conv2d(64 * layer_multiplier, 128 * layer_multiplier,
                      kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128 * layer_multiplier)
        )

        # Residual connection layers for encoder
        self.res_enc1 = nn.Conv2d(128 * layer_multiplier, 128 * layer_multiplier,
                                  kernel_size=3, stride=1, padding=1)

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
        self.res_dec1 = nn.ConvTranspose2d(
            64*2 * layer_multiplier,
            64*2 * layer_multiplier,
            kernel_size=3, stride=1, padding=1
        )

        self.decoder3 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64*2 * layer_multiplier, 4,
                               kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, content_image, style_image=None):
        """
        Forward pass through the generator for style transfer.

        This method performs image-to-image translation by combining features 
        from a content image and a style image. If a style image is provided, 
        its latent style representation is computed and stored internally. 
        If no style image is given, the previously stored latent style is used 
        for generation.

        Args:
            content_image (torch.Tensor): The input content image tensor of shape 
                [batch_size, channels, height, width].
            style_image (torch.Tensor, optional): The input style image tensor of the same 
                shape as content_image. If not provided, the method uses a stored 
                latent style for style transfer. Default is None.
        """

        if style_image is None:
            latent_style = self.stored_latent_style
        else:
            e1_style = self.encoder1(style_image)
            e2_style = self.encoder2(e1_style)
            e2_style = e2_style + self.res_enc1(e2_style)
            latent_style = self.encoder3(e2_style)

            self.stored_latent_style = latent_style

        e1_content = self.encoder1(content_image)
        e2_content = self.encoder2(e1_content)

        d1 = torch.cat([self.decoder1(latent_style), e2_content], dim=1)
        d2 = torch.cat([self.decoder2(d1), e1_content], dim=1)
        d2 = d2 + self.res_dec1(d2)
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
