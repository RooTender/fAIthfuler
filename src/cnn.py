"""Provides CNN utils made for training neural network."""
from torch import nn


class FaithfulNet(nn.Module):
    """Convolutional neural network for Faithful textures generation."""
    def __init__(self):
        super(FaithfulNet, self).__init__()
        
        # Upscaling layers (for example, using Transposed Convolution)
        self.upscale = nn.Sequential(
            nn.ConvTranspose2d(4, 64, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=1, padding=0),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 4, kernel_size=3, stride=1, padding=0),
            nn.ReLU()
        )

    def forward(self, image):
        """Forward function of the neural network."""
        out = self.upscale(image)
        return out
