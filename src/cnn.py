"""Provides CNN utils made for training neural network."""
import sys
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import piq
import numpy as np
from tqdm import tqdm
import wandb


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

class CNN:
    """Trains neural networks and evaluates them."""

    def calculate_psnr(self, predictions, targets):
        """Calculate peak signal-to-noise ratio."""
        mse = torch.mean((predictions - targets) ** 2)
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
        return psnr

    def run(self, input_path: str, output_path: str):
        """Just a test..."""

        learning_rate = 0.1
        batch_size = 64
        epochs = 10

        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project="FAIthfuler",
            entity="rootender",
            name="FaithfulNet",
            # track hyperparameters and run metadata
            config={
                "learning_rate": learning_rate,
                "architecture": "CNN",
                "batch_size": batch_size,
                "hidden_layer_sizes": [64, 32, 4]
            }
        )

        if not torch.cuda.is_available():
            print("CUDA architecture is unavailable!")
            sys.exit(1)
        else:
            print("CUDA detected! Initializing...")
        print(f'Input dataset: {input_path}')
        print(f'Output dataset: {output_path}')
        print('Reading datasets...')

        transformator = transforms.Compose([
            transforms.ToTensor(),  # Convert images to tensors
        ])

        input_data = datasets.ImageFolder(input_path, transform=transformator)
        print('Input loaded!')

        output_data = datasets.ImageFolder(output_path, transform=transformator)
        print('Output loaded!')

        input_data_loader = DataLoader(input_data, batch_size=batch_size, pin_memory=True)
        output_data_loader = DataLoader(output_data, batch_size=batch_size, pin_memory=True)
        model = FaithfulNet().cuda()

        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        loss_function = nn.MSELoss()

        # simulate training
        for epoch in range(epochs):
            total_psnr = 0.0
            total_samples = 0
            loss = 0

            for batch, (input_batch, output_batch) in tqdm(
                enumerate(zip(input_data_loader, output_data_loader)),
                unit="batch",
                desc=f'epoch {epoch}'):

                input_batch = input_batch[batch].cuda()
                output_batch = output_batch[batch].cuda()

                optimizer.zero_grad()
                prediction = model(input_batch)

                # Metrics
                loss = loss_function(prediction, output_batch)

                psnr = piq.psnr(prediction, output_batch)
                total_psnr += np.mean(psnr)
                total_samples += len(psnr)

                # Step
                optimizer.step()

            wandb.log({"avg_psnr": total_psnr / total_samples, "loss": loss})
        wandb.finish()
