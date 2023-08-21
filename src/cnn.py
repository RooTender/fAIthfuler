"""Provides CNN utils made for training neural network."""
import os
import sys
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import piq
from PIL import Image
from tqdm import tqdm
import wandb


class FaithfulNet(nn.Module):
    """Convolutional neural network for Faithful textures generation."""

    def __init__(self):
        super(FaithfulNet, self).__init__()
        # Upscaling layers (for example, using Transposed Convolution)
        self.upscale = nn.Sequential(
            nn.ConvTranspose2d(4, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 4, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, image):
        """Forward function of the neural network."""
        return self.upscale(image)


class ImageLoaderDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, file)
                            for file in os.listdir(root_dir)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        return image


class CNN:
    """Trains neural networks and evaluates them."""

    def calculate_psnr(self, predictions, targets):
        """Calculate peak signal-to-noise ratio."""
        mse = torch.mean((predictions - targets) ** 2)
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
        return psnr

    def __load_data(self, input_path: str, output_path: str, batch_size: int):
        print(f'Input dataset: {input_path}')
        print(f'Output dataset: {output_path}')
        print('Reading datasets...')

        transformator = transforms.Compose([
            transforms.ToTensor()  # Convert images to tensors
        ])

        input_data = ImageLoaderDataset(input_path, transform=transformator)
        print('Input loaded!')

        output_data = ImageLoaderDataset(output_path, transform=transformator)
        print('Output loaded!')

        return DataLoader(input_data, batch_size=batch_size, pin_memory=True), DataLoader(
            output_data, batch_size=batch_size, pin_memory=True)

    def __train(self, dir_for_models: str, input_data: DataLoader, output_data: DataLoader,
                num_of_epochs: int, learning_rate: float):
        model = FaithfulNet().cuda()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)

        criterion_mse = nn.MSELoss()

        best_loss = float('inf')

        for epoch in range(num_of_epochs):
            total_psnr = 0.0
            total_ssim = 0.0

            total_loss = 0.0
            total_batches = 0

            for input_batch, output_batch in tqdm(
                    zip(input_data, output_data),
                    unit="batch",
                    desc=f'epoch {epoch}'):

                input_batch = input_batch.cuda()
                output_batch = output_batch.cuda()

                optimizer.zero_grad()
                prediction = model(input_batch)

                # Metrics
                # Rescale from [-1, 1] to [0, 1]
                prediction = (prediction + 1) / 2.0
                output_batch = (output_batch + 1) / 2.0

                total_psnr += piq.psnr(prediction, output_batch).item()
                total_ssim += piq.ssim(
                    prediction, output_batch).item()  # type: ignore

                loss = criterion_mse(prediction, output_batch)

                total_loss += loss.item()
                total_batches += 1

                # Step
                loss.backward()
                optimizer.step()

            avg_psnr = total_psnr / total_batches
            avg_ssim = total_ssim / total_batches
            avg_loss = total_loss / total_batches

            wandb.log({
                "psnr": avg_psnr,
                "ssim": avg_ssim,
                "loss": avg_loss})

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_state = model.state_dict()

                os.makedirs(os.path.join(
                    '..', 'models', dir_for_models), exist_ok=True)
                torch.save(best_model_state, os.path.join(
                    '..', 'models', dir_for_models, f'e{epoch}_l{avg_loss:.4f}.pth'))

    def run(self, directory_for_saving_models: str, input_path: str, output_path: str):
        """Just a test..."""

        learning_rate = 0.35
        batch_size = 64
        epochs = 1000

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

        original_data, style_data = self.__load_data(
            input_path=input_path,
            output_path=output_path,
            batch_size=batch_size
        )

        # simulate training
        self.__train(directory_for_saving_models, original_data,
                     style_data, epochs, learning_rate)

        wandb.finish()
