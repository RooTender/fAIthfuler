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
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.01, inplace=True),

            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01, inplace=True),

            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01, inplace=True),

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
                num_of_epochs: int, learning_rate: float, early_stopping_patience: int,
                optimizer_type: str = 'adam',
                fine_tune_model: str = ''):
        model = FaithfulNet().cuda()

        if optimizer_type == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_type == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        else:
            raise ValueError("Unsupported optimizer type:", optimizer_type)

        criterion_mse = nn.MSELoss()
        criterion_ssim = piq.SSIMLoss()

        if fine_tune_model != '':
            checkpoint = torch.load(fine_tune_model)

            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch']
            best_loss = checkpoint['loss']

        else:
            start_epoch = 0
            best_loss = float('inf')

        no_improvement_count = 0

        for epoch in range(start_epoch, start_epoch + num_of_epochs):
            total_psnr = 0.0
            total_ssim = 0.0
            total_mse = 0.0

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

                # Loss
                # Rescale from [-1, 1] to [0, 1]
                prediction = (prediction + 1) / 2.0
                output_batch = (output_batch + 1) / 2.0

                mse_loss = criterion_mse(prediction, output_batch)
                ssim_loss = criterion_ssim(prediction, output_batch)
                loss = 0.8 * mse_loss + 0.2 * ssim_loss

                total_loss += loss.item()
                total_batches += 1

                # Metrics
                total_mse += mse_loss.item()
                total_psnr += piq.psnr(prediction, output_batch).item()
                total_ssim += piq.ssim(
                    prediction, output_batch).item()  # type: ignore

                # Step
                loss.backward()
                nn.utils.clip_grad.clip_grad_norm_(
                    model.parameters(), max_norm=1.0)
                optimizer.step()

            avg_mse = total_mse / total_batches
            avg_psnr = total_psnr / total_batches
            avg_ssim = total_ssim / total_batches
            avg_loss = total_loss / total_batches

            wandb.log({
                "mse": avg_mse,
                "psnr": avg_psnr,
                "ssim": avg_ssim,
                "loss": avg_loss})

            if avg_loss < best_loss:
                best_loss = avg_loss
                best_model_state = model.state_dict()
                no_improvement_count = 0

                os.makedirs(os.path.join(
                    '..', 'models', dir_for_models), exist_ok=True)
                torch.save(best_model_state, os.path.join(
                    '..', 'models', dir_for_models, f'e{epoch}_l{avg_loss:.4f}.pth'))
            else:
                no_improvement_count += 1
                if no_improvement_count >= early_stopping_patience:
                    print("Early stopping: No improvement for",
                          early_stopping_patience, "epochs.")
                    break

    def run(self, directory_for_saving_models: str,
            input_path: str, output_path: str,
            fine_tune_model: str = ''):
        """Just a test..."""

        # SGD performs the best on 0.3
        # Adam on 0.001
        learning_rate = 0.3

        # SGD on 32
        # Adam on ???
        batch_size = 16
        epochs = 10
        patience = epochs
        optimizer = 'sgd'

        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project="FAIthfuler",
            entity="rootender",
            name="FaithfulNet",
            # track hyperparameters and run metadata
            config={
                "architecture": "CNN",
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "hidden_layer_sizes": [256, 128, 64, 4],
                "optimizer": optimizer
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
        self.__train(f'{directory_for_saving_models}_b{batch_size}_lr{learning_rate}',
                     original_data, style_data,
                     epochs, learning_rate, patience,
                     optimizer_type=optimizer,
                     fine_tune_model=fine_tune_model)

        wandb.finish()
