"""
This module provides classes and methods for training and 
evaluating image translation models using PyTorch and WANDB.

Classes:
    - ImageLoaderDataset: A PyTorch dataset for loading pairs of 
    input and output images from specified directories.
    - CNN: Trains neural networks and evaluates them.
"""
import os
import shutil
import sys
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms
import numpy as np
from PIL import Image
from tqdm import tqdm
import network
import wandb

class ImageLoaderDataset(Dataset):
    """
    A PyTorch dataset for loading pairs of input and output images from specified directories.

    Args:
        input_root_dir (str): The root directory containing input images.
        output_root_dir (str): The root directory containing corresponding output images.
        transform (callable, optional): A function/transform to apply to the images.

    Attributes:
        input_root_dir (str): The root directory containing input images.
        output_root_dir (str): The root directory containing corresponding output images.
        transform (callable, optional): A function/transform to apply to the images.
        image_paths (list of tuple): A list of tuples containing input and output image file paths.
    """
    def __init__(self, input_root_dir, output_root_dir, transform=None):
        self.input_root_dir = input_root_dir
        self.output_root_dir = output_root_dir
        self.transform = transform
        self.image_paths = [(os.path.join(input_root_dir, file),
                             os.path.join(output_root_dir, file))
                            for file in os.listdir(input_root_dir)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        input_image_path, output_image_path = self.image_paths[idx]
        input_image = Image.open(input_image_path)
        output_image = Image.open(output_image_path)

        input_image = input_image.resize(output_image.size, Image.Resampling.NEAREST)

        if self.transform:
            input_image = self.transform(input_image)
            output_image = self.transform(output_image)

        return input_image, output_image


class CNN:
    """
    The CNN (Convolutional Neural Network) class for training and 
    evaluating image translation models.

    This class provides methods for loading data, training the model, 
    and running the training process. It utilizes
    WANDB for tracking and visualization of training progress.
    """
    def __load_data(self, input_path: str, output_path: str,
                    batch_size: int, validation_split: float = 0.2, test_split: float = 0.1):
        print(f'Input dataset: {input_path}')
        print(f'Output dataset: {output_path}')
        print('Reading datasets...')

        transformator = transforms.Compose([
            transforms.ToTensor()  # Convert images to tensors
        ])

        input_data = ImageLoaderDataset(
            input_path, output_path, transform=transformator)

        # Splitting the dataset into training, validation, and test sets
        num_train = len(input_data)
        indices = list(range(num_train))

        v_split = int(np.floor(validation_split * num_train))
        t_split = int(np.floor(test_split * num_train))

        rng = np.random.default_rng(seed=123)
        rng.shuffle(indices)

        test_idx, val_idx, train_idx = indices[:t_split], indices[t_split:t_split+v_split], indices[t_split+v_split:]

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(val_idx)
        test_sampler = SubsetRandomSampler(test_idx)

        train_loader = DataLoader(input_data, batch_size=batch_size,
                                sampler=train_sampler, pin_memory=True, num_workers=4)

        valid_loader = DataLoader(input_data, batch_size=batch_size,
                                sampler=test_sampler, pin_memory=True, num_workers=4)

        test_loader = DataLoader(input_data, batch_size=batch_size,
                                sampler=valid_sampler, pin_memory=True, num_workers=4)

        return train_loader, valid_loader, test_loader

    def __train(self, dir_for_models: str,
                train_set: DataLoader, valid_set: DataLoader,
                num_of_epochs: int, learning_rate: float, finetune_from_epoch: int = -1):
        models_path = os.path.join('..', 'models', dir_for_models)

        # Initialize Generator and Discriminator
        generator = network.Generator().cuda()
        discriminator = network.Discriminator().cuda()

        if finetune_from_epoch >= 0:
            for model_name in os.listdir(os.path.join(models_path, f'e{finetune_from_epoch}')):
                model_path = os.path.join(models_path, f'e{finetune_from_epoch}', model_name)
                if "generator" in model_name:
                    generator.load_state_dict(torch.load(model_path))
                else:
                    discriminator.load_state_dict(torch.load(model_path))

        # Optimizers
        optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

        # Losses
        criterion = nn.BCELoss()
        l1_lambda = 100

        training_start = finetune_from_epoch + 1
        for epoch in range(training_start, num_of_epochs + training_start):
            total_train_real_loss = 0.0
            total_train_fake_loss = 0.0
            total_train_gan_loss = 0.0

            total_batches = 0

            # Training loop
            for (input_batch, output_batch) in tqdm(
                    train_set,
                    unit="batch",
                    desc=f'epoch {epoch}'):

                input_batch = input_batch.cuda()
                output_batch = output_batch.cuda()

                # Train discriminator
                optimizer_d.zero_grad()

                # Real Images
                real_data = torch.cat([input_batch, output_batch], dim=1)
                prediction = discriminator(real_data)
                labels = torch.ones(size=prediction.shape, dtype=torch.float).cuda()

                real_loss = 0.5 * criterion(prediction, labels)  # slow down learning
                total_train_real_loss += real_loss
                real_loss.backward()

                # Fake Images
                gan_batch = generator(input_batch).detach()

                fake_data = torch.cat([gan_batch, output_batch], dim=1)
                prediction = discriminator(fake_data)
                labels = torch.zeros(size=prediction.shape, dtype=torch.float).cuda()

                fake_loss = 0.5 * criterion(prediction, labels)  # slow down learning
                total_train_fake_loss += fake_loss
                fake_loss.backward()

                optimizer_d.step()

                # Train Generator
                for _ in range(2):
                    optimizer_g.zero_grad()

                    gan_batch = generator(input_batch)

                    gan_data = torch.cat([input_batch, gan_batch], dim=1)
                    prediction = discriminator(gan_data).detach()
                    labels = torch.ones(size=prediction.shape, dtype=torch.float).cuda()

                    gan_loss = criterion(prediction, labels) + l1_lambda * torch.mean(torch.abs(gan_batch - output_batch))
                    total_train_gan_loss += gan_loss

                    gan_loss.backward()
                    optimizer_g.step()

                total_batches += 1

            # Validation loop
            generator.eval()
            discriminator.eval()

            with torch.no_grad():
                total_val_real_loss = 0.0
                total_val_fake_loss = 0.0
                total_val_gan_loss = 0.0

                for (input_batch, output_batch) in tqdm(
                    valid_set,
                    unit="batch",
                    desc=f'validation epoch {epoch}'):
                    input_batch = input_batch.cuda()
                    output_batch = output_batch.cuda()

                    # Discriminator validation
                    # Real Images
                    real_data = torch.cat([input_batch, output_batch], dim=1)
                    prediction = discriminator(real_data)
                    labels = torch.ones(size=prediction.shape, dtype=torch.float).cuda()

                    real_loss = 0.5 * criterion(prediction, labels)  # slow down learning
                    total_val_real_loss += real_loss

                    # Fake Images
                    gan_batch = generator(input_batch)

                    fake_data = torch.cat([gan_batch, output_batch], dim=1)
                    prediction = discriminator(fake_data)
                    labels = torch.zeros(size=prediction.shape, dtype=torch.float).cuda()

                    fake_loss = 0.5 * criterion(prediction, labels)  # slow down learning
                    total_val_fake_loss += fake_loss

                    # GAN validation
                    gan_batch = generator(input_batch)

                    gan_data = torch.cat([input_batch, gan_batch], dim=1)
                    prediction = discriminator(gan_data)
                    labels = torch.ones(size=prediction.shape, dtype=torch.float).cuda()

                    gan_loss = criterion(prediction, labels) + l1_lambda * torch.mean(torch.abs(gan_batch - output_batch))
                    total_val_gan_loss += gan_loss

            generator.train()
            discriminator.train()

            # Log metrics
            # Train...
            avg_train_real_loss = total_train_real_loss / total_batches
            avg_train_fake_loss = total_train_fake_loss / total_batches
            avg_train_dicriminator_loss = (avg_train_real_loss + avg_train_fake_loss) / 2
            avg_train_gan_loss = total_train_gan_loss / total_batches
            avg_train_loss = (avg_train_gan_loss + avg_train_dicriminator_loss) / 2

            # Validation...
            avg_val_real_loss = total_val_real_loss / total_batches
            avg_val_fake_loss = total_val_fake_loss / total_batches
            avg_val_dicriminator_loss = (avg_val_real_loss + avg_val_fake_loss) / 2
            avg_val_gan_loss = total_val_gan_loss / total_batches
            avg_val_loss = (avg_val_gan_loss + avg_val_dicriminator_loss) / 2

            wandb.log({"train": {"Discriminator real images loss": avg_train_real_loss,
                                 "Discriminator fake images loss": avg_train_fake_loss,
                                 "Discriminator loss": avg_train_dicriminator_loss,
                                 "GAN loss": avg_train_gan_loss,
                                 "Loss": avg_train_loss},

                       "validation": {"Discriminator real images loss": avg_val_real_loss,
                                      "Discriminator fake images loss": avg_val_fake_loss,
                                      "Discriminator loss": avg_val_dicriminator_loss,
                                      "GAN loss": avg_val_gan_loss,
                                      "Loss": avg_val_loss}})

            if epoch % 5 == 0 or epoch == num_of_epochs - 1:
                path_to_save = os.path.join(models_path, f'e{epoch}')
                try:
                    os.makedirs(path_to_save)
                except OSError:
                    shutil.rmtree(path_to_save)
                    os.makedirs(path_to_save)

                torch.save(generator.state_dict(), os.path.join(
                    path_to_save, f'generator_{avg_val_gan_loss:4f}.pth'))
                torch.save(discriminator.state_dict(), os.path.join(
                    path_to_save, f'discriminator_{avg_val_dicriminator_loss:4f}.pth'))

    def run(self, input_path: str, output_path: str, finetune_from_epoch: int = -1):
        """
        Run the image translation model training and logging process.

        Args:
            input_path (str): The directory path containing input images.
            output_path (str): The directory path containing corresponding output images.
            finetune_from_epoch (int, optional): The epoch from which to fine-tune training.
                Default is `-1`, which means starting a new training session.

        This method initializes a new training run using WANDB for tracking and visualization.
        It sets up hyperparameters, loads data, and starts training the image translation model.
        """

        learning_rate = 0.0002
        batch_size = 64
        epochs = 100

        dimensions = os.path.basename(os.path.normpath(input_path))

        # start a new wandb run to track this script
        wandb.init(
            resume = (finetune_from_epoch >= 0),
            # set the wandb project where this run will be logged
            project="FAIthfuler",
            entity="rootender",
            name=f"FaithfulNet_{dimensions}",
            # track hyperparameters and run metadata
            config={
                "architecture": "CNN",
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "optimizer": "Adam"
            }
        )

        if not torch.cuda.is_available():
            print("CUDA architecture is unavailable!")
            sys.exit(1)
        else:
            print("CUDA detected! Initializing...")

        train_set, validation_set, _ = self.__load_data(
            input_path=input_path,
            output_path=output_path,
            batch_size=batch_size
        )

        # simulate training
        self.__train(f'{dimensions}_b{batch_size}_lr{learning_rate}',
                     train_set, validation_set, epochs, learning_rate, finetune_from_epoch)

        wandb.finish()
