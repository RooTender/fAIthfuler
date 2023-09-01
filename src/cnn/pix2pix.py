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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils.dataset_loader import DatasetLoader, DataLoader
from tqdm import tqdm
from architectures._16x16 import pix2pix as network
import wandb


class CNN:
    """
    The CNN (Convolutional Neural Network) class for training and 
    evaluating image translation models.

    This class provides methods for loading data, training the model, 
    and running the training process. It utilizes
    WANDB for tracking and visualization of training progress.
    """

    def __init__(self, loader: DatasetLoader):
        """
        Initialize the CNN object with input and output paths for data.

        Args:
            input_path (str): Path to the directory containing input images.
            output_path (str): Path to the directory containing corresponding ground truth images.
        """
        self.dataloader = loader

    def __train(self, dir_for_models: str, train_set: DataLoader,
                learning_rate: float, num_of_epochs: int = -1, finetune_from_epoch: int = -1):
        models_path = os.path.join('..', 'models', dir_for_models)

        # Initialize Generator and Discriminator
        generator = network.Generator().cuda()
        discriminator = network.Discriminator().cuda()

        if finetune_from_epoch >= 0:
            for model_name in os.listdir(os.path.join(models_path, f'e{finetune_from_epoch}')):
                model_path = os.path.join(
                    models_path, f'e{finetune_from_epoch}', model_name)
                if "generator" in model_name:
                    generator.load_state_dict(torch.load(model_path))
                else:
                    discriminator.load_state_dict(torch.load(model_path))

        # Optimizers
        optimizer_g = optim.Adam(
            generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
        optimizer_d = optim.Adam(
            discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))

        # Schedulers
        scheduler_g = ReduceLROnPlateau(
            optimizer_g, 'min', cooldown=2, patience=3, factor=0.5, threshold=0.01)

        plateau_count = 0
        last_lr = learning_rate

        # Losses
        criterion = nn.BCELoss()
        l1_lambda = 10

        training_start = finetune_from_epoch + 1
        if num_of_epochs == -1:
            num_of_epochs = 2147483647  # python doesn't have 'max' value so it's theoretical max

        for epoch in range(training_start, training_start + num_of_epochs):
            total_real_loss = 0.0
            total_fake_loss = 0.0
            total_gan_loss = 0.0

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
                labels = torch.ones(size=prediction.shape,
                                    dtype=torch.float).cuda()

                # slow down learning
                real_loss = 0.5 * criterion(prediction, labels)
                total_real_loss += real_loss
                real_loss.backward()

                # Fake Images
                gan_batch = generator(input_batch).detach()

                fake_data = torch.cat([gan_batch, output_batch], dim=1)
                prediction = discriminator(fake_data)
                labels = torch.zeros(size=prediction.shape,
                                     dtype=torch.float).cuda()

                # slow down learning
                fake_loss = 0.5 * criterion(prediction, labels)
                total_fake_loss += fake_loss
                fake_loss.backward()

                optimizer_d.step()

                # Train Generator
                for _ in range(5):
                    optimizer_g.zero_grad()

                    gan_batch = generator(input_batch)

                    gan_data = torch.cat([input_batch, gan_batch], dim=1)
                    prediction = discriminator(gan_data).detach()
                    labels = torch.ones(
                        size=prediction.shape, dtype=torch.float).cuda()

                    gan_loss = criterion(
                        prediction, labels) + l1_lambda * torch.abs(gan_batch - output_batch).mean()
                    total_gan_loss += gan_loss

                    gan_loss.backward()
                    optimizer_g.step()

                total_batches += 1

            # Log metrics
            avg_real_loss = total_real_loss / total_batches
            avg_fake_loss = total_fake_loss / total_batches
            avg_dicriminator_loss = (
                avg_real_loss + avg_fake_loss) / 2
            avg_gan_loss = total_gan_loss / total_batches
            avg_loss = (avg_gan_loss +
                        avg_dicriminator_loss) / 2

            wandb.log({"Discriminator real images loss": avg_real_loss,
                       "Discriminator fake images loss": avg_fake_loss,
                       "Discriminator loss": avg_dicriminator_loss,
                       "GAN loss": avg_gan_loss,
                       "Loss": avg_loss})

            # Update scheduler
            scheduler_g.step(avg_gan_loss)

            if optimizer_g.param_groups[0]['lr'] < last_lr:
                last_lr = optimizer_g.param_groups[0]['lr']
                print(f"Learning rate set to: {last_lr}")
                plateau_count += 1

            if epoch % 5 == 0 or epoch == training_start + num_of_epochs - 1:
                path_to_save = os.path.join(models_path, f'e{epoch}')
                try:
                    os.makedirs(path_to_save)
                except OSError:
                    shutil.rmtree(path_to_save)
                    os.makedirs(path_to_save)

                torch.save(generator.state_dict(), os.path.join(
                    path_to_save, f'generator_{avg_gan_loss:4f}.pth'))
                torch.save(discriminator.state_dict(), os.path.join(
                    path_to_save, f'discriminator_{avg_dicriminator_loss:4f}.pth'))

    def train(self, finetune_from_epoch: int = -1, wandb_id: str = ''):
        """
        Run the image translation model training and logging process.

        Args:
            finetune_from_epoch (int, optional): The epoch from which to fine-tune training.
                Default is `-1`, which means starting a new training session.

        This method initializes a new training run using WANDB for tracking and visualization.
        It sets up hyperparameters, loads data, and starts training the image translation model.
        """

        learning_rate = 0.002
        batch_size = 64
        epochs = -1

        dimensions = self.dataloader.get_images_dimension()

        # start a new wandb run to track this script
        wandb.init(
            resume=f'{wandb_id}',
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

        train_set = self.dataloader.load_data(batch_size)

        # simulate training
        self.__train(f'{dimensions}_b{batch_size}',
                     train_set, learning_rate,
                     epochs, finetune_from_epoch)

        wandb.finish()
