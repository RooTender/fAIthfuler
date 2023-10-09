"""
This module provides classes and methods for training and 
evaluating image translation models using PyTorch and WANDB.
However, it also uses ResNet for more stable and faster learning.

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
from dataset_loader import DatasetLoader, DataLoader # pylint: disable=import-error
from model_tester import ModelTester
from tqdm import tqdm
from architectures._32x32 import tiny_pix2pix as network
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

    def __train(self, dir_for_models: str, train_set: DataLoader, val_set: DataLoader,
                learning_rate: float, num_of_epochs: int = -1, finetune_from_epoch: int = -1,
                save_models: bool = True, generator_iterations: int = 1, l1_lambda: int = 100):
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

        # Losses
        criterion_bce = nn.BCELoss()
        criterion_l1 = nn.L1Loss()
        # l1_lambda = 100

        training_start = finetune_from_epoch + 1
        if num_of_epochs == -1:
            num_of_epochs = sys.maxsize

        for epoch in range(training_start, training_start + num_of_epochs):
            total_real_loss = 0.0
            total_fake_loss = 0.0
            total_gan_loss = 0.0

            total_batches = 0

            # Training
            generator.train()
            discriminator.train()

            for (input_batch, output_batch) in tqdm(
                    train_set,
                    unit="batch",
                    desc=f'epoch {epoch}'):

                input_batch = input_batch.cuda()
                output_batch = output_batch.cuda()

                # Train discriminator
                optimizer_d.zero_grad()

                # Real Images
                prediction = discriminator(input_batch, output_batch)
                labels = torch.ones(size=prediction.shape,
                                    dtype=torch.float).cuda()

                real_loss = criterion_bce(prediction, labels)
                total_real_loss += real_loss
                real_loss.backward()

                # Fake Images
                gan_batch = generator(input_batch).detach()

                prediction = discriminator(gan_batch, output_batch)
                labels = torch.zeros(size=prediction.shape,
                                     dtype=torch.float).cuda()

                fake_loss = criterion_bce(prediction, labels)
                total_fake_loss += fake_loss
                fake_loss.backward()

                optimizer_d.step()

                # Train Generator
                for _ in range(generator_iterations):
                    optimizer_g.zero_grad()

                    gan_batch = generator(input_batch)

                    prediction = discriminator(input_batch, gan_batch).detach()
                    labels = torch.ones(
                        size=prediction.shape, dtype=torch.float).cuda()

                    gan_loss = criterion_bce(prediction, labels
                                             ) + l1_lambda * criterion_l1(gan_batch, output_batch)

                    total_gan_loss += gan_loss / (l1_lambda + 1)

                    gan_loss.backward()
                    optimizer_g.step()

                total_batches += 1

            total_gan_loss /= generator_iterations

            # Validating
            generator.eval()
            discriminator.eval()

            val_loss = 0.0
            total_val_batches = 0

            with torch.no_grad():
                for (input_batch, output_batch) in tqdm(
                    val_set,
                    unit="batch",
                    desc=f'validating epoch {epoch}'):

                    input_batch = input_batch.cuda()
                    output_batch = output_batch.cuda()

                    # Generator
                    gan_batch = generator(input_batch)

                    prediction = discriminator(input_batch, gan_batch)
                    labels = torch.ones(
                        size=prediction.shape, dtype=torch.float).cuda()
                    gan_loss = criterion_bce(prediction, labels
                                ) + l1_lambda * criterion_l1(gan_batch, output_batch)

                    val_loss += gan_loss / (l1_lambda + 1)

                    total_val_batches += 1

            # Log train metrics
            avg_real_loss = total_real_loss / total_batches
            avg_fake_loss = total_fake_loss / total_batches
            avg_dicriminator_loss = (
                avg_real_loss + avg_fake_loss) / 2
            avg_gan_loss = total_gan_loss / total_batches
            avg_loss = (avg_gan_loss +
                        avg_dicriminator_loss) / 2

            # Log validation metrics
            avg_val_loss = val_loss / total_val_batches

            wandb.log({"Discriminator real images loss": avg_real_loss,
                       "Discriminator fake images loss": avg_fake_loss,
                       "Discriminator loss": avg_dicriminator_loss,
                       "GAN loss": avg_gan_loss,
                       "Loss": avg_loss,
                       "Validation GAN loss": avg_val_loss})

            if save_models and (epoch % 5 == 0 or epoch == training_start + num_of_epochs - 1):
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

    def __sweep_train(self):
        wandb.init()
        config = wandb.config

        dimensions = self.dataloader.get_images_dimension()
        train_set, val_set = self.dataloader.load_data(config.batch_size)

        self.__train(dimensions, train_set, val_set,
                     config.learning_rate,
                     25,
                     save_models=False,
                     generator_iterations=config.generator_iterations,
                     l1_lambda=config.l1_lambda)

    def train(self, finetune_from_epoch: int = -1, wandb_id: str = ''):
        """
        Run the image translation model training and logging process.

        Args:
            finetune_from_epoch (int, optional): The epoch from which to fine-tune training.
                Default is `-1`, which means starting a new training session.

        This method initializes a new training run using WANDB for tracking and visualization.
        It sets up hyperparameters, loads data, and starts training the image translation model.
        """

        learning_rate = 0.001
        batch_size = 64
        epochs = 100

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
                "architecture": "pix2pix",
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

        train_set, val_set = self.dataloader.load_data(batch_size)

        # simulate training
        self.__train(f'{dimensions}_b{batch_size}',
                     train_set, val_set, learning_rate,
                     epochs, finetune_from_epoch,
                     l1_lambda=2)

        wandb.finish()

    def sweep(self):
        dimensions = self.dataloader.get_images_dimension()

        sweep_config = {
            'name': f"FaithfulNet_{dimensions}",
            'method': 'grid',
            'metric': {
                'name': 'Validation GAN loss',
                'goal': 'minimize'   
            },
            'parameters': {
                'learning_rate': {
                    'values': [0.001]
                },
                'batch_size': {
                    'values': [64]
                },
                'generator_iterations': {
                    'values': [1, 5, 10]
                },
                'l1_lambda': {
                    'values': [10, 50, 100]
                }
            }
        }

        sweep_id = wandb.sweep(sweep_config, project="FAIthfuler", entity="rootender")
        wandb.agent(sweep_id, function=self.__sweep_train)

    def test_model(self, model_path: str, images_to_test: int = 9):
        model = network.Generator()
        tester = ModelTester(model, self.dataloader)
        tester.test(model_path, images_to_test)
