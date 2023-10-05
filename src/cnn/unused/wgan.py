"""
This module implements the Wasserstein GAN (WGAN) 
algorithm for image translation.

It provides methods for training the generator and
critic networks using the WGAN loss function.
"""

import os
import shutil
import sys
import torch
from torch import optim
from dataset_loader import DatasetLoader, DataLoader
from tqdm import tqdm
from architectures._16x16 import wgan as network
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

    def __toggle_computation(self, model, enable_gradient_computation: bool):
        for parameter in model.parameters():
            parameter.requires_grad = enable_gradient_computation

    def __train(self, dir_for_models: str, train_set: DataLoader,
                learning_rate: float, num_of_epochs: int = -1, finetune_models_path: str = ''):
        models_path = os.path.join('..', 'models', dir_for_models)

        # Initialize Generator and Discriminator
        generator = network.Generator().cuda()
        critic = network.Critic().cuda()

        if finetune_models_path != '':
            for model_name in os.listdir(finetune_models_path):
                model_path = os.path.join(
                    finetune_models_path, model_name)
                if "generator" in model_name:
                    generator.load_state_dict(torch.load(model_path))
                else:
                    critic.load_state_dict(torch.load(model_path))

        # Optimizers
        optimizer_g = optim.RMSprop(generator.parameters(), lr=learning_rate)
        optimizer_c = optim.RMSprop(critic.parameters(), lr=learning_rate)

        if num_of_epochs == -1:
            num_of_epochs = 2147483647  # python doesn't have 'max' value so it's theoretical max

        one = torch.tensor(1, dtype=torch.float).cuda()
        minus_one = torch.tensor(-1, dtype=torch.float).cuda()

        for epoch in range(num_of_epochs):
            total_critic_loss = 0.0
            total_gan_loss = 0.0

            total_wasserstein_distance = 0.0
            total_real_loss = 0.0
            total_fake_loss = 0.0

            total_batches = 0

            # Training loop
            for (input_batch, output_batch) in tqdm(
                    train_set,
                    unit="batch",
                    desc=f'epoch {epoch}'):

                input_batch = input_batch.cuda()
                output_batch = output_batch.cuda()

                self.__toggle_computation(generator, False)
                self.__toggle_computation(critic, True)

                # In WGAN the critic is updated more frequently
                critic_iterations = 5
                for _ in range(critic_iterations):

                    # Train discriminator
                    optimizer_c.zero_grad()

                    # Real Images
                    real_data = output_batch
                    real_loss = torch.mean(critic(real_data))
                    real_loss.backward(minus_one)

                    # Fake Images
                    fake_data = generator(input_batch)
                    fake_loss = torch.mean(critic(fake_data))
                    fake_loss.backward(one)

                    # => The goal is to have balance, so we aim at value = 0
                    total_critic_loss += (fake_loss - real_loss) / critic_iterations
                    total_wasserstein_distance += (real_loss -
                                                   fake_loss) / critic_iterations
                    total_real_loss += real_loss / critic_iterations
                    total_fake_loss += fake_loss / critic_iterations

                    optimizer_c.step()

                # Train Generator
                self.__toggle_computation(generator, True)
                self.__toggle_computation(critic, False)

                optimizer_g.zero_grad()

                generated_data = generator(input_batch)
                generator_loss = torch.mean(critic(generated_data))
                generator_loss.backward(minus_one)

                total_gan_loss += -generator_loss

                optimizer_g.step()

                total_batches += 1

            # Log metrics
            avg_critic_loss = total_critic_loss / total_batches
            avg_gan_loss = total_gan_loss / total_batches
            avg_wasserstein_distance = total_wasserstein_distance / total_batches
            avg_real_loss = total_real_loss / total_batches
            avg_fake_loss = total_fake_loss / total_batches

            wandb.log({"Wasserstein distance": avg_wasserstein_distance,
                       "Critic loss": avg_critic_loss,
                       "Critic real loss": avg_real_loss,
                       "Critic fake loss": avg_fake_loss,
                       "GAN loss": avg_gan_loss})

            if epoch % 5 == 0 or epoch == num_of_epochs - 1:
                path_to_save = os.path.join(models_path, f'e{epoch}')
                try:
                    os.makedirs(path_to_save)
                except OSError:
                    shutil.rmtree(path_to_save)
                    os.makedirs(path_to_save)

                torch.save(generator.state_dict(), os.path.join(
                    path_to_save, f'generator_{avg_gan_loss}.pth'))
                torch.save(critic.state_dict(), os.path.join(
                    path_to_save, f'critic_{avg_critic_loss}.pth'))

    def train(self, finetune_model_path: str = '', wandb_id: str = ''):
        """
        Run the image translation model training and logging process.

        Args:
            finetune_from_epoch (int, optional): The epoch from which to fine-tune training.
                Default is `-1`, which means starting a new training session.

        This method initializes a new training run using WANDB for tracking and visualization.
        It sets up hyperparameters, loads data, and starts training the image translation model.
        """

        learning_rate = 0.0001
        batch_size = 16
        epochs = 500

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
                "architecture": "WGAN",
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "optimizer": "RMSProp"
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
                     epochs, finetune_model_path)

        wandb.finish()
