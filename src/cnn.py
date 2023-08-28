"""
This module provides classes and methods for training and 
evaluating image translation models using PyTorch and WANDB.

Classes:
    - ImageLoaderDataset: A PyTorch dataset for loading pairs of 
    input and output images from specified directories.
    - CNN: Trains neural networks and evaluates them.
"""
import os
import random
import shutil
import sys
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
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

        input_image = input_image.resize(
            output_image.size, Image.Resampling.NEAREST)

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

    def __init__(self, input_path: str, output_path: str):
        """
        Initialize the CNN object with input and output paths for data.

        Args:
            input_path (str): Path to the directory containing input images.
            output_path (str): Path to the directory containing corresponding ground truth images.
        """
        self.input_path = input_path
        self.output_path = output_path

    def __load_data(self, batch_size: int):
        print(f'Input dataset: {self.input_path}')
        print(f'Output dataset: {self.output_path}')
        print('Reading datasets...')

        transformator = transforms.Compose([
            transforms.ToTensor()  # Convert images to tensors
        ])

        input_data = ImageLoaderDataset(
            self.input_path, self.output_path, transform=transformator)

        train_loader = DataLoader(input_data, batch_size=batch_size,
                                  shuffle=True, pin_memory=True, num_workers=4)

        return train_loader

    def __train(self, dir_for_models: str, train_set: DataLoader,
                num_of_epochs: int, learning_rate: float, finetune_from_epoch: int = -1):
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
            generator.parameters(), lr=learning_rate, betas=(0.99, 0.999))
        optimizer_d = optim.Adam(
            discriminator.parameters(), lr=learning_rate, betas=(0.99, 0.999))

        # Losses
        criterion = nn.BCELoss()
        l1_lambda = 100

        training_start = finetune_from_epoch + 1
        for epoch in range(training_start, training_start + num_of_epochs):
            total_real_loss = 0.0
            total_fake_loss = 0.0
            total_gan_loss = 0.0

            total_batches = 0

            if epoch < 200:
                l1_lambda = 100
            else:
                l1_lambda = 50

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
                for _ in range(2):
                    optimizer_g.zero_grad()

                    gan_batch = generator(input_batch)

                    gan_data = torch.cat([input_batch, gan_batch], dim=1)
                    prediction = discriminator(gan_data).detach()
                    labels = torch.ones(
                        size=prediction.shape, dtype=torch.float).cuda()

                    gan_loss = criterion(
                        prediction, labels) + l1_lambda * torch.mean(torch.abs(gan_batch - output_batch))
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

    def run(self, finetune_from_epoch: int = -1):
        """
        Run the image translation model training and logging process.

        Args:
            finetune_from_epoch (int, optional): The epoch from which to fine-tune training.
                Default is `-1`, which means starting a new training session.

        This method initializes a new training run using WANDB for tracking and visualization.
        It sets up hyperparameters, loads data, and starts training the image translation model.
        """

        learning_rate = 0.0002
        batch_size = 64
        epochs = 100

        dimensions = os.path.basename(os.path.normpath(self.input_path))

        # start a new wandb run to track this script
        wandb.init(
            resume=(finetune_from_epoch >= 0),
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

        train_set = self.__load_data(batch_size)

        # simulate training
        self.__train(f'{dimensions}_b{batch_size}_lr{learning_rate}',
                     train_set, epochs, learning_rate, finetune_from_epoch)

        wandb.finish()

    def test(self, model_path: str, images_to_test: int = 9):
        """
        Test and visualize the performance of a generator model on input images.

        Args:
            model_path (str): Path to the trained generator model's checkpoint.
            images_to_test (int, optional): Number of images to test and visualize. Default is `9`.
        """

        # Select 9 common random files between input_files and output_files
        selected_files = random.sample(
            os.listdir(self.input_path), images_to_test)

        # Load the model
        generator = network.Generator().cuda()
        generator.load_state_dict(torch.load(model_path))
        generator.eval()

        # Initialize matplotlib
        _, axes = plt.subplots(len(selected_files), 4, figsize=(20, 45))
        plt.subplots_adjust(hspace=0.4, wspace=0.3)
        columns = ['Filename', 'Input', 'Ground Truth', 'Generated']

        # Populate each row
        for i, ax_row in enumerate(axes):
            ax_filename, ax_input, ax_ground, ax_generated = ax_row

            # Display filename
            ax_filename.axis('off')
            ax_filename.text(
                0.5, 0.5, selected_files[i], horizontalalignment='center', verticalalignment='center')

            # Load and display the input image
            input_image_path = os.path.join(self.input_path, selected_files[i])
            input_image = Image.open(input_image_path)
            ax_input.imshow(input_image)

            # Load and display the ground truth
            ground_image_path = os.path.join(
                self.output_path, selected_files[i])
            ground_image = Image.open(ground_image_path)
            ax_ground.imshow(ground_image)

            # Preprocess the input image and run it through the model
            input_image = input_image.resize(
                ground_image.size, Image.Resampling.NEAREST)
            transform = transforms.ToTensor()
            input_tensor = transform(input_image).unsqueeze(
                0).cuda()  # Add batch dimension and move to GPU
            with torch.no_grad():
                generated_tensor = generator(input_tensor)

            # Convert tensor to PIL Image and display it
            generated_image = transforms.ToPILImage()(generated_tensor.squeeze(0).cpu())
            ax_generated.imshow(generated_image)

        # Set column titles
        for axes, column in zip(axes[0], columns):
            axes.set_title(column)

        # Show the plot
        plt.show()
