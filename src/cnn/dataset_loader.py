"""
This module provides classes and functions for loading datasets and creating data loaders
for training and processing image pairs.

Classes:
    - ImageLoaderDataset: A PyTorch dataset for loading pairs of input and output images.
    - DatasetLoader: A class for loading datasets and providing data loaders.

Functions:
    - None

Usage:
    - Import the classes and functions from this module to load and process image datasets.
"""

import os
import random
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


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


class DatasetLoader:
    """
    A class for loading datasets and providing data loaders.

    Args:
        input_path (str): Path to the input dataset.
        ground_truth_path (str): Path to the ground truth dataset.
    """

    def __init__(self, input_path: str, ground_truth_path: str):
        self.input_path = input_path
        self.output_path = ground_truth_path

    def load_data(self, batch_size: int):
        """
        Load the dataset and create a data loader.

        Args:
            batch_size (int): Batch size for the data loader.

        Returns:
            DataLoader: A PyTorch DataLoader for the dataset.
        """
        print(f'Input dataset: {self.input_path}')
        print(f'Output dataset: {self.output_path}')
        print('Reading datasets...')

        transformator = transforms.Compose([
            transforms.ToTensor()  # Convert images to tensors
        ])

        train_data = ImageLoaderDataset(
            os.path.join(self.input_path, 'train'),
            os.path.join(self.output_path, 'train'),
            transform=transformator)

        valid_data = ImageLoaderDataset(
            os.path.join(self.input_path, 'valid'),
            os.path.join(self.output_path, 'valid'),
            transform=transformator)

        train_loader = DataLoader(train_data, batch_size=batch_size,
                                  shuffle=True, pin_memory=True, num_workers=4)
        val_loader = DataLoader(valid_data, batch_size=batch_size,
                                shuffle=False, pin_memory=True, num_workers=4)

        return train_loader, val_loader

    def get_random_images(self, amount: int):
        """
        Get a list of random image pairs (input path, output path).

        Args:
            amount (int): Number of random image pairs to retrieve.

        Returns:
            list: List of tuples containing random input and ground truth image file paths.
        """

        basenames = random.sample(os.listdir(os.path.join(self.input_path, 'valid')), amount)
        return [
            (
                os.path.join(self.input_path, 'valid', basenames[i]),
                os.path.join(self.output_path, 'valid', basenames[i])
            )
            for i in range(amount)
        ]

    def get_images_dimension(self):
        return os.path.basename(os.path.normpath(self.input_path))
