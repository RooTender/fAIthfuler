"""
Contains classes and functions for testing and visualizing
the performance of generator models on input images.

Classes:
    ModelTester: Class for testing and visualizing generator models.
"""

import os
from dataset_loader import DatasetLoader
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import transforms


class ModelTester:
    """
    Class for testing and visualizing the performance of a generator model on input images.

    Args:
        generator_model (nn.Module): The generator model to be tested.
        loader (DatasetLoader): An instance of the DatasetLoader class for loading data.
    """

    def __init__(self, generator_model, loader: DatasetLoader):
        self.model = generator_model
        self.dataloader = loader

    def test(self, model_path: str, images_to_test: int = 9):
        """
        Test and visualize the performance of a generator model on input images.

        Args:
            model_path (str): Path to the trained generator model's checkpoint.
            images_to_test (int, optional): Number of images to test and visualize. Default is `9`.
        """

        # Select 9 common random files between input_files and output_files
        images = self.dataloader.get_random_images(images_to_test)

        # Load the model
        generator = self.model().cuda()
        generator.load_state_dict(torch.load(model_path))
        generator.eval()

        # Initialize matplotlib
        _, axes = plt.subplots(len(images), 4, figsize=(20, 45))
        plt.subplots_adjust(hspace=0.4, wspace=0.3)

        # Populate each row
        for i, ax_row in enumerate(axes):
            ax_filename, ax_input, ax_ground, ax_generated = ax_row

            # Display filename
            ax_filename.axis('off')
            ax_filename.text(
                0.5, 0.5, os.path.basename(images[i][0]),
                horizontalalignment='center', verticalalignment='center')

            # Load and display the input image
            input_image = Image.open(images[i][0])
            ax_input.imshow(input_image)

            # Load and display the ground truth
            ground_image = Image.open(images[i][1])
            ax_ground.imshow(ground_image)

            # Preprocess the input image and run it through the model
            input_image = input_image.resize(
                ground_image.size, Image.Resampling.NEAREST)
            input_tensor = transforms.ToTensor()(
                input_image).unsqueeze(0).cuda()  # Add batch dimension and move to GPU
            with torch.no_grad():
                generated_tensor = generator(input_tensor)

            # Convert tensor to PIL Image and display it
            generated_image = transforms.ToPILImage()(generated_tensor.squeeze(0).cpu())
            ax_generated.imshow(generated_image)

        # Set column titles
        for axes, column in zip(axes[0], ['Filename', 'Input', 'Ground Truth', 'Generated']):
            axes.set_title(column)

        # Show the plot
        plt.show()
