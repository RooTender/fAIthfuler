"""Provides augmentation techniques for images."""
import os
from pathlib import Path
from typing import Callable, List
from PIL import Image


class Augmentator:
    """Augments the raw data and outputs it to the desired folder"""

    def __init__(self, input_dir, output_dir):
        self.input_directory = Path(input_dir)
        self.output_directory = Path(output_dir)

    def _get_path_with_suffix(self, path: str, suffix: str):
        directory = os.path.dirname(path)
        if str(self.input_directory) in path:
            directory = directory.replace(str(self.input_directory), "")[1:]
        else:
            directory = directory.replace(str(self.output_directory), "")[1:]
        filename = Path(path).stem
        return os.path.join(
            self.output_directory,
            Path(directory),
            Path(f"{filename}_{suffix}.png")
        )

    def _save_with_suffix(self, img, path: str, suffix: str):
        """Saves image with suffix. Returns the path under which file is saved."""
        output = self._get_path_with_suffix(path, suffix)
        os.makedirs(os.path.dirname(output), exist_ok=True)
        img.save(output)
        return output

    def greyscale(self, image_path):
        """Apply greyscale to image."""
        img = Image.open(image_path).convert('LA')
        self._save_with_suffix(img, image_path, 'greyscaled')

    def rotate(self, image_path):
        """Rotate the texture by the specified angle (degrees)."""
        degrees = [90, 180, 270]
        for rotation in degrees:
            img = Image.open(image_path)
            img = img.rotate(rotation, resample=Image.NEAREST)
            self._save_with_suffix(img, image_path, f'rotated_{rotation}')

    def bulk_apply(self, function: Callable, use_processed_images=False):
        """Apply on bulk functions to the choosen input directory."""
        files = []
        if use_processed_images is False:
            for (dirpath, _, filenames) in os.walk(self.input_directory):
                files += [os.path.join(dirpath, file) for file in filenames]
        else:
            for (dirpath, _, filenames) in os.walk(self.output_directory):
                files += [os.path.join(dirpath, file) for file in filenames]

        for file in files:
            function(file)
