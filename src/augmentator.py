"""Provides augmentation techniques for images."""
import os
from pathlib import Path
from typing import Callable
import shutil
from PIL import Image, ImageOps
from tqdm import tqdm


class Techniques:
    """Augmentation techniques that are applicable to this problem."""

    def __init__(self, input_dir, output_dir):
        self.input_directory = Path(input_dir)
        self.output_directory = Path(output_dir)
        os.makedirs(os.path.dirname(output_dir), exist_ok=True)

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
        """Rotate the image by the specified angle (90, 180 and 270 degrees)."""
        degrees = [90, 180, 270]
        for rotation in degrees:
            img = Image.open(image_path)
            img = img.rotate(rotation, resample=Image.NEAREST, expand=True)
            self._save_with_suffix(img, image_path, f'rotated_{rotation}')

    def mirror(self, image_path):
        """Mirror the image horizontally."""
        img = Image.open(image_path)
        img = ImageOps.mirror(img)
        self._save_with_suffix(img, image_path, 'mirrored')

    def flip(self, image_path):
        """Flip the image vertically."""
        img = Image.open(image_path)
        img = ImageOps.flip(img)
        self._save_with_suffix(img, image_path, 'flipped')


class Utils:
    """Utils for easier usage of augmentator."""

    def __init__(self, input_dir, output_dir):
        self.input_directory = Path(input_dir)
        self.output_directory = Path(output_dir)
        os.makedirs(self.output_directory, exist_ok=True)

    def clone(self):
        """Clones images from input to output directory."""
        for (dirpath, _, filenames) in tqdm(
                os.walk(self.input_directory), desc="clone", unit="dir"):
            for filename in filenames:
                output_dir = dirpath.replace(
                    str(self.input_directory), str(self.output_directory))
                os.makedirs(output_dir, exist_ok=True)
                shutil.copy(os.path.join(dirpath, filename),
                            os.path.join(output_dir, filename))

    def bulk_apply(self, function: Callable, use_processed_images=False):
        """Apply on bulk functions to the choosen input directory."""
        files = []
        if use_processed_images is False:
            for (dirpath, _, filenames) in os.walk(self.input_directory):
                files += [os.path.join(dirpath, file) for file in filenames]
        else:
            for (dirpath, _, filenames) in os.walk(self.output_directory):
                files += [os.path.join(dirpath, file) for file in filenames]

        for file in tqdm(files, desc=function.__name__, unit="img"):
            function(file)
