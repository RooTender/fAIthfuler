"""Provides augmentation techniques for images."""
import os
from pathlib import Path
from PIL import Image


class Augmentator:
    """Augments the raw data and outputs it to the desired folder"""

    def __init__(self, input_dir, output_dir):
        self.dir_to_strip = Path(input_dir)
        self.root_directory = Path(output_dir)

    def _get_path_with_suffix(self, path: str, suffix: str):
        directory = os.path.dirname(path)
        print(directory)
        if str(self.dir_to_strip) in path:
            directory = directory.replace(str(self.dir_to_strip), "")[1:]
        else:
            print(directory)
            directory = directory.replace(str(self.root_directory), "")[1:]
        filename = Path(path).stem
        return os.path.join(
            self.root_directory,
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

    def rotate(self, image_path, apply_function=None):
        """Rotate the texture by the specified angle (degrees)."""
        degrees = [90, 180, 270]
        for rotation in degrees:
            img = Image.open(image_path)
            img = img.rotate(rotation, resample=Image.NEAREST)
            path = self._save_with_suffix(
                img, image_path, f'rotated_{rotation}')
            if apply_function is not None:
                apply_function(path)
