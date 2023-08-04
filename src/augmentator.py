"""Provides augmentation techniques for images."""
import os
from multiprocessing import Pool
from pathlib import Path
from typing import Callable, List, Optional
import shutil
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from tqdm import tqdm


class Techniques:
    """Augmentation techniques that are applicable to this problem."""

    def __init__(self, input_dir: str, output_dir: str):
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

    def greyscale(self, image_path: str):
        """Apply greyscale to image."""
        img = Image.open(image_path).convert('LA').convert('RGBA')
        self._save_with_suffix(img, image_path, 'greyscaled')

    def rotate(self, image_path: str):
        """Rotate the image by the specified angle (90, 180 and 270 degrees)."""
        degrees = [90, 180, 270]
        for rotation in degrees:
            img = Image.open(image_path)
            img = img.rotate(rotation, resample=Image.NEAREST, expand=True)
            self._save_with_suffix(img, image_path, f'rotated_{rotation}')

    def mirror(self, image_path: str):
        """Mirror the image horizontally."""
        img = Image.open(image_path)
        img = ImageOps.mirror(img)
        self._save_with_suffix(img, image_path, 'mirrored')

    def flip(self, image_path: str):
        """Flip the image vertically."""
        img = Image.open(image_path)
        img = ImageOps.flip(img)
        self._save_with_suffix(img, image_path, 'flipped')

    def _jittering(self, image_path: str, value: float):
        img = Image.open(image_path)
        img_array = np.array(img)

        try:
            rgb = img_array[..., :3]
            rgb = np.clip(rgb * value, 0, 255).astype(np.uint8)
            img_array[..., :3] = rgb
            img = Image.fromarray(img_array)
            self._save_with_suffix(
                img,
                image_path,
                f'jittered{value}')
        except ValueError:
            pass
        except UnidentifiedImageError:
            print(f"{image_path} is corrupted.")

    def color_jitter(self, image_path: str):
        """Apply color jittering to image."""

        values = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.4, 1.5])
        np.vectorize(self._jittering)(image_path, values)


class Utils:
    """Utils for easier usage of augmentator."""

    def __init__(self, input_dir: str, output_dir: str):
        self.input_directory = Path(input_dir)
        self.output_directory = Path(output_dir)
        os.makedirs(self.output_directory, exist_ok=True)

    def equalize_matching_data(self, raw_input_relpath: str, raw_output_relpath: str):
        """Erases all the files that unmatch the output data"""
        input_files = []
        for root, _, files in os.walk(os.path.join(self.input_directory, raw_input_relpath)):
            for file in files:
                path = os.path.join(root, file)
                path = path.replace(os.path.join(
                    self.input_directory, raw_input_relpath), "")[1:]
                input_files.append(path)

        output_files = []
        for root, _, files in os.walk(os.path.join(self.input_directory, raw_output_relpath)):
            for file in files:
                path = os.path.join(root, file)
                path = path.replace(os.path.join(
                    self.input_directory, raw_output_relpath), "")[1:]
                output_files.append(path)

        unmatched_input_files = list(set(input_files) - set(output_files))
        unmatched_output_files = list(set(output_files) - set(input_files))

        for i, file in enumerate(unmatched_input_files):
            unmatched_input_files[i] = os.path.join(
                self.input_directory, raw_input_relpath, file)

        for i, file in enumerate(unmatched_output_files):
            unmatched_output_files[i] = os.path.join(
                self.input_directory, raw_output_relpath, file)

        for file_path in unmatched_input_files:
            if os.path.isfile(file_path):
                os.remove(file_path)

        for file_path in unmatched_output_files:
            if os.path.isfile(file_path):
                os.remove(file_path)

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

    def bulk_apply(self, function: Callable,
                   to_files: Optional[List[str]] = None, use_processed_images=False):
        """Apply on bulk functions to the choosen input directory."""
        files = []

        if to_files is None:
            if use_processed_images is False:
                for (dirpath, _, filenames) in os.walk(self.input_directory):
                    files += [os.path.join(dirpath, file)
                              for file in filenames]
            else:
                for (dirpath, _, filenames) in os.walk(self.output_directory):
                    files += [os.path.join(dirpath, file)
                              for file in filenames]
        else:
            files = to_files

        with Pool() as pool:
            list(tqdm(pool.imap_unordered(function, files), total=len(
                files), desc=function.__name__, unit="img"))

        return files
