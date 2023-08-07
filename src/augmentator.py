"""Provides augmentation techniques for images."""
import os
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Callable, List, Optional
import numpy as np
from PIL import Image, ImageOps
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

    def _save_with_suffix(self, img: Image.Image, path: str, suffix: str):
        """Saves image with suffix. Returns the path under which file is saved."""
        output = self._get_path_with_suffix(path, suffix)
        os.makedirs(os.path.dirname(output), exist_ok=True)
        img.convert('RGBA').save(output)
        return output

    def rotate(self, image_path: str):
        """Rotate the image by the specified angle (90, 180 and 270 degrees)."""
        degrees = [90, 180, 270]
        for rotation in degrees:
            img = Image.open(image_path)
            img = img.rotate(
                rotation, resample=Image.Resampling.NEAREST, expand=True)
            self._save_with_suffix(img, image_path, f'rotated_{rotation}')
            img.close()

    def mirror(self, image_path: str):
        """Mirror the image horizontally."""
        img = Image.open(image_path)
        img = ImageOps.mirror(img)
        self._save_with_suffix(img, image_path, 'mirrored')
        img.close()

    def flip(self, image_path: str):
        """Flip the image vertically."""
        img = Image.open(image_path)
        img = ImageOps.flip(img)
        self._save_with_suffix(img, image_path, 'flipped')
        img.close()

    def invert(self, image_path: str):
        """Invert colors of the image."""
        img = Image.open(image_path)

        red, green, blue, alpha = img.split()

        red = ImageOps.invert(red)
        green = ImageOps.invert(green)
        blue = ImageOps.invert(blue)

        img = Image.merge('RGBA', (red, green, blue, alpha))

        self._save_with_suffix(img, image_path, 'inverted')
        img.close()

    def _jittering(self, image_path: str, hue: float, sat: float, val: float):
        img = Image.open(image_path)

        alpha = img.getchannel('A')
        hsv_image = np.array(img.convert('HSV'))
        hsv_image[:, :, 0] = np.clip(hsv_image[:, :, 0] * hue, 0, 255)
        hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * sat, 0, 255)
        hsv_image[:, :, 2] = np.clip(hsv_image[:, :, 2] * val, 0, 255)
        img = Image.fromarray(hsv_image, 'HSV').convert('RGBA')
        img.putalpha(alpha)

        self._save_with_suffix(
            img,
            image_path,
            f'_hue{hue}_saturation{sat}_value{val}')
        img.close()

    def color_jitter(self, image_path: str):
        """Apply color jittering to image."""

        values = np.array([0.5, 1.5, 2])

        combinations = np.array(list(np.unique(np.array(np.meshgrid(
            values, values, values)).T.reshape(-1, 3), axis=0)))
        for combination in combinations:
            np.vectorize(self._jittering)(
                image_path, combination[0], combination[1], combination[2])


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
        def convert_to_rgba(image_path: str, output_path: str):
            """Converts files to RGBA channel."""
            img = Image.open(image_path)
            img = img.convert('RGBA')
            img.save(output_path)
            img.close()

        for (dirpath, _, filenames) in tqdm(
                os.walk(self.input_directory), desc="clone", unit="dir"):
            for filename in filenames:
                output_dir = dirpath.replace(
                    str(self.input_directory), str(self.output_directory))
                os.makedirs(output_dir, exist_ok=True)
                convert_to_rgba(os.path.join(dirpath, filename),
                                os.path.join(output_dir, filename))

    def _batch_process(self, args: tuple[Callable, List[str]]) -> None:
        """Helper function to apply the function to a batch of files."""
        function, file_batch = args
        for file in file_batch:
            function(file)

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

        batch_size = min(1000, max(
            1, int(len(files) / (np.log(len(files)) * cpu_count()))))

        # Split files into batches
        file_batches = [files[i:i + batch_size]
                        for i in range(0, len(files), batch_size)]

        with Pool() as pool, tqdm(
                total=len(file_batches),
                desc=function.__name__,
                unit="batch") as pbar:
            for _ in pool.imap_unordered(self._batch_process,
                                         [(function, file_batch) for file_batch in file_batches]):
                pbar.update(1)

        return files
