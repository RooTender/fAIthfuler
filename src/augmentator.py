"""Provides augmentation techniques for images."""
import os
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Callable, List, Optional
import shutil
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

    def _batch_process(self, args: tuple[Callable, List[str]]) -> None:
        """Helper function to apply the function to a batch of files."""
        function, file_batch = args
        for file in file_batch:
            function(file)

    def augment_data(self, techniques_instance: Techniques, function: Callable,
                     to_files: Optional[List[str]] = None, use_processed_images=False):
        """Apply on bulk functions to the choosen input directory."""
        files = []

        if to_files is None:
            if use_processed_images is False:
                for (dirpath, _, filenames) in os.walk(techniques_instance.input_directory):
                    files += [os.path.join(dirpath, file)
                              for file in filenames]
            else:
                for (dirpath, _, filenames) in os.walk(techniques_instance.output_directory):
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

    def preprocess_data(self, input_dir: str):
        for root, _, files in tqdm(os.walk(input_dir), desc=f"sort ({os.path.basename(os.path.normpath(input_dir))})", unit="dir"):
            for file in files:
                file_path = os.path.join(root, file)
                with Image.open(file_path) as img:
                    dim = img.size

                    if dim[1] > dim[0]:
                        dim = (dim[1], dim[0])
                    img = Image.open(file_path)
                    img = img.convert('RGBA')

                    output_dir = os.path.join(
                        self.output_directory,
                        'preprocessed',
                        os.path.basename(os.path.normpath(input_dir)),
                        f'{dim[0]}x{dim[1]}')
                    os.makedirs(output_dir, exist_ok=True)
                    img.save(os.path.join(output_dir, file))

    def normalize_data(self, input_dir: str, output_dir: str, output_scale_ratio: int):

        def collect_files_info(directory: str):
            result = []
            for root, _, files in os.walk(directory):
                for file in files:
                    path = os.path.join(root, file)
                    path = path.replace(directory, "")[1:]
                    result.append(path)
            return result

        input_files = collect_files_info(input_dir)
        output_files = collect_files_info(output_dir)

        def scaled_filename(file: str, scale: int):
            dimensions, rest = os.path.split(file)
            width, height = map(int, dimensions.split('x'))
            new_dimensions = f"{width * scale}x{height * scale}"
            return os.path.join(new_dimensions, rest)

        aux_input_files = list(
            map(lambda x: scaled_filename(x, output_scale_ratio), input_files))

        # Find common files
        common_filenames = list(set(aux_input_files) & set(output_files))

        output_dir_for_input = os.path.join(
            self.output_directory,
            'normalized',
            os.path.basename(os.path.normpath(input_dir)))

        for file in input_files:
            scaled_file = scaled_filename(file, output_scale_ratio)
            if scaled_file in common_filenames:
                os.makedirs(os.path.join(output_dir_for_input,
                            os.path.dirname(file)), exist_ok=True)
                shutil.copy(os.path.join(input_dir, file),
                            os.path.join(output_dir_for_input, file))

        output_dir_for_output = os.path.join(
            self.output_directory,
            'normalized',
            os.path.basename(os.path.normpath(output_dir)))

        for file in output_files:
            if file in common_filenames:
                os.makedirs(os.path.join(output_dir_for_output,
                            os.path.dirname(file)), exist_ok=True)
                shutil.copy(os.path.join(output_dir, file),
                            os.path.join(output_dir_for_output, file))

    def postprocess_data(self, input_dir: str):

        for dirpath, _, filenames in os.walk(input_dir):

            if dirpath is input_dir:
                continue

            if len(os.listdir(dirpath)) < 10:
                continue

            output_dir = os.path.join(
                self.output_directory,
                'postprocessed',
                os.path.basename(input_dir),
                os.path.basename(dirpath)
            )
            os.makedirs(output_dir, exist_ok=True)
            for file in filenames:
                shutil.copy(os.path.join(dirpath, file),
                            os.path.join(output_dir, file))

    def prepare_data(self, output_scale_ratio: int):

        print('preprocessing...')
        for directory in os.listdir(self.input_directory):
            self.preprocess_data(os.path.join(
                self.input_directory, os.path.basename(os.path.normpath(directory))))

        def get_style_directories_for_category(category: str):
            styles_directories = os.path.join(
                self.output_directory,
                category
            )
            styles_directories = os.listdir(styles_directories)
            styles_directories.remove("original")
            return list(map(lambda x: os.path.join(
                self.output_directory,
                category,
                x
            ), styles_directories))

        print('normalizing...')
        directories = get_style_directories_for_category('preprocessed')
        for directory in directories:
            self.normalize_data(
                os.path.join(self.output_directory,
                             'preprocessed', 'original'),
                os.path.join(self.output_directory, 'preprocessed', os.path.basename(
                    os.path.normpath(directory))),
                output_scale_ratio
            )

        print('postprocessing...')
        for directory in os.listdir(os.path.join(self.output_directory, 'normalized')):
            self.postprocess_data(os.path.join(
                self.output_directory, 'normalized', os.path.basename(os.path.normpath(directory))))

        print('prepare for augmentation...')
        final_directory = os.path.join(self.output_directory, 'augmented')
        if os.path.exists(final_directory):
            shutil.rmtree(final_directory)

        shutil.copytree(os.path.join(self.output_directory,
                        'postprocessed'), final_directory)
