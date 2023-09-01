"""
This module provides utility classes and methods for image data augmentation and preprocessing.

Classes:
    - Techniques: A utility class for applying various image processing techniques to images.
    - Utils: Utility class for easier usage of an augmentator.
"""
import os
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Callable, List
import shutil
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm


class Techniques:
    """
    A utility class for applying various image processing techniques to images.

    Args:
        input_dir (str): The directory containing input images.
        output_dir (str): The directory where processed images will be saved.
    """

    def __init__(self, input_dir: str, output_dir: str):
        self.input_directory = Path(input_dir)
        self.output_directory = Path(output_dir)
        os.makedirs(os.path.dirname(output_dir), exist_ok=True)

    def __get_path_with_suffix(self, path: str, suffix: str):
        directory = os.path.dirname(path)
        directory = directory.replace('postprocessed', 'augmented')
        filename = Path(path).stem
        return os.path.join(
            Path(directory),
            Path(f"{filename}_{suffix}.png")
        )

    def __save_with_suffix(self, img: Image.Image, path: str, suffix: str):
        """Saves image with suffix. Returns the path under which file is saved."""
        output = self.__get_path_with_suffix(path, suffix)
        os.makedirs(os.path.dirname(output), exist_ok=True)
        img.convert('RGBA').save(output)
        return output

    def rotate(self, image_path: str):
        """
        Rotate the image by 90, 180, and 270 degrees and save the rotated versions.

        Args:
            image_path (str): The path to the input image.
        """
        degrees = [90, 180, 270]
        for rotation in degrees:
            img = Image.open(image_path)
            img = img.rotate(
                rotation, resample=Image.Resampling.NEAREST, expand=True)
            self.__save_with_suffix(img, image_path, f'rotated_{rotation}')
            img.close()

    def mirror(self, image_path: str):
        """
        Mirror the image horizontally and save the mirrored version.

        Args:
            image_path (str): The path to the input image.
        """
        img = Image.open(image_path)
        img = ImageOps.mirror(img)
        self.__save_with_suffix(img, image_path, 'mirrored')
        img.close()

    def flip(self, image_path: str):
        """
        Flip the image vertically and save the flipped version.

        Args:
            image_path (str): The path to the input image.
        """
        img = Image.open(image_path)
        img = ImageOps.flip(img)
        self.__save_with_suffix(img, image_path, 'flipped')
        img.close()

    def invert(self, image_path: str):
        """
        Invert the colors of the image and save the inverted version.

        Args:
            image_path (str): The path to the input image.
        """
        img = Image.open(image_path)

        red, green, blue, alpha = img.split()

        red = ImageOps.invert(red)
        green = ImageOps.invert(green)
        blue = ImageOps.invert(blue)

        img = Image.merge('RGBA', (red, green, blue, alpha))

        self.__save_with_suffix(img, image_path, 'inverted')
        img.close()

    def __jittering(self, image_path: str, hue: float, sat: float, val: float):
        img = Image.open(image_path)

        alpha = img.getchannel('A')
        hsv_image = np.array(img.convert('HSV'))
        hsv_image[:, :, 0] = np.clip(hsv_image[:, :, 0] * hue, 0, 255)
        hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * sat, 0, 255)
        hsv_image[:, :, 2] = np.clip(hsv_image[:, :, 2] * val, 0, 255)
        img = Image.fromarray(hsv_image, 'HSV').convert('RGBA')
        img.putalpha(alpha)

        self.__save_with_suffix(
            img,
            image_path,
            f'_hue{hue}_saturation{sat}_value{val}')
        img.close()

    def color_jitter(self, image_path: str, values_for_variations=None):
        """
        Apply color jittering variations to an image using specified adjustment values.

        Args:
            image_path (str): The path to the input image.
            values_for_variations (list or None, optional): A list of adjustment values 
                for hue, saturation, and value. If None, default values [0.5, 1.5, 2] are used.

        Note: 
            1.0 means 100%, so the same value. It's about scaling: 0.75 = 75% of default, etc.
        """
        if values_for_variations is None:
            values = np.array([0.5, 1.5, 2])
        else:
            values = np.array(values_for_variations)

        combinations = np.array(list(np.unique(np.array(np.meshgrid(
            values, values, values)).T.reshape(-1, 3), axis=0)))
        for combination in combinations:
            np.vectorize(self.__jittering)(
                image_path, combination[0], combination[1], combination[2])


class Utils:
    """
    Utility class for easier usage of an augmentator.

    Args:
        input_dir (str): The directory containing input images.
        output_dir (str): The directory where processed images will be saved.
    """

    def __init__(self, input_dir: str, output_dir: str):
        self.input_directory = Path(input_dir)
        self.output_directory = Path(output_dir)
        os.makedirs(self.output_directory, exist_ok=True)

    def _batch_process(self, data: tuple[Callable, List[str], tuple]) -> None:
        function, file_batch, *args = data
        for file in file_batch:
            function(file, *args)

    def smart_augmentation(self, techniques: Techniques, goal: int):
        """
        Apply smart data augmentation to images using a combination
        of techniques to reach a specified goal.

        Args:
            techniques (Techniques): An instance of the Techniques class
            containing image processing methods.
            goal (int): The target number of augmented images to generate.
        """

        def get_deepest_directories(root_dir):
            max_depth = 0
            deepest_dirs = []

            for root, _, _ in os.walk(root_dir):
                depth = str(root).count(os.sep) - str(root_dir).count(os.sep)

                if depth > max_depth:
                    max_depth = depth
                    deepest_dirs = [os.path.abspath(root)]
                elif depth == max_depth:
                    deepest_dirs.append(os.path.abspath(root))

            return deepest_dirs

        def get_files(input_path: str):
            files = []
            for (dirpath, _, filenames) in os.walk(input_path):
                files += [os.path.join(dirpath, file)
                          for file in filenames]
            return files

        def apply_method(function: Callable, files: List[str], *args):
            batch_size = max(
                1, int(len(files) / (np.log(len(files)) * cpu_count())))

            file_batches = [files[i:i + batch_size]
                            for i in range(0, len(files), batch_size)]

            with Pool() as pool, tqdm(
                    total=len(file_batches),
                    desc=function.__name__,
                    unit="batch") as pbar:
                for _ in pool.imap_unordered(self._batch_process,
                                             [(function, file_batch, *args
                                               ) for file_batch in file_batches]):
                    pbar.update(1)

        total_directories = len(
            get_deepest_directories(techniques.input_directory))

        for i, (input_dir, output_dir) in enumerate(zip(
                get_deepest_directories(techniques.input_directory),
                get_deepest_directories(techniques.output_directory))):
            print(
                f'Augmenting {os.path.basename(input_dir)} ({i}/{total_directories})')

            files = get_files(input_dir)
            if len(files) * 2 < goal:
                apply_method(techniques.mirror, files)
            else:
                continue

            if len(get_files(output_dir)) * 2 < goal:
                apply_method(techniques.flip, files)
            else:
                continue

            if len(get_files(output_dir)) * 3 < goal:
                apply_method(techniques.rotate, files)
            else:
                continue

            # (x + x * (1 + 1 + 3)) * y^3 = goal
            exponent = int(np.cbrt(goal / (6 * len(files))))
            array_for_jittering = list(
                np.around(np.linspace(0.5, 2, exponent), 2))

            files = get_files(output_dir)
            if len(files) * (exponent ** 3) < goal:
                apply_method(techniques.color_jitter, files, array_for_jittering)
            else:
                continue

            files = get_files(output_dir)
            if len(files) * 2 < goal:
                apply_method(techniques.invert, files)

    def preprocess_data(self, input_dir: str):
        """
        Preprocess and sort images in the specified input directory.

        Args:
            input_dir (str): The directory containing input images to preprocess.
        """
        for root, _, files in tqdm(
                os.walk(input_dir),
                desc=f"sort ({os.path.basename(os.path.normpath(input_dir))})",
                unit="dir"):
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
        """
        Normalize input and output images to a specified scale ratio while preserving common files.

        Args:
            input_dir (str): The directory containing input images.
            output_dir (str): The directory containing output images.
            output_scale_ratio (int): The scale ratio for normalization.

        This method scales input images to match a specified ratio and preserves 
        common files found in both input and output directories. Scaled input images 
        are placed in corresponding subdirectories within the 'normalized' folder
        of the output directory.

        Note:
            - The `output_scale_ratio` determines the scale for normalization.
        """
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
        """
        Postprocess and organize images in the specified input directory.

        Args:
            input_dir (str): The directory containing input images to be postprocessed.

        This method performs postprocessing on images found in the specified 
        input directory, including organizing them into subdirectories within the 
        'postprocessed' directory of the output directory.
        
        Note:
            Images with fewer than 10 files in a directory are skipped during postprocessing.
        """

        for dirpath, _, filenames in os.walk(input_dir):

            if dirpath is input_dir:
                continue

            height, width = os.path.basename(dirpath).split('x')
            if height != width:
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
        """
        Prepare data for augmentation, including preprocessing, normalization, and postprocessing.

        Args:
            output_scale_ratio (int): The scale ratio for image normalization.

        This method automates the preparation of data for augmentation. 
        It performs the following steps:
        1. Preprocesses input images by converting them to RGBA format.
        2. Normalizes images to a specified scale ratio while preserving common files.
        3. Postprocesses images, organizing them into subdirectories.
        4. Prepares data for augmentation by creating a clean 'augmented' directory.

        Note:
            The `output_scale_ratio` parameter determines the scale for image normalization.
        """
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
