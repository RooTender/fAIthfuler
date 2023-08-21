"""Main module"""
import os
from multiprocessing import freeze_support
from texture_tools import ArchiveHandler
from augmentator import Utils, Techniques
from cnn import CNN


def unpack_textures():
    """Unpacks the archives placed in 'data/to_unpack' folder to 'data/raw' folder."""
    archieve_handler = ArchiveHandler()
    input_path = os.path.join('..', 'data', 'to_unpack')
    for archieve in os.listdir(input_path):
        archieve_handler.extract_textures(os.path.join(input_path, archieve))


def augment_datasets(output_scale_ratio: int):
    """Does the augmentation of a raw datasets."""
    utils = Utils(os.path.join("..", "data", "raw"),
                  os.path.join("..", "data", "output"))
    techniques = Techniques(
        os.path.join("..", "data", "output", "postprocessed"),
        os.path.join("..", "data", "output", "augmented"))

    utils.prepare_data(output_scale_ratio)
    utils.augment_data(techniques, techniques.flip)
    utils.augment_data(techniques, techniques.mirror)
    utils.augment_data(techniques, techniques.rotate)
    utils.augment_data(techniques, techniques.color_jitter,
                       use_processed_images=True)
    utils.augment_data(techniques, techniques.invert,
                       use_processed_images=True)


if __name__ == '__main__':
    # add freeze support for multiprocessing
    freeze_support()

    # augment_datasets(2)

    test = CNN()
    test.run(
        os.path.join("..", "data", "output", "augmented", "original", "8x8"),
        os.path.join("..", "data", "output", "augmented", "x32", "16x16"))
