"""Main module"""
import os
import sys
from multiprocessing import freeze_support
from texture_tools import ArchiveHandler
from augmentator import Utils, Techniques

sys.path.append('cnn')
from cnn.pix2pix import CNN, DatasetLoader



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
    utils.smart_augmentation(techniques, 10000)


if __name__ == '__main__':
    # add freeze support for multiprocessing
    freeze_support()

    # augment_datasets(2)
    loader = DatasetLoader(
        os.path.join("..", "data", "output",
                     "augmented", "original", "8x8"),
        os.path.join("..", "data", "output",
                     "augmented", "x32", "16x16"))
    network = CNN(loader)
    network.train()
    #network.test_model(os.path.join('..','models','8x8_b16','e285','generator_1.803540.pth'))
