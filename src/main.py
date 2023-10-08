"""Main module"""
import os
import sys
from multiprocessing import freeze_support
from augmentator import Utils, Techniques
sys.path.append('cnn')
from cnn.tiny_pix2pix import CNN, DatasetLoader


def augment_datasets(output_scale_ratio: int):
    """Does the augmentation of a raw datasets."""
    utils = Utils(os.path.join("..", "data", "raw"),
                  os.path.join("..", "data", "output"))
    techniques = Techniques(
        os.path.join("..", "data", "output", "postprocessed"),
        os.path.join("..", "data", "output", "augmented"))

    utils.prepare_data(output_scale_ratio)
    utils.smart_augmentation(techniques, 100000)


if __name__ == '__main__':
    # add freeze support for multiprocessing
    freeze_support()

    #augment_datasets(2)
    loader = DatasetLoader(
        os.path.join("..", "data", "output",
                     "augmented", "original", "16x16"),
        os.path.join("..", "data", "output",
                     "augmented", "x32", "32x32"))

    network = CNN(loader)
    network.train()
    #network.test_model(os.path.join('..','models','16x16_b64','e19','generator_0.117144.pth'))
