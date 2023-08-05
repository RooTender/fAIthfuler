"""Main module"""
import os
from multiprocessing import freeze_support
from texture_tools import ArchiveHandler
from augmentator import Utils, Techniques

def unpack_textures():
    """Unpacks the archives placed in 'data/to_unpack' folder to 'data/raw' folder."""
    archieve_handler = ArchiveHandler()
    input_path = os.path.join('..', 'data', 'to_unpack')
    for archieve in os.listdir(input_path):
        archieve_handler.extract_textures(os.path.join(input_path, archieve))

def augment_datasets():
    """Does the augmentation of a raw datasets."""
    input_dir = os.path.join("..", "data", "raw")
    output_dir = os.path.join("..", "data", "augmented")

    techniques = Techniques(input_dir, output_dir)
    utils = Utils(input_dir, output_dir)

    utils.equalize_matching_data('original', 'x32')
    utils.clone()
    utils.bulk_apply(techniques.flip)
    utils.bulk_apply(techniques.mirror)
    utils.bulk_apply(techniques.rotate)
    utils.bulk_apply(techniques.color_jitter,
                        use_processed_images=True)
    utils.bulk_apply(techniques.invert,
                        use_processed_images=True)

if __name__ == '__main__':
    # add freeze support for multiprocessing
    freeze_support()
