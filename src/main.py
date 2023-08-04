"""Main module"""
import os
from multiprocessing import freeze_support
from augmentator import Utils, Techniques

if __name__ == '__main__':
    # add freeze support for multiprocessing
    freeze_support()

    input_dir = os.path.join("..", "data", "raw")
    output_dir = os.path.join("..", "data", "augmented")

    Aug_Techniques = Techniques(input_dir, output_dir)
    Aug_Utils = Utils(input_dir, output_dir)

    Aug_Utils.equalize_matching_data('original', 'x32')
    Aug_Utils.clone()
    Aug_Utils.bulk_apply(Aug_Techniques.flip)
    Aug_Utils.bulk_apply(Aug_Techniques.mirror)
    Aug_Utils.bulk_apply(Aug_Techniques.rotate)
    Aug_Utils.bulk_apply(Aug_Techniques.color_jitter,
                         use_processed_images=True)
    Aug_Utils.bulk_apply(Aug_Techniques.invert,
                         use_processed_images=True)
