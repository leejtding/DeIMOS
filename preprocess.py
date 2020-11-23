import numpy as np
import tensorflow as tf
import os

import sample_images

def get_data(img_dir, labeled=False):
    sample_images(img_dir)

get_data('data')