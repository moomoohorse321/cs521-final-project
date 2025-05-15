import iree.compiler.tf
import iree.runtime
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from iree.tf.support import module_utils
from iree import runtime as ireert
from iree.compiler import compile_str
from substitute import FuncSubstitute, get_approx_kernel
import os

from common import test_load


IMG_DIR = "../imgs/plot3/"

if __name__ == "__main__":
    # Declare and create the directory for saving images
    if not os.path.exists(IMG_DIR):
        os.makedirs(IMG_DIR)
    test_load(img_dir=IMG_DIR)