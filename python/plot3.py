import os

from common import test_load


IMG_DIR = "../imgs/plot3/"

if __name__ == "__main__":
    # Declare and create the directory for saving images
    if not os.path.exists(IMG_DIR):
        os.makedirs(IMG_DIR)
    test_load(img_dir=IMG_DIR)
