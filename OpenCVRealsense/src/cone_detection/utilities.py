import os
import time


def current_milli_time():
    return int(round(time.time() * 1000))


def create_dirs(dirpath):
    img_output_dir = dirpath
    if not os.path.exists(img_output_dir):
        os.makedirs(img_output_dir)
