import cv2
import numpy as np
import os
import shutil
from tqdm import tqdm

from colorizer import Colorizer


def load_color_backgrounds(dir, extension):
  # TODO: make sure images are returned as float32 between 0 and 1

  filenames = get_filenames(dir, extension)

  for filename in filenames:
    img_bgr = cv2.imread(os.path.join(dir, filename), cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    yield img_rgb

def get_filenames(dir, extension):
  return sorted([x for x in os.listdir(dir) if x.endswith(f'.{extension}')])

def load_colorized_foregrounds(dir_low, dir_high, extension, low_energy,
                               high_energy):
  # TODO: test with jpg and png
  # TODO: make sure images are returned as float32 between 0 and 1

  filenames_low = [x for x in os.listdir(dir_low)
                   if x.endswith(f'.{extension}')]

  imgs = []
  for filename in tqdm(filenames_low):
    img_low = cv2.imread(os.path.join(dir_low, filename),
                         cv2.IMREAD_UNCHANGED)
    img_high = cv2.imread(os.path.join(dir_high, filename),
                         cv2.IMREAD_UNCHANGED)

    colorizer = Colorizer(low_energy, high_energy)
    img_color = colorizer.colorize(img_low, img_high)
    imgs.append((img_color * 255).astype(np.uint8))

  return imgs


def delete_and_create_yolo_dirs(data_dir):
  for category in ['train', 'valid', 'test']:
    delete_and_create_dir(os.path.join(data_dir, 'images', category))
    delete_and_create_dir(os.path.join(data_dir, 'labels', category))


def delete_and_create_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)