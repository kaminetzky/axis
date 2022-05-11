import cv2
from functools import cache
import matplotlib.pyplot as plt
import numpy as np
import os
import random
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


@cache
def get_filenames(dir, extension):
  filenames = [x for x in os.listdir(dir) if x.endswith(f'.{extension}')]
  random.shuffle(filenames)
  return filenames


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


def save_data(filename_with_extension, img_extension, dir_output, img,
              bounding_boxes, category):
    filename = os.path.splitext(filename_with_extension)[0]
    img_path = os.path.join(dir_output, 'images', category,
                            f'{filename}.{img_extension}')
    plt.imsave(img_path, img, cmap='gray')

    bbox_path = os.path.join(dir_output, 'labels', category,
                             filename + '.txt')

    bbox_text = ''
    for bbox in bounding_boxes:
      obj_cat = 0
      center_y = bbox['pos_y'] + bbox['size_y'] / 2
      center_x = bbox['pos_x'] + bbox['size_x'] / 2
      center_y_rel = center_y / img.shape[0]
      center_x_rel = center_x / img.shape[1]
      size_y_rel = bbox['size_y'] / img.shape[0]
      size_x_rel = bbox['size_x'] / img.shape[1]
      bbox_text += (f'{obj_cat} {center_x_rel} {center_y_rel} {size_x_rel} '
                    f'{size_y_rel}\n')
    with open(bbox_path, 'w') as file:
      file.write(bbox_text)
