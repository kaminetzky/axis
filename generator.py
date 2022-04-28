import numpy as np
import random
from tqdm import tqdm

import file
import overlay


def sample_images_with_qty_prob(imgs, qty_prob):
  qty = np.random.choice(list(qty_prob.keys()), p=list(qty_prob.values()))
  sampled_imgs = random.sample(imgs, qty)
  return sampled_imgs


def get_category(index, total, train_val_ratio):
  if index < total * train_val_ratio['train']:
      category = 'train'
  elif (index < total
        * (train_val_ratio['train']
            + train_val_ratio['valid'])):
      category = 'valid'
  else:
      category = 'test'
  return category


def generate_simulated_images(
    dir_bgnd, extension_bgnd, dir_fgnd_low, dir_fgnd_high, extension_fgnd,
    dir_out, extension_out, low_energy, high_energy, scale_min, scale_max,
    fgnd_qty_prob, train_val_ratio):
  file.delete_and_create_yolo_dirs(dir_out)

  bgnds = file.load_color_backgrounds(dir_bgnd, extension_bgnd)
  fgnds = file.load_colorized_foregrounds(dir_fgnd_low, dir_fgnd_high,
                                                extension_fgnd, low_energy,
                                                high_energy)


  bgnd_filenames = file.get_filenames(dir_bgnd, extension_bgnd)
  bgnd_qty = len(bgnd_filenames)

  for index, bgnd in enumerate(tqdm(bgnds, total=bgnd_qty)):
    overlaid, bboxes = overlay.overlay_fgnds_over_bgnd(
      bgnd, fgnds, fgnd_qty_prob, scale_min, scale_max)

    category = get_category(index, bgnd_qty, train_val_ratio)

    img_filename = bgnd_filenames[index]
    file.save_data(img_filename, extension_out, dir_out, overlaid, bboxes,
                    category)
          
