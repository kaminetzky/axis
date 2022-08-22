# TODO: replace ratios better names, such as organic_inorganic_threshold
# TODO: add fallback in case there's no internet connection
# TODO: check exceptions when the xcom scraper fails to fetch info
# TODO: Optimize colorizer. Stop doing it pixelwise

import cv2
import numpy as np
from perlin_numpy import generate_perlin_noise_2d
import random
import skimage

import xcom


class Colorizer:
  def __init__(self, low_energy=None, high_energy=None, ratios=None):
    # TODO: Add checker that either energies or ratios are defined
    # TODO: Then code the logic.

    if low_energy is None and high_energy is None and ratios is None:
      raise TypeError('Energies or ratios should be passed')
    if ratios is None and (low_energy is None or high_energy is None):
      raise TypeError('Both energies should be passed')

    self._low_energy = low_energy
    self._high_energy = high_energy

    if ratios:
      self.ratios = ratios
    else:
      self._ratios = xcom.get_transition_ratios(low_energy, high_energy)

  @property
  def low_energy(self):
    return self._low_energy

  @low_energy.setter
  def low_energy(self, value):
    self._low_energy = value
    self.ratios = xcom.get_transition_ratios(self.low_energy, self.high_energy)

  @property
  def high_energy(self):
    return self._high_energy

  @high_energy.setter
  def high_energy(self, value):
    self._high_energy = value
    self.ratios = xcom.get_transition_ratios(self.low_energy, self.high_energy)

  @property
  def ratios(self):
    return self._ratios

  @ratios.setter
  def ratios(self, value):
    # Manually setting the ratios means that they won't necessarily match
    # the real ratios corresponding to this instance's energies
    self._ratios = value

  def colorize(self, img_low, img_high, organic_hue=30, inorganic_hue=120,
               metal_hue=205):
    img_color = np.zeros((*img_low.shape, 3), dtype=np.float32)
    metal_hue = random.randint(200, 240)

    for j in range(img_color.shape[0]):
      for i in range(img_color.shape[1]):
        low_pixel = img_low[j, i]
        high_pixel = img_high[j, i]
        if low_pixel == 0:
          # Prevents calculating the log of zero
          value = (0, 0, 0)
        elif high_pixel == 0:
          # Prevents calculating the log of zero
          value = (0, 0, 0)
        elif high_pixel == 1:
          # Prevents division by zero
          value = (0, 0, 1)
        else:
          mean_pixel = (high_pixel + low_pixel) / 2
          ratio = np.log(low_pixel) / np.log(high_pixel)

          sat = 1 - mean_pixel ** 3
          val = 1 - (1 - mean_pixel) ** 3

          # TODO: define the first condition in a better way.
          # In theory, some plastic's ratio can be lower than air's.
          # Air's ratio is 1.519, while PE's is 1.358.
          # Maybe take into account mean_pixel value and ratio.
          if ratio < 1.3:
            # Air
            hue = 0
            sat = 0
            val = 1
          elif ratio < self.ratios[0]:
            # Organic
            hue = organic_hue # Orange
          elif ratio < self.ratios[1]:
            # Inorganic
            hue = inorganic_hue # Green
          else:
            # Metal
            hue = metal_hue # Blue
          value = (hue, sat, val)

        img_color[j, i] = value

    img_color = cv2.cvtColor(img_color, cv2.COLOR_HSV2RGB)
    return img_color

  def add_spots(self, img_color):
    # Tidy up this
    img_hsv = cv2.cvtColor(img_color, cv2.COLOR_RGB2HSV)

    location_blobs = generate_perlin_noise_2d((400, 400), (4, 4)) * 0.5 + 0.5
    dark_mask = (generate_perlin_noise_2d((400, 400), (4, 4)) * 0.5 + 0.5) > 0.7
    big_blobs = generate_perlin_noise_2d((400, 400), (8, 8)) * 0.5 + 0.5
    small_blobs = generate_perlin_noise_2d((400, 400), (16, 16)) * 0.5 + 0.5
    blobs_mask = ((big_blobs > 0.7) + (small_blobs > 0.7)) * (location_blobs > 0.5)
    
    img_hsv_hue = img_hsv[:, :, 0]
    img_hsv_sat = img_hsv[:, :, 1]
    img_hsv_val = img_hsv[:, :, 2]

    object_mask = img_hsv_val < 0.99

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    blobs_outline = cv2.morphologyEx(blobs_mask.astype(np.uint8), cv2.MORPH_GRADIENT, kernel)
    blobs_outline_in_object = blobs_outline * object_mask

    img_hsv_hue[blobs_mask & object_mask] = random.randint(90, 120)
    img_hsv_val[blobs_mask & object_mask] *= 0.7
    img_hsv_val[blobs_mask & object_mask & dark_mask] *= 0.5

    img_color = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
    img_color = skimage.filters.gaussian(img_color, sigma=1)


    img_hsv = cv2.cvtColor(img_color, cv2.COLOR_RGB2HSV)
    img_hsv_sat_new = img_hsv[:, :, 1]
    img_hsv_val_new = img_hsv[:, :, 2]

    img_hsv_sat_new = np.where(blobs_outline_in_object, img_hsv_sat_new, img_hsv_sat)
    img_hsv_val_new = np.where(blobs_outline_in_object, img_hsv_val_new, img_hsv_val)

    img_hsv[:, :, 1] = img_hsv_sat_new
    img_hsv[:, :, 2] = img_hsv_val_new
    
    img_color = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)

    return img_color


  def add_hue_variations(self, img_color):
    # Tidy up this, too
    noise_periods = random.choice([5, 8, 10, 16, 20])
    noise_multiplier = random.uniform(5, 15)
    noise = generate_perlin_noise_2d((400, 400), (noise_periods, noise_periods))
    img_hsv = cv2.cvtColor(img_color, cv2.COLOR_RGB2HSV)
    img_hue = img_hsv[:, :, 0]
    img_hue = skimage.filters.gaussian(img_hue, sigma=0.5)
    img_hue += noise * noise_multiplier
    img_hsv[:, :, 0] = img_hue
    img_color = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
    return img_color


  def generate_spectrum(self, size=256):
    img_low = np.tile(np.linspace(0, 1, size, dtype=np.float32), (size, 1))
    img_high = img_low.T
    return self.colorize(img_low, img_high)
