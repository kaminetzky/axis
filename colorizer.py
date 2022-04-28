# TODO: replace ratios better names, such as organic_inorganic_threshold
# TODO: add fallback in case there's no internet connection
# TODO: check exceptions when the xcom scraper fails to fetch info
# TODO: Optimize colorizer. Stop doing it pixelwise

import cv2
import numpy as np

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

  def colorize(self, img_low, img_high):
    img_color = np.zeros((*img_low.shape, 3), dtype=np.float32)

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
            hue = 30 # Orange
          elif ratio < self.ratios[1]:
            # Inorganic
            hue = 120 # Green
          else:
            # Metals
            hue = 220 # Blue
          value = (hue, sat, val)

        img_color[j, i] = value

    img_color = cv2.cvtColor(img_color, cv2.COLOR_HSV2RGB)
    return img_color

  def generate_spectrum(self, size=256):
    img_low = np.tile(np.linspace(0, 1, size, dtype=np.float32), (size, 1))
    img_high = img_low.T
    return self.colorize(img_low, img_high)
