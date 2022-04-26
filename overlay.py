import cv2
import numpy as np

import image


def overlay_grayscale(bgnd, fgnd, pos, b=0, c=255):
  # source: https://github.com/computervision-xray-testing/pyxvis
  # bgnd (background) and fgnd (foreground) must be numpy ndarrays with uint8
  # values between 0 and 255
  # pos is (y, x) with respect to the top-left corner
  # b and c are calibration parameters

  bgnd_y_start = max(0, pos[0])
  bgnd_y_end = min(max(0, pos[0] + fgnd.shape[0]), bgnd.shape[0])
  bgnd_x_start = max(0, pos[1])
  bgnd_x_end = min(max(0, pos[1] + fgnd.shape[1]), bgnd.shape[1])
  bgnd_cropped = (bgnd[bgnd_y_start:bgnd_y_end, bgnd_x_start:bgnd_x_end] - b) / c

  fgnd_y_start = max(-pos[0], 0)
  fgnd_y_end = min(max(0, bgnd.shape[0] - pos[0]), fgnd.shape[0])
  fgnd_x_start = max(-pos[1], 0)
  fgnd_x_end = min(max(0, bgnd.shape[1] - pos[1]), fgnd.shape[1])
  fgnd_cropped = (fgnd[fgnd_y_start:fgnd_y_end, fgnd_x_start:fgnd_x_end] - b) / c

  result = bgnd.copy()
  result[bgnd_y_start:bgnd_y_end, bgnd_x_start:bgnd_x_end] = (
    c * fgnd_cropped * bgnd_cropped + b
  )

  return result


def overlay_color(bgnd, fgnd, pos, alpha=0.9):
  # source: http://arxiv.org/abs/1909.11508
  # pos is the top left corner
  # TODO: modify code so that bag mask is calculated only once
  bgnd_gray = image.rgb_to_gray(bgnd)
  bgnd_mask = calculate_bag_mask(bgnd)
  threat_threshold = calculate_threat_threshold(bgnd_gray, bgnd_mask)

  bgnd_size = bgnd.shape[:2]

  fgnd_mask = image.binarize(fgnd, threat_threshold)
  fgnd_mask_padded = image.zero_pad_image(fgnd_mask, bgnd_size, pos)
  fgnd_mask_padded_inv = 255 - fgnd_mask_padded

  fgnd_padded = image.zero_pad_image(fgnd, bgnd_size, pos)
  fgnd_padded_masked = cv2.bitwise_and(fgnd_padded, fgnd_padded, mask=fgnd_mask_padded)

  bgnd_masked = cv2.bitwise_and(bgnd, bgnd, mask=fgnd_mask_padded_inv)
  bgnd_masked_inv = cv2.bitwise_and(bgnd, bgnd, mask=fgnd_mask_padded)

  bgnd_fgnd_merged = (alpha * fgnd_padded_masked
                  + (1 - alpha) * bgnd_masked_inv).astype(np.uint8)

  return bgnd_masked + bgnd_fgnd_merged


def calculate_threat_threshold(target, mask):
  region_intensity = calculate_normal_average_intensity(target, mask)
  threshold = int(round(255 * min(np.exp(region_intensity**5) - 0.5, 0.95)))
  return threshold


def calculate_normal_average_intensity(target, mask):
  target_masked = cv2.bitwise_and(target, target, mask=mask)
  target_masked_sum = np.sum(target_masked)
  mask_sum = np.sum(mask)
  return target_masked_sum / mask_sum


def calculate_bag_mask(img_rgb, bin_thresh=240, kernel_size=5):
  img = image.rgb_to_gray(img_rgb)
  img = image.binarize(img, bin_thresh)
  img = image.dilate(img, kernel_size)
  img = image.fill_holes(img)
  img = image.erode(img, kernel_size)
  img = image.get_largest_region(img)
  return img


def generate_insertion_pos(bgnd, fgnd, max_iterations=1000):
  # TODO: modify code so that bag mask is calculated only once
  bgnd_mask = calculate_bag_mask(bgnd)
  fgnd_size = fgnd.shape[:2]

  for _ in range(max_iterations):
    pos = (np.random.randint(0, bgnd_mask.shape[0] - fgnd_size[0]),
           np.random.randint(0, bgnd_mask.shape[1] - fgnd_size[1]))
    if is_valid_insertion_position(bgnd_mask, fgnd_size, pos):
      return pos

  raise RuntimeError('Couldn\'t find an insertion position')


def is_valid_insertion_position(bgnd_mask, fgnd_size, pos):
  source_mask = np.zeros(bgnd_mask.shape, dtype=np.uint8)
  source_mask[pos[0]:pos[0]+fgnd_size[0], pos[1]:pos[1]+fgnd_size[1]] = 255
  colissions = np.logical_and(source_mask, np.logical_not(bgnd_mask))
  all_zeros = not colissions.any()
  return all_zeros
