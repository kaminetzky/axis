import cv2
import numpy as np


def overlay_grayscale(bg, fg, pos, b=0, c=255):
  # source: https://github.com/computervision-xray-testing/pyxvis
  # bg (background) and fg (foreground) must be numpy ndarrays with uint8
  # values between 0 and 255
  # pos is (y, x) with respect to the top-left corner
  # b and c are calibration parameters

  bg_y_start = max(0, pos[0])
  bg_y_end = min(max(0, pos[0] + fg.shape[0]), bg.shape[0])
  bg_x_start = max(0, pos[1])
  bg_x_end = min(max(0, pos[1] + fg.shape[1]), bg.shape[1])
  bg_cropped = (bg[bg_y_start:bg_y_end, bg_x_start:bg_x_end] - b) / c

  fg_y_start = max(-pos[0], 0)
  fg_y_end = min(bg.shape[0] - pos[0], fg.shape[0])
  fg_x_start = max(-pos[1], 0)
  fg_x_end = min(bg.shape[1] - pos[1], fg.shape[1])
  fg_cropped = (fg[fg_y_start:fg_y_end, fg_x_start:fg_x_end] - b) / c

  result = bg.copy()
  result[bg_y_start:bg_y_end, bg_x_start:bg_x_end] = (
    c * fg_cropped * bg_cropped + b
  )

  return result


def overlay_color(bg, fg, pos, alpha=0.9):
  # source: http://arxiv.org/abs/1909.11508
  # pos is the top left corner
  bg_gray = rgb_to_gray(bg)
  bg_mask = calculate_bag_mask(bg)
  threat_threshold = calculate_threat_threshold(bg_gray, bg_mask)

  bg_size = bg.shape[:2]

  fg_mask = binarize(fg, threat_threshold)
  fg_mask_padded = zero_pad_image(fg_mask, bg_size, pos)
  fg_mask_padded_inv = 255 - fg_mask_padded

  fg_padded = zero_pad_image(fg, bg_size, pos)
  fg_padded_masked = cv2.bitwise_and(fg_padded, fg_padded, mask=fg_mask_padded)

  bg_masked = cv2.bitwise_and(bg, bg, mask=fg_mask_padded_inv)
  bg_masked_inv = cv2.bitwise_and(bg, bg, mask=fg_mask_padded)

  bg_fg_merged = (alpha * fg_padded_masked
                  + (1 - alpha) * bg_masked_inv).astype(np.uint8)

  return bg_masked + bg_fg_merged


def calculate_threat_threshold(target, mask):
  region_intensity = calculate_normal_average_intensity(target, mask)
  threshold = int(round(255 * min(np.exp(region_intensity**5) - 0.5, 0.95)))
  return threshold


def calculate_normal_average_intensity(target, mask):
  target_masked = cv2.bitwise_and(target, target, mask=mask)
  target_masked_sum = np.sum(target_masked)
  mask_sum = np.sum(mask)
  return target_masked_sum / mask_sum


def zero_pad_image(img, size, pos):
  if len(img.shape) == 2:
    img_padded = np.zeros(size, dtype=np.uint8)
  else:
    img_padded = np.zeros((*size, 3), dtype=np.uint8)

  img, size, pos = crop_excess_image(img, size, pos)

  if img.shape[0] == 0 or img.shape[1] == 0:
    # Foreground image is completely out of frame
    return img_padded

  img_padded[pos[0]:pos[0]+img.shape[0], pos[1]:pos[1]+img.shape[1]] = img

  return img_padded


def crop_excess_image(img, size, pos):
  if pos[0] < 0:
    img = img[-pos[0]:, :]
    pos = (0, pos[1])
  if pos[1] < 0:
    img = img[:, -pos[1]:]
    pos = (pos[0], 0)
  if pos[0] + img.shape[0] > size[0]:
    img = img[:size[0] - img.shape[0] - pos[0], :]
  if pos[1] + img.shape[1] > size[1]:
    img = img[:, :size[1] - img.shape[1] - pos[1]]
  
  return img, size, pos


def calculate_bag_mask(img_rgb):
  img = rgb_to_gray(img_rgb)
  img = binarize(img, 240)
  img = dilate(img, 5)
  img = fill_holes(img)
  img = erode(img, 5)
  img = get_largest_region(img)
  return img


def rgb_to_gray(img):
  return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def binarize(img, threshold):
  if len(img.shape) == 3:
    img = rgb_to_gray(img)
  return np.where(img < threshold, 255, 0).astype(np.uint8)


def dilate(img, kernel_size):
  kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
  return cv2.dilate(img, kernel, iterations=1)


def fill_holes(img):
  img_filled = np.copy(img)
  contours, _ = cv2.findContours(img_filled, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
  for contour in contours:
      cv2.drawContours(img_filled, [contour], 0, 255, -1)
  return img_filled


def erode(img, kernel_size):
  kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
  return cv2.erode(img, kernel, iterations=1)


def get_largest_region(img):
  contours, _ = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
  largest_contour = sorted(contours, key=cv2.contourArea, reverse=True)[0]
  img_largest_region = np.zeros(img.shape, dtype=np.uint8)
  cv2.drawContours(img_largest_region, [largest_contour], 0, 255, -1)
  return img_largest_region