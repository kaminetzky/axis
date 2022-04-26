import cv2
from cv2 import IMWRITE_PNG_COMPRESSION
import numpy as np
from scipy import ndimage


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


def zero_pad_image(img, size, pos):
  if len(img.shape) == 2:
    img_padded = np.zeros(size, dtype=np.uint8)
  else:
    img_padded = np.zeros((*size, 3), dtype=np.uint8)

  img, size, pos = crop_overflowing_image(img, size, pos)

  if img.shape[0] == 0 or img.shape[1] == 0:
    # Foreground image is completely out of frame
    return img_padded

  img_padded[pos[0]:pos[0]+img.shape[0], pos[1]:pos[1]+img.shape[1]] = img

  return img_padded


def crop_overflowing_image(img, size, pos):
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


def scale_relative(fgnd, bgnd, scale_min, scale_max):
  height_ratio = fgnd.shape[0] / bgnd.shape[0]
  width_ratio = fgnd.shape[1] / bgnd.shape[1]
  scale = (np.random.uniform(scale_min, scale_max)
           / max(height_ratio, width_ratio))
  width = int(fgnd.shape[0] * scale)
  height = int(fgnd.shape[1] * scale)

  fgnd_scaled = cv2.resize(fgnd, (width, height), interpolation=cv2.INTER_AREA)

  return fgnd_scaled


def rotate_random(img):
  img_rot = ndimage.rotate(img, np.random.randint(0, 359), order=1,
                           reshape=True, mode='constant', cval=255)
  return img_rot


def crop_white_borders(img, threshold=250, offset=2):
  dark_coords = np.where(img < threshold)

  y_start = np.min(dark_coords[0]) - offset
  y_end = np.max(dark_coords[0]) + offset
  x_start = np.min(dark_coords[1]) - offset
  x_end = np.max(dark_coords[1]) + offset

  img_cropped = img[y_start:y_end, x_start:x_end]
  return img_cropped