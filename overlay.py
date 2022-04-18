import numpy as np


def overlay_grayscale(bg, fg, pos, b=0, c=255):
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
