import os
from tqdm import tqdm

import file
import generator
import overlay


DIR_BGND = 'img/color/bags'
EXTENSION_BGND = 'jpg'

DIR_FGND_LOW = 'img/dual/wrench/wrench-50'
DIR_FGND_HIGH = 'img/dual/wrench/wrench-150'
EXTENSION_FGND = 'tiff'

DIR_OUT = 'data'
EXTENSION_OUT = 'png'

LOW_ENERGY = 50
HIGH_ENERGY = 150

SCALE_MIN = 0.1
SCALE_MAX = 0.2

FGND_QTY_PROB = {0: 0.1, 1: 0.5, 2: 0.3, 3: 0.1}

TRAIN_VAL_RATIO = {'train': 0.8, 'valid': 0.1}


if __name__ == '__main__':
  generator.generate_simulated_images(
    DIR_BGND, EXTENSION_BGND, DIR_FGND_LOW, DIR_FGND_HIGH, EXTENSION_FGND,
    DIR_OUT, EXTENSION_OUT, LOW_ENERGY, HIGH_ENERGY, SCALE_MIN, SCALE_MAX,
    FGND_QTY_PROB, TRAIN_VAL_RATIO
  )