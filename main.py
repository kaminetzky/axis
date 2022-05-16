import numpy as np
import random

import generator


DIR_BGND = '/Users/ak/Documents/University/Masters/Research/Data/datasets/SIXray/SIXrayN10'
EXTENSION_BGND = 'jpg'

DIR_FGND_LOW = 'data/input/wrench-experiments2-flat/dual/wrench-100-test'
DIR_FGND_HIGH = 'data/input/wrench-experiments2-flat/dual/wrench-150-test'
EXTENSION_FGND = 'tiff'

DIR_OUT = 'data/output/wrench-sim-only-test'
EXTENSION_OUT = 'jpg'

LOW_ENERGY = 100
HIGH_ENERGY = 150

SCALE_MIN = 0.3
SCALE_MAX = 0.6

FGND_QTY_PROB = {0: 0.1, 1: 0.5, 2: 0.3, 3: 0.1}

TRAIN_VAL_RATIO = {'train': 0.9, 'valid': 0.1}

RANDOM_SEED = 0


if __name__ == '__main__':
  random.seed(RANDOM_SEED)
  np.random.seed(RANDOM_SEED)

  generator.generate_simulated_images(
    DIR_BGND, EXTENSION_BGND, DIR_FGND_LOW, DIR_FGND_HIGH, EXTENSION_FGND,
    DIR_OUT, EXTENSION_OUT, LOW_ENERGY, HIGH_ENERGY, SCALE_MIN, SCALE_MAX,
    FGND_QTY_PROB, TRAIN_VAL_RATIO
  )
