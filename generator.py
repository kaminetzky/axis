import numpy as np
import random

def sample_images_with_qty_prob(imgs, qty_prob):
  qty = np.random.choice(list(qty_prob.keys()), p=list(qty_prob.values()))
  sampled_imgs = random.sample(imgs, qty)
  return sampled_imgs