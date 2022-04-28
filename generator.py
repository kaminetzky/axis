import numpy as np
import random

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
