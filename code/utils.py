from __future__ import print_function
from __future__ import division

import numpy as np

def filter_extreme_examples(batch, windowsize=100):
  if windowsize >= len(batch):
    return np.ones_like(batch, dtype=bool)
  
  batch_sort = np.sort(batch)
  thresh = batch_sort[windowsize] # (k+1)th lowest value
  candidates = (batch<thresh) # bottom k elements
  return candidates


def make_batch(data, n):
  try:
    batch = np.random.choice(data, [n], replace=False) # works if n<=len(data)
  except ValueError: 
    # not very elegant way of coping if there are too few samples for desired batchsize
    batch1 = np.repeat(data, n//len(data))
    batch2 = np.random.choice(data,[n-len(batch1)], replace=False)
    batch = np.concatenate([batch1,batch2])
  return batch

def perturb_inplace(data, eps):
  """Perturb data to make values unique. 
  We avoid leaving the [0,1] interval by perturbing very small values
  only to the positive and very large values only to the negative.
  """
  tiny_noise = np.random.uniform(0.1,0.9,len(data))
  close_to_zero_set = (data < eps)
  close_to_one_set = (data > 1.-eps)
  center_set = np.logical_not(close_to_zero_set+close_to_one_set)
  data[close_to_zero_set] += eps*tiny_noise[close_to_zero_set]
  data[close_to_one_set] -= eps*tiny_noise[close_to_one_set]
  data[center_set] += eps*(tiny_noise[center_set]-.5)
  return data
  
