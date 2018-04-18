from __future__ import print_function
from __future__ import division

import numpy as np
from utils import perturb_inplace, make_batch, filter_extreme_examples

class KSconf:
  """Naive implementation of KS(conf). More efficient ways would be possible."""
  def __init__(self, val_data, eps=1e-6):
    m = len(val_data)
    self.eps = eps
    val_data = perturb_inplace(val_data, self.eps)
  
    self.values = np.sort( np.append( np.insert(val_data, 0, 0.), 1.)  )
    self.uniform = np.linspace(0, 1, m+2, endpoint=True)
    self.cdf = lambda batch: np.interp(batch, xp=self.values, fp=self.uniform)
    self.invcdf = lambda batch: np.interp(batch, xp=self.uniform, fp=self.values)
    import KStab
    self.KStab = KStab.KStab

  def flatten_cdf(self, Y):
    return self.cdf(Y)    # turn x~P into x~U([0,1])

  def flat_test(self, batch, alpha=0.01):
    batch = self.flatten_cdf(batch) # try to make ~U([0,1])
    batch = np.sort(batch)
    
    n = len(batch)
    uniform = np.linspace(0,1,n+1,endpoint=True)
    left_diff = np.abs(batch - uniform[:-1])    # |F(y_k)-k/N|
    right_diff = np.abs(uniform[1:] - batch)    # |k+1/N-F(y_k+1)|
    KS_stat = np.max([left_diff, right_diff]) # = max(max(D+,D-))
    
    try:
      thresh = self.KStab[n,alpha]  # check if we have tabulated value
    except KeyError:
      c=np.sqrt(-0.5*np.log(alpha/2.))  # otherwise use closed-form
      thresh = c/np.sqrt(n)
    
    return KS_stat>thresh # return test result: True/False

  def repeated_test(self, test_data, extra_data, ratio, batchsize, alpha, nrepeats, windowsize=0):
    n_extra = np.int(np.floor(ratio*batchsize)) # can be 0, no problem
    n_test = batchsize-n_extra                 # can be 0, no problem
    
    RES = [] # result of out-of-specs tests
    ACC = [] # result of filtering by density or score  (only for positive tests)
    for i in range(nrepeats):
      batch_test = make_batch(test_data, n_test)
      if n_extra == 0:
        batch = batch_test
      else:
        batch_extra = make_batch(extra_data, n_extra)
        batch = np.concatenate((batch_test, batch_extra))

      perturb_inplace(batch, self.eps)
      res = self.flat_test(batch, alpha=alpha)
      RES.append(res)
      
      if windowsize>0: # apply filtering
        suspicious_examples = self.filter_suspicious_examples(batch, windowsize)
        
        labels = np.zeros_like(batch_test)
        if n_extra > 0:
          label_extra = np.ones_like(batch_extra)
          labels = np.concatenate((labels, label_extra))
        assert( suspicious_examples.any() )
        acc_suspicious = labels[suspicious_examples].mean()
        
        extreme_examples = filter_extreme_examples(batch, windowsize)
        assert( extreme_examples.any() )
        acc_extreme = labels[extreme_examples].mean()
        ACC.append([acc_suspicious,acc_extreme])

    return np.asarray(RES),np.asarray(ACC) 

  def filter_suspicious_examples(self, batch, windowsize): # identify overpopulated bin
    m = len(batch)
    if windowsize >= m:
      return np.ones_like(batch, dtype=bool)
    
    batch = self.flatten_cdf(batch) # after cdf, distribution should be uniform
    
    # TODO: find more elegant way of doing this...
    nbins = m//windowsize
    bins = np.linspace(0, 1, nbins, endpoint=False)
    batch_hist = np.histogram(batch, bins )[0]
    batch_bins = np.digitize(batch, bins)-1 # put samples into bins
    batch_maxbin = np.argmax(batch_hist)  # most populated bin
    batch_candidates = (batch_bins == batch_maxbin)
    return batch_candidates
  
