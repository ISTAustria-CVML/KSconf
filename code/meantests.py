from __future__ import print_function
from __future__ import division

import numpy as np
from utils import perturb_inplace, make_batch
from scipy.stats import norm as normal_dist # cdf for z-tests

class Meantests:
  """Naive implementation of mean-based tests."""
  def __init__(self, val_data, eps):
    self.eps = eps
    # compute reference means
    self.valmean = np.mean(val_data)
    self.valstd = np.std(val_data)
    self.vallogmean = np.mean(np.log(val_data))
    self.vallogstd = np.std(np.log(val_data))
    
  
  def adjust_thresholds(self, val_data, alpha, batchsize, nrepeats=10000, maxrepeats=1e6):

    self.thr_Z = normal_dist.ppf(alpha, loc=self.valmean, scale=self.valstd/np.sqrt(batchsize)), np.inf # no upper bound, only lower 
    self.thr_logZ = normal_dist.ppf(alpha, loc=self.vallogmean, scale=self.vallogstd/np.sqrt(batchsize)), np.inf # no upper bound, only lower 
    self.thr_symZ = normal_dist.ppf(0.5*alpha, loc=self.valmean, scale=self.valstd/np.sqrt(batchsize)), normal_dist.ppf(1.-0.5*alpha, loc=self.valmean, scale=self.valstd/np.sqrt(batchsize))
    self.thr_symlogZ = normal_dist.ppf(0.5*alpha, loc=self.vallogmean, scale=self.vallogstd/np.sqrt(batchsize)), normal_dist.ppf(1.-0.5*alpha, loc=self.vallogmean, scale=self.vallogstd/np.sqrt(batchsize))
    
    nrepeats = max( nrepeats, np.int(np.ceil(2./alpha)) )
    if nrepeats <= maxrepeats:
      mean_stat,logmean_stat = [],[]
      for i in range(nrepeats):
        batch = make_batch(val_data, batchsize)
        mean_stat.append( np.mean(batch) )
        logmean_stat.append( np.mean(np.log(batch) ) )
      
      mean_stat = np.sort(mean_stat)
      logmean_stat = np.sort(logmean_stat)
      index = np.int(np.floor(alpha*nrepeats)) # number of permitted outliers
      
      self.thr_mean = mean_stat[index], np.inf
      self.thr_logmean = logmean_stat[index], np.inf
      self.thr_symmean = mean_stat[(index-1)//2], mean_stat[-index//2]
      self.thr_symlogmean = logmean_stat[(index-1)//2], mean_stat[-index//2]
    else: # disable tests
      self.thr_mean = -np.inf, np.inf
      self.thr_logmean = -np.inf, np.inf
      self.thr_symmean = -np.inf, np.inf
      self.thr_symlogmean = -np.inf, np.inf

  def threshold_tests(self, batch):
    mean_stat = np.mean(batch)
    logmean_stat = np.mean(np.log(batch))
    
    return (mean_stat<self.thr_mean[0], 
            logmean_stat<self.thr_logmean[0], 
            mean_stat<self.thr_Z[0], 
            logmean_stat<self.thr_logZ[0], 
            np.logical_or(mean_stat<self.thr_symmean[0], mean_stat>self.thr_symmean[1]), 
            np.logical_or(logmean_stat<self.thr_symlogmean[0], logmean_stat>self.thr_symlogmean[1]), 
            np.logical_or(mean_stat<self.thr_symZ[0], mean_stat>self.thr_symZ[1]), 
            np.logical_or(logmean_stat<self.thr_symlogZ[0], logmean_stat>self.thr_symlogZ[1])
           )
  
  def repeated_test(self, test_data, extra_data, ratio, batchsize, alpha, nrepeats, windowsize=0):
    n_extra = np.int(np.floor(ratio*batchsize)) # can be 0, no problem
    n_test = batchsize-n_extra                # can be 0, no problem
    
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
      res = self.threshold_tests(batch)
      RES.append(res)
      
    return np.asarray(RES)


