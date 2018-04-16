''' Run different tests if given confidence scores match a reference 
    distribution.
    
    Author: Christoph Lampert, http://cvml.ist.ac.at
    Project: https://github.com/ISTAustria-CVML/KSconf

    Reference:
    [R. Sun, C.H. Lampert, "KS(conf) A Light-Weight Test if a ConvNet 
    Operates Outside of Its Specifications", arXiv 2018]

    Example use: 

'''

from __future__ import print_function
from __future__ import division

import argparse
import numpy as np
import sys


def extreme_examples(self, batch, windowsize=100):
  if windowsize >= len(batch):
    return np.ones_like(batch, dtype=bool)
  
  batch_sort = np.sort(batch)
  thresh = batch_sort[windowsize] # (k+1)th lowest value
  batch_candidates = (batch<thresh) # bottom k elements
  return Y_candidates

  
class KSconf:
  """Naive implementation of KS(conf). More efficient ways would be possible."""
  def __init__(self, val):
    m = len(val)
    self.values = np.sort( np.append( np.insert(val, 0, 0.), 1.)  )
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

  def repeated_test(self, test_data, extra_data, ratio=1., batchsize=100, windowsize=0, alpha=0.01, nrepeats=10, do_thresholded=True):
    n_test = np.int(np.ceil(ratio*batchsize)) # can be 0, no problem
    n_extra = batchsize-n_test                # can be 0, no problem
    
    RES = []
    for i in range(nrepeats):
      batch_test = np.random.choice(test_data,[n_test], replace=False) # it's the user's job to make sure there's enough test samples
      
      try:
        batch_extra = np.random.choice(extra_data,[n_extra], replace=False) # works if nZ<=len(Z)
      except ValueError: 
        # our not very elegant way of coping if there are not enough samples in extra_data
        batch_extra1 = np.repeat(Z, nZ//len(Z))
        batch_extra2 = np.random.choice(Z,[nZ-len(batch_extra1)], replace=False)
        batch_extra = np.concatenate([batch_extra1,batch_extra2])
        perturb_inplace(batch_extra) # prevent duplicates
      
      batch = np.concatenate((batch_test, label_extra))

      res = self.flat_test(batch, alpha=alpha)
      RES.append(res)

      if windowsize>0: # apply filtering
        suspicious_examples = self.suspicious_examples(batch, windowsize)
        
        label_test = np.zeros_like(batch_test)
        label_extra = np.ones_like(label_extra)
        labels = np.concatenate((label_extra, label_extra))
        if suspicious_examples.any():
          accS = labels[suspicious_examples].mean()
        else:
          accS = np.nan
        extreme_examples = self.extreme_examples(batch, windowsize)
        if extreme_examples.any():
          accX = labels[extreme_examples].mean()
        else:
          accX = np.nan
        ACC.append([accS,accX])

    return RES,ACC,THR

  def suspicious_examples(self, batch, windowsize=100, do_plot=False, eps=1e-6):
    # identify overpopulated bin
    m = len(batch)
    if windowsize >= m:
      return np.ones_like(batch, dtype=bool)
    
    batch = self.flatten_cdf(batch)   # should be uniform distribution now
    
    # TODO: find more elegant way of doing this...
    nbins = m//windowsize
    bins = np.linspace(0, 1, nbins, endpoint=False)
    batch_hist = np.histogram(batch, bins )[0]
    batch_bins = np.digitize(batch, bins)-1 # put samples into bins
    batch_maxbin = np.argmax(batch_hist)  # most populated bin
    batch_candidates = (batch_bins == batch_maxbin)
    return batch_candidates
  
    
class Meantests:
  def __init__(self, val):
    self.valmean = np.mean(val) # for mean tests
    self.valstd = np.std(val)
    self.vallogmean = np.mean(np.log(val)) # for log-mean tests
    self.vallogstd = np.std(np.log(val))
  
  def adjust_thresholds(self, alpha, batchsize, nrepeats=1000):
    nrepeats = max( nrepeats, np.int(np.ceil(2./alpha)) )
    from scipy.stats import norm

    self.thr_Z = norm.ppf(alpha, loc=self.valmean, scale=self.valstd/np.sqrt(batchsize)), self.valmean, np.inf # no upper, only lower bound
    self.thr_logZ = norm.ppf(alpha, loc=self.vallogmean, scale=self.vallogstd/np.sqrt(batchsize)), self.vallogmean, np.inf # no upper, only lower bound
    self.thr_Z2 = norm.ppf(0.5*alpha, loc=self.valmean, scale=self.valstd/np.sqrt(batchsize)), self.valmean, norm.ppf(1.-0.5*alpha, loc=self.valmean, scale=self.valstd/np.sqrt(batchsize))
    self.thr_logZ2 = norm.ppf(0.5*alpha, loc=self.vallogmean, scale=self.vallogstd/np.sqrt(batchsize)), self.vallogmean, norm.ppf(1.-0.5*alpha, loc=self.vallogmean, scale=self.vallogstd/np.sqrt(batchsize))
    
    mean_stat,logmean_stat = [],[]
    OT_stat = [] 
    for i in range(nrepeats):
      batch = np.random.random([batchsize])  # random uniform 
      batch = np.sort(batch)
      
      n = len(batch)
      uniform = np.linspace(0,1,n+1,endpoint=True)
      left_diff = np.abs(batch - uniform[:-1])
      right_diff = np.abs(uniform[1:] - batch)
      
      self.cdf = lambda batch: np.interp(batch, xp=self.values, fp=self.uniform)
      self.invcdf = lambda batch: np.interp(batch, xp=self.uniform, fp=self.values)
    
      OT_stat.append( 0.5*np.abs(left_diff-right_diff).mean() )   # = EMD
      
      batch = self.invcdf(batch)                 # uniform -> P
      mean_stat.append( np.mean(batch) )
      logmean_stat.append( np.mean(np.log(batch) ) )
    
    mean_stat = np.sort(mean_stat)
    logmean_stat = np.sort(logmean_stat)
    
    OT_stat  = np.sort(OT_stat)

    index = np.int(np.floor(alpha*nrepeats)) # number of permitted outliers
    assert index >= 2
    
    self.thr_mean = mean_stat[index], np.mean(mean_stat), np.inf
    self.thr_mean2 = mean_stat[(index-1)//2], np.mean(mean_stat), mean_stat[-index//2]
    self.thr_logmean = logmean_stat[index], np.mean(logmean_stat), np.inf
    self.thr_logmean2 = logmean_stat[(index-1)//2], np.mean(logmean_stat), mean_stat[-index//2]
    self.thr_OT = OT_stat[(index-1)//2], np.mean(OT_stat), OT_stat[-index//2]
  def threshold_tests(self, batch):
    # original test without flattening
    
    #max_stat = np.max(batch)
    #min_stat = np.min(batch)
    mean_stat = np.mean(batch)
    logmean_stat = np.mean(np.log(batch))
    #prod_stat = np.exp(np.mean(np.log(batch)))
    
    batch = self.flatten_cdf(batch) # try to make ~U([0,1])
    batch = np.sort(batch)
    
    n = len(batch)
    uniform = np.linspace(0,1,n+1,endpoint=True)
    left_diff = np.abs(batch - uniform[:-1])
    right_diff = np.abs(uniform[1:] - batch)
      
    #KS_stat = np.max([left_diff, right_diff])
    #V_stat = left_diff.max()+right_diff.max()
    OT_stat = np.abs(batch-0.5*(uniform[1:]+uniform[:-1])).mean()  # approx. L1 between CDFs

    #print("Z:",mean_stat,self.thr_Z)
    #print("logZ:",logmean_stat,self.thr_logZ,self.thr_logZ2)
    return mean_stat<self.thr_mean[0], \
           logmean_stat<self.thr_logmean[0], \
           mean_stat<self.thr_Z[0], \
           logmean_stat<self.thr_logZ[0], \
           np.logical_or(mean_stat<self.thr_mean2[0], mean_stat>self.thr_mean2[2]), \
           np.logical_or(logmean_stat<self.thr_logmean2[0], logmean_stat>self.thr_logmean2[2]), \
           np.logical_or(mean_stat<self.thr_Z2[0], mean_stat>self.thr_Z2[2]), \
           np.logical_or(logmean_stat<self.thr_logZ2[0], logmean_stat>self.thr_logZ2[2]), \
           np.logical_or(OT_stat<self.thr_OT[0], OT_stat>self.thr_OT[2]) 
        
  

  
def main():
  parser = argparse.ArgumentParser(description='KS(conf) and other tests for distribution change.')
  parser.add_argument('--val', '-V', type=str, default=None, required=True, help='Text file containing within-specs confidence scores')
  parser.add_argument('--test', '-T', type=str, default='test', help='Text file containing confidence scores to be tested')
  parser.add_argument('--extra', '-X', type=str, default='test', help='Text file containing additional confidence scores')

  parser.add_argument('--batchsize', '-b', type=int, default=1000, help='Size of test batch, default: 1000')

  parser.add_argument('--ratio', '-r', type=float, default=1.0, help='Ratio for mixing "test" and "extra" data, default: 1.')
  parser.add_argument('--alpha', '-a', type=float, default=0.01, help='Target false positive rate, default: 0.01')
  parser.add_argument('--repeats', '-N', type=int, default=10000, help='Number of time to repeat each test, default: 10000')

  parser.add_argument('--windowsize', '-w', type=int, default=0, help='Number of suspicious examples to report (0 to switch off), default: 0.')
  parser.add_argument('--seed', '-s', type=int, default=None, help='Random seed (default: none)')

  parser.add_argument('--eps', type=float, default=1e-6, help='Amount of noise to use for making scores unique, (default: 1e-6)')
  parser.add_argument('--verbose', '-v', type=bool, default=False, help='Print status/debug messages, default: off')
  
  args = parser.parse_args()
  do_tests(args)

def perturb_inplace(data, eps):
  """Perturb data to make values unique. 
  We avoid leaving the [0,1] interval by perturbing very small values
  only to the positive and very large values only to the negative.
  """
  tiny_noise = np.random.uniform(0.1,0.9,len(val))
  close_to_zero_set = (data < eps)
  close_to_one_set = (data > 1.-eps)
  center_set = np.logical_not(close_to_zero_set+close_to_one_set)
  data[close_to_zero_set] += eps*tiny_noise[close_to_zero_set]
  data[close_to_one_set] -= eps*tiny_noise[close_to_one_set]
  data[center_set] += eps*(tiny_noise[center_set]-.5)
  return data
  
def do_tests(args):
  if args.seed is not None:
    np.random.seed(args.seed)
  
  assert(0 <= args.ratio <= 1)
  
  val_data = np.loadtxt(args.val)
  perturb_inplace(val_data)
  KS = KSconf(val_data)
  MT = Meantests(val_data)
  
  test_data = np.loadtxt(args.test)
  perturb_inplace(test_data)
  
  if args.ration == 1:
    extra_data = None
  else:
    if not args.extra:
      print("ratio<1 needs 'extra' data")
      raise SystemExit
    
    extra_data = np.loadtxt(args.extra)
    perturb_inplace(extra_data)
    
    if args.ratio>0.:  # true mixture of test and extra
      ntest = len(test_extra)
      if len(test_extra) == ntest:
        # specific for test and extra both generated from same images.
        # use different halfs (by indices) to make sure the same 
        # images don't end up in both sets
        idx = np.random.permutation(ntest)
        test_data = test_data[idx[::2]]
        extra_data = extra_data[idx[1::2]]    # disjoint subsets
  
  alpha = args.alpha
  batchsize = args.batchsize
  ratio = args.ratio

  if args.verbose:
    print("Batchsize effect for ratio {}, alpha {}, batchsize {}".format(self.ratio, self.alpha, self.batchsize))

  if args.alpha >= 0.0001:
    do_thresholded = True # too slow to determine thresholds otherwise
  else:
    do_thresholded = False

  if do_thresholded:
    MT.adjust_thresholds(alpha, batchsize)
  
  RES,ACC,THR = KS.repeated_test(test_data, other_data, 
                                 alpha=args.alpha, ratio=args.ratio, 
                                 batchsize=args.batchsize, 
                                 windowsize=args.windowsize, 
                                 nrepeats=args.repeats, 
                                 do_thresholded=do_thresholded)

  RES,ACC,THR = np.asarray(RES),np.asarray(ACC),np.asarray(THR)
  if args.windowsize>0 and RES.any():
    accS,accX = ACC[RES].mean(axis=0)
  else:
    accS,accX = ratio, ratio
  
  thr = np.mean(THR,axis=0)
  
  if args.latex:
    format_string = "{:5.2f}\t& {:7.5f}\t& {:7d}\t& {:7.4f}\t& {:7.4f}\t& {:7.4f}" + "\t& {:7.4f}"*len(thr) + "   \\\\" 
  else:
    format_string = "{:5.2f}\t{:7.5f}\t{:7d}\t{:7.4f}\t{:6.4f}\t{:6.4f}" + "\t{:6.4f}"*len(thr) 
  print(format_string.format(ratio, alpha, batchsize, np.mean(RES), accS, accX, *thr)) # np.sum(RES), len(RES)

if __name__ == "__main__":
  main()
