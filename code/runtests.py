''' Run different tests if given confidence scores match a reference 
    distribution.
    
    Author: Christoph Lampert, http://cvml.ist.ac.at
    Project: https://github.com/ISTAustria-CVML/KSconf

    Reference:
    [R. Sun, C.H. Lampert, "KS(conf) A Light-Weight Test if a ConvNet 
    Operates Outside of Its Specifications", arXiv:1804.04171 [stat.ML]]

    Example uses: 
    ******************************************************************
    Check false positives by testing 'test' vs. 'val' with defaults
    alpha (0.01) and batchsize (1000)

    > python runtests.py --val ../conf/ResNet50-val-max.npy \
                         --test ../conf/ResNet50-test-max.npy

    output:
    #ratio alpha batchsz  KSconf  mean   logmean    z      logz   s.mean  s.logm    s.z   s.logz
     1.00 0.01000    1000 0.0090 0.0114 0.0083 0.0123 0.0101 0.0109 0.0042 0.0107 0.0094

    ******************************************************************
    Check detection rate on 'blue+whale' class with alpha=0.01 and batchsize=100. 
    Set the random seed to 0, do only 100 repeats, print the results without header line.

    > python runtests.py --val ../conf/ResNet50-val-max.npy \
                         --test ../conf/ResNet50-blue+whale-max.npy \
                         --alpha 0.01 --batchsize 100 --seed 0 \
                         --repeats 100 --noheader

    output:
     1.00  0.010   100    1.0000 0.0100 0.0000 0.0100 0.0000 0.0100 0.0000 0.0100 0.0000

    ******************************************************************
    Check detection rate when mixing 25% 'seal' with 75% 'ILSVRC test' 
    for VGG19 network confidence scores and filtering results for a 
    windowsize of 10. Use short argument names.

    > python runtests.py -V ../conf/VGG19-val-max.npy \
                         -T ../conf/VGG19-seal-max.npy \
                         -X ../conf/VGG19-test-max.npy \
                         -r 0.25 -w 10 

    output:
    #ratio  alpha batchsz KSconf  mean   logmean    z      logz   s.mean  s.logm    s.z   s.logz F-susp. F-extr.
     0.25   0.010    1000 0.6247 0.0000   0.0000  0.0000  0.0000  0.7308  0.0000   0.7128 0.6591 0.7070  0.9821
'''

from __future__ import print_function
from __future__ import division

import argparse
import numpy as np
import sys

from KSconf import KSconf
from meantests import Meantests
from utils import perturb_inplace
  
def main():
  parser = argparse.ArgumentParser(description='KS(conf) and other tests for distribution change.')
  parser.add_argument('--val', '-V', type=str, default=None, required=True, help='NPY file containing within-specs confidence scores')
  parser.add_argument('--test', '-T', type=str, default=None, required=True, help='NPY file containing confidence scores to be tested')
  parser.add_argument('--extra', '-X', type=str, default=None, help='NPY file containing additional confidence scores')

  parser.add_argument('--batchsize', '-b', type=int, default=1000, help='Size of test batch, default: 1000')

  parser.add_argument('--ratio', '-r', type=float, default=0.0, help='Ratio for mixing "extra" and "test" data, default: 0.')
  parser.add_argument('--alpha', '-a', type=float, default=0.01, help='Target false positive rate, default: 0.01')
  parser.add_argument('--repeats', '-N', type=int, default=10000, help='Number of time to repeat each test, default: 10000')

  parser.add_argument('--windowsize', '-w', type=int, default=0, help='Number of suspicious examples to report (0 to switch off), default: 0.')
  parser.add_argument('--seed', '-s', type=int, default=None, help='Random seed (default: none)')

  parser.add_argument('--eps', type=float, default=1e-6, help='Amount of noise to use for making scores unique, (default: 1e-6)')
  parser.add_argument('--verbose', '-v', action='store_true', help='Print status/debug messages, default: off')
  parser.add_argument('--noheader', action='store_false', dest='header', help='Do not print header explaining result columns, default: off')
  
  args = parser.parse_args()
  do_tests(args)

def do_tests(args):
  if args.seed is not None:
    np.random.seed(args.seed)
  
  assert(0 <= args.ratio <= 1)
  
  val_data = np.load(args.val)
  test_data = np.load(args.test)
  n_test = len(test_data)
  
  if args.ratio == 0:
    extra_data = None
  else:
    if not args.extra:
      print("ratio>0 needs 'extra' data")
      raise SystemExit
    
    extra_data = np.load(args.extra)
    
    if args.ratio>0.:  # mixed batches of test and extra data?
      n_extra = len(extra_data)
      if n_extra == n_test: # TODO: add proper flag instead of this heuristic
        # specific for test and extra both generated from same images.
        # use different halfs (by indices) to make sure the same 
        # images don't end up in both sets
        idx = np.random.permutation(n_extra)
        test_data = test_data[idx[::2]]
        extra_data = extra_data[idx[1::2]]    # disjoint subsets
  
  if args.verbose:
    if args.extra:
      print("#Testing '{}' vs '{}' and '{}'".format(args.val,args.test,args.extra))
    else:
      print("#Testing '{}' vs '{}'".format(args.val,args.test))
  
  
  # perform KS-based tests and potentially filtering
  
  KS = KSconf(val_data, eps=args.eps)
  
  res_KSconf,ACC = KS.repeated_test(test_data, extra_data, 
                             alpha=args.alpha, 
                             ratio=args.ratio, 
                             batchsize=args.batchsize, 
                             nrepeats=args.repeats,
                             windowsize=args.windowsize)
  rate_KSconf = res_KSconf.mean(axis=0)
  
  # perform mean-based tests
  
  MT = Meantests(val_data, eps=args.eps)
  MT.adjust_thresholds(val_data, args.alpha, args.batchsize)
  res_meantests = MT.repeated_test(test_data, extra_data, 
                                   alpha=args.alpha, 
                                   ratio=args.ratio, 
                                   batchsize=args.batchsize, 
                                   nrepeats=args.repeats,
                                   windowsize=args.windowsize)
  rate_meantests = res_meantests.mean(axis=0)
  
  # print results
  
  if args.header:
    header_string = "#ratio\talpha\tbatchsz\tKSconf"
  output_string = "{:5.2f}\t{:7.5f}\t{:7d}".format(args.ratio, args.alpha, args.batchsize)
  output_string += "\t{:6.4f}".format(rate_KSconf) # KS(conf)

  if args.header:
    header_string += "\tmean\tlogmean\tz\tlogz\ts.mean\ts.logm\ts.z\ts.logz"
  output_string += ("\t{:6.4f}"*len(rate_meantests)).format(*rate_meantests) # mean-based tests

  if args.windowsize>0:
    if res_KSconf.any():
      acc_suspicious,acc_extreme = ACC[res_KSconf].mean(axis=0)
    else:
      acc_suspicious,acc_extreme = args.ratio, args.ratio # without any positive tests, return result of random guessing

    if args.header:
      header_string += "\tF-susp.\tF-extr."
    output_string += "\t{:6.4f}\t{:6.4f}".format(acc_suspicious,acc_extreme)

  if args.header:
    print(header_string)
  print(output_string)


if __name__ == "__main__":
  main()
