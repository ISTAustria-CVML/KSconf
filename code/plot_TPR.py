#!/usr/bin/python
from __future__ import print_function
from __future__ import division

import argparse
import numpy as np
import matplotlib.pyplot as plt

def main():
  parser = argparse.ArgumentParser(description='Plot true positive rates for different tests.')
  parser.add_argument('--filepattern', type=str, default="res-%s-AwA2-TPR.txt", help='Data file to plot. Default: "res-%s-AwA2-TPR.txt"')
  parser.add_argument('--caption', type=str, default=None, help='Caption to give to plot. Default: none')
  parser.add_argument('--title', action='store_true', help='Display model name in figure title. Default: off')
  parser.add_argument('--legend', action='store_true', help='Display legend. Default: off')
  parser.add_argument('--output', '-o', type=str, default=None, help='Filename to save figure. Default: none')
  args = parser.parse_args()

  MODELS_IN_ORDER = ['MobileNet25', 'SqueezeNet', 'ResNet50' ,'VGG19', 'NASNetAlarge']
  nmodels = len(MODELS_IN_ORDER)
  TESTS_IN_ORDER = ['KS(conf)', 'mean', 'log-mean', 'z', 'log-z', 'sym.mean', 'sym.log-mean', 'sym.z', 'sym.log-z']
  
  left_offset,right_offset = 0,0
  if args.caption:
    left_offset = 0.07  # leave space on left for caption
  if args.legend:
    right_offset = 0.13  # leave space on right for legend
  
  TEST_TO_COL = {'KS(conf)': 3, 'mean':4, 'log-mean':5, 'z':6, 'log-z':7, 
                 'sym.mean':8, 'sym.log-mean':9, 'sym.z':10, 'sym.log-z':11}

  Xvals_all = np.linspace(0.,1.,21) # 0.05 steps
  Xvals_show = np.linspace(0.,1.,11)
  
  COLOR = ['b','#E07030','r','c','m','#E07030','r','c','m']
  STYLE = ['-']*5 + [(0, (2, 2))]*4 # first solid, then densely dashed

  fig = plt.figure(figsize=(17,1.8))
  plt.subplots_adjust(left=0.1+left_offset, right=0.9-right_offset, top=0.87, bottom=0.12)

  for k,model in enumerate(MODELS_IN_ORDER):
    data = np.loadtxt(args.filepattern % model)
    ax = plt.subplot(1, nmodels, k+1)
    if args.caption and k==0:
      plt.text(-1.3,0.45, args.caption, fontsize=18)
    if args.title:
      ax.set_title(model)
    
    for i,test_name in enumerate(TESTS_IN_ORDER):
      col = TEST_TO_COL[test_name]
        
      if args.legend and k == nmodels-1 and i==5:
        ax.plot( [0],[0], ',', c="w", label=" ")  # empty entry for prettier legend
      line = ax.plot( Xvals_all, data[::-1,col], zorder=2, color=COLOR[i], linestyle=STYLE[i], linewidth=2) # reverse order for better interpretability
      if args.legend and k == nmodels-1:
        line[0].set_label(test_name)
      
      ax.set_xticks([0,0.25,0.5,0.75,1])
      ax.set_xticklabels([r'0.0',r'0.25',r'0.5',r'0.75',r'1.0'], fontsize=12)
      ax.set_yticks([0,0.25,0.5,0.75,1])
      ax.set_yticklabels([r'0.0',r'0.25',r'0.5',r'0.75',r'1.0'], fontsize=12)

      ax.grid(True, zorder=3)
      ax.axis([-0.04,1.04,-0.04,1.04])

    if args.legend and k == nmodels-1:
      ax.legend(ncol=2, bbox_to_anchor=(3.1, 1),fontsize=12)
        
  if args.output:
    plt.savefig(args.output)
  else:
    plt.show()


if __name__ == "__main__":
  main()
