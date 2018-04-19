#!/usr/bin/python
from __future__ import print_function
from __future__ import division

import argparse
import numpy as np
import matplotlib.pyplot as plt
  
NETWORKS=['MobileNet25','NASNetAlarge','ResNet50','SqueezeNet','VGG19']
BS = [1, 10, 100, 1000]
ALPHA=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
  
FILEPATTERN = '../res/res-%s-FPR.txt'

TEST_TO_COL = {'KS(conf)': 3, 'mean':4, 'log-mean':5, 'z':6, 'log-z':7, 
               'sym.mean':8, 'sym.log-mean':9, 'sym.z':10, 'sym.log-z':11}

COL_TO_TEST = dict(zip(TEST_TO_COL.values(),TEST_TO_COL.keys()))

COLOR = "kbrcmg"
STYLE = "xo+d"

def main():
  parser = argparse.ArgumentParser(description='Plot false positive rates for different tests.')
  parser.add_argument('--test', type=str, default="KS(conf)", help='Test to plot (name or column id). Default: "KS(conf)"')
  parser.add_argument('--output', '-o', type=str, default=None, help='Filename to save figure. Default: none')
  parser.add_argument('--filepattern', type=str, default='res-%s-FPR.txt', help='File pattern to read data from. Default: "res-%s-FPR.txt"')
  parser.add_argument('--legend', action='store_true', help='Display legend. Default: off')
  args = parser.parse_args()
  
  try:
    col = TEST_TO_COL[args.test]
  except KeyError:
    try:
      col = int(args.test)
    except ValueError:
      print("Unknown test. '%s'".args.test)
      raise SystemExit
  
  fig=plt.figure(figsize=(4.2,4.2))
  plt.title(COL_TO_TEST[col], fontsize=16)
  plt.loglog([1e-5,1],[1e-5,1],linestyle='-',color='k')

  plt1,plt2 = [],[]  # collect lines for legend
  for k,network in enumerate(NETWORKS):
    X = np.loadtxt(args.filepattern % network)
    X = X.reshape(len(BS),len(ALPHA),-1)

    for i,bs in enumerate(BS):
      for j,alpha in enumerate(ALPHA):
        val = X[i,j,col]
        if args.legend and i==0 and j==0:
          plt1.extend( plt.loglog([alpha],[val],marker=STYLE[i],color=COLOR[k],label=network) )
        elif args.legend and k==0 and j==1:
          plt2.extend( plt.loglog([alpha],[val],marker=STYLE[i],color=COLOR[k],label="bs %d"%bs) )
        else:
          plt.loglog([alpha],[val],marker=STYLE[i],color=COLOR[k])

  if args.legend:
    leg1 = plt.legend(handles=plt1, fontsize=12, loc='upper left')
    ax = plt.gca().add_artist(leg1)
    plt.legend(handles=plt2, loc='lower right')

  plt.tight_layout()
  if args.output:
    plt.savefig(args.output)
  else:
    plt.show()


if __name__ == "__main__":
  main()
