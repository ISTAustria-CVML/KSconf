#!/usr/bin/python
from __future__ import print_function
from __future__ import division

import argparse
import numpy as np
import matplotlib.pyplot as plt

def main():
  parser = argparse.ArgumentParser(description='Plot score distributions for different networks.')

  parser.add_argument('--filepattern', type=str, default="%s-test-max.npy", help='Data file to plot. Default: "%s-test-max.npy"')
  parser.add_argument('--caption', type=str, default=None, help='Caption to give to plot. Default: none')
  parser.add_argument('--title', action='store_true', help='Display model name in figure title. Default: off')
  parser.add_argument('--output', '-o', type=str, default=None, help='Filename to save figure. Default: none')
  parser.add_argument('--nbins', type=int, default=50, help='Number of bins in plot (default: 50)')
  args = parser.parse_args()

  MODELS_IN_ORDER = ['MobileNet25', 'SqueezeNet', 'ResNet50' ,'VGG19', 'NASNetAlarge']
  nmodels = len(MODELS_IN_ORDER)

  nbins = args.nbins
  
  left_offset = 0
  if args.caption:
    left_offset = 0.07  # leave space on left for caption
  
  fig = plt.figure(figsize=(14,1+1.5))
  plt.subplots_adjust(left=0.1+left_offset, right=0.98, top=0.87, bottom=0.12)
  
  bins = np.linspace(0,1,nbins+1)
  for k,model in enumerate(MODELS_IN_ORDER):
    data = np.load(args.filepattern % model)
    H = np.histogram(data, bins=bins, density=True)[0]
    
    ax = plt.subplot(1, nmodels, k+1)
    caption_text = args.caption.replace('_',' ').replace('+',' ')
    if args.caption and k==0:
        text = caption_text.split('/')
        if len(text) == 1:
            ax.text(-1.2,7.5, text[0], fontsize=16)
        else:
            ax.text(-1.2,8.5, text[0], fontsize=16)
            ax.text(-1.2,4.5, text[1], fontsize=16)
    if args.title:
      ax.set_title(model, fontsize=14)

    try:  # bar syntax differs between version of matplotlib
      ax.bar(x=bins[:-1], width=1./nbins, bottom=0, height=H, color='b', zorder=2, edgecolor='b')
    except TypeError:
      ax.bar(left=bins[:-1], height=H, width=1./nbins, bottom=0, color='b', zorder=2, edgecolor='b')
    Hmean = np.mean(data)
    ax.plot([Hmean,Hmean],[0,15],color='#E07030', zorder=1, linewidth=2, linestyle="--")

    ax.set_xticks([0,0.5,1])
    for tick in ax.yaxis.get_major_ticks():
      tick.label.set_fontsize(14)
    ax.set_yticks([0,5,10,15])
    for tick in ax.xaxis.get_major_ticks():
      tick.label.set_fontsize(14)
    plt.axis([-0.05,1.05,-.5,15.5])
    ax.grid(True, zorder=3)

  if args.output:
    plt.savefig(args.output)
  else:
    plt.show()
      
if __name__ == "__main__":
  main()
