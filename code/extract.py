''' Extract ConvNet confidence scores from images, potentially after 
    distorting them first. Supports different pretrained networks 
    that are accessible through the fabulous TensorNet framework 
    https://github.com/taehoonlee/tensornets

    Author: Christoph Lampert, http://cvml.ist.ac.at
    Project: https://github.com/ISTAustria-CVML/KSconf

    Reference:
    [R. Sun, C.H. Lampert, "KS(conf) A Light-Weight Test if a ConvNet 
    Operates Outside of Its Specifications", arXiv 2018]

    Example use: 
    
    - extract VGG19 confidence scores from an image

    > python extract.py -m VGG19 -i image.jpg -o VGG19-conf.txt

    - extract MobileNet25 and ResNet50 confidence scores for all 
      images in a directory after flipping horizontally and  
      blurring by different amounts (sigma=1,5,10):
    
    > for M in MobileNet25 ResNet50 ; do
    >   for BLUR in 1 5 10 ; do 
    >      python extract.py -m ${M} -i "${IMAGEPATH}/*.JPEG" -o ${M}-conf-blur${B}.txt --blur ${B} --geometry flipH; 
    >   done 
    > done
'''

from __future__ import print_function
from __future__ import division

import argparse
import glob
import numpy as np
import sys
import tensorflow as tf
import tensornets as nets
import time

from tensornets import utils

def main():
  parser = argparse.ArgumentParser(description='extract confidence scores from images.')
  parser.add_argument('--images', '-i', type=str, default=None, help='Filename or -pattern of images', required=True)
  parser.add_argument('--model', '-m', type=str, default='ResNet50', 
                      choices=['MobileNet25', 'ResNet50', 'NASNetAlarge', 'SqueezeNet' ,'VGG19'],
                      help='Model name (default: ResNet50)')
  
  parser.add_argument('--output', '-o', type=str, default=None, help='Output filename for confidence values, default: none')
  parser.add_argument('--rawoutput', type=str, default=None, help='Output filename for complete activations (in NPZ format), default: none')
  parser.add_argument('--labeloutput', type=str, default=None, help='Output filename for predicted labels, default: none')
  
  parser.add_argument('--batchsize', '-b', type=int, default=None, help='Batchsize for ConvNet evaluation, default: automatic')

  parser.add_argument('--blur', type=float, default=None, help='Amount of blur to apply (Gaussian sigma), default: none')
  parser.add_argument('--noise', type=float, default=None, help='Amount of noise to apply (Gaussian sigma), default: none')
  parser.add_argument('--dead', type=float, default=None, help='Percentage of dead pixels to create, default: none')
  parser.add_argument('--dark', type=float, default=None, help='Scale factor towards 0 for underexposure, default: none')
  parser.add_argument('--bright', type=float, default=None, help='Scale factor towards 255 for overexposure, default: none')
  
  parser.add_argument('--geometry', type=str, default=None, 
                      choices=['flipH', 'flipV', 'rotate90', 'rotate180' ,'rotate270'],
                      help='Geometric transformation to apply, default: none')
  parser.add_argument('--swapcolors', action='store_true', help='Swap color channels RGB->BGR')
  parser.add_argument('--verbose', '-v', action='store_true', help='Print extra information')

  parser.add_argument('--seed', '-s', type=int, default=None, help='Random seed, default: none')

  args = parser.parse_args()
  do_extract(args)

  
def apply_blur(img, sigma):
  from scipy.ndimage.filters import gaussian_filter
  img = gaussian_filter(img, sigma=sigma)
  return img

def apply_noise(img, sigma):
  noise = np.random.standard_normal(img.shape)
  img = np.clip( img+sigma*noise, 0., 255.)
  return img

def apply_deadpixels(img, p):
  orig_shape = img.shape
  img = img.reshape(-1, img.shape[-1])  # flatten into 1D list of BGR
  num_pixels = len(img)
  num_defects = int( (p*num_pixels)//100 )
  idx = np.random.choice(num_pixels, num_defects, replace=False)
  img[idx[::2]] = [0,0,0]        # half pixels black
  img[idx[1::2]] = [255,255,255] # half pixels white
  img = img.reshape(orig_shape)
  return img

def apply_scaletoblack(img, factor):
  img = img*(1./factor)
  return img
  
def apply_scaletowhite(img, factor):
  img = 255-(1./factor)*(255-img)
  return img

def apply_geometry(img, geometry):
  if geometry == 'flipH':
    img = img[:,:,::-1,:]  
  elif geometry == 'flipV':
    img = img[:,::-1,:,:]
  elif geometry == 'rotate90':
    img = np.rot90(img, k=1, axes=(1,2))
  elif geometry == 'rotate180':
    img = np.rot90(img, k=2, axes=(1,2))
  elif geometry == 'rotate270':
    img = np.rot90(img, k=-1, axes=(1,2))
  return img


def do_extract(args):
  if args.seed is not None:
    np.random.seed(args.seed)

  FILENAMES = sorted(glob.glob(args.images))
  nimages = len(FILENAMES)
  if nimages == 0:
    print("Did not find any images in '{}".format(args.images))
    return 
  if args.verbose:
    print("Processing {} images".format(nimages))
    
  if args.model == 'NASNetAlarge':
    target_size, crop_size = 331, 331
  else:
    target_size, crop_size = 256, 224
  
  inputs = tf.placeholder(tf.float32, [None, crop_size, crop_size, 3])
  
  if args.model == 'NASNetAlarge':
    model = nets.NASNetAlarge(inputs)
    bs = 250
  elif args.model == 'VGG19':
    model = nets.VGG19(inputs)
    bs = 250
  elif args.model == 'MobileNet25':
    model = nets.MobileNet25(inputs)
    bs = 2500
  elif args.model == 'SqueezeNet':
    model = nets.SqueezeNet(inputs)
    bs = 1000
  elif args.model == 'ResNet50':
    model = nets.ResNet50(inputs)
    bs = 500
  else:
    raise ValueError # this should not happen
  
  model_pretrained = model.pretrained()

  if args.batchsize: 
    bs = args.batchsize # overwrite default batchsize

  nchunks = (nimages+bs-1)//bs
  
  PREDS = [] # beware: we store all activations (#images x #classes x sizeof(float))
  with tf.Session() as sess:
      sess.run(model_pretrained)
      
      for i in xrange(nchunks):
          images = []
          for filename in FILENAMES[i*bs:(i+1)*bs]:
              img = utils.load_img(filename, target_size=target_size, crop_size=crop_size)
              if args.blur:
                  img = apply_blur(img, args.blur)
              if args.noise:
                  img = apply_noise(img, args.noise)
              if args.dead:
                  img = apply_deadpixels(img, args.dead)
              if args.dark:
                  img = apply_scaletoblack(img, args.dark)
              if args.bright:
                  img = apply_scaletoblack(img, args.bright)
              if args.geometry:
                  img = apply_geometry(img, args.geometry)

              images.append( img.squeeze() )

          images = model.preprocess(np.asarray(images))

          preds = sess.run(model, {inputs: images})
          PREDS.extend(preds)

          if args.verbose:
            print('Processed chunk {} of {}'.format(i, nchunks))
            print('Most recent prediction:', utils.decode_predictions(preds, top=1)[0])

      PREDS = np.asarray(PREDS)
      if args.output:
          np.savetxt(args.output, PREDS.max(axis=1), fmt='%.12f')
      if args.rawoutput:
          np.savez_compressed(args.rawoutput, PREDS)
      if args.labeloutput:
          np.savetxt(args.labeloutput, PREDS.argmax(axis=1), fmt='%d')

  if args.verbose:
    print("Done.")

if __name__ == "__main__":
  main()
