#!/usr/bin/env python3
''' Extract ConvNet confidence scores from images, potentially after 
    distorting them first. Supports different pretrained networks 
    that are accessible through the fabulous TensorNet framework 
    https://github.com/taehoonlee/tensornets

    Author: Christoph Lampert, http://cvml.ist.ac.at
    Project: https://github.com/ISTAustria-CVML/KSconf

    Reference:
    [R. Sun, C.H. Lampert, "KS(conf) A Light-Weight Test if a ConvNet 
    Operates Outside of Its Specifications", GCPR 2019]
    [R. Sun, C.H. Lampert, "KS(conf) A Light-Weight Test if a Classifier
    Operates Outside of Its Specifications", IJCV 2019]

    EXAMPLE: extract VGG19 confidence scores from an image

    > python extract.py -m VGG19 -i image.jpg -o VGG19-conf.txt

    EXAMPLE: extract MobileNet25 and ResNet50 confidence scores for 
      all images in a directory after flipping horizontally and  
      blurring by different amounts (sigma=1,5,10), and applying 
      ODIN preprocessing/postprocessing with temperature T=1000 and 
      perturbation strength epsilon=0.02:
    
    > for M in MobileNet25 ResNet50 ; do
    >   for BLUR in 1 5 10 ; do 
    >      python extract.py -m ${M} -i "${IMAGEPATH}/*.JPEG" \
    >                        -o conf-${M}-blur${B}.txt --blur ${B} \
    >                        --geometry flipH --T ${T} --perturb 0.02 ; 
    >   done 
    > done
    
    EXAMPLE: extract predictions for all classes for uniform noise:

    > python extract.py -m SqueezeNet -S Uniform --rawoutput squeezeNet-uniform.npz

'''

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
  parser.add_argument('--images', '-i', type=str, default=None, help='Filename or -pattern of images', required=False)
  parser.add_argument('--model', '-m', type=str, default='ResNet50', 
                      choices=['MobileNet25', 'ResNet50', 'ResNet152v2', 'NASNetAlarge', 'SqueezeNet' ,'VGG19'],
                      help='Model name (default: ResNet50)')
  
  parser.add_argument('--synth', '-S', type=str, default=None, help='Synthetic transformation to use as input', 
                      choices=['Uniform', 'Gaussian', 'Binary'], required=False)
  
  parser.add_argument('--output', '-o', type=str, default=None, help='Output filename for confidence values, default: none')
  parser.add_argument('--rawoutput', type=str, default=None, help='Output filename for complete activations (in NPZ format), default: none')
  parser.add_argument('--labeloutput', type=str, default=None, help='Output filename for predicted labels, default: none')
  parser.add_argument('--temperature', '-T', type=float, default=None, help='Temperature to apply before softmax')  
  parser.add_argument('--perturb', type=float, default=None, help='Strength of image perturbation following [Liang etal, "ODIN", ICLR 2018]')
 
  parser.add_argument('--batchsize', '-b', type=int, default=None, help='Batchsize for ConvNet evaluation, default: automatic')

  parser.add_argument('--blur', type=float, default=None, help='Amount of blur to apply (Gaussian sigma), default: none')
  parser.add_argument('--noise', type=float, default=None, help='Amount of noise to apply (Gaussian sigma), default: none')
  parser.add_argument('--dead', type=float, default=None, help='Percentage of dead pixels to create, default: none')
  parser.add_argument('--dark', type=float, default=None, help='Scale factor towards 0 for underexposure, default: none')
  parser.add_argument('--bright', type=float, default=None, help='Scale factor towards 255 for overexposure, default: none')
  
  parser.add_argument('--geometry', type=str, default=None, 
                      choices=['flipH', 'flipV', 'rotate90', 'rotate180' ,'rotate270'],
                      help='Geometric transformation to apply, default: none')
                      
  parser.add_argument('--swapRGB', action='store_true', help='Swap color channels RGB->BGR')
  parser.add_argument('--verbose', '-v', action='store_true', help='Print extra information')

  parser.add_argument('--seed', '-s', type=int, default=None, help='Random seed, default: none')

  args = parser.parse_args()
  
  if args.images or args.synth:
    do_extract(args)
  else:
    print("ERROR: You must specify either image direction or synthetic transformation")
    raise SystemExit

  
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
  img = img.reshape(-1, img.shape[-1])  # flatten into 1D list of pixels
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

def apply_swapRBG(img):
  img = img[:,:,:,::-1]
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
    tf.set_random_seed(args.seed)

  if args.synth:
    FILENAMES = None
    nimages = 10000
    print("Using {} synthetic images of type {}".format(nimages, args.synth))
  elif args.images:
    FILENAMES = sorted(glob.glob(args.images))
    nimages = len(FILENAMES)
    if nimages == 0:
      print("Did not find any images in '{}".format(args.images))
      return 
    if args.verbose:
      print("Processing {} images".format(nimages))
  else:
     return     
    
  if args.model == 'NASNetAlarge':
    target_size, crop_size = 331, 331
  else:
    target_size, crop_size = 256, 224
  
  inputs = tf.placeholder(tf.float32, [None, crop_size, crop_size, 3])
  
  if args.model == 'NASNetAlarge':
    model = nets.NASNetAlarge(inputs)
    bs = 100
    if args.perturb:
      bs = 20 # why only so small? who knows...
  elif args.model == 'VGG19':
    model = nets.VGG19(inputs)
    bs = 180 
    if args.perturb:
      bs //= 2
  elif args.model == 'MobileNet25':
    model = nets.MobileNet25(inputs)
    bs = 2500
    if args.perturb:
      bs //= 3
  elif args.model == 'SqueezeNet':
    model = nets.SqueezeNet(inputs)
    bs = 1000
    if args.perturb:
      bs //= 2
  elif args.model == 'ResNet50':
    model = nets.ResNet50(inputs)
    bs = 500
    if args.perturb:
      bs = 100 #(bs+4)//5
  elif args.model == 'ResNet152v2':
    model = nets.ResNet152v2(inputs)
    bs = 500
    if args.perturb:
      bs //= 3
  else:
    raise ValueError # this should not happen
  
  if bs > nimages:
    bs = nimages
   
  model_pretrained = model.pretrained()
  if args.perturb:
    max_output = tf.reduce_max(model, axis=1) # maximum activation value (=confidence of chosen class)
    odin_perturbations = tf.sign(tf.gradients(max_output, inputs)[0]) # [0] because gradients returns a list
    # this is a bit indirect. tf.gradients returns the gradient of the sum of the confidence scores, 
    # but since each entry depends only on a single image, the gradient of the sum is the same as the
    # gradient of just the entry with respect to its individual images (hopefully)
    # channel_weights = tf.constant([63./255., 62.1/255., 66.7/255.]) 
    # odin_perturbations = image_gradient * channel_weights # auto broadcasting
 
  if args.temperature:
    #model_preds = tf.nn.softmax(tf.log(model)/args.temperature)
    model_preds = tf.nn.softmax(model.logits/args.temperature) # new version of tensornets should support this
  else:
    model_preds = model

  if args.batchsize: 
    bs = args.batchsize # overwrite default batchsize

  nchunks = (nimages+bs-1)//bs
  
  PREDS,CONF,CLASS = [],[],[] # beware: args.rawoutputs keeps all activations (#images x #classes x sizeof(float))

  with tf.Session() as sess:
      sess.run(model_pretrained)
      
      for i in range(nchunks):
          images = []
          if FILENAMES:
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
                  if args.swapRGB:
                      img = apply_swapRGB(img)
                  images.append( img.squeeze() )
              images = model.preprocess(np.asarray(images))
          
          elif args.synth == "Uniform":
              images = np.floor(256*np.random.rand(bs, crop_size, crop_size, 3)) # [0,255]
          elif args.synth == "Gaussian":
              images = (np.random.randn(bs, crop_size, crop_size, 3)+0.5)*256     # mu=128, sigma=256, clipped to [0,255]
              images = np.clip(images, 0, 255)
          elif args.synth == "Binary": 
              images = 255.*np.random.randint(0, 2, size=(bs, crop_size, crop_size, 1)) # {0,255}
              images = np.tile(images, (1,1,3)) # repeat across channels
          else:   # unknown
              print("Unknown synthetic transformation")
              raise SystemExit
          
          target_class = None
          if args.perturb:  # perturb image before running other steps
            image_perturbations = sess.run(odin_perturbations, {inputs: images})
            images += args.perturb*image_perturbations

          preds = sess.run(model_preds, {inputs: images})

          if args.rawoutput:
            PREDS.extend(preds) # store only if needed to save memory
          CONF.extend(np.max(preds,axis=1)) # confidence score
          CLASS.extend(np.argmax(preds,axis=1)) # predicted class
           
          if args.verbose:
            print('Processed chunk {} of {}'.format(i, nchunks))
            print('Most recent prediction:', utils.decode_predictions(preds, top=1)[0])

      if args.output:
          np.savetxt(args.output, CONF, fmt='%.12f')
      if args.rawoutput:
          np.savez_compressed(args.rawoutput, np.asarray(PREDS))
      if args.labeloutput:
          np.savetxt(args.labeloutput, CLASS, fmt='%d')

  if args.verbose:
    print("Done.")

if __name__ == "__main__":
  main()
