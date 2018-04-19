# KS(conf)
A Light-Weight Test if a ConvNet Operates Outside of Its Specifications

Code and example data for KS(conf) test and baselines.
For examples of use, see **code/runtests.py**

Additional data files with confidence scores of synthetic image 
manipulations can be downloaded from 

* https://cvml.ist.ac.at/data/KSconf-MobileNet25-synth.zip  (27MB)
* https://cvml.ist.ac.at/data/KSconf-NASNetAlarge-synth.zip   (26MB) 
* https://cvml.ist.ac.at/data/KSconf-ResNet50-synth.zip	  (27MB)
* https://cvml.ist.ac.at/data/KSconf-SqueezeNet-synth.zip (27MB)
* https://cvml.ist.ac.at/data/KSconf-VGG19-synth.zip  (27MB)

Each file contains the confidence scores for the following 78 synthetic manipulations of the ILSVRC 'test' images:

* **blur**: Gaussian blur, sigma={1,2,3,4,5,6,7,8,9,10,11}
* **noise**: Gaussian additive noise, sigma={5,10,15,20,30,50,100}
* **dead**: Dead pixels, percentage={1,2,3,4,5,6,7,8,9,10,15,20,25,30,40,50,60,70,80,90,100}
* **dark**: Reduced brightness, factor={1,2,3,4,5,6,7,8,9,10,15,20,30,40,50,100}
* **bright**: Increased brightness, factor={1,2,3,4,5,6,7,8,9,10,15,20,30,40,50,100}
* **geo**: Geometric transformation: {0=none, 1=flipH, 2=flipV, 3=rotate90, 4=rotate180, 5=rotate270}
* **bgr**: Swapped color channels, RGB<->BGR

## Reference:
[* R. Sun, C.H. Lampert, "KS(conf) A Light-Weight Test if a ConvNet Operates Outside of Its Specifications", arXiv:1804.04171 [stat.ML]](https://arxiv.org/abs/1804.04171)
