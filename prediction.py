#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 15:29:56 2020

@author: kai yi
"""

from libs.pconv_model import PConvUnet
from optparse import OptionParser
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from copy import deepcopy
import os
parser = OptionParser()
parser.add_option("--weight_dir",
                  dest="weight", default='./',
                  help="output file")

parser.add_option("--index",
                  dest="id", default=0,
                  help="the path and file name of CMB Commander archive file ")

options, args = parser.parse_args()

def complete_image(pred,true,mask):
    com_image = np.zeros((400,400,3))
    for i in range(400):
      for j in range(400):
        if mask[i,j,0] == 1:#vaild pixel
          com_image[i,j,0] = true[i,j,0]
          com_image[i,j,1] = true[i,j,1]
          com_image[i,j,2] = true[i,j,2]
        else:#hole pixel
          com_image[i,j,0] = pred[i,j,0]
          com_image[i,j,1] = pred[i,j,1]
          com_image[i,j,2] = pred[i,j,2]
    return com_image

weight  = options.weight
model = PConvUnet(vgg_weights=None, inference_only=True)
model.load(weight, train_bn=False)
imid = int(options.id)
masks = mpimg.imread('./datasets/mask/data/'+str(imid)+'pict.png')[:,:,:3]
Planck_Image = mpimg.imread('./datasets/test/data/'+str(imid)+'pict.png')[:,:,:3]
CosmoVae = complete_image(model.predict([np.expand_dims(Planck_Image,0), np.expand_dims(masks,0)])[0],Planck_Image,masks)
masked = deepcopy(Planck_Image)
masked[masks==0] = 1
if not os.path.exists('./datasets/predicted/'):
        os.makedirs('./datasets/predicted/')
fig=plt.imsave('./datasets/predicted/'+str(imid)+'pred.png',CosmoVae)

#_, axes = plt.subplots(1, 3, figsize=(20, 5))
#axes[0].imshow(masks)
#axes[1].imshow(masked)
#axes[2].imshow(CosmoVae)
#axes[0].set_title('Mask')
#axes[1].set_title('Masked Image')
#axes[2].set_title('Predicted Image')
#        
#plt.savefig(r'dataset/predicted/predicted_{imid}_img.png')
#plt.close()