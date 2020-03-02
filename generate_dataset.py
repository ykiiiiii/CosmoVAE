#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: kai yi

"""
from __future__ import division
import numpy as np
import matplotlib
import os
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from optparse import OptionParser
from libs.Map_Cutter import MapCutter
from sklearn.model_selection import train_test_split
parser = OptionParser()
parser.add_option("--sv_dir",
                  dest="sv_dir", default='datasets',
                  help="output file")

parser.add_option("--map_loc",
                  dest="dustmap_loc", default='data/COM_CMB_IQU-commander_2048_R3.00_full.fits',
                  help="the path and file name of CMB Commander archive file ")

parser.add_option("--mask_loc",
                  dest="map_loc", default='data/COM_Mask_CMB-Inpainting-Mask-Int_2048_R3.00.fits',
                  help="the path and file name of CMB mask archive file ")

parser.add_option("--cropping_step",
                  dest="cropping_step", default=2,
                  help="the step of cropping full-sky CMB map when generate the dataset"
                  +"if you want to generate a larger dataset , you can make this number smaller"
                  )

options, args = parser.parse_args()
MP = MapCutter(options.dustmap_loc)
MP_mask = MapCutter(options.map_loc)
map_cuts_train,map_cuts_test,mask_cuts, dic= [],[],[],{}

lat_range=list(range(-20,25,int(options.cropping_step)))
for j, theta in enumerate(lat_range):
    lon_range = np.arange(0, 360, int(options.cropping_step )/ np.cos(theta / 180. * np.pi))
    for i, phi in enumerate(lon_range):
        map_cut = MP.cut_map([phi, theta],res=400)
        mask_cut = MP_mask.cut_map([phi, theta],res=400)
        if (400**2 - mask_cut.sum() > 10890):
            map_cuts_test.append(map_cut)
            dic[str(theta)+'-'+str(round(phi,2))] = map_cut
            mask_cuts.append(mask_cut)
            dic[str(theta)+'-'+str(round(phi,2))+'mask'] = mask_cut
        else:
            map_cuts_train.append(map_cut)
            dic[str(theta)+'-'+str(round(phi,2)) ] = map_cut

def triplec(mask):
    a = np.zeros((400,400,3))
    a[:,:,0] = mask
    a[:,:,1] = mask
    a[:,:,2] = mask
    return a
X_train, X_valid = train_test_split(np.asarray(map_cuts_train) , test_size=0.1, random_state=42)

colombi1_cmap = ListedColormap(np.loadtxt("./data/Planck_Parchment_RGB.txt")/255.)
colombi1_cmap.set_bad("gray") # color of missing pixels
colombi1_cmap.set_under("white") # color of background, necessary if you want to use
cmap = colombi1_cmap
sv_dir = options.sv_dir
print('start saving image')
for  i ,image in enumerate(X_train):
    if not os.path.exists(sv_dir+'/train/data/'):
        os.makedirs(sv_dir+'/train/data/')
    sv  = sv_dir +'/train/data/'+ str(i)+'pict.png'
    fig=plt.imsave(sv,image,cmap=cmap)

for i ,image in enumerate(X_valid):
    if not os.path.exists(sv_dir+'/valid/data/'):
        os.makedirs(sv_dir+'/valid/data/')
    sv  = sv_dir +'/valid/data/'+ str(i)+'pict.png'
    fig=plt.imsave(sv,image,cmap=cmap)
print('start saving test data')

for i ,image in enumerate(map_cuts_test):
    if not os.path.exists(sv_dir+'/test/data/'):
        os.makedirs(sv_dir+'/test/data/')
    sv  = sv_dir +'/test/data/'+ str(i)+'pict.png'
    fig=plt.imsave(sv,image,cmap=cmap)
for i ,image in enumerate(mask_cuts):
    if not os.path.exists(sv_dir+'/mask/data/'):
        os.makedirs(sv_dir+'/mask/data/')
    sv  = sv_dir +'/mask/data/'+ str(i)+'pict.png'
    fig=plt.imsave(sv,triplec(image))

