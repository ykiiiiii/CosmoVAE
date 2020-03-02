# CosmoVAE
This is a Keras implementation of CosmoVAE for the task of inpainting the Cosmic Microwave Background, as described in our paper:https://arxiv.org/abs/2001.11651





The code is testing on Keras 2.2.5
## Abstract
Cosmic microwave background radiation (CMB) is critical to the understanding of the early universe and precise estimation of cosmological constants. Due to the contamination of thermal dust noise in the galaxy, the CMB map that is an image on the two-dimensional sphere has missing observations, mainly concentrated on the equatorial region. The noise of the CMB map has a significant impact on the estimation precision for cosmological parameters. Inpainting the CMB map can effectively reduce the uncertainty of parametric estimation. In this paper, we propose a deep learning-based variational autoencoder --- CosmoVAE, to restoring the missing observations of the CMB map. The input and output of CosmoVAE are square images. To generate training, validation, and test data sets, we segment the full-sky CMB map into many small images by Cartesian projection. CosmoVAE assigns physical quantities to the parameters of the VAE network by using the angular power spectrum of the Gaussian random field as latent variables. CosmoVAE adopts a new loss function to improve the learning performance of the model, which consists of $\ell_1$ reconstruction loss, Kullback-Leibler divergence between the posterior distribution of encoder network and the prior distribution of latent variables, perceptual loss, and total-variation regularizer. The proposed model achieves state of the art performance for Planck \texttt{Commander} 2018 CMB map inpainting.
## Prerequisites

In order to generate dataset, one need to download the archive file from Planck Legacy Archive: http://pla.esac.esa.int/pla/#home, then go to “MAPS” -> “CMB maps” and “MAPS” -> “Masks” -> “CMB”.  In our paper, we use “\texttt{Commander}  full Mission CMB map”  and “2018 Component Separation Inpainting Common mask in Intensity”. Make sure these two files located in the "data" folder.

One also need to download the weight of Vgg-16 model from: [VGG 16 weights](https://drive.google.com/open?id=1HOzmKQFljTdKWftEP-kWD7p2paEaeHM0) for training and also put it into "data" folder.

## Generate dataset
Make sure two archive files from Planck Legacy Archive are located in "./data"
```
python generate_dataset.py
```

## Train
```
python train.py
```
## Prediction
There are two parameters for the prediction program.
```
weight_dir   -  the path and name of your weight 
index         -  the index of test image 
```
For example, you can run
```
python prediction.py--weight_dir './2weights.09-2.05.h5' \
                    --index 0
```
Our trained weights can be downloaded [here](https://drive.google.com/file/d/1WEPpDZNWHFt708ssoVervP-xlFr4nQ-j/view?usp=sharing) 

<img src="https://github.com/ykiiiiii/CosmoVAE/blob/master/data/img_2.png" alt="acd_commander15" width="800">


## Cite
Please cite our paper if you use this code in your own work:
```
@article{yi2020cosmovae,
  title={CosmoVAE: Variational Autoencoder for CMB Image Inpainting},
  author={Yi, Kai and Guo, Yi and Fan, Yanan and Hamann, Jan and Wang, Yu Guang},
  journal={arXiv preprint arXiv:2001.11651},
  year={2020}
}
```
