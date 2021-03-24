# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 21:24:49 2021

@author: thuli
"""

import cv2
import numpy as np
import math
import huffman
from collections import Counter
import matplotlib.pyplot as plt
from scipy.fftpack import dct,idct
from DCT_compression_functions import *
#Load Img
image_path="4.2.06.tiff";
img=cv2.imread(image_path);

#Img Compression

blocks=ImgToBlocks(img);
dct_blocks=DCTBlocks(blocks);
q_dct_blocks,Q3=BlockQuantization(dct_blocks,1);
compressed_img,sortedHfm=SavingQuantizedDctBlocks(q_dct_blocks)
save = open("compressed_img_new_2.bin", "wb")
save.write(compressed_img);
save.close();

#Decompression image
load=open('compressed_img_new.bin','rb');
loadedbytes=load.read()
LoadedBlocks=LoadingQuantizedDctBlocks(loadedbytes,sortedHfm);
recon_blocks=IDCTBlocks(LoadedBlocks*Q3);
recon_img=BlocksToImg(recon_blocks)
plt.figure(figsize=(20,20))
plt.imshow(recon_img.astype('uint8'))
plt.figure(figsize=(25,25))
plt.imshow(img.astype('uint8'))