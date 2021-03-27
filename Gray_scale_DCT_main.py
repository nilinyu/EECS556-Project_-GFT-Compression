from DCT_compression_functions import *

import cv2
import numpy as np
import math
import huffman
from collections import Counter
import matplotlib.pyplot as plt
from scipy.fftpack import dct,idct
import scipy.io
from mpl_toolkits.axes_grid1 import make_axes_locatable

mat = scipy.io.loadmat('C:/Users/nilin/Desktop/EECS556/project/EECS556-Project_-GFT-Compression-main/EECS556-Project_-GFT-Compression-main/img/Peppers/class1.mat')
#img=mat.get('img')
img=mat.get('Imgg')
fig, ax = plt.subplots(3)
ax[0].imshow(img,cmap=plt.cm.gray)
ax[0].title.set_text("Original Image")
blocks=ImgToBlocks(img,w=8,h=8)
xLen=blocks.shape[1]
yLen=blocks.shape[0]

#print(xLen)
#x1=int(0.4*xLen)#x1,x2,y1,y2 is location of the face
#y1=int(0.4*yLen)
#plt.figure(figsize=(5,5))
#plotBlocks(blocks[10:30,10:30],0,255)

DCTblocks=DCTBlocks(blocks)
IDCTblocks=IDCTBlocks(DCTblocks)
#plotBlocks(IDCTblocks[10:30,10:30],0,255)
q_dct_blocks,Q3=BlockQuantization(DCTblocks,1);
compressed_img,sortedHfm=SavingQuantizedDctBlocks(q_dct_blocks)
save = open("compressed_img_new_2.bin", "wb")
save.write(compressed_img);
save.close();

load=open("compressed_img_new_2.bin",'rb');
loadedbytes=load.read()
LoadedBlocks=LoadingQuantizedDctBlocks(loadedbytes,sortedHfm);
recon_blocks=IDCTBlocks(LoadedBlocks*Q3);
print(recon_blocks.shape)
recon_img=BlocksToImg(recon_blocks)
ax[1].imshow(recon_img,cmap=plt.cm.gray)
ax[1].title.set_text("Reconstructed Image")
divider = make_axes_locatable(ax[2])
cax = divider.append_axes('right', size='5%', pad=0.05)
im=ax[2].imshow(recon_img-img,cmap=plt.cm.gray)
ax[2].title.set_text("Error")
plt.colorbar(im, cax=cax, orientation='vertical')
plt.show()
