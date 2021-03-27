# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 14:57:32 2021

@author: thuli
"""
#######################BLOCK 1#########################
########LOADING IMAGES AND TRANSFORM COLOR SPACE#######
#######################################################

#Temporary example image

import cv2
import numpy as np
import math
import huffman
from collections import Counter
import matplotlib.pyplot as plt
from scipy.fftpack import dct,idct


def ImgToBlocks(img,w=8,h=8):
    xLen = img.shape[1]//h
    yLen = img.shape[0]//h
    blocks = np.zeros((yLen,xLen,h,w),dtype='int16')
    for y in range(yLen):
        for x in range(xLen):
            blocks[y][x]=img[y*h:(y+1)*h,x*w:(x+1)*w]
    return np.array(blocks)
    
def BlocksToImg(blocks):
	xLen=blocks.shape[1];
	yLen=blocks.shape[0];
	h=blocks.shape[2];
	w=blocks.shape[3];
	W=xLen*w
	H=yLen*h
	img = np.zeros((H,W))
	for y in range(yLen):
		for x in range(xLen):
			img[y*h:y*h+h,x*w:x*w+w]=blocks[y][x]
	return img


def plotBlocks(blocks,a,b):
	xLen=blocks.shape[1]
	yLen=blocks.shape[0]
	for y in range(yLen):
		for x in range(xLen):
			plt.subplot(yLen,xLen,1+xLen*y+x)
			plt.imshow(blocks[y][x],cmap=plt.cm.gray, vmin=a, vmax=b)
			plt.axis('off')  
	plt.show()
   

def DCTBlocks(blocks):
	xLen=blocks.shape[1];
	yLen=blocks.shape[0];
	h=blocks.shape[2];
	w=blocks.shape[3];
	output_blocks=np.zeros((yLen,xLen,h,w));
	for y in range(yLen):
		for x in range(xLen):
			d = np.zeros((h,w))
			block=blocks[y][x][:,:]
			d[:,:]=dct(dct(block.T, norm = 'ortho').T, norm = 'ortho')
			output_blocks[y][x]=d
	return output_blocks;

def IDCTBlocks(blocks):
	xLen=blocks.shape[1];
	yLen=blocks.shape[0];
	h=blocks.shape[2];
	w=blocks.shape[3];
	output_blocks=np.zeros((yLen,xLen,h,w));
	for y in range(yLen):
		for x in range(xLen):
			d = np.zeros((h,w))
			block=blocks[y][x][:,:]
			d[:,:]=idct(idct(block.T, norm = 'ortho').T, norm = 'ortho')
			d=d.round().astype('int16')
			output_blocks[y][x]=d
	return output_blocks;

def BlockQuantization(blocks,quantization_ratio):
    h=blocks.shape[2];
    w=blocks.shape[3];
    QY=np.array([[16,11,10,16,24,40,51,61],
     [12,12,14,19,26,58,60,55],
     [14,13,16,24,40,57,69,56],
     [14,17,22,29,51,87,80,62],
     [18,22,37,56,68,109,103,77],
     [24,35,55,64,81,104,113,92],
     [49,64,78,87,103,121,120,101],
     [72,92,95,98,112,100,103,99]])
    QY=QY[:w,:h]
    quantized_blocks=np.copy(blocks);
    #print(QY.shape)
    Q3 = QY*quantization_ratio
    Q3=Q3
    #print(Q3.shape)
    quantized_blocks=(quantized_blocks/Q3).round().astype('int16') 
    return  quantized_blocks,Q3;

#Block for Zigzag transform
def ZigZag(block): 
    h=block.shape[0];
    w=block.shape[1];
    lines=[[] for i in range(h+w-1)] 
    for y in range(h): 
        for x in range(w): 
            i=y+x 
            if(i%2 ==0): 
                lines[i].insert(0,block[y][x]) 
            else:  
                lines[i].append(block[y][x]) 
    return np.array([coefficient for line in lines for coefficient in line])

def InvZigZag(zigZagArr,w=8,h=8): 
	gaps=[i for i in range(1,8)]+[8-i for i in range(8)]+[-1]
	locations=[[int(sum(range(gaps[i-1]+1))),sum(range(gaps[i]+1))] if gaps[i]>gaps[i-1]  else [64-sum(range(gaps[i-1])),64-sum(range(gaps[i]))] for i in range(len(gaps)-1)]
	block=np.zeros((h,w),dtype='int16')
	zigZagArr=[zigZagArr[l[0]:l[1]] for l in locations]
	for y in range(h): 
		for x in range(w): 
			i=y+x 
			if(i%2 != 0): 
				block[y][x]=zigZagArr[i][0]
				zigZagArr[i]=zigZagArr[i][1:]
			else: 
				block[y][x]=zigZagArr[i][-1:][0]
				zigZagArr[i]=zigZagArr[i][:-1]
	#print(block.shape)
	return block


#Huffman Coding Block
def HuffmanCodingOneArray(zigZagArr,bitBits=3,runBits=1):
    rbBits=runBits+bitBits
    rbCount=[]
    run=0
    for AC in zigZagArr[1:]:
        if(AC!=0):
            AC=max(AC,1-2**(2**bitBits-1)) if AC<0 else min(AC,2**(2**bitBits-1)-1)
            if(run>2**runBits-1):
                runGap=2**runBits
                k=run//runGap
                for i in range(k):
                    rbCount.append('1'*runBits+'0'*bitBits)
                run-=k*runGap
            run=min(run,2**runBits-1) 
            bitSize=min(int(np.ceil(np.log(np.abs(AC)+0.000000001)/np.log(2))),2**bitBits-1)
            rbCount.append(format(run<<bitBits|bitSize,'0'+str(rbBits)+'b'))
            run=0
        else:
            run+=1
    rbCount.append("0"*(rbBits))
    return Counter(rbCount);
    
def HuffmanCodingWholeImg(blocks):
	xLen=blocks.shape[1];
	yLen=blocks.shape[0];
	rbCount=np.zeros(xLen*yLen,dtype=Counter)
	zz=np.zeros(xLen*yLen,dtype=object)
	for y in range(yLen):
		for x in range(xLen):
			zz[y*xLen+x]=ZigZag(blocks[y, x,:,:])
			rbCount[y*xLen+x]=HuffmanCodingOneArray(zz[y*xLen+x])
	return np.sum(rbCount),zz

#Runlength
def RunLength(zigZagArr,lastDC,hfm,bitBits=3,runBits=1):
    rbBits=runBits+bitBits;
    rlc=[]
    run=0
    newDC=min(zigZagArr[0],2**(2**bitBits-1))
    DC=newDC-lastDC
    bitSize=max(0,min(int(np.ceil(np.log(np.abs(DC)+0.000000001)/np.log(2))),2**bitBits-1))
    code=format(bitSize, '0'+str(bitBits)+'b')
    if (bitSize>0):
        code+=(format(DC,"b") if DC>0 else ''.join([str((int(b)^1)) for b in format(abs(DC),"b")]))
    for AC in zigZagArr[1:]:
        if(AC!=0):
            AC=max(AC,1-2**(2**bitBits-1)) if AC<0 else min(AC,2**(2**bitBits-1)-1)
            if(run>2**runBits-1):
                runGap=2**runBits
                k=run//runGap
                for i in range(k):
                    code+=  hfm['1'*runBits+'0'*bitBits]
                run-=k*runGap
            run=min(run,2**runBits-1) 
            bitSize=min(int(np.ceil(np.log(np.abs(AC)+0.000000001)/np.log(2))),2**bitBits-1)
            rb= hfm[format(run<<bitBits|bitSize,'0'+str(rbBits)+'b')]
            code+=rb+(format(AC,"b") if AC>=0 else ''.join([str((int(b)^1)) for b in format(abs(AC),"b")]))
            run=0
        else:
            run+=1
    code+= hfm["0"*(rbBits)]#end
    return code,newDC;

def RunLengthToBytes(code):
    return bytes([len(code)%8]+[int(code[i:i+8],2) for i in range(0, len(code), 8)])
def BytesToRunLength(bytes):
    return "".join([format(i,'08b') for i in list(bytes)][1:-1 if list(bytes)[-1]!=0 else None])+(format(list(bytes)[-1],'0'+str(list(bytes)[0])+'b')if list(bytes)[-1]!=0 else"")


#Saving block
def SavingQuantizedDctBlocks(blocks,bitBits=3,runBits=1):
	xLen=blocks.shape[1];
	yLen=blocks.shape[0];
	rbCount,zigZag=HuffmanCodingWholeImg(blocks)
	hfm=huffman.codebook(rbCount.items())
	sortedHfm=[[hfm[i[0]],i[0]] for i in rbCount.most_common()]
	code=""
	DC=0
	for y in range(yLen):
		for x in range(xLen):
			codeNew,DC=RunLength(zigZag[y*xLen+x],DC,hfm )
			code+=codeNew
	savedImg=RunLengthToBytes(code)
	print("Compression image size: %.3f MB"%(len(savedImg)/2**20))
	return bytes([int(format(xLen,'012b')[:8],2),int(format(xLen,'012b')[8:]+format(yLen,'012b')[:4],2),int(format(yLen,'012b')[4:],2)])+savedImg,sortedHfm

#test

#Loading Block

def LoadingQuantizedDctBlocks(loadedbytes,sortedHfm,runBits=1,bitBits=3,w=8,h=8):
	rbBits=bitBits+runBits
	runMax=2**runBits-1
	xLen=int(format(loadedbytes[0],'b')+format(loadedbytes[1],'08b')[:4],2)
	yLen=int(format(loadedbytes[1],'08b')[4:]+format(loadedbytes[2],'08b'),2)
	code=BytesToRunLength(loadedbytes[3:])
	blocks = np.zeros((yLen,xLen,h,w),dtype='int16')
	lastDC=0
	rbBitsTmp=rbBits
	rbTmp=""
	cursor=0 #don't use code=code[index:] to remove readed strings when len(String) is large like 1,000,000. It will be extremely slow
	for y in range(yLen):
		for x in range(xLen):
			zz=np.zeros(64)
			bitSize=int(code[cursor:cursor+bitBits],2)
			DC=code[cursor+bitBits:cursor+bitBits+bitSize]
			DC=(int(DC,2) if DC[0]=="1" else -int(''.join([str((int(b)^1)) for b in DC]),2)) if bitSize>0 else 0
			cursor+=(bitBits+bitSize)
			zz[0]=DC+lastDC
			lastDC=zz[0]
			r=1
			while(True):
				if(sortedHfm!=None):
					for ii in sortedHfm:
						if (ii[0]==code[cursor:cursor+len(ii[0])]):
							rbTmp=ii[1]
							rbBitsTmp=len(ii[0])
							break
					run=int(rbTmp[:runBits],2)
					bitSize=int(rbTmp[runBits:],2)
				else:
					run=int(code[cursor:cursor+runBits],2)
					bitSize=int(code[cursor+runBits:cursor+rbBitsTmp],2)
				if (bitSize==0):
					cursor+=rbBitsTmp
					if(run==runMax):
						r+=(run+1)
						continue
					else:
						break
				coefficient=code[cursor+rbBitsTmp:cursor+rbBitsTmp+bitSize]
				if(coefficient[0]=="0"):
					coefficient=-int(''.join([str((int(b)^1)) for b in coefficient]),2)
				else:
					coefficient=int(coefficient,2)
				zz[r+run]=coefficient
				r+=(run+1)
				cursor+=rbBitsTmp+bitSize    
			blocks[y,x]=InvZigZag(zz)
	return blocks
