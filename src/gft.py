import cvxpy as cp
import numpy as np
import huffman
import collections
from bitarray import bitarray
import math
import matplotlib.pyplot as plt
import statistics
import joblib
from joblib import Parallel, delayed

###
# Utilities, no sanity check in most cases
###
def graph_edge(width,height):
	edge=[]
	for i in range(width):
		for j in range(height):
			if (j+1)%width==0:
				edge.append([j+1+i*width,height+j+1+i*width])
			elif (i+1)==width:
				edge.append([j+1+i*width,j+1+i*width+1])
			else:
				edge.append([j+1+i*width,j+1+i*width+1])
				edge.append([j+1+i*width,j+1+i*width+height])
	npedge=np.array(edge)
	edge_sort = np.delete(npedge, npedge.shape[0]-1, 0)
	return edge_sort

def get_M(width,height):
    # Compute the number of edges in the 4 neighbor graph setup
    return ((2*width-1)*(2*height-1)-1)//2

def get_incidence_matrix(width,height):
    # Incidence matrix B of the graph G, which is designed to be a 4 neighbor graph
	N = width*height # number of nodes
	M = ((2*width-1)*(2*height-1)-1)//2 # number of edges
	B = np.zeros((N,M))
	edge=graph_edge(width,height)
	for i in range(N):
		pos=np.where(edge[0:,0] == i+1)
		pos=np.array(pos)
		B[i,pos[0,0:]]=1
		pos=np.where(edge[0:,1] == i+1)
		pos=np.array(pos)
		B[i,pos[0,0:]]=-1
	return B

def get_M_diag(M):
    M_diag = np.zeros((M**2,M))
    for i in range(M):
        M_diag[i*M+i, i] = 1
    return M_diag

def eig_decompose(A):
    eigenValues, eigenVectors = np.linalg.eig(A)
    idx = np.argsort(-abs(eigenValues))
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    return (eigenValues, eigenVectors)

def get_psi_d(width, height):
    # W is the adjacency matrix of the dual graph, constructed based on the rule that 2 nodes are connected iff. their corresponding edges in the original graph share an endpoint
	M = get_M(width, height)
	W = np.zeros((M,M))
	edge=graph_edge(width,height)
	for i in range(M):
		for j in range(M):
			temp = [val for val in edge[i,0:] if val in edge[j,0:]]
			if 2>len(temp)>0:
				W[i,j] = W[j,i] = 1
	# D is the degree matrix of the dual graph. Since it is an unweighted graph, D can be derived from D fairly easily
	D = np.diag(W.sum(axis=0))
	L = D - W
	_, psi_d = eig_decompose(L)
	return psi_d

def cover_multiple(current_length, multiple):
    return ((current_length - 1) // multiple + 1) * multiple

def slicer(a, chunk_i, chunk_j, two_d=True):
    n = cover_multiple(a.shape[0], chunk_i)
    m = cover_multiple(a.shape[1], chunk_j)
    c = np.empty((n, m))
    c.fill(0.0)
    c[:a.shape[0], :a.shape[1]] = a
    c = c.reshape(n // chunk_i, chunk_i, m // chunk_j, chunk_j)
    c = c.transpose(0, 2, 1, 3)
    if not two_d:
        c = c.reshape(-1, chunk_i, chunk_j)
    return c
    
def BlocksToImg(u,w=16,h=16,xLen=32,yLen=32):
	W=xLen*w
	H=yLen*h
	#print(W,H)
	img = np.zeros((H,W))
	for y in range(yLen):
		for x in range(xLen):
			if yLen==1:
				data=abs(u[:])
			else:
				data=u[y+x*yLen,:]
			#print(data)
			img[y*h:y*h+h,x*w:x*w+w]=data.reshape(w,h)
	return img
    
def BlocksToImgF(blocks,w=16,h=16,xLen=32,yLen=32):
	W=xLen*w
	H=yLen*h
	img = np.zeros((H,W))
	for x in range(xLen):
		for y in range(yLen):
			img[x*h:x*h+h,y*w:y*w+w]=blocks[x*xLen+y,:,:]
	return img

###
# Functions that you really called
###

def convex_optimize(u, width, height=None, alpha=100, beta=1):
	if height is None:
		height = width
	M = get_M(width, height)
	B = get_incidence_matrix(width, height)
	M_diag = get_M_diag(M)
	psi_d = get_psi_d(width, height)
	w = cp.Variable(M) # The weight of each edge in a vector. With B fixed, the graph G can be uniquely determined by it. 
	# Optimization summary: constant array inner product with w + constant times l1 norm of contant matrix times w - something similar to l1 norm
	step1=u.T@B
	step2=np.dot(u[:, None], step1[None, :])
	prob = cp.Problem(cp.Minimize(((B.T@step2).reshape(-1)).T@(M_diag@w)+ alpha*cp.norm(psi_d.T@w,1) - beta*np.ones(M)@cp.log(w)),[w <= np.ones(M)])

	prob.solve()
	W_hat = np.diag(w.value)
	L = B@W_hat@B.T # The incidence matrix definition of graph Laplacian
	_, psi = eig_decompose(L) # psi is the graph Fourier transformation matrix
	u_hat = psi.T@u # The coefficients in the Fourier domain. Theoretical one not the quantized one
	return w.value, u_hat

def quantize(w, width, height, step_size, M_thresh):
    psi_d = get_psi_d(width, height)
    w_hat = psi_d.T@w
    w_hat_r = w_hat.real
    w_hat_r[abs(w_hat_r).argsort()[:-M_thresh]]=0
    w_hat_r_quant = (w_hat_r/step_size).round()
    return w_hat_r_quant

def reconstruct(u_hat, w_hat_quant, step_size, width, height):
    psi_d = get_psi_d(width, height)
    w_hat = w_hat_quant*step_size
    w = psi_d@w_hat
    W_hat = np.diag(w)
    B = get_incidence_matrix(width, height)
    L = B@W_hat@B.T
    _, psi = eig_decompose(L)
    u = psi@u_hat
    
    return u

def entropy_coding(w_hat_quant):
    w_list = w_hat_quant.tolist()
    word_cnt_tb = collections.Counter(w_list)
    code_book = huffman.codebook(word_cnt_tb.items())
    w_code = ''.join([code_book[k] for k in w_list])
    return w_code, code_book
    
def compute_distortion(u, u_recon):
    return np.linalg.norm(u - u_recon)

def argmin(iterable):
    return min(enumerate(iterable), key=lambda x: x[1])[0]


###
# Driver functions (non-parallel)
###


def gft_encode(img, alpha,beta):
	step_size_list = [0.001]#[0.1,0.05,0.01,0.005,0.001,0.0005,0.0001]
	blk_height = blk_width = 16
	gamma = 1
	M_thresh = 64 # 64 for real?

	blocks = slicer(img, blk_height, blk_width)
	num_row, num_col, _, _ = blocks.shape
    
	opt_rate_distortion = np.inf

	rst_code = None
	rst_codbook = None
	rst_step_size = None

	for idx, step_size in enumerate(step_size_list):
		rst_w = []
		rst_u = []
		meanu=[]
		rate_distortion = 0
		s=None
		for row in range(num_row):
			for col in range(num_col):
				block = blocks[row,col]
				u = block.flatten()
				mu=statistics.mean(u)
				w, u_hat = convex_optimize(u, blk_width, blk_height, alpha=alpha, beta=beta)         
				w_hat_quant = quantize(w, blk_width, blk_height, step_size, M_thresh)
				u_recon = reconstruct(u_hat, w_hat_quant, step_size, blk_width, blk_height)
				mur=statistics.mean(u_recon.real)
				dmean=mu-mur
				u_recon=dmean+u_recon.real
				#imgg=BlocksToImg(u_recon,w=16,h=16,xLen=1,yLen=1)
				#plt.show(imgg)
				#plt.show()
					
				#plt.plot(u)
				#plt.plot(u_recon)
				#plt.show()
   
				distortion = ((u-u_recon)**2).sum()
				rate_distortion += distortion
				rst_w.append(w_hat_quant)
				rst_u.append(u_hat)
				meanu.append(dmean)
				#print(row, col)
		#imgg=BlocksToImg(np.array(s),w=16,h=16,xLen=2,yLen=2)
		#plt.imshow(imgg)
		#plt.show()
        
		w_code, codebook = entropy_coding(np.concatenate(rst_w))
		rate = len(w_code)/(img.size*8)
		rate_distortion += gamma*rate

		if rate_distortion < opt_rate_distortion:
			rst_code = (w_code, rst_u)
			rst_codebook = codebook
			rst_step_size = idx
		return rst_code, rst_codebook, rst_step_size, meanu# code book is not compressed yet, should be the same as DCT

def gft_decode(code,codebook,step_size,dmean):
	step_size_list = [0.001]#[0.1,0.05,0.01,0.005,0.001,0.0005,0.0001]
	step_size=step_size_list[step_size]
	blk_width=blk_height=16
	codebook_formatted={k:bitarray(v)for k,v in codebook.items()}
	w_code,u_list=code
	decoded=np.array(bitarray(w_code).decode(codebook_formatted))
	w_list=np.split(decoded,decoded.size//get_M(blk_width,blk_height))
	blocks=[]
	for w,u,meanu in zip(w_list,u_list,dmean):
		u_recon=reconstruct(u,w,step_size,blk_width,blk_height)
		#mur=statistics.mean(u_recon.real)
		u_recon=meanu+u_recon
		blocks.append(u_recon.reshape(blk_width,blk_height))
	return blocks

###
# Drive function (parallel encode)
###

def gft_encode_p_blk(u, blk_width, blk_height, alpha, beta, step_size, M_thresh):
    mu = statistics.mean(u)
    w, u_hat = convex_optimize(u, blk_width, blk_height, alpha=alpha, beta=beta)         
    w_hat_quant = quantize(w, blk_width, blk_height, step_size, M_thresh)
    u_recon = reconstruct(u_hat, w_hat_quant, step_size, blk_width, blk_height)
    mur = statistics.mean(u_recon.real)
    dmean = mu-mur
    return {"distortion":((u-u_recon)**2).sum(), "w_hat_quant":w_hat_quant, "u_hat":u_hat, "meanu":dmean}

def gft_encode_p(img, alpha, beta):
    step_size_list = [0.001]#[0.1,0.05,0.01,0.005,0.001,0.0005,0.0001]
    blk_height = blk_width = 16
    gamma = 1
    M_thresh = 64 # 64 for real?
    
    number_of_cpu = joblib.cpu_count()
    parallel_pool = Parallel(n_jobs=number_of_cpu)

    blocks = slicer(img, blk_height, blk_width)
    num_row, num_col, _, _ = blocks.shape
    
    opt_rate_distortion = np.inf

    rst_code = None
    rst_codbook = None
    rst_step_size = None

    for idx, step_size in enumerate(step_size_list):
        u_list = [blocks[row, col].flatten() for row in range(num_row) for col in range(num_col)]
        delayed_funcs = [delayed(gft_encode_p_blk)(u_list[i], blk_width, blk_height, alpha, beta, step_size, M_thresh) for i in range(num_row * num_col)]
        rst = parallel_pool(delayed_funcs)
        rate_distortion = sum([i['distortion'] for i in rst])
        rst_w = [i['w_hat_quant'] for i in rst]
        rst_u = [i['u_hat'] for i in rst]
        meanu = [i['meanu'] for i in rst]

        w_code, codebook = entropy_coding(np.concatenate(rst_w))
        rate = len(w_code)/(img.size*8)
        rate_distortion += gamma*rate

        if rate_distortion < opt_rate_distortion:
            rst_code = (w_code, rst_u)
            rst_codebook = codebook
            rst_step_size = idx
        return rst_code, rst_codebook, rst_step_size, meanu# code book is not compressed yet, should be the same as DCT
