import cvxpy as cp
import numpy as np
import huffman
import collections

###
# Utilities, no sanity check in most cases
###
def edge_idx_parse(idx, width, height):
    quo, rem = divmod(idx, width+height-2)
    if rem < width - 1: # horizontal
        return (quo*width + rem, quo*width + rem + 1)
    rem = rem - (width-1)
    return (quo*width + rem, quo*width + rem + width)

def edge_name_parse(src, dest, width, height):
    if dest == src + 1:
        return src//width * (height+width-2) + src%(width - 1)
    if dest == src + width:
        return src//width * (height+width-2) + width - 1 + src%width
    return None

def get_M(width,height):
    # Compute the number of edges in the 4 neighbor graph setup
    return ((2*width-1)*(2*height-1)-1)//2

def get_incidence_matrix(width,height):
    # Incidence matrix B of the graph G, which is designed to be a 4 neighbor graph
    N = width*height # number of nodes
    M = ((2*width-1)*(2*height-1)-1)//2 # number of edges
    B = np.zeros((N,M), int)
    for i in range(N):
        if (i+1)%width:
            B[i, edge_name_parse(i, i+1, width, height)] = 1
            B[i+1, edge_name_parse(i, i+1, width, height)] = -1
        if (i + width) < N:
            B[i, edge_name_parse(i, i+width, width, height)] = 1
            B[i+width, edge_name_parse(i, i+width, width, height)] = -1
    return B

def get_M_diag(M):
    M_diag = np.zeros((M**2,M), int)
    for i in range(M):
        M_diag[i*M+i, i] = 1
    return M_diag

def eig_decompose(A):
    eigenValues, eigenVectors = np.linalg.eig(A)
    idx = np.argsort(-eigenValues)
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:,idx]
    return (eigenValues, eigenVectors)

def get_psi_d(width, height):
    # W is the adjacency matrix of the dual graph, constructed based on the rule that 2 nodes are connected iff. their corresponding edges in the original graph share an endpoint
    M = get_M(width, height)
    W = np.zeros((M,M), int)
    for i in range(M):
        for j in range(M):
            a,b = edge_idx_parse(i, width, height)
            c,d = edge_idx_parse(j, width, height)
            if i != j and (a==c or b==c or a==d or b==d):
                W[i,j] = 1
                W[j,i] = 1
    # D is the degree matrix of the dual graph. Since it is an unweighted graph, D can be derived from D fairly easily
    D = np.diag(W.sum(axis=0))
    L = D - W
    _, psi_d = eig_decompose(L)
    return psi_d


###
# Functions that you really called
###

def convex_optimize(u, width, height=None, alpha=100, beta=1):
    """
    The convex optimization problem is to minimize

    vec(B'uu'B)'M_{diag}w + alpha||psi_d'w||_1 - beta1'log(w)
    s.t. w<=1

    To clarify:
    u: input signal
    B: N*M incidence matrix of the graph that holds the input signal
        The graph is a fixed grid graph, s.t. each node is connected
        to its 4 neighbors.
    M_diag: matrix that turns a vector v into vec(diag(v))
    psi_d: eigenvectors of dual graph
    alpha: based on smoothness of the block
    beta: 1 for all cases in the paper
    """
    if height is None:
        height = width
    M = get_M(width, height)
    B = get_incidence_matrix(width, height)
    M_diag = get_M_diag(M)
    psi_d = get_psi_d(width, height)
    w = cp.Variable(M) # The weight of each edge in a vector. With B fixed, the graph G can be uniquely determined by it. 
    # Optimization summary: constant array inner product with w + constant times l1 norm of contant matrix times w - something similar to l1 norm
    prob = cp.Problem(cp.Minimize(((B.T@u).reshape(-1,1)@(B.T@u).reshape(1,-1)).reshape(-1)@M_diag@w + alpha*cp.norm(psi_d.T@w,1) - beta*np.ones(M)@cp.log(w)),
            [w <= np.ones(M)])
    prob.solve()
    W_hat = np.diag(w.value)
    L = B@W_hat@B.T # The incidence matrix definition of graph Laplacian
    _, psi = eig_decompose(L) # psi is the graph Fourier transformation matrix
    u_hat = psi.T@u # The coefficients in the Fourier domain. Theoretical one not the quantized one
    return w.value, u_hat

def quantize(w, width, height, step_size, M_thresh):
    psi_d = get_psi_d(width, height)
    w_hat = psi_d.T@w
    w_hat_r = w_hat
    w_hat_r[M_thresh:] = 0 # Reduce by keeping only M_thresh number of weights
    w_hat_r_quant = (w_hat/step_size).round()
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

def entropy_coding(u_hat_quant, w_hat_quant):
    u_list = u_hat_quant.tolist()
    w_list = w_hat_quant.tolist()
    word_cnt_tb = collections.Counter(u_list + w_list)
    code_book = huffman.codebook(word_cnt_tb.items())
    u_code = ''.join([code_book[k] for k in u_list])
    w_code = ''.join([code_book[k] for k in w_list])
    return u_code, w_code
    
    
def compute_distortion(u, u_recon):
    return np.linalg.norm(u - u_recon)

def argmin(iterable):
    return min(enumerate(iterable), key=lambda x: x[1])[0]

def choose_quantization(u, u_hat, w, width, height, step_sizes, M_thresh, gamma):
    w_hat_quant_list = [quantize(w, width, height, step_size, M_thresh, encode=True) for step_size in step_sizes]
    u_hat_quant_list = [(u_hat/step_sizes).round() for step_size in step_sizes]
    u_recon_list = [reconstruct(u_hat_quant, w_hat_quant, step_size, width, height) for u_hat_quant, w_hat_quant, step_size in zip(u_hat_quant_list, w_hat_quant_list, step_sizes)]
    distortions = [compute_distortion(u, u_recon) for u_recon in u_recon_list]
    coded = [entropy_coding(u_hat_quant, w_hat_quant) for u_hat_quant, w_hat_quant in zip(u_hat_quant_list, w_hat_quant_list)]
    rates = [len(''.join(code))/(width*height*4) for code in coded]
    loss = [distortion + gamma*rate for distortion,rate in zip(distortions, rates)]
    opt_idx = argmin(loss)
    return u_hat_quant_list[opt_idx], w_hat_quant_list[opt_idx]
    