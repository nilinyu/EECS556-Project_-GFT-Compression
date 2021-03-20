import cvxpy as cp
import numpy as np

###
# Utilities, no sanity check in most cases
###
def edge_idx_parse(idx, w, h):
    quo, rem = divmod(idx, w+h-2)
    if rem < w - 1: # horizontal
        return (quo*w + rem, quo*w + rem + 1)
    rem = rem - (w-1)
    return (quo*w + rem, quo*w + rem + w)

def edge_name_parse(src, dest, w, h):
    if dest == src + 1:
        return src//w * (h+w-2) + src%(w - 1)
    if dest == src + w:
        return src//w * (h+w-2) + w - 1 + src%w
    return None

def get_incidence_matrix(w,h):
    N = w*h # number of nodes
    M = ((2*w-1)*(2*h-1)-1)//2 # number of edges
    B = np.zeros((N,M), int)
    for i in range(N):
        if (i+1)%w:
            B[i, edge_name_parse(i, i+1, w, h)] = 1
            B[i+1, edge_name_parse(i, i+1, w, h)] = -1
        if (i + w) < N:
            B[i, edge_name_parse(i, i+w, w, h)] = 1
            B[i+w, edge_name_parse(i, i+w, w, h)] = -1
    return B

def get_M_diag(M):
    M_diag = np.zeros((M**2,M), int)
    for i in range(M):
        M_diag[i*M+i, i] = 1
    return M_diag

def get_psi_d(M, w, h):
    # is self loop included?
    psi_d = np.zeros((M,M), int)
    for i in range(M):
        for j in range(M):
            a,b = edge_idx_parse(i, w, h)
            c,d = edge_idx_parse(j, w, h)
            if i != j and (a==c or b==c or a==d or b==d):
                psi_d[i,j] = 1
                psi_d[j,i] = 1
    return psi_d

def convex_optimize(u, W, H=None, alpha=100, beta=1):
    """
    The convex optimization problem is to minimize

    vec(B'uu'B)'M_{diag}w + alpha||psi_d'w||_1 - beta1'log(w)
    s.t. w<=1

    The clarify:
    u: input signal
    B: N*M incidence matrix of the graph that holds the input signal
        The graph is a fixed grid graph, s.t. each node is connected
        to its 4 neighbors.
    M_diag: matrix that turns a vector v into vec(diag(v))
    psi_d: eigenvectors of dual graph
    alpha: based on smoothness of the block
    beta: 1 for all cases in the paper
    """
    if H is None:
        H = W
    M = ((2*W-1)*(2*H-1)-1)//2
    B = get_incidence_matrix(W, H)
    M_diag = get_M_diag(M)
    psi_d = get_psi_d(M, W, H)
    w = cp.Variable(M)
    # constant array inner product with w + constant times l1 norm of contant matrix times w - something similar to l1 norm
    prob = cp.Problem(cp.Minimize(((B.T@u).reshape(-1,1)@(B.T@u).reshape(1,-1)).reshape(-1)@M_diag@w + alpha*cp.norm(psi_d.T@w,1) - beta*np.ones(M)@cp.log(w)),
            [w >= 0, w <= np.ones(M)])
    prob.solve()
    return w.value


    