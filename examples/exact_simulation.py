import os
import numpy as np
import scvelo as scv
from anndata import AnnData, read_h5ad
from dsne import DSNE, DSNE_approximate



N=500
D=300
d=2
K=16
perplexity =6
n_rep=1
exact = False
with_norm = True
basis = 'exact_embeddings'
verbose = False

def unitLength(V):
    V_ = V/np.sqrt(np.sum(V*V,axis=1,keepdims=True))
    return V_
def velocity_accuracy(V, V_exact):
    V_unit = unitLength(V)
    V_exact_unit = unitLength(V_exact)
    accu = np.sum( V_unit* V_exact_unit )/(V.shape[0]*1.)
    return accu

def simulate_data(N=50, D=3, d=2, save =True, file_name_prefix ="./data" ):
    if not os.path.exists(file_name_prefix):
        print("Directory: {} do not exist, create it! \n".format(os.path.abspath(file_name_prefix)))
        os.makedirs(os.path.abspath(file_name_prefix))
    V_2d = np.random.randn(*(N * 3, d)) * 6
    err_2d = np.random.randn(*(N * 3, d))*2
    x_1 = np.asarray([0, ] * d)
    x_2 = np.asarray([50, ] * d)
    x_3 = np.asarray([160, ] * d)
    X_2d = np.zeros_like(V_2d)
    X_2d[0, :] = x_1
    X_2d[N, :] = x_2
    X_2d[N * 2, :] = x_3
    for i in np.arange(N - 1):
        X_2d[i + 1, :] = X_2d[i, :] + V_2d[i, :] + err_2d[i,:]
        X_2d[i + N + 1, :] = X_2d[i + N, :] + V_2d[i + N, :] + err_2d[i + N, :]
        X_2d[i + N * 2 + 1, :] = X_2d[i + N * 2, :] + V_2d[i + N * 2, :] +  err_2d[i + N * 2, :]


    y = np.asarray([0, ] * N + [1, ] * N + [2, ] * N)
    U = np.array(np.random.randn(*(d, D)))
    X = X_2d.__matmul__(U)
    V = V_2d.__matmul__(U)


    adata = AnnData(X=X, layers={"velocity": V},obs={"clusters": y}, obsm={"X_exact_embeddings":X_2d, "V_exact_embeddings":V_2d})
    if save:
        file_name = file_name_prefix+"simulated_data_N_{}_D_{}.h5hd".format(N,D)
        adata.write_h5ad(file_name)
    return adata


adata = simulate_data(N=N,D=D,d=d,save=False)
X = adata.X
V = adata.layers["velocity"]
X_basis = f"X_{basis}"

for method in ['DSNE', "DSNE_approximate","scvelo_velocity_original"  ]:
    X = np.asarray(X, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    Y = None
    if (X_basis in adata.obsm.keys()) and adata.obsm[X_basis] is not None:
        Y = adata.obsm[f"X_{basis}"]

    if Y is None:
        print("Do not get the low dimesnional embedding Y! \n")
        # raise

    Y = np.asarray(Y, dtype=np.float64)
    if method == "DSNE":
        W = DSNE(X, V, Y=Y,
                 K= K,
                 perplexity=perplexity,
                 pca_d=None,
                 threshold_V=1e-8,
                 separate_threshold=1e-8,
                 max_iter=1000,
                 mom_switch_iter=250,
                 momentum=0.5,
                 final_momentum=0.8,
                 eta=0.1,
                 epsilon_kl=1e-16,
                 epsilon_dsne=1e-16,
                 with_norm=with_norm,
                 seed=16,
                 random_state=None,
                 copy_data=True,
                 verbose=verbose)
        vkey = "velocity_dsne"
    elif method == "DSNE_approximate":
        W = DSNE_approximate(X, V, Y=Y,
                                      perplexity=perplexity,
                                      pca_d=None,
                                      threshold_V=1e-8,
                                      separate_threshold=1e-8,
                                      seed=16,
                                      random_state=None,
                                      copy_data=False,
                                      verbose=verbose)
        vkey = "velocity_scvelo"
    elif method == "scvelo_velocity_original":
        adata_tmp = AnnData(X=X, obsm={"X_umap": Y}, layers={"velocity": V, "spliced": X})
        scv.tl.velocity_graph(adata_tmp, xkey='spliced')
        scv.tools.velocity_embedding(adata_tmp, basis="umap")
        W = adata_tmp.obsm["velocity_umap"]
        vkey = "velocity_scvelo_original"
    else:
        print("method: {} do not implemented!\n".format(method))
        # raise
    str_exact = "exact" if exact else "approx"
    adata.obsm[f"{vkey}_{str_exact}_{basis}"] = W
    W_exact = adata.obsm["V_exact_embeddings"]
    accu = velocity_accuracy(W, W_exact)
    print(f"  {method}, {str_exact},  accu: {accu}\n")
    if method == 'DSNE':
        method_str = "DSNE"
    elif method == 'scvelo_velocity_original':
        method_str = "scVelo"
    elif method =="DSNE_approximate":
        method_str = "DSNE-approximate"
    title = "{} on exact embeddings with accuracy {:5.3f}".format(method_str, accu)
    scv.pl.velocity_embedding(adata, basis=basis, V=W, title=title,density=2,)
    scv.pl.velocity_embedding_stream(adata, basis=basis, V=W, title=title,density=2,)
    scv.pl.velocity_embedding_grid(adata, basis=basis, V=W, title=title,)
