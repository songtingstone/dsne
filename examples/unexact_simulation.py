import os
import numpy as np
import umap
from anndata import AnnData, read_h5ad
from matplotlib import pyplot as plt
from tsne import bh_sne
from dsne import DSNE, DSNE_approximate, scvelo_velocity_embedding


def unitLength(V):
    V_ = V/np.sqrt(np.sum(V*V,axis=1,keepdims=True))
    return V_
def velocity_accuracy(V, V_exact, N):
    index = np.asarray(list(range(N-1))+ list(np.arange(N, 2*N-1)) + list(np.arange(2*N, 3*N-1)), dtype=np.int)
    V_ = V[index,:]
    V_exact_ = V_exact[index,:]
    V_unit = unitLength(V_)
    V_exact_unit = unitLength(V_exact_)
    accu = np.sum( V_unit* V_exact_unit )/(V_.shape[0]*1.)
    return accu

def simulate_data_base(N = 100, D = 30, theta=0.5, perplexity=20,  nrep=1, resimulate =False, save =True, file_prefix = "./data/"):
    # 3 types
    file_base = f"unexact_simulation_data_N_{N}_D_{D}_nrep_{nrep}.h5ad"
    file = os.path.join(file_prefix, file_base)
    if os.path.exists(file) and not resimulate:
        adata = read_h5ad(file)
        return adata

    V = np.abs(np.random.randn(*(N * 3, D)))*6
    x_1 = np.asarray([0,]*D)
    x_2 =  np.asarray([50,]*D)
    x_3 = np.asarray([160, ]*D)
    X = np.zeros_like(V)
    X[0, :] = x_1
    X[N, :] = x_2
    X[N*2, :] = x_3
    for i in np.arange(N-1):
        X[i+1,:] = X[i,:] + V[i,:]
        X[i + N +1, :] = X[i+ N, :] + V[i+ N, :]
        X[i + N*2 + 1, :] = X[i + N*2, :] + V[i + N*2, :]

    y = np.asarray([0,] *N + [1,]  * N + [ 2,]*N)
    X_tsne = bh_sne(X,copy_data=True,perplexity=perplexity, theta=theta, verbose=True)
    X_umap = umap.UMAP(n_components=2).fit_transform(X)
    X_tsne = np.asarray(X_tsne,dtype=np.float64)
    X_umap = np.asarray(X_umap, dtype=np.float64)
    V_2d_tsne_true = np.zeros_like(X_tsne)
    V_2d_umap_true = np.zeros_like(X_umap)
    for i in range(V_2d_tsne_true.shape[0]-1):
        V_2d_tsne_true[i,:] = X_tsne[i+1,:] - X_tsne[i,:]
        V_2d_umap_true[i,:] = X_umap[i+1,:] - X_umap[i,:]
    adata = AnnData(X=X,layers = {"V":V}, obsm={"X_umap":X_umap,"X_tsne":X_tsne,
                                                "V_2d_tsne_true": V_2d_tsne_true,
                                                "V_2d_umap_true": V_2d_umap_true,
                                                }, obs={"clusters":y})
    if save:
        adata.write_h5ad(file)
    return adata


nrep = 6
N = 50
D = 30
K = 6
perplexity = 1
theta = 0.5
exact = False
basis = "tsne"
# basis ="umap"

adata = simulate_data_base(N = N, D = D, nrep=nrep,theta=theta, resimulate =False, save =False, file_prefix = "./data/")
X = adata.X
V = adata.layers["V"]
X_2d = adata.obsm[f"X_{basis}"]
y =  np.asarray([0,] *N + [1,]  * N + [ 2,]*N)
adata.obs['clusters'] = y
V_2d_true = adata.obsm[f"V_2d_{basis}_true"]

V_2d = DSNE(X, V, Y=X_2d,
            K=K,
            perplexity=perplexity,
            verbose=True)

plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y,s=0.1)
plt.quiver(X_2d[:, 0], X_2d[:, 1],V_2d[:,0], V_2d[:,1],y,
           # width = 1,
           lw=0.01,
           headwidth=3,
           headlength=2,
           headaxislength =1,
           # scale=1
           )
# plt.legend("0","1","2")
for i in np.arange(N):
    plt.text(X_2d[i, 0], X_2d[i, 1], s=eval(f"i"))
    plt.text(X_2d[i+N, 0], X_2d[i+N, 1], s=eval(f"i"))
    plt.text(X_2d[i + N*2, 0], X_2d[i + N*2, 1], s=eval(f"i"))

plt.colorbar()
accu = velocity_accuracy(V_2d, V_2d_true, N)
basis_name = "UMAP" if basis == 'umap' else "t-SNE"
title = "DSNE velocity embedding on {} with accuracy {:5.3f}".format(basis_name, accu)
print(title)
plt.title(title)
plt.show()


V_2d = DSNE_approximate(X, V, Y=X_2d,
                        perplexity=perplexity,
                        K=K,
                        verbose=True)

plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y,s=0.1)
plt.quiver(X_2d[:, 0], X_2d[:, 1],V_2d[:,0], V_2d[:,1],y,
           # width = 1,
           lw=0.01,
           headwidth=3,
           headlength=2,
           headaxislength =1,
           # scale=1
           )
plt.colorbar()

for i in np.arange(N):
    plt.text(X_2d[i, 0], X_2d[i, 1], s=eval(f"i"))
    plt.text(X_2d[i+N, 0], X_2d[i+N, 1], s=eval(f"i"))
    plt.text(X_2d[i + N*2, 0], X_2d[i + N*2, 1], s=eval(f"i"))

accu = velocity_accuracy(V_2d, V_2d_true, N)
print(f"DSNE_approximate accu {accu} \n")

plt.title(f"DSNE_approximate, accu {accu}")

plt.show()

V_2d = scvelo_velocity_embedding(X, V, X_2d)
plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y,s=0.1)
plt.quiver(X_2d[:, 0], X_2d[:, 1],V_2d[:,0], V_2d[:,1],y,
           lw=0.01,
           headwidth=3,
           headlength=2,
           headaxislength =1,
           )
plt.colorbar()

# plt.legend("0","1","2")
for i in np.arange(N):
    plt.text(X_2d[i, 0], X_2d[i, 1], s=eval(f"i"))
    plt.text(X_2d[i+N, 0], X_2d[i+N, 1], s=eval(f"i"))
    plt.text(X_2d[i + N*2, 0], X_2d[i + N*2, 1], s=eval(f"i"))

accu = velocity_accuracy(V_2d, V_2d_true, N)
basis_name = "UMAP" if basis == 'umap' else "t-SNE"
title = "scVelo velocity embedding on {} with accuracy {:5.3f}".format(basis_name, accu)
print(title)
plt.title(title)

plt.show()
