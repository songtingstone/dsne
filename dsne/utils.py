import scvelo as scv
from anndata import AnnData


def scvelo_velocity_embedding(X, V, Y):
    adata_tmp = AnnData(X=X,obsm={ "X_umap":Y}, layers={"velocity":V, "spliced":X})
    scv.pp.moments(adata_tmp, n_pcs=30, n_neighbors=30)
    scv.tl.velocity_graph(adata_tmp, xkey='spliced')
    scv.tools.velocity_embedding(adata_tmp,basis="umap")
    W = adata_tmp.obsm["velocity_umap"]
    return W
