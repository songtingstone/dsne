import numpy as np
import scvelo as scv
from scipy.sparse import issparse
from dsne import DSNE, DSNE_approximate


scv.settings.verbosity = 3  # show errors(0), warnings(1), info(2), hints(3)
scv.settings.presenter_view = True  # set max width size for presenter view
scv.settings.set_figure_params('scvelo')  # for beautified visualization
adata = scv.datasets.pancreas()

scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
scv.pp.moments(adata, n_pcs=30, n_neighbors=30)

scv.tl.velocity(adata)


def get_X_V_Y(adata,vkey="velocity",
              xkey="Ms",
              basis=None,
              gene_subset=None,
              ):

    subset = np.ones(adata.n_vars, bool)
    if gene_subset is not None:
        var_names_subset = adata.var_names.isin(gene_subset)
        subset &= var_names_subset if len(var_names_subset) > 0 else gene_subset
    elif f"{vkey}_genes" in adata.var.keys():
        subset &= np.array(adata.var[f"{vkey}_genes"].values, dtype=bool)

    xkey = xkey if xkey in adata.layers.keys() else "spliced"
    basis = 'umap' if basis is None else basis
    X = np.array(
        adata.layers[xkey].A[:, subset]
        if issparse(adata.layers[xkey])
        else adata.layers[xkey][:, subset]
    )
    V = np.array(
        adata.layers[vkey].A[:, subset]
        if issparse(adata.layers[vkey])
        else adata.layers[vkey][:, subset]
    )
    # V -= np.nanmean(V, axis=1)[:, None]
    Y =np.array(
        adata.obsm[f"X_{basis}"]
    )


    nans = np.isnan(np.sum(V, axis=0))
    if np.any(nans):
        X = X[:, ~nans]
        V = V[:, ~nans]
    return X,V,Y


X,V,X_2d = get_X_V_Y(adata,vkey="velocity",xkey="Ms",basis="umap")


V_2d = DSNE(X, V, Y=X_2d,
            perplexity=3.0,
            K=16,
            threshold_V=1e-8,
            separate_threshold=1e-8,
            max_iter=600,
            mom_switch_iter=250,
            momentum=0.5,
            final_momentum=0.8,
            eta=0.1,
            epsilon_kl=1e-16,
            epsilon_dsne=1e-16,
            seed=6,
            random_state=None,
            copy_data=False,
            with_norm=True,
            verbose=True)

adata.obsm["X_DSNE"] = X_2d
adata.obsm["V_DSNE"] = V_2d
title ="DSNE"
scv.pl.velocity_embedding_stream(adata, title=title, basis='umap',V=adata.obsm["V_DSNE"], smooth=0.5,density=2,)

scv.pl.velocity_embedding_grid(adata, title=title, basis='umap',V=adata.obsm["V_DSNE"], smooth=0.5,density=2,)


scv.pl.velocity_embedding(adata,  title=title,basis='umap',V = adata.obsm["V_DSNE"])


title ="DSNE_approximate"
V_2d = DSNE_approximate(X, V, Y=X_2d,
                        perplexity=3.0,
                        K=16,
                        threshold_V=1e-8,
                        separate_threshold=1e-8,
                        seed=6,
                        random_state=None,
                        copy_data=False,
                        with_norm=True,
                        verbose=True)

adata.obsm["X_DSNE_approximate"] = X_2d
adata.obsm["V_DSNE_approximate"] = V_2d

scv.pl.velocity_embedding_stream(adata, basis='umap',V=adata.obsm["V_DSNE_approximate"],  title=title, smooth=0.5,density=2,)

scv.pl.velocity_embedding_grid(adata, basis='umap',V=adata.obsm["V_DSNE_approximate"],  title=title, smooth=0.5,density=2,)

scv.pl.velocity_embedding(adata, basis='umap',V = adata.obsm["V_DSNE_approximate"], title=title)
