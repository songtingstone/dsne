def test_seed():
    from dsne import DSNE
    from sklearn.datasets import load_iris
    from sklearn.decomposition import PCA
    import numpy as np

    iris = load_iris()

    X = iris.data
    V = iris.data
    # y = iris.target
    pca = PCA(n_components=2)
    Y = pca.fit_transform(X)


    t1 = DSNE(X,V,Y, random_state=np.random.RandomState(0), copy_data=True)
    # t2 = dsne(X,V,Y, random_state=np.random.RandomState(0), copy_data=True)

    assert t1.shape[0] == 150
    assert t1.shape[1] == 2
