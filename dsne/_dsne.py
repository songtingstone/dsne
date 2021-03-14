import numpy as np
import scipy.linalg as la
# dynamical lib
from st_dsne import ST_DSNE


def DSNE(
    data,
    V,
    Y,
    pca_d=None,
    perplexity=3.0,
    K = 16,
    threshold_V=1e-9,
    separate_threshold=1e-9,
    max_iter=1000,
    mom_switch_iter=250,
    momentum=0.5,
    final_momentum=0.8,
    eta=0.1,
    epsilon_kl=1e-9,
    epsilon_dsne=1e-9,
    with_norm=True,
    seed=None,
    random_state=None,
    copy_data=True,
    verbose=False,
):
    """
    Run DSNE on data (data points), V (velocity), Y (map points).

    @param data         The data points matrix, with shape (N,D).

    @param V            The velocity matrix, with shape (N,D), each V_i with shape D is the velocity of point data_i.

    @param Y            The low dimension map points matrix, with shape (N,d), each Y_i with shape d is the map points of point data_i.

    @param pca_d        The dimensionality of data and V is reduced via PCA
                        to this dimensionality, when D>5000, can use pca to reduce the dimension of data and V.

    @param perplexity   The perplexity controls the effective number of
                        neighbors.

    @param K            The number of KNN neighbors, which is used to compute the similarity matrix P and Q.

    @param threshold_V  A threshold. When max(abs(V_i)) < thershold_V, will omit to find its velocity embedding and set w_i=0

    @param separate_threshold.  A threshold. When we find the KNN neighbor of data_i, only when j satisfies
                            || data_j - data_i||^2 > separate_threshold and max( abs(Y_j- Y_i) ) > separate_threshold
                        will be considered as the neighbor of data_i.

    @param max_iter     The max number of iteration to update the velocity embedding W

    @mom_switch_iter    The momentum swith times. On iter in  [0, mom_switch_iter) will use momentum, on iter
                        in [mom_switch_iter, max_iter) will use final_momentum.

    @momentum           The initial momentum scalar.

    @final_momentum     The final momentum scalar.

    @eta                Learning rate. Scalar.

    @epsilon_kl         The threshold to check the convergence of velocity embedding for fixed \beta_Q each round

    @epsilon_dsne       The threshold to check the convergence of DSNE.

    @with_norm          Bool. If true, return the velocity embedding W wiht approximate norm based on data,V and Y;
                        If false, return velocity embddding W with unit length for each W_i, i.e ||W_i|| =1

    @seed               Int. The random seed.

    @param random_state A numpy RandomState object; if None, use
                        the numpy.random singleton. Init the RandomState
                        with a fixed seed to obtain consistent results
                        from run to run.

    @param copy_data    Copy the data to prevent it from being modified
                        by the C code.

    @param verbose      Verbose output from the training process.
    """
    N, D = data.shape
    N_V, D_V = V.shape
    N_Y, d = Y.shape
    if(not (N== N_Y and  N_Y == N_V and D ==D_V)):
        print("The shape of data: {}, V: {}, Y: {} mismatch!".format(data.shape, V.shape, Y.shape))
        raise


    if pca_d is None:
        if copy_data:
            X = np.copy(data)
            V = np.copy(V)
            Y = np.copy(Y)
        else:
            X = data
    else:
        # do PCA
        # data -= data.mean(axis=0)

        # working with covariance + (svd on cov.) is
        # much faster than svd on data directly.
        cov = np.dot(data.T, data) / N
        u, s, v = la.svd(cov, full_matrices=False)
        u = u[:, 0:pca_d]
        X = np.dot(data, u)
        V = np.dot(V, u)
    if seed is None:
        if random_state is None:
            seed = np.random.randint(2 ** 32 - 1)
        else:
            seed = random_state.randint(2 ** 32 - 1)

    if K is None:
        K = int(3*perplexity)

    if K > N - 1:
        if verbose:
            print("K: {} is too large, set it to N - 1".format(K, N-1))
        K = N - 1

    st_dsne= ST_DSNE()
    original_dtype = X.dtype
    dtype_changed = False
    if X.dtype is not np.dtype(np.float64):
        dtype_changed = True
        X = np.asarray(X, dtype=np.float64)
    if V.dtype is not np.dtype(np.float64):
        V = np.asarray(V, dtype=np.float64)
        dtype_changed = True
    if Y.dtype is not np.dtype(np.float64):
        Y = np.asarray(Y, dtype=np.float64)
        dtype_changed = True
    W = st_dsne.run(X, V, Y, N, K, X.shape[1], d, perplexity, threshold_V,
                    separate_threshold,  max_iter, mom_switch_iter, momentum, final_momentum, eta,epsilon_kl,
                        epsilon_dsne, with_norm,
    seed, verbose)
    if dtype_changed:
        W = np.asarray(W, dtype=original_dtype)
    return W


def DSNE_approximate(
    data,
    V,
    Y,
    pca_d=None,
    perplexity=3.0,
    K = 16,
    threshold_V=1e-9,
    separate_threshold=1e-9,
    with_norm = True,
    seed=None,
    random_state=None,
    copy_data=True,
    verbose=False,
):
    """
        Run DSNE approximately on data (data points), V (velocity), Y (map points).

        @param data         The data points matrix, with shape (N,D).

        @param V            The velocity matrix, with shape (N,D), each V_i with shape D is the velocity of point data_i.

        @param Y            The low dimension map points matrix, with shape (N,d), each Y_i with shape d is the map points of point data_i.

        @param pca_d        The dimensionality of data and V is reduced via PCA
                            to this dimensionality, when D>5000, can use pca to reduce the dimension of data and V.

        @param perplexity   The perplexity controls the effective number of
                            neighbors.

        @param K            The number of KNN neighbors, which is used to compute the similarity matrix P and Q.

        @param threshold_V  A threshold. When max(abs(V_i)) < thershold_V, will omit to find its velocity embedding and set w_i=0

        @param separate_threshold.  A threshold. When we find the KNN neighbor of data_i, only when j satisfies
                                || data_j - data_i||^2 > separate_threshold and max( abs(Y_j- Y_i) ) > separate_threshold
                            will be considered as the neighbor of data_i.

        @with_norm          Bool. If true, return the velocity embedding W wiht approximate norm based on data,V and Y;
                            If false, return velocity embddding W with unit length for each W_i, i.e ||W_i|| =1

        @seed               Int. The random seed.

        @param random_state A numpy RandomState object; if None, use
                            the numpy.random singleton. Init the RandomState
                            with a fixed seed to obtain consistent results
                            from run to run.

        @param copy_data    Copy the data to prevent it from being modified
                            by the C code.

        @param verbose      Verbose output from the training process.
        """

    N, D = data.shape
    N_V, D_V = V.shape
    N_Y, d = Y.shape
    if (not (N == N_Y and N_Y == N_V and D == D_V)):
        print("The shape of data: {}, V: {}, Y: {} mismatch!".format(data.shape, V.shape, Y.shape))
        raise


    if pca_d is None:
        if copy_data:
            X = np.copy(data)
            V = np.copy(V)
            Y = np.copy(Y)
        else:
            X = data
    else:
        # do PCA
        data -= data.mean(axis=0)

        # working with covariance + (svd on cov.) is
        # much faster than svd on data directly.
        cov = np.dot(data.T, data) / N
        u, s, v = la.svd(cov, full_matrices=False)
        u = u[:, 0:pca_d]
        X = np.dot(data, u)
        V = np.dot(V, u)
    if seed is None:
        if random_state is None:
            seed = np.random.randint(2 ** 32 - 1)
        else:
            seed = random_state.randint(2 ** 32 - 1)
    if K is None:
        K = int(6*perplexity)

    if K > N - 1:
        if verbose:
            print("K: {} is too large, set it to N - 1".format(K, N-1))
        K = N - 1
    st_dsne = ST_DSNE()
    original_dtype = X.dtype
    dtype_changed = False
    if X.dtype is not np.dtype(np.float64):
        dtype_changed = True
        X = np.asarray(X,dtype=np.float64)
    if V.dtype is not np.dtype(np.float64):
        V = np.asarray(V,dtype=np.float64)
        dtype_changed = True
    if Y.dtype is not np.dtype(np.float64):
        Y = np.asarray(Y,dtype=np.float64)
        dtype_changed = True
        # X, V, N, D, d, alpha, perplexity, theta, seed, verbose = False):
    # X, V, Y, N_full, D, d, perplexity, max_iter, mom_switch_iter,
    # momentum, final_momentum, eta, exact, seed, verbose
    # run(self, X, V, Y, N_full, D, d, perplexity, max_iter, threshold_V,
    #     separate_threshold, mom_switch_iter,
    #     momentum, final_momentum, eta, exact, seed, verbose=False)
    # print("perplexity", perplexity)
    W = st_dsne.run_approximate(X, V, Y, N, K, X.shape[1], d, perplexity, threshold_V,
                    separate_threshold, with_norm,
            seed, verbose)
    if dtype_changed:
        W = np.asarray(W, dtype=original_dtype)
    return W
