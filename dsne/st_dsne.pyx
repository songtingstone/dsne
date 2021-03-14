# cython: linetrace=True
# distutils: define_macros=CYTHON_TRACE=1
# distutils: language=c++

import numpy as np
cimport numpy as np
cimport cython

cdef extern from "dsne.h":
    cdef cppclass DSNE:
        DSNE()
        void run(double* X, double* V, int N_full, int K, int D, double* Y, double* W_full, int no_dims,
                double perplexity, double threshold_V, double separate_threshold,  int max_iter,
                int mom_switch_iter, double momentum, double final_momentum,  double eta, double epsilon_kl, double epsilon_dsne,
                 int with_norm,
                unsigned int seed,  int verbose);
        void run_approximate(double* X, double* V, int N_full, int K, int D, double* Y, double* W_full, int no_dims,
            double perplexity, double threshold_V, double separate_threshold,
            int with_norm,
            unsigned int seed, int verbose);

cdef class ST_DSNE:
    cdef DSNE* thisptr # hold a C++ instance

    def __cinit__(self):
        self.thisptr = new DSNE()

    def __dealloc__(self):
        del self.thisptr

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def run(self, X, V, Y, N_full, K, D, d, perplexity, threshold_V,
                    separate_threshold, max_iter,   mom_switch_iter,
        momentum, final_momentum, eta, epsilon_kl, epsilon_dsne, with_norm,  seed, verbose=False):
        cdef np.ndarray[np.float64_t, ndim=2, mode='c'] _X = np.ascontiguousarray(X)
        cdef np.ndarray[np.float64_t, ndim=2, mode='c'] _V = np.ascontiguousarray(V)
        # cdef np.ndarray[np.float64_t, ndim=2, mode='c'] Y = np.zeros((N_full, d), dtype=np.float64)
        cdef np.ndarray[np.float64_t, ndim=2, mode='c'] _Y = np.ascontiguousarray(Y)
        cdef np.ndarray[np.float64_t, ndim=2, mode='c'] W_full= np.zeros((N_full, d), dtype=np.float64)
        self.thisptr.run(&_X[0,0], &_V[0,0], N_full, K, D, &_Y[0,0], &W_full[0,0], d, perplexity, threshold_V,
                    separate_threshold,  max_iter,
                mom_switch_iter, momentum, final_momentum,  eta, epsilon_kl, epsilon_dsne,
                with_norm,  seed,   verbose)
        return W_full

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def run_approximate(self, X, V, Y, N_full, K, D, d, perplexity, threshold_V,
                    separate_threshold, with_norm,  seed, verbose=False):
        cdef np.ndarray[np.float64_t, ndim=2, mode='c'] _X = np.ascontiguousarray(X)
        cdef np.ndarray[np.float64_t, ndim=2, mode='c'] _V = np.ascontiguousarray(V)
        cdef np.ndarray[np.float64_t, ndim=2, mode='c'] _Y = np.ascontiguousarray(Y)
        cdef np.ndarray[np.float64_t, ndim=2, mode='c'] W_full= np.zeros((N_full, d), dtype=np.float64)
        self.thisptr.run_approximate(&_X[0,0], &_V[0,0], N_full, K, D, &_Y[0,0], &W_full[0,0], d, perplexity, threshold_V,
                    separate_threshold,
                    with_norm,
                    seed, verbose)
        return W_full
