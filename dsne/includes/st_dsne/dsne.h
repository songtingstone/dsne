/*
 *  dsne.h
 *  Header file for DSNE.
 *
 *  Created by Songting Shi.
 *  Copyright 2021. All rights reserved.
 */


#ifndef DSNE_H
#define DSNE_H
#include <stdlib.h>
#include <vector>
#include <stdio.h>

static inline double sign(double x) { return (x == .0 ? .0 : (x < .0 ? -1.0 : 1.0)); }

// todo  support the already computed knn index
class DSNE
{
public:
    void run(double* X, double* V, int N_full, int K, int D, double* Y, double* W_full, int no_dims,
        double perplexity, double threshold_V, double separate_threshold,  int max_iter,
        int mom_switch_iter, double momentum, double final_momentum,  double eta, double epsilon_kl, double epsilon_dsne,
        bool with_norm,
        unsigned int seed,   bool verbose);

    void run(double* X, double* V, int N_full, int K, int D, int* row_P, int* col_P, double* Y, double* W_full, int no_dims,
        double perplexity, double threshold_V, double separate_threshold,  int max_iter,
        int mom_switch_iter, double momentum, double final_momentum,  double eta, double epsilon_kl, double epsilon_dsne,
        bool with_norm,
        unsigned int seed,   bool verbose);  // for same K

    // with K_mean for y_mean for use K for P Q
    void run(double* X, double* V, int N_full, int K, int K_mean, int D, double* Y, double* W_full, int no_dims,
        double perplexity, double threshold_V, double separate_threshold,  int max_iter,
        int mom_switch_iter, double momentum, double final_momentum,  double eta, double epsilon_kl, double epsilon_dsne,
        bool with_norm,
        unsigned int seed,   bool verbose);

    // with K_mean for y_mean for use K for P Q  with already computed index
    void run(double* X, double* V, int N_full, int K, int K_mean,  int D, int* row_P, int* col_P, double* Y,
        double* W_full, int no_dims,
        double perplexity, double threshold_V, double separate_threshold,  int max_iter,
        int mom_switch_iter, double momentum, double final_momentum,  double eta, double epsilon_kl, double epsilon_dsne,
        bool with_norm,
        unsigned int seed,   bool verbose);


    void run_approximate(double* X, double* V, int N_full, int K, int D, double* Y, double* W_full, int no_dims,
        double perplexity, double threshold_V, double separate_threshold,
        bool with_norm,
        unsigned int seed, bool verbose);
    void computeScaledNorm(double* X, double* V, double* Y, int N_full, int N, int D, int no_dims, bool* mask_V, double *scaled_norm);
    std::vector<double> objective_dsne, objective_kl;

private:
    void computeGradient(double* P, double* beta_Q, double* dYhat,
        double* W, int N, int K, int D, double* Q, double* dW);
    double evaluateErrorKnn(double* P, double* Q, int N, int K );

    void unitLength(double* V, int N, int D);
    void clip(double* X, int N, int D, double min_val, double max_val);
    double dmax(unsigned int N, double* X, unsigned int inc, double max_lower_bound);
    void computeDirectionGaussianPerplexity(double* X, double* V, double* Y,  bool* mask_V, int N_full, int N,  int K, int D, int no_dims,
        bool with_ii,
        int** _row_P, int** _col_P, double* P, double* beta_P, double* H_P, double perplexity, double separate_threshold,
        bool verbose);

    // precomputed knn index store in row_P and col_P
    void computeDirectionGaussianPerplexity(double* X, double* V, double* Y,  bool* mask_V, int N_full, int N,  int K,  int D, int no_dims,
        bool with_ii,
        int* row_P, int* col_P, double* P, double* beta_P, double* H_P, double perplexity, double separate_threshold,
        bool verbose);

    void computeUnitYhatMatrix(int* row_P, int* col_P, double*Y, bool* mask_V, int N_full, int N, int K, int no_dims,
        double* Yhat, bool verbose);
    void computeDeltaYhatMatrix(double* Yhat, int N, int K, int no_dims,  bool verbose);
    // use Yhat_mean computed previously.
    void computeDeltaYhatMatrix(double* Yhat, int N, int K, int no_dims, double* Yhat_mean, bool verbose);
    void computeMaskV(double* V, int N_full, int D, bool* mask_V, int &N, double threshold);
    void computeCosineDistance(double* Y_hat, double* W, int N, int K, int no_dims, double* cosineD);
    double randn();
    void write_matrix(double* data, int N, int D, FILE* fp);
    bool check_convergence(int type, int window_size, double epsilon_kl, double epsilon_dsne);
    void update_beta_Q_knn(double* P, double* beta_Q, double* dYhat,
        double* W, int N, int K,  int no_dims,
        double perplexity, double* Q);
    void update_beta_Q_knn_new(double* P, double* beta_P_mul_Dist_sum, double* beta_Q, double* Yhat, double* W, int N, int K,  int no_dims,
        double perplexity, double* Q);
};

#endif
