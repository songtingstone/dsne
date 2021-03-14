/*
 *  dsne.cpp
 *  Implementation of DSNE.
 *
 *  Created by Songting Shi.
 *  Copyright 2021. All rights reserved.
 *
 */

#include <math.h>
#include <float.h>
#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <cstring>
#include <time.h>
#include "vptree.h"
#include "dsne.h"

extern "C" {
    #include <cblas.h>
}

using namespace std;

// Perform DSNE
void DSNE::run(double* X, double* V, int N_full, int K, int D, double* Y, double* W_full, int no_dims,
    double perplexity, double threshold_V, double separate_threshold,  int max_iter,
    int mom_switch_iter, double momentum, double final_momentum,  double eta,  double epsilon_kl, double epsilon_dsne,
    bool with_norm,
    unsigned int seed,  bool verbose) {
    // Initalize the pseudorandom number generator
    srand(seed);
    objective_kl.clear();
    objective_dsne.clear();

    if(N_full - 1 < K)
    {
        printf("N_full: %i, K: %f,  K too large for the number of data points!\n", N_full, K);
        exit(1);
    }

    if (verbose) {
        printf("N_full: %i, no_dims = %d, using perplexity = %f, K: %i, seed=%d\n", N_full, no_dims, perplexity,  seed);
    }


    // Set learning parameters
    float total_time = .0;
    clock_t start, end;

    // Allocate some memory
    int window_size=3;
    double C = 0;
    bool* mask_V = (bool *) malloc(N_full*sizeof(bool));
    if( mask_V == NULL) { printf("Memory allocation failed!\n"); exit(1); }

    int N=0;
    computeMaskV(V, N_full, D, mask_V, N, threshold_V);
    if(verbose){
        printf("The full data has %i data points, with   %i effective velocity points.", N_full, N);
    }
    double* beta_P = (double*) malloc( N * sizeof(double));
    double* H_P = (double*) malloc( N * sizeof(double));
    double* beta_Q = (double*) malloc( N * sizeof(double));
    double* W    = (double*) malloc(N * no_dims * sizeof(double));
    double* dW    = (double*) malloc(N * no_dims * sizeof(double));
    double* uW    = (double*) malloc(N * no_dims * sizeof(double));
    double* gains_W = (double*) malloc(N * no_dims * sizeof(double));
    if(beta_P == NULL || beta_Q == NULL || H_P == NULL || dW == NULL || uW == NULL || gains_W == NULL || W == NULL )
    { printf("Memory allocation failed!\n"); exit(1); }
    for(int i = 0; i < N * no_dims; i++)    uW[i] =  .0;
    for(int i = 0; i < N * no_dims; i++) gains_W[i] = 1.0;
    double *scaled_norm;
    if(with_norm) {
        scaled_norm =  (double*) malloc(N_full  * sizeof(double));
        if(scaled_norm == NULL ) { printf("Memory allocation failed!\n"); exit(1); }
        computeScaledNorm(X, V, Y,  N_full, N,  D, no_dims, mask_V, scaled_norm);
    }

    // Normalize input data (to prevent numerical problems)
    if (verbose) {
        printf("Computing input similarities...\n");
    }
    start = clock();

    double* P; int* row_P; int* col_P; double *dYhat; double* Q;  double* dYhat_norm_square_plus_one;

    // Compute asymmetric pairwise input similarities
    P = (double*) malloc(N * K * sizeof(double));
    Q = (double*) malloc(N * K * sizeof(double));
    double* sum_Ps = (double*) malloc(N * sizeof(double));
    dYhat =  (double*) malloc(N * K * no_dims * sizeof(double));
    dYhat_norm_square_plus_one =  (double*) malloc(N * K * sizeof(double));
    if(P == NULL || sum_Ps == NULL || Q == NULL || dYhat == NULL || beta_P ==NULL || dYhat_norm_square_plus_one ==NULL )
     { printf("Memory allocation failed!\n"); exit(1); }
    computeDirectionGaussianPerplexity(X, V, Y, mask_V, N_full,  N,  K, D,  no_dims, true, // with_ii
            &row_P, &col_P,  P, beta_P, H_P, perplexity, separate_threshold, verbose);
    for(int n =0; n<N; n++) beta_Q[n] = 1.;

    // \tilde P
    int ind_nK = 0;
    for(int n = 0; n < N; n++){
        sum_Ps[n] = DBL_MIN;
        ind_nK = n*K;
        for(int k = 0; k < K; k++){
            sum_Ps[n] += P[ind_nK + k];
        }

    }
    for(int n = 0; n < N; n++){
        ind_nK = n*K;
        for(int k = 0; k < K; k++){
            P[ind_nK + k] /= sum_Ps[n];
        }
    }
    free(sum_Ps); sum_Ps = NULL;
    computeUnitYhatMatrix(row_P, col_P, Y,  mask_V,  N_full,  N,   K,  no_dims,  dYhat, verbose);
    computeDeltaYhatMatrix(dYhat, N, K, no_dims, verbose );

    end = clock();
	for(int i = 0; i < N * no_dims; i++) W[i] = randn() * .0001;
    unitLength(W, N, no_dims);

	// Perform main training loop
    if (verbose) {
        printf("Done in %4.2f seconds (sparsity = %f)!\nLearning embedding...\n", (float) (end - start) / CLOCKS_PER_SEC, (double) row_P[N] / ((double) N * (double) N));
    }


    bool updated_beta_Q = false;
    start = clock();
	for(int iter = 0; iter < max_iter; iter++) {
//	    printf("begin iter: %i\n",iter);

        computeGradient( P,  beta_Q, dYhat,  W,  N, K,  no_dims, Q, dW);


        if((!updated_beta_Q) && iter>window_size && check_convergence(0,window_size, epsilon_kl,epsilon_dsne)){
            objective_dsne.push_back(C);
            if((!updated_beta_Q)&&check_convergence(1,window_size, epsilon_kl,epsilon_dsne)){
                // convergent
                break;
            }else{
                 // update beta_Q
                 if(verbose)  printf("Update beta Q\n");
                update_beta_Q_knn(P, beta_Q, dYhat, W, N, K, no_dims, perplexity, Q);
                for(int i = 0; i < N * no_dims; i++)    uW[i] =  .0;
                for(int i = 0; i < N * no_dims; i++) gains_W[i] = 1.0;
                updated_beta_Q = true;
                continue;
            }


        }else{
            updated_beta_Q = false;
        }
        C = evaluateErrorKnn(P, Q, N, K);
        objective_kl.push_back(C);
        if(iter==0) objective_dsne.push_back(C);


        for(int i = 0; i < N * no_dims; i++) gains_W[i] = (sign(dW[i]) != sign(uW[i])) ? (gains_W[i] + .2) : (gains_W[i] * .8);
        for(int i = 0; i < N * no_dims; i++) if(gains_W[i] < .0001) gains_W[i] = .0001;


	    for(int i = 0; i < N * no_dims; i++) uW[i] = momentum * uW[i] - eta * gains_W[i]  * dW[i];
		for(int i = 0; i < N * no_dims; i++)  W[i] = W[i] + uW[i];

//        for(int i =0; i<6;i++){
//            printf("i: %d, W[i]: %f,  gains_W[i]: %f, dW[i]: %f, uW[i]: %f\n",
//                i,  W[i],  gains_W[i], dW[i], uW[i] );
//        }

 		// Make W on the unit ball
		unitLength(W, N, no_dims);
//		for(int i =0; i<6;i++){
//            printf("After nomalizatioin, i: %d, W[i]: %f,  gains_W[i]: %f, dW[i]: %f, uW[i]: %f\n",
//                i, W[i],  gains_W[i], dW[i], uW[i] );
//        }

        if(iter == mom_switch_iter) momentum = final_momentum;

        if (verbose) {
            printf("Iteration %d: err is %5.2f. Y[0] is %5.2f, W[0] is %5.2f\n", iter + 1, C, Y[0], W[0]);
        }

    }
    // fill the direction ( with_norm )
    int n_curr =0, ind_nD, ind_nfD;
    for(int n=0; n<N_full; n++){
        ind_nfD = n*no_dims;
        ind_nD = n_curr*no_dims;
        if(!mask_V[n]){
            if(with_norm){
                for(int d=0; d<no_dims; d++) W_full[ind_nfD + d] = W[ ind_nD +d] * scaled_norm[n];
            }else{
                for(int d=0; d<no_dims; d++) W_full[ind_nfD + d] = W[ ind_nD +d];
            }
        }else{
            for(int d=0; d<no_dims; d++) W_full[ind_nfD + d] = .0;
        }
        n_curr += 1;
    }

    if(n_curr>N) {printf("N_full:%i, n_cur: %i > N: %i\n",N_full, n_curr,N); exit(1);}
    end = clock(); total_time += (float) (end - start) / CLOCKS_PER_SEC;

    // Clean up memory
    free(W);
    free(dW);
    free(uW);
    free(gains_W);
    free(beta_P);
    free(beta_Q);
    free(H_P);
    free(dYhat); dYhat=NULL;
    free(P);
    free(row_P); row_P = NULL;
    free(col_P); col_P = NULL;
    free(dYhat_norm_square_plus_one); dYhat_norm_square_plus_one = NULL;
    if(with_norm) free(scaled_norm);
    if (verbose) {
        printf("Fitting performed in %4.2f seconds.\n", total_time);
    }
}


void run(double* X, double* V, int N_full, int K, int D, int* row_P, int* col_P, double* Y, double* W_full, int no_dims,
        double perplexity, double threshold_V, double separate_threshold,  int max_iter,
        int mom_switch_iter, double momentum, double final_momentum,  double eta, double epsilon_kl, double epsilon_dsne,
        bool with_norm, bool exact,
        unsigned int seed,   bool verbose);  // for same K

    // with K_mean for y_mean for use K for P Q
void DSNE::run(double* X, double* V, int N_full, int K, int K_mean, int D, double* Y, double* W_full, int no_dims,
        double perplexity, double threshold_V, double separate_threshold,  int max_iter,
        int mom_switch_iter, double momentum, double final_momentum,  double eta, double epsilon_kl, double epsilon_dsne,
        bool with_norm,
        unsigned int seed,   bool verbose){
    printf("todo\n");
}

    // with K_mean for y_mean for use K for P Q  with already computed index
void DSNE::run(double* X, double* V, int N_full, int K, int K_mean,  int D, int* row_P, int* col_P, double* Y,
        double* W_full, int no_dims,
        double perplexity, double threshold_V, double separate_threshold,  int max_iter,
        int mom_switch_iter, double momentum, double final_momentum,  double eta, double epsilon_kl, double epsilon_dsne,
        bool with_norm,
        unsigned int seed,   bool verbose){
    printf("todo\n");
}


void DSNE::run_approximate(double* X, double* V, int N_full, int K, int D, double* Y, double* W_full, int no_dims,
    double perplexity, double threshold_V, double separate_threshold,
    bool with_norm,
    unsigned int seed,   bool verbose){
    // Initalize the pseudorandom number generator
    srand(seed);

    if(N_full - 1 < K)
    {
        printf("N_full: %i, K: %f,  K too large for the number of data points!\n", N_full, K);
        exit(1);
    }

    if (verbose) {
        printf("N_full: %i, using no_dims = %d, perplexity = %f, K: %i, seed=%d\n", N_full, no_dims, perplexity,  seed);
    }

    // Set learning parameters
    float total_time = .0;
    clock_t start, end;

    // Allocate some memory
    bool* mask_V = (bool *) malloc(N_full*sizeof(bool));
    if( mask_V == NULL) { printf("Memory allocation failed!\n"); exit(1); }

    int N=0;
    computeMaskV(V, N_full, D, mask_V, N, threshold_V);
    if(verbose){
        printf("The full data has %i data points, with   %i effective velocity points.", N_full, N);
    }
    double* beta_P = (double*) malloc( N * sizeof(double));
    double* H_P = (double*) malloc( N * sizeof(double));

    if(beta_P == NULL || H_P == NULL  )
    { printf("Memory allocation failed!\n"); exit(1); }
    double *scaled_norm;
    if(with_norm) {
        scaled_norm =  (double*) malloc(N_full  * sizeof(double));
        if(scaled_norm == NULL ) { printf("Memory allocation failed!\n"); exit(1); }
        computeScaledNorm(X, V, Y,  N_full, N,  D, no_dims, mask_V, scaled_norm);
    }

    // Normalize input data (to prevent numerical problems)
    if (verbose) {
        printf("Computing input similarities...\n");
    }
    start = clock();

    double* P; int* row_P; int* col_P; double *dYhat; double* Q; double* beta_P_mul_Dist_sum;

    // Compute asymmetric pairwise input similarities
    P = (double*) malloc(N * K * sizeof(double));
    Q = (double*) malloc(N * K * sizeof(double));
    double* sum_Ps = (double*) malloc(N * sizeof(double));
    beta_P_mul_Dist_sum = (double*) malloc(N * sizeof(double));
    dYhat =  (double*) malloc(N * K * no_dims * sizeof(double));
    if(P == NULL || sum_Ps == NULL || Q == NULL || dYhat == NULL )
    { printf("Memory allocation failed!\n"); exit(1); }
    computeDirectionGaussianPerplexity(X,  V,  Y,  mask_V, N_full,  N,  K, D,  no_dims, true, // with_ii
            &row_P, &col_P,  P, beta_P_mul_Dist_sum, H_P, perplexity, separate_threshold, verbose);
    int ind_nK = 0;
    for(int n = 0; n < N; n++){
//            sum_Ps[n] = DBL_MIN;
        sum_Ps[n] = DBL_MIN;// set P_ii = 1
        ind_nK = n*K;
        for(int k = 0; k < K; k++){
            sum_Ps[n] += P[ind_nK + k];
        }

    }
    for(int n = 0; n < N; n++){
        ind_nK = n*K;
        for(int k = 0; k < K; k++){
            P[ind_nK + k] /= sum_Ps[n];
        }
    }
    free(sum_Ps); sum_Ps = NULL;
    computeUnitYhatMatrix(row_P, col_P, Y,  mask_V,  N_full,  N,   K,  no_dims,  dYhat, verbose);
    computeDeltaYhatMatrix(dYhat, N, K, no_dims, verbose );
    end = clock();

	// Perform main training loop
    if (verbose) {
        printf("Done in %4.2f seconds (sparsity = %f)!\nLearning embedding...\n", (float) (end - start) / CLOCKS_PER_SEC, (double) row_P[N] / ((double) N * (double) N));
    }

//        double* Yhatfull = (double *) malloc(N*N_full*no_dims * sizeof(double));
//        double* meanW = (double *) malloc(N*no_dims * sizeof(double));
//        if(Yhatfull == NULL|| meanW == NULL) { printf("Memory allocation failed!\n"); exit(1); }
//        computeUnitYhatMatrix(Y, mask_V, N_full, N,  no_dims,
//            Yhatfull, verbose);
//        for(int i=0; i< N*no_dims; i++) meanW[i] = 0;
//        for(int n=0; n<N; n++){
//            int ind_nd = n*no_dims;
//            int ind_nNfulld = n*N_full*no_dims;
//            for(int m=0; m<N_full; m++){
//                if(m!=n){
//                    int ind_md = m*no_dims;
//                    for(int d=0; d<no_dims;d++){
//                        meanW[ind_nd +d] += Yhatfull[ind_nNfulld + ind_md +d];
//                    }
//                }
//            }
//        }
//        free(Yhatfull); Yhatfull=NULL;
//        double N_full_sub_one = (double ) N_full - 1.;
//        for(int i=0; i< N*no_dims; i++) meanW[i] /= N_full_sub_one;


    for(int i=0; i<N_full*no_dims; i++){ W_full[i] = 0;}
    int n_curr =0;
    for(int n=0; n<N_full; n++){
        if(!mask_V[n]){
            // P: N x K
            // W: N x no_dims
            int ind_nd = n*no_dims;
            int ind_ncurrK = n_curr*K;
            int ind_ncurrKd = n_curr*K*no_dims;
            for(int k=0; k<K; k++){
                int ind_kd = k*no_dims;
                for(int d=0; d<no_dims; d++){
                    // Yhat N*N_full*no_dims
                     W_full[ind_nd+d] += P[ind_ncurrK+k] * dYhat[ind_ncurrKd + ind_kd + d];
                }
            }
            n_curr += 1;
         }
    }

    unitLength(W_full, N_full, no_dims);

    if(n_curr>N) {printf("N_full:%i, n_cur: %i > N: %i\n",N_full, n_curr,N); exit(1);}

    // fill the direction (with_norm)
    for(int n=0; n<N_full; n++){
        int ind_nfD = n*no_dims;
        if(!mask_V[n]){
            if(with_norm){
                for(int d=0; d<no_dims; d++) W_full[ind_nfD + d] *= scaled_norm[n];
            }
        }else{
            for(int d=0; d<no_dims; d++) W_full[ind_nfD + d] = .0;
        }
    }

    end = clock(); total_time += (float) (end - start) / CLOCKS_PER_SEC;

    // Clean up memory
    free(beta_P);
    free(H_P);
    free(dYhat); dYhat=NULL;
    free(P);
    free(row_P); row_P = NULL;
    free(col_P); col_P = NULL;
    free(beta_P_mul_Dist_sum);
    if(with_norm) free(scaled_norm);
    if (verbose) {
        printf("Fitting performed in %4.2f seconds.\n", total_time);
    }
}

// Compute gradient of the DSNE cost function (using Barnes-Hut algorithm)
void DSNE::computeGradient(double* P, double* beta_Q, double* dYhat,
 double* W, int N, int K, int no_dims, double* Q, double* dW)
{
    // g = (p -q) ( - dYhat + cosine W)
    int D = no_dims;
    double* cosineD = (double*) malloc(N*K* sizeof(double));
    double* sum_Qs = (double*) malloc(N * sizeof(double));
    if(cosineD == NULL || sum_Qs == NULL ) { printf("Memory allocation failed!\n"); exit(1); }
    // unitLength W before this function
    computeCosineDistance(dYhat, W, N, K, D, cosineD);
    int ind_nK= 0,ind_nD = 0, ind_nKD = 0, ind_kD =0,  ND = N*D;
//    for(int n=0; n<N; n++) sum_Qs[n] = DBL_MIN;
    for(int n=0; n<N; n++) sum_Qs[n] = 1.; // acounts for Q_ii = 1;
    for(int n =0; n<N; n++){
        ind_nK = n*K;
        for(int k=0; k<K; k++){
            Q[ind_nK + k] = exp( -beta_Q[n] * 2. * (1. - cosineD[ind_nK + k]));
            sum_Qs[n] += Q[ind_nK + k];
        }
    }
    for(int n =0; n<N; n++){
        ind_nK = n*K;
        for(int k=0; k<K; k++){
            Q[ind_nK + k] /= sum_Qs[n] ;
        }
    }

    for(int i=0; i < ND; i++) dW[i]=.0;
    for(int n =0; n<N; n++){
        ind_nD = n*D;
        for(int d=0; d<D; d++){
            ind_nK = n*K;
            ind_nKD = n*K*D;
            ind_nD = n*D;
            for(int k=0; k<K; k++){
                ind_kD = k*D;
                dW[ind_nD + d] += ( P[ind_nK+k] - Q[ind_nK+k] ) * (-dYhat[ind_nKD + ind_kD + d] + cosineD[ind_nK + k]*W[ind_nD +d]);
            }
        }
    }
    free(cosineD); cosineD =NULL;
    free(sum_Qs); sum_Qs =NULL;
}







void DSNE::update_beta_Q_knn(double* P, double* beta_Q, double* dYhat,
    double* W, int N, int K,  int no_dims,
    double perplexity, double* Q){
    // update beta Q to approach the fixed perplexity and also reduce the error
    int D = no_dims;
    double *tmp_cosineD = (double *) malloc(K*sizeof(double));
    double g=0,H=0,  Hdiff =0;
    double log_perplexity = log(perplexity);
    if(tmp_cosineD== NULL){ printf("Memory allocation failed!\n"); exit(1); }
    // only update beta_Q if the gradient direction sign was different with Hdiff sign
    // compute gradient and H
    // g = ( p - q ) * 2 * ( 1 -cos )
    int ind_nD = 0, ind_nKD = 0, ind_nK =0;
    for(int n=0; n<N; n++){
        ind_nD = n*D;
        ind_nKD = n*K*D;
        ind_nK = n*K;
        int iter = 0;
        bool found = false;
		double beta = beta_Q[n];
		double min_beta = -DBL_MAX;
		double max_beta =  DBL_MAX;
		double tol = 1e-5;
        double sum_Q;
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, K, 1, D, 1., dYhat+ind_nKD, D, W+ind_nD, D, 0.0, tmp_cosineD, K);
        for(int k=0; k<K; k++) tmp_cosineD[k] = 2. - 2.* tmp_cosineD[k]; // d_ij
		while(!found && iter < 200) {
//            sum_Q = DBL_MIN;
            sum_Q = 1.; // account for Q_ii = 1;
            for(int k=0; k<K; k++){
                Q[ind_nK + k] = exp(- beta *  tmp_cosineD[k]);
                sum_Q += Q[ind_nK + k];
            }
            for(int k=0; k<K; k++) Q[ind_nK + k]/= sum_Q;
            g=0.0; H=0.0;
            for(int k=0; k<K; k++) {
                g += (P[ind_nK + k] - Q[ind_nK + k])*  tmp_cosineD[k];
                H += beta * tmp_cosineD[k] * Q[ind_nK + k];
            }
            // todo check which is right
            H += log(sum_Q);
            Hdiff = H - log_perplexity;
            if(abs(g)<tol || abs(Hdiff)<tol || g*Hdiff>=0){
                found=true;
            }else{
               if(Hdiff > 0) {
					min_beta = beta;
					if(max_beta == DBL_MAX || max_beta == -DBL_MAX)
						beta *= 2.0;
					else
						beta = (beta + max_beta) / 2.0;
				}
				else {
					max_beta = beta;
					if(min_beta == -DBL_MAX || min_beta == DBL_MAX)
						beta /= 2.0;
					else
						beta = (beta + min_beta) / 2.0;
				}
            }
//            printf("n: %i, iter: %i, beta: %e, g: %e, Hdiff: %e \n", n, iter, beta, g, Hdiff);
            // Update iteration counter
			iter++;

        }
        beta_Q[n] = beta;
//        printf("n: %i, beta_q[n]: %e \n",n, beta);
    }
    double C = evaluateErrorKnn(P,Q,N,K);
//    printf("loss before updating beta_Q: %e, atter updating beta_Q: %e\n", objective_kl.back(),C);
    objective_kl.push_back(C+2);
    objective_kl.push_back(C+1);
    objective_kl.push_back(C);
    objective_dsne.push_back(C);
    free(tmp_cosineD); tmp_cosineD=NULL;
}


void DSNE::update_beta_Q_knn_new(double* P, double* beta_P_mul_Dist_sum, double* beta_Q, double* Yhat, double* W, int N, int K,  int no_dims,
    double perplexity, double* Q){
    int D = no_dims;
    double *tmp_cosineD = (double *) malloc(K*sizeof(double));
    double g=0,H=0,  Hdiff =0;
    double log_perplexity = log(perplexity);
    if(tmp_cosineD== NULL){ printf("Memory allocation failed!\n"); exit(1); }
    // only update beta_Q if the gradient direction sign was different with Hdiff sign
    // compute gradient and H
    // g = ( p - q ) * 2 * ( 1 -cos )
    int ind_nD = 0, ind_nKD = 0, ind_nK =0;
    for(int n=0; n<N; n++){
        ind_nD = n*D;
        ind_nKD = n*K*D;
        ind_nK = n*K;
		double tol = 1e-5;
        double sum_Q;
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, K, 1, D, 1., Yhat+ind_nKD, D, W+ind_nD, D, 0.0, tmp_cosineD, K);
        double Dist_Q_sum = 0;
        for(int k=0; k<K; k++){
            Dist_Q_sum += 2*(1. - tmp_cosineD[k]);
        }
        double beta = beta_P_mul_Dist_sum[n]/ (Dist_Q_sum + FLT_MIN);
//
        sum_Q = 1.; // account for Q_ii = 1;
        for(int k=0; k<K; k++){
            Q[ind_nK + k] = exp(- beta * 2. *(1. - tmp_cosineD[k]));
            sum_Q += Q[ind_nK + k];
        }
        for(int k=0; k<K; k++) Q[ind_nK + k]/= sum_Q;
        g=0.0; H=0.0;
        for(int k=0; k<K; k++) {
            g += (P[ind_nK + k] - Q[ind_nK + k])*(1-tmp_cosineD[k]);
            H += beta * (2. - 2. * tmp_cosineD[k]) * Q[ind_nK + k];
        }
        // todo check which is right
        H += log(sum_Q);
        Hdiff = H - log_perplexity;
        bool found =false;
        if(abs(g)<tol || abs(Hdiff)<tol || g*Hdiff>=0){
            found=true;
        }
        printf("n:%i, found: %i, Hdiff: %e, g: %e\n", n, found, Hdiff, g);
        beta_Q[n] = beta;
    }
    double C = evaluateErrorKnn(P,Q,N,K);
    printf("loss before updating beta_Q: %e, atter updating beta_Q: %e\n", objective_kl.back(),C);
    objective_kl.push_back(C+2);
    objective_kl.push_back(C+1);
    objective_kl.push_back(C);
    objective_dsne.push_back(C);
    free(tmp_cosineD); tmp_cosineD=NULL;
}



// Evaluate DSNE cost function (approximately)
double DSNE::evaluateErrorKnn(double* P, double* Q, int N, int K )
{
//    printf("begin evaluateError aproximately\n");
    // Sum DSNE error
    double C = .0, tmp_log=0;
    int ind_nK = 0;
	for(int n = 0; n < N; n++) {
	    ind_nK = n*K;
		for(int k = 0; k < K; k++) {
		    tmp_log = log((P[ind_nK + k] + FLT_MIN) / (Q[ind_nK + k] + FLT_MIN));
		    if(isnan(tmp_log)){printf("n: %i, k: %i, P_nm: %f, Q_nm: %f, tmp_log: %f \n",
		        n, k, P[ind_nK + k], Q[ind_nK + k], tmp_log );}
//            C += P[n * N + m] * log((P[n * N + m] + FLT_MIN) / (Q[n * N + m] + FLT_MIN));
            C += P[ind_nK + k] * tmp_log;
		}
	}
	C /=(double) N;

	return C;
}


bool DSNE::check_convergence(int type, int window_size, double epsilon_kl, double epsilon_dsne) {
  float obj_new, obj_old;
  switch (type) {
  case 0:
    // upaate W
    // compute new window mean
    obj_old = 0;
    obj_new = 0;
    for (int i = 0; i < window_size; i++) {
      obj_old += objective_kl[objective_kl.size() - 2 - i];
      obj_new += objective_kl[objective_kl.size() - 1 - i];
    }
    if ((obj_old - obj_new) / abs(obj_old) < epsilon_kl) {
      return(true);
    } else { return(false); }
  case 1:
    // dsne
    obj_old = objective_dsne[objective_dsne.size() - 2];
    obj_new = objective_dsne[objective_dsne.size() - 1];
    bool convergent_objective =(obj_old - obj_new) / abs(obj_old) < epsilon_dsne;
    if (convergent_objective) { return(true); }
    else { return(false);  }
  }

  // gives warning if we don't give default return value
  return(true);
}





// Compute input similarities with a fixed perplexity using ball trees (this function allocates memory another function should free)
void DSNE::computeDirectionGaussianPerplexity(double* X, double* V, double* Y,  bool* mask_V, int N_full,
    int N, int K, int D, int no_dims,
    bool with_ii,
    int** _row_P, int** _col_P, double* P, double* beta_P,
    double* H_P,
    double perplexity, double separate_threshold,
    bool verbose) {
    // Note normalized V previously
    if(perplexity > K) printf("Perplexity should be lower than K!\n");

    // Allocate the memory we need
    *_row_P = (int*)    malloc((N + 1 ) * sizeof(int));
    *_col_P = (int*)    calloc(N * (K), sizeof(int));
    if(*_row_P == NULL || *_col_P == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    int* row_P = *_row_P;
    int* col_P = *_col_P;
    double* cur_P = (double*) malloc((N -1) * sizeof(double));
    if(cur_P == NULL) { printf("Memory allocation failed!\n"); exit(1); }
    double* cosine_distances = (double *) malloc(K*sizeof(double));
    double* buff_x = (double *) malloc(D * sizeof(double));
    double* buff_y = (double *) malloc(no_dims * sizeof(double));
    double* tmp_dXhat = ( double* ) malloc(K*D*sizeof(double));
    double* tmp_dXhat_norm_square_plus_one =  ( double* ) malloc(K*sizeof(double));
    if(cosine_distances == NULL || buff_x == NULL || buff_y == NULL || tmp_dXhat==NULL || tmp_dXhat_norm_square_plus_one == NULL )
    { printf("Memory allocation failed\n"); exit(1);}

    row_P[0] = 0;
    // should provide previously, the common element do not has exactly K elements
    for(int n = 0; n < N; n++) row_P[n + 1] = row_P[n] + K;

    // Build ball tree on data set
    VpTree<DataPoint, euclidean_distance>* tree = new VpTree<DataPoint, euclidean_distance>();
    vector<DataPoint> obj_X(N, DataPoint(D, -1, X));
    // Build the tree do not use the P information, just provide [0,1] closely distance representaion
    for(int n = 0; n < N; n++) obj_X[n] = DataPoint(D, n, X + n * D);
    tree->create(obj_X);

    // Loop over all points to find nearest neighbors
    if (verbose) {
        printf("Building tree...\n");
    }
    vector<DataPoint> indices;
    vector<double> distances;
    // K_additional for too close points ||x_i - x_j||
    int K_additional = 6, K_total = -1;
    bool search_success = false;
    int  ind_n, ind_m, n_curr, k_curr;
    n_curr = 0;
    for(int n = 0; n < N_full; n++){
        if (verbose) {
            if(n % 10000 == 0) printf(" - point %d of %d\n", n, N);
        }
        if(mask_V[n]) continue;
        search_success = false;
        K_additional = 6;
        double tmp_dist_x_ij=0, tmp_dist_y_ij = 0;
        while(!search_success){
            // Find nearest neighbors
            indices.clear();
            distances.clear();
            // ii in the index
            K_total = K + K_additional;
            tree->search(obj_X[n], K_total+1, &indices, &distances);
            // record the distance in the val to accelerate the later computation
            ind_n = n * no_dims;
            k_curr = 0;
            for(int k = 0; k < K_total; k++) {
                tmp_dist_x_ij = distances[ k + 1 ];
                ind_m = indices[k + 1].index()*no_dims;
                for(int d=0; d < no_dims; d++){
                    buff_y[d] = Y[ind_n + d] - Y[ind_m + d];
                }
                tmp_dist_y_ij = dmax(no_dims, buff_y,1,0);
//                printf("i: %i, j: %i, k: %i, k_curr: %i, dxij: %e, dyij, %e, thershold:%e \n",
//                 n, indices[k + 1].index(),k, k_curr,tmp_dist_x_ij, tmp_dist_y_ij, separate_threshold  );
                if(tmp_dist_y_ij > separate_threshold && tmp_dist_x_ij > separate_threshold){
                    col_P[row_P[n_curr] + k_curr] = indices[k + 1].index();
                    k_curr += 1;
                    if(k_curr==K) {
                        search_success = true;
                        break;
                    }
                }
            }
            if(K_additional > N_full - K -2)
            { printf("The x or y points close to each other closely! Checking the data! find k_curr: %i, K: %i\n",k_curr, K); exit(1);}
            if(!search_success) K_additional += 6;
        }
        n_curr += 1;
    }

    //Search for beta (1/(2sigma^2))

    n_curr = 0;
    for(int n = 0; n < N_full; n++) {
        if (verbose) {
            if(n % 10000 == 0) printf(" - point %d of %d\n", n, N);
        }
        if(mask_V[n]) continue;

        // compute the cosine distance < x_j - x_i, v_i>/ || x_j - x_i || || v_i ||
        int flat_i0, k, ind_n, ind_m ;
        flat_i0 =row_P[n_curr];
        double max_x_ij =0, max_v = 0;
        ind_n = n*D;
        max_v = dmax(D, V+ind_n,1, 1e-9);
        if(max_v>1){
           for(int d = 0; d < D; d++) V[ind_n +d] /= max_v;
        }
        max_v = cblas_dnrm2(D, V + ind_n, 1);

        for(int i=row_P[n_curr]; i < row_P[n_curr+1] ; i++){
            // compute Xhat[i,j] = x_j - x_i
            k = i - flat_i0;
            ind_m = col_P[i]*D;

            for(int d=0; d<D; d++){
                buff_x[d] = X[ ind_m + d] - X[ ind_n + d ];
            }
            max_x_ij = dmax(D, buff_x, 1, 1e-9);
            if(max_x_ij>1.){
               for(int d = 0; d < D; d++){ buff_x[d] /= max_x_ij; }
            }
            max_x_ij = cblas_dnrm2(D, buff_x, 1);
            if(max_x_ij < separate_threshold || max_x_ij < FLT_MIN){
                 printf("n: %i, m: %i, max_x_ij : %e is too small \n",n, col_P[i], max_x_ij);
            }
            int kD = k*D;
            for(int d=0; d<D; d++){
                tmp_dXhat[kD + d] = buff_x[d]/(max_x_ij+FLT_MIN) ;
            }
        }
        for(int d=0; d<D; d++){ buff_x[d] = 0;} // for mean_dy_i
        for(int k=0; k<K; k++){
            // compute mean_dy_i
            int ind_kD = k*D;
            for(int d=0; d<D; d++){
                buff_x[d]+= tmp_dXhat[ind_kD +d];
            }
        }
        for(int d=0; d<D; d++){
            // compute mean_dy_i
            buff_x[d] /= (double)K;
        }
        for(int k=0; k<K; k++){
            // compute dXhat[ij] = x_j - x_i /||x_j - x_i||
            int ind_kD = k*D;
            double tmp_sum = 0;
            for(int d=0; d<D; d++){
                tmp_dXhat[ind_kD +d] -= buff_x[d] ;
                tmp_sum += tmp_dXhat[ind_kD +d] * tmp_dXhat[ind_kD +d] ;
            }
            tmp_dXhat_norm_square_plus_one[k] = 1. +  tmp_sum;
        }

        // unitLength tmp_dXhat
         for(int k=0; k<K; k++){
            // compute dXhat[ij] = x_j - x_i /||x_j - x_i||
            int ind_kD = k*D;
            double tmp_norm = cblas_dnrm2(D, tmp_dXhat + ind_kD,1);
            if(tmp_norm < FLT_MIN){
                printf("n: %i, k:%i, dXhat_ij_norm too small: %e \n", n, k,tmp_norm );
            }
            for(int d=0; d<D; d++){
                tmp_dXhat[ind_kD +d] /= (tmp_norm+FLT_MIN) ;
            }

        }


        for(int i=row_P[n_curr]; i < row_P[n_curr+1] ; i++){
              // D_ij
              k = i - flat_i0;
              cosine_distances[k]=0;
              int ind_kD = k*D;
              cosine_distances[k] = cblas_ddot(D, V+ind_n, 1, tmp_dXhat+ind_kD, 1);
              cosine_distances[k] /= (max_v+FLT_MIN); // cos_ij
//              cosine_distances[k] = tmp_dXhat_norm_square_plus_one[k] - 2.* cosine_distances[k]; // d_ij
              cosine_distances[k] = 2.*(1. -cosine_distances[k]); // d_ij
              if(max_v< separate_threshold || max_v < FLT_MIN){
                 printf("n: %i  max_v : %e is too small \n",n, max_v);
              }
        }

        // Initialize some variables for binary search
		bool found = false;
		double beta = 1.0;
		double min_beta = -DBL_MAX;
		double max_beta =  DBL_MAX;
		double tol = 1e-5;

		// Iterate until we found a good perplexity
		int iter = 0; double sum_P; double H = .0;
		while(!found && iter < 200) {

			// Compute Gaussian kernel row
			for(int m = 0; m < K; m++) cur_P[m] = exp( - beta * cosine_distances[m] );
			// Compute entropy of current row
			if(with_ii){
			    sum_P = 1.; // accout for P_ii = 1; K do not contain i
			}else{
			    sum_P = DBL_MIN;
			}
//
			for(int m = 0; m < K; m++) sum_P += cur_P[m];
			H = .0;
			for(int m = 0; m < K; m++) H += beta *  ( cosine_distances[m] * cur_P[m] );
			H = (H / sum_P) + log(sum_P) ;

			// Evaluate whether the entropy is within the tolerance level
			double Hdiff = H - log(perplexity) ;
			if(Hdiff < tol && -Hdiff < tol) {
				found = true;
			}
			else {
				if(Hdiff > 0) {
					min_beta = beta;
					if(max_beta == DBL_MAX || max_beta == -DBL_MAX)
						beta *= 2.0;
					else
						beta = (beta + max_beta) / 2.0;
				}
				else {
					max_beta = beta;
					if(min_beta == -DBL_MAX || min_beta == DBL_MAX)
						beta /= 2.0;
					else
						beta = (beta + min_beta) / 2.0;
				}
			}

			// Update iteration counter
			iter++;
		}
		if(!found) printf("n: %i, perplexity not find the beta\n", n);
        beta_P[n_curr] = beta;
//        printf("n: %i, beta_p[n]: %e \n",n_curr, beta);
        H_P[n_curr] = H;
//        printf("n: %i, perplexity: %f, beta: %f, H: %f, Hdiff: %f, found: %i\n",
//        n, perplexity, beta, H, H-log(perplexity), found);
		// Row-normalize current row of P and store in matrix
        for(int m = 0; m < K; m++) cur_P[m] /= sum_P;
//        for(int m = 0; m < K; m++) printf("i: %i, j: %i, P_j|i: %5.2f \n", n,col_P[row_P[n_curr]+m], cur_P[m]);
        ind_n = n_curr * K;
        for(int m = 0; m < K; m++) {
            P[ind_n + m] = cur_P[m];
        }
        n_curr += 1;
    }

    // Clean up memory
    obj_X.clear();
    free(cur_P);
    free(cosine_distances);
    free(buff_x);
    free(buff_y);
    free(tmp_dXhat);
    free(tmp_dXhat_norm_square_plus_one);
    delete tree;
//    printf("End of computeGaussionPreplexity approximately\n");
}

void DSNE::computeDirectionGaussianPerplexity(double* X, double* V, double* Y,  bool* mask_V, int N_full, int N,  int K,  int D, int no_dims,
        bool with_ii,
        int* row_P, int* col_P, double* P, double* beta_P, double* H_P, double perplexity, double separate_threshold,
        bool verbose)
{
    printf("todo\n");
}

void DSNE::computeUnitYhatMatrix(int* row_P, int* col_P, double*Y, bool* mask_V, int N_full, int N, int K, int no_dims,
    double* Yhat, bool verbose){
    int D = no_dims;
    int n_curr = 0, ind_n =0, ind_m =0, ind_kD=0;
    double* buff = (double* ) malloc( D * sizeof(double));
    if(buff == NULL){ printf("Memory allocation failed\n"); exit(1);}
    double max_y_ij;
    for(int n=0; n < N_full; n++){
        if (verbose) {
            if(n % 10000 == 0) printf(" - point %d of %d\n", n, N_full);
        }
        if(mask_V[n]) continue;
        ind_n = n * D; // here check that D==2
        int ind_ncurrKD = n_curr*K*D;
        for(int k=0; k<K; k++){
            ind_kD = k*D;
            ind_m = col_P[row_P[n_curr] + k] * D;
            for(int d=0; d<D; d++){
                buff[d] =  Y[ind_m + d] - Y[ind_n + d];
            }
            max_y_ij = dmax(D, buff, 1, 0);
            if(max_y_ij>1){
               for(int d=0; d<D; d++){
                    buff[d] /= max_y_ij;
               }
            }
            max_y_ij = cblas_dnrm2(D, buff, 1);
            if(max_y_ij>FLT_MIN){
                for(int d=0; d<D; d++){
                    Yhat[ind_ncurrKD + ind_kD + d] = buff[d] / max_y_ij;
                }
            }else{
                printf("i: %i, j: %i, norm_y_ij: %e too small \n",n, col_P[row_P[n_curr] + k], max_y_ij );
                for(int d=0; d<D; d++){
                    Yhat[ind_ncurrKD + ind_kD + d] = 0.0;
                }
            }
        }
        n_curr += 1;
    }

    free(buff); buff = NULL;
}

void DSNE::computeDeltaYhatMatrix(double* Yhat, int N, int K, int no_dims,  bool verbose){
    // Yhat: N x K x no_dims
    // dYhat_norm_square_plus_one: N x K,  dYhat_norm_square_plus_one[ik] = 1 + || dYhat[ik] ||^2
    double* dYhat_mean = (double* ) malloc( N * no_dims * sizeof(double) ); // \bar y_i
    for(int i =0; i< N*no_dims; i++ ) dYhat_mean[i] = 0;
    for(int n = 0; n<N; n++){
        int ind_nKd = n*K*no_dims;
        int ind_nd = n*no_dims;
        for(int k=0; k <K; k++){
            int ind_kd = k*no_dims;
            for(int d = 0; d<no_dims;d++){
                dYhat_mean[ind_nd +d ] += Yhat[ind_nKd + ind_kd + d ];
            }
        }
    }

    for(int n = 0; n<N; n++){
        int ind_nd = n*no_dims;
        for(int d = 0; d<no_dims;d++){
            dYhat_mean[ind_nd +d ] /= (double) K;
        }
    }

    // compute dYhat
    for(int n = 0; n<N; n++){
        int ind_nKd = n*K*no_dims;
        int ind_nd = n*no_dims;
        for(int k=0; k <K; k++){
            int ind_kd = k*no_dims;
            double tmp_sum = 0;
            for(int d = 0; d<no_dims;d++){
                Yhat[ind_nKd + ind_kd + d ] -= dYhat_mean[ind_nd +d ] ;
                tmp_sum += Yhat[ind_nKd + ind_kd + d ]  * Yhat[ind_nKd + ind_kd + d ];
            }
//            dYhat_norm_square_plus_one[ind_nK +k ] = 1. + tmp_sum;
        }
    }

    // unit dYhat
    for(int n = 0; n<N; n++){
        int ind_nKd = n*K*no_dims;
        for(int k=0; k <K; k++){
            int ind_kd = k*no_dims;
            double tmp_norm = cblas_dnrm2(no_dims, Yhat + (ind_nKd + ind_kd),1 );
            if(tmp_norm < FLT_MIN) printf("n: %i, k: %i, dYhat_ij norm too small:%e\n", n,k,tmp_norm);
            for(int d = 0; d<no_dims;d++){
                Yhat[ind_nKd + ind_kd + d ] /= (tmp_norm + FLT_MIN) ;
            }
        }
    }

}

void DSNE::computeDeltaYhatMatrix(double* Yhat, int N, int K, int no_dims, double* Yhat_mean, bool verbose){
    printf("todo\n");
}

void DSNE::computeMaskV(double* V, int N_full, int D, bool* mask_V, int &N, double threshold){
    double v_max = 0;
    N=0;
    for(int n=0; n<N_full; n++){
        v_max = dmax(D, V+n*D, 1, 0);
        if(v_max < threshold){ mask_V[n] = true; }
        else { mask_V[n] = false; N+=1; }
    }
}

void DSNE::computeScaledNorm(double* X, double* V, double* Y, int N_full, int N, int D, int no_dims,
    bool* mask_V, double *scaled_norm){
    double v_max = 0;
    int ind_nD = 0;
    double scale = 0;
    double sum_norm_Y_div_norm_X = DBL_MIN;
    for(int n=0; n< N_full; n++){
        if(mask_V[n]){
            continue;
        }else{
            sum_norm_Y_div_norm_X += ( cblas_dnrm2(no_dims, Y+n*no_dims, 1) + no_dims ) / ( cblas_dnrm2(D, X+n*D, 1) + D);
        }
    }
    scale = sum_norm_Y_div_norm_X / (double) N;

    for(int n=0; n<N_full; n++){
        if(mask_V[n]){
            scaled_norm[n] = 0.0;
        }else{
             v_max = dmax(D, V+n*D, 1, 0);
             ind_nD = n*D;
             if(v_max>1){
                for(int d=0; d<D; d++) V[ind_nD +d ] /= v_max;
                scaled_norm[n] = scale * v_max * cblas_dnrm2(D, V+ind_nD, 1);
             }else{
                scaled_norm[n] = scale * cblas_dnrm2(D, V+ind_nD, 1);
             }
        }
    }
}

void DSNE::computeCosineDistance(double* Yhat, double* W, int N, int K, int no_dims, double* cosineD){
    // Yhat: N * K * no_dims
    // W:  N * no_dims
    // make sure that W being unit
    // cosineD N * K
    int D = no_dims;
    for( int i=0; i< N*K; i++ ) cosineD[i] =0.0;
    int ind_nD = 0, ind_nKD = 0, ind_nK =0;
    for(int n=0; n<N; n++){
        ind_nD = n*D;
        ind_nKD = n*K*D;
        ind_nK = n*K;
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, K, 1, D,  1., Yhat+ind_nKD, D, W+ind_nD, D, 0.0, cosineD+ind_nK, K);
    }
//    printf("Cosine distances, %i x %i\n", N, K);
//    for(int n=0; n<N; n++){
//        for(int k=0; k<K; k++){
//            if(k<K-1){
//                printf("%5.5f\t",cosineD[n*K + k]);
//            }else  printf("%5.5f\n",cosineD[n*K + k]);
//        }
//    }
}

// Makes velocity unit-length
void DSNE::unitLength(double* V, int N, int D) {

	// Compute data mean
    double scale =1.;
    double tmp = 1./sqrt((double)D);
	for(int n = 0; n < N; n++) {
	    scale = dmax(D, &(V[n*D]),1,FLT_MIN);
	    int ind_n = n*D;
	    if(scale > 1.){
	        for(int d=0; d < D; d++){
	            V[ind_n + d] /= scale;
	        }
	    }else if(scale<=FLT_MIN){
	        for(int d=0; d < D; d++){
	           V[ind_n + d] = tmp;
	        }

	        continue;
	    }

	    scale = cblas_dnrm2(D, &V[ind_n],1);
	    if(scale >FLT_MIN){
	        for(int d=0; d < D; d++){
	            V[ind_n + d] /= scale;
	        }
	    }
	    else{
	        for(int d=0; d < D; d++){
	            V[ind_n+ d] = tmp;
	        }
	    }
    }

	for(int i = 0; i<N*D; i++){
//	    if(isnan(V[i])|| V[i] >1. || V[i]<-1.){
//	        printf("i %d, V[i]: %f \n", i, V[i]);
//	    }
	    if(isnan(V[i])) { V[i]= 0;}
	    else if(V[i] >1.) { V[i] = 1.;}
	    else if(V[i] <-1.){ V[i] = -1.; }
	}
}

void DSNE::clip(double* X, int N, int D, double min_val, double max_val){
    // clip the value of X to [min_val,max_val]
    for(int i=0; i< N*D; i++) {
        if(X[i]>max_val){
            X[i] = max_val;
        }else if(X[i] < min_val){
            X[i] = min_val;
        }
    }
}
double DSNE::dmax(unsigned int N, double* X, unsigned int inc, double max_lower_bound){
    // find the absolute max value of X with size N and step size inc
    int top = 1 + (N-1)*inc;
    double tmp = 0.;
    double max = max_lower_bound;
    for(int i=0; i<top; i=i+inc){
        tmp = abs(X[i]);
        if(max < tmp){
            max = tmp;
        }
    }
    return max;
}

// Generates a Gaussian random number
double DSNE::randn() {
	double x, y, radius;
	do {
		x = 2 * (rand() / ((double) RAND_MAX + 1)) - 1;
		y = 2 * (rand() / ((double) RAND_MAX + 1)) - 1;
		radius = (x * x) + (y * y);
	} while((radius >= 1.0) || (radius == 0.0));
	radius = sqrt(-2 * log(radius) / radius);
	x *= radius;
	y *= radius;
	return x;
}


void DSNE::write_matrix(double* data, int N, int D, FILE* fp) {
    // wite the matrix data with size N x D
	if(fp == NULL) {
		printf("Error: could not open data file.\n");
		exit(1);
	}
    for(int n=0; n<N; n++){
        for(int d=0; d<D; d++){
            if(d<D-1){
                fseek(fp,0,SEEK_END);
                fprintf(fp, "%f\t",data[n*D+d]);
            }else{
                fseek(fp,0,SEEK_END);
                fprintf(fp,"%f\n",data[n*D+d]);
            }
        }
    }
    //	printf("Wrote the %i x %i data matrix successfully!\n", N, D);
}

