// EMG_PR.cpp : Defines the entry point for the console application.
//

//#include "stdafx.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "math.h"

#ifdef _WIN32
#define DLL_EXPORT __declspec(dllexport)
#else
#define DLL_EXPORT
#endif

int Choleski_LU_Decomposition(float *A, int n);
int Choleski_LU_Solve(float *LU, float B[], float x[], int n);

/*  Matrix Multiply
    C=A*B
    A:(am x an)
    B:(an x bn)
    C:(am x bn)
*/
void mulAB(float *A,float *B,float *C,int am,int an,int bn)
{
    int i,j,l,u;

    for (i=0; i<am; i++)
        for (j=0; j<bn; j++)
        {
            u=i*bn+j; C[u]=0.0;
            for (l=0; l<an; l++)
                C[u]=C[u]+A[i*an+l]*B[l*bn+j];
        }
    return;
}

/*  Matrix Addition
    C=A+B
    A,B,C:(m x n)
*/
void addition(float *A,float *B,float *C, int m,int n)
{
    int i,j;

    for (i=0; i<m; i++)
        for (j=0; j<n; j++)
          {
              C[i*m+j]=A[i*m+j]+B[i*m+j];
          }
    return;
}
/*  Matrix Subtraction
    C=A-B
    A,B,C:(m x n)
*/
void subtract(float *A,float *B,float *C, int m,int n)
{
    int i,j;

    for (i=0; i<m; i++)
        for (j=0; j<n; j++)
          {
              C[i*m+j]=A[i*m+j]-B[i*m+j];
          }
    return;
}

/*  Matrix Transpose
    Input: A (m x n)
    Output: B (n x m)
*/
void Transpose(float *A,float *B,int m,int n)
{
    int i,j;

    for (j=0; j<n; j++)
        for (i=0; i<m; i++)
          {
              B[j*m+i]=A[i*n+j];
          }
    return;
}

/*  Covariance Matrix
    Input: A (row x col)
    Output: B (col x col)
*/
void cov(float *A,float *B,int row,int col)
{
    int i,j;
    float *tmp;
    float *tmp1;
    float *t_tmp1;
    float sum=0;

    printf("row:%d, col:%d\n",row,col);
    t_tmp1=(float*)malloc(col*row*sizeof(float));
    tmp1=(float*)malloc(row*col*sizeof(float));
    tmp=(float*)malloc(col*sizeof(float));

    memset(tmp,0,col*sizeof(float));
    memset(tmp1,0,row*col*sizeof(float));
    memset(t_tmp1,0,row*col*sizeof(float));
    //printf("tmp:%d, tmp1:%d, t_tmp1:%d\n",&tmp[0],&tmp1[0],&t_tmp1[0]);
    //printf("cov1\n");
    //calculate mean of each column
    for (i=0; i<col; i++)
    {
        sum = 0;
        for (j=0; j<row; j++)
        {
            sum+=A[j*col+i];
        }
        tmp[i]=sum/row;
		//printf("%f\n",tmp[i]);
    }

    //substract matrix A with mean
    for (i=0; i<col; i++)
    {
        for (j=0; j<row; j++)
        {
            tmp1[j*col+i]=A[j*col+i]-tmp[i];
            //float tmptmp = tmp1[j*n+i];
        }
    }
    //printf("cov2\n");
    //multiply tmp1 with transpose of tmp1
    Transpose(tmp1,t_tmp1,row,col);
    //printf("cov3\n");
    mulAB(t_tmp1,tmp1,B,col,row,col);
    //printf("cov4\n");
    //calculate mean of each element of B
    //printf("tmp:%d, tmp1:%d, t_tmp1:%d\n",&tmp[0],&tmp1[0],&t_tmp1[0]);
   // printf("tmp:%d, tmp1:%d, t_tmp1:%d\n",&tmp[col-1],&tmp1[row*col-1],&t_tmp1[row*col-1]);
    for (i=0; i<col; i++)
    {
        for (j=0; j<col; j++)
        {
            B[j*col+i]=B[j*col+i]/(row-1);
        }
    }

    free(t_tmp1);
    free(tmp1);
    //printf("cov5\n");
    free(tmp);

    return;
}
/*
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% TDFEATS	Compute four time domain features
%   1. mav: mean absolute value
%   2. len: waveform length
%   3. zero_count: number of zero crossings
%   4. turns: number of slope sign changes
%
%     Inputs:
%         DataSet_in: pointer to the matrix of the raw EMG data (just one window)
%         win_length: window length
%         channel: number of channels in DataSet_in
%         Nframe: the index of the window in the training data set.
%     Outputs:
%         Features: the output feature matrix (feature (mave, len, zero_count, turns) x window)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
*/
DLL_EXPORT void tdfeats(float *DataSet_in, int win_length, int channel, int Nframe,
                        float *Features, float deadzone_zc, float deadzone_turn,
                        int scale_mav, int scale_zc, int td_features)
{
    float ruler=0;
    float rulersq;
    float lscale;
    float tscale;

    int Nsig;

    int i,j;
    float sum=0;
    float *mean;
	float *DataSet;

    float zero_count;
    float len;
    float mav;
    float turns;
    int index;
    int range;
    float sum_sig;
    int flag1;
    int flag2;
    int idx;
    float fst;
    float mid;
    float lst;
    int SigNum;

    ruler=1/(float)win_length;
    rulersq=ruler*ruler;
    lscale=(float)win_length/40;
    tscale=((float)win_length/40)*10;


    Nsig=channel;


    mean=(float*)malloc(1*channel*sizeof(float));
	DataSet=(float*)malloc(1*win_length*channel*sizeof(float));

    for(i=0;i<channel;i++)
    {
        sum=0;
        for(j=0;j<win_length;j++)
        {
            sum=sum+DataSet_in[j*channel+i];
        }
        mean[i]=sum/win_length;//mean of a sigal (column)
    }

    for(i=0;i<win_length;i++)
    {
        for(j=0;j<channel;j++)
        {
            DataSet[i*channel+j]=DataSet_in[i*channel+j]-mean[j];
        }
    }

    free(mean);
    /*for(i=0;i<DataSet_Row;i++)
    {
        for(j=0;j<channel;j++)
        {
            printf("%.4f ",DataSet[i*channel+j]);
        }
        printf("\n");
    }
    printf("***************************************\n");*/

    for(SigNum=0;SigNum<Nsig;SigNum++)
    {
        zero_count=0;
        len=0;
        mav=0;
        turns=0;

        index=0;
        sum_sig=0;
        for(range=index;range<index+win_length;range++)
        {
            sum_sig=sum_sig+(float)fabs(DataSet[range*channel+SigNum]);//sum of a signal (a column)
        }
        mav=sum_sig/win_length;//mean of a signal

        flag1=1;
        flag2=1;
        for(i=1;i<win_length-1;i++)//win_length=DATASET_ROW
        {
            idx=i;    //idx=2:DATASET_ROW-1
            fst=(float)fabs(DataSet[(idx-1)*channel+SigNum]);//DataSet[0,0]:DataSet[DATASET_ROW-3,0]
            mid=(float)fabs(DataSet[(idx)*channel+SigNum]);  //DataSet[1,0]:DataSet[DATASET_ROW-2,0]
            lst=(float)fabs(DataSet[(idx+1)*channel+SigNum]);//DataSet[2,0]:DataSet[DATASET_ROW-1,0]

            //% Compute Zero Crossings
            if(((DataSet[(idx)*channel+SigNum]>=0) &&(DataSet[(idx-1)*channel+SigNum]>=0))
                ||((DataSet[(idx)*channel+SigNum]<=0) &&(DataSet[(idx-1)*channel+SigNum]<=0)))
            {
                flag1=flag2;//if (DataSet[1,0]>=0&&DataSet[0,0]>=0)||(DataSet[1,0]<=0&&DataSet[0,0]<=0)
            }else
            {
                if((mid<deadzone_zc)&&(fst<deadzone_zc))//approximately zero
                {
                    flag1=flag2;
                }
                else
                {
                    flag1=(-1)*flag2;
                }
            }
            if(flag1!=flag2)
            {
                zero_count=zero_count+1;
            }
            //% Compute Turns (Slope Changes
            if(((mid>fst)&&(mid>lst))||((mid<fst)&&(mid<lst)))
            {
                //% turns threshold of 15mV (i.e. 3uV noise)
                if((fabs(mid)-fabs(fst))>deadzone_turn||(fabs(mid)-fabs(lst))>deadzone_turn)
                {
                    turns=turns+1;
                }
            }
            //% Compute Waveform Length
            //len=len+(float)sqrt(((fst-mid)/20.0)*((fst-mid)/20.0)+rulersq);
            len=len+(float)sqrt(((fst-mid)/20.0)*((fst-mid)/20.0)+rulersq);//rulersq=(1/DATASET_ROW)^2
        }

        //% Scale the features to normalize for the neural network

        zero_count=(zero_count/scale_zc)*40/win_length;

        //% scaling based on 40 ms

        mav=mav/scale_mav;
        len=(len-1)/lscale;
        turns=turns/tscale;
        Features[td_features*channel*Nframe+SigNum*td_features] = mav;
        Features[td_features*channel*Nframe+SigNum*td_features+1] = len;
        Features[td_features*channel*Nframe+SigNum*td_features+2] = zero_count;
        Features[td_features*channel*Nframe+SigNum*td_features+3] = turns;

    }
	free(DataSet);
}
/*  Feature Normalization in the training phase
    mapstd in Matlab
    Input:
        features: the feature matrix of training data (FEATURE_DIM x num)
        num: number of feature vectors (windows) in the feature matrix
    Output:
        xmean: parameter of the normalization to be used in the testing phase (mean)
        xtd: parameter of the normalization to be used in the testing phase (std)
*/
DLL_EXPORT void feature_normalization(float *features, float * xmean, float * xstd,
                                      int num, int feature_dim)
{
	int j,m;
	//float *tmp;
	//float *tmp1;
	float *xstd_mean;
	//int test_pro;
	float sum=0.0;
	float xstd_sum=0.0;


	xstd_mean=(float*)malloc(1*feature_dim*sizeof(float));

	for(m=0;m<feature_dim;m++)
    {
		sum=0;
		xstd_sum=0;
        for(j=0;j<num;j++)
        {
            sum+=features[m+feature_dim*j];
        }
        xmean[m]=sum/num;

		for(j=0;j<num;j++)
        {
            features[m+feature_dim*j]=features[m+feature_dim*j]-xmean[m];
			xstd_sum+=features[m+feature_dim*j]*features[m+feature_dim*j];
        }
		xstd_mean[m]=(num > 1) ? xstd_sum/(num-1) : 0.0f;
		xstd[m]=(float)sqrt(xstd_mean[m]);
        if (xstd[m] == 0.0f) {
            xstd[m] = 1.0f;
        }

		for(j=0;j<num;j++)
        {
            features[m+feature_dim*j]=features[m+feature_dim*j]/xstd[m];
        }
    }

	free(xstd_mean);

	return;
}
/*  Feature Normalization in the testing phase
    mapstd('apply') in Matlab
    Inputs:
        features: the feature vector to be normalized (FEATURE_DIM x 1)
        xmean: parameter of normalization obtained in the training phase
        xtd: parameter of normalization obtained in the training phase
    Output:
        features: the normalized feature vector
*/
void feature_normalization_apply(float *features, float * xmean, float * xstd, int feature_dim)
{
	int m;

	for(m=0;m<feature_dim;m++)
    {
		features[m]=features[m]-xmean[m];
        features[m]=features[m]/xstd[m];
    }

	return;
}


/*
%%-------LDA testing procedure---------%%
%   Integrate three steps:
%   1. feature extraction
%   2. feature normalization
%   3. classification
% Inputs:
%     TestData: the raw EMG data for one analysis window
%     Wg: parameter of the LDA classifier (feature vector dimension x # of classes)
%     Cg: parameter of the LDA classifier (1 x # of classes)
%     xmean: parameter for normalization
%     xstd: parameter for normalization
% return:
%     test_decision: the classification decision
%
%%% By Xiaorong Zhang, 7/28/2014  %%%%%%%%
*/

DLL_EXPORT int LDA_test(float *TestData, float *Wg, float *Cg, float *xmean, float *xstd,
                        int win_length, int channel, int feature_dim, int num_class,
                        float deadzone_zc, float deadzone_turn,
                        int scale_mav, int scale_zc, int td_features)
{
	int j;
	float *Feature_test;
	float *tmp;
	float *tmp1;
	float maxdata=-9999.0;
	int test_decision;


	maxdata=-9999.0;
	Feature_test = (float*)malloc(feature_dim*sizeof(float));
	tmp=(float*)malloc(1*num_class*sizeof(float));
	tmp1=(float*)malloc(1*num_class*sizeof(float));

	tdfeats(TestData, win_length, channel, 0, Feature_test, deadzone_zc,
            deadzone_turn, scale_mav, scale_zc, td_features);
	feature_normalization_apply(Feature_test,xmean,xstd,feature_dim);

	mulAB(Feature_test,Wg,tmp,1,feature_dim,num_class);
	addition(Cg,tmp,tmp1,1,num_class);

	for(j=0;j<num_class;j++){
		if(tmp1[j]>maxdata)
		{
			maxdata=tmp1[j];
			test_decision=j+1;
		}
	}

	free(Feature_test);
	free(tmp);
	free(tmp1);
	return test_decision;
}

DLL_EXPORT void LDA_train(float *features, int *classes, float *Wg, float *Cg,
                          int feature_dim, int num_class,
                          int win_per_trial, int trial_per_class)
{
    int i, j, c;
    int num_samples = num_class * win_per_trial * trial_per_class;
    float *means = (float*)calloc(num_class * feature_dim, sizeof(float));
    float *pooled_cov = (float*)calloc(feature_dim * feature_dim, sizeof(float));
    float *lu = (float*)malloc(feature_dim * feature_dim * sizeof(float));
    float *rhs = (float*)malloc(feature_dim * sizeof(float));
    float *solution = (float*)malloc(feature_dim * sizeof(float));
    int *counts = (int*)calloc(num_class, sizeof(int));
    float regularization = 1.0e-5f;
    int decomp_status;

    if (!means || !pooled_cov || !lu || !rhs || !solution || !counts || num_samples <= 0) {
        goto cleanup;
    }

    for (i = 0; i < num_samples; i++) {
        c = classes[i] - 1;
        if (c < 0 || c >= num_class) {
            continue;
        }
        counts[c]++;
        for (j = 0; j < feature_dim; j++) {
            means[c * feature_dim + j] += features[i * feature_dim + j];
        }
    }

    for (c = 0; c < num_class; c++) {
        if (counts[c] == 0) {
            continue;
        }
        for (j = 0; j < feature_dim; j++) {
            means[c * feature_dim + j] /= counts[c];
        }
    }

    for (i = 0; i < num_samples; i++) {
        c = classes[i] - 1;
        if (c < 0 || c >= num_class) {
            continue;
        }
        for (j = 0; j < feature_dim; j++) {
            int k;
            float dj = features[i * feature_dim + j] - means[c * feature_dim + j];
            for (k = 0; k < feature_dim; k++) {
                float dk = features[i * feature_dim + k] - means[c * feature_dim + k];
                pooled_cov[j * feature_dim + k] += dj * dk;
            }
        }
    }

    {
        int denom = num_samples - num_class;
        if (denom < 1) {
            denom = 1;
        }
        for (i = 0; i < feature_dim * feature_dim; i++) {
            pooled_cov[i] /= denom;
        }
    }

    memcpy(lu, pooled_cov, feature_dim * feature_dim * sizeof(float));
    for (i = 0; i < feature_dim; i++) {
        lu[i * feature_dim + i] += regularization;
    }
    decomp_status = Choleski_LU_Decomposition(lu, feature_dim);
    while (decomp_status < 0 && regularization < 1.0f) {
        memcpy(lu, pooled_cov, feature_dim * feature_dim * sizeof(float));
        regularization *= 10.0f;
        for (i = 0; i < feature_dim; i++) {
            lu[i * feature_dim + i] += regularization;
        }
        decomp_status = Choleski_LU_Decomposition(lu, feature_dim);
    }

    for (c = 0; c < num_class; c++) {
        float bias = 0.0f;
        float prior = (num_samples > 0) ? ((float)counts[c] / (float)num_samples) : 0.0f;

        for (j = 0; j < feature_dim; j++) {
            rhs[j] = means[c * feature_dim + j];
            solution[j] = 0.0f;
        }

        if (decomp_status == 0) {
            Choleski_LU_Solve(lu, rhs, solution, feature_dim);
        }

        for (j = 0; j < feature_dim; j++) {
            Wg[c + j * num_class] = solution[j];
            bias += means[c * feature_dim + j] * solution[j];
        }
        Cg[c] = -0.5f * bias;
        if (prior > 0.0f) {
            Cg[c] += (float)log(prior);
        }
    }

cleanup:
    free(means);
    free(pooled_cov);
    free(lu);
    free(rhs);
    free(solution);
    free(counts);
}

DLL_EXPORT float LDA_train_accuracy(float *features, int *classes, float *Wg, float *Cg,
                                    int feature_dim, int num_class,
                                    int win_per_trial, int trial_per_class)
{
    int i, j, c;
    int num_samples = num_class * win_per_trial * trial_per_class;
    int num_correct = 0;

    if (num_samples <= 0) {
        return 0.0f;
    }

    for (i = 0; i < num_samples; i++) {
        float max_score = -3.4e38f;
        int decision = 1;
        for (c = 0; c < num_class; c++) {
            float score = Cg[c];
            for (j = 0; j < feature_dim; j++) {
                score += features[i * feature_dim + j] * Wg[c + j * num_class];
            }
            if (score > max_score) {
                max_score = score;
                decision = c + 1;
            }
        }
        if (decision == classes[i]) {
            num_correct++;
        }
    }

    return (float)num_correct / (float)num_samples;
}

////////////////////////////////////////////////////////////////////////////////
//  int Lower_Triangular_Solve(double *L, double *B, double x[], int n)       //
//                                                                            //
//  Description:                                                              //
//     This routine solves the linear equation Lx = B, where L is an n x n    //
//     lower triangular matrix.  (The superdiagonal part of the matrix is     //
//     not addressed.)                                                        //
//     The algorithm follows:                                                 //
//                      x[0] = B[0]/L[0][0], and                              //
//     x[i] = [B[i] - (L[i][0] * x[0]  + ... + L[i][i-1] * x[i-1])] / L[i][i],//
//     for i = 1, ..., n-1.                                                   //
//                                                                            //
//  Arguments:                                                                //
//     double *L   Pointer to the first element of the lower triangular       //
//                 matrix.                                                    //
//     double *B   Pointer to the column vector, (n x 1) matrix, B.           //
//     double *x   Pointer to the column vector, (n x 1) matrix, x.           //
//     int     n   The number of rows or columns of the matrix L.             //
//                                                                            //
//  Return Values:                                                            //
//     0  Success                                                             //
//    -1  Failure - The matrix L is singular.                                 //
//                                                                            //
//  Example:                                                                  //
//     #define N                                                              //
//     double A[N][N], B[N], x[N];                                            //
//                                                                            //
//     (your code to create matrix A and column vector B)                     //
//     err = Lower_Triangular_Solve(&A[0][0], B, x, n);                       //
//     if (err < 0) printf(" Matrix A is singular\n");                        //
//     else printf(" The solution is \n");                                    //
//           ...                                                              //
////////////////////////////////////////////////////////////////////////////////
//                                                                            //
int Lower_Triangular_Solve(float *L, float B[], float x[], int n)
{
   int i, k;

//         Solve the linear equation Lx = B for x, where L is a lower
//         triangular matrix.

   for (k = 0; k < n; L += n, k++) {
      if (*(L + k) == 0.0) return -1;           // The matrix L is singular
      x[k] = B[k];
      for (i = 0; i < k; i++) x[k] -= x[i] * *(L + i);
      x[k] /= *(L + k);
   }

   return 0;
}
////////////////////////////////////////////////////////////////////////////////
//  int Upper_Triangular_Solve(double *U, double *B, double x[], int n)       //
//                                                                            //
//  Description:                                                              //
//     This routine solves the linear equation Ux = B, where U is an n x n    //
//     upper triangular matrix.  (The subdiagonal part of the matrix is       //
//     not addressed.)                                                        //
//     The algorithm follows:                                                 //
//                  x[n-1] = B[n-1]/U[n-1][n-1], and                          //
//     x[i] = [B[i] - (U[i][i+1] * x[i+1]  + ... + U[i][n-1] * x[n-1])]       //
//                                                                 / U[i][i], //
//     for i = n-2, ..., 0.                                                   //
//                                                                            //
//  Arguments:                                                                //
//     double *U   Pointer to the first element of the upper triangular       //
//                 matrix.                                                    //
//     double *B   Pointer to the column vector, (n x 1) matrix, B.           //
//     double *x   Pointer to the column vector, (n x 1) matrix, x.           //
//     int     n   The number of rows or columns of the matrix U.             //
//                                                                            //
//  Return Values:                                                            //
//     0  Success                                                             //
//    -1  Failure - The matrix U is singular.                                 //
//                                                                            //
//  Example:                                                                  //
//     #define N                                                              //
//     double A[N][N], B[N], x[N];                                            //
//                                                                            //
//     (your code to create matrix A and column vector B)                     //
//     err = Upper_Triangular_Solve(&A[0][0], B, x, n);                       //
//     if (err < 0) printf(" Matrix A is singular\n");                        //
//     else printf(" The solution is \n");                                    //
//           ...                                                              //
////////////////////////////////////////////////////////////////////////////////
//                                                                            //
int Upper_Triangular_Solve(float *U, float B[], float x[], int n)
{
   int i, k;

//         Solve the linear equation Ux = B for x, where U is an upper
//         triangular matrix.

   for (k = n-1, U += n * (n - 1); k >= 0; U -= n, k--) {
      if (*(U + k) == 0.0) return -1;           // The matrix U is singular
      x[k] = B[k];
      for (i = k + 1; i < n; i++) x[k] -= x[i] * *(U + i);
      x[k] /= *(U + k);
   }

   return 0;
}

////////////////////////////////////////////////////////////////////////////////
//  int Choleski_LU_Decomposition(double *A, int n)                           //
//                                                                            //
//  Description:                                                              //
//     This routine uses Choleski's method to decompose the n x n positive    //
//     definite symmetric matrix A into the product of a lower triangular     //
//     matrix L and an upper triangular matrix U equal to the transpose of L. //
//     The original matrix A is replaced by L and U with L stored in the      //
//     lower triangular part of A and the transpose U in the upper triangular //
//     part of A. The original matrix A is therefore destroyed.               //
//                                                                            //
//     Choleski's decomposition is performed by evaluating, in order, the     //
//     following pair of expressions for k = 0, ... ,n-1 :                    //
//       L[k][k] = sqrt( A[k][k] - ( L[k][0] ^ 2 + ... + L[k][k-1] ^ 2 ) )    //
//       L[i][k] = (A[i][k] - (L[i][0]*L[k][0] + ... + L[i][k-1]*L[k][k-1]))  //
//                          / L[k][k]                                         //
//     and subsequently setting                                               //
//       U[k][i] = L[i][k], for i = k+1, ... , n-1.                           //
//                                                                            //
//     After performing the LU decomposition for A, call Choleski_LU_Solve    //
//     to solve the equation Ax = B or call Choleski_LU_Inverse to calculate  //
//     the inverse of A.                                                      //
//                                                                            //
//  Arguments:                                                                //
//     double *A   On input, the pointer to the first element of the matrix   //
//                 A[n][n].  On output, the matrix A is replaced by the lower //
//                 and upper triangular Choleski factorizations of A.         //
//     int     n   The number of rows and/or columns of the matrix A.         //
//                                                                            //
//  Return Values:                                                            //
//     0  Success                                                             //
//    -1  Failure - The matrix A is not positive definite symmetric (within   //
//                  working accuracy).                                        //
//                                                                            //
//  Example:                                                                  //
//     #define N                                                              //
//     double A[N][N];                                                        //
//                                                                            //
//     (your code to initialize the matrix A)                                 //
//     err = Choleski_LU_Decomposition((double *) A, N);                      //
//     if (err < 0) printf(" Matrix A is singular\n");                        //
//     else { printf(" The LLt decomposition of A is \n");                    //
//           ...                                                              //
////////////////////////////////////////////////////////////////////////////////
//                                                                            //
int Choleski_LU_Decomposition(float *A, int n)
{
   int i, k, p;
   float *p_Lk0;                   // pointer to L[k][0]
   float *p_Lkp;                   // pointer to L[k][p]
   float *p_Lkk;                   // pointer to diagonal element on row k.
   float *p_Li0;                   // pointer to L[i][0]
   float reciprocal;

   for (k = 0, p_Lk0 = A; k < n; p_Lk0 += n, k++) {

//            Update pointer to row k diagonal element.

      p_Lkk = p_Lk0 + k;

//            Calculate the difference of the diagonal element in row k
//            from the sum of squares of elements row k from column 0 to
//            column k-1.

      for (p = 0, p_Lkp = p_Lk0; p < k; p_Lkp += 1,  p++)
         *p_Lkk -= *p_Lkp * *p_Lkp;

//            If diagonal element is not positive, return the error code,
//            the matrix is not positive definite symmetric.

      if ( *p_Lkk <= 0.0 ) return -1;

//            Otherwise take the square root of the diagonal element.

      *p_Lkk = sqrt( *p_Lkk );
      reciprocal = 1.0 / *p_Lkk;

//            For rows i = k+1 to n-1, column k, calculate the difference
//            between the i,k th element and the inner product of the first
//            k-1 columns of row i and row k, then divide the difference by
//            the diagonal element in row k.
//            Store the transposed element in the upper triangular matrix.

      p_Li0 = p_Lk0 + n;
      for (i = k + 1; i < n; p_Li0 += n, i++) {
         for (p = 0; p < k; p++)
            *(p_Li0 + k) -= *(p_Li0 + p) * *(p_Lk0 + p);
         *(p_Li0 + k) *= reciprocal;
         *(p_Lk0 + i) = *(p_Li0 + k);
      }
   }
   return 0;
}


////////////////////////////////////////////////////////////////////////////////
//  int Choleski_LU_Solve(double *LU, double *B, double *x,  int n)           //
//                                                                            //
//  Description:                                                              //
//     This routine uses Choleski's method to solve the linear equation       //
//     Ax = B.  This routine is called after the matrix A has been decomposed //
//     into a product of a lower triangular matrix L and an upper triangular  //
//     matrix U which is the transpose of L. The matrix A is the product LU.  //
//     The solution proceeds by solving the linear equation Ly = B for y and  //
//     subsequently solving the linear equation Ux = y for x.                 //
//                                                                            //
//  Arguments:                                                                //
//     double *LU  Pointer to the first element of the matrix whose elements  //
//                 form the lower and upper triangular matrix factors of A.   //
//     double *B   Pointer to the column vector, (n x 1) matrix, B            //
//     double *x   Solution to the equation Ax = B.                           //
//     int     n   The number of rows and/or columns of the matrix LU.        //
//                                                                            //
//  Return Values:                                                            //
//     0  Success                                                             //
//    -1  Failure - The matrix L is singular.                                 //
//                                                                            //
//  Example:                                                                  //
//     #define N                                                              //
//     double A[N][N], B[N], x[N];                                            //
//                                                                            //
//     (your code to create matrix A and column vector B)                     //
//     err = Choleski_LU_Decomposition(&A[0][0], N);                          //
//     if (err < 0) printf(" Matrix A is singular\n");                        //
//     else {                                                                 //
//        err = Choleski_LU_Solve(&A[0][0], B, x, n);                         //
//        if (err < 0) printf(" Matrix A is singular\n");                     //
//        else printf(" The solution is \n");                                 //
//           ...                                                              //
//     }                                                                      //
////////////////////////////////////////////////////////////////////////////////
//                                                                            //
int Choleski_LU_Solve(float *LU, float B[], float x[], int n)
{

//         Solve the linear equation Ly = B for y, where L is a lower
//         triangular matrix.

   if ( Lower_Triangular_Solve(LU, B, x, n) < 0 ) return -1;

//         Solve the linear equation Ux = y, where y is the solution
//         obtained above of Ly = B and U is an upper triangular matrix.

   return Upper_Triangular_Solve(LU, x, x, n);
}



DLL_EXPORT int majority_vote(int * decision_buffer, int size)
{
	int i,j;
	int maxdata=0;
	int test_decision=0;
    int max_label=0;
    int *modes;

    for(i=0;i<size;i++)
	{
        if(decision_buffer[i] > max_label)
        {
            max_label = decision_buffer[i];
        }
    }

    modes=(int*)calloc(max_label+1,sizeof(int));
    if(!modes)
    {
        return 0;
    }

	for(i=0;i<size;i++)
	{
		modes[decision_buffer[i]]=modes[decision_buffer[i]]+1;
	}
	for(j=0;j<max_label+1;j++){
		if(modes[j]>maxdata)
		{
			maxdata=modes[j];
			test_decision=j;
		}
	}
    free(modes);
	return test_decision;
}

