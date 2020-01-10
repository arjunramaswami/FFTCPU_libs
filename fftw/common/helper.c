/******************************************************************************
 *  Author: Arjun Ramaswami
 *****************************************************************************/

// global dependencies
#define _POSIX_C_SOURCE 199309L  
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <float.h>
#define _USE_MATH_DEFINES
#include <fftw3.h>

// common dependencies
#include "fft_api.h"

/* Compute (K*L)%M accurately */
static double moda(int K, int L, int M){
    return (double)(((long long)K * L) % M);
}

// --- CODE ------------------------------------------------------------------

/******************************************************************************
 * \brief  create random single precision floating point values for FFT 
 *         computation or read existing ones if already saved in a file
 * \param  fftw_data : pointer to fft3d sized allocation of sp complex data for fftw cpu computation
 * \param  N : 3 element integer array containing the size of FFT3d  
 * \param  H1, H2, H3 : harmonic to modify frequency of discrete time signal
 *****************************************************************************/
void get_sp_input_data(fftwf_complex *fftw_data, int N[3], int H1, int H2, int H3){

  int i = 0, j = 0, k = 0, index;
  int N1 = N[0], N2 = N[1], N3 = N[2];
  int S1 = N2*N3, S2 = N3, S3 = 1;

  float TWOPI = 6.2831853071795864769;
  float phase, phase1, phase2, phase3;

  for (i = 0; i < N[0]; i++) {
    for (j = 0; j < N[1]; j++) {
      for (k = 0; k < N[2]; k++) {
        phase1 = moda(i,H1,N1) / N1;
        phase2 = moda(j,H2,N2) / N2;
        phase3 = moda(k,H3,N3) / N3;
        phase = phase1 + phase2 + phase3;

        index = i*S1 + j*S2 + k*S3;

        fftw_data[index][0] = cosf( TWOPI * phase ) / (N1*N2*N3);
        fftw_data[index][1] = sinf( TWOPI * phase ) / (N1*N2*N3);

#ifdef DEBUG          
          printf(" %d %d %d : fftw[%d] = (%f, %f) \n", i, j, k, index, fftw_data[index][0], fftw_data[index][1]);
#endif
      }
    }
  }
}

/******************************************************************************
 * \brief  create random double precision floating point values for FFT 
 *         computation or read existing ones if already saved in a file
 * \param  fftw_data : pointer to fft3d sized allocation of dp complex data for fftw cpu computation
 * \param  N : 3 element integer array containing the size of FFT3d  
 * \param  H1, H2, H3 : harmonic to modify frequency of discrete time signal
 *****************************************************************************/
void get_dp_input_data(fftw_complex *fftw_data, int N[3], int H1, int H2, int H3){
  int i = 0, j = 0, k = 0, index;
  int N1 = N[0], N2 = N[1], N3 = N[2];
  int S1 = N2*N3, S2 = N3, S3 = 1;

  double TWOPI = 6.2831853071795864769;
  double phase, phase1, phase2, phase3;

  for (i = 0; i < N[0]; i++) {
    for (j = 0; j < N[1]; j++) {
      for (k = 0; k < N[2]; k++) {
        phase1 = moda(i,H1,N1) / N1;
        phase2 = moda(j,H2,N2) / N2;
        phase3 = moda(k,H3,N3) / N3;
        phase = phase1 + phase2 + phase3;

        index = i*S1 + j*S2 + k*S3;

        fftw_data[index][0] = cos( TWOPI * phase ) / (N1*N2*N3);
        fftw_data[index][1] = sin( TWOPI * phase ) / (N1*N2*N3);

#ifdef DEBUG          
          printf(" %d %d %d : fftw[%d] = (%lf, %lf) \n", i, j, k, index, fftw_data[index][0], fftw_data[index][1]);
#endif
      }
    }
  }


}

/******************************************************************************
 * \brief  compute walltime in milliseconds
 * \retval time in milliseconds
 *****************************************************************************/
double getTimeinMilliSec(){
   struct timespec a;
   clock_gettime(CLOCK_MONOTONIC, &a);
   return (double)(a.tv_nsec) * 1.0e-6 + (double)(a.tv_sec) * 1.0E3;
}

/******************************************************************************
 * \brief  print time taken for fftw runs to a file
 * \param  fftw_time: double
 * \param  iter - number of iterations of each
 * \param  fname - filename given through cmd line arg
 * \param  N - fft size
 *****************************************************************************/
void compute_metrics( double fftw_runtime, unsigned iter, int N[3]){
  double avg_fftw_runtime = 0.0;

  printf("\nNumber of runs: %d\n\n", iter);
  printf("      FFTSize  TotalRuntime(ms)  AvgRuntime(ms)  Throughput(GFLOPS)    \n");

  printf("fftw:"); 
  if(fftw_runtime != 0.0){
    avg_fftw_runtime = fftw_runtime / (iter);  
    double gpoints_per_sec = ( N[0] * N[1] * N[2] / (fftw_runtime * 1E-3)) * 1E-9;
    double gflops = 3 * 5 * N[0] * N[1] * N[2]* (log((double)N[0])/log((double)2))/(avg_fftw_runtime * 1E-3 * 1E9);
    printf("%5d³       %.4f            %.4f          %.4f    \n", N[0], fftw_runtime, avg_fftw_runtime, gflops);
  }
  else{
    printf("ERROR in FFT3d\n");
  }
}

/******************************************************************************
 * \brief  Verify double precision FFT3d computation
 * \param  x : fftw_complex - 3d FFT data after transformation
 * \param  N - fft size
 * \param  H1, H2, H3 : harmonic to modify frequency of discrete time signal
 *****************************************************************************/
int verify_dp(fftw_complex *x, int N[3], int H1, int H2, int H3){
    /* Verify that x(n1,n2,n3) is a peak at H1,H2,H3 */
    double err, errthr, maxerr;
    int n1, n2, n3, index;

    int N1 = N[0], N2 = N[1], N3 = N[2];
    /* Generalized strides for row-major addressing of x */
    int S1 = N2*N3, S2 = N3, S3 = 1;

    /*
     * Note, this simple error bound doesn't take into account error of
     * input data
     */
    errthr = 5.0 * log( (double)N1*N2*N3 ) / log(2.0) * DBL_EPSILON;
    printf("Verify the result, errthr = %.3lg\n", errthr);

    maxerr = 0;
    for (n1 = 0; n1 < N1; n1++){
        for (n2 = 0; n2 < N2; n2++){
            for (n3 = 0; n3 < N3; n3++){
                double re_exp = 0.0, im_exp = 0.0, re_got, im_got;

                if ((n1-H1)%N1==0 && (n2-H2)%N2==0 && (n3-H3)%N3==0) {
                    re_exp = 1;
                }

                index = n1*S1 + n2*S2 + n3*S3;
                re_got = x[index][0];
                im_got = x[index][1];
                err  = fabs(re_got - re_exp) + fabs(im_got - im_exp);
                if (err > maxerr) maxerr = err;
                if (!(err < errthr))
                {
                    printf(" x[%i][%i][%i]: ",n1,n2,n3);
                    printf(" expected (%.17lg,%.17lg), ",re_exp,im_exp);
                    printf(" got (%.17lg,%.17lg), ",re_got,im_got);
                    printf(" err %.3lg\n", err);
                    printf(" Verification FAILED\n");
                    return 1;
                }
            }
        }
    }
    printf("Verified, maximum error was %.3lg\n", maxerr);

#ifdef DEBUG
    printf("Output Frequencies: \n");
    for(int i = 0; i < N1; i++){
        for(int j = 0; j < N2; j++){
            for(int k = 0; k < N3; k++){
                index = (i * N1 * N2) + (j * N3) + k;
                printf(" %d %d %d : fftw[%d] = (%lf, %lf) \n", i, j, k, index, x[index][0], x[index][1]);
            }
        }
    }
#endif

    return 0;
}

/******************************************************************************
 * \brief  Verify single precision FFT3d computation
 * \param  x : fftwf_complex - 3d FFT data after transformation
 * \param  N - fft size
 * \param  H1, H2, H3 : harmonic to modify frequency of discrete time signal
 *****************************************************************************/
int verify_sp(fftwf_complex *x, int N[3], int H1, int H2, int H3){
    /* Verify that x(n1,n2,n3) is a peak at H1,H2,H3 */
    double err, errthr, maxerr;
    int n1, n2, n3, index;

    int N1 = N[0], N2 = N[1], N3 = N[2];
    /* Generalized strides for row-major addressing of x */
    int S1 = N2*N3, S2 = N3, S3 = 1;

    /*
     * Note, this simple error bound doesn't take into account error of
     * input data
     */
    errthr = 5.0 * log( (float)N1*N2*N3 ) / log(2.0) * FLT_EPSILON;
    printf("Verify the result, errthr = %.3lg\n", errthr);

    maxerr = 0;
    for (n1 = 0; n1 < N1; n1++){
        for (n2 = 0; n2 < N2; n2++){
            for (n3 = 0; n3 < N3; n3++){
                float re_exp = 0.0, im_exp = 0.0, re_got, im_got;

                if ((n1-H1)%N1==0 && (n2-H2)%N2==0 && (n3-H3)%N3==0) {
                    re_exp = 1;
                }

                index = n1*S1 + n2*S2 + n3*S3;
                re_got = x[index][0];
                im_got = x[index][1];
                err  = fabs(re_got - re_exp) + fabs(im_got - im_exp);
                if (err > maxerr) maxerr = err;
                if (!(err < errthr))
                {
                    printf(" x[%i][%i][%i]: ",n1,n2,n3);
                    printf(" expected (%.17lg,%.17lg), ",re_exp,im_exp);
                    printf(" got (%.17lg,%.17lg), ",re_got,im_got);
                    printf(" err %.3lg\n", err);
                    printf(" Verification FAILED\n");
                    return 1;
                }
            }
        }
    }
    printf("Verified, maximum error was %.3lg\n", maxerr);

#ifdef DEBUG
    printf("Output Frequencies: \n");
    for(int i = 0; i < N1; i++){
        for(int j = 0; j < N2; j++){
            for(int k = 0; k < N3; k++){
                index = (i * N1 * N2) + (j * N3) + k;
                printf(" %d %d %d : fftw[%d] = (%f, %f) \n", i, j, k, index, x[index][0], x[index][1]);
            }
        }
    }
#endif

    return 0;
}