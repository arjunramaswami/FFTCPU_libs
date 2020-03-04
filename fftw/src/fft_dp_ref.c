//  Author: Arjun Ramaswami

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <float.h> // DBL_EPSILON
#include <fftw3.h>
#include "helper.h"

static void get_dp_input_data(fftw_complex *fftw_data, int N[3], int H1, int H2, int H3);
static int verify_dp(fftw_complex *x, int N[3], int H1, int H2, int H3);

void fftw_dp(int N[3], int nthreads, int inverse, int iter){
    
    int H1 = 1, H2 = 1, H3 = 1;
    double start = 0.0, stop = 0.0, diff = 0.0, total_diff = 0.0, plan_start = 0.0, plan_end = 0.0;

    printf("Configuring plan for double precision FFT\n\n");
    int status = fftw_init_threads(); 
    if(status == 0){
      printf("Something went wrong with Multithreaded FFTW! Exiting... \n");
      exit(EXIT_FAILURE);
    }

#ifdef DEBUG
    printf("Using %d threads \n", nthreads);
#endif
    fftw_plan_with_nthreads(nthreads);

    fftw_complex *fftw_dp_data = (fftw_complex*)fftw_malloc(sizeof(fftw_complex) * N[0] * N[1] * N[2]);

    int direction = FFTW_FORWARD;
    if(inverse){
      direction = FFTW_BACKWARD;
    }

    plan_start = getTimeinMilliSec();
#ifdef MEASURE
    const fftw_plan plan = fftw_plan_dft_3d(N[0], N[1], N[2], fftw_dp_data, fftw_dp_data, direction, FFTW_MEASURE);
#elif PATIENT
    const fftw_plan plan = fftw_plan_dft_3d(N[0], N[1], N[2], fftw_dp_data, fftw_dp_data, direction, FFTW_PATIENT);
#elif EXHAUSTIVE
    const fftw_plan plan = fftw_plan_dft_3d(N[0], N[1], N[2], fftw_dp_data, fftw_dp_data, direction, FFTW_EXHAUSTIVE);
#else
    const fftw_plan plan = fftw_plan_dft_3d(N[0], N[1], N[2], fftw_dp_data, fftw_dp_data, direction, FFTW_ESTIMATE);
#endif
    plan_end = getTimeinMilliSec();

    printf("Threads %d: time to plan - %lf sec\n\n", nthreads, (plan_end - plan_start) / 1000);

    for(int i = 0; i < iter; i++){
      if(inverse){
        get_dp_input_data(fftw_dp_data, N, -H1, -H2, -H3);
      }
      else{
        get_dp_input_data(fftw_dp_data, N, H1, H2, H3);
      }

      start = getTimeinMilliSec();
      fftw_execute(plan);
      stop = getTimeinMilliSec();

      diff = stop - start;

#ifdef VERBOSE
      printf("Iter %d: %lf ms\n", i, diff);
#endif

      total_diff += diff;

      //printf(" Time : %lf %lf - %lf \n", start, stop, diff);
      status = verify_dp(fftw_dp_data, N, H1, H2, H3);
      if(status == 1){
        printf("Error in transformation \n");
        exit(0);
      }
    }

    double add, mul, fma, flops;
    fftw_flops(plan, &add, &mul, &fma);
    flops = add + mul + fma; 
    print_results(total_diff, iter, N, nthreads, flops);

    printf("Cleaning up \n");
    // Cleanup : fftw data, plans and threads
    fftw_free(fftw_dp_data);
    fftw_destroy_plan(plan);
    fftw_cleanup_threads();

}

/******************************************************************************
 * \brief  Verify double precision FFT3d computation
 * \param  x : fftw_complex - 3d FFT data after transformation
 * \param  N - fft size
 * \param  H1, H2, H3 : harmonic to modify frequency of discrete time signal
 *****************************************************************************/
static int verify_dp(fftw_complex *x, int N[3], int H1, int H2, int H3){
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
#ifdef DEBUG
    printf("Verify the result, errthr = %.3lg\n", errthr);
#endif

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

#ifdef DEBUG
    printf("Verified, maximum error was %.3lg\n", maxerr);
    printf("\n\nOutput Frequencies: \n");
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
 * \brief  create random double precision floating point values for FFT 
 *         computation or read existing ones if already saved in a file
 * \param  fftw_data : pointer to fft3d sized allocation of dp complex data for fftw cpu computation
 * \param  N : 3 element integer array containing the size of FFT3d  
 * \param  H1, H2, H3 : harmonic to modify frequency of discrete time signal
 *****************************************************************************/
static void get_dp_input_data(fftw_complex *fftw_data, int N[3], int H1, int H2, int H3){
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