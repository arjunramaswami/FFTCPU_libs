/* Author : Arjun Ramaswami
* email   : ramaswami.arjun@gmail.com
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "common/argparse.h"
#include "common/helper.h"

/*
*  Include FFT Files
*  MKL Memory Allocation functions - mkl_malloc, mkl_free
*  Can also separately include mkl_dfti.h and mkl_service.h
*/
#include "mkl.h"  

/*
*  Multi threading support
*/
#if defined(_OPENMP)
#include <omp.h>
#endif

/* Compute (K*L)%M accurately */
static double moda(int K, int L, int M)
{
    return (double)(((long long)K * L) % M);
}

/* Initialize array with harmonic {H1, H2, H3} */
static void init(MKL_Complex16 *x, int N1, int N2, int N3,
                 int H1, int H2, int H3)
{
    double TWOPI = 6.2831853071795864769, phase;
    int n1, n2, n3, index;

    /* Generalized strides for row-major addressing of x */
    int S1 = N2*N3, S2 = N3, S3 = 1;

    for (n1 = 0; n1 < N1; n1++)
    {
        for (n2 = 0; n2 < N2; n2++)
        {
            for (n3 = 0; n3 < N3; n3++)
            {
                phase =  moda(n1,H1,N1) / N1;
                phase += moda(n2,H2,N2) / N2;
                phase += moda(n3,H3,N3) / N3;
                index = n1*S1 + n2*S2 + n3*S3;
                x[index].real = cos( TWOPI * phase ) / (N1*N2*N3);
                x[index].imag = sin( TWOPI * phase ) / (N1*N2*N3);
            }
        }
    }
}

/* Verify that x(n1,n2,n3) is a peak at H1,H2,H3 */
static int verify(MKL_Complex16 *x, int N1, int N2, int N3,
                  int H1, int H2, int H3)
{
    double err, errthr, maxerr;
    int n1, n2, n3, index;

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
                re_got = x[index].real;
                im_got = x[index].imag;
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
    return 0;
}


static void error_msg(MKL_LONG status){
    if(status != DFTI_NO_ERROR){
        char *error_message = DftiErrorMessage(status);
        printf("Failed with message %s \n", error_message);
        exit(1);
    }
}

static const char *const usage[] = {
    "./host [options]",
    NULL,
};

int main(int argc, const char **argv){
    int err = 0, where = 0, iter = 1;
    int i, j, k;
    int N1 = 8, N2 = 8, N3 = 8;
    MKL_LONG status = 0;
    int thread_id = 0, team = 1; // Multi threaded 
    
    /* Arbitrary harmonic used to verify FFT */
    int H1 = -2, H2 = -3, H3 = -4;

    // Cmd Line arguments
    struct argparse_option options[] = {
        OPT_HELP(),
        OPT_GROUP("Basic Options"),
        OPT_INTEGER('m',"n1", &N1, "FFT 1st Dim Size"),
        OPT_INTEGER('n',"n2", &N2, "FFT 2nd Dim Size"),
        OPT_INTEGER('p',"n3", &N3, "FFT 3rd Dim Size"),
        OPT_INTEGER('i',"iter", &iter, "Number of iterations"),
        OPT_INTEGER('t',"threads", &team, "Num Threads"),
        OPT_END(),
    };

    struct argparse argparse;
    argparse_init(&argparse, options, usage, 0);
    argparse_describe(&argparse, "Computing FFT3d using MKL", "FFT size is mandatory, default number of iterations is 1");
    argc = argparse_parse(&argparse, argc, argv);

#if defined(_OPENMP)
    printf("Total number of threads %d \n", team);
#endif

    // Print Version of Intel MKL
    char version[DFTI_VERSION_LENGTH];
    DftiGetValue(0, DFTI_VERSION, version);

    // Command Line Arguments
    MKL_LONG dim = 3;
    MKL_LONG size[3]; size[0] = N1; size[1] = N2; size[2] = N3;

    printf("\n------------------------------------\n");
    printf("MKL Configuration: \n");
    printf("------------------------------------\n");
    printf("MKL VERSION : %s\n", version);
    printf("Forward and Backward Double precision complex 3d FFT");
    printf("Parameters: \n");
    printf("DFTI_DIMENSION      =  3\n");
    printf("DFTI_PRECISION      = DFTI_DOUBLE \n");
    printf("DFTI_LENGTHS        = {%i, %i, %i} \n", size[0], size[1], size[2]);
    printf("DFTI_FORWARD_DOMAIN = DFTI_COMPLEX\n");
    printf("DFTI_PLACEMENT      = DFTI_INPLACE\n");
    printf("THREADS             = %i \n", team);
    printf("------------------------------------\n\n");

    DFTI_DESCRIPTOR_HANDLE ffti_desc_handle = 0;

    status = DftiCreateDescriptor( &ffti_desc_handle, DFTI_DOUBLE, DFTI_COMPLEX, dim, size);
    error_msg(status);

    status = DftiSetValue(ffti_desc_handle, DFTI_PLACEMENT, DFTI_INPLACE);  // this is the default
    error_msg(status);

    status = DftiSetValue(ffti_desc_handle, DFTI_THREAD_LIMIT, team);
    error_msg(status);

    status = DftiCommitDescriptor(ffti_desc_handle);
    error_msg(status);

    MKL_Complex16 *fft_data = (MKL_Complex16*)mkl_malloc(N1 * N2 * N3 * sizeof(MKL_Complex16), 64);
    if (fft_data == NULL){
        printf("Data allocation failed \n");
        exit(1);
    }

    printf("Computing Forward followed by Backward Transforms for %d iterations\n\n", iter);
    double start_fwd = 0.0, stop_fwd = 0.0;
    double start_bwd = 0.0, stop_bwd = 0.0;
    double diff_fwd = 0.0, diff_bwd = 0.0;

    for(i = 0; i < iter; i++){

        init(fft_data, N1, N2, N3, H1, H2, H3);

        start_fwd = getTimeinMilliSec();
        status = DftiComputeForward(ffti_desc_handle, fft_data);
        stop_fwd = getTimeinMilliSec();
        error_msg(status);

        status = verify(fft_data, N1, N2, N3, H1, H2, H3);
        diff_fwd += stop_fwd - start_fwd;

        init(fft_data, N1, N2, N3, -H1, -H2, -H3);

        start_bwd = getTimeinMilliSec();
        status = DftiComputeBackward(ffti_desc_handle, fft_data);
        stop_bwd = getTimeinMilliSec();
        error_msg(status);

        status = verify(fft_data, N1, N2, N3, H1, H2, H3);

        diff_bwd += stop_bwd - start_bwd;
    }

    printf("\nForward Transform Performance: \n");
    compute_metrics(diff_fwd, iter, N1, N2, N3);
    printf("\nBackward Transform Performance: \n");
    compute_metrics(diff_bwd, iter, N1, N2, N3);

#ifdef DEBUG
    for(i = 0; i < N1; i++){
        for(j = 0; j < N2; j++){
            for(k = 0; k < N3; k++){
                where = (i * N1 * N2) + (j * N3) + k;
                printf(" %d %d %d : fft[%d] = (%lf, %lf)\n", i, j, k, where, fft_data[where].real, fft_data[where].imag);
            }
        }
    }
#endif

    status = DftiFreeDescriptor(&ffti_desc_handle);
    error_msg(status);

    // free array
    mkl_free(fft_data);
    return 0;
}