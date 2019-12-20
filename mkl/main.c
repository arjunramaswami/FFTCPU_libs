#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include "common/argparse.h"

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

/* 
*  Initialize input array to FFT
*  Input  : pointer to 3d array of MKL_Complex16 type
*         : size of FFT3d - N1, N2, N3
*  Output : 1 if error in initialization
*/
static void init(MKL_Complex16 *fft_data, int N1, int N2, int N3){
    int where = 0, i, j, k;

    for(i = 0; i < N1; i++){
        for(j = 0; j < N2; j++){
            for(k = 0; k < N3; k++){
                where = (i * N1 * N2) + (j * N3) + k;
                fft_data[where].real = where;
                fft_data[where].imag = where;
            }
        }
    }
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

    printf("Create DFTI descriptor\n");
    DFTI_DESCRIPTOR_HANDLE ffti_desc_handle = 0;
    status = DftiCreateDescriptor( &ffti_desc_handle, DFTI_DOUBLE, DFTI_COMPLEX, dim, size);
    error_msg(status);

    //status = DftiSetValue(ffti_desc_handle, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
    //status = DftiSetValue(ffti_desc_handle, DFTI_OUTPUT_STRIDES, strides_out);
    status = DftiSetValue(ffti_desc_handle, DFTI_PLACEMENT, DFTI_INPLACE);  // this is the default
    if(status != DFTI_NO_ERROR){
        char *error_message = DftiErrorMessage(status);
        printf("MKL FFT Description Creation Failed with message %s \n", error_message);
        return 1;
    }

    printf("Setting thread limit %i\n", team);
    status = DftiSetValue(ffti_desc_handle, DFTI_THREAD_LIMIT, team);
    error_msg(status);

    /* Initializes the descriptor
    * Performs optimization
    * Computes Twiddle Factors
    */
    printf("Committing Descriptor\n");
    status = DftiCommitDescriptor(ffti_desc_handle);
    error_msg(status);

    MKL_Complex16 *fft_data = (MKL_Complex16*)mkl_malloc(N1 * N2 * N3 * sizeof(MKL_Complex16), 64);
    if (fft_data == NULL){
        printf("Data allocation failed \n");
        exit(1);
    }

    printf("Initializing Input of %ix%ix%i FFT\n\n", N1, N2, N3);
    init(fft_data, N1, N2, N3);

    printf("Computing Forward followed by Backward Transforms for %d iterations\n", iter);
    for(i = 0; i < iter; i++){
        status = DftiComputeForward(ffti_desc_handle, fft_data);
        error_msg(status);

        status = DftiComputeBackward(ffti_desc_handle, fft_data);
        error_msg(status);
    }

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

    printf("Free descriptor\n");
    status = DftiFreeDescriptor(&ffti_desc_handle);
    error_msg(status);

    // free array
    mkl_free(fft_data);
    return 0;
}