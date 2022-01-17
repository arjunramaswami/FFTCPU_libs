#include <iostream>
#include <cmath>
#include <omp.h>
#include "mkl.h"  

#include "cxxopts.hpp" // Cmd-Line Args parser
#include "mkl_many.hpp"
#include "helper.hpp"
using namespace std;

/*
int main(int argc, const char **argv){
    int err = 0, where = 0, iter = 1, inverse = 0;
    int i, j, k;
    int N1 = 8, N2 = 8, N3 = 8;
    MKL_LONG status = 0;
    int thread_id = 0, team = 1; // Multi threaded 
    
    int H1 = 1, H2 = 1, H3 = 1;

    // Cmd Line arguments
    struct argparse_option options[] = {
        OPT_HELP(),
        OPT_GROUP("Basic Options"),
        OPT_INTEGER('m',"n1", &N1, "FFT 1st Dim Size"),
        OPT_INTEGER('n',"n2", &N2, "FFT 2nd Dim Size"),
        OPT_INTEGER('p',"n3", &N3, "FFT 3rd Dim Size"),
        OPT_INTEGER('i',"iter", &iter, "Number of iterations"),
        OPT_INTEGER('t',"threads", &team, "Num Threads"),
        OPT_BOOLEAN('b',"inverse", &inverse, "Backward FFT"),
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
    printf("\n%sDOUBLE PRECISION COMPLEX 3d FFT\n\n", inverse?"BACKWARD ":"FORWARD ");
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

    double start_fwd = 0.0, stop_fwd = 0.0;
    double start_bwd = 0.0, stop_bwd = 0.0;
    double diff_fwd = 0.0, diff_bwd = 0.0;

    for(i = 0; i < iter; i++){
        if(!inverse){
            // Computing Forward
            init(fft_data, N1, N2, N3, H1, H2, H3);

            start_fwd = getTimeinMilliSec();
            status = DftiComputeForward(ffti_desc_handle, fft_data);
            stop_fwd = getTimeinMilliSec();
            error_msg(status);

            status = verify(fft_data, N1, N2, N3, H1, H2, H3);
            diff_fwd += stop_fwd - start_fwd;
        }
        else{
            // Computing Backward
            init(fft_data, N1, N2, N3, -H1, -H2, -H3);

            start_bwd = getTimeinMilliSec();
            status = DftiComputeBackward(ffti_desc_handle, fft_data);
            stop_bwd = getTimeinMilliSec();
            error_msg(status);

            status = verify(fft_data, N1, N2, N3, H1, H2, H3);

            diff_bwd += stop_bwd - start_bwd;
        }
    }

    if(!inverse){
        printf("\nForward Transform Performance: \n");
        compute_metrics(diff_fwd, iter, N1, N2, N3);
    }
    else{
        printf("\nBackward Transform Performance: \n");
        compute_metrics(diff_bwd, iter, N1, N2, N3);
    }

#ifdef DEBUG
    printf("Output Frequencies: \n");
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
*/

/**
 * \brief create single precision floating points values for FFT computation for each process level block
 * \param fft_data    : pointer to 3d number of sp points for FFTW
 * \param verify_data : pointer to 3d number of sp points for verification
 * \param N           : number of points in each dimension
 * \param how_many    : number of batched implementations of FFTW
 */
void get_data(MKL_Complex8 *fft_data, MKL_Complex8 *verify_data, unsigned N, unsigned how_many){

  float re_val = 0.0f, img_val = 0.0f;

  for(unsigned i = 0; i < how_many * N * N * N; i++){
    re_val = ((float) rand() / (RAND_MAX));
    img_val = ((float) rand() / (RAND_MAX));

    verify_data[i].real = fft_data[i].real = re_val;
    verify_data[i].imag = fft_data[i].imag = img_val;

#ifndef NDEBUG          
  //printf("fft[%u]: (%f, %f)\n", i, fft_data[i].real, fft_data[i].imag);
#endif
  }
}

static void error_msg(MKL_LONG status){
  if(status != DFTI_NO_ERROR){
    char *error_message = DftiErrorMessage(status);
    printf("Failed with message %s \n", error_message);
  }
}


void mkl_openmp(unsigned N, unsigned how_many, unsigned nthreads, bool inverse, unsigned iter){

  if ( (how_many == 0) || (nthreads == 0) || (iter == 0) )
    throw "Invalid value, should be >=1!";

  MKL_Complex8 *fft_data = (MKL_Complex8*)mkl_malloc(N * N * N * sizeof(MKL_Complex8), 64);
  if (fft_data == NULL)
    throw "FFT data allocation failed";

  MKL_Complex8 *verify_data = (MKL_Complex8*)mkl_malloc(N * N * N * sizeof(MKL_Complex8), 64);
  if (fft_data == NULL)
    throw "FFT data allocation failed";

  get_data(fft_data, verify_data, N, how_many);

  //MKL_NUM_THREADS = nthreads;
  const MKL_LONG dim = 3;
  const MKL_LONG size[3] = {N, N, N};
  MKL_LONG status;

  DFTI_DESCRIPTOR_HANDLE fft_desc_handle = NULL;

  status = DftiCreateDescriptor(&fft_desc_handle, DFTI_SINGLE, DFTI_COMPLEX, dim, size);
  error_msg(status);

  status = DftiSetValue(fft_desc_handle, DFTI_PLACEMENT, DFTI_INPLACE);
  error_msg(status);
  status = DftiSetValue(fft_desc_handle, DFTI_THREAD_LIMIT, nthreads);
  error_msg(status);

  status = DftiCommitDescriptor(fft_desc_handle);
  error_msg(status);

  double start = getTimeinMilliSec();
  DftiComputeForward(fft_desc_handle, fft_data);
  double stop = getTimeinMilliSec();

  cout << "Time to FFT3D: " << stop-start << endl;

  DftiComputeBackward(fft_desc_handle, fft_data);

  DftiFreeDescriptor(&fft_desc_handle);

  bool status_out = verify_mkl(fft_data, verify_data, N, how_many);
  if(!status_out){
    cout << "Error in Transformation\n";
  }

  mkl_free(fft_data);
  mkl_free(verify_data);
}