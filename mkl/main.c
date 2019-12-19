#include <stdio.h>
#include <stdlib.h>

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
static int init(MKL_Complex16 *fft_data, int N1, int N2, int N3){
    int where = 0;
    int i, j, k;

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

/* 
*  Initialize input array to FFT
*  Input  : pointer to initialized 3d array of MKL_Complex16 type 
*         : size of FFT3d - N1, N2, N3
*         : num_threads 
*  Output : 1 if error in execution
*/
static int run_fft(MKL_Complex16 *fft_data, int N1, int N2, int N3, int num_threads){
    int i, j, k, where;
    MKL_LONG status = 0;
    printf("Create DFTI descriptor for %ix%ix%i FFT\n", N1, N2, N3);

    DFTI_DESCRIPTOR_HANDLE ffti_desc_handle;

    MKL_LONG dim = 3;
    MKL_LONG size[3];
    size[0] = N1;
    size[1] = N2;
    size[2] = N3;

    /* Configuration
    * ------------------
    * ffti_desc_handle - fft intel description handle / DFTI_DESCRIPTOR HANDLE type
    * double precision - precision                    / enum MACRO 
    * DFTI_COMPLEX     - data type or domain          / enum MACRO
    *            3     - dimension                    / MKL_LONG
    *   {64,64,64}     - points in each dimension     / array of MKL_LONG
    * ------------------- */
    status = DftiCreateDescriptor( &ffti_desc_handle, DFTI_DOUBLE, DFTI_COMPLEX, dim, size);
    if(status != DFTI_NO_ERROR){
        char *error_message = DftiErrorMessage(status);
        printf("MKL FFT Description Creation Failed with message %s \n", error_message);
        return 1;
    }

    //status = DftiSetValue(ffti_desc_handle, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
    //status = DftiSetValue(ffti_desc_handle, DFTI_OUTPUT_STRIDES, strides_out);
    status = DftiSetValue(ffti_desc_handle, DFTI_PLACEMENT, DFTI_INPLACE);  // this is the default
    if(status != DFTI_NO_ERROR){
        char *error_message = DftiErrorMessage(status);
        printf("MKL FFT Description Creation Failed with message %s \n", error_message);
        return 1;
    }

    printf("Setting thread limit %i\n", num_threads);
    status = DftiSetValue(ffti_desc_handle, DFTI_THREAD_LIMIT, num_threads);
    if(status != DFTI_NO_ERROR){
        char *error_message = DftiErrorMessage(status);
        printf("MKL FFT Description Creation Failed with message %s \n", error_message);
        return 1;
    }

    /* Initializes the descriptor
    * Performs optimization
    * Computes Twiddle Factors
    */
    printf("Committing Descriptor\n");
    status = DftiCommitDescriptor(ffti_desc_handle);
    if(status != 0){
        char *error_message = DftiErrorMessage(status);
        printf("MKL FFT Description Creation Failed with message %s \n", error_message);
        return 1;
    }

    printf("Computing Forward Transform\n");
    status = DftiComputeForward(ffti_desc_handle, fft_data);
    if(status != 0){
        char *error_message = DftiErrorMessage(status);
        printf("MKL FFT Description Creation Failed with message %s \n", error_message);
        return 1;
    }
    for(i = 0; i < N1; i++){
        for(j = 0; j < N2; j++){
            for(k = 0; k < N3; k++){
                where = (i * N1 * N2) + (j * N3) + k;
                printf(" %d %d %d : fft[%d] = (%lf, %lf)\n", i, j, k, where, fft_data[where].real, fft_data[where].imag);
            }
        }
    }
    printf("Computing Backward Transform\n");
    status = DftiComputeBackward(ffti_desc_handle, fft_data);
    if(status != 0){
        char *error_message = DftiErrorMessage(status);
        printf("MKL FFT Description Creation Failed with message %s \n", error_message);
        return 1;
    }

    printf("Free descriptor\n");
    status = DftiFreeDescriptor(&ffti_desc_handle);
    if(status != 0){
        char *error_message = DftiErrorMessage(status);
        printf("MKL FFT Description Creation Failed with message %s \n", error_message);
        return 1;
    }

    return 0;
}


int main(){
    int err = 0;
    int N1 = 64, N2 = 64, N3 = 64;
    int where = 0;
    int i, j, k;

    // Print Version of Intel MKL
    char version[DFTI_VERSION_LENGTH];
    DftiGetValue(0, DFTI_VERSION, version);
    printf("%s\n", version);

    // Multi threaded
    int thread_id = 0, team = 1;
#if defined(_OPENMP)
    thread_id = omp_get_thread_num();
    team  = omp_get_num_threads();
    printf("Total number of threads %d \n", team);
#endif
    
    MKL_Complex16 *fft_data = (MKL_Complex16*)mkl_malloc(N1 * N2 * N3 * sizeof(MKL_Complex16), 64);
    if (fft_data == NULL){
        printf("Data allocation failed \n");
        return 1;
    }

    printf("Initializing Input of %ix%ix%i FFT\n", N1, N2, N3);
    err = init(fft_data, N1, N2, N3);
    if(err){
        printf("FFT failed \n");
    }

    err = run_fft(fft_data, N1, N2, N3, team);
    if(err){
        printf("FFT failed \n");
    }


    // free array
    mkl_free(fft_data);
    return 0;
}