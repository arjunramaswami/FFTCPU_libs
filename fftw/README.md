
## Configuring FFTW with OpenMP

Steps to configure multithreaded execution of FFTW:

1. To compile with OpenMPI, link this additional flag `-lfftw3_omp` along with 
`-fopenmp` other than the regular `fftw` flags

2. 



### Details on the configuration
- Initialize the environment:
 ```
#include<omp.h>
int fftw_init_threads(void); 
```

- Make the plan with the necessary number of threads to execute
```
void fftw_plan_with_nthreads(int nthreads);
```

- Execute with the normal 
```
fftw_execute(plan) 
```

- Cleanup plan and threads after execution
```
fftw_destroy_plan()
void fftw_cleanup_threads(void);
```

#### Note
Execution is thread-safe but not plan creation and destruction, therefore use a
single thread for the latter. 

