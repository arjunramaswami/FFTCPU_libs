# Author : Arjun Ramaswami
# email  : ramaswami.arjun@gmail.com
#
# Compiles Code to execute FFTW3 library for 3d FFT

# Variables
DEBUG := 0

# Compilers
# Gcc with OpenMPI, FFTW with OpenMPI
CC := gcc -std=c99  
MPICC := mpicc 
LIBS := -fopenmp -lm
CFLAGS := -march=native -Wall

# Intel MPI requires installing FFTW with icc 
#CC := icc -std=c99 
#MPICC := mpiicc 
#LIBS := -qopenmp
#CFLAGS := -O3 -xHOST -Wall

# debug flag enables heap memory leak sanitization along with prints
ifeq ($(DEBUG),1)
	CFLAGS := $(CFLAGS) -O1 -fsanitize=leak -ggdb3 -DDEBUG
else
	CFLAGS := $(CFLAGS) -O3 -funroll-loops -mtune=native
endif

ifeq ($(VERBOSE),1)
	CFLAGS := $(CFLAGS) -DVERBOSE
endif

ifeq ($(MEASURE),1)
	CFLAGS := $(CFLAGS) -DMEASURE
endif

ifeq ($(PATIENT),1)
	CFLAGS := $(CFLAGS) -DPATIENT
endif

ifeq ($(EXHAUSTIVE),1)
	CFLAGS := $(CFLAGS) -DEXHAUSTIVE
endif

SRCS := $(wildcard main.c src/*.c)

CFLAGS := $(CFLAGS)

INC := -Iinclude 

LIBS += -lfftw3 -lfftw3f -lfftw3_omp -lfftw3f_omp -lfftw3_mpi -lfftw3f_mpi

# Target
TARGETDIR := bin
TARGET := fftw

all :
	mkdir -p $(TARGETDIR)
	$(MPICC) $(LIBS) $(CFLAGS) $(SRCS) $(INC) -o $(TARGETDIR)/$(TARGET)

clean : 
	rm -rf $(TARGETDIR)/

.PHONY : all clean
