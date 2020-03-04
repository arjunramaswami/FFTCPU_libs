# Author : Arjun Ramaswami
# email  : ramaswami.arjun@gmail.com
#
# Compiles Code to execute FFTW3 library for 3d FFT

# Variables
DEBUG := 0

# Compilers
CC := gcc

# Compiler Flags
CFLAGS := -march=native -Wall

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

LIBS := -lfftw3 -lfftw3f -lfftw3_omp -lfftw3f_omp -lm -fopenmp

# Target
TARGETDIR := bin
TARGET := fftw

all :
	mkdir -p $(TARGETDIR)
	$(CC) $(LIBS) $(CFLAGS) $(SRCS) $(INC) -o $(TARGETDIR)/$(TARGET)

clean : 
	rm -f $(TARGETDIR)/$(TARGET)

.PHONY : all clean
