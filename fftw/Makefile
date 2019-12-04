# Author : Arjun Ramaswami
# email  : ramaswami.arjun@gmail.com
#
# Compiles Code to run FFTW FFT3d

# Variables
DEBUG := 0

# Compilers
CC := gcc

# Compiler Flags
CFLAGS := -march=native
# debug flag enables heap memory leak sanitization along with prints
ifeq ($(DEBUG),1)
	CFLAGS := $(CFLAGS) -O1 -fsanitize=leak -ggdb3 -DDEBUG
else
	CFLAGS := $(CFLAGS) -O3 -funroll-loops -mtune=native
endif

SRCS := $(wildcard main.c common/*.c)
CFLAGS := $(CFLAGS) $(SRCS) 
OMPFLAGS := -DOMP -fopenmp -D __FFT_DP

# Include Directories 
INC := -Icommon 
INC_DIRS :=  $(FFTW_INC) $(INC)

# Linker Directories
LIB_DIRS :=  $(FFTW_DIR) -lm

# Libraries
LIBS := -lrt -lpthread -lfftw3 -lfftw3f
OMPLIBS := -lfftw3_omp

# Target
TARGETDIR := .
TARGET := host
TARGET_DP := host_dp
TARGET_SP := host_sp
TARGET_OMP := host_omp

all : info 
	mkdir -p $(TARGETDIR)
	$(CC) $(CFLAGS) -D __FFT_SP $(INC_DIRS) $(LIB_DIRS)	$(LIBS) -o $(TARGETDIR)/$(TARGET)

dp : info
	$(CC) $(CFLAGS) -D __FFT_DP $(INC_DIRS) $(LIB_DIRS) $(LIBS) -o $(TARGETDIR)/$(TARGET_DP)

sp : info
	$(CC) $(CFLAGS) -D __FFT_SP $(INC_DIRS) $(LIB_DIRS) $(LIBS) -o $(TARGETDIR)/$(TARGET_SP)

omp : info
	mkdir -p $(TARGETDIR)
	$(CC) $(CFLAGS) $(OMPFLAGS) $(INC_DIRS) $(LIB_DIRS) $(LIBS) $(OMPLIBS) -o $(TARGETDIR)/$(TARGET_OMP)

clean : 
	rm -f $(TARGETDIR)/$(TARGET) $(TARGETDIR)/$(TARGET_DP) $(TARGETDIR)/$(TARGET_SP) $(TARGETDIR)/$(TARGET_OMP)

info :
	$(info -------------------------------)
	$(info FFTW Version     : $(FFTW_VERSION))
	$(info FFTW Include Dir : $(FFTW_INC) ) 
	$(info FFTW Shared Libs : $(FFTW_DIR) ) 
	$(info -------------------------------)

.PHONY : all clean dp sp omp info # ---------------
