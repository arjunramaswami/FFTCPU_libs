# Author : Arjun Ramaswami
# email  : ramaswami.arjun@gmail.com
#
# Compiles Code to run FFTW FFT3d

# Variables
FFT_DP := 0
DEBUG := 0

# Compilers
CC := gcc

# Compiler Flags
CFLAGS := -march=native
# debug flag enables heap memory leak sanitization along with prints
ifeq ($(DEBUG),1)
	CFLAGS := $(CFLAGS) -O1 -fsanitize=leak -ggdb3 -DDEBUG
else
	CFLAGS := $(CFLAGS) -O3 -funroll-loops
endif

SRCS := $(wildcard main.c common/*.c)

# Pass FFT_DP=1 as cmd line arg to activate double precision compilation 
ifeq ($(FFT_DP),1)
 	FPFLAGS :=  $(FPFLAGS)
else
	FPFLAGS := $(FPFLAGS) -D __FFT_SP
endif
CFLAGS := $(CFLAGS) $(SRCS) $(FPFLAGS)

# Include Directories 
INC := -Icommon 
INC_DIRS :=  $(FFTW_INC) $(INC)

# Linker Directories
LIB_DIRS :=  $(FFTW_DIR) -lm

# Libraries
LIBS := -lrt -lpthread -lfftw3 -lfftw3f

# Target
TARGETDIR := .
TARGET := host

all : info
	mkdir -p $(TARGETDIR)
	$(CC) $(CFLAGS) $(INC_DIRS) $(LIB_DIRS)	$(LIBS) -o $(TARGETDIR)/$(TARGET)

clean : 
	rm -f $(TARGETDIR)/$(TARGET)

info :
	$(info -------------------------------)
	$(info FFTW Version     : $(FFTW_VERSION))
	$(info FFTW Include Dir : $(FFTW_INC) ) 
	$(info FFTW Shared Libs : $(FFTW_DIR) ) 
	$(info Precision        : $(FPGA_DP) (0/sp 1/dp))
	$(info -------------------------------)

.PHONY : all clean # ---------------
