# Author : Arjun Ramaswami
# email  : ramaswami.arjun@gmail.com
#
# Compiles Code to run MKL FFT3d

CC := icc

CFLAGS :=
# debug flag enables heap memory leak sanitization along with prints
ifeq ($(DEBUG),1)
	CFLAGS := $(CFLAGS) -O1 -DDEBUG -g -check-pointers=rw -check-pointers-dangling=all
else
	CFLAGS := $(CFLAGS) -O3 -funroll-loops -mtune=native
endif

SRCS     := main.c $(wildcard common/*.c)
INC_DIRS := -Icommon
LIB_DIRS := 

LIBS := -mkl -lm

TARGET := host_dp

all: 
	$(CC) $(CFLAGS) $(SRCS) $(INC_DIRS) $(LIB_DIRS) $(LIBS) -o $(TARGET)

clean: 
	rm $(TARGET)

.PHONY : all clean