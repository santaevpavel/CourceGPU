##############################################################
# Makefile                                                   #
#                                                            #
# Author      : Chong Ding <cding@mymail.mines.edu>          #
# Version     : 0.1                                          #
# Date        : 20100915                                     #
# Discription : generic Makefile for making cuda programs    #
##############################################################

BIN               := out 
CXXFLAGS          := -O3 -g -Wall


CUDA_INSTALL_PATH ?= /opt/cuda_80ea
CUDA_SDK_PATH ?= /opt/cudasdk_5.0


NVCC ?= $(CUDA_INSTALL_PATH)/bin/nvcc
INCD = -I"$(CUDA_SDK_PATH)/C/common/inc" -I"$(CUDA_INSTALL_PATH)/include" -I"./" -I/usr/include/libpng12
LIBS = -L"$(CUDA_INSTALL_PATH)/lib" -L"$(CUDA_INSTALL_PATH)/lib64" -lcudart -L"$(CUDA_SDK_PATH)/C/common/lib" -lstdc++ -lpthread -lpng12
CUDA_SDK?=3
COMMONFLAGS = -DCUDA_SDK=$(CUDA_SDK)
#NVCCFLAGS := --ptxas-options=-v -O3 -G -g 
NVCCFLAGS_WITH_MATH := -use_fast_math
NVCCFLAGS := 


# files 
CU_SOURCES        := main.cu libpng.cu
HEADERS           := $(wildcard *.h)
CPP_OBJS          := $(patsubst %.cpp, %.o, $(CPP_SOURCES))
CU_OBJS           := $(patsubst %.cu, %.cu_o, $(CU_SOURCES))

%.cu_o : %.cu
	$(NVCC) $(NVCCFLAGS) -c $(INCD) -o $@ $<

%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) $(INCD) -o $@ $<

$(BIN): $(CPP_OBJS) $(CU_OBJS)
	$(CXX) -o $(BIN) $(CU_OBJS) $(CPP_OBJS) $(LDFLAGS) $(INCD) $(LIBS)

clean:
	rm -f $(BIN) *.o *.cu_o


