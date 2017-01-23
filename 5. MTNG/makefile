##############################################################
# Makefile                                                   #
#                                                            #
# Author      : Chong Ding <cding@mymail.mines.edu>          #
# Version     : 0.1                                          #
# Date        : 20100915                                     #
# Discription : generic Makefile for making cuda programs    #
##############################################################

BIN               := out 
//CXXFLAGS          := -O3 -g -Wall -std=c++11
CXXFLAGS          := -O3 -g -std=c++11 -ggdb
#CXXFLAGS          := -O3 -std=c++11


CUDA_INSTALL_PATH ?= /opt/cuda_80ea
CUDA_SDK_PATH ?= /opt/cudasdk_5.0


NVCC ?= $(CUDA_INSTALL_PATH)/bin/nvcc
INCD = -I"$(CUDA_SDK_PATH)/C/common/inc" -I"$(CUDA_INSTALL_PATH)/include" -I"./" -I/usr/include/libpng12
LIBS = -L"$(CUDA_INSTALL_PATH)/lib" -L"$(CUDA_INSTALL_PATH)/lib64" -lcudart -L"$(CUDA_SDK_PATH)/C/common/lib" -lstdc++ -lpthread -lpng12 -std=c++11
CUDA_SDK?=3
COMMONFLAGS = -DCUDA_SDK=$(CUDA_SDK)
NVCCFLAGS_WITH_MATH := -use_fast_math
#NVCCFLAGS := --compiler-options "-std=c++11"
NVCCFLAGS := -std=c++11 -G -g
#NVCCFLAGS := -std=c++11

# CPP_SOURCES		  := $(shell echo *.cpp)
# files  /Geometry /MonteCarlo /random
#CU_SOURCES        := $(shell echo *.cu)
CU_SOURCES        :=Box.cu Cylinder.cu  MaterialFactory.cu Random.cu Shape.cu ShapesExclusion.cu ShapesIntersection.cu ShapesUnion.cu Sphere.cu Main.cu 
HEADERS           := $(wildcard *.h)
CPP_OBJS          := $(patsubst %.cpp, %.o, $(CPP_SOURCES))
CU_OBJS           := $(patsubst %.cu, %.cu_o, $(CU_SOURCES))

#Box.cu Cylinder.cu  MaterialFactory.cu Random.cu Shape.cu ShapesExclusion.cu ShapesIntersection.cu ShapesUnion.cu Sphere.cu Main.cu
#
%.cu_o : %.cu
	$(NVCC) $(NVCCFLAGS) -c $(INCD) -o $@ $<

%.cu_o : %.cpp
	$(NVCC) $(NVCCFLAGS) -std=c++11 -c $(INCD) -o $@ $<

%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) $(INCD) -std=c++11 -o $@ $<

$(BIN): $(CPP_OBJS) $(CU_OBJS)
	$(CXX) -o $(BIN) $(CU_OBJS) $(CPP_OBJS) $(LDFLAGS) $(INCD) $(LIBS)

clean:
	rm -f $(BIN) *.o *.cu_o


