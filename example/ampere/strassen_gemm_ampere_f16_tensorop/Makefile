# Makefile for CUTLASS-based GEMM examples

# Variables
SPLIT_COMPILE=1
CUTLASS = ../../../cutlass/include/
CUTLASS_COMMON = ../../../cutlass/examples/common
UTIL_INCLUDE = ../../../cutlass/tools/util/include/
GENCODE = --generate-code=arch=compute_80,code=[compute_80,sm_80]
NVCC_FLAGS = -std=c++20 -O3 -Xptxas -v,-O3 $(GENCODE) --split-compile=$(SPLIT_COMPILE) -DCUTLASS_API_v2 -DNDEBUG -lineinfo #--keep
CUDA_LIBS = -L/usr/local/cuda-12.2/lib64 -lcudart  -lcublas 
STRASSEN_CUTLASS = ../../../include
INCLUDES = -I $(CUTLASS) -I $(CUTLASS_COMMON) -I $(UTIL_INCLUDE) -I $(STRASSEN_CUTLASS) -I ../../../src/

# Conditional flags based on the compiler

COMPILER = /usr/local/cuda/bin/nvcc
CXXFLAGS = $(NVCC_FLAGS)
CUDA_LIBS =

# Targets
all: strassen_ampere_f16_tensorop_gemm strassen_ampere_f16_tensorop_level_2_gemm

strassen_ampere_f16_tensorop_gemm: strassen_ampere_f16_tensorop_gemm.cu
	$(COMPILER) $(CXXFLAGS_COMMON) $(CXXFLAGS) $< $(INCLUDES) -o $@ $(CUDA_LIBS) -DSTRASSEN_FOUR_PRESUM_LEVEL1

strassen_ampere_f16_tensorop_level_2_gemm: strassen_ampere_f16_tensorop_level_2_gemm.cu
	$(COMPILER) $(CXXFLAGS_COMMON) $(CXXFLAGS) $< $(INCLUDES) -o $@ $(CUDA_LIBS) -DSTRASSEN_FOUR_PRESUM_LEVEL1

# Clean up generated files
clean: 
	find . -type f -exec sh -c "file -i '{}' | grep -q 'charset=binary'" \; -print | xargs rm -f
	
