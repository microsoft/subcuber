# Makefile for CUTLASS-based GEMM examples

# Variables
SPLIT_COMPILE=1
CUTLASS = ../../cutlass/include/
CUTLASS_COMMON = ../../cutlass/examples/common
UTIL_INCLUDE = ../../cutlass/tools/util/include/
GENCODE = --generate-code=arch=compute_80,code=[compute_80,sm_80]
NVCC_FLAGS = -std=c++17 -O3 -Xptxas -v,-O3 $(GENCODE) --split-compile=$(SPLIT_COMPILE) -lineinfo #--keep
CLANG_PATH = /home/anmoreira/llvm-project/build-gpu/bin
CXXFLAGS_COMMON =
CLANG_FLAGS = --cuda-gpu-arch=sm_86 -x cuda --cuda-path=/usr/local/cuda-12.2
INCLUDES = -I $(CUTLASS) -I $(CUTLASS_COMMON) -I $(UTIL_INCLUDE)
CUDA_LIBS = -L/usr/local/cuda-12.2/lib64 -lcudart  -lcublas 
STRASSEN_CUTLASS =  -I ../../include

# Conditional flags based on the compiler

ifeq ($(COMPILER),clang)
    COMPILER = $(CLANG_PATH)/clang++
    CXXFLAGS = $(CLANG_FLAGS)
else ifeq ($(COMPILER), nvcc)
    COMPILER = /usr/local/cuda/bin/nvcc
    CXXFLAGS = $(NVCC_FLAGS)
	CUDA_LIBS =
else
    COMPILER = /usr/local/cuda/bin/nvcc
	CXXFLAGS = $(NVCC_FLAGS)
	CUDA_LIBS =
endif

# Targets
all: strassen_winograd_presum4_ampere_tf32_tensorop_gemm

strassen_global_ampere_f16_tensorop_gemm: strassen_ampere_f16_tensorop_gemm.cu
	$(COMPILER) $(CXXFLAGS_COMMON) $(CXXFLAGS) $< $(INCLUDES) $(STRASSEN_CUTLASS) -o $@ $(CUDA_LIBS) -DSTRASSEN_GLOBAL_LEVEL1

strassen_winograd_presum4_ampere_tf32_tensorop_gemm: strassen_ampere_tf32_tensorop_gemm.cu
	$(COMPILER) $(CXXFLAGS_COMMON) $(CXXFLAGS) $< $(INCLUDES) $(STRASSEN_CUTLASS) -o $@ $(CUDA_LIBS) -DSTRASSEN_FOUR_PRESUM_LEVEL1

# Clean up generated files
clean: 
	rm -f strassen_global_ampere_f16_tensorop_gemm strassen_winograd_presum4_ampere_tf32_tensorop_gemm
	
