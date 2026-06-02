# Makefile for CUTLASS-based GEMM examples

# Variables
CUTLASS = ../../cutlass/include/
CUTLASS_COMMON = ../../cutlass/examples/common
UTIL_INCLUDE = ../../cutlass/tools/util/include/
GENCODE = --generate-code=arch=compute_80,code=[compute_80,sm_80]
NVCC_FLAGS = -std=c++17 -O3 -Xptxas -v $(GENCODE) -lineinfo
CLANG_PATH = /home/anmoreira/llvm-project/build-gpu/bin
CXXFLAGS_COMMON = -std=c++17 -O3
CLANG_FLAGS = --cuda-gpu-arch=sm_86 -x cuda --cuda-path=/usr/local/cuda-12.2
INCLUDES = -I $(CUTLASS) -I $(CUTLASS_COMMON) -I $(UTIL_INCLUDE)
CUDA_LIBS = -L/usr/local/cuda-12.2/lib64 -lcudart  -lcublas 

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
all: ampere_f16_tensorop_gemm

ampere_tf32_tensorop_gemm: ampere_tf32_tensorop_gemm.cu
	$(COMPILER) $(CXXFLAGS_COMMON) $(CXXFLAGS) $< $(INCLUDES) -o $@ $(CUDA_LIBS)

# Clean up generated files
clean: 
	rm -f ampere_tf32_tensorop_gemm

