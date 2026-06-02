# Makefile for CUTLASS-based GEMM examples

# Variables
CUTLASS = ../../../cutlass/include/
CUTLASS_COMMON = ../../../cutlass/examples/common
UTIL_INCLUDE = ../../../cutlass/tools/util/include/
HOPPER_GENCODE = --generate-code=arch=compute_90a,code=[compute_90a,sm_90a]
AMPERE_GENCODE = --generate-code=arch=compute_80,code=[compute_80,sm_80]
ARCH = AMPERE

ifeq ($(ARCH), HOPPER)
        GENCODE = $(HOPPER_GENCODE)
endif
ifeq ($(ARCH), AMPERE)
        GENCODE = $(AMPERE_GENCODE)
endif

NVCC_FLAGS = -std=c++17 -Xptxas -v $(GENCODE)
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
all: basic_gemm_128 basic_gemm_256 basic_gemm_128_splitk basic_gemm_256_splitk

basic_gemm_128: basic_gemm.cu
	$(COMPILER) $(CXXFLAGS_COMMON) $(CXXFLAGS) basic_gemm.cu $(INCLUDES) -DTILE_SIZE_128 -DSPLIT_K=0 -o $@ $(CUDA_LIBS)

basic_gemm_256: basic_gemm.cu
	$(COMPILER) $(CXXFLAGS_COMMON) $(CXXFLAGS) basic_gemm.cu $(INCLUDES) -DTILE_SIZE_256 -DSPLIT_K=0 -o $@ $(CUDA_LIBS)

basic_gemm_128_splitk: basic_gemm.cu
	$(COMPILER) $(CXXFLAGS_COMMON) $(CXXFLAGS) basic_gemm.cu $(INCLUDES) -DTILE_SIZE_128 -DSPLIT_K=1 -o $@ $(CUDA_LIBS)

basic_gemm_256_splitk: basic_gemm.cu
	$(COMPILER) $(CXXFLAGS_COMMON) $(CXXFLAGS) basic_gemm.cu $(INCLUDES) -DTILE_SIZE_256 -DSPLIT_K=1 -o $@ $(CUDA_LIBS)

# Clean up generated files
clean: 
	rm -f basic_gemm_128 basic_gemm_256 basic_gemm_128_splitk basic_gemm_256_splitk

