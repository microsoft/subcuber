# Makefile for CUTLASS-based GEMM examples

# Variables
SPLIT_COMPILE=1
CUTLASS = ../../../cutlass/include/
CUTLASS_COMMON = ../../../cutlass/examples/common
UTIL_INCLUDE = ../../../cutlass/tools/util/include/
GENCODE = --generate-code=arch=compute_90a,code=[compute_90a,sm_90a]
NVCC_FLAGS = -std=c++17 -O3 -Xptxas -v,-O3 $(GENCODE) --split-compile=$(SPLIT_COMPILE) -lineinfo --expt-relaxed-constexpr -DCUTLASS_VERSIONS_GENERATED -DNDEBUG -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -DCUTLASS_ENABLE_GDC_FOR_SM100=1 --expt-relaxed-constexpr -ftemplate-backtrace-limit=0 -DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED -DCUTLASS_TEST_LEVEL=0 -DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1 -DCUTLASS_CONV_UNIT_TEST_RIGOROUS_SIZE_ENABLED=1 -DCUTLASS_DEBUG_TRACE_LEVEL=0 #--keep
CXXFLAGS_COMMON =
INCLUDES = -I $(CUTLASS) -I $(CUTLASS_COMMON) -I $(UTIL_INCLUDE)
CUDA_LIBS = -lcudart  -lcublas 
STRASSEN_CUTLASS =  -I ../../include

# Conditional flags based on the compiler

COMPILER = /usr/local/cuda/bin/nvcc
CXXFLAGS = $(NVCC_FLAGS)

# Targets
all: hopper_f16_tensorop_gemm_pingpong hopper_f16_tensorop_gemm_cooperative

hopper_f16_tensorop_gemm_pingpong: hopper_f16_tensorop_gemm.cu
	$(COMPILER) $(CXXFLAGS_COMMON) $(CXXFLAGS) $< $(INCLUDES) $(STRASSEN_CUTLASS) -o $@ $(CUDA_LIBS) -DPINGPONG

hopper_f16_tensorop_gemm_cooperative: hopper_f16_tensorop_gemm.cu
	$(COMPILER) $(CXXFLAGS_COMMON) $(CXXFLAGS) $< $(INCLUDES) $(STRASSEN_CUTLASS) -o $@ $(CUDA_LIBS) -DCOOPERATIVE

# Clean up generated files
clean: 
	rm -f hopper_f16_tensorop_gemm
