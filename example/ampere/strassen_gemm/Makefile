# Makefile for Strassen-based GEMM examples

# Variables
CUTLASS = ../../../cutlass/include/
CUTLASS_COMMON = ../../../cutlass/examples/common
UTIL_INCLUDE = ../../../cutlass/tools/util/include/
STRASSEN_CUTLASS = ../../../include
STRASSEN_CUTLASS_COMMON = ../../
SRC_INCLUDE = ../../../src

SPLIT_COMPILE=1
AMPERE_GENCODE = --generate-code=arch=compute_80,code=[compute_80,sm_80]
HOPPER_GENCODE = --generate-code=arch=compute_90a,code=[compute_90a,sm_90a]
ARCH = AMPERE
ifeq ($(ARCH), HOPPER)
	GENCODE = $(HOPPER_GENCODE)
endif
ifeq ($(ARCH), AMPERE)
	GENCODE = $(AMPERE_GENCODE)
endif

NVCC = /usr/local/cuda/bin/nvcc
NVCC_FLAGS = -std=c++17 -O3 -Xptxas -v --split-compile=$(SPLIT_COMPILE) -DCUTLASS_API_v2
INCLUDES = -I $(STRASSEN_CUTLASS) -I $(CUTLASS) -I $(CUTLASS_COMMON) -I $(UTIL_INCLUDE) -I $(STRASSEN_CUTLASS_COMMON) -I $(SRC_INCLUDE)

# Targets
all: strassen_winograd_threadblock strassen_winograd_fused_in_sum strassen_winograd_presum_256 strassen_winograd_presum_256_splitk strassen_winograd_presum_256_level2

# strassen_winograd_min_lds_presum0_global_256 \
		 strassen_winograd_min_lds_presum1_global_256 strassen_winograd_min_lds_presum2_global_256 \


# Use Ampere Gencode because H100 gencode generates nans
strassen_winograd_threadblock: strassen_gemm.cu
	$(NVCC) $(NVCC_FLAGS) $< $(INCLUDES) -DTHREADBLOCK -DTILE_SIZE_128 -o $@ $(AMPERE_GENCODE) -DSPLIT_K

strassen_winograd_fused_in_sum: strassen_gemm.cu
	$(NVCC) $(NVCC_FLAGS) $< $(INCLUDES) -DFUSED_IN_SUM -DTILE_SIZE_256 -o $@ $(GENCODE) -DSPLIT_K

strassen_winograd_presum_256: strassen_gemm.cu
	$(NVCC) $(NVCC_FLAGS) $< $(INCLUDES) -DPRESUM -DTILE_SIZE_256 -o $@ $(GENCODE) -DSUB_GEMM_PARALLEL

strassen_winograd_presum_256_splitk: strassen_gemm.cu
	$(NVCC) $(NVCC_FLAGS) $< $(INCLUDES) -DPRESUM -DTILE_SIZE_256 -o $@ $(GENCODE) -DSPLIT_K

strassen_winograd_presum_256_level2: strassen_gemm_level2.cu
	$(NVCC) $(NVCC_FLAGS) $< $(INCLUDES) -DPRESUM -DTILE_SIZE_256 -o $@ $(GENCODE)

# Clean up generated files
clean:
	find . -type f -exec sh -c "file -i '{}' | grep -q 'charset=binary'" \; -print | xargs rm -f
