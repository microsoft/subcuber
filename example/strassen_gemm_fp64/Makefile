# Makefile for Strassen-based GEMM examples

# Variables
CUTLASS = ../../cutlass/include/
CUTLASS_COMMON = ../../cutlass/examples/common
UTIL_INCLUDE = ../../cutlass/tools/util/include/
STRASSEN_CUTLASS = ../../include
SPLIT_COMPILE=1
AMPERE_GENCODE = --generate-code=arch=compute_80,code=[compute_80,sm_80]
HOPPER_GENCODE = --generate-code=arch=compute_90a,code=[compute_90a,sm_90a]

ifeq ($(ARCH), HOPPER)
	GENCODE = $(HOPPER_GENCODE)
endif
ifeq ($(ARCH), AMPERE)
	GENCODE = $(AMPERE_GENCODE)
endif

NVCC_FLAGS = -std=c++17 -O3 -Xptxas -v --split-compile=$(SPLIT_COMPILE)
INCLUDES = -I $(STRASSEN_CUTLASS) -I $(CUTLASS) -I $(CUTLASS_COMMON) -I $(UTIL_INCLUDE)

# Targets
all: strassen_winograd_threadblock strassen_winograd_threadblock_splitk strassen_global_256 strassen_winograd_min_lds_presum0_global_256 \
		 strassen_winograd_min_lds_presum1_global_256 strassen_winograd_min_lds_presum2_global_256 \
		 strassen_winograd_min_lds_presum4_global_256

# Use Ampere Gencode because H100 gencode generates nans
strassen_winograd_threadblock: strassen_gemm.cu
	nvcc $(NVCC_FLAGS) strassen_gemm.cu $(INCLUDES) -DSTRASSEN_THREADBLOCK -DTILE_SIZE_128 -o $@ $(AMPERE_GENCODE) -DALGO_STRASSEN_WINOGRAD

# strassen_global_128: strassen_gemm.cu
# 	nvcc $(NVCC_FLAGS) strassen_gemm.cu $(INCLUDES) -DSTRASSEN_GLOBAL_LEVEL1 -DTILE_SIZE_128 -o $@ $(GENCODE) -DALGO_STRASSEN

strassen_winograd_global_128: strassen_gemm.cu
	nvcc $(NVCC_FLAGS) strassen_gemm.cu $(INCLUDES) -DSTRASSEN_GLOBAL_LEVEL1 -DTILE_SIZE_128 -o $@ $(GENCODE) -DALGO_STRASSEN_WINOGRAD

strassen_winograd_min_lds_presum0_global_128: strassen_gemm.cu
	nvcc $(NVCC_FLAGS) strassen_gemm.cu $(INCLUDES) -DMIN_LDS_NO_PRESUM -DTILE_SIZE_128 -o $@ $(GENCODE) -DALGO_STRASSEN_WINOGRAD

strassen_global_256: strassen_gemm.cu
	nvcc $(NVCC_FLAGS) strassen_gemm.cu $(INCLUDES) -DSTRASSEN_GLOBAL_LEVEL1 -DTILE_SIZE_256 -o $@ $(GENCODE) -DALGO_STRASSEN

strassen_winograd_global_256: strassen_gemm.cu
	nvcc $(NVCC_FLAGS) strassen_gemm.cu $(INCLUDES) -DSTRASSEN_GLOBAL_LEVEL1 -DTILE_SIZE_256 -o $@ $(GENCODE) -DALGO_STRASSEN_WINOGRAD

strassen_winograd_min_lds_presum0_global_256: strassen_gemm.cu
	nvcc $(NVCC_FLAGS) strassen_gemm.cu $(INCLUDES) -DMIN_LDS_NO_PRESUM -DTILE_SIZE_256 -o $@ $(GENCODE) -DALGO_STRASSEN_WINOGRAD

strassen_winograd_min_lds_presum1_global_256: strassen_gemm.cu
	nvcc $(NVCC_FLAGS) strassen_gemm.cu $(INCLUDES) -DMIN_LDS_ONE_PRESUM -DTILE_SIZE_256 -o $@ $(GENCODE) -DALGO_STRASSEN_WINOGRAD

strassen_winograd_min_lds_presum2_global_256: strassen_gemm.cu
	nvcc $(NVCC_FLAGS) strassen_gemm.cu $(INCLUDES) -DMIN_LDS_TWO_PRESUM -DTILE_SIZE_256 -o $@ $(GENCODE) -DALGO_STRASSEN_WINOGRAD

strassen_winograd_min_lds_presum4_global_128_fp64: strassen_gemm_fp64.cu
	nvcc $(NVCC_FLAGS) strassen_gemm_fp64.cu $(INCLUDES) -DMIN_LDS_FOUR_PRESUM -DTILE_SIZE_128 -o $@ $(GENCODE) -DALGO_STRASSEN_WINOGRAD

strassen_winograd_min_lds_presum4_global_256_splitk: strassen_gemm.cu
	nvcc $(NVCC_FLAGS) strassen_gemm.cu $(INCLUDES) -DMIN_LDS_FOUR_PRESUM -DTILE_SIZE_256 -o $@ $(GENCODE) -DALGO_STRASSEN_WINOGRAD -DSPLIT_K

strassen_winograd_min_lds_presum4_fused_global_256: strassen_gemm.cu
	nvcc $(NVCC_FLAGS) strassen_gemm.cu $(INCLUDES) -DMIN_LDS_FOUR_PRESUM_FUSED -DTILE_SIZE_256 -o $@ $(GENCODE) -DALGO_STRASSEN_WINOGRAD

# Use Ampere Gencode because H100 gencode generates nans
strassen_winograd_threadblock_splitk: strassen_gemm.cu
	nvcc $(NVCC_FLAGS) strassen_gemm.cu $(INCLUDES) -DSTRASSEN_THREADBLOCK -DTILE_SIZE_128 -DSPLIT_K=1 -o $@ $(AMPERE_GENCODE) -DALGO_STRASSEN_WINOGRAD

# strassen_global_128_splitk: strassen_gemm.cu
# 	nvcc $(NVCC_FLAGS) strassen_gemm.cu $(INCLUDES) -DSTRASSEN_GLOBAL_LEVEL1 -DTILE_SIZE_128 -DSPLIT_K=1 -o $@ $(GENCODE) -DALGO_STRASSEN

strassen_winograd_global_128_splitk: strassen_gemm.cu
	nvcc $(NVCC_FLAGS) strassen_gemm.cu $(INCLUDES) -DSTRASSEN_GLOBAL_LEVEL1 -DTILE_SIZE_128 -DSPLIT_K=1 -o $@ $(GENCODE) -DALGO_STRASSEN_WINOGRAD

strassen_winograd_min_lds_presum0_global_128_splitk: strassen_gemm.cu
	nvcc $(NVCC_FLAGS) strassen_gemm.cu $(INCLUDES) -DMIN_LDS_NO_PRESUM -DTILE_SIZE_128 -o $@ $(GENCODE) -DALGO_STRASSEN_WINOGRAD -DSPLIT_K=1

strassen_global_256_splitk: strassen_gemm.cu
	nvcc $(NVCC_FLAGS) strassen_gemm.cu $(INCLUDES) -DSTRASSEN_GLOBAL_LEVEL1 -DTILE_SIZE_256 -DSPLIT_K=1 -o $@ $(GENCODE)

strassen_global_128_subgemm_parallel_splitk: strassen_gemm.cu
	nvcc $(NVCC_FLAGS) strassen_gemm.cu $(INCLUDES) -DSTRASSEN_GLOBAL_LEVEL1 -DTILE_SIZE_128 -DSPLIT_K=1 -DSUB_GEMM_PARALLEL=1 -o $@ $(GENCODE)

strassen_global_256_subgemm_parallel_splitk: strassen_gemm.cu
	nvcc $(NVCC_FLAGS) strassen_gemm.cu $(INCLUDES) -DSTRASSEN_GLOBAL_LEVEL1 -DTILE_SIZE_256 -DSPLIT_K=1 -DSUB_GEMM_PARALLEL=1 -o $@ $(GENCODE)

# Clean up generated files
clean:
	rm -f strassen_threadblock strassen_global_128 strassen_global_256 \
	      strassen_threadblock_splitk strassen_global_128_splitk strassen_global_256_splitk \
		  strassen_threadblock_subgemm_parallel_splitk strassen_global_128_subgemm_parallel_splitk strassen_global_256_subgemm_parallel_splitk
