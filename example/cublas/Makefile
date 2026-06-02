all: cublas_fp16_gemm_bench

cublas_fp16_gemm_bench: cublas_fp16_gemm_bench.cu
	nvcc $< -lcublas -lcublasLt -o $@

