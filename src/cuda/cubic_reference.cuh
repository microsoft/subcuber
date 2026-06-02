#pragma once

/// Naive reference GEMM computation.
template<
  typename ElementA,
  typename ElementB,
  typename ElementC,
  typename ElementScalar,
  typename ElementAccumulator>
__global__ void ReferenceGemm_kernel(
  int M,
  int N,
  int K,
  ElementScalar alpha,
  ElementA const *A,
  int lda,
  ElementB const *B,
  int ldb,
  ElementScalar beta,
  ElementC *C,
  int ldc) {

  cutlass::NumericConverter<ElementC, ElementAccumulator> convert_output;

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < M && j < N) {
    ElementAccumulator accumulator = ElementAccumulator();

    for (int k = 0; k < K; ++k) {
      accumulator += ElementAccumulator(A[i * lda + k]) * ElementAccumulator(B[k * ldb + j]);
    }

    ElementAccumulator output = ElementAccumulator(alpha) * accumulator +
      ElementAccumulator(beta) * ElementAccumulator(C[i * ldc + j]);

    C[i * ldc + j] = convert_output(output);
  }
}

/// Reference GEMM computation.
template<
  typename ElementA,
  typename ElementB,
  typename ElementC,
  typename ElementScalar,
  typename ElementAccumulator = ElementScalar>
cudaError_t ReferenceGemm(
  int M,
  int N,
  int K,
  ElementScalar alpha,
  ElementA const *A,
  int lda,
  ElementB const *B,
  int ldb,
  ElementScalar beta,
  ElementC *C,
  int ldc) {

  dim3 block(16, 16);
  dim3 grid(
    (M + block.x - 1) / block.x,
    (N + block.y - 1) / block.y
  );

  ReferenceGemm_kernel<ElementA, ElementB, ElementC, ElementScalar, ElementAccumulator><<< grid, block >>>(
    M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);

  return cudaDeviceSynchronize();
}