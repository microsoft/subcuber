#include "cuda/cubic_reference.cuh"

#pragma once

/// Level-1 Strassen-Winograd reference GEMM computation.
template<
  typename ElementA,
  typename ElementB,
  typename ElementC,
  typename ElementScalar,
  typename ElementAccumulator>
__global__ void ReferenceStrassenWinogradGemm_kernel(
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
  using ElementPresumA = ElementAccumulator;
  using ElementPresumB = ElementAccumulator;
  using ElementPostsum = ElementAccumulator;

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < M && j < N) {
    int half_m = M / 2;
    int half_n = N / 2;
    int half_k = K / 2;

    bool bottom_half = (i >= half_m);
    bool right_half = (j >= half_n);

    int row = bottom_half ? (i - half_m) : i;
    int col = right_half ? (j - half_n) : j;

    int top_row = row;
    int bottom_row = half_m + row;
    int left_col = col;
    int right_col = half_n + col;

    ElementAccumulator m0 = ElementAccumulator();
    ElementAccumulator m1 = ElementAccumulator();
    ElementAccumulator m2 = ElementAccumulator();
    ElementAccumulator m3 = ElementAccumulator();
    ElementAccumulator m4 = ElementAccumulator();
    ElementAccumulator m5 = ElementAccumulator();
    ElementAccumulator m6 = ElementAccumulator();

    for (int k = 0; k < half_k; ++k) {
      ElementPresumA a0 = ElementPresumA(A[top_row * lda + k]);
      ElementPresumA a1 = ElementPresumA(A[top_row * lda + half_k + k]);
      ElementPresumA a2 = ElementPresumA(A[bottom_row * lda + k]);
      ElementPresumA a3 = ElementPresumA(A[bottom_row * lda + half_k + k]);

      ElementPresumB b0 = ElementPresumB(B[k * ldb + left_col]);
      ElementPresumB b1 = ElementPresumB(B[k * ldb + right_col]);
      ElementPresumB b2 = ElementPresumB(B[(half_k + k) * ldb + left_col]);
      ElementPresumB b3 = ElementPresumB(B[(half_k + k) * ldb + right_col]);

      ElementPresumA s1_ = a2 + a3;
      ElementPresumA s2_ = s1_ - a0;
      ElementPresumB b10_ = b1 - b0;
      ElementPresumB b31_ = b3 - b1;
      ElementPresumB s3_ = b31_ + b0;
      ElementPresumA a02_ = a0 - a2;
      ElementPresumA a1s2_ = a1 - s2_;
      ElementPresumA s3b2_ = s3_ - b2;

      ElementA s1 = ElementA(s1_);
      ElementA s2 = ElementA(s2_);
      ElementA a02 = ElementA(a02_);
      ElementA a1s2 = ElementA(a1s2_);
      
      ElementB b10 = ElementB(b10_);
      ElementB b31 = ElementB(b31_);
      ElementB s3 = ElementB(s3_);
      ElementB s3b2 = ElementB(s3b2_);

      m0 += ElementAccumulator(a0) * ElementAccumulator(b0);
      m1 += ElementAccumulator(a1) * ElementAccumulator(b2);
      m2 += ElementAccumulator(s2) * ElementAccumulator(s3);
      m3 += ElementAccumulator(a02) * ElementAccumulator(b31);
      m4 += ElementAccumulator(s1) * ElementAccumulator(b10);
      m5 += ElementAccumulator(a1s2) * ElementAccumulator(b3);
      m6 += ElementAccumulator(a3) * ElementAccumulator(s3b2);
    }

    ElementPostsum v1 = ElementPostsum(m0) + ElementPostsum(m2);
    ElementPostsum v2 = ElementPostsum(v1) + ElementPostsum(m3);
    ElementPostsum v3 = ElementPostsum(v1) + ElementPostsum(m4);

    ElementPostsum accumulator = ElementPostsum(0.0f);

    if (!bottom_half && !right_half) {
      accumulator = ElementPostsum(m0) + ElementPostsum(m1);
    } else if (!bottom_half && right_half) {
      accumulator = v3 + ElementPostsum(m5);
    } else if (bottom_half && !right_half) {
      accumulator = v2 - ElementPostsum(m6);
    } else {
      accumulator = v2 + ElementPostsum(m4);
    }

    ElementPostsum output = ElementPostsum(alpha) * accumulator +
      ElementPostsum(beta) * ElementPostsum(C[i * ldc + j]);

    C[i * ldc + j] = ElementC(output);
  }
}

template<
  typename ElementA,
  typename ElementB,
  typename ElementSum>
__global__ void ReferenceStrassenWinogradPresums_kernel(
  int M,
  int N,
  int K,
  ElementA const *A,
  int lda,
  ElementB const *B,
  int ldb,
  ElementSum *sums) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  int k = threadIdx.z + blockIdx.z * blockDim.z;

  int half_m = M / 2;
  int half_n = N / 2;
  int half_k = K / 2;

  if (i < M && j < N && k < half_k) {
    bool bottom_half = (i >= half_m);
    bool right_half = (j >= half_n);

    int row = bottom_half ? (i - half_m) : i;
    int col = right_half ? (j - half_n) : j;

    int top_row = row;
    int bottom_row = half_m + row;
    int left_col = col;
    int right_col = half_n + col;

    ElementA a0 = ElementA(A[top_row * lda + k]);
    ElementA a1 = ElementA(A[top_row * lda + half_k + k]);
    ElementA a2 = ElementA(A[bottom_row * lda + k]);
    ElementA a3 = ElementA(A[bottom_row * lda + half_k + k]);

    ElementB b0 = ElementB(B[k * ldb + left_col]);
    ElementB b1 = ElementB(B[k * ldb + right_col]);
    ElementB b2 = ElementB(B[(half_k + k) * ldb + left_col]);
    ElementB b3 = ElementB(B[(half_k + k) * ldb + right_col]);

    ElementA s1 = a2 + a3;
    ElementA s2 = s1 - a0;
    ElementB b10 = b1 - b0;
    ElementB b31 = b3 - b1;
    ElementB s3 = b31 + b0;
    ElementA a02 = a0 - a2;
    ElementA a1s2 = a1 - s2;
    ElementB s3b2 = s3 - b2;

    size_t offset = ((size_t(i) * size_t(N) + size_t(j)) * size_t(half_k) + size_t(k)) * 8;

    sums[offset + 0] = ElementSum(s1);
    sums[offset + 1] = ElementSum(s2);
    sums[offset + 2] = ElementSum(b10);
    sums[offset + 3] = ElementSum(b31);
    sums[offset + 4] = ElementSum(s3);
    sums[offset + 5] = ElementSum(a02);
    sums[offset + 6] = ElementSum(a1s2);
    sums[offset + 7] = ElementSum(s3b2);
  }
}

template<
  typename ElementA,
  typename ElementB,
  typename ElementSum>
cudaError_t ReferenceStrassenWinogradPresums(
  int M,
  int N,
  int K,
  ElementA const *A,
  int lda,
  ElementB const *B,
  int ldb,
  ElementSum *sums) {

  if ((M % 2) || (N % 2) || (K % 2) || !A || !B || !sums) {
    return cudaErrorInvalidValue;
  }

  dim3 block(8, 8, 4);
  dim3 grid(
    (M + block.x - 1) / block.x,
    (N + block.y - 1) / block.y,
    ((K / 2) + block.z - 1) / block.z
  );

  ReferenceStrassenWinogradPresums_kernel<ElementA, ElementB, ElementSum><<< grid, block >>>(
    M, N, K, A, lda, B, ldb, sums);

  return cudaDeviceSynchronize();
}

template<
  typename ElementA,
  typename ElementB,
  typename ElementC,
  typename ElementScalar,
  typename ElementAccumulator = ElementScalar>
cudaError_t ReferenceStrassenWinogradGemm(
  int M,
  int N,
  int K,
  int level,
  ElementScalar alpha,
  ElementA const *A,
  int lda,
  ElementB const *B,
  int ldb,
  ElementScalar beta,
  ElementC *C,
  int ldc) {

  if (level != 1 || (M % 2) || (N % 2) || (K % 2)) {
    return ReferenceGemm<ElementA, ElementB, ElementC, ElementScalar, ElementAccumulator>(
      M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
  }

  dim3 block(16, 16);
  dim3 grid(
    (M + block.x - 1) / block.x,
    (N + block.y - 1) / block.y
  );

  ReferenceStrassenWinogradGemm_kernel<ElementA, ElementB, ElementC, ElementScalar, ElementAccumulator><<< grid, block >>>(
    M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);

  return cudaDeviceSynchronize();
}