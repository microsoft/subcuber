#include <algorithm>
#include <cstddef>

#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cublas_v2.h>

#include "cutlass/numeric_types.h"
#include "cuda/kernel_runner_support.cuh"

template <typename Element>
struct CublasLtType;

template <>
struct CublasLtType<float> {
  static constexpr cudaDataType_t data_type = CUDA_R_32F;
  static constexpr cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
};

template <>
struct CublasLtType<cutlass::half_t> {
  static constexpr cudaDataType_t data_type = CUDA_R_16F;
  static constexpr cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F_FAST_16F;
};

static int cublaslt_status_to_error(cublasStatus_t status) {
  return status == CUBLAS_STATUS_SUCCESS ? 0 : 10000 + static_cast<int>(status);
}

template <typename Element>
int run_cublaslt_gemm(KernelRunnerBuffers buffers, int m, int n, int k,
                      int warmup_iterations, int iterations, cudaStream_t *streams,
                      int num_streams, int, float *avg_ms) {
  using Type = CublasLtType<Element>;

  cublasLtHandle_t handle = nullptr;
  cublasLtMatmulDesc_t matmul_desc = nullptr;
  cublasLtMatrixLayout_t layout_a = nullptr;
  cublasLtMatrixLayout_t layout_b = nullptr;
  cublasLtMatrixLayout_t layout_c = nullptr;
  cublasLtMatrixLayout_t layout_d = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;
  cudaStream_t stream = streams == nullptr || num_streams == 0 ? nullptr : streams[0];
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  void *workspace = nullptr;
  constexpr size_t kWorkspaceSize = 32ULL * 1024ULL * 1024ULL;

  auto cleanup = [&]() {
    if (start != nullptr) cudaEventDestroy(start);
    if (stop != nullptr) cudaEventDestroy(stop);
    if (workspace != nullptr) cudaFree(workspace);
    if (preference != nullptr) cublasLtMatmulPreferenceDestroy(preference);
    if (layout_d != nullptr) cublasLtMatrixLayoutDestroy(layout_d);
    if (layout_c != nullptr) cublasLtMatrixLayoutDestroy(layout_c);
    if (layout_b != nullptr) cublasLtMatrixLayoutDestroy(layout_b);
    if (layout_a != nullptr) cublasLtMatrixLayoutDestroy(layout_a);
    if (matmul_desc != nullptr) cublasLtMatmulDescDestroy(matmul_desc);
    if (handle != nullptr) cublasLtDestroy(handle);
  };

  cublasStatus_t cublas_status = cublasLtCreate(&handle);
  if (cublas_status != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return cublaslt_status_to_error(cublas_status);
  }

  cublas_status = cublasLtMatmulDescCreate(&matmul_desc, Type::compute_type, CUDA_R_32F);
  if (cublas_status != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return cublaslt_status_to_error(cublas_status);
  }

  cublasOperation_t op = CUBLAS_OP_N;
  cublas_status = cublasLtMatmulDescSetAttribute(
      matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op, sizeof(op));
  if (cublas_status != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return cublaslt_status_to_error(cublas_status);
  }
  cublas_status = cublasLtMatmulDescSetAttribute(
      matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op, sizeof(op));
  if (cublas_status != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return cublaslt_status_to_error(cublas_status);
  }

  cublas_status = cublasLtMatrixLayoutCreate(&layout_a, Type::data_type, m, k, k);
  if (cublas_status != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return cublaslt_status_to_error(cublas_status);
  }
  cublas_status = cublasLtMatrixLayoutCreate(&layout_b, Type::data_type, k, n, n);
  if (cublas_status != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return cublaslt_status_to_error(cublas_status);
  }
  cublas_status = cublasLtMatrixLayoutCreate(&layout_c, Type::data_type, m, n, n);
  if (cublas_status != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return cublaslt_status_to_error(cublas_status);
  }
  cublas_status = cublasLtMatrixLayoutCreate(&layout_d, Type::data_type, m, n, n);
  if (cublas_status != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return cublaslt_status_to_error(cublas_status);
  }

  cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
  cublas_status = cublasLtMatrixLayoutSetAttribute(
      layout_a, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
  if (cublas_status != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return cublaslt_status_to_error(cublas_status);
  }
  cublas_status = cublasLtMatrixLayoutSetAttribute(
      layout_b, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
  if (cublas_status != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return cublaslt_status_to_error(cublas_status);
  }
  cublas_status = cublasLtMatrixLayoutSetAttribute(
      layout_c, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
  if (cublas_status != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return cublaslt_status_to_error(cublas_status);
  }
  cublas_status = cublasLtMatrixLayoutSetAttribute(
      layout_d, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
  if (cublas_status != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return cublaslt_status_to_error(cublas_status);
  }

  cublas_status = cublasLtMatmulPreferenceCreate(&preference);
  if (cublas_status != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return cublaslt_status_to_error(cublas_status);
  }
  cublas_status = cublasLtMatmulPreferenceSetAttribute(
      preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &kWorkspaceSize,
      sizeof(kWorkspaceSize));
  if (cublas_status != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return cublaslt_status_to_error(cublas_status);
  }

  cudaError_t cuda_status = cudaMalloc(&workspace, kWorkspaceSize);
  if (cuda_status != cudaSuccess) {
    cleanup();
    return static_cast<int>(cuda_status);
  }

  cublasLtMatmulHeuristicResult_t heuristic_result{};
  int returned_results = 0;
  cublas_status = cublasLtMatmulAlgoGetHeuristic(handle, matmul_desc, layout_a,
                                                 layout_b, layout_c, layout_d,
                                                 preference, 1, &heuristic_result,
                                                 &returned_results);
  if (cublas_status != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return cublaslt_status_to_error(cublas_status);
  }
  if (returned_results == 0) {
    cleanup();
    return cublaslt_status_to_error(CUBLAS_STATUS_NOT_SUPPORTED);
  }

  float alpha = 1.0f;
  float beta = 0.0f;
  auto launch = [&]() {
    return cublasLtMatmul(handle, matmul_desc, &alpha, buffers.a, layout_a,
                          buffers.b, layout_b, &beta, buffers.c, layout_c,
                          buffers.d, layout_d, &heuristic_result.algo, workspace,
                          kWorkspaceSize, stream);
  };

  for (int i = 0; i < warmup_iterations; ++i) {
    cublas_status = launch();
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
      cleanup();
      return cublaslt_status_to_error(cublas_status);
    }
  }
  cuda_status = cudaStreamSynchronize(stream);
  if (cuda_status != cudaSuccess) {
    cleanup();
    return static_cast<int>(cuda_status);
  }

  cuda_status = cudaEventCreate(&start);
  if (cuda_status != cudaSuccess) {
    cleanup();
    return static_cast<int>(cuda_status);
  }
  cuda_status = cudaEventCreate(&stop);
  if (cuda_status != cudaSuccess) {
    cleanup();
    return static_cast<int>(cuda_status);
  }
  cuda_status = cudaEventRecord(start, stream);
  if (cuda_status != cudaSuccess) {
    cleanup();
    return static_cast<int>(cuda_status);
  }

  for (int i = 0; i < iterations; ++i) {
    cublas_status = launch();
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
      cleanup();
      return cublaslt_status_to_error(cublas_status);
    }
  }

  cuda_status = cudaEventRecord(stop, stream);
  if (cuda_status != cudaSuccess) {
    cleanup();
    return static_cast<int>(cuda_status);
  }
  cuda_status = cudaEventSynchronize(stop);
  if (cuda_status != cudaSuccess) {
    cleanup();
    return static_cast<int>(cuda_status);
  }

  float elapsed_ms = 0.0f;
  cuda_status = cudaEventElapsedTime(&elapsed_ms, start, stop);
  if (cuda_status != cudaSuccess) {
    cleanup();
    return static_cast<int>(cuda_status);
  }
  *avg_ms = elapsed_ms / float(iterations);

  cleanup();
  return 0;
}

extern "C" int run_cublaslt_f32(KernelRunnerBuffers buffers, int m, int n, int k,
                                 int warmup_iterations, int iterations,
                                 cudaStream_t *streams, int num_streams,
                                 int split_k_slices, float *avg_ms) {
  return run_cublaslt_gemm<float>(buffers, m, n, k, warmup_iterations,
                                  iterations, streams, num_streams, split_k_slices, avg_ms);
}

extern "C" int run_cublaslt_f16(KernelRunnerBuffers buffers, int m, int n, int k,
                                 int warmup_iterations, int iterations,
                                 cudaStream_t *streams, int num_streams,
                                 int split_k_slices, float *avg_ms) {
  return run_cublaslt_gemm<cutlass::half_t>(buffers, m, n, k, warmup_iterations,
                                            iterations, streams, num_streams, split_k_slices, avg_ms);
}