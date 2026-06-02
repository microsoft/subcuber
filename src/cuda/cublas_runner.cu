#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "cutlass/numeric_types.h"
#include "cuda/kernel_runner_support.cuh"

template <typename Element>
struct CublasType;

template <>
struct CublasType<float> {
  static constexpr cudaDataType_t data_type = CUDA_R_32F;
  static constexpr cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F_PEDANTIC;
  static constexpr cublasMath_t math_mode = CUBLAS_DEFAULT_MATH;
  static constexpr cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT;
};

template <>
struct CublasType<cutlass::half_t> {
  static constexpr cudaDataType_t data_type = CUDA_R_16F;
  static constexpr cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F_FAST_16F;
  static constexpr cublasMath_t math_mode = CUBLAS_TENSOR_OP_MATH;
  static constexpr cublasGemmAlgo_t algo = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
};

static int cublas_status_to_error(cublasStatus_t status) {
  return status == CUBLAS_STATUS_SUCCESS ? 0 : 10000 + static_cast<int>(status);
}

template <typename Element>
int run_cublas_gemm(KernelRunnerBuffers buffers, int m, int n, int k,
                    int warmup_iterations, int iterations, cudaStream_t *streams,
                    int num_streams, int, float *avg_ms) {
  using Type = CublasType<Element>;

  cublasHandle_t handle = nullptr;
  cudaStream_t stream = streams == nullptr || num_streams == 0 ? nullptr : streams[0];
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;

  auto cleanup = [&]() {
    if (start != nullptr) cudaEventDestroy(start);
    if (stop != nullptr) cudaEventDestroy(stop);
    if (handle != nullptr) cublasDestroy(handle);
  };

  cudaError_t cuda_status = cudaSuccess;
  cublasStatus_t cublas_status = cublasCreate(&handle);
  if (cublas_status != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return cublas_status_to_error(cublas_status);
  }

  cublas_status = cublasSetStream(handle, stream);
  if (cublas_status != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return cublas_status_to_error(cublas_status);
  }

  cublas_status = cublasSetMathMode(handle, Type::math_mode);
  if (cublas_status != CUBLAS_STATUS_SUCCESS) {
    cleanup();
    return cublas_status_to_error(cublas_status);
  }

  float alpha = 1.0f;
  float beta = 0.0f;
  auto launch = [&]() {
    return cublasGemmEx(handle,
                        CUBLAS_OP_N, CUBLAS_OP_N,
                        n, m, k,
                        &alpha,
                        buffers.b, Type::data_type, n,
                        buffers.a, Type::data_type, k,
                        &beta,
                        buffers.d, Type::data_type, n,
                        Type::compute_type,
                        Type::algo);
  };

  for (int i = 0; i < warmup_iterations; ++i) {
    cublas_status = launch();
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
      cleanup();
      return cublas_status_to_error(cublas_status);
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
      return cublas_status_to_error(cublas_status);
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

extern "C" int run_cublas_f32(KernelRunnerBuffers buffers, int m, int n, int k,
                               int warmup_iterations, int iterations,
                               cudaStream_t *streams, int num_streams,
                               int split_k_slices, float *avg_ms) {
  return run_cublas_gemm<float>(buffers, m, n, k, warmup_iterations,
                                iterations, streams, num_streams, split_k_slices, avg_ms);
}

extern "C" int run_cublas_f16(KernelRunnerBuffers buffers, int m, int n, int k,
                               int warmup_iterations, int iterations,
                               cudaStream_t *streams, int num_streams,
                               int split_k_slices, float *avg_ms) {
  return run_cublas_gemm<cutlass::half_t>(buffers, m, n, k, warmup_iterations,
                                          iterations, streams, num_streams, split_k_slices, avg_ms);
}
