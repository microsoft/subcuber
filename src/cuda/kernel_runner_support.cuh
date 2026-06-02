#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

#include "cutlass/cutlass.h"
#include "cutlass/kernel_hardware_info.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/numeric_types.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"

struct KernelRunnerBuffers {
  void const *a;
  void const *b;
  void const *c;
  void *d;
};

constexpr int kKernelRunnerMaxStreams = 49;

inline cudaError_t kernel_runner_create_streams(cudaStream_t *streams, int num_streams) {
  for (int i = 0; i < num_streams; ++i) {
    cudaError_t err = cudaStreamCreate(&streams[i]);
    if (err != cudaSuccess) {
      for (int j = 0; j < i; ++j) {
        cudaStreamDestroy(streams[j]);
      }
      return err;
    }
  }
  return cudaSuccess;
}

inline void kernel_runner_destroy_streams(cudaStream_t *streams, int num_streams) {
  for (int i = 0; i < num_streams; ++i) {
    cudaStreamDestroy(streams[i]);
  }
}

inline int kernel_runner_status_to_error(cutlass::Status status) {
  return status == cutlass::Status::kSuccess ? 0 : static_cast<int>(status);
}

template <typename Gemm>
int kernel_runner_run_cutlass2(KernelRunnerBuffers buffers, int m, int n, int k,
                               int warmup_iterations, int iterations,
                               cudaStream_t *streams, int num_streams,
                               int split_k_slices, float *avg_ms) {
  using Kernel = typename Gemm::StrassenGemmKernel;
  using ElementA = typename Kernel::ElementA;
  using ElementB = typename Kernel::ElementB;
  using ElementC = typename Kernel::ElementC;
  using LayoutA = typename Kernel::LayoutA;
  using LayoutB = typename Kernel::LayoutB;
  using LayoutC = typename Kernel::LayoutC;
  using Arguments = typename Gemm::Arguments;

  if (m < 2 * Kernel::ThreadblockShape::kM ||
      n < 2 * Kernel::ThreadblockShape::kN ||
      k < 2 * Kernel::ThreadblockShape::kK) {
    return kernel_runner_status_to_error(cutlass::Status::kErrorInvalidProblem);
  }

  if constexpr (requires { typename Kernel::ChildGemmM0; }) {
    if (m % (4 * Kernel::ThreadblockShape::kM) != 0 ||
        n % (4 * Kernel::ThreadblockShape::kN) != 0 ||
        k % (4 * Kernel::ThreadblockShape::kK) != 0) {
      return kernel_runner_status_to_error(cutlass::Status::kErrorInvalidProblem);
    }
  }

  Arguments args(
      cutlass::gemm::GemmCoord(m, n, k),
      {reinterpret_cast<ElementA const *>(buffers.a), LayoutA::packed({m, k})},
      {reinterpret_cast<ElementB const *>(buffers.b), LayoutB::packed({k, n})},
      {reinterpret_cast<ElementC const *>(buffers.c), LayoutC::packed({m, n})},
      {reinterpret_cast<ElementC *>(buffers.d), LayoutC::packed({m, n})},
      {1.0f, 0.0f},
      split_k_slices);

  cutlass::Status status = Gemm::can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    return kernel_runner_status_to_error(status);
  }

  size_t workspace_size = Gemm::get_workspace_size(args);
  cutlass::device_memory::allocation<unsigned char> workspace(workspace_size);

  num_streams = std::max(1, std::min(num_streams, kKernelRunnerMaxStreams));
  cudaError_t err = cudaSuccess;

  Gemm gemm;
  status = gemm.initialize(args, workspace.get(), streams[0]);
  if (status != cutlass::Status::kSuccess) {
    return kernel_runner_status_to_error(status);
  }

  for (int i = 0; i < warmup_iterations; ++i) {
    status = gemm.run(streams, num_streams);
    if (num_streams > 1) {
      err = cudaDeviceSynchronize();
      if (err != cudaSuccess) {
        return static_cast<int>(err);
      }
    }
    if (status != cutlass::Status::kSuccess) {
      return kernel_runner_status_to_error(status);
    }
  }
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    return static_cast<int>(err);
  }

  cudaEvent_t start, stop;
  err = cudaEventCreate(&start);
  if (err != cudaSuccess) {
    return static_cast<int>(err);
  }
  err = cudaEventCreate(&stop);
  if (err != cudaSuccess) {
    cudaEventDestroy(start);
    return static_cast<int>(err);
  }

  err = cudaEventRecord(start);
  if (err != cudaSuccess) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return static_cast<int>(err);
  }

  for (int i = 0; i < iterations; ++i) {
    status = gemm.run(streams, num_streams);
    if (num_streams > 1) {
      err = cudaDeviceSynchronize();
      if (err != cudaSuccess) {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return static_cast<int>(err);
      }
    }
    if (status != cutlass::Status::kSuccess) {
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
      return kernel_runner_status_to_error(status);
    }
  }

  err = cudaEventRecord(stop);
  if (err != cudaSuccess) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return static_cast<int>(err);
  }
  err = cudaEventSynchronize(stop);
  if (err != cudaSuccess) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return static_cast<int>(err);
  }
  float elapsed_ms = 0.0f;
  err = cudaEventElapsedTime(&elapsed_ms, start, stop);
  if (err != cudaSuccess) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return static_cast<int>(err);
  }
  *avg_ms = elapsed_ms / float(iterations);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return 0;
}

template <typename Gemm>
int kernel_runner_run_gemm_cutlass2(KernelRunnerBuffers buffers, int m, int n, int k,
                                    int warmup_iterations, int iterations,
                                    cudaStream_t *streams, int num_streams,
                                    int split_k_slices, float *avg_ms) {
  using Kernel = typename Gemm::CutlassGemm;
  using ElementA = typename Kernel::ElementA;
  using ElementB = typename Kernel::ElementB;
  using ElementC = typename Kernel::ElementC;
  using LayoutA = typename Kernel::LayoutA;
  using LayoutB = typename Kernel::LayoutB;
  using LayoutC = typename Kernel::LayoutC;
  using Arguments = typename Gemm::Arguments;

  Arguments args(
      cutlass::gemm::GemmCoord(m, n, k),
      {reinterpret_cast<ElementA const *>(buffers.a), LayoutA::packed({m, k})},
      {reinterpret_cast<ElementB const *>(buffers.b), LayoutB::packed({k, n})},
      {reinterpret_cast<ElementC const *>(buffers.c), LayoutC::packed({m, n})},
      {reinterpret_cast<ElementC *>(buffers.d), LayoutC::packed({m, n})},
      {1.0f, 0.0f},
      split_k_slices);

  cutlass::Status status = Gemm::can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    return kernel_runner_status_to_error(status);
  }

  size_t workspace_size = Gemm::get_workspace_size(args);
  cutlass::device_memory::allocation<unsigned char> workspace(workspace_size);

  num_streams = std::max(1, std::min(num_streams, kKernelRunnerMaxStreams));
  cudaError_t err = cudaSuccess;

  Gemm gemm;
  status = gemm.initialize(args, workspace.get(), streams[0]);
  if (status != cutlass::Status::kSuccess) {
    return kernel_runner_status_to_error(status);
  }

  for (int i = 0; i < warmup_iterations; ++i) {
    status = gemm.run(streams, num_streams);
    if (num_streams > 1) {
      err = cudaDeviceSynchronize();
      if (err != cudaSuccess) {
        return static_cast<int>(err);
      }
    }
    if (status != cutlass::Status::kSuccess) {
      return kernel_runner_status_to_error(status);
    }
  }
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    return static_cast<int>(err);
  }

  cudaEvent_t start, stop;
  err = cudaEventCreate(&start);
  if (err != cudaSuccess) {
    return static_cast<int>(err);
  }
  err = cudaEventCreate(&stop);
  if (err != cudaSuccess) {
    cudaEventDestroy(start);
    return static_cast<int>(err);
  }

  err = cudaEventRecord(start);
  if (err != cudaSuccess) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return static_cast<int>(err);
  }

  for (int i = 0; i < iterations; ++i) {
    status = gemm.run(streams, num_streams);
    if (status != cutlass::Status::kSuccess) {
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
      return kernel_runner_status_to_error(status);
    }
  }

  err = cudaEventRecord(stop);
  if (err != cudaSuccess) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return static_cast<int>(err);
  }
  err = cudaEventSynchronize(stop);
  if (err != cudaSuccess) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return static_cast<int>(err);
  }
  float elapsed_ms = 0.0f;
  err = cudaEventElapsedTime(&elapsed_ms, start, stop);
  if (err != cudaSuccess) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return static_cast<int>(err);
  }
  *avg_ms = elapsed_ms / float(iterations);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return 0;
}

template <typename Gemm>
int kernel_runner_run_cutlass3(KernelRunnerBuffers buffers, int m, int n, int k,
                               int warmup_iterations, int iterations,
                               cudaStream_t *streams, int num_streams, int,
                               float *avg_ms) {
  using Kernel = typename Gemm::GemmKernel;
  using ElementA = typename Gemm::StrassenGemmKernel::ElementA;
  using ElementB = typename Gemm::StrassenGemmKernel::ElementB;
  using ElementC = typename Gemm::StrassenGemmKernel::ElementC;
  using StrideA = typename Kernel::StrideA;
  using StrideB = typename Kernel::StrideB;
  using StrideC = typename Kernel::StrideC;
  using StrideD = typename Kernel::StrideD;
  using Arguments = typename Gemm::Arguments;
  using RasterOrderOptions = typename Gemm::RasterOrderOptions;

  if (m < 2 * int(cute::size<0>(typename Kernel::TileShape{})) ||
      n < 2 * int(cute::size<1>(typename Kernel::TileShape{})) ||
      k < 2 * int(cute::size<2>(typename Kernel::TileShape{}))) {
    return kernel_runner_status_to_error(cutlass::Status::kErrorInvalidProblem);
  }

  int device_id = 0;
  cutlass::KernelHardwareInfo hw_info = cutlass::KernelHardwareInfo::make_kernel_hardware_info<Kernel>(device_id);
  StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
  StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, {k, n, 1});
  StrideC stride_c = cutlass::make_cute_packed_stride(StrideC{}, {m, n, 1});
  StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});

  Arguments args(
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k},
      {reinterpret_cast<ElementA const *>(buffers.a), stride_a,
       reinterpret_cast<ElementB const *>(buffers.b), stride_b},
      {{1.0f, 0.0f}, reinterpret_cast<ElementC const *>(buffers.c), stride_c,
       reinterpret_cast<ElementC *>(buffers.d), stride_d},
      hw_info);

  args.scheduler.raster_order = RasterOrderOptions::AlongN;

  cutlass::Status status = Gemm::can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    return kernel_runner_status_to_error(status);
  }

  size_t workspace_size = Gemm::get_workspace_size(args);
  cutlass::device_memory::allocation<unsigned char> workspace(workspace_size);

  num_streams = std::max(1, std::min(num_streams, kKernelRunnerMaxStreams));
  cudaError_t err = cudaSuccess;

  int swizzles[7] = {2, 2, 1, 1, 1, 1, 1};
  Gemm gemm;
  status = gemm.initialize(args, swizzles, workspace.get(), streams[0]);
  if (status != cutlass::Status::kSuccess) {
    return kernel_runner_status_to_error(status);
  }

  for (int i = 0; i < warmup_iterations; ++i) {
    status = gemm.run(streams, num_streams);
    if (num_streams > 1) {
      err = cudaDeviceSynchronize();
      if (err != cudaSuccess) {
        return static_cast<int>(err);
      }
    }
    if (status != cutlass::Status::kSuccess) {
      return kernel_runner_status_to_error(status);
    }
  }
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    return static_cast<int>(err);
  }

  cudaEvent_t start, stop;
  err = cudaEventCreate(&start);
  if (err != cudaSuccess) {
    return static_cast<int>(err);
  }
  err = cudaEventCreate(&stop);
  if (err != cudaSuccess) {
    cudaEventDestroy(start);
    return static_cast<int>(err);
  }

  err = cudaEventRecord(start);
  if (err != cudaSuccess) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return static_cast<int>(err);
  }

  for (int i = 0; i < iterations; ++i) {
    status = gemm.run(streams, num_streams);
    if (num_streams > 1) {
      err = cudaDeviceSynchronize();
      if (err != cudaSuccess) {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return static_cast<int>(err);
      }
    }
    if (status != cutlass::Status::kSuccess) {
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
      return kernel_runner_status_to_error(status);
    }
  }

  err = cudaEventRecord(stop);
  if (err != cudaSuccess) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return static_cast<int>(err);
  }
  err = cudaEventSynchronize(stop);
  if (err != cudaSuccess) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return static_cast<int>(err);
  }
  float elapsed_ms = 0.0f;
  err = cudaEventElapsedTime(&elapsed_ms, start, stop);
  if (err != cudaSuccess) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return static_cast<int>(err);
  }
  *avg_ms = elapsed_ms / float(iterations);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return 0;
}

template <typename Gemm>
int kernel_runner_run_gemm_cutlass3(KernelRunnerBuffers buffers, int m, int n, int k,
                                    int warmup_iterations, int iterations,
                                    cudaStream_t *streams, int num_streams, int,
                                    float *avg_ms) {
  using Adapter = typename Gemm::CutlassGemm;
  using Kernel = typename Adapter::GemmKernel;
  using ElementA = typename Adapter::ElementA;
  using ElementB = typename Adapter::ElementB;
  using ElementC = typename Adapter::ElementC;
  using ElementD = typename Adapter::ElementD;
  using StrideA = typename Kernel::StrideA;
  using StrideB = typename Kernel::StrideB;
  using StrideC = typename Kernel::StrideC;
  using StrideD = typename Kernel::StrideD;
  using Arguments = typename Gemm::Arguments;
  using RasterOrderOptions = typename Gemm::RasterOrderOptions;

  int device_id = 0;
  cutlass::KernelHardwareInfo hw_info = cutlass::KernelHardwareInfo::make_kernel_hardware_info<Kernel>(device_id);
  StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
  StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, {k, n, 1});
  StrideC stride_c = cutlass::make_cute_packed_stride(StrideC{}, {m, n, 1});
  StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});

  Arguments args(
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k},
      {reinterpret_cast<ElementA const *>(buffers.a), stride_a,
       reinterpret_cast<ElementB const *>(buffers.b), stride_b},
      {{1.0f, 0.0f}, reinterpret_cast<ElementC const *>(buffers.c), stride_c,
       reinterpret_cast<ElementD *>(buffers.d), stride_d},
      hw_info);

  args.scheduler.raster_order = RasterOrderOptions::AlongN;
  args.scheduler.max_swizzle_size = 1;

  cutlass::Status status = Gemm::can_implement(args);
  if (status != cutlass::Status::kSuccess) {
    return kernel_runner_status_to_error(status);
  }

  size_t workspace_size = Gemm::get_workspace_size(args);
  cutlass::device_memory::allocation<unsigned char> workspace(workspace_size);

  num_streams = std::max(1, std::min(num_streams, kKernelRunnerMaxStreams));
  cudaError_t err = cudaSuccess;

  Gemm gemm;
  status = gemm.initialize(args, workspace.get(), streams[0]);
  if (status != cutlass::Status::kSuccess) {
    return kernel_runner_status_to_error(status);
  }

  for (int i = 0; i < warmup_iterations; ++i) {
    status = gemm.run(streams, num_streams);
    if (status != cutlass::Status::kSuccess) {
      return kernel_runner_status_to_error(status);
    }
  }
  err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    return static_cast<int>(err);
  }

  cudaEvent_t start, stop;
  err = cudaEventCreate(&start);
  if (err != cudaSuccess) {
    return static_cast<int>(err);
  }
  err = cudaEventCreate(&stop);
  if (err != cudaSuccess) {
    cudaEventDestroy(start);
    return static_cast<int>(err);
  }

  err = cudaEventRecord(start);
  if (err != cudaSuccess) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return static_cast<int>(err);
  }

  for (int i = 0; i < iterations; ++i) {
    status = gemm.run(streams, num_streams);
    if (status != cutlass::Status::kSuccess) {
      cudaEventDestroy(start);
      cudaEventDestroy(stop);
      return kernel_runner_status_to_error(status);
    }
  }

  err = cudaEventRecord(stop);
  if (err != cudaSuccess) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return static_cast<int>(err);
  }
  err = cudaEventSynchronize(stop);
  if (err != cudaSuccess) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return static_cast<int>(err);
  }
  float elapsed_ms = 0.0f;
  err = cudaEventElapsedTime(&elapsed_ms, start, stop);
  if (err != cudaSuccess) {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return static_cast<int>(err);
  }
  *avg_ms = elapsed_ms / float(iterations);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return 0;
}

#define STRASSEN_RUNNER_EXPORT_CUTLASS2(function_name, gemm_type) \
  extern "C" int function_name(KernelRunnerBuffers buffers, int m, int n, int k, \
                               int warmup_iterations, int iterations, \
                               cudaStream_t *streams, int num_streams, \
                               int split_k_slices, float *avg_ms) { \
    return kernel_runner_run_cutlass2<gemm_type>(buffers, m, n, k, warmup_iterations, \
                                                iterations, streams, num_streams, split_k_slices, avg_ms); \
  }

#define STRASSEN_RUNNER_EXPORT_GEMM_CUTLASS2(function_name, gemm_type) \
  extern "C" int function_name(KernelRunnerBuffers buffers, int m, int n, int k, \
                               int warmup_iterations, int iterations, \
                               cudaStream_t *streams, int num_streams, \
                               int split_k_slices, float *avg_ms) { \
    return kernel_runner_run_gemm_cutlass2<gemm_type>(buffers, m, n, k, warmup_iterations, \
                                                     iterations, streams, num_streams, split_k_slices, avg_ms); \
  }

#define STRASSEN_RUNNER_EXPORT_CUTLASS3(function_name, gemm_type) \
  extern "C" int function_name(KernelRunnerBuffers buffers, int m, int n, int k, \
                               int warmup_iterations, int iterations, \
                               cudaStream_t *streams, int num_streams, \
                               int split_k_slices, float *avg_ms) { \
    return kernel_runner_run_cutlass3<gemm_type>(buffers, m, n, k, warmup_iterations, \
                                                iterations, streams, num_streams, split_k_slices, avg_ms); \
  }

#define STRASSEN_RUNNER_EXPORT_GEMM_CUTLASS3(function_name, gemm_type) \
  extern "C" int function_name(KernelRunnerBuffers buffers, int m, int n, int k, \
                               int warmup_iterations, int iterations, \
                               cudaStream_t *streams, int num_streams, \
                               int split_k_slices, float *avg_ms) { \
    return kernel_runner_run_gemm_cutlass3<gemm_type>(buffers, m, n, k, warmup_iterations, \
                                                     iterations, streams, num_streams, split_k_slices, avg_ms); \
  }
