#pragma once

#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"

class AmpereF16Cutlass128x256 {
  // The code section below describes datatype for input, output matrices and computation between
// elements in input matrices.
  using ElementAccumulator = float;                   // <- data type of accumulator
  using ElementComputeEpilogue = float;  // <- data type of epilogue operations
  using ElementInputA = cutlass::half_t;                        // <- data type of elements in input matrix A
  using ElementInputB = cutlass::half_t;                        // <- data type of elements in input matrix B
  using ElementOutput = cutlass::half_t;                        // <- data type of elements in output matrix D

  // The code section below describes matrix layout of input and output matrices. Column Major for
  // Matrix A, Row Major for Matrix B and Row Major for Matrix C
  using LayoutInputA = cutlass::layout::RowMajor;
  using LayoutInputB = cutlass::layout::RowMajor;
  using LayoutOutput = cutlass::layout::RowMajor;

  // This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
  using MMAOp = cutlass::arch::OpClassTensorOp;

  // This code section describes CUDA SM architecture number
  using SmArch = cutlass::arch::Sm80;

  // This code section describes the tile size a thread block will compute
  using ShapeMMAThreadBlock =
  //128x256x32 provides a little better perf
      cutlass::gemm::GemmShape<128, 256, 32>;  // <- threadblock tile M = 128, N = 128, K = 16 
  // This code section describes tile size a warp will compute
  using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>;  // <- warp tile M = 64, N = 64, K = 16
  // This code section describes the size of MMA op
  using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 16>;  // <- MMA Op tile M = 16, N = 8, K = 8

  // This code section describes how threadblocks are scheduled on GPU
  using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;  // <- ??

  // This code section describes the epilogue part of the kernel
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      ElementOutput,                                     // <- data type of output matrix
      128 / cutlass::sizeof_bits<ElementOutput>::value,  // <- the number of elements per vectorized
                                                        // memory access. For a byte, it's 16
                                                        // elements. This becomes the vector width of
                                                        // math instructions in the epilogue too
      ElementAccumulator,                                // <- data type of accumulator
      ElementComputeEpilogue>;  // <- data type for alpha/beta in linear combination function

  // Number of pipelines you want to use
  static constexpr int NumStages = 3;

public:
  using CutlassGemm = cutlass::gemm::device::Gemm<ElementInputA,
                                          LayoutInputA,
                                          ElementInputB,
                                          LayoutInputB,
                                          ElementOutput,
                                          LayoutOutput,
                                          ElementAccumulator,
                                          MMAOp,
                                          SmArch,
                                          ShapeMMAThreadBlock,
                                          ShapeMMAWarp,
                                          ShapeMMAOp,
                                          EpilogueOp,
                                          SwizzleThreadBlock,
                                          NumStages, 8, 8, true>;

  using Arguments = typename CutlassGemm::Arguments;

  static cutlass::Status can_implement(Arguments const &args);

  static size_t get_workspace_size(Arguments const &args);

  cutlass::Status init(Arguments const &args, void *workspace = nullptr, cudaStream_t stream = nullptr);

  cutlass::Status launch(cudaStream_t *streams = nullptr, int num_streams = 0);

  cutlass::Status initialize(Arguments const &args, void *workspace = nullptr, cudaStream_t stream = nullptr);

  cutlass::Status run(cudaStream_t *streams = nullptr, int num_streams = 0);

  cutlass::Status operator()(Arguments const &args, void *workspace = nullptr,
                             cudaStream_t *streams = nullptr, int num_streams = 0);

private:
  CutlassGemm gemm_;
};