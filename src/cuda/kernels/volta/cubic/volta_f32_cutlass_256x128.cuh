#pragma once

#include "cutlass/gemm/device/gemm.h"
#include "cutlass/cutlass.h"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"

#ifndef SPLIT_K
#define SPLIT_K 1
#endif

class VoltaF32Cutlass256x128 {
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    float,                                     // <- data type of output matrix
    1,  // <- This is the number of elements per
                                                       // vectorized memory access. For half
                                                       // precision, it's 8 elements. This becomes
                                                       // the vector width of math instructions in
                                                       // epilogue too
    float,                                // <- data type of accumulator
    float>;  // <- data type for alpha/beta in linear combination function


  using RowMajor = cutlass::layout::RowMajor;
  using ThreadBlockShape = cutlass::gemm::GemmShape<256, 128, 8>;
  using WarpShape = cutlass::gemm::GemmShape<64, 64, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1,1,1>;
  static constexpr bool splitK = SPLIT_K;

public:
  using CutlassGemm = cutlass::gemm::device::Gemm<float,        // Data-type of A matrix
                                                  RowMajor,  // Layout of A matrix
                                                  float,        // Data-type of B matrix
                                                  RowMajor,  // Layout of B matrix
                                                  float,        // Data-type of C matrix
                                                  RowMajor,
                                                  float,
                                                  cutlass::arch::OpClassSimt,
                                                  cutlass::arch::Sm70,
                                                  ThreadBlockShape,
                                                  WarpShape,
                                                  InstructionShape,
                                                  EpilogueOp,
                                                  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
                                                  2,1,1,splitK>; // Layout of C matrix

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