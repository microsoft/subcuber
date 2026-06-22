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

class HopperF64Cutlass128x128 {
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    double,
    1,
    double,
    double>;

  using RowMajor = cutlass::layout::RowMajor;
  using ThreadBlockShape = cutlass::gemm::GemmShape<128, 128, 16>;
  using WarpShape = cutlass::gemm::GemmShape<32, 64, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<16, 8, 4>;
  static constexpr bool splitK = SPLIT_K;

public:
  using CutlassGemm = cutlass::gemm::device::Gemm<double,
                                                  RowMajor,
                                                  double,
                                                  RowMajor,
                                                  double,
                                                  RowMajor,
                                                  double,
                                                  cutlass::arch::OpClassTensorOp,
                                                  cutlass::arch::Sm90,
                                                  ThreadBlockShape,
                                                  WarpShape,
                                                  InstructionShape,
                                                  EpilogueOp,
                                                  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
                                                  3, 1, 1, splitK>;

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