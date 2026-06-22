#pragma once

#include "cutlass/gemm/device/gemm.h"
#include "cutlass/cutlass.h"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"

class AmpereF64Cutlass64x64 {
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    double,
    1,
    double,
    double>;

  using RowMajor = cutlass::layout::RowMajor;
  using ThreadBlockShape = cutlass::gemm::GemmShape<64, 64, 16>;
  using WarpShape = cutlass::gemm::GemmShape<32, 32, 16>;
  using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;

public:
  using CutlassGemm = cutlass::gemm::device::Gemm<double,
                                                  RowMajor,
                                                  double,
                                                  RowMajor,
                                                  double,
                                                  RowMajor,
                                                  double,
                                                  cutlass::arch::OpClassTensorOp,
                                                  cutlass::arch::Sm80,
                                                  ThreadBlockShape,
                                                  WarpShape,
                                                  InstructionShape,
                                                  EpilogueOp,
                                                  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
                                                  4, 1, 1, false>;

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