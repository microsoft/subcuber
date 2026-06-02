/***************************************************************************************************
 * Copyright (c) 2017 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

/*
  This example demonstrates how to call a CUTLASS GEMM kernel and provides a naive reference
  matrix multiply kernel to verify its correctness.

  The CUTLASS Gemm template is instantiated in the function CutlassSgemmNN. This is kernel computes
  the general matrix product (GEMM) using single-precision floating-point arithmetic and assumes
  all matrices have column-major layout.

  The threadblock tile size is chosen as 128x128x8 which offers good performance for large matrices.
  See the CUTLASS Parallel for All blog post for more exposition on the tunable parameters available
  in CUTLASS.

  https://devblogs.nvidia.com/cutlass-linear-algebra-cuda/

  Aside from defining and launching the SGEMM kernel, this example does not use any other components
  or utilities within CUTLASS. Such utilities are demonstrated elsewhere in other examples and are
  prevalent in the CUTLASS unit tests.

  This example has delibrately been kept similar to the basic_gemm example from cutlass-1.3 to
  highlight the minimum amount of differences needed to transition to cutlass-2.0.

  Cutlass-1.3 sgemm: https://github.com/NVIDIA/cutlass/blob/master/examples/00_basic_gemm/basic_gemm.cu
*/

// Standard Library includes
#include <iostream>
#include <sstream>
#include <vector>

// Helper methods to check for errors
#include "helper.h"

//
// CUTLASS includes needed for single-precision GEMM kernel
//

// Defines cutlass::gemm::device::Gemm, the generic Gemm computation template class.
#include "cutlass/gemm/device/strassen_gemm.h"
#include "cutlass/cutlass.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"

#include "cuda/allocate_float_matrices.cuh"
#include "cuda/strassen_reference_l1.cuh"

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// This function defines a CUTLASS GEMM kernel instantiation, constructs its parameters object,
// and launches it on the CUDA device.
//
///////////////////////////////////////////////////////////////////////////////////////////////////

using EpilogueOp = cutlass::epilogue::thread::StrassenLinearCombination<
  float,                                     // <- data type of output matrix
  1,                                         // <- This is the number of elements per
                                             // vectorized memory access. For half
                                             // precision, it's 8 elements. This becomes
                                             // the vector width of math instructions in
                                             // epilogue too
  float,                                // <- data type of accumulator
  float>;  // <- data type for alpha/beta in linear combination function

using InterimEpilogueOp = cutlass::epilogue::thread::StrassenLinearCombination<
  float,                                     // <- data type of output matrix
  4,                                         // <- This is the number of elements per
                                             // vectorized memory access. For half
                                             // precision, it's 8 elements. This becomes
                                             // the vector width of math instructions in
                                             // epilogue too
  float,                                // <- data type of accumulator
  float>;  // <- data type for alpha/beta in linear combination function

using ColumnMajor = cutlass::layout::ColumnMajor;
using RowMajor = cutlass::layout::RowMajor;

#ifndef SUB_GEMM_PARALLEL
  #define SUB_GEMM_PARALLEL 0
#endif

#ifndef SPLIT_K
  #define SPLIT_K 0
#endif

const auto StrassenKind = StrassenType::StrassenWinograd;

#if defined(TILE_SIZE_256)
  const int TileShapeM = 256;
  const int WarpShapeN = 64;
#elif defined(TILE_SIZE_128)
  const int TileShapeM = 128;
  const int WarpShapeN = 32;
#endif

using ThreadBlockShape256 = cutlass::gemm::GemmShape<256, 128, 8>;
using WarpShape64 = cutlass::gemm::GemmShape<64, 64, 8>;
using ThreadBlockShape128 = cutlass::gemm::GemmShape<128,128,8>;
using WarpShape32 = cutlass::gemm::GemmShape<64, 32, 8>;

using namespace MmaStrassen;

constexpr int kStrassenLevel = 1;

#if defined(THREADBLOCK)
  using ThreadBlockShape = cutlass::gemm::GemmShape<128, 128, 8>;
  using WarpShape = cutlass::gemm::GemmShape<64, 32, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1,1,1>;
  const bool splitK = SPLIT_K;
using StrassenGroups = MmaStrassen::StrassenLevel1Groups<StrassenPresum<kStrassenLevel, 0, ThreadBlockShape,
                                                                          AllPresums<>>,
                                                           MmaStrassen::StrassenLevel1MiGroup<kStrassenLevel, 0, ThreadBlockShape, WarpShape, 2,
                                                                                              MmaStrassen::RWMTypes<>, MmaStrassen::RWCTypes<>,
                                                                                              MmaStrassen::AllPresums<>, 0, 0,1,2,3,4,5,6>>;
  using ScheduleStrassenGroups1 = ScheduleStrassenGroups<ParallelMiGroups<true, FusedMiGroup<7, 0>>>;

  using CutlassGemm = cutlass::gemm::device::StrassenGemm<StrassenKind, StrassenGroups, ScheduleStrassenGroups1,
                                                          float,        // Data-type of A matrix
                                                          RowMajor,  // Layout of A matrix
                                                          float,        // Data-type of B matrix
                                                          RowMajor,  // Layout of B matrix
                                                          float,        // Data-type of C matrix
                                                          RowMajor,
                                                          float,
                                                          cutlass::arch::OpClassSimt,
                                                          cutlass::arch::Sm90,
                                                          ThreadBlockShape,
                                                          WarpShape,
                                                          InstructionShape,
                                                          EpilogueOp,
                                                          InterimEpilogueOp,
                                                          cutlass::gemm::threadblock::StrassenGemmIdentityThreadblockSwizzle<8>,
                                                          2, 1, 1, splitK>; // Layout of C matrix
#elif defined(FUSED_IN_SUM)
  using ThreadBlockShape = cutlass::gemm::GemmShape<TileShapeM, 128, 8>;
  using WarpShape = cutlass::gemm::GemmShape<64, WarpShapeN, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1,1,1>;
  const bool splitK = SPLIT_K;
  const bool sub_gemm_parallel = SUB_GEMM_PARALLEL;
  //[m0], [m1], [m5, m4, m3, m2], [m6]
  // using StrassenGroups = MmaStrassen::StrassenLevel1Groups<MmaStrassen::StrassenLevel1SingleMiGroup::Group0,
  //                                                          MmaStrassen::StrassenLevel1SingleMiGroup::Group1,
  //                                                          MmaStrassen::StrassenLevel1MiGroup<false, false, true, true, true, true, false>,
  //                                                          MmaStrassen::StrassenLevel1SingleMiGroup::Group6>;

  //[m0], [m1], [m3, m2], [m5, m4], [m6]

  using StrassenGroups = StrassenLevel1Groups<StrassenPresum<kStrassenLevel, 0, ThreadBlockShape,
                                                              AllPresums<>>,
                                              StrassenLevel1M0Group<kStrassenLevel, 0, ThreadBlockShape, WarpShape, 
                                                                    RWMTypes<KeepAccums>,
                                                                    RWCTypes<CUW<1, LayoutInterim1D, LayoutNone, Expr<Plus<0>>>>,//C1 = M0
                                                                    AllPresums<>>,
                                              StrassenLevel1M1Group<kStrassenLevel, 0, ThreadBlockShape, WarpShape,
                                                                    RWMTypes<ContinueAccums>,
                                                                    RWCTypes<CUW<0, LayoutFinal,   LayoutNone, Expr<Plus<1>>>>,//C0 = C0+M1
                                                                    AllPresums<>>,
                                              StrassenLevel1M2Group<kStrassenLevel, 0, ThreadBlockShape, WarpShape,
                                                                    RWMTypes<>,
                                                                    // RWCTypes<CUW<2, LayoutFinal, LayoutNone, Expr<Plus<2>>>>,
                                                                    RWCTypes<CUW<1, LayoutInterim1D, LayoutNone, Expr<Plus<2>>, Expr<Plus<1, MemGlobal, LayoutInterim1D>>>>,//C1 = C1+M2 ; Reg = C1
                                                                    AllPresums<>>,
                                              StrassenLevel1M3Group<kStrassenLevel, 0, ThreadBlockShape, WarpShape,
                                                                    RWMTypes<>,
                                                                    // RWCTypes<CUW<3, LayoutFinal, LayoutNone, Expr<Plus<3>>>>,
                                                                    RWCTypes<CUW<2, LayoutInterim1D, LayoutNone, Expr<Plus<3>>, Expr<Plus<1, MemGlobal, LayoutInterim1D>>>>,//C2 = C1(Reg)+M3 
                                                                    AllPresums<>>,
                                              StrassenLevel1M4Group<kStrassenLevel, 0, ThreadBlockShape, WarpShape,
                                                                    RWMTypes<>,
                                                                    // RWCTypes<CUW<0, LayoutFinal, LayoutNone, Expr<Plus<4>>>>,
                                                                    RWCTypes<CUW<0, LayoutInterim1D, LayoutNone, Expr<Plus<4>>>, //C1 = C1+M4
                                                                             CUW<3, LayoutFinal,   LayoutNone, Expr<Plus<4>>, Expr<Plus<2, MemGlobal, LayoutInterim1D>>>>,//C3 = C2+M4
                                                                    AllPresums<>>,
                                              StrassenLevel1M5Group<kStrassenLevel, 0, ThreadBlockShape, WarpShape,
                                                                    RWMTypes<>,
                                                                    // RWCTypes<CUW<1, LayoutFinal, LayoutNone, Expr<Plus<5>>>>,
                                                                    RWCTypes<CUW<1, LayoutFinal, LayoutNone, Expr<Plus<5>>, Expr<Plus<1, MemGlobal, LayoutInterim1D>,
                                                                                                                                 Plus<0, MemGlobal, LayoutInterim1D>>>>, //C1 = C1+M5
                                                                    AllPresums<>>,
                                              StrassenLevel1M6Group<kStrassenLevel, 0, ThreadBlockShape, WarpShape,
                                                                    RWMTypes<>,
                                                                    // RWCTypes<CUW<2, LayoutFinal, LayoutNone, Expr<Plus<6>>>>,
                                                                    RWCTypes<CUW<2, LayoutFinal, LayoutNone, Expr<Neg<6>>, Expr<Plus<2, MemGlobal, LayoutInterim1D>>>>, //C2 = C2-M6
                                                                    AllPresums<>>
                                              >;
  using ScheduleStrassenGroups1 = ScheduleStrassenGroups<ParallelMiGroups<true, FusedMiGroup<7, 0, 1>>,
                                                         ParallelMiGroups<true, FusedMiGroup<7, 2>>,
                                                         ParallelMiGroups<true, FusedMiGroup<7, 3>>,
                                                         ParallelMiGroups<true, FusedMiGroup<7, 4>>,
                                                         ParallelMiGroups<true, FusedMiGroup<7, 5>>,
                                                         ParallelMiGroups<true, FusedMiGroup<7, 6>>>;

  using CutlassGemm = cutlass::gemm::device::StrassenGemm<StrassenKind, StrassenGroups, ScheduleStrassenGroups1,
                                                          float,        // Data-type of A matrix
                                                          RowMajor,  // Layout of A matrix
                                                          float,        // Data-type of B matrix
                                                          RowMajor,  // Layout of B matrix
                                                          float,        // Data-type of C matrix
                                                          RowMajor,
                                                          float,
                                                          cutlass::arch::OpClassSimt,
                                                          cutlass::arch::Sm80,
                                                          ThreadBlockShape,
                                                          WarpShape,
                                                          InstructionShape,
                                                          EpilogueOp,
                                                          InterimEpilogueOp,
                                                          cutlass::gemm::threadblock::StrassenGemmIdentityThreadblockSwizzle<8>,
                                                          5,
                                                          1, 1, splitK,
                                                          sub_gemm_parallel>; // Layout of C matrix
#elif defined(PRESUM)
  using ThreadBlockShape = cutlass::gemm::GemmShape<TileShapeM, 128, 8>;
  using WarpShape = cutlass::gemm::GemmShape<64, WarpShapeN, 8>;
  using InstructionShape = cutlass::gemm::GemmShape<1,1,1>;
  const bool splitK = SPLIT_K;
  const bool sub_gemm_parallel = SUB_GEMM_PARALLEL;

  //[m0], [m1], [m2], [m3], [m4], [m5], [m6]
  using AllPresumsKernel = AllPresums<>;
  using AllPresumsM0    = AllPresums<PresumCompute,   PresumCompute,   PresumCompute,   PresumCompute,   PresumCompute,    PresumCompute,  PresumCompute,    PresumCompute>;
  // using AllPresumsM0    = AllPresums<PresumGlobalKernel,   PresumGlobalKernel,   PresumGlobalKernel,   PresumGlobalKernel,   PresumGlobalKernel,    PresumGlobalKernel,  PresumGlobalKernel,    PresumGlobalKernel>;
  using AllPresumsM1To6 = AllPresums<PresumAvailable, PresumAvailable, PresumAvailable, PresumAvailable, PresumAvailable, PresumAvailable, PresumAvailable, PresumAvailable>;

  using StrassenGroups = StrassenLevel1Groups<StrassenPresum<kStrassenLevel, 0, ThreadBlockShape,
                                                              AllPresumsKernel>,
                                              StrassenLevel1M0Group<kStrassenLevel, 0, ThreadBlockShape, WarpShape, 5,
                                                                    RWMTypes<KeepAccums>,
                                                                    RWCTypes<CUW<1, LayoutInterim1D, LayoutNone, Expr<Plus<0>>>>,//C1 = M0
                                                                    AllPresumsM0>,
                                              StrassenLevel1M1Group<kStrassenLevel, 0, ThreadBlockShape, WarpShape, 5,
                                                                    RWMTypes<ContinueAccums>,
                                                                    RWCTypes<CUW<0, LayoutFinal,   LayoutNone, Expr<Plus<1>>>>,//C0 = C0+M1
                                                                    AllPresumsM1To6>,
                                              StrassenLevel1M2Group<kStrassenLevel, 0, ThreadBlockShape, WarpShape, 5,
                                                                    RWMTypes<>,
                                                                    // RWCTypes<CUW<2, LayoutFinal, LayoutNone, Expr<Plus<2>>>>,
                                                                    RWCTypes<CUW<1, LayoutInterim1D, LayoutNone, Expr<Plus<2>>, Expr<Plus<1, MemGlobal, LayoutInterim1D>>>>,//C1 = C1+M2 ; Reg = C1
                                                                    AllPresumsM1To6>,
                                              StrassenLevel1M3Group<kStrassenLevel, 0, ThreadBlockShape, WarpShape, 5,
                                                                    RWMTypes<>,
                                                                    // RWCTypes<CUW<3, LayoutFinal, LayoutNone, Expr<Plus<3>>>>,
                                                                    RWCTypes<CUW<2, LayoutInterim1D, LayoutNone, Expr<Plus<3>>, Expr<Plus<1, MemGlobal, LayoutInterim1D>>>>,//C2 = C1(Reg)+M3 
                                                                    AllPresumsM1To6>,
                                              StrassenLevel1M4Group<kStrassenLevel, 0, ThreadBlockShape, WarpShape, 5,
                                                                    RWMTypes<>,
                                                                    // RWCTypes<CUW<0, LayoutFinal, LayoutNone, Expr<Plus<4>>>>,
                                                                    RWCTypes<CUW<0, LayoutInterim1D, LayoutNone, Expr<Plus<4>>>, //C1 = C1+M4
                                                                             CUW<3, LayoutFinal,   LayoutNone, Expr<Plus<4>>, Expr<Plus<2, MemGlobal, LayoutInterim1D>>>>,//C3 = C2+M4
                                                                    AllPresumsM1To6>,
                                              StrassenLevel1M5Group<kStrassenLevel, 0, ThreadBlockShape, WarpShape, 5,
                                                                    RWMTypes<>,
                                                                    // RWCTypes<CUW<1, LayoutFinal, LayoutNone, Expr<Plus<5>>>>,
                                                                    RWCTypes<CUW<1, LayoutFinal, LayoutNone, Expr<Plus<5>>, Expr<Plus<1, MemGlobal, LayoutInterim1D>,
                                                                                                                                 Plus<0, MemGlobal, LayoutInterim1D>>>>, //C1 = C1+M5
                                                                    AllPresumsM1To6>,
                                              StrassenLevel1M6Group<kStrassenLevel, 0, ThreadBlockShape, WarpShape, 5,
                                                                    RWMTypes<>,
                                                                    // RWCTypes<CUW<2, LayoutFinal, LayoutNone, Expr<Plus<6>>>>,
                                                                    RWCTypes<CUW<2, LayoutFinal, LayoutNone, Expr<Neg<6>>, Expr<Plus<2, MemGlobal, LayoutInterim1D>>>>, //C2 = C2-M6
                                                                    AllPresumsM1To6>
                                              >;
  using ScheduleStrassenGroups1 = ScheduleStrassenGroups<ParallelMiGroups<true, FusedMiGroup<7, 0, 1>>,
                                                         ParallelMiGroups<true, FusedMiGroup<7, 2>>,
                                                         ParallelMiGroups<true, FusedMiGroup<7, 3>>,
                                                         ParallelMiGroups<true, FusedMiGroup<7, 4>>,
                                                         ParallelMiGroups<true, FusedMiGroup<7, 5>>,
                                                         ParallelMiGroups<true, FusedMiGroup<7, 6>>
                                                         >;

  using CutlassGemm = cutlass::gemm::device::StrassenGemm<StrassenKind, StrassenGroups, ScheduleStrassenGroups1,
                                                          float,        // Data-type of A matrix
                                                          RowMajor,  // Layout of A matrix
                                                          float,        // Data-type of B matrix
                                                          RowMajor,  // Layout of B matrix
                                                          float,        // Data-type of C matrix
                                                          RowMajor,
                                                          float,
                                                          cutlass::arch::OpClassSimt,
                                                          cutlass::arch::Sm80,
                                                          ThreadBlockShape,
                                                          WarpShape,
                                                          InstructionShape,
                                                          EpilogueOp,
                                                          InterimEpilogueOp,
                                                          cutlass::gemm::threadblock::StrassenGemmIdentityThreadblockSwizzle<8>,
                                                          3,
                                                          1, 1, splitK,
                                                          sub_gemm_parallel>; // Layout of C matrix
#endif

// Command line options parsing
struct Options {

  bool help;

  cutlass::gemm::GemmCoord problem_size;
  int batch_count;
  float alpha;
  float beta;

  bool reference_check;
  int iterations;
  int split_k_slices;
  int streams;

  int level;

  Options():
    help(false),
    problem_size({5120, 4096, 4096}),
    batch_count(1),
    reference_check(true),
    iterations(20),
    alpha(1),
    beta(),
    split_k_slices(1),
    streams(1), level(1) { }

  bool valid() {
    return true;
  }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
    }

    if (cmd.check_cmd_line_flag("mnk")) {
      int mnk = 0;
      cmd.get_cmd_line_argument("mnk", mnk);
      problem_size = cutlass::gemm::GemmCoord(mnk,mnk,mnk);
    } else {
      cmd.get_cmd_line_argument("m", problem_size.m());
      cmd.get_cmd_line_argument("n", problem_size.n());
      cmd.get_cmd_line_argument("k", problem_size.k());
    }

    cmd.get_cmd_line_argument("alpha", alpha);
    cmd.get_cmd_line_argument("beta", beta);
    
    cmd.get_cmd_line_argument("iterations", iterations);
    cmd.get_cmd_line_argument("split_k_slices", split_k_slices);
    cmd.get_cmd_line_argument("streams", streams);
    cmd.get_cmd_line_argument("check", reference_check);
    cmd.get_cmd_line_argument("level", level);
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "strassen_ampere_f32_gemm example\n\n"
      << "  This example uses the CUTLASS Library to execute F32 tensorop GEMM computations.\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement.\n\n"
      << "  --m=<int>                   GEMM M dimension\n"
      << "  --n=<int>                   GEMM N dimension\n"
      << "  --k=<int>                   GEMM K dimension\n"
      << "  --alpha=<f32>               Epilogue scalar alpha\n"
      << "  --beta=<f32>                Epilogue scalar beta\n\n"
      << "  --iterations=<int>          Number of profiling iterations to perform\n\n"
      << "  --split_k_slices=<int>      Split K Slices.\n\n"
      << "  --streams=<int>             Number of overlapping streams.\n\n"
      << "  --check=<0|1>               Do reference check\n\n"
      << "  --level=<1|2>               Strassen Recursion Level";

    out << "\n\nExamples:\n\n"
      << "$ ./examples/14_ampere_tf32_tensorop_gemm/14_ampere_tf32_tensorop_gemm --m=1024 --n=512 --k=1024 \\\n"
      << "     --alpha=2 --beta=0.707 \n\n";

    return out;
  }

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s) const {

    // Number of real-valued multiply-adds 
    int64_t fmas = problem_size.product() * batch_count;
    
    // Two flops per multiply-add
    return 2.0 * double(fmas) / double(1.0e9) / runtime_s;
  }
};

/// Define a CUTLASS GEMM template and launch a GEMM kernel.
cudaError_t CutlassSgemmNN(
  int M,
  int N,
  int K,
  int level,
  float alpha,
  float const *A,
  int lda,
  float const *B,
  int ldb,
  float beta,
  float *C,
  int ldc,
  cudaStream_t stream[7],
  int num_streams,
  float& elapsedTime,
  int runs = 1,
  int split_k_slices = 1) {

  // Define type definition for single-precision CUTLASS GEMM with column-major
  // input matrices and 128x128x8 threadblock tile size (chosen by default).
  //
  // To keep the interface manageable, several helpers are defined for plausible compositions
  // including the following example for single-precision GEMM. Typical values are used as
  // default template arguments. See `cutlass/gemm/device/default_gemm_configuration.h` for more details.
  //
  // To view the full gemm device API interface, see `cutlass/gemm/device/gemm.h`

  
  // Define a CUTLASS GEMM type
  CutlassGemm gemm_operator;

  // Construct the CUTLASS GEMM arguments object.
  //
  // One of CUTLASS's design patterns is to define gemm argument objects that are constructible
  // in host code and passed to kernels by value. These may include pointers, strides, scalars,
  // and other arguments needed by Gemm and its components.
  //
  // The benefits of this pattern are (1.) a structured, composable strategy for passing host-constructible
  // arguments to kernels and (2.) minimized initialization overhead on kernel entry.
  //

  CutlassGemm::Arguments args({M , N, K},  // Gemm Problem dimensions
                              {A, lda},    // Tensor-ref for source matrix A
                              {B, ldb},    // Tensor-ref for source matrix B
#if defined(STRASSEN_GLOBAL_LEVEL1) || defined(MIN_LDS_NO_PRESUM) || defined(MIN_LDS_ONE_PRESUM) || defined(MIN_LDS_TWO_PRESUM) || defined(MIN_LDS_THREE_PRESUM) || defined(MIN_LDS_FOUR_PRESUM) || defined(MIN_LDS_FOUR_PRESUM_FUSED)
                              {C, ldc},    // Tensor-ref for source matrix C
#else
                              {nullptr, ldc},
#endif
                              {C, ldc},    // Tensor-ref for destination matrix D (may be different memory than source C matrix)
#if defined(STRASSEN_GLOBAL_LEVEL1)
                              {1.0f, 1.0f},//((StrassenKind == StrassenType::Normal) ? 0.0f : 0.0f)},
#else
                              {1.0f, 0.0f},
#endif
                              split_k_slices); // Scalars used in the Epilogue
  size_t workspace_size = CutlassGemm::get_workspace_size(args);

  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
  cutlass::Status status = gemm_operator.initialize(args, workspace.get());// stream[1]);

  cudaDeviceSynchronize();
  //
  // Launch the CUTLASS GEMM kernel.
  //
  cudaEvent_t start;
  cudaEvent_t end;

  cudaEventCreate(&start);
  cudaEventCreate(&end);
  cudaEventRecord(start);

  for (int r = 0; r < runs; r++) {
    cutlass::Status status = gemm_operator.run(stream, num_streams);
    if (num_streams > 1) cudaDeviceSynchronize();
    //
    // Return a cudaError_t if the CUTLASS GEMM operator returned an error code.
    //

    if (status != cutlass::Status::kSuccess) {
      return cudaErrorUnknown;
    }
    // if (StrassenKind != StrassenType::Normal and split_k_slices > 1) {
    //   //TODO: Have to do this in GlobalStrassen with split k
    //   cudaError_t err = cudaDeviceSynchronize();
    //   if (err != cudaSuccess) {
    //     printf("%s\n", cudaGetErrorString(err));
    //     return err;
    //   }
    // }
  }

  cudaEventRecord(end);
  cudaEventSynchronize(end);
  cudaEventElapsedTime(&elapsedTime, start, end);
  elapsedTime = elapsedTime/runs;

  if (false) {
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      printf("%s\n", cudaGetErrorString(err));
      return err;
    }
    float* hC = new float[M * K];
    cudaMemcpy(hC, C, sizeof(float) * M * K, cudaMemcpyDeviceToHost);
    for (int i = 0; i < M * K; i++) {
      uint r = i / 128;
      uint c = i % 128;
      if (r < 32 && c < 32)
        printf("i %d %f\n", i, hC[i]);
    }
  }

  // Return success, if no errors were encountered.
  return cudaSuccess;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Allocate several matrices in GPU device memory and call a single-precision
/// CUTLASS GEMM kernel.
cudaError_t TestCutlassGemm(int M, int N, int K, int level, float alpha, float beta, int split_k_slices, int runs, bool check, int num_streams) {
  cudaError_t result;

  //
  // Define several matrices to be used as operands to GEMM kernels.
  //

  // Compute leading dimensions for each matrix.
  int lda = K;
  int ldb = N;
  int ldc = N;

  // Compute size in bytes of the C matrix.
  size_t sizeof_C = sizeof(float) * M * ldc;

  // Define pointers to matrices in GPU device memory.
  float *A;
  float *B;
  float *C_cutlass;
  float *C_reference;

  //
  // Allocate matrices in GPU device memory with arbitrary seeds.
  //

  result = AllocateMatrix<random_float<>>(&A, M, K, 0);
  if (false) {
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
      printf("%s\n", cudaGetErrorString(err));
      return err;
    }
    float* hA = new float[M * K];
    cudaMemcpy(hA, A, sizeof(float) * M * K, cudaMemcpyDeviceToHost);
    for (int i = 0; i < M * K; i++) {
      uint r = i / 128;
      uint c = i % 128;
      // if (r < 32 && c < 32)
        printf("i %d %f\n", i, hA[i]);
    }
  }
  if (result !=  cudaSuccess) {
    return result;
  }

  result = AllocateMatrix<random_float<>>(&B, K, N, 17);

  if (result !=  cudaSuccess) {
    cudaFree(B);
    return result;
  }
  
  // result = InitializeMatrixB(B, K, N);
  result = cudaDeviceSynchronize();
  if (result != cudaSuccess) {
    std::cerr << "Error: "
      << cudaGetErrorString(result) << std::endl;
    return result;
  }

  if (false) {
    float* hA = new float[M*K];
    float* hB = new float[K*N];

    cudaMemcpy(hA, A, sizeof(float) * M * K, cudaMemcpyDeviceToHost);
    cudaMemcpy(hB, B, sizeof(float) * N * K, cudaMemcpyDeviceToHost);

    float m1 = 0, m3 = 0;

    for (int k = 0; k < K/2; k++) {
      float a2 = hA[64*K + k];
      float a3 = hA[64*K + K/2 + k];
      float b0 = hB[k*N];
      float b2 = hB[(K/2+k)*N];
      printf("442: %d %f+%f = %f\n", k, a2, a3, (a2+a3));
      // printf("442: %d %f\n", k, b0);
      m1 += (a2+a3)*b0;
      m3 += a3*(b2 - b0);
    }

    printf("445: %f + %f = %f\n", m1, m3, m1+m3);
  }

  result = AllocateMatrix<zero>(&C_cutlass, M, N, 101);

  if (result != cudaSuccess) {
    cudaFree(A);
    cudaFree(B);
    return result;
  }

  printf("A %p B %p C %p sizeof_C %ld\n", A, B, C_cutlass, sizeof_C);

  result = AllocateMatrix<zero>(&C_reference, M, N, 101);

  if (result != cudaSuccess) {
    cudaFree(A);
    cudaFree(B);
    cudaFree(C_cutlass);
    return result;
  }
  printf("C_ref %p C_cut %p\n", C_reference, C_cutlass);
  result = cudaMemcpy(C_reference, C_cutlass, sizeof_C, cudaMemcpyDeviceToDevice);

  if (result != cudaSuccess) {
    std::cerr << "Failed to copy C_cutlass matrix to C_reference: "
      << cudaGetErrorString(result) << std::endl;

    cudaFree(C_reference);
    cudaFree(C_cutlass);
    cudaFree(B);
    cudaFree(A);

    return result;
  }

  //
  // Launch CUTLASS GEMM.
  //

  cudaStream_t streams[7];

  cudaStreamCreate(&streams[0]);
  cudaStreamCreate(&streams[1]);
  cudaStreamCreate(&streams[2]);
  cudaStreamCreate(&streams[3]);
  cudaStreamCreate(&streams[4]);
  cudaStreamCreate(&streams[5]);
  cudaStreamCreate(&streams[6]);
  float elapsedTime = 0;

  result = CutlassSgemmNN(M, N, K, level, alpha, A, lda, B, ldb, beta, C_cutlass, ldc, streams, num_streams, elapsedTime, 1, split_k_slices);
  result = cudaDeviceSynchronize();
  // printf("executed\n");
  if (result != cudaSuccess) {
    std::cerr << "CUTLASS GEMM kernel failed: "
      << cudaGetErrorString(result) << std::endl;

    cudaFree(C_reference);
    cudaFree(C_cutlass);
    cudaFree(B);
    cudaFree(A);

    return result;
  }

  //
  // Verify.
  //
  if (check) {
    std::vector<float> host_reference(M * ldc, 0);
    // Launch reference GEMM
    result = ReferenceStrassenWinogradGemm<
      float,
      float,
      float,
      float,
      float>(M, N, K, level, alpha, A, lda, B, ldb, beta, C_reference, ldc);
    printf("reference\n");
    if (result != cudaSuccess) {
      std::cerr << "Reference GEMM kernel failed: "
        << cudaGetErrorString(result) << std::endl;

      cudaFree(C_reference);
      cudaFree(C_cutlass);
      cudaFree(B);
      cudaFree(A);

      return result;
    }

    // Copy to host and verify equivalence.
    std::vector<float> host_cutlass(M * ldc, 0);

    result = cudaMemcpy(host_cutlass.data(), C_cutlass, sizeof_C, cudaMemcpyDeviceToHost);

    if (result != cudaSuccess) {
      std::cerr << "Failed to copy CUTLASS GEMM results: "
        << cudaGetErrorString(result) << std::endl;

      cudaFree(C_reference);
      cudaFree(C_cutlass);
      cudaFree(B);
      cudaFree(A);

      return result;
    }

    result = cudaMemcpy(host_reference.data(), C_reference, sizeof_C, cudaMemcpyDeviceToHost);

    if (result != cudaSuccess) {
      std::cerr << "Failed to copy Reference GEMM results: "
        << cudaGetErrorString(result) << std::endl;

      cudaFree(C_reference);
      cudaFree(C_cutlass);
      cudaFree(B);
      cudaFree(A);

      return result;
    }

    //
    // Test for bit equivalence of results.
    //
    {
      bool eq = true;
      for (int i = 0; i < host_cutlass.size(); i++) {
        float c = host_cutlass[i];
        float r = host_reference[i];

        if ((r == 0 && c == 0) || 
            (isnan(r) && isnan(c))) {
        } else if (r != 0 && c == 0) {
          eq = false;
        } else if (r == 0 && c != 0) {
          eq = false;
        } else if (isnan(r) && !isnan(c)) {
          eq = false;
        } else if (!isnan(r) && isnan(c)) {
          eq = false;
        } else if (abs(r-c)/abs(r + 1e-6) > 1e-4) {
          eq = false;
        }
        if (eq == false) {
          printf("%d, %d : %f != %f\n", i/N, i%N, c, r);
          std::cerr << "CUTLASS results incorrect." << std::endl;
          break;
        }
      }

      if (eq) printf("Passed\n");
      else {
        printf("Failed\n");
        return cudaErrorUnknown;
      }
    }
  }
  
  //warmup
  result = CutlassSgemmNN(M, N, K, level, alpha, A, lda, B, ldb, beta, C_cutlass, ldc, streams, num_streams, elapsedTime, 10, split_k_slices);
  cudaDeviceSynchronize();
  elapsedTime = 0;
  result = CutlassSgemmNN(M, N, K, level, alpha, A, lda, B, ldb, beta, C_cutlass, ldc, streams, num_streams, elapsedTime, runs, split_k_slices);
  
  std::cout << "Time elapsed " << (elapsedTime) << " ms" << std::endl;
  std::cout << "GFLOPS " << ((2L*((long)M)*((long)N)*K)/(elapsedTime/1e3))/1e9 << std::endl;
  //
  // Free device memory allocations.
  //

  cudaFree(C_reference);
  cudaFree(C_cutlass);
  cudaFree(B);
  cudaFree(A);

  return cudaSuccess;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Entry point to basic_gemm example.
//
// usage:
//
//   00_basic_gemm <M> <N> <K> <alpha> <beta>
//
int main(int argc, const char *argv[]) {

  //
  // Parse the command line to obtain GEMM dimensions and scalar values.
  //

  cudaDeviceProp props;

  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (error != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
    return -1;
  }

  bool notSupported = false;
  if (!((props.major * 10 + props.minor) >= 80)) {
    std::cerr << "Ampere Tensor Core operations must be run on a machine with compute capability at least 80."
              << std::endl;
    notSupported = true;
  }

  if (notSupported) {
    // Returning zero so this test passes on older Toolkits. Its actions are no-op.
    return 0;
  }

  Options options;
  options.parse(argc, argv);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  printf("%d x %d x %d F32 tensor op Matrix Multiply\n", \
    options.problem_size.m(), options.problem_size.n(), options.problem_size.k());

  if (!options.valid()) {
    std::cerr << "Invalid problem." << std::endl;
    return -1;
  }

  //
  // Run the CUTLASS GEMM test.
  //

  cudaError_t result = TestCutlassGemm(
    options.problem_size.m(),     // GEMM M dimension
    options.problem_size.n(),     // GEMM N dimension
    options.problem_size.k(),     // GEMM K dimension
    options.level,
    options.alpha,     // alpha
    options.beta,      // beta
    options.split_k_slices,
    options.iterations,
    options.reference_check,
    options.streams
  );

  // Exit.
  return result == cudaSuccess ? 0 : -1;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
