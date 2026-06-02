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

/**
Please check example 07 and 08 for the basics of tensor op gemm kernels.  On NVIDIA Ampere
architecture, most concept still holds.  The two main differences are

1. NVIDIA Ampere architecture introduces a new series of tensor core instructions (see 
   include/cutlass/arch/mma_sm80.h) which are more efficient on Ampere.

2. NVIDIA Ampere architecture uses cp_async() to build multistage software pipeline to better hide
   latency (see include/cutlass/gemm/threadblock/mma_multistage.h)

Moreover, NVIDIA Ampere architecture starts supporting tfloat32 (see include/cutlass/tfloat32.h)
data types in tensor cores.  One big advantage is that we can load in fp32 data and convert them
implicitly to tf32 inside the GEMM kernel which means no change is needed to accelerate traditional
fp32 data by using NVIDIA Ampere architecture.
*/

#include <cmath>
#include <iostream>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/strassen_gemm.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"

#include "cuda/strassen_reference_l1.cuh"
#include "helper.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Result structure
struct Result {

  double runtime_ms;
  double gflops;
  cutlass::Status status;
  cudaError_t error;
  bool passed;

  //
  // Methods
  //

  Result(
    double runtime_ms = 0,
    double gflops = 0,
    cutlass::Status status = cutlass::Status::kSuccess,
    cudaError_t error = cudaSuccess
  ):
    runtime_ms(runtime_ms), gflops(gflops), status(status), error(error), passed(true) { }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

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

    out << "14_ampere_tf32_tensorop_gemm example\n\n"
      << "  This example uses the CUTLASS Library to execute TF32 tensorop GEMM computations.\n\n"
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
      << "  --check=<0|1>               Check results or not.\n\n"
      << "  --level=<1|2>               Strassen Level 1 or 2";

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

///////////////////////////////////////////////////////////////////////////////////////////////////

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
    cutlass::gemm::GemmShape<128, 256, 32>;  // <- threadblock tile M = 128, N = 128, K = 16
// This code section describes tile size a warp will compute
using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>;  // <- warp tile M = 64, N = 64, K = 16
// This code section describes the size of MMA op
using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 16>;  // <- MMA Op tile M = 16, N = 8, K = 8

constexpr int kStrassenLevel = 1;

// This code section describes how threadblocks are scheduled on GPU
//1 for 5k x 5k x 5k and 8 for others
using SwizzleThreadBlock = cutlass::gemm::threadblock::StrassenGemmIdentityThreadblockSwizzle<8>;  // <- ??

// This code section describes the epilogue part of the kernel
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,                                     // <- data type of output matrix
    128 / cutlass::sizeof_bits<ElementOutput>::value,  // <- the number of elements per vectorized
                                                       // memory access. For a byte, it's 16
                                                       // elements. This becomes the vector width of
                                                       // math instructions in the epilogue too
    ElementAccumulator,                                // <- data type of accumulator
    ElementComputeEpilogue>;  // <- data type for alpha/beta in linear combination function

using InterimEpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,                                     // <- data type of output matrix
    128 / cutlass::sizeof_bits<ElementOutput>::value,  // <- the number of elements per vectorized
                                                       // memory access. For a byte, it's 16
                                                       // elements. This becomes the vector width of
                                                       // math instructions in the epilogue too
    ElementAccumulator,                                // <- data type of accumulator
    ElementComputeEpilogue>;  // <- data type for alpha/beta in linear combination function

// Number of pipelines you want to use
constexpr int NumStages = 4; //TODO: Try with NumStages=3

using AllPresumsKernel = AllPresums<>;
                          //  AllPresums<PresumGlobalKernel,   PresumGlobalKernel,  PresumGlobalKernel,   PresumGlobalKernel,  //A Presums
                                        // PresumGlobalKernel,   PresumGlobalKernel,  PresumGlobalKernel,   PresumGlobalKernel>; //B Presums
using AllPresumsM0    = AllPresums<PresumCompute, PresumCompute, PresumCompute, PresumCompute,
                                    PresumCompute, PresumCompute, PresumCompute, PresumCompute>;
using AllPresumsM1To6 = AllPresums<PresumAvailable, PresumAvailable, PresumAvailable, PresumAvailable,
                                    PresumAvailable, PresumAvailable, PresumAvailable, PresumAvailable>;

#if defined(STRASSEN_FOUR_PRESUM_LEVEL1)
#if 1
  using StrassenGroups = StrassenLevel1Groups<StrassenPresum<kStrassenLevel, 0, ShapeMMAThreadBlock,
                                                             AllPresumsKernel>,
                                              StrassenLevel1MiGroup<kStrassenLevel, 0, ShapeMMAThreadBlock, ShapeMMAWarp, 3,
                                                                    RWMTypes<>,
                                                                    RWCTypes<//CUW<1, LayoutInterim, LayoutNone, Expr<Plus<0>>>,//C1 = M0
                                                                             CUW<0, LayoutFinal, LayoutNone, Expr<Plus<1>>>>,//C0 = M1
                                                                    AllPresumsM0, 1, 0, 1>,
                                              StrassenLevel1M2Group<kStrassenLevel, 0, ShapeMMAThreadBlock, ShapeMMAWarp, 3,
                                                                    RWMTypes<>,
                                                                    RWCTypes<CUW<1, LayoutInterim1D, LayoutInterim1D, Expr<Plus<2>>, Expr<Plus<1, MemGlobal, LayoutInterim1D>>>>,//C1 = C1+M2 ; Reg = C1 //TODO: pass C1 through registers
                                                                    AllPresumsM1To6>,
                                              StrassenLevel1M3Group<kStrassenLevel, 0, ShapeMMAThreadBlock, ShapeMMAWarp, 3,
                                                                    RWMTypes<>,
                                                                    RWCTypes<CUW<2, LayoutNone, LayoutInterim1D, Expr<Plus<3>>, Expr<Plus<1, MemShared, LayoutInterim1D>>>>,//C2 = C1(Reg)+M3 
                                                                    AllPresumsM1To6>,
                                              StrassenLevel1M4Group<kStrassenLevel, 0, ShapeMMAThreadBlock, ShapeMMAWarp, 3,
                                                                    RWMTypes<>,
                                                                    RWCTypes<CUW<0, LayoutInterim1D, LayoutNone, Expr<Plus<4>>>, //C1 (stored at M0) = C1+M4
                                                                             CUW<3, LayoutFinal,   LayoutNone, Expr<Plus<4>>, Expr<Plus<2, MemShared, LayoutInterim1D>>>>,//C3 = C2+M4
                                                                    AllPresumsM1To6>,
                                              StrassenLevel1M5Group<kStrassenLevel, 0, ShapeMMAThreadBlock, ShapeMMAWarp, 3,
                                                                    RWMTypes<>,
                                                                    RWCTypes<CUW<1, LayoutFinal, LayoutNone, Expr<Plus<5>>, Expr<Plus<0, MemGlobal, LayoutInterim1D>,
                                                                                                                                 Plus<1, MemGlobal, LayoutInterim1D>>>>, //C1 = C1+M5 //TODO: in code M5 reads M1 and M0 (written by M4)
                                                                    AllPresumsM1To6>,
                                              StrassenLevel1M6Group<kStrassenLevel, 0, ShapeMMAThreadBlock, ShapeMMAWarp, 3,
                                                                    RWMTypes<>,
                                                                    RWCTypes<CUW<2, LayoutFinal, LayoutNone, Expr<Neg<6>>, Expr<Plus<2, MemShared, LayoutInterim1D>>>>, //C2 = C2-M6
                                                                    AllPresumsM1To6>
                                              >;
  using ScheduleStrassenGroups1 = ScheduleStrassenGroups<ParallelMiGroups<true, FusedMiGroup<7, 0>>,
                                                         ParallelMiGroups<true, FusedMiGroup<7, 1, 2, 5, 3>,
                                                        //  ParallelMiGroups<FusedMiGroup<7, 2>>,
                                                                                FusedMiGroup<7, 4>>
                                                        //  ParallelMiGroups<true, FusedMiGroup<7, 4>>
                                                        //  ParallelMiGroups<FusedMiGroup<7, 5>>
                                                         >;
#elif 1
  using StrassenGroups = StrassenLevel1Groups<StrassenPresum<kStrassenLevel, 0, ThreadBlockShape,
                                                             AllPresumsKernel>,
                                              StrassenLevel1M0Group<kStrassenLevel, ShapeMMAThreadBlock, ShapeMMAWarp, 3,
                                                                    RWMTypes<>,
                                                                    RWCTypes<CUW<1, LayoutInterim1D, LayoutInterim1D, Expr<Plus<0>>>>,//C1 = M0
                                                                            //  CUW<0, LayoutFinal, LayoutNone, Expr<Plus<1>>>>,//C0 = M1
                                                                    AllPresumsM0>,
                                              StrassenLevel1M1Group<kStrassenLevel, ShapeMMAThreadBlock, ShapeMMAWarp, 3,
                                                                    RWMTypes<>,
                                                                    RWCTypes<//CUW<1, LayoutInterim, LayoutNone, Expr<Plus<0>>>,//C1 = M0
                                                                             CUW<0, LayoutFinal, LayoutNone, Expr<Plus<1>>, Expr<Plus<1, MemShared, LayoutInterim1D>> >>,//C0 = M1
                                                                    AllPresumsM1To6>,
                                              StrassenLevel1M2Group<kStrassenLevel, ShapeMMAThreadBlock, ShapeMMAWarp, 3,
                                                                    RWMTypes<>,
                                                                    RWCTypes<CUW<1, LayoutInterim1D, LayoutInterim1D, Expr<Plus<2>>, Expr<Plus<1, MemGlobal, LayoutInterim1D>>>>,//C1 = C1+M2 ; Reg = C1 //TODO: pass C1 through registers
                                                                    AllPresumsM1To6>,
                                              StrassenLevel1M3Group<kStrassenLevel, ShapeMMAThreadBlock, ShapeMMAWarp, 3,
                                                                    RWMTypes<>,
                                                                    RWCTypes<CUW<2, LayoutNone, LayoutInterim1D, Expr<Plus<3>>, Expr<Plus<1, MemShared, LayoutInterim1D>>>>,//C2 = C1(Reg)+M3 
                                                                    AllPresumsM1To6>,
                                              StrassenLevel1M4Group<kStrassenLevel, ShapeMMAThreadBlock, ShapeMMAWarp, 3,
                                                                    RWMTypes<>,
                                                                    RWCTypes<CUW<0, LayoutInterim1D, LayoutNone, Expr<Plus<4>>>, //C1 (stored at M0) = C1+M4
                                                                             CUW<3, LayoutFinal,   LayoutNone, Expr<Plus<4>>, Expr<Plus<2, MemShared, LayoutInterim1D>>>>,//C3 = C2+M4
                                                                    AllPresumsM1To6>,
                                              StrassenLevel1M5Group<kStrassenLevel, ShapeMMAThreadBlock, ShapeMMAWarp, 3,
                                                                    RWMTypes<>,
                                                                    RWCTypes<CUW<1, LayoutFinal, LayoutNone, Expr<Plus<5>>, Expr<Plus<1, MemGlobal, LayoutInterim1D>,
                                                                                                                                 Plus<0, MemGlobal, LayoutInterim1D>>>>, //C1 = C1+M5 //TODO: in code M5 reads M1 and M0 (written by M4)
                                                                    AllPresumsM1To6>,
                                              StrassenLevel1M6Group<kStrassenLevel, ShapeMMAThreadBlock, ShapeMMAWarp, 3,
                                                                    RWMTypes<>,
                                                                    RWCTypes<CUW<2, LayoutFinal, LayoutNone, Expr<Neg<6>>, Expr<Plus<2, MemShared, LayoutInterim1D>>>>, //C2 = C2-M6
                                                                    AllPresumsM1To6>
                                              >;
  using ScheduleStrassenGroups1 = ScheduleStrassenGroups<ParallelMiGroups<true, FusedMiGroup<7, 0, 1>>,
                                                         ParallelMiGroups<true, FusedMiGroup<7, 2, 3, 6, 4>,
                                                        //  ParallelMiGroups<FusedMiGroup<7, 2>>,
                                                                           FusedMiGroup<7, 5>>
                                                        //  ParallelMiGroups<FusedMiGroup<7, 4>>,
                                                        //  ParallelMiGroups<FusedMiGroup<7, 5>>
                                                         >;
#else
  //For 4k, 4k, 4k acheives 0.55 ms 
  using StrassenGroups = StrassenLevel1Groups<StrassenPresum<kStrassenLevel, 0, ThreadBlockShape,
                                                             AllPresumsKernel>,
                                              StrassenLevel1MiGroup<kStrassenLevel, 0, ShapeMMAThreadBlock, ShapeMMAWarp, 3,
                                                                    RWMTypes<>,
                                                                    RWCTypes<//CUW<1, LayoutInterim, LayoutNone, Expr<Plus<0>>>,//C1 = M0
                                                                             CUW<0, LayoutFinal, LayoutNone, Expr<Plus<1>>>>,//C0 = M1
                                                                    AllPresumsM0, 1, 0, 1>,
                                              StrassenLevel1M2Group<kStrassenLevel, ShapeMMAThreadBlock, ShapeMMAWarp, 3,
                                                                    RWMTypes<>,
                                                                    RWCTypes<CUW<1, LayoutInterim1D, LayoutInterim1D, Expr<Plus<2>>, Expr<Plus<1, MemGlobal, LayoutInterim1D>>>>,//C1 = C1+M2 ; Reg = C1 //TODO: pass C1 through registers
                                                                    AllPresumsM1To6>,
                                              StrassenLevel1M3Group<kStrassenLevel, ShapeMMAThreadBlock, ShapeMMAWarp, 3,
                                                                    RWMTypes<>,
                                                                    RWCTypes<CUW<2, LayoutInterim1D, LayoutInterim1D, Expr<Plus<3>>, Expr<Plus<1, MemShared, LayoutInterim1D>>>>,//C2 = C1(Reg)+M3 
                                                                    AllPresumsM1To6>,
                                              StrassenLevel1M4Group<kStrassenLevel, ShapeMMAThreadBlock, ShapeMMAWarp, 3,
                                                                    RWMTypes<>,
                                                                    RWCTypes<CUW<0, LayoutInterim1D, LayoutNone, Expr<Plus<4>>>, //C1 (stored at M0) = C1+M4
                                                                             CUW<3, LayoutFinal, LayoutNone, Expr<Plus<4>>, Expr<Plus<2, MemGlobal, LayoutInterim1D>>>>,//C3 = C2+M4
                                                                    AllPresumsM1To6>,
                                              StrassenLevel1M5Group<kStrassenLevel, ShapeMMAThreadBlock, ShapeMMAWarp, 3,
                                                                    RWMTypes<>,
                                                                    RWCTypes<CUW<1, LayoutFinal, LayoutNone, Expr<Plus<5>>, Expr<Plus<1, MemGlobal, LayoutInterim1D>,
                                                                                                                                 Plus<0, MemGlobal, LayoutInterim1D>>>>, //C1 = C1+M5 //TODO: in code M5 reads M1 and M0 (written by M4)
                                                                    AllPresumsM1To6>,
                                              StrassenLevel1M6Group<kStrassenLevel, ShapeMMAThreadBlock, ShapeMMAWarp, 3,
                                                                    RWMTypes<>,
                                                                    RWCTypes<CUW<2, LayoutFinal, LayoutNone, Expr<Neg<6>>, Expr<Plus<2, MemShared, LayoutInterim1D>>>>, //C2 = C2-M6
                                                                    AllPresumsM1To6>
                                              >;
  using ScheduleStrassenGroups1 = ScheduleStrassenGroups<ParallelMiGroups<true, FusedMiGroup<7, 0>>,
                                                         ParallelMiGroups<true, FusedMiGroup<7, 1, 2, 5>,
                                                                                FusedMiGroup<7, 3>,
                                                                                FusedMiGroup<7, 4>>
                                                        //  ParallelMiGroups<FusedMiGroup<7, 2>>,
                                                        //  ParallelMiGroups<FusedMiGroup<7, 4>>,
                                                        //  ParallelMiGroups<FusedMiGroup<7, 5>>
                                                         >;
#endif
  const auto StrassenKind = StrassenType::StrassenWinograd;
  using Gemm = cutlass::gemm::device::StrassenGemm<StrassenKind, StrassenGroups, ScheduleStrassenGroups1,
                                                   ElementInputA,
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
                                                   InterimEpilogueOp,
                                                   SwizzleThreadBlock,
                                                   NumStages, 8, 8,
                                                   false, true>;
#endif


int run(Options &options) {

  // Create a tuple of problem size for matrix multiplication
  cutlass::gemm::GemmCoord problem_size = options.problem_size;
  // bool allP2PAccess = true;
  // cudaDeviceEnablePeerAccess(0, 1);
  // cudaDeviceEnablePeerAccess(1, 0);
  // cudaSetDevice(1);
  // Initialize tensors using CUTLASS helper functions
  cutlass::HostTensor<ElementInputA, LayoutInputA> tensor_a(
      problem_size.mk());  // <- Create matrix A with dimensions M x K
  cutlass::HostTensor<ElementInputB, LayoutInputB> tensor_b(
      problem_size.kn());  // <- Create matrix B with dimensions K x N
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_c(
      problem_size.mn());  // <- Create matrix C with dimensions M x N
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_d(
      problem_size.mn());  // <- Create matrix D with dimensions M x N used to store output from
                           // CUTLASS kernel
  cutlass::HostTensor<ElementOutput, LayoutOutput> tensor_ref_d(
      problem_size.mn());  // <- Create matrix D with dimensions M x N used to store output from
                           // reference kernel

  // Fill input and output matrices on host using CUTLASS helper functions
  if (true) {
    cutlass::reference::host::TensorFillRandomUniform(
         tensor_a.host_view(),
         1,
         ElementInputA(4),
         ElementInputA(-4),
         0);  // <- Fill matrix A on host with uniform-distribution random data
    // cutlass::reference::host::TensorFill(tensor_a.host_view(), ElementInputA(1));
    cutlass::reference::host::TensorFillRandomUniform(
        tensor_b.host_view(),
        1,
        ElementInputB(4),
        ElementInputB(-4),
        0);  // <- Fill matrix B on host with uniform-distribution random data
    // cutlass::reference::host::TensorFill(tensor_b.host_view(), ElementInputB(1));
  } else {
    cutlass::reference::host::TensorFill(tensor_a.host_view(), ElementInputA(1));
    cutlass::reference::host::TensorFill(tensor_b.host_view(), ElementInputB(1));
  }
  cutlass::reference::host::TensorFill(tensor_c.host_view());
  cutlass::reference::host::TensorFill(
      tensor_d.host_view());  // <- fill matrix D on host with zeros
  cutlass::reference::host::TensorFill(
      tensor_ref_d.host_view());  // <- fill matrix D for reference on host with zeros

  // Copy data from host to GPU
  tensor_a.sync_device();
  tensor_b.sync_device();
  // tensor_c.sync_device();
  tensor_d.sync_device();
  tensor_ref_d.sync_device();
  // cudaSetDevice(0);
  // Initialize alpha and beta for dot product computation
  ElementComputeEpilogue alpha = ElementComputeEpilogue(options.alpha);
  ElementComputeEpilogue beta = ElementComputeEpilogue(0);//ElementComputeEpilogue(options.beta);

  // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
  // instantiated CUTLASS kernel
  if (options.level == 2)
    problem_size = cutlass::gemm::GemmCoord{problem_size.m()/2, problem_size.n()/2, problem_size.k()/2};

  typename Gemm::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                     tensor_a.device_ref(),  // <- reference to matrix A on device
                                     tensor_b.device_ref(),  // <- reference to matrix B on device
                                     {nullptr, tensor_d.device_ref().layout()},  // <- reference to matrix C on device
                                     tensor_d.device_ref(),  // <- reference to matrix D on device
                                     {alpha, 
                                     #if defined(STRASSEN_THREADBLOCK) || defined(STRASSEN_GLOBAL_PRESUM_LEVEL1) || defined(STRASSEN_FOUR_PRESUM_LEVEL1)
                                     ElementComputeEpilogue(0.0f),
                                     #elif defined(STRASSEN_GLOBAL_LEVEL1)
                                     ElementComputeEpilogue(1.0f)
                                     #endif
                                     },          // <- tuple of alpha and beta
                                     options.split_k_slices};        // <- k-dimension split factor

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);

  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Instantiate CUTLASS kernel depending on templates
  Gemm gemm_op;

  // Check the problem size is supported or not 
  cutlass::Status status = gemm_op.can_implement(arguments);
  CUTLASS_CHECK(status);

  // Initialize CUTLASS kernel with arguments and workspace pointer
  status = gemm_op.initialize(arguments, workspace.get());
  CUTLASS_CHECK(status);

  cudaStream_t streams[7];
  cudaStreamCreate(&streams[0]);
  cudaStreamCreate(&streams[1]);
  cudaStreamCreate(&streams[2]);
  cudaStreamCreate(&streams[3]);
  cudaStreamCreate(&streams[4]);
  cudaStreamCreate(&streams[5]);
  cudaStreamCreate(&streams[6]);

  // int streamLeastPriority, streamGreatestPriority;
  // cudaDeviceGetStreamPriorityRange(&streamLeastPriority, &streamGreatestPriority);
  // cudaStreamCreateWithPriority(&streams[0], 0, streamGreatestPriority);
  // cudaStreamCreateWithPriority(&streams[1], 0, streamGreatestPriority + 1);
  // cudaStreamCreateWithPriority(&streams[2], 0, streamGreatestPriority+2);
  // cudaStreamCreateWithPriority(&streams[3], 0, streamGreatestPriority+3);
  // cudaStreamCreateWithPriority(&streams[4], 0, streamGreatestPriority+4);
  // cudaStreamCreateWithPriority(&streams[5], 0, streamGreatestPriority+5);
  // cudaStreamCreate(&streams[6]);

  // Result structure
  Result result;
  uint num_streams = options.streams;

  if (options.reference_check) {
    cudaError_t reference_result = ReferenceStrassenWinogradGemm<
      ElementInputA,
      ElementInputB,
      ElementOutput,
      ElementComputeEpilogue,
      ElementComputeEpilogue>(
        problem_size.m(),
        problem_size.n(),
        problem_size.k(),
        options.level,
        alpha,
        tensor_a.device_data(),
        tensor_a.layout().stride(0),
        tensor_b.device_data(),
        tensor_b.layout().stride(0),
        beta,
        tensor_ref_d.device_data(),
        tensor_ref_d.layout().stride(0));
    if (reference_result != cudaSuccess) {
      std::cerr << "Reference GEMM kernel failed: "
        << cudaGetErrorString(reference_result) << std::endl;
      return -1;
    }

    // Wait for kernels to finish
    cudaDeviceSynchronize();
    tensor_ref_d.sync_host();
    
    status = gemm_op.run(streams, num_streams);
    CUTLASS_CHECK(status);
    auto err = cudaDeviceSynchronize();
    if (err != cudaSuccess){
      std::cout << __LINE__ << " : " << cudaGetErrorString(err) << std::endl;
      abort();
    }
    
    // Copy output data from CUTLASS and reference kernel to host for comparison
    tensor_d.sync_host();

    // Check if output from CUTLASS kernel and reference kernel are equal or not
    bool passed = true;
    // cutlass::reference::host::TensorEquals(
      // tensor_d.host_view(),
      // tensor_ref_d.host_view());
    if (true) {
      for (int i = 0; i < tensor_d.size(); i++) {
        cutlass::half_t e1 = tensor_d.host_view().data()[i];
        cutlass::half_t e2 = tensor_ref_d.host_view().data()[i];
        float computed = static_cast<float>(e1);
        float reference = static_cast<float>(e2);
        float abs_err = std::fabs(computed - reference);
        float relative_error = std::fabs(computed - reference) /
          std::fmax(std::fabs(reference), 1e-6f);
        
        //Reference gemm and this gemm might have different accumulator types
        if (!(computed == reference or abs_err <= 6 or relative_error <= 1e-2)) {
          printf("389: %d, %d %f, %f (rel err=%f)\n", i/problem_size.n(),i%problem_size.n(), computed, reference, relative_error);
          passed = false;
          break;
        }
      }
    }

    if (passed) std::cout << "Passed" << std::endl;
    else        {std::cout << "Failed" << std::endl; return -1;}
  }

  //
  // Construct events
  //

  //Warmups
  for (int iter = 0; iter < max(10, options.iterations/10); ++iter) {
    // Launch initialized CUTLASS kernel
    status = gemm_op.run(streams, num_streams);
    if (num_streams > 1) result.error = cudaDeviceSynchronize();
    CUTLASS_CHECK(status);
  }

  cudaEvent_t events[2];

  for (auto & event : events) {
    result.error = cudaEventCreate(&event);
    if (result.error != cudaSuccess) {
      std::cerr << "cudaEventCreate() failed: " << cudaGetErrorString(result.error) << std::endl;
      return -1;
    }
  }

  // Record an event at the start of a series of GEMMs
  result.error = cudaEventRecord(events[0]);
  if (result.error != cudaSuccess) {
    std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result.error) << std::endl;
    return -1;
  }

  //
  // Run profiling loop
  //

  for (int iter = 0; iter < options.iterations; ++iter) {
    // Launch initialized CUTLASS kernel
    status = gemm_op.run(streams, num_streams);
    if (num_streams > 1) result.error = cudaDeviceSynchronize();
    CUTLASS_CHECK(status);
  }

  //
  // Stop profiling loop
  //

  result.error = cudaDeviceSynchronize();
  if (result.error != cudaSuccess) {
    std::cerr << "cudaDeviceSynchronize() failed: " << cudaGetErrorString(result.error) << std::endl;
    return -1;
  }

  // Record an event when the GEMMs are complete
  result.error = cudaEventRecord(events[1]);
  if (result.error != cudaSuccess) {
    std::cerr << "cudaEventRecord() failed: " << cudaGetErrorString(result.error) << std::endl;
    return -1;
  }

  // Wait for work on the device to complete.
  result.error = cudaEventSynchronize(events[1]);
  if (result.error != cudaSuccess) {
    std::cerr << "cudaEventSynchronize() failed: " << cudaGetErrorString(result.error) << std::endl;
    return -1;
  }

  // Measure elapsed runtime
  float runtime_ms = 0;
  result.error = cudaEventElapsedTime(&runtime_ms, events[0], events[1]);
  if (result.error != cudaSuccess) {
    std::cerr << "cudaEventElapsed() failed: " << cudaGetErrorString(result.error) << std::endl;
    return -1;
  }

  // Compute average runtime and GFLOPs.
  result.runtime_ms = (options.level == 1 ? 1 : 7)*double(runtime_ms) / double(options.iterations);
  result.gflops = options.gflops(result.runtime_ms / 1000.0);

  // Cleanup
  for (auto event : events) {
    (void)cudaEventDestroy(event);
  }

  std::cout << "Runtime: " << result.runtime_ms << " ms" << std::endl;
  std::cout << " GFLOPs: " << result.gflops << std::endl;

  return 0;
}

int main(int argc, const char **argv) {
  
  bool notSupported = false;

  // Ampere Tensor Core operations exposed with mma.sync and ldmatrix are first available
  // in CUDA 11.0. 
  //
  // CUTLASS must be compiled with CUDA 11.0 Toolkit to run these examples.
  if (!(__CUDACC_VER_MAJOR__ >= 11)) {
    std::cerr << "Ampere Tensor Core operations must be compiled with CUDA 11.0 Toolkit or later." << std::endl;
    notSupported = true;
  }

  cudaDeviceProp props;

  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (error != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties() returned an error: " << cudaGetErrorString(error) << std::endl;
    return -1;
  }

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

  printf("%d x %d x %d F16 tensor op Matrix Multiply\n", \
    options.problem_size.m(), options.problem_size.n(), options.problem_size.k());

  if (!options.valid()) {
    std::cerr << "Invalid problem." << std::endl;
    return -1;
  }

  return run(options);
}
