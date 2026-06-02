/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/*! \file
    \brief Simple Hopper GEMM example using CUTLASS 3.0 APIs for NVIDIA Hopper architecture

    This example demonstrate a simple way to instantiate and run a TF32 GEMM using the new CUTLASS 3.0
    APIs on NVIDIA Hopper architecture. New features that will be showcased in this example are as follows:

    1. NVIDIA Hopper architecture introduces a new series of tensor core instructions (GMMA)
    which are more efficient than the Ampere tensor core instructions.

    2. NVIDIA Hopper architecture includes new Tensor Memory Accelerator (TMA) unit to transfer large
    blocks of data efficiently between global memory and shared memory. TMA also supports asynchronous
    copies between thread blocks in a cluster. Another advantage is that TMA can load in FP32 data and
    convert them implicitly to TF32.

    3. This example uses the Warp Specialized kernel design (see /media/docs/efficient_gemm.md for details).

    4. A simple way to tune the CTA rasterization direction and swizzle pattern of Hopper kernels. Both the 
    CTA rasterization direction and swizzle pattern impact cross-CTA locality of accesses. By tuning we can 
    improve performance.

    Examples:

      $ ./examples/48_hopper_warp_specialized_gemm/48_hopper_warp_specialized_gemm --m=2048 --n=2048 --k=2048 --rasterization=N --swizzle=2
*/

#include <iostream>

#define MY_PRINTF(...) ;//printf(__VA_ARGS__)

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_strassen_gemm_builder.hpp"
#include "cutlass/epilogue/collective/collective_strassen_builder.hpp"
#include "cutlass/gemm/device/strassen_gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/strassen_gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/tensor_view_io.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/device/tensor_fill.h"

#include "cutlass/gemm/device/strassen_decls.h"
using namespace MmaStrassen;

#include "helper.h"

using namespace cute;

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM kernel configurations
/////////////////////////////////////////////////////////////////////////////////////////////////

// A matrix configuration
using         ElementA    = cutlass::half_t;                                // Element type for A matrix operand
using         LayoutA     = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<ElementA>::value;    // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

// B matrix configuration
using         ElementB    = cutlass::half_t;                                // Element type for B matrix operand
using         LayoutB     = cutlass::layout::RowMajor;                   // Layout type for B matrix operand
constexpr int AlignmentB  = 128 / cutlass::sizeof_bits<ElementB>::value;    // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

// C/D matrix configuration
using         ElementC    = cutlass::half_t;                                // Element type for C and D matrix operands
using         LayoutC     = cutlass::layout::RowMajor;                   // Layout type for C and D matrix operands
constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)

// Core kernel configurations
using ElementAccumulator  = float;                                          // Element type for internal accumulation
using ArchTag             = cutlass::arch::Sm90;                            // Tag indicating the minimum SM that supports the intended feature
using OperatorClass       = cutlass::arch::OpClassTensorOp;                 // Operator class tag
using TileShape           = Shape<_128,_128,_64>;                           // Threadblock-level tile size
using ClusterShape        = Shape<_2,_1,_1>;                                // Shape of the threadblocks in a cluster
const uint StageCountTypeM0 = 6 ; //cutlass::gemm::collective::StageCountAuto;           // Stage count maximized based on the tile size
const uint StageCountTypeM2M6 = 6 ;
using PresumTileShapeA    = Shape<_2, _128>;
using PresumTileShapeB    = Shape<_2, _128>;
using KernelSchedule = cutlass::gemm::collective::KernelScheduleAuto;       // Kernel to launch based on the default setting in the Collective Builder
using PresumOpts = cutlass::gemm::device::PresumOpt<>;//0,0,0,0>;
//StageCount = 6 is a little slower than this with swizzle = 8.
//TODO: Stages 5 produces wrong results for C2

using AllPresumsKernel = AllPresums<>;
                          //  AllPresums<PresumGlobalKernel,   PresumGlobalKernel,  PresumGlobalKernel,   PresumGlobalKernel,  //A Presums
                                        // PresumGlobalKernel,   PresumGlobalKernel,  PresumGlobalKernel,   PresumGlobalKernel>; //B Presums
using AllPresumsM0    = AllPresums<PresumCompute, PresumCompute, PresumCompute, PresumCompute,
                                   PresumCompute, PresumCompute, PresumCompute, PresumCompute>;
//TODO: Can also divide presum among M0 and M1 if K * K/N is not big enough
//TODO: If PresumShape and K/TK cannot cover all of A and B then report error

using AllPresumsM1To6 = AllPresums<PresumAvailable, PresumAvailable, PresumAvailable, PresumAvailable, PresumAvailable,    PresumAvailable,    PresumAvailable,    PresumAvailable>;

#if 1
using StrassenGroups = StrassenLevel1Groups<StrassenPresum<1, 0, TileShape, AllPresumsKernel>,
                                            StrassenLevel1MiGroup<1, 0, TileShape, ClusterShape, StageCountTypeM0,
                                                                  RWMTypes<>,
                                                                  RWCTypes<CUW<1, LayoutInterim1D, LayoutNone, Expr<Plus<0>>>,//C1 = M0
                                                                           CUW<0, LayoutFinal, LayoutNone, Expr<Plus<1>>>>,//C0 = M1
                                                                  AllPresumsM0, 0, 0, 1>,
                                            StrassenLevel1M1Group<1, 0, TileShape, ClusterShape, StageCountTypeM0,
                                                                  RWMTypes<>,
                                                                  RWCTypes<//CUW<1, LayoutInterim, LayoutNone, Expr<Plus<0>>>,//C1 = M0
                                                                            CUW<0, LayoutFinal, LayoutNone, Expr<Plus<1>>, Expr<Plus<1, MemGlobal, LayoutInterim1D>>>>,//C0 = M1
                                                                  AllPresumsM0>,
                                            StrassenLevel1MiGroup<1, 0, TileShape, ClusterShape, StageCountTypeM2M6,
                                                                  RWMTypes<>,
                                                                  RWCTypes<CUW<1, LayoutInterim1D, LayoutInterim1D, Expr<Plus<2>>, Expr<Plus<1, MemGlobal, LayoutInterim1D>>>, //C1 = Sh = C1+M2 ; Reg = C1 //TODO: pass C1 through registers
                                                                           CUW<2, LayoutInterim1D, LayoutNone, Expr<Plus<3>>, Expr<Plus<1, MemShared, LayoutInterim1D>>> >, //C2 = C1Sh+M3
                                                                  AllPresumsM1To6, 0, 2, 3>,
                                            StrassenLevel1M3Group<1, 0, TileShape, ClusterShape, StageCountTypeM2M6,
                                                                  RWMTypes<>,
                                                                  RWCTypes<CUW<2, LayoutNone, LayoutInterim1D, Expr<Plus<3>>, Expr<Plus<1, MemGlobal, LayoutInterim1D>>>>,//C2 = C1(Reg)+M3 
                                                                  AllPresumsM1To6>,
                                            StrassenLevel1MiGroup<1, 0, TileShape, ClusterShape, StageCountTypeM2M6,
                                                                  RWMTypes<>,
                                                                  RWCTypes<CUW<0, LayoutNone,  LayoutInterim1D,  Expr<Plus<4>>>, //C1 (stored at M0) = C1+M4
                                                                           CUW<3, LayoutFinal, LayoutNone, Expr<Plus<4>>, Expr<Plus<2, MemGlobal, LayoutInterim1D>>>,//C3 = C2+M4
                                                                           CUW<1, LayoutFinal, LayoutNone, Expr<Plus<5>>, Expr<Plus<1, MemGlobal, LayoutInterim1D>,
                                                                                                                               Plus<0, MemShared, LayoutInterim1D>
                                                                                                                               >>
                                                                           >,
                                                                  AllPresumsM1To6, 0, 4, 5>,
                                            StrassenLevel1M5Group<1, 0, TileShape, ClusterShape, StageCountTypeM2M6,
                                                                  RWMTypes<>,
                                                                  RWCTypes<CUW<1, LayoutFinal, LayoutNone, Expr<Plus<5>>, Expr<Plus<1, MemGlobal, LayoutInterim1D>,
                                                                                                                               Plus<0, MemGlobal, LayoutInterim1D>>>>, //C1 = C1+M5 //TODO: in code M5 reads M1 and M0 (written by M4)
                                                                  AllPresumsM1To6>,
                                            StrassenLevel1M6Group<1, 0, TileShape, ClusterShape, StageCountTypeM2M6,
                                                                  RWMTypes<>,
                                                                  RWCTypes<CUW<2, LayoutFinal, LayoutNone, Expr<Neg<6>>, Expr<Plus<2, MemGlobal, LayoutInterim1D>>>>, //C2 = C2-M6
                                                                  AllPresumsM1To6>
                                            >;
using ScheduleStrassenGroups1 = ScheduleStrassenGroups<ParallelMiGroups<false, FusedMiGroup<7, 0>>,
                                                       ParallelMiGroups<false, FusedMiGroup<7, 2>, //TODO: Change this to true
                                                                               FusedMiGroup<7, 4>,
                                                                               FusedMiGroup<7, 6>>
                                                      //  ParallelMiGroups<true, FusedMiGroup<7, 2>>,
                                                      //  ParallelMiGroups<true, FusedMiGroup<7, 3>>,
                                                      //  ParallelMiGroups<false, FusedMiGroup<7, 4>>,
                                                      //  ParallelMiGroups<true, FusedMiGroup<7, 5>>,
                                                      //  ParallelMiGroups<false, FusedMiGroup<7, 6>>
                                                        >;

#else

using StrassenGroups = StrassenLevel1Groups<StrassenPresum<1, 0, TileShape, AllPresumsKernel>,
                                            StrassenLevel1MiGroup<1, 0, TileShape, ClusterShape, StageCountTypeM0,
                                                                  RWMTypes<>,
                                                                  RWCTypes<CUW<1, LayoutInterim1D, LayoutNone, Expr<Plus<0>>>,//C1 = M0
                                                                           CUW<0, LayoutFinal, LayoutNone, Expr<Plus<1>>>>,//C0 = M1
                                                                  AllPresumsM0, 0, 0, 1>,
                                            StrassenLevel1M1Group<1, 0, TileShape, ClusterShape, StageCountTypeM0,
                                                                  RWMTypes<>,
                                                                  RWCTypes<//CUW<1, LayoutInterim, LayoutNone, Expr<Plus<0>>>,//C1 = M0
                                                                            CUW<0, LayoutFinal, LayoutNone, Expr<Plus<1>>, Expr<Plus<1, MemGlobal, LayoutInterim1D>>>>,//C0 = M1
                                                                  AllPresumsM0>,
                                            StrassenLevel1MiGroup<1, 0, TileShape, ClusterShape, StageCountTypeM2M6,
                                                                  RWMTypes<>,
                                                                  RWCTypes<CUW<1, LayoutInterim1D, LayoutInterim1D, Expr<Plus<2>>, Expr<Plus<1, MemGlobal, LayoutInterim1D>>>, //C1 = Sh = C1+M2 ; Reg = C1 //TODO: pass C1 through registers
                                                                           CUW<2, LayoutInterim1D, LayoutNone, Expr<Plus<3>>, Expr<Plus<1, MemShared, LayoutInterim1D>>> >, //C2 = C1Sh+M3
                                                                  AllPresumsM1To6, 0, 2, 3>,
                                            StrassenLevel1M3Group<1, 0, TileShape, ClusterShape, StageCountTypeM2M6,
                                                                  RWMTypes<>,
                                                                  RWCTypes<CUW<2, LayoutNone, LayoutInterim1D, Expr<Plus<3>>, Expr<Plus<1, MemGlobal, LayoutInterim1D>>>>,//C2 = C1(Reg)+M3 
                                                                  AllPresumsM1To6>,
                                            StrassenLevel1M4Group<1, 0, TileShape, ClusterShape, StageCountTypeM2M6,
                                                                  RWMTypes<>,
                                                                  RWCTypes<CUW<0, LayoutInterim1D, LayoutNone,  Expr<Plus<4>>>, //C1 (stored at M0) = C1+M4
                                                                           CUW<3, LayoutFinal, LayoutNone, Expr<Plus<4>>, Expr<Plus<2, MemGlobal, LayoutInterim1D>>>,//C3 = C2+M4
                                                                           >,
                                                                  AllPresumsM1To6>,
                                            StrassenLevel1M5Group<1, 0, TileShape, ClusterShape, StageCountTypeM2M6,
                                                                  RWMTypes<>,
                                                                  RWCTypes<CUW<1, LayoutFinal, LayoutNone, Expr<Plus<5>>, Expr<Plus<1, MemGlobal, LayoutInterim1D>,
                                                                                                                               Plus<0, MemGlobal, LayoutInterim1D>>>>, //C1 = C1+M5 //TODO: in code M5 reads M1 and M0 (written by M4)
                                                                  AllPresumsM1To6>,
                                            StrassenLevel1M6Group<1, 0, TileShape, ClusterShape, StageCountTypeM2M6,
                                                                  RWMTypes<>,
                                                                  RWCTypes<CUW<2, LayoutFinal, LayoutNone, Expr<Neg<6>>, Expr<Plus<2, MemGlobal, LayoutInterim1D>>>>, //C2 = C2-M6
                                                                  AllPresumsM1To6>
                                            >;
//For 8kx8kx8k. do not create a single kernel and run as
// strassen_winograd_presum4_hopper_f16_tensorop_gemm --m=$((8*1024)) --n=$((8*1024)) --k=$((8*1024)) --iterations=100 --check=0 --streams=7 --swizzles=2,2,1,1,1,1,1 --beta=0 --raster=N
using ScheduleStrassenGroups1 = ScheduleStrassenGroups<ParallelMiGroups<false, FusedMiGroup<7, 0>>,
                                                       ParallelMiGroups<false, FusedMiGroup<7, 2>>, //TODO: Change this to true
                                                                              //  FusedMiGroup<7, 4>,
                                                                              //  FusedMiGroup<7, 5>,
                                                                              //  FusedMiGroup<7, 6>>
                                                      //  ParallelMiGroups<true, FusedMiGroup<7, 2>>,
                                                      //  ParallelMiGroups<true, FusedMiGroup<7, 3>>,
                                                       ParallelMiGroups<false, FusedMiGroup<7, 4>>,
                                                       ParallelMiGroups<false, FusedMiGroup<7, 5>>,
                                                      //  ParallelMiGroups<true, FusedMiGroup<7, 5>>,
                                                       ParallelMiGroups<false, FusedMiGroup<7, 6>>
                                                        >;
#endif

using StrassenGemmKernels = cutlass::gemm::device::StrassenGemmKernels<StrassenGroups,
                                                                       ElementA, LayoutA, ElementB, LayoutB,
                                                                       ElementC, LayoutC,
                                                                       ElementAccumulator, TileShape, ClusterShape,
                                                                       cute::Int<StageCountTypeM0>,
                                                                       PresumTileShapeA, PresumTileShapeB,
                                                                       PresumOpts>;

using Gemm = cutlass::gemm::device::StrassenGemmUniversalAdapter<ScheduleStrassenGroups1, StrassenGemmKernels>;

// Reference device GEMM implementation type
using DeviceGemmReference = cutlass::reference::device::Gemm<
  ElementA,
  LayoutA,
  ElementB,
  LayoutB,
  ElementC,
  LayoutC,
  ElementAccumulator,
  ElementAccumulator>;

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

//
// Data members
//

/// Initialization
StrideA stride_A;
StrideB stride_B;
StrideC stride_C;
StrideD stride_D;
uint64_t seed;

cutlass::DeviceAllocation<typename Gemm::ElementA> block_A;
cutlass::DeviceAllocation<typename Gemm::ElementB> block_B;
cutlass::DeviceAllocation<typename Gemm::ElementC> block_C;
cutlass::DeviceAllocation<typename Gemm::EpilogueOutputOp::ElementOutput> block_D;
cutlass::DeviceAllocation<typename Gemm::EpilogueOutputOp::ElementOutput> block_ref_D;

#endif // defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// Testbed utility types
/////////////////////////////////////////////////////////////////////////////////////////////////

using RasterOrderOptions = typename cutlass::gemm::kernel::detail::PersistentTileSchedulerSm90Params::RasterOrderOptions;

// Command line options parsing
struct Options {

  bool help;

  float alpha, beta;
  int iterations;
  int m, n, k;
  RasterOrderOptions raster;
  int swizzles[7];
  bool reference_check;
  int streams;

  Options():
    help(false),
    m(5120), n(4096), k(4096),
    alpha(1.f), beta(0.f),
    reference_check(true),
    iterations(1000),
    raster(RasterOrderOptions::Heuristic),
    streams(1)

  { }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    if (cmd.check_cmd_line_flag("mnk")) {
      int mnk = 0;
      cmd.get_cmd_line_argument("mnk", mnk);
      m = n = k = mnk;
    } else {
      cmd.get_cmd_line_argument("m", m);
      cmd.get_cmd_line_argument("n", n);
      cmd.get_cmd_line_argument("k", k);
    }

    cmd.get_cmd_line_argument("streams", streams);
    cmd.get_cmd_line_argument("check", reference_check);
    cmd.get_cmd_line_argument("alpha", alpha, 1.f);
    cmd.get_cmd_line_argument("beta", beta, 0.f);
    cmd.get_cmd_line_argument("iterations", iterations);

    char raster_char;
    cmd.get_cmd_line_argument("raster", raster_char);

    if (raster_char == 'N' || raster_char == 'n') {
      raster = RasterOrderOptions::AlongN;
    }
    else if (raster_char == 'M' || raster_char == 'm') {
      raster = RasterOrderOptions::AlongM;
    }
    else if (raster_char == 'H' || raster_char == 'h') {
      raster = RasterOrderOptions::Heuristic;
    }
    
    std::string str_swizzles;
    std::string default_swizzle = "1,1,1,1,1,1,1";
    cmd.get_cmd_line_argument("swizzles", str_swizzles, default_swizzle);

    std::stringstream str_swizzles_stream(str_swizzles);

    std::string str_swizzle;
    int idx = 0;
    while(std::getline(str_swizzles_stream, str_swizzle, ','))
    {
      if (idx < 7)
        swizzles[idx] = stoi(str_swizzle);
      idx++; 
    }
    for (int ii = idx; ii < 7; ii++) swizzles[ii] = 1;
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "48_hopper_warp_specialized_gemm\n\n"
      << "  Hopper FP16 GEMM using a Warp Specialized kernel.\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement\n\n"
      << "  --m=<int>                   Sets the M extent of the GEMM\n"
      << "  --n=<int>                   Sets the N extent of the GEMM\n"
      << "  --k=<int>                   Sets the K extent of the GEMM\n"
      << "  --alpha=<f32>               Epilogue scalar alpha\n"
      << "  --beta=<f32>                Epilogue scalar beta\n\n"
      << "  --raster=<char>             CTA Rasterization direction (N for along N, M for along M, and H for heuristic)\n\n"
      << "  --swizzle=<int>             CTA Rasterization swizzle\n\n"
      << "  --iterations=<int>          Number of profiling iterations to perform.\n\n"
      << "  --streams=<int>             Number of overlapping streams.\n\n"
      << "  --check=<0|1>               Check results or not.\n\n";


    out
      << "\n\nExamples:\n\n"
      << "$ " << "48_hopper_warp_specialized_gemm" << " --m=1024 --n=512 --k=1024 --alpha=2 --beta=0.707 \n\n";

    return out;
  }

  /// Compute performance in GFLOP/s
  double gflops(double runtime_s) const
  {
    // Two flops per multiply-add
    uint64_t flop = uint64_t(2) * m * n * k;
    double gflop = double(flop) / double(1.0e9);
    return gflop / runtime_s;
  }
};

/// Result structure
struct Result
{
  double avg_runtime_ms;
  double gflops;
  cutlass::Status status;
  cudaError_t error;
  bool passed;

  Result(
    double avg_runtime_ms = 0,
    double gflops = 0,
    cutlass::Status status = cutlass::Status::kSuccess,
    cudaError_t error = cudaSuccess)
  :
    avg_runtime_ms(avg_runtime_ms), gflops(gflops), status(status), error(error), passed(false)
  {}

};

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

/////////////////////////////////////////////////////////////////////////////////////////////////
/// GEMM setup and evaluation
/////////////////////////////////////////////////////////////////////////////////////////////////

/// Helper to initialize a block of device data
template <class Element>
bool initialize_block(
  cutlass::DeviceAllocation<Element>& block,
  uint64_t seed=2023, bool mod = false, bool ones = false) {

  Element scope_max, scope_min;
  int bits_input = cutlass::sizeof_bits<Element>::value;

  if (bits_input == 1) {
    scope_max = Element(2);
    scope_min = Element(0);
  } else if (bits_input <= 8) {
    scope_max = Element(2);
    scope_min = Element(-2);
  } else {
    scope_max = Element(4);
    scope_min = Element(-4);
  }

  if (!mod and !ones) {
    cutlass::reference::device::BlockFillRandomUniform(
      block.get(), block.size(), seed, scope_max, scope_min, 0);
  } else if (mod) {
    Element* host =  new Element[block.size()];
    for (int i = 0; i < block.size(); i++)
      host[i] = (static_cast<Element>(i/(9*1024)));
    CUDA_CHECK(cudaMemcpy(block.get(), host, block.size() * sizeof(Element), cudaMemcpyHostToDevice));
    delete[] host;
  } else if (ones) {
    using Layout = cutlass::layout::PackedVectorLayout;
    Layout::TensorCoord size(static_cast<Layout::Index>(block.size())); // -Wconversion
    Layout layout = Layout::packed(size);
    cutlass::TensorView<Element, Layout> view(block.get(), layout, size);

    cutlass::reference::device::TensorFill(
      view, Element(1));
  }

  return true;
}

/// Initialize operands to be used in the GEMM and reference GEMM
void initialize(const Options &options) {

  stride_A = cutlass::make_cute_packed_stride(StrideA{}, {options.m, options.k, 1});
  stride_B = cutlass::make_cute_packed_stride(StrideB{}, {options.n, options.k, 1});
  stride_C = cutlass::make_cute_packed_stride(StrideC{}, {options.m, options.n, 1});
  stride_D = cutlass::make_cute_packed_stride(StrideD{}, {options.m, options.n, 1});

  block_A.reset(options.m * options.k);
  block_B.reset(options.k * options.n);
  block_C.reset(options.m * options.n);
  block_D.reset(options.m * options.n);
  block_ref_D.reset(options.m * options.n);

  initialize_block(block_A, seed + 2023, false, false);
  initialize_block(block_B, seed + 2022, false, false);
  initialize_block(block_C, seed + 2021);
}

/// Populates a Gemm::Arguments structure from the given commandline options
typename Gemm::Arguments args_from_options(const Options &options)
{
  // Change device_id to another value if you are running on a machine with multiple GPUs and wish
  // to use a GPU other than that with device ID 0.
  int device_id = 0;
  cutlass::KernelHardwareInfo kernel_hw_info = cutlass::KernelHardwareInfo::make_kernel_hardware_info<Gemm::GemmKernel>(device_id);

  typename Gemm::Arguments arguments(
    cutlass::gemm::GemmUniversalMode::kGemm,
    {options.m, options.n, options.k},
    {block_A.get(), stride_A, block_B.get(), stride_B},
    {{options.alpha, options.beta}, block_C.get(), stride_C, block_D.get(), stride_D},
    kernel_hw_info
  );

  arguments.scheduler.raster_order = options.raster;
  // The tile scheduler will swizzle up to 8 and with the nearest multiple of 2 (i.e., 1, 2, 4, and 8) 
  // arguments.scheduler.max_swizzle_size = options.swizzles;

  return arguments;
}

bool verify(const Options &options) {
  cutlass::TensorRef ref_A(block_A.get(), Gemm::LayoutA::packed({options.m, options.k}));
  cutlass::TensorRef ref_B(block_B.get(), Gemm::LayoutB::packed({options.k, options.n}));
  cutlass::TensorRef ref_C(block_C.get(), Gemm::LayoutC::packed({options.m, options.n}));
  cutlass::TensorRef ref_D(block_ref_D.get(), Gemm::LayoutD::packed({options.m, options.n}));

  //
  // Compute reference output
  //

  // Create instantiation for device reference gemm kernel
  DeviceGemmReference gemm_reference;

  // Launch device reference gemm kernel
  gemm_reference(
    {options.m, options.n, options.k},
    ElementAccumulator(options.alpha),
    ref_A,
    ref_B,
    ElementAccumulator(options.beta),
    ref_C,
    ref_D);

  // Wait for kernel to finish
  CUDA_CHECK(cudaDeviceSynchronize());

  // Check if output from CUTLASS kernel and reference kernel are equal or not
  // bool passed = cutlass::reference::device::BlockCompareEqual(block_ref_D.get(), block_D.get(), block_D.size());
  bool passed = true;

  ElementC* host_ref_D = new ElementC[options.m*options.n];
  CUDA_CHECK(cudaMemcpy(host_ref_D, block_ref_D.get(), options.m*options.n*sizeof(ElementC), cudaMemcpyDeviceToHost));
  ElementC* host_D = new ElementC[options.m*options.n];
  CUDA_CHECK(cudaMemcpy(host_D, block_D.get(), options.m*options.n*sizeof(ElementC), cudaMemcpyDeviceToHost));

  float MAX_REL_ERR = 1e-2;
  float MAX_ABS_ERR = 5;

  for (int r = 0; r < options.m; r++) {
  for (int c = 0; c < options.n; c++) {
    auto idx = r*options.n + c;
    cutlass::half_t e1 = host_ref_D[idx];
    cutlass::half_t e2 = host_D[idx];
    float err = fabs((float)e1 -(float)e2)/fabs((float)e1 + 1e-6);
    float abs_err = fabs((float)e1 -(float)e2);

    //C0
    if (r < options.m/2 && c < options.n/2) {
      if (!((float)e1 == (float)e2 or err <= MAX_REL_ERR or abs_err <= MAX_ABS_ERR)) {
        printf("389: %d, %d at ref: %f, computed: %f\n", r, c, (float)e1, (float)e2);
        passed = false;
      }
    }

    //C1
    if (r < options.m/2 && c >= options.n/2) {
      if (!((float)e1 == (float)e2 or err <= MAX_REL_ERR or abs_err <= MAX_ABS_ERR)) {
        printf("389: %d, %d at ref: %f, computed: %f\n", r, c, (float)e1, (float)e2);
        passed = false;
      }
    }

    //C2
    if (r >= options.m/2 && c < options.n/2) {
      if (!((float)e1 == (float)e2 or err <= MAX_REL_ERR or abs_err <= MAX_ABS_ERR)) {
        printf("389: %d, %d at ref: %f, computed: %f\n", r, c, (float)e1, (float)e2);
        passed = false;
      }
    }

    //C3
    if (r >= options.m/2 && c >= options.n/2) {
      if (!((float)e1 == (float)e2 or err <= MAX_REL_ERR or abs_err <= MAX_ABS_ERR)) {
        printf("389: %d, %d at ref: %f, computed: %f\n", r, c, (float)e1, (float)e2);
        passed = false;
      }
    }
    // if (r > 0 || c > 4112) 
    if (!passed) break;
  }
  if (!passed) break;
  }

  return passed;
}

/// Execute a given example GEMM computation
template <typename Gemm>
int run(Options &options)
{
  initialize(options);

  cudaEvent_t events[7];

  cudaStream_t streams[7];
  for (int i = 0; i < 7; i++) {
    CUDA_CHECK(cudaStreamCreateWithPriority(&streams[i], cudaStreamDefault, 0+i));
    CUDA_CHECK(cudaEventCreateWithFlags(&events[i], cudaEventDisableTiming));
  }

  // Instantiate CUTLASS kernel depending on templates
  Gemm gemm;

  // Create a structure of gemm kernel arguments suitable for invoking an instance of Gemm
  auto arguments = args_from_options(options);

  // Using the arguments, query for extra workspace required for matrix multiplication computation
  size_t workspace_size = Gemm::get_workspace_size(arguments);
  std::cout << 464 << " " <<workspace_size << std::endl;
  // Allocate workspace memory
  cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

  // Check if the problem size is supported or not
  CUTLASS_CHECK(gemm.can_implement(arguments));

  // Initialize CUTLASS kernel with arguments and workspace pointer
  CUTLASS_CHECK(gemm.initialize(arguments, options.swizzles, workspace.get()));

  CUDA_CHECK(cudaDeviceSynchronize());

  // Correctness / Warmup iteration
  CUTLASS_CHECK(gemm.run(streams, options.streams));

  // Check if output from CUTLASS kernel and reference kernel are equal or not
  Result result;

  if (options.reference_check) {
    CUDA_CHECK(cudaDeviceSynchronize());
    result.passed = verify(options);

    std::cout << "  Disposition: " << (result.passed ? "Passed" : "Failed") << std::endl;

    if (!result.passed) {
      exit(-1);
    }
  }

  // Run profiling loop
  if (options.iterations > 0)
  {
    // CUTLASS_CHECK(gemm.initialize(arguments, options.swizzles, workspace.get()));
    for (int iter = 0; iter < options.iterations/10; ++iter) {
      CUTLASS_CHECK(gemm.run(streams, options.streams));
      if (options.streams > 1) result.error = cudaDeviceSynchronize();
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    GpuTimer timer;
    timer.start();
    for (int iter = 0; iter < options.iterations; ++iter) {
      CUTLASS_CHECK(gemm.run(streams, options.streams));
      if (options.streams > 1) {
        // for (int s1 = 0; s1 < options.streams; s1++) {
        //   CUDA_CHECK(cudaEventRecord(events[s1], streams[s1]));
        // }
        // for (int s1 = 0; s1 < options.streams; s1++) {
        //   // for (int s2 = 0; s2 < options.streams; s2++) {
        //     // if (s1 != s2) {
        //       CUDA_CHECK(cudaStreamWaitEvent(streams[s1], events[0]));
        //       CUDA_CHECK(cudaStreamWaitEvent(streams[0], events[s1]));
        //     // }
        //   // }
        // }
        result.error = cudaDeviceSynchronize();
      }
    }
    timer.stop();

    // Compute average runtime and GFLOPs.
    float elapsed_ms = timer.elapsed_millis();
    result.avg_runtime_ms = double(elapsed_ms) / double(options.iterations);
    result.gflops = options.gflops(result.avg_runtime_ms / 1000.0);

    std::string raster = "Heuristic";

    if (options.raster == RasterOrderOptions::AlongN) {
      raster = "Along N";
    }
    else if (options.raster == RasterOrderOptions::AlongM) {
      raster = "Along M";
    }

    std::cout << "  Problem Size: " << options.m << 'x' << options.n << 'x' << options.k << std::endl;
    std::cout << "  Rasterization: " << raster << " with a maximum CTA swizzle of ";
    for (int i = 0; i < 7; i++) std::cout << options.swizzles[i] << ", ";
    std::cout <<std::endl;
    std::cout << "  Avg runtime: " << result.avg_runtime_ms << " ms" << std::endl;
    std::cout << "  GFLOPS: " << result.gflops << std::endl;

    for (int s = 0; s < options.streams; s++) {
      CUDA_CHECK(cudaEventDestroy(events[s]));
      CUDA_CHECK(cudaStreamDestroy(streams[s]));
    }
  }

  return 0;
}

#endif // defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

///////////////////////////////////////////////////////////////////////////////////////////////////

int main(int argc, char const **args) {

  // CUTLASS must be compiled with CUDA 12.0 Toolkit to run this example
  // and must have compute capability at least 90.
  if (__CUDACC_VER_MAJOR__ < 12) {
    std::cerr << "This example requires CUDA 12 or newer.\n";
    // Returning zero so this test passes on older Toolkits. Its actions are no-op.
    return 0;
  }

  cudaDeviceProp props;
  int current_device_id;
  CUDA_CHECK(cudaGetDevice(&current_device_id));
  CUDA_CHECK(cudaGetDeviceProperties(&props, current_device_id));
  cudaError_t error = cudaGetDeviceProperties(&props, 0);
  if (props.major != 9 || props.minor != 0) {
    std::cerr
      << "This example requires a GPU of NVIDIA's Hopper Architecture (compute capability 90).\n";
    return 0;
  }

  
  

  //
  // Parse options
  //

  Options options;

  options.parse(argc, args);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  //
  // Evaluate CUTLASS kernels
  //

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
  run<Gemm>(options);
#endif

  return 0;
}

/////////////////////////////////////////////////////////////////////////////////////////////////
