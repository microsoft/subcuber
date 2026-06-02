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
/*! \file
    \brief Template for a pipelined GEMM kernel. Does not compute batching or support split-K.
*/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/arch/arch.h"
#include "cutlass/strassen_device_kernel.h"

#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/gemm/threadblock/strassen_threadblock_swizzle.h"
#include "cutlass/gemm/kernel/strassen_gemm.h"

#include "cutlass/gemm/kernel/default_strassen_gemm.h"
#include "cutlass/gemm/device/default_gemm_configuration.h"

#include "cutlass/layout/permute.h"
#include "cutlass/strassen_presum_global_kernel.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace device {

/////////////////////////////////////////////////////////////////////////////////////////////////

static __global__ void presumcheck(float* A, float* presum) {
  int row = blockIdx.x;
  int R = 2048, C = 2048;
  // if (row == 0)
  // printf("%d %d: %f + %f = %f\n", row, threadIdx.x, A[(512+row) * 1024 + threadIdx.x], A[(512+row) * 1024 + 512+threadIdx.x], presum[row*512+threadIdx.x]);
  for (int col = threadIdx.x ; col < C; col += blockDim.x)
    if (row == 0 && presum[2*R*C+row*C+col] != 1.0f)
      printf("63: %d %d: %f; %p\n", row, col,
            presum[2*R*C+row*C+col],
            &presum[2*R*C+row*C+col]);
}


static __global__ void postsumcheck(cutlass::half_t* postsum) {
  // int row = blockIdx.x;
  int R = 4096, C = 4096;
  // if (row == 0)
  // printf("%d %d: %f + %f = %f\n", row, threadIdx.x, A[(512+row) * 1024 + threadIdx.x], A[(512+row) * 1024 + 512+threadIdx.x], presum[row*512+threadIdx.x]);
  if (threadIdx.x==0)
  for (int i = 0 ; i < C*R; i += 1)
    if (float(postsum[R*C + i]) != 4096.0f)
      printf("63: %d: %f; %p\n", i,
            float(postsum[R*C + i]),
            &postsum[R*C + i]);
}

template<typename TensorRef>
CUTLASS_HOST_DEVICE
static typename TensorRef::Layout layoutforM(TensorRef ref_A_or_B, int level = 1) {
  typename TensorRef::Layout layout = ref_A_or_B.layout();

  layout.stride() /= typename TensorRef::Stride(1 << level);
  // printf("78 %ld %ld %d\n", ref_A_or_B.layout().stride(0), layout.stride(0), level);
  return layout;
}
/*! Gemm device-level operator. This is an interface to efficient CUTLASS GEMM kernels that may
  be invoked from host code.

  The contributions of this class are:
    
    1. At compile time, it maps data types and high-level structural parameters onto 
       specific CUTLASS components.

    2. At runtime, it maps logical arguments to GEMM problems to kernel parameters.

    3. At runtime, it launches kernels on the device.

  The intent is to provide a convenient mechanism for interacting with most plausible GEMM
  configurations for each supported architecture. Consequently, not all parameters are exposed
  to the top-level interface. Rather, sensible defaults at each level of the CUTLASS hierarchy
  are selected to tradeoff simplicity of the interface with flexibility. We expect 
  most configurations to be specified at this level. Applications with more exotic requirements 
  may construct their kernels of interest using CUTLASS components at the threadblock, warp, 
  and thread levels of abstraction.

  CUTLASS exposes computations using the functor design pattern in which objects compose some
  internal state with an overloaded function call operator. This enables decoupling of
  initialization from execution, possibly reducing overhead during steady state phases of
  application execution.

  CUTLASS device-level operators expose an Arguments structure encompassing each logical
  input to the computation. This is distinct from the kernel-level Params structure pattern
  which contains application-specific precomputed state needed by the device code.

  Example of a CUTLASS GEMM operator implementing the functionality of cuBLAS's SGEMM NN
  is as follows:

    //
    // Instantiate the CUTLASS GEMM operator.
    //

    cutlass::gemm::device::Gemm<
      float,
      cutlass::layout::ColumnMajor,
      float,
      cutlass::layout::ColumnMajor,
      float,
      cutlass::layout::ColumnMajor
    > gemm_op;

    //
    // Launch the GEMM operation on the device
    //

    cutlass::Status status = gemm_op({
      {m, n, k},                          // GemmCoord problem_size,
      {A, lda},                           // TensorRef<float, layout::ColumnMajor> ref_A,
      {B, ldb},                           // TensorRef<float, layout::ColumnMajor> ref_B,
      {C, ldc},                           // TensorRef<float, layout::ColumnMajor> ref_C,
      {D, ldd},                           // TensorRef<float, layout::ColumnMajor> ref_D,
      {alpha, beta}                       // EpilogueOutputOp::Params epilogue_op_params
    });


  A simplified view of the template is listed below.

    template <
      /// Element type for A matrix operand
      typename ElementA,
      
      /// Layout type for A matrix operand
      typename LayoutA,
      
      /// Element type for B matrix operand
      typename ElementB,
      
      /// Layout type for B matrix operand
      typename LayoutB,
      
      /// Element type for C and D matrix operands
      typename ElementC,
      
      /// Layout type for C and D matrix operands
      typename LayoutC,
      
      /// Element type for internal accumulation
      typename ElementAccumulator,

      /// Operator class tag
      typename OperatorClass,
      
      /// Tag indicating architecture to tune for.  This is the minimum SM that
      /// supports the intended feature. The device kernel can be built
      /// targeting any SM larger than this number.
      typename ArchTag,
      
      /// Threadblock-level tile size (concept: GemmShape)
      typename ThreadblockShape,
      
      /// Warp-level tile size (concept: GemmShape)
      typename WarpShape,
      
      /// Warp-level tile size (concept: GemmShape)
      typename InstructionShape,
      
      /// Epilogue output operator
      typename EpilogueOutputOp,
      
      /// Threadblock-level swizzling operator
      typename ThreadblockSwizzle,
      
      /// Number of stages used in the pipelined mainloop
      int Stages
    >
    class Gemm;
*/
template <
    StrassenType::ENUM StrassenKind,
    typename StrassenMiGroup,
    typename ScheduleStrassenGroups,
    /// Element type for A matrix operand
    typename ElementA_,
    /// Layout type for A matrix operand
    typename LayoutA_,
    /// Element type for B matrix operand
    typename ElementB_,
    /// Layout type for B matrix operand
    typename LayoutB_,
    /// Element type for C and D matrix operands
    typename ElementC_,
    /// Layout type for C and D matrix operands
    typename LayoutC_,
    /// Element type for internal accumulation
    typename ElementAccumulator_ = ElementC_,
    /// Operator class tag
    typename OperatorClass_ = arch::OpClassSimt,
    /// Tag indicating architecture to tune for
    typename ArchTag_ = arch::Sm70,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::InstructionShape,
    /// Epilogue output operator
    typename EpilogueOutputOp_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::EpilogueOutputOp,
    typename InterimEpilogueOp_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::EpilogueOutputOp,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle_ =
        typename threadblock::GemmIdentityThreadblockSwizzle<>,
    /// Number of stages used in the pipelined mainloop
    int Stages =
        DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_,
                                 ElementC_, ElementAccumulator_>::kStages,
    /// Access granularity of A matrix in units of elements
    int AlignmentA =
        DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_,
                                 ElementC_, ElementAccumulator_>::kAlignmentA,
    /// Access granularity of B matrix in units of elements
    int AlignmentB =
        DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_,
                                 ElementC_, ElementAccumulator_>::kAlignmentB,
    /// If true, kernel supports split-K with serial reduction
    bool SplitKSerial = false,
    bool SubGemmParallel = false,
    /// Operation performed by GEMM
    typename Operator_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::Operator,
    /// Gather operand A by using an index array
    bool GatherA = false,
    /// Gather operand B by using an index array
    bool GatherB = false,
    /// Scatter result D by using an index array
    bool ScatterD = false,
    /// Permute result D
    typename PermuteDLayout = layout::NoPermute>
class StrassenGemm//<, StrassenMiGroup, ElementA_, LayoutA_, ElementB_, LayoutB_, ElementC_, LayoutC_, ElementAccumulator_, OperatorClass_, ArchTag_, ThreadblockShape_, WarpShape_, InstructionShape_, EpilogueOutputOp_, ThreadblockSwizzle_, Stages, AlignmentA, AlignmentB, SplitKSerial, SubGemmParallel, Operator_, GatherA, GatherB, ScatterD, PermuteDLayout> 
  {
 public:
  using ElementA = ElementA_;
  using LayoutA = LayoutA_;
  using TensorRefA = TensorRef<ElementA const, LayoutA>;
  using ElementB = ElementB_;
  using LayoutB = LayoutB_;
  using TensorRefB = TensorRef<ElementB const, LayoutB>;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  using TensorRefC = TensorRef<ElementC const, LayoutC>;
  using TensorRefD = TensorRef<ElementC, LayoutC>;
  using ElementAccumulator = ElementAccumulator_;
  using OperatorClass = OperatorClass_;
  using ArchTag = ArchTag_;
  using ThreadblockShape = ThreadblockShape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  using EpilogueOutputOp = EpilogueOutputOp_;
  using InterimEpilogueOp = InterimEpilogueOp_;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  using Operator = Operator_;
  using StrassenGroups = StrassenMiGroup;
  static int const kStages = Stages;
  static int const kAlignmentA = AlignmentA;
  static int const kAlignmentB = AlignmentB;
  static int const kAlignmentC = EpilogueOutputOp::kCount;
  static bool const kSplitKSerial = SplitKSerial;
  static ComplexTransform const kTransformA = ComplexTransform::kNone;
  static ComplexTransform const kTransformB = ComplexTransform::kNone;

  /// Define the kernel
  template<MmaStrassen::Type MmaKind, typename StrassenMiGroup_, typename DefaultThreadBlockShape, typename DefaultWarpShape>
  using GemmKernel_ = typename kernel::DefaultStrassenGemm<
    ElementA,
    LayoutA,
    kAlignmentA,
    ElementB,
    LayoutB,
    kAlignmentB,
    ElementC,
    LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    typename std::conditional<StrassenMiGroup_::hasAnyM(), typename StrassenMiGroup_::ThreadBlockShape, DefaultThreadBlockShape>::type,
    typename std::conditional<StrassenMiGroup_::hasAnyM(), typename StrassenMiGroup_::WarpShape, DefaultWarpShape>::type,
    InstructionShape,
    MmaKind,
    StrassenMiGroup_,
    MmaStrassen::Consts<1,1,1,1>,
    EpilogueOutputOp,
    InterimEpilogueOp,
    ThreadblockSwizzle,
    kStages,
    kSplitKSerial,SubGemmParallel,
    Operator,
    SharedMemoryClearOption::kNone,
    GatherA,
    GatherB,
    ScatterD,
    PermuteDLayout
  >; 
  using GemmKernelM0 = typename GemmKernel_<(MmaStrassen::Type)StrassenKind, typename StrassenMiGroup::Group0, ThreadblockShape, WarpShape>::GemmKernel;
  using GemmKernelM1 = typename GemmKernel_<(MmaStrassen::Type)StrassenKind, typename StrassenMiGroup::Group1, ThreadblockShape, WarpShape>::GemmKernel;
  using GemmKernelM2 = typename GemmKernel_<(MmaStrassen::Type)StrassenKind, typename StrassenMiGroup::Group2, ThreadblockShape, WarpShape>::GemmKernel;
  using GemmKernelM3 = typename GemmKernel_<(MmaStrassen::Type)StrassenKind, typename StrassenMiGroup::Group3, ThreadblockShape, WarpShape>::GemmKernel;
  using GemmKernelM4 = typename GemmKernel_<(MmaStrassen::Type)StrassenKind, typename StrassenMiGroup::Group4, ThreadblockShape, WarpShape>::GemmKernel;
  using GemmKernelM5 = typename GemmKernel_<(MmaStrassen::Type)StrassenKind, typename StrassenMiGroup::Group5, ThreadblockShape, WarpShape>::GemmKernel;
  using GemmKernelM6 = typename GemmKernel_<(MmaStrassen::Type)StrassenKind, typename StrassenMiGroup::Group6, ThreadblockShape, WarpShape>::GemmKernel;

  // using ParallelMiKernels0 = MmaStrassen::ParallelMiKernels<typename ScheduleStrassenGroups::FusedMiGroup0,
  //                                                           GemmKernelM0, GemmKernelM1, GemmKernelM2, GemmKernelM3, GemmKernelM4, GemmKernelM5, GemmKernelM6>;
  // using ParallelMiKernels1 = MmaStrassen::ParallelMiKernels<typename ScheduleStrassenGroups::FusedMiGroup1,
  //                                                     GemmKernelM0, GemmKernelM1, GemmKernelM2, GemmKernelM3, GemmKernelM4, GemmKernelM5, GemmKernelM6>;
  // using ParallelMiKernels2 = MmaStrassen::ParallelMiKernels<typename ScheduleStrassenGroups::FusedMiGroup2,
  //                                                     GemmKernelM0, GemmKernelM1, GemmKernelM2, GemmKernelM3, GemmKernelM4, GemmKernelM5, GemmKernelM6>;
  // using ParallelMiKernels3 = MmaStrassen::ParallelMiKernels<typename ScheduleStrassenGroups::FusedMiGroup3,
  //                                                     GemmKernelM0, GemmKernelM1, GemmKernelM2, GemmKernelM3, GemmKernelM4, GemmKernelM5, GemmKernelM6>;
  // using ParallelMiKernels4 = MmaStrassen::ParallelMiKernels<typename ScheduleStrassenGroups::FusedMiGroup4,
  //                                                     GemmKernelM0, GemmKernelM1, GemmKernelM2, GemmKernelM3, GemmKernelM4, GemmKernelM5, GemmKernelM6>;
  // using ParallelMiKernels5 = MmaStrassen::ParallelMiKernels<typename ScheduleStrassenGroups::FusedMiGroup5,
  //                                                     GemmKernelM0, GemmKernelM1, GemmKernelM2, GemmKernelM3, GemmKernelM4, GemmKernelM5, GemmKernelM6>;
  // using ParallelMiKernels6 = MmaStrassen::ParallelMiKernels<typename ScheduleStrassenGroups::FusedMiGroup6,
  //                                                     GemmKernelM0, GemmKernelM1, GemmKernelM2, GemmKernelM3, GemmKernelM4, GemmKernelM5, GemmKernelM6>;

  /// Argument structure
  struct Arguments {

    //
    // Data members
    //

    GemmCoord problem_size;
    TensorRef<ElementA const, LayoutA> ref_A;
    TensorRef<ElementB const, LayoutB> ref_B;
    TensorRef<ElementC const, LayoutC> ref_C;
    TensorRef<ElementC, LayoutC> ref_D;
    TensorRef<ElementC const, LayoutC> ref_C2;
    TensorRef<ElementC, LayoutC> ref_D2;
    typename EpilogueOutputOp::Params epilogue;
    int split_k_slices;
    LayoutA layout_MA;
    LayoutB layout_MB;
    int level_1_idx;
    // For gather+scatter operations
    int const *gather_A_indices;
    int const *gather_B_indices;
    int const *scatter_D_indices;

    //
    // Methods
    //

    /// Default ctor
    CUTLASS_HOST_DEVICE
    Arguments(): problem_size(0, 0, 0), split_k_slices(1), level_1_idx(0) {

    }

    /// Constructs an Arguments structure
    CUTLASS_HOST_DEVICE
    Arguments(
      GemmCoord problem_size_,
      TensorRef<ElementA const, LayoutA> ref_A_,
      TensorRef<ElementB const, LayoutB> ref_B_,
      TensorRef<ElementC const, LayoutC> ref_C_,
      TensorRef<ElementC, LayoutC> ref_D_,
      typename EpilogueOutputOp::Params epilogue_ = 
        typename EpilogueOutputOp::Params(),
      int split_k_slices = 1,
      int const *gather_A_indices_ = nullptr,
      int const *gather_B_indices_ = nullptr,
      int const *scatter_D_indices_ = nullptr
    ): Arguments(problem_size_, ref_A_, ref_B_, ref_C_, ref_D_, {nullptr, ref_C_.layout()}, {nullptr, ref_D_.layout()},
                 epilogue_, split_k_slices, layoutforM(ref_A_), layoutforM(ref_B_), 0,
                 gather_A_indices_, gather_B_indices_, scatter_D_indices_) {}
 
    CUTLASS_HOST_DEVICE
    Arguments(
      GemmCoord problem_size_,
      TensorRef<ElementA const, LayoutA> ref_A_,
      TensorRef<ElementB const, LayoutB> ref_B_,
      TensorRef<ElementC const, LayoutC> ref_C_,
      TensorRef<ElementC, LayoutC> ref_D_,
      TensorRef<ElementC const, LayoutC> ref_C2_,
      TensorRef<ElementC, LayoutC> ref_D2_,
      typename EpilogueOutputOp::Params epilogue_,
      int split_k_slices,
      LayoutA layout_MA_,
      LayoutB layout_MB_,
      int level_1_idx,
      int const *gather_A_indices_ = nullptr,
      int const *gather_B_indices_ = nullptr,
      int const *scatter_D_indices_ = nullptr
    ):
      problem_size(problem_size_),
      ref_A(ref_A_),
      ref_B(ref_B_),
      ref_C(ref_C_),
      ref_D(ref_D_),
      ref_C2(ref_C2_),
      ref_D2(ref_D2_),
      epilogue(epilogue_),
      split_k_slices(split_k_slices),
      layout_MA(layout_MA_),
      layout_MB(layout_MB_),
      level_1_idx(level_1_idx),
      gather_A_indices(gather_A_indices_),
      gather_B_indices(gather_B_indices_),
      scatter_D_indices(scatter_D_indices_) {

    }
  };

private:

  /// Kernel parameters object
  typename GemmKernelM0::Params paramsM0_;
  typename GemmKernelM1::Params paramsM1_;
  typename GemmKernelM2::Params paramsM2_;
  typename GemmKernelM3::Params paramsM3_;
  typename GemmKernelM4::Params paramsM4_;
  typename GemmKernelM5::Params paramsM5_;
  typename GemmKernelM6::Params paramsM6_;

public:

  /// Constructs the GEMM.
  StrassenGemm() {
    /*int kM = GemmKernel_::Mma_::MmaCore::LaneMmaShape::kM;
    int kN = GemmKernel_::Mma_::MmaCore::LaneMmaShape::kN;
    printf("361: LaneMmaShape %d %d \n", kM, kN);
    using WarpCount = typename GemmKernel_::Mma_::MmaCore::WarpCount;
    printf("363: WarpCount %d %d\n", WarpCount::kM, WarpCount::kN);

    int WarpNumThreadsM = GemmKernel_::Mma_::MmaCore::WarpNumThreadsM;
    int WarpNumThreadsN = GemmKernel_::Mma_::MmaCore::WarpNumThreadsN;
    printf("367: %d %d\n", WarpNumThreadsM, WarpNumThreadsN);
    
    using WarpShape = typename GemmKernel_::Mma_::MmaCore::MmaWarpSimt::Shape;
    using PolicyWarpShape = typename GemmKernel_::Mma_::MmaCore::MmaWarpSimt::Policy::WarpShape;
    printf("368: %d %d; %d %d\n", WarpShape::kM, WarpShape::kN, PolicyWarpShape::kRow, PolicyWarpShape::kColumn);

    int MmaCorekThreads = GemmKernel_::Mma_::MmaCore::kThreads;
    printf("MmaCore::kThreads %d\n", MmaCorekThreads);

    using ThreadShape = typename GemmKernel_::Mma_::MmaCore::MmaWarpSimt::ThreadMma::Shape;
    printf("374: %d %d\n", ThreadShape::kM, ThreadShape::kN);

    using MmaCoreShape = typename GemmKernel_::Mma_::MmaCore::Shape;
    printf("377: %d %d %d\n", MmaCoreShape::kM, MmaCoreShape::kN, MmaCoreShape::kK);

    using ThreadMapA = typename GemmKernel_::Mma_::MmaCore::IteratorThreadMapA;
    printf("380: %d %d\n", ThreadMapA::Iterations::kCount, ThreadMapA::kElementsPerAccess);
    printf("381: %d %d %d\n", ThreadMapA::kThreads, ThreadMapA::Detail::ShapeVec::kContiguous, ThreadMapA::Detail::ShapeVec::kStrided);

    uint WarpCountkCount = GemmKernel::WarpCount::kCount;
    printf("384: WarpCountkCount %d\n", WarpCountkCount);*/
  }

  /// Determines whether the GEMM can execute the given problem.
  static Status can_implement(Arguments const &args) {

    if (!kSplitKSerial && args.split_k_slices > 1) {
      return Status::kErrorInvalidProblem;
    }

    Status status = GemmKernelM0::can_implement(
      args.problem_size,
      args.ref_A.non_const_ref(),
      args.ref_B.non_const_ref(),
      args.ref_C.non_const_ref(),
      args.ref_D
    );

    if (!GemmKernelM0::StrassenMiGroup::hasAllM()) {
      //Multiple kernels
      if (!GemmKernelM0::StrassenMiGroup::hasM0())
        return Status::kErrorInvalidProblem;
      
      bool has_A_presums = GemmKernelM0::StrassenMiGroup::AllPresums::computeAnyAPresum(MmaStrassen::PresumCompute);
      bool has_B_presums = GemmKernelM0::StrassenMiGroup::AllPresums::computeAnyBPresum(MmaStrassen::PresumCompute);
      const int presum_a_log_tile_multiplier = GemmKernelM0::get_presum_log_multiplier(args.problem_size.k(), args.problem_size.n());
      const int presum_b_log_tile_multiplier = GemmKernelM0::get_presum_log_multiplier(args.problem_size.k(), args.problem_size.m());

      const int total_presum_iterations = GemmKernelM0::Mma::kPresumComputeIterationsA*has_A_presums*(1<<presum_a_log_tile_multiplier) +
                                          GemmKernelM0::Mma::kPresumComputeIterationsB*has_B_presums*(1<<presum_b_log_tile_multiplier);
      int required_k = total_presum_iterations * GemmKernelM0::Mma::Shape::kK;
      if (args.problem_size.k()/2 < required_k)
        return Status::kErrorInvalidProblem;
    }

    if (status != Status::kSuccess) {
      return status;
    }

    return Status::kSuccess;
  }
  
  static size_t get_presum_a_workspace_size(Arguments const &args) {
    return (StrassenMiGroup::Group0::AllPresums::numAPresumYes() + StrassenMiGroup::PresumGroup::AllPresums::numAPresumYes()) *
            sizeof(ElementA) * size_t(args.problem_size.m()/2) * size_t(args.problem_size.k()/2);
  }

  static size_t get_presum_b_workspace_size(Arguments const &args) {
    return (StrassenMiGroup::Group0::AllPresums::numBPresumYes() + StrassenMiGroup::PresumGroup::AllPresums::numBPresumYes()) *
            sizeof(ElementB) * size_t(args.problem_size.n()/2) * size_t(args.problem_size.k()/2);
  }

  static size_t get_postsum_m_workspace_size(Arguments const &args) {
    return 7*(size_t(args.problem_size.m()/2) * size_t(args.problem_size.n()/2)) * sizeof(ElementC);
  }

  static size_t get_postsum_semaphore_size(Arguments const &args) {
    // Determine grid shape
    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord tiled_shape = threadblock_swizzle.get_tiled_shape(
      args.problem_size,
      {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
      args.split_k_slices);
    return 7*(size_t(tiled_shape.m()/2) * size_t(tiled_shape.n()/2)) * sizeof(int);
  }

  /// Gets the workspace size
  static size_t get_workspace_size(Arguments const &args) {
    
    size_t bytes = 0;

    // Determine grid shape
    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord tiled_shape = threadblock_swizzle.get_tiled_shape(
      args.problem_size,
      {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
      args.split_k_slices);

    bytes += get_presum_a_workspace_size(args) + get_presum_b_workspace_size(args) +
             get_postsum_m_workspace_size(args);

    bytes += get_postsum_semaphore_size(args);

    if (kSplitKSerial && args.split_k_slices > 1) {
      bytes += sizeof(int) * size_t(2*tiled_shape.m()) * size_t(2*tiled_shape.n()) * (SubGemmParallel ? 1 : 7);
    }
    return bytes;
  }

  
  /// Initializes GEMM state from arguments.
  template<typename GemmKernel>
  Status initialize(Arguments const &args, typename GemmKernel::Params& params, ElementA* presum_m_a, ElementB* presum_m_b,
                    ElementC* postsum_m, int* m_sync_semaphore,
                    int*& sem_workspace, cudaStream_t stream = nullptr) {
    // Determine grid shape
    ThreadblockSwizzle threadblock_swizzle;
    
    auto new_problem_size = GemmCoord{args.problem_size.m()/2 * GemmKernel::StrassenMiGroup::GridMMultiplier(),
                                      args.problem_size.n()/2 * GemmKernel::StrassenMiGroup::GridNMultiplier(),
                                      args.problem_size.k()};
    using KernelThreadBlockShape = typename GemmKernel::StrassenMiGroup::ThreadBlockShape;
    cutlass::gemm::GemmCoord grid_shape = threadblock_swizzle.get_tiled_shape(
      new_problem_size,
      {KernelThreadBlockShape::kM, KernelThreadBlockShape::kN, KernelThreadBlockShape::kK},
      args.split_k_slices);

    cutlass::gemm::GemmCoord orig_grid_shape = threadblock_swizzle.get_tiled_shape(
      args.problem_size,
      {KernelThreadBlockShape::kM, KernelThreadBlockShape::kN, KernelThreadBlockShape::kK},
      args.split_k_slices);

    // Initialize the Params structure
    params = typename GemmKernel::Params{
      args.problem_size,
      grid_shape,
      args.ref_A.non_const_ref(),
      args.ref_B.non_const_ref(),
      args.ref_C.non_const_ref(),
      args.ref_D,
      args.layout_MA,
      args.layout_MB,
      args.level_1_idx,
      args.ref_C2.non_const_ref(), args.ref_D2,
      args.epilogue,
      presum_m_a,
      presum_m_b,
      postsum_m,
      m_sync_semaphore,
      static_cast<int *>(sem_workspace),
      args.gather_A_indices,
      args.gather_B_indices,
      args.scatter_D_indices
    };

    sem_workspace += 1*(orig_grid_shape.m() * orig_grid_shape.n());

    return Status::kSuccess;
  }
  
  static ElementA* get_presum_a_ptr(Arguments const &args, void *workspace = nullptr) {
    return (ElementA*)workspace;
  }

  static ElementB* get_presum_b_ptr(Arguments const &args, void *workspace = nullptr) {
    ElementA* ptr = (ElementA*)get_presum_a_ptr(args, workspace) + get_presum_a_workspace_size(args)/sizeof(ElementA);
    return (ElementA*)ptr;
  }

  static ElementC* get_postsum_m_ptr(Arguments const &args, void *workspace = nullptr) {
    ElementC* ptr = (ElementA*)get_presum_b_ptr(args, workspace) + get_presum_b_workspace_size(args)/sizeof(ElementA);
    return (ElementC*)ptr;
  }

  Status initialize(Arguments const &args, void *workspace = nullptr, cudaStream_t stream = nullptr) {
    ElementA* presum_a_workspace = (ElementA*)workspace;
    ElementA* presum_b_workspace = (ElementA*)presum_a_workspace + get_presum_a_workspace_size(args)/sizeof(ElementA);
    ElementC* postsum_m_workspace = (ElementC*)presum_b_workspace + get_presum_b_workspace_size(args)/sizeof(ElementB);
    int* postsum_semaphore = (int*)(postsum_m_workspace + get_postsum_m_workspace_size(args)/sizeof(ElementC));
    int* sem_workspace = (int*)(postsum_semaphore + get_postsum_semaphore_size(args)/sizeof(int));
    
    if (SubGemmParallel || kSplitKSerial) {
      if (SubGemmParallel || args.split_k_slices > 1) {
        if (!workspace) {
          return Status::kErrorWorkspaceNull;
        }

        size_t bytes = get_workspace_size(args) - get_presum_a_workspace_size(args) - get_presum_b_workspace_size(args) -
                       get_postsum_m_workspace_size(args);
        cudaError_t result = cudaMemsetAsync(postsum_semaphore, 0, bytes, stream);

        if (result != cudaSuccess) {
          return Status::kErrorInternal;
        }
      }
    }
    else {
      if (args.split_k_slices > 1) {
        return Status::kErrorInvalidProblem;
      }
    }

    Status err;
    err = initialize<GemmKernelM0>(args, paramsM0_, presum_a_workspace, presum_b_workspace, postsum_m_workspace, postsum_semaphore, sem_workspace, stream);
    if (err == Status::kErrorInternal) return err;
    err = initialize<GemmKernelM1>(args, paramsM1_, presum_a_workspace, presum_b_workspace, postsum_m_workspace, postsum_semaphore, sem_workspace, stream);
    if (err == Status::kErrorInternal) return err;
    err = initialize<GemmKernelM2>(args, paramsM2_, presum_a_workspace, presum_b_workspace, postsum_m_workspace, postsum_semaphore, sem_workspace, stream);
    if (err == Status::kErrorInternal) return err;
    err = initialize<GemmKernelM3>(args, paramsM3_, presum_a_workspace, presum_b_workspace, postsum_m_workspace, postsum_semaphore, sem_workspace, stream);
    if (err == Status::kErrorInternal) return err;
    err = initialize<GemmKernelM4>(args, paramsM4_, presum_a_workspace, presum_b_workspace, postsum_m_workspace, postsum_semaphore, sem_workspace, stream);
    if (err == Status::kErrorInternal) return err;
    err = initialize<GemmKernelM5>(args, paramsM5_, presum_a_workspace, presum_b_workspace, postsum_m_workspace, postsum_semaphore, sem_workspace, stream);
    if (err == Status::kErrorInternal) return err;
    err = initialize<GemmKernelM6>(args, paramsM6_, presum_a_workspace, presum_b_workspace, postsum_m_workspace, postsum_semaphore, sem_workspace, stream);
    if (err == Status::kErrorInternal) return err;
    return Status::kSuccess;
  }

  /// Lightweight update given a subset of arguments
  Status update(Arguments const &args, void *workspace = nullptr) {
    assert(false);
    if (kSplitKSerial && args.split_k_slices > 1) {
      if (!workspace) {
        return Status::kErrorWorkspaceNull;
      }
    }

    paramsM0_.ref_A.reset(args.ref_A.non_const_ref().data());
    paramsM0_.ref_B.reset(args.ref_B.non_const_ref().data());
    paramsM0_.ref_C.reset(args.ref_C.non_const_ref().data());
    paramsM0_.ref_D.reset(args.ref_D.data());
    paramsM0_.output_op = args.epilogue;
    paramsM0_.semaphore = static_cast<int *>(workspace);


    paramsM1_.ref_A.reset(args.ref_A.non_const_ref().data());
    paramsM1_.ref_B.reset(args.ref_B.non_const_ref().data());
    paramsM1_.ref_C.reset(args.ref_C.non_const_ref().data());
    paramsM1_.ref_D.reset(args.ref_D.data());
    paramsM1_.output_op = args.epilogue;
    paramsM1_.semaphore = static_cast<int *>(workspace);


    paramsM2_.ref_A.reset(args.ref_A.non_const_ref().data());
    paramsM2_.ref_B.reset(args.ref_B.non_const_ref().data());
    paramsM2_.ref_C.reset(args.ref_C.non_const_ref().data());
    paramsM2_.ref_D.reset(args.ref_D.data());
    paramsM2_.output_op = args.epilogue;
    paramsM2_.semaphore = static_cast<int *>(workspace);


    paramsM3_.ref_A.reset(args.ref_A.non_const_ref().data());
    paramsM3_.ref_B.reset(args.ref_B.non_const_ref().data());
    paramsM3_.ref_C.reset(args.ref_C.non_const_ref().data());
    paramsM3_.ref_D.reset(args.ref_D.data());
    paramsM3_.output_op = args.epilogue;
    paramsM3_.semaphore = static_cast<int *>(workspace);

    paramsM4_.ref_A.reset(args.ref_A.non_const_ref().data());
    paramsM4_.ref_B.reset(args.ref_B.non_const_ref().data());
    paramsM4_.ref_C.reset(args.ref_C.non_const_ref().data());
    paramsM4_.ref_D.reset(args.ref_D.data());
    paramsM4_.output_op = args.epilogue;
    paramsM4_.semaphore = static_cast<int *>(workspace);

    paramsM5_.ref_A.reset(args.ref_A.non_const_ref().data());
    paramsM5_.ref_B.reset(args.ref_B.non_const_ref().data());
    paramsM5_.ref_C.reset(args.ref_C.non_const_ref().data());
    paramsM5_.ref_D.reset(args.ref_D.data());
    paramsM5_.output_op = args.epilogue;
    paramsM5_.semaphore = static_cast<int *>(workspace);

    paramsM6_.ref_A.reset(args.ref_A.non_const_ref().data());
    paramsM6_.ref_B.reset(args.ref_B.non_const_ref().data());
    paramsM6_.ref_C.reset(args.ref_C.non_const_ref().data());
    paramsM6_.ref_D.reset(args.ref_D.data());
    paramsM6_.output_op = args.epilogue;
    paramsM6_.semaphore = static_cast<int *>(workspace);

    return Status::kSuccess;
  }


  template<typename FusedOperator1, typename FusedOperator2, typename FusedOperator3,
           typename FusedOperator4, typename FusedOperator5, typename FusedOperator6,
           typename FusedOperator7>
  cudaError_t run(typename FusedOperator1::Operator1::Params& params1,
                  typename FusedOperator2::Operator1::Params& params2,
                  typename FusedOperator3::Operator1::Params& params3,
                  typename FusedOperator4::Operator1::Params& params4,
                  typename FusedOperator5::Operator1::Params& params5,
                  typename FusedOperator6::Operator1::Params& params6,
                  typename FusedOperator7::Operator1::Params& params7,
                  cudaStream_t stream = nullptr) {
    ThreadblockSwizzle threadblock_swizzle;

    dim3 grid = threadblock_swizzle.get_grid_shape(params1.grid_tiled_shape);
    dim3 block(FusedOperator1::Operator1::kThreadCount, 1, 1);
    params1.run += 1; params2.run += 1; params3.run += 1; params4.run += 1; params5.run += 1;
    params6.run += 1; params7.run += 1;
    cudaError_t result;
    int smem_size = int(sizeof(typename FusedOperator1::Operator1::SharedStorage));
    // int epilogue_size = int(sizeof(typename GemmKernel::Epilogue::SharedStorage));
    dim3 origGrid = grid;
    grid.z = 7*grid.z;
    // std::cout << "504 " << smem_size << " " << GemmKernel::kThreadCount << " " << grid.x << " " << grid.y << " " << grid.z << " " << epilogue_size <<std::endl;
    if (smem_size >= (48 << 10)) {
        result = cudaFuncSetAttribute(Kernel7<FusedOperator1, FusedOperator2, FusedOperator3, FusedOperator4, FusedOperator5, FusedOperator6, FusedOperator7>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      smem_size);

      if (result != cudaSuccess) {
        return result;
      }
    }

    cutlass::Kernel7<FusedOperator1, FusedOperator2, FusedOperator3, FusedOperator4, FusedOperator5, FusedOperator6, FusedOperator7><<<grid, block, smem_size, stream>>>(params1, params2, params3, params4, params5, params6, params7, origGrid);
    result = cudaGetLastError();
    return result;
  }

  template<typename FusedOperator1, typename FusedOperator2, typename FusedOperator3,
           typename FusedOperator4, typename FusedOperator5>
  cudaError_t run(typename FusedOperator1::Operator1::Params& params1,
                  typename FusedOperator2::Operator1::Params& params2,
                  typename FusedOperator3::Operator1::Params& params3,
                  typename FusedOperator4::Operator1::Params& params4,
                  typename FusedOperator5::Operator1::Params& params5,
                  cudaStream_t stream = nullptr) {
    ThreadblockSwizzle threadblock_swizzle;

    dim3 grid = threadblock_swizzle.get_grid_shape(params1.grid_tiled_shape);
    dim3 block(FusedOperator1::Operator1::kThreadCount, 1, 1);
    params1.run += 1; params2.run += 1; params3.run += 1; params4.run += 1; params5.run += 1;
    cudaError_t result;
    int smem_size = int(sizeof(typename FusedOperator1::Operator1::SharedStorage));
    // int epilogue_size = int(sizeof(typename GemmKernel::Epilogue::SharedStorage));
    dim3 origGrid = grid;
    grid.z = 5*grid.z;
    // std::cout << "504 " << smem_size << " " << GemmKernel::kThreadCount << " " << grid.x << " " << grid.y << " " << grid.z << " " << epilogue_size <<std::endl;
    if (smem_size >= (48 << 10)) {
        result = cudaFuncSetAttribute(Kernel5<FusedOperator1, FusedOperator2, FusedOperator3, FusedOperator4, FusedOperator5>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      smem_size);

      if (result != cudaSuccess) {
        return result;
      }
    }

    cutlass::Kernel5<FusedOperator1, FusedOperator2, FusedOperator3, FusedOperator4, FusedOperator5><<<grid, block, smem_size, stream>>>(params1, params2, params3, params4, params5, origGrid);
    result = cudaGetLastError();
    return result;
  }

  template<typename FusedOperator1, typename FusedOperator2, typename FusedOperator3, typename FusedOperator4>
  cudaError_t run(typename FusedOperator1::Operator1::Params& params1,
                  typename FusedOperator2::Operator1::Params& params2,
                  typename FusedOperator3::Operator1::Params& params3,
                  typename FusedOperator4::Operator1::Params& params4,
                  cudaStream_t stream = nullptr) {
    ThreadblockSwizzle threadblock_swizzle;

    dim3 grid = threadblock_swizzle.get_grid_shape(params1.grid_tiled_shape);
    dim3 block(FusedOperator1::Operator1::kThreadCount, 1, 1);
    params1.run += 1; params2.run += 1; params3.run += 1; params4.run += 1;
    cudaError_t result;
    int smem_size = int(sizeof(typename FusedOperator1::Operator1::SharedStorage));
    // int epilogue_size = int(sizeof(typename GemmKernel::Epilogue::SharedStorage));
    dim3 origGrid = grid;
    grid.z = 4*grid.z;
    // std::cout << "504 " << smem_size << " " << GemmKernel::kThreadCount << " " << grid.x << " " << grid.y << " " << grid.z << " " << epilogue_size <<std::endl;
    if (smem_size >= (48 << 10)) {
        result = cudaFuncSetAttribute(Kernel4<FusedOperator1, FusedOperator2, FusedOperator3, FusedOperator4>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      smem_size);

      if (result != cudaSuccess) {
        return result;
      }
    }

    cutlass::Kernel4<FusedOperator1, FusedOperator2, FusedOperator3, FusedOperator4><<<grid, block, smem_size, stream>>>(params1, params2, params3, params4, origGrid);
    result = cudaGetLastError();
    return result;
  }

  template<typename FusedOperator1, typename FusedOperator2, typename FusedOperator3>
  cudaError_t run(typename FusedOperator1::Operator1::Params& params1,
                  typename FusedOperator2::Operator1::Params& params2,
                  typename FusedOperator3::Operator1::Params& params3,
                  cudaStream_t stream = nullptr) {
    ThreadblockSwizzle threadblock_swizzle;

    dim3 grid = threadblock_swizzle.get_grid_shape(params1.grid_tiled_shape);
    dim3 block(FusedOperator1::Operator1::kThreadCount, 1, 1);
    params1.run += 1; params2.run += 1; params3.run += 1;
    cudaError_t result;
    int smem_size = int(sizeof(typename FusedOperator1::Operator1::SharedStorage));
    // int epilogue_size = int(sizeof(typename GemmKernel::Epilogue::SharedStorage));
    dim3 origGrid = grid;
    grid.z = 3*grid.z;
    // std::cout << "504 " << smem_size << " " << GemmKernel::kThreadCount << " " << grid.x << " " << grid.y << " " << grid.z << " " << epilogue_size <<std::endl;
    if (smem_size >= (48 << 10)) {
        result = cudaFuncSetAttribute(Kernel3<FusedOperator1, FusedOperator2, FusedOperator3>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      smem_size);

      if (result != cudaSuccess) {
        return result;
      }
    }

    cutlass::Kernel3<FusedOperator1, FusedOperator2, FusedOperator3><<<grid, block, smem_size, stream>>>(params1, params2, params3, origGrid);
    result = cudaGetLastError();
    return result;
  }

  template<typename FusedOperator1, typename FusedOperator2>
  cudaError_t run(typename FusedOperator1::Operator1::Params& params1,
                  typename FusedOperator2::Operator1::Params& params2,
                  cudaStream_t stream = nullptr) {
    ThreadblockSwizzle threadblock_swizzle;

    dim3 grid = threadblock_swizzle.get_grid_shape(params1.grid_tiled_shape);
    dim3 block(FusedOperator1::Operator1::kThreadCount, 1, 1);
    params1.run += 1; params2.run += 1;
    cudaError_t result;
    int smem_size = int(sizeof(typename FusedOperator1::Operator1::SharedStorage));
    // int epilogue_size = int(sizeof(typename GemmKernel::Epilogue::SharedStorage));
    dim3 origGrid = grid;
    grid.z = 2*grid.z;
    // std::cout << "504 " << smem_size << " " << grid.x << " " << grid.y << " " << grid.z << " " << std::endl;
    if (smem_size >= (48 << 10)) {
        result = cudaFuncSetAttribute(Kernel2<FusedOperator1, FusedOperator2>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      smem_size);

      if (result != cudaSuccess) {
        return result;
      }
    }

    cutlass::Kernel2<FusedOperator1, FusedOperator2><<<grid, block, smem_size, stream>>>(params1, params2, origGrid);
    result = cudaGetLastError();
    return result;
  }

  template<typename ParallelMiKernels>
  cudaError_t run_parallel_kernels(cudaStream_t stream = nullptr) {
    ThreadblockSwizzle threadblock_swizzle;
    
    ParallelMiKernels parallel_kernels(paramsM0_, paramsM1_, paramsM2_, paramsM3_,
                                       paramsM4_, paramsM5_, paramsM6_);
    dim3 grid = threadblock_swizzle.get_grid_shape(parallel_kernels.grid_single_tiled_shape());
    dim3 block(ParallelMiKernels::ThreadCount(), 1, 1);

    cudaError_t result;
    dim3 origGrid = grid;
    grid.z = ParallelMiKernels::NumKernels()*grid.z;
    int smem_size = ParallelMiKernels::SharedStorageSize();
    // int epilogue_size = int(sizeof(typename ParallelMiKernels::GemmKernel0_::Mma::SharedStorage));
    // std::cout << "504 " << smem_size << " " << grid.x << " " << grid.y << " " << grid.z <<std::endl;
    if (smem_size >= (48 << 10)) {
        result = cudaFuncSetAttribute(KernelParallelMiGroup<ParallelMiKernels>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      smem_size);

      if (result != cudaSuccess) {
        return result;
      }
    }

    cutlass::KernelParallelMiGroup<ParallelMiKernels><<<grid, block, smem_size, stream>>>(parallel_kernels, origGrid);
    result = cudaGetLastError();
    // result = cudaDeviceSynchronize();
    return result;
  }

  template<typename GemmKernel>
  cudaError_t run(typename GemmKernel::Params& params, cudaStream_t stream = nullptr) {
    ThreadblockSwizzle threadblock_swizzle;

    dim3 grid = threadblock_swizzle.get_grid_shape(params.grid_tiled_shape);

    dim3 block(GemmKernel::kThreadCount, 1, 1);

    cudaError_t result;
    int smem_size = int(sizeof(typename GemmKernel::SharedStorage));
    // std::cout << "504 " << smem_size << " " << GemmKernel::kThreadCount << " " << grid.x << " " << grid.y << " " << grid.z << " " << stream << std::endl;
    if (smem_size >= (48 << 10)) {
      result = cudaFuncSetAttribute(Kernel<GemmKernel>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    smem_size);

      if (result != cudaSuccess) {
        return result;
      }
    }

    cutlass::Kernel<GemmKernel><<<grid, block, smem_size, stream>>>(params);
    result = cudaGetLastError();
    return result;
  }

  /// Runs the kernel using initialized state.
  Status run(cudaStream_t* streams = nullptr, int num_streams = 0) {
    cudaError_t result;

    if (false && SubGemmParallel) {//TODO:What to do here
      cudaStream_t defaultStreams[7] = {0};
      if (streams == nullptr || num_streams == 0) {
        streams = defaultStreams;
      } else {
        assert (num_streams >= 7);
      }
      // result = run<GemmKernelM0>(paramsM0_, streams[0]);
      // if (result != cudaSuccess)
      // {printf("Error at %d: %s\n", __LINE__, cudaGetErrorString(result)); return Status::kErrorInternal;}

      // result = run<GemmKernelM1>(paramsM1_, streams[1]);
      // if (result != cudaSuccess)
      // {printf("Error at %d: %s\n", __LINE__, cudaGetErrorString(result)); return Status::kErrorInternal;}

      // result = run<GemmKernelM2>(paramsM2_, streams[2]);
      // if (result != cudaSuccess)
      // {printf("Error at %d: %s\n", __LINE__, cudaGetErrorString(result)); return Status::kErrorInternal;}

      // result = run<GemmKernelM3>(paramsM3_, streams[3]);
      // if (result != cudaSuccess)
      // {printf("Error at %d: %s\n", __LINE__, cudaGetErrorString(result)); return Status::kErrorInternal;}

      // result = run<GemmKernelM4>(paramsM4_, streams[4]);
      // if (result != cudaSuccess)
      // {printf("Error at %d: %s\n", __LINE__, cudaGetErrorString(result)); return Status::kErrorInternal;}

      // result = run<GemmKernelM5>(paramsM5_, streams[5]);
      // if (result != cudaSuccess)
      // {printf("Error at %d: %s\n", __LINE__, cudaGetErrorString(result)); return Status::kErrorInternal;}

      // result = run<GemmKernelM6>(paramsM6_, streams[6]);
      // if (result != cudaSuccess)
      // {printf("Error at %d: %s\n", __LINE__, cudaGetErrorString(result)); return Status::kErrorInternal;}
    } else {
      cudaStream_t streams2[7]; 
      if (num_streams == 1) {
        for (int i = 0; i < 7; i++)
          streams2[i] = streams[0];
      } else {
        for (int i = 0; i < 7; i++)
          streams2[i] = streams[i];
      }

      using ParallelGroup0 = ParallelMiKernels<typename ScheduleStrassenGroups::ParallelGroups0,
                                               GemmKernelM0, GemmKernelM1, GemmKernelM2, GemmKernelM3,
                                               GemmKernelM4, GemmKernelM5, GemmKernelM6>;
      using ParallelGroup1 = ParallelMiKernels<typename ScheduleStrassenGroups::ParallelGroups1,
                                               GemmKernelM0, GemmKernelM1, GemmKernelM2, GemmKernelM3,
                                               GemmKernelM4, GemmKernelM5, GemmKernelM6>;
      using ParallelGroup2 = ParallelMiKernels<typename ScheduleStrassenGroups::ParallelGroups2,
                                               GemmKernelM0, GemmKernelM1, GemmKernelM2, GemmKernelM3,
                                               GemmKernelM4, GemmKernelM5, GemmKernelM6>;
      using ParallelGroup3 = ParallelMiKernels<typename ScheduleStrassenGroups::ParallelGroups3,
                                               GemmKernelM0, GemmKernelM1, GemmKernelM2, GemmKernelM3,
                                               GemmKernelM4, GemmKernelM5, GemmKernelM6>;
      using ParallelGroup4 = ParallelMiKernels<typename ScheduleStrassenGroups::ParallelGroups4,
                                               GemmKernelM0, GemmKernelM1, GemmKernelM2, GemmKernelM3,
                                               GemmKernelM4, GemmKernelM5, GemmKernelM6>;
      using ParallelGroup5 = ParallelMiKernels<typename ScheduleStrassenGroups::ParallelGroups5,
                                               GemmKernelM0, GemmKernelM1, GemmKernelM2, GemmKernelM3,
                                               GemmKernelM4, GemmKernelM5, GemmKernelM6>;
      using ParallelGroup6 = ParallelMiKernels<typename ScheduleStrassenGroups::ParallelGroups6,
                                               GemmKernelM0, GemmKernelM1, GemmKernelM2, GemmKernelM3,
                                               GemmKernelM4, GemmKernelM5, GemmKernelM6>;

      if (StrassenMiGroup::PresumGroup::AllPresums::APresumComputeLoads(PresumGlobalKernel).numAccess() > 0 ||
          StrassenMiGroup::PresumGroup::AllPresums::BPresumComputeLoads(PresumGlobalKernel).numAccess() > 0) {
        dim3 grid = {(uint)paramsM0_.grid_tiled_shape.n(),
                     (uint)paramsM0_.grid_tiled_shape.m(),
                     1};
        KernelPresumGlobalCompute<typename StrassenMiGroup::PresumGroup, GemmKernelM0, GemmKernelM0::kThreadCount>
          <<<grid, GemmKernelM0::kThreadCount, 0, streams2[0]>>>(paramsM0_);
        result = cudaStreamSynchronize(streams2[0]);
        if (result != cudaSuccess)
        {printf("Error at %d: %s\n", __LINE__, cudaGetErrorString(result)); return Status::kErrorInternal;}
      }

      if (ParallelGroup0::HasAKernel()) {
        result = run_parallel_kernels<ParallelGroup0>(streams2[0]);
        if (result != cudaSuccess)
        {printf("Error at %d: %s\n", __LINE__, cudaGetErrorString(result)); return Status::kErrorInternal;}
      }
      
      // if (StrassenMiGroup::Group0::Level1Idx == 0) {
        // presumcheck<<<2048,1024,0,streams[0]>>>(paramsM0_.ref_A.data(), paramsM0_.presum_m_a_workspace);
        // postsumcheck<<<1,1,0,streams[0]>>>(paramsM0_.postsum_m_workspace);
        // cudaDeviceSynchronize();
      // }
      // exit(EXIT_SUCCESS);
      
      if (ParallelGroup1::HasAKernel()) {
        result = run_parallel_kernels<ParallelGroup1>(streams2[1]);
        if (result != cudaSuccess)
        {printf("Error at %d: %s\n", __LINE__, cudaGetErrorString(result)); return Status::kErrorInternal;}
      }

      if (ParallelGroup2::HasAKernel()) {
        result = run_parallel_kernels<ParallelGroup2>(streams2[2]);
        if (result != cudaSuccess)
        {printf("Error at %d: %s\n", __LINE__, cudaGetErrorString(result)); return Status::kErrorInternal;}
      }

      if (ParallelGroup3::HasAKernel()) {
        result = run_parallel_kernels<ParallelGroup3>(streams2[3]);
        if (result != cudaSuccess)
        {printf("Error at %d: %s\n", __LINE__, cudaGetErrorString(result)); return Status::kErrorInternal;}
      }

      if (ParallelGroup4::HasAKernel()) {
        result = run_parallel_kernels<ParallelGroup4>(streams2[4]);
        if (result != cudaSuccess)
        {printf("Error at %d: %s\n", __LINE__, cudaGetErrorString(result)); return Status::kErrorInternal;}
      }

      if (ParallelGroup5::HasAKernel()) {
        result = run_parallel_kernels<ParallelGroup5>(streams2[5]);
        if (result != cudaSuccess)
        {printf("Error at %d: %s\n", __LINE__, cudaGetErrorString(result)); return Status::kErrorInternal;}
      }

      if (ParallelGroup6::HasAKernel()) {
        result = run_parallel_kernels<ParallelGroup6>(streams2[6]);
        if (result != cudaSuccess)
        {printf("Error at %d: %s\n", __LINE__, cudaGetErrorString(result)); return Status::kErrorInternal;}
      }

      paramsM0_.run += 1; paramsM1_.run += 1; paramsM2_.run += 1; paramsM3_.run += 1;
      paramsM4_.run += 1; paramsM5_.run += 1; paramsM6_.run += 1;

      // if (FusedMiKernels1::HasAKernel()) {
      //   FusedMiKernels1 fusedMiKernels(paramsM0_, paramsM1_, paramsM2_, paramsM3_,
      //                                  paramsM4_, paramsM5_, paramsM6_);
      //   result = run<FusedMiKernels1>(fusedMiKernels, streams[1]);
      //   if (result != cudaSuccess)
      //   {printf("Error at %d: %s\n", __LINE__, cudaGetErrorString(result)); return Status::kErrorInternal;}
      // }

      // if (FusedMiKernels2::HasAKernel()) {
      //   FusedMiKernels2 fusedMiKernels(paramsM0_, paramsM1_, paramsM2_, paramsM3_,
      //                                  paramsM4_, paramsM5_, paramsM6_);
      //   result = run<FusedMiKernels2>(fusedMiKernels, streams[2]);
      //   if (result != cudaSuccess)
      //   {printf("Error at %d: %s\n", __LINE__, cudaGetErrorString(result)); return Status::kErrorInternal;}
      // }

      // if (FusedMiKernels3::HasAKernel()) {
      //   FusedMiKernels3 fusedMiKernels(paramsM0_, paramsM1_, paramsM2_, paramsM3_,
      //                                  paramsM4_, paramsM5_, paramsM6_);
      //   result = run<FusedMiKernels3>(fusedMiKernels, streams[3]);
      //   if (result != cudaSuccess)
      //   {printf("Error at %d: %s\n", __LINE__, cudaGetErrorString(result)); return Status::kErrorInternal;}
      // }

      // if (FusedMiKernels4::HasAKernel()) {
      //   FusedMiKernels4 fusedMiKernels(paramsM0_, paramsM1_, paramsM2_, paramsM3_,
      //                                  paramsM4_, paramsM5_, paramsM6_);
      //   result = run<FusedMiKernels4>(fusedMiKernels, streams[4]);
      //   if (result != cudaSuccess)
      //   {printf("Error at %d: %s\n", __LINE__, cudaGetErrorString(result)); return Status::kErrorInternal;}
      // }

      // if (FusedMiKernels5::HasAKernel()) {
      //   FusedMiKernels5 fusedMiKernels(paramsM0_, paramsM1_, paramsM2_, paramsM3_,
      //                                  paramsM4_, paramsM5_, paramsM6_);
      //   result = run<FusedMiKernels5>(fusedMiKernels, streams[5]);
      //   if (result != cudaSuccess)
      //   {printf("Error at %d: %s\n", __LINE__, cudaGetErrorString(result)); return Status::kErrorInternal;}
      // }

      // if (FusedMiKernels6::HasAKernel()) {
      //   FusedMiKernels6 fusedMiKernels(paramsM0_, paramsM1_, paramsM2_, paramsM3_,
      //                                  paramsM4_, paramsM5_, paramsM6_);
      //   result = run<FusedMiKernels6>(fusedMiKernels, streams[6]);
      //   if (result != cudaSuccess)
      //   {printf("Error at %d: %s\n", __LINE__, cudaGetErrorString(result)); return Status::kErrorInternal;}
      // }

      // if (stream2 != stream1) {
      //   result = cudaStreamSynchronize(stream1);
      //   if (result != cudaSuccess)
      //   {printf("Error at %d: %s\n", __LINE__, cudaGetErrorString(result)); return Status::kErrorInternal;}
      //   result = cudaStreamSynchronize(stream2);
      //   if (result != cudaSuccess)
      //   {printf("Error at %d: %s\n", __LINE__, cudaGetErrorString(result)); return Status::kErrorInternal;}
      // }

     
      // if (stream2 != stream1) {
      //   result = cudaStreamSynchronize(stream1);
      //   if (result != cudaSuccess)
      //   {printf("Error at %d: %s\n", __LINE__, cudaGetErrorString(result)); return Status::kErrorInternal;}
      //   result = cudaStreamSynchronize(stream2);
      //   if (result != cudaSuccess)
      //   {printf("Error at %d: %s\n", __LINE__, cudaGetErrorString(result)); return Status::kErrorInternal;}
      // }
    }

    return Status::kSuccess;
  }

  /// Runs the kernel using initialized state.
  // Status operator()(cudaStream_t stream = nullptr) {
  //   return run(stream);
  // }

  /// Runs the kernel using initialized state.
  Status operator()(
    Arguments const &args,
    void *workspace = nullptr,
    cudaStream_t* streams = nullptr,
    int num_streams = 0) {
    cudaStream_t init_stream;

    if (streams == nullptr || num_streams == 0) {
      init_stream = nullptr;
    } else {
      init_stream = streams[0];
    }

    Status status = initialize(args, workspace, init_stream);

    if (status == Status::kSuccess) {
      status = run(streams, num_streams);
    }

    return status;
  }
};



////////////////////////////////////////////////////////////////////////////////

template <
    StrassenType::ENUM StrassenKind,
    typename StrassenMiGroupM0,
    typename StrassenMiGroupM1,
    typename StrassenMiGroupM2,
    typename StrassenMiGroupM3,
    typename StrassenMiGroupM4,
    typename StrassenMiGroupM5,
    typename StrassenMiGroupM6,
    typename ScheduleStrassenGroups,
    /// Element type for A matrix operand
    typename ElementA_,
    /// Layout type for A matrix operand
    typename LayoutA_,
    /// Element type for B matrix operand
    typename ElementB_,
    /// Layout type for B matrix operand
    typename LayoutB_,
    /// Element type for C and D matrix operands
    typename ElementC_,
    /// Layout type for C and D matrix operands
    typename LayoutC_,
    /// Element type for internal accumulation
    typename ElementAccumulator_ = ElementC_,
    /// Operator class tag
    typename OperatorClass_ = arch::OpClassSimt,
    /// Tag indicating architecture to tune for
    typename ArchTag_ = arch::Sm70,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::InstructionShape,
    /// Epilogue output operator
    typename EpilogueOutputOp_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::EpilogueOutputOp,
    typename InterimEpilogueOp_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::EpilogueOutputOp,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle_ =
        typename threadblock::GemmIdentityThreadblockSwizzle<>,
    /// Number of stages used in the pipelined mainloop
    int Stages =
        DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_,
                                 ElementC_, ElementAccumulator_>::kStages,
    /// Access granularity of A matrix in units of elements
    int AlignmentA =
        DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_,
                                 ElementC_, ElementAccumulator_>::kAlignmentA,
    /// Access granularity of B matrix in units of elements
    int AlignmentB =
        DefaultGemmConfiguration<OperatorClass_, ArchTag_, ElementA_, ElementB_,
                                 ElementC_, ElementAccumulator_>::kAlignmentB,
    /// If true, kernel supports split-K with serial reduction
    bool SplitKSerial = false,
    bool SubGemmParallel = false,
    /// Operation performed by GEMM
    typename Operator_ = typename DefaultGemmConfiguration<
        OperatorClass_, ArchTag_, ElementA_, ElementB_, ElementC_,
        ElementAccumulator_>::Operator,
    /// Gather operand A by using an index array
    bool GatherA = false,
    /// Gather operand B by using an index array
    bool GatherB = false,
    /// Scatter result D by using an index array
    bool ScatterD = false,
    /// Permute result D
    typename PermuteDLayout = layout::NoPermute>
class StrassenGemmLevel2//<, StrassenMiGroup, ElementA_, LayoutA_, ElementB_, LayoutB_, ElementC_, LayoutC_, ElementAccumulator_, OperatorClass_, ArchTag_, ThreadblockShape_, WarpShape_, InstructionShape_, EpilogueOutputOp_, ThreadblockSwizzle_, Stages, AlignmentA, AlignmentB, SplitKSerial, SubGemmParallel, Operator_, GatherA, GatherB, ScatterD, PermuteDLayout> 
  {
 public:
  using ElementA = ElementA_;
  using LayoutA = LayoutA_;
  using TensorRefA = TensorRef<ElementA const, LayoutA>;
  using ElementB = ElementB_;
  using LayoutB = LayoutB_;
  using TensorRefB = TensorRef<ElementB const, LayoutB>;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  using TensorRefC = TensorRef<ElementC const, LayoutC>;
  using TensorRefD = TensorRef<ElementC, LayoutC>;
  using ElementAccumulator = ElementAccumulator_;
  using OperatorClass = OperatorClass_;
  using ArchTag = ArchTag_;
  using ThreadblockShape = ThreadblockShape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  using EpilogueOutputOp = EpilogueOutputOp_;
  using InterimEpilogueOp = InterimEpilogueOp_;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  using Operator = Operator_;
  static int const kStages = Stages;
  static int const kAlignmentA = AlignmentA;
  static int const kAlignmentB = AlignmentB;
  static int const kAlignmentC = EpilogueOutputOp::kCount;
  static bool const kSplitKSerial = SplitKSerial;
  static ComplexTransform const kTransformA = ComplexTransform::kNone;
  static ComplexTransform const kTransformB = ComplexTransform::kNone;

    using ChildGemmM0 = StrassenGemm<
      StrassenKind,
      StrassenMiGroupM0,
      ScheduleStrassenGroups,
      ElementA,
      LayoutA,
      ElementB,
      LayoutB,
      ElementC,
      LayoutC,
      ElementAccumulator,
      OperatorClass,
      ArchTag,
      ThreadblockShape,
      WarpShape,
      InstructionShape,
      EpilogueOutputOp,
      InterimEpilogueOp,
      ThreadblockSwizzle,
      kStages,
      kAlignmentA,
      kAlignmentB,
      kSplitKSerial,
      SubGemmParallel,
      Operator,
      GatherA,
      GatherB,
      ScatterD,
      PermuteDLayout>;

    using ChildGemmM1 = StrassenGemm<
      StrassenKind,
      StrassenMiGroupM1,
      ScheduleStrassenGroups,
      ElementA,
      LayoutA,
      ElementB,
      LayoutB,
      ElementC,
      LayoutC,
      ElementAccumulator,
      OperatorClass,
      ArchTag,
      ThreadblockShape,
      WarpShape,
      InstructionShape,
      EpilogueOutputOp,
      InterimEpilogueOp,
      ThreadblockSwizzle,
      kStages,
      kAlignmentA,
      kAlignmentB,
      kSplitKSerial,
      SubGemmParallel,
      Operator,
      GatherA,
      GatherB,
      ScatterD,
      PermuteDLayout>;

    using ChildGemmM2 = StrassenGemm<
      StrassenKind,
      StrassenMiGroupM2,
      ScheduleStrassenGroups,
      ElementA,
      LayoutA,
      ElementB,
      LayoutB,
      ElementC,
      LayoutC,
      ElementAccumulator,
      OperatorClass,
      ArchTag,
      ThreadblockShape,
      WarpShape,
      InstructionShape,
      EpilogueOutputOp,
      InterimEpilogueOp,
      ThreadblockSwizzle,
      kStages,
      kAlignmentA,
      kAlignmentB,
      kSplitKSerial,
      SubGemmParallel,
      Operator,
      GatherA,
      GatherB,
      ScatterD,
      PermuteDLayout>;

    using ChildGemmM3 = StrassenGemm<
      StrassenKind,
      StrassenMiGroupM3,
      ScheduleStrassenGroups,
      ElementA,
      LayoutA,
      ElementB,
      LayoutB,
      ElementC,
      LayoutC,
      ElementAccumulator,
      OperatorClass,
      ArchTag,
      ThreadblockShape,
      WarpShape,
      InstructionShape,
      EpilogueOutputOp,
      InterimEpilogueOp,
      ThreadblockSwizzle,
      kStages,
      kAlignmentA,
      kAlignmentB,
      kSplitKSerial,
      SubGemmParallel,
      Operator,
      GatherA,
      GatherB,
      ScatterD,
      PermuteDLayout>;

    using ChildGemmM4 = StrassenGemm<
      StrassenKind,
      StrassenMiGroupM4,
      ScheduleStrassenGroups,
      ElementA,
      LayoutA,
      ElementB,
      LayoutB,
      ElementC,
      LayoutC,
      ElementAccumulator,
      OperatorClass,
      ArchTag,
      ThreadblockShape,
      WarpShape,
      InstructionShape,
      EpilogueOutputOp,
      InterimEpilogueOp,
      ThreadblockSwizzle,
      kStages,
      kAlignmentA,
      kAlignmentB,
      kSplitKSerial,
      SubGemmParallel,
      Operator,
      GatherA,
      GatherB,
      ScatterD,
      PermuteDLayout>;

    using ChildGemmM5 = StrassenGemm<
      StrassenKind,
      StrassenMiGroupM5,
      ScheduleStrassenGroups,
      ElementA,
      LayoutA,
      ElementB,
      LayoutB,
      ElementC,
      LayoutC,
      ElementAccumulator,
      OperatorClass,
      ArchTag,
      ThreadblockShape,
      WarpShape,
      InstructionShape,
      EpilogueOutputOp,
      InterimEpilogueOp,
      ThreadblockSwizzle,
      kStages,
      kAlignmentA,
      kAlignmentB,
      kSplitKSerial,
      SubGemmParallel,
      Operator,
      GatherA,
      GatherB,
      ScatterD,
      PermuteDLayout>;

    using ChildGemmM6 = StrassenGemm<
      StrassenKind,
      StrassenMiGroupM6,
      ScheduleStrassenGroups,
      ElementA,
      LayoutA,
      ElementB,
      LayoutB,
      ElementC,
      LayoutC,
      ElementAccumulator,
      OperatorClass,
      ArchTag,
      ThreadblockShape,
      WarpShape,
      InstructionShape,
      EpilogueOutputOp,
      InterimEpilogueOp,
      ThreadblockSwizzle,
      kStages,
      kAlignmentA,
      kAlignmentB,
      kSplitKSerial,
      SubGemmParallel,
      Operator,
      GatherA,
      GatherB,
      ScatterD,
      PermuteDLayout>;

  /// Argument structure
  struct Arguments {

    //
    // Data members
    //

    GemmCoord problem_size;
    TensorRef<ElementA const, LayoutA> ref_A;
    TensorRef<ElementB const, LayoutB> ref_B;
    TensorRef<ElementC const, LayoutC> ref_C;
    TensorRef<ElementC, LayoutC> ref_D;
    typename EpilogueOutputOp::Params epilogue;
    int split_k_slices;
    LayoutA layout_MA;
    LayoutB layout_MB;
    int level_1_idx;
    // For gather+scatter operations
    int const *gather_A_indices;
    int const *gather_B_indices;
    int const *scatter_D_indices;

    //
    // Methods
    //

    /// Default ctor
    CUTLASS_HOST_DEVICE
    Arguments(): problem_size(0, 0, 0), split_k_slices(1), level_1_idx(1) {

    }

    /// Constructs an Arguments structure
    CUTLASS_HOST_DEVICE
    Arguments(
      GemmCoord problem_size_,
      TensorRef<ElementA const, LayoutA> ref_A_,
      TensorRef<ElementB const, LayoutB> ref_B_,
      TensorRef<ElementC const, LayoutC> ref_C_,
      TensorRef<ElementC, LayoutC> ref_D_,
      typename EpilogueOutputOp::Params epilogue_ = 
        typename EpilogueOutputOp::Params(),
      int split_k_slices = 1,
      int const *gather_A_indices_ = nullptr,
      int const *gather_B_indices_ = nullptr,
      int const *scatter_D_indices_ = nullptr
    ): Arguments(problem_size_, ref_A_, ref_B_, ref_C_, ref_D_,
                 epilogue_, split_k_slices, layoutforM(ref_A_), layoutforM(ref_B_), 0,
                 gather_A_indices_, gather_B_indices_, scatter_D_indices_) {}
             
    /// Constructs an Arguments structure 
    CUTLASS_HOST_DEVICE
    Arguments(
      GemmCoord problem_size_,
      TensorRef<ElementA const, LayoutA> ref_A_,
      TensorRef<ElementB const, LayoutB> ref_B_,
      TensorRef<ElementC const, LayoutC> ref_C_,
      TensorRef<ElementC, LayoutC> ref_D_,
      typename EpilogueOutputOp::Params epilogue_,
      int split_k_slices ,
      LayoutA layout_MA_,
      LayoutB layout_MB_,
      int level_1_idx,
      int const *gather_A_indices_ = nullptr,
      int const *gather_B_indices_ = nullptr,
      int const *scatter_D_indices_ = nullptr
    ):
      problem_size(problem_size_),
      ref_A(ref_A_),
      ref_B(ref_B_),
      ref_C(ref_C_),
      ref_D(ref_D_),
      epilogue(epilogue_),
      split_k_slices(split_k_slices),
      layout_MA(layout_MA_),
      layout_MB(layout_MB_),
      level_1_idx(level_1_idx),
      gather_A_indices(gather_A_indices_),
      gather_B_indices(gather_B_indices_),
      scatter_D_indices(scatter_D_indices_) {

    }
  };

private:
  static constexpr size_t kChildWorkspaceAlignment = 32;

  static size_t align_child_workspace_size(size_t bytes) {
    return SubGemmParallel ? ((bytes + kChildWorkspaceAlignment - 1) / kChildWorkspaceAlignment) * kChildWorkspaceAlignment : bytes;
  }

  ChildGemmM0 gemm_m0_;
  ChildGemmM1 gemm_m1_;
  ChildGemmM2 gemm_m2_;
  ChildGemmM3 gemm_m3_;
  ChildGemmM4 gemm_m4_;
  ChildGemmM5 gemm_m5_;
  ChildGemmM6 gemm_m6_;

  template <typename ChildGemm>
  static typename ChildGemm::Arguments to_child_arguments(Arguments const &args, TensorRef<ElementA const, LayoutA> ref_A,
                                                          TensorRef<ElementB const, LayoutB> ref_B, TensorRef<ElementC const, LayoutC> ref_C,
                                                          TensorRef<ElementC, LayoutC> ref_D, TensorRef<ElementC const, LayoutC> ref_C2,
                                                          TensorRef<ElementC, LayoutC> ref_D2,
                                                          LayoutA layout_MA, LayoutB layout_MB, int level_1_idx,
                                                          bool halve_problem_size = true) {
    GemmCoord child_problem_size(
      halve_problem_size ? (args.problem_size.m() / 2) : args.problem_size.m(),
      halve_problem_size ? (args.problem_size.n() / 2) : args.problem_size.n(),
      halve_problem_size ? (args.problem_size.k() / 2) : args.problem_size.k());
    return typename ChildGemm::Arguments{
      child_problem_size,
        ref_A,
        ref_B,
        ref_C,
        ref_D,
        ref_C2, ref_D2,
        {(level_1_idx == 6) ? -1.0f : args.epilogue.alpha, (level_1_idx == 0 || level_1_idx == 1 || level_1_idx == 2) ? 0.0f : 1.0f},
        args.split_k_slices,
        layout_MA, layout_MB,
        level_1_idx,
        args.gather_A_indices,
        args.gather_B_indices,
        args.scatter_D_indices};
  }

  static cudaStream_t stream_for_index(cudaStream_t* streams, int num_streams, int index) {
    if (streams == nullptr || num_streams == 0) {
      return nullptr;
    }
    if (num_streams == 1) {
      return streams[0];
    }
    assert(num_streams > index);
    return streams[index];
  }

public:

  /// Constructs the GEMM.
  StrassenGemmLevel2() {
    /*int kM = GemmKernel_::Mma_::MmaCore::LaneMmaShape::kM;
    int kN = GemmKernel_::Mma_::MmaCore::LaneMmaShape::kN;
    printf("361: LaneMmaShape %d %d \n", kM, kN);
    using WarpCount = typename GemmKernel_::Mma_::MmaCore::WarpCount;
    printf("363: WarpCount %d %d\n", WarpCount::kM, WarpCount::kN);

    int WarpNumThreadsM = GemmKernel_::Mma_::MmaCore::WarpNumThreadsM;
    int WarpNumThreadsN = GemmKernel_::Mma_::MmaCore::WarpNumThreadsN;
    printf("367: %d %d\n", WarpNumThreadsM, WarpNumThreadsN);
    
    using WarpShape = typename GemmKernel_::Mma_::MmaCore::MmaWarpSimt::Shape;
    using PolicyWarpShape = typename GemmKernel_::Mma_::MmaCore::MmaWarpSimt::Policy::WarpShape;
    printf("368: %d %d; %d %d\n", WarpShape::kM, WarpShape::kN, PolicyWarpShape::kRow, PolicyWarpShape::kColumn);

    int MmaCorekThreads = GemmKernel_::Mma_::MmaCore::kThreads;
    printf("MmaCore::kThreads %d\n", MmaCorekThreads);

    using ThreadShape = typename GemmKernel_::Mma_::MmaCore::MmaWarpSimt::ThreadMma::Shape;
    printf("374: %d %d\n", ThreadShape::kM, ThreadShape::kN);

    using MmaCoreShape = typename GemmKernel_::Mma_::MmaCore::Shape;
    printf("377: %d %d %d\n", MmaCoreShape::kM, MmaCoreShape::kN, MmaCoreShape::kK);

    using ThreadMapA = typename GemmKernel_::Mma_::MmaCore::IteratorThreadMapA;
    printf("380: %d %d\n", ThreadMapA::Iterations::kCount, ThreadMapA::kElementsPerAccess);
    printf("381: %d %d %d\n", ThreadMapA::kThreads, ThreadMapA::Detail::ShapeVec::kContiguous, ThreadMapA::Detail::ShapeVec::kStrided);

    uint WarpCountkCount = GemmKernel::WarpCount::kCount;
    printf("384: WarpCountkCount %d\n", WarpCountkCount);*/
  }

  /// Determines whether the GEMM can execute the given problem.
  static Status can_implement(Arguments const &args) {

    if (!kSplitKSerial && args.split_k_slices > 1) {
      return Status::kErrorInvalidProblem;
    }
    auto args_m0 = to_child_arguments<ChildGemmM0>(args, args.ref_A, args.ref_B, args.ref_C, args.ref_D, args.ref_C, args.ref_D, layoutforM(args.ref_A, 1), layoutforM(args.ref_B, 1), 0, false);
    auto args_m1 = to_child_arguments<ChildGemmM1>(args, args.ref_A, args.ref_B, args.ref_C, args.ref_D, args.ref_C, args.ref_D, layoutforM(args.ref_A, 2), layoutforM(args.ref_B, 2), 1);
    auto args_m2 = to_child_arguments<ChildGemmM2>(args, args.ref_A, args.ref_B, args.ref_C, args.ref_D, args.ref_C, args.ref_D, layoutforM(args.ref_A, 2), layoutforM(args.ref_B, 2), 2);
    auto args_m3 = to_child_arguments<ChildGemmM3>(args, args.ref_A, args.ref_B, args.ref_C, args.ref_D, args.ref_C, args.ref_D, layoutforM(args.ref_A, 2), layoutforM(args.ref_B, 2), 3);
    auto args_m4 = to_child_arguments<ChildGemmM4>(args, args.ref_A, args.ref_B, args.ref_C, args.ref_D, args.ref_C, args.ref_D, layoutforM(args.ref_A, 2), layoutforM(args.ref_B, 2), 4);
    auto args_m5 = to_child_arguments<ChildGemmM5>(args, args.ref_A, args.ref_B, args.ref_C, args.ref_D, args.ref_C, args.ref_D, layoutforM(args.ref_A, 2), layoutforM(args.ref_B, 2), 5);
    auto args_m6 = to_child_arguments<ChildGemmM6>(args, args.ref_A, args.ref_B, args.ref_C, args.ref_D, args.ref_C, args.ref_D, layoutforM(args.ref_A, 2), layoutforM(args.ref_B, 2), 6);

    Status status = ChildGemmM0::can_implement(args_m0);
    if (status != Status::kSuccess) return status;
    status = ChildGemmM1::can_implement(args_m1);
    if (status != Status::kSuccess) return status;
    status = ChildGemmM2::can_implement(args_m2);
    if (status != Status::kSuccess) return status;
    status = ChildGemmM3::can_implement(args_m3);
    if (status != Status::kSuccess) return status;
    status = ChildGemmM4::can_implement(args_m4);
    if (status != Status::kSuccess) return status;
    status = ChildGemmM5::can_implement(args_m5);
    if (status != Status::kSuccess) return status;
    status = ChildGemmM6::can_implement(args_m6);
    if (status != Status::kSuccess) return status;

    return Status::kSuccess;
  }

  /// Gets the workspace size
  static size_t get_workspace_size(Arguments const &args) {
    auto args_m0 = to_child_arguments<ChildGemmM0>(args, args.ref_A, args.ref_B, args.ref_C, args.ref_D, args.ref_C, args.ref_D, layoutforM(args.ref_A, 1), layoutforM(args.ref_B, 1), 0, false);
    auto args_m1 = to_child_arguments<ChildGemmM1>(args, args.ref_A, args.ref_B, args.ref_C, args.ref_D, args.ref_C, args.ref_D, layoutforM(args.ref_A, 2), layoutforM(args.ref_B, 2), 1);
    auto args_m2 = to_child_arguments<ChildGemmM2>(args, args.ref_A, args.ref_B, args.ref_C, args.ref_D, args.ref_C, args.ref_D, layoutforM(args.ref_A, 2), layoutforM(args.ref_B, 2), 2);
    auto args_m3 = to_child_arguments<ChildGemmM3>(args, args.ref_A, args.ref_B, args.ref_C, args.ref_D, args.ref_C, args.ref_D, layoutforM(args.ref_A, 2), layoutforM(args.ref_B, 2), 3);
    auto args_m4 = to_child_arguments<ChildGemmM4>(args, args.ref_A, args.ref_B, args.ref_C, args.ref_D, args.ref_C, args.ref_D, layoutforM(args.ref_A, 2), layoutforM(args.ref_B, 2), 4);
    auto args_m5 = to_child_arguments<ChildGemmM5>(args, args.ref_A, args.ref_B, args.ref_C, args.ref_D, args.ref_C, args.ref_D, layoutforM(args.ref_A, 2), layoutforM(args.ref_B, 2), 5);
    auto args_m6 = to_child_arguments<ChildGemmM6>(args, args.ref_A, args.ref_B, args.ref_C, args.ref_D, args.ref_C, args.ref_D, layoutforM(args.ref_A, 2), layoutforM(args.ref_B, 2), 6);

        return align_child_workspace_size(ChildGemmM0::get_workspace_size(args_m0)) +
          align_child_workspace_size(ChildGemmM1::get_workspace_size(args_m1)) +
          align_child_workspace_size(ChildGemmM2::get_workspace_size(args_m2)) +
          align_child_workspace_size(ChildGemmM3::get_workspace_size(args_m3)) +
          align_child_workspace_size(ChildGemmM4::get_workspace_size(args_m4)) +
          align_child_workspace_size(ChildGemmM5::get_workspace_size(args_m5)) +
          align_child_workspace_size(ChildGemmM6::get_workspace_size(args_m6));
  }

  Status initialize(Arguments const &args, void *workspace = nullptr, cudaStream_t stream = nullptr) {
    auto args_m0 = to_child_arguments<ChildGemmM0>(args, args.ref_A, args.ref_B, args.ref_C, args.ref_D, args.ref_C, args.ref_D, layoutforM(args.ref_A, 1), layoutforM(args.ref_B, 1), 0, false);
    auto args_m1 = to_child_arguments<ChildGemmM1>(args, args.ref_A, args.ref_B, args.ref_C, args.ref_D, args.ref_C, args.ref_D, layoutforM(args.ref_A, 2), layoutforM(args.ref_B, 2), 1);
    auto args_m2 = to_child_arguments<ChildGemmM2>(args, args.ref_A, args.ref_B, args.ref_C, args.ref_D, args.ref_C, args.ref_D, layoutforM(args.ref_A, 2), layoutforM(args.ref_B, 2), 2);
    auto args_m3 = to_child_arguments<ChildGemmM3>(args, args.ref_A, args.ref_B, args.ref_C, args.ref_D, args.ref_C, args.ref_D, layoutforM(args.ref_A, 2), layoutforM(args.ref_B, 2), 3);
    auto args_m4 = to_child_arguments<ChildGemmM4>(args, args.ref_A, args.ref_B, args.ref_C, args.ref_D, args.ref_C, args.ref_D, layoutforM(args.ref_A, 2), layoutforM(args.ref_B, 2), 4);
    auto args_m5 = to_child_arguments<ChildGemmM5>(args, args.ref_A, args.ref_B, args.ref_C, args.ref_D, args.ref_C, args.ref_D, layoutforM(args.ref_A, 2), layoutforM(args.ref_B, 2), 5);
    auto args_m6 = to_child_arguments<ChildGemmM6>(args, args.ref_A, args.ref_B, args.ref_C, args.ref_D, args.ref_C, args.ref_D, layoutforM(args.ref_A, 2), layoutforM(args.ref_B, 2), 6);

    size_t workspace_bytes_m0 = ChildGemmM0::get_workspace_size(args_m0);
    size_t workspace_bytes_m1 = ChildGemmM1::get_workspace_size(args_m1);
    size_t workspace_bytes_m2 = ChildGemmM2::get_workspace_size(args_m2);
    size_t workspace_bytes_m3 = ChildGemmM3::get_workspace_size(args_m3);
    size_t workspace_bytes_m4 = ChildGemmM4::get_workspace_size(args_m4);
    size_t workspace_bytes_m5 = ChildGemmM5::get_workspace_size(args_m5);
    size_t workspace_bytes_m6 = ChildGemmM6::get_workspace_size(args_m6);

    uint8_t* workspace_bytes = static_cast<uint8_t*>(workspace);
    void* workspace_m0 = nullptr;
    void* workspace_m1 = nullptr;
    void* workspace_m2 = nullptr;
    void* workspace_m3 = nullptr;
    void* workspace_m4 = nullptr;
    void* workspace_m5 = nullptr;
    void* workspace_m6 = nullptr;

    if (workspace_bytes) {
      size_t offset = 0;
      workspace_m0 = workspace_bytes + offset;
      offset += align_child_workspace_size(workspace_bytes_m0);
      workspace_m1 = workspace_bytes + offset;
      offset += align_child_workspace_size(workspace_bytes_m1);
      workspace_m2 = workspace_bytes + offset;
      offset += align_child_workspace_size(workspace_bytes_m2);
      workspace_m3 = workspace_bytes + offset;
      offset += align_child_workspace_size(workspace_bytes_m3);
      workspace_m4 = workspace_bytes + offset;
      offset += align_child_workspace_size(workspace_bytes_m4);
      workspace_m5 = workspace_bytes + offset;
      offset += align_child_workspace_size(workspace_bytes_m5);
      workspace_m6 = workspace_bytes + offset;
    }

    using AllPresums = typename StrassenMiGroupM0::Group0::AllPresums;

    Status err = gemm_m0_.initialize(args_m0, workspace_m0, stream);
    if (err != Status::kSuccess) return err;
    ElementA* presum_a_ptr  = ChildGemmM0::get_presum_a_ptr(args_m0, workspace_m0);
    ElementA* presum_b_ptr  = ChildGemmM0::get_presum_b_ptr(args_m0, workspace_m0);
    ElementC* postsum_m_ptr = ChildGemmM0::get_postsum_m_ptr(args_m0, workspace_m0);

    const uint M = args.problem_size.m();
    const uint N = args.problem_size.n();
    const uint K = args.problem_size.k();
    const uint halfM = args.problem_size.m()/2;
    const uint halfN = args.problem_size.n()/2;
    const uint halfK = args.problem_size.k()/2;

    bool is_fp16 = std::is_same<ElementA, cutlass::half_t>::value;//TODO:Fix this
    {
      //TODO: StrassenGeMML2 takes a 2-level schedule
      //m1=a1@b2
      TensorRefA ref_A = {args.ref_A.data() + halfK, args.ref_A.layout()}; //TensorRefC ref_C = {}; TensorRefD ref_D = {};
      TensorRefB ref_B = {args.ref_B.data() + halfK*N, args.ref_B.layout()};
      TensorRefC ref_m0l1 = {postsum_m_ptr, halfN};
      TensorRefC ref_C2 = TensorRefC{nullptr, halfN};
      TensorRefD ref_D2 = TensorRefD{nullptr, halfN};
      // printf("1796 %p ; %p \n", ref_m0l1.data(), ref_A.data());
      args_m1 = to_child_arguments<ChildGemmM1>(args, ref_A, ref_B, ref_m0l1, args.ref_D, ref_C2, ref_D2, layoutforM(args.ref_A, 2), layoutforM(args.ref_B, 2)/*divide by 1 level */, 1);
      err = gemm_m1_.initialize(args_m1, workspace_m1, stream);
      // printf("1799\n");
      if (err != Status::kSuccess) return err;

      //m2 = s2@s3
      TensorRefA ref_s2 = {presum_a_ptr + AllPresums::indexAPresum(MmaStrassen::APresums::S2)*halfM*halfK,
                          halfK};
      TensorRefB ref_s3 = {presum_b_ptr + AllPresums::indexBPresum(MmaStrassen::BPresums::S3)*halfN*halfK,
                          halfN};
      //m2 is stored at [1]
      TensorRefD ref_m2l1 = {postsum_m_ptr + (is_fp16+1)*halfM*halfN, halfN};
      args_m2 = to_child_arguments<ChildGemmM2>(args, ref_s2, ref_s3, ref_m0l1, ref_m2l1, ref_C2, ref_D2, layoutforM(args.ref_A, 2), layoutforM(args.ref_B, 2), 2);
      err = gemm_m2_.initialize(args_m2, workspace_m2, stream);
      if (err != Status::kSuccess) return err;
    }

    {
      // printf("1817 %p\n", presum_b_ptr +AllPresums::indexBPresum(MmaStrassen::BPresums::B31)*halfN*halfK);
      //m3 = a02@b31
      TensorRefA ref_a02 = {presum_a_ptr + AllPresums::indexAPresum(MmaStrassen::APresums::A02)*halfM*halfK,
                          halfK};
      TensorRefB ref_b31 = {presum_b_ptr + AllPresums::indexBPresum(MmaStrassen::BPresums::B31)*halfK*halfN,
                          halfN};
      TensorRefC ref_m2l1 = {postsum_m_ptr + (is_fp16+1)*halfM*halfN, halfN};
      //m3 is stored at [2]
      TensorRefD ref_m3l1 = {postsum_m_ptr + (is_fp16+2)*halfM*halfN, halfN};
      TensorRefC ref_C2 = TensorRefC{nullptr, halfN};
      TensorRefD ref_D2 = TensorRefD{nullptr, halfN};
      args_m3 = to_child_arguments<ChildGemmM3>(args, ref_a02, ref_b31, ref_m2l1, ref_m3l1, ref_C2, ref_D2,
                                                layoutforM(args.ref_A, 2), layoutforM(args.ref_B, 2), 3);
      err = gemm_m3_.initialize(args_m3, workspace_m3, stream);
      if (err != Status::kSuccess) return err;
    }
    
    {
      //m4 = s1@b10
      TensorRefA ref_s1 = {presum_a_ptr + AllPresums::indexAPresum(MmaStrassen::APresums::S1)*halfM*halfK,
                          halfK};
      TensorRefB ref_b10 = {presum_b_ptr + AllPresums::indexBPresum(MmaStrassen::BPresums::B10)*halfN*halfK,
                          halfN};
      TensorRefC ref_m3l1 = {postsum_m_ptr + (is_fp16+2)*halfM*halfN, halfN};
      TensorRefD ref_m4l1 = {postsum_m_ptr + (is_fp16+3)*halfM*halfN, halfN};
      TensorRefD ref_d11 = {args.ref_D.data() + halfM*N + halfN, args.ref_D.layout()};
      TensorRefC ref_C2 = TensorRefC{nullptr, halfN};
      // TensorRefD ref_D2 = TensorRefD{, halfN};

      args_m4 = to_child_arguments<ChildGemmM4>(args, ref_s1, ref_b10, ref_m3l1, ref_d11, ref_C2, ref_m4l1,
                                                layoutforM(args.ref_A, 2), layoutforM(args.ref_B, 2), 4);
      err = gemm_m4_.initialize(args_m4, workspace_m4, stream);
      if (err != Status::kSuccess) return err;
    }
    {
      //m5 = a1s2@b3
      TensorRefA ref_a1s2 = {presum_a_ptr + AllPresums::indexAPresum(MmaStrassen::APresums::A1S2)*halfM*halfK,
                          halfK};
      TensorRefB ref_b3 = {args.ref_B.data() + halfK*N + halfN, args.ref_B.layout()};
      TensorRefC ref_m2l1 = {postsum_m_ptr + (is_fp16+1)*halfM*halfN, halfN};
      TensorRefC ref_m4l1 = {postsum_m_ptr + (is_fp16+3)*halfM*halfN, halfN};
      TensorRefD ref_d01 = {args.ref_D.data() + halfN, args.ref_D.layout()};
      TensorRefD ref_D2 = TensorRefD{nullptr, halfN};
      // TensorRefD ref_D2 = TensorRefD{, halfN};

      args_m5 = to_child_arguments<ChildGemmM5>(args, ref_a1s2, ref_b3, ref_m2l1, ref_d01, ref_m4l1, ref_D2,
                                                layoutforM(args.ref_A, 2), layoutforM(args.ref_B, 2), 5);
      err = gemm_m5_.initialize(args_m5, workspace_m5, stream);
      if (err != Status::kSuccess) return err;
    }
    {
      //m6 = a3@s3b2
      TensorRefA ref_a02 = {args.ref_A.data() + halfM*K + halfK, args.ref_A.layout()};
      TensorRefB ref_s3b2 = {presum_b_ptr + AllPresums::indexBPresum(MmaStrassen::BPresums::S3B2)*halfN*halfK,
                          halfN};
      TensorRefC ref_m3l1 = {postsum_m_ptr + (is_fp16+2)*halfM*halfN, halfN};
      TensorRefD ref_d10 = {args.ref_D.data() + halfM*N, args.ref_D.layout()};
      TensorRefC ref_C2 = TensorRefC{nullptr, halfN};
      TensorRefD ref_D2 = TensorRefD{nullptr, halfN};

      args_m6 = to_child_arguments<ChildGemmM6>(args, ref_a02, ref_s3b2, ref_m3l1, ref_d10, ref_C2, ref_D2,
                                                layoutforM(args.ref_A, 2), layoutforM(args.ref_B, 2), 6);
      err = gemm_m6_.initialize(args_m6, workspace_m6, stream);
      if (err != Status::kSuccess) return err;
    }

    return Status::kSuccess;
  }

  /// Lightweight update given a subset of arguments
  Status update(Arguments const &args, void *workspace = nullptr) {
    (void)args;
    (void)workspace;
    assert(false);
    return Status::kErrorInternal;
  }

  /// Runs the kernel using initialized state.
  Status run(cudaStream_t* streams = nullptr, int num_streams = 0) {
    Status status = Status::kSuccess;

    if (num_streams < 0) {
      return Status::kErrorInvalidProblem;
    }

    if (num_streams <= 1) {
      status = gemm_m0_.run(streams, 1);
      if (status != Status::kSuccess) return status;
      if (ChildGemmM0::StrassenGroups::Group0::FusedOrContinueMMA() == 0) {
        status = gemm_m1_.run(streams, 1);
        if (status != Status::kSuccess) return status;
      }
      status = gemm_m2_.run(streams, 1);
      if (status != Status::kSuccess) return status;
      status = gemm_m3_.run(streams, 1);
      if (status != Status::kSuccess) return status;
      status = gemm_m4_.run(streams, 1);
      if (status != Status::kSuccess) return status;
      status = gemm_m5_.run(streams, 1);
      if (status != Status::kSuccess) return status;
      status = gemm_m6_.run(streams, 1);
      if (status != Status::kSuccess) return status;

      return Status::kSuccess;
    }

    if (num_streams != 49 || streams == nullptr) {
      return Status::kErrorInvalidProblem;
    }

    status = gemm_m0_.run(streams + 0 * 7, 7);
    if (status != Status::kSuccess) return status;
    if (ChildGemmM0::StrassenGroups::Group0::FusedOrContinueMMA() == 0) {
      status = gemm_m1_.run(streams + 1 * 7, 7);
      if (status != Status::kSuccess) return status;
    }

    status = gemm_m2_.run(streams + 2 * 7, 7);
    if (status != Status::kSuccess) return status;
    status = gemm_m3_.run(streams + 3 * 7, 7);
    if (status != Status::kSuccess) return status;
    status = gemm_m4_.run(streams + 4 * 7, 7);
    if (status != Status::kSuccess) return status;
    status = gemm_m5_.run(streams + 5 * 7, 7);
    if (status != Status::kSuccess) return status;
    status = gemm_m6_.run(streams + 6 * 7, 7);
    if (status != Status::kSuccess) return status;

    return Status::kSuccess;
  }

  /// Runs the kernel using initialized state.
  // Status operator()(cudaStream_t stream = nullptr) {
  //   return run(stream);
  // }

  /// Runs the kernel using initialized state.
  Status operator()(
    Arguments const &args,
    void *workspace = nullptr,
    cudaStream_t* streams = nullptr,
    int num_streams = 0) {
    cudaStream_t init_stream;

    if (streams == nullptr || num_streams == 0) {
      init_stream = nullptr;
    } else {
      init_stream = streams[0];
    }

    Status status = initialize(args, workspace, init_stream);

    if (status == Status::kSuccess) {
      status = run(streams, num_streams);
    }

    return status;
  }
};

////////////////////////////////////////////////////////////////////////////////

CUTLASS_DEVICE
void sumHalf8(half2* frag_op1, half2* frag_op2, half2* frag_out, half2 mul) {
  for (int i = 0; i < 4; i++) {
    frag_out[i] = frag_op1[i] + mul * frag_op2[i];
  }
}

CUTLASS_DEVICE
void sumHalf8(half2* frag_op1, half2* frag_op2, half2* frag_out, float mul) {
  for (int i = 0; i < 4; i++) {
    float2 op2f = __half22float2(frag_op2[i]);
    float2 op1f = __half22float2(frag_op1[i]);
    frag_out[i] = __float22half2_rn(float2{op1f.x + mul * op2f.x, op1f.y + mul * op2f.y});
  }
}

template<typename ElementB, int THREAD_TILE_M, int THREAD_TILE_N>
__global__ void presumB(ElementB* B, ElementB* B_out, int M, int N, uint64_t schedule) {
  typedef float4 MemAccessVector;
  const uint MemAccessVectorElems = sizeof(float4)/sizeof(half);

  int bid_n = blockIdx.x * blockDim.x * THREAD_TILE_N * MemAccessVectorElems;
  int bid_m = blockIdx.y * blockDim.y * THREAD_TILE_M;

  #pragma unroll
  for (int m_ = threadIdx.y; m_ < blockDim.y * THREAD_TILE_M; m_ += blockDim.y) {
  #pragma unroll
  for (int n_ = threadIdx.x * MemAccessVectorElems; n_ < blockDim.x * THREAD_TILE_N * MemAccessVectorElems && bid_n + n_ < N;
       n_ += blockDim.x * MemAccessVectorElems) {
    uint m = bid_m + m_;
    uint n = bid_n + n_;
    uint idx = m * N + n;
    const uint sizeB = M*N;
    MemAccessVector b0 = *(MemAccessVector*)(B + idx);
    MemAccessVector b1 = *(MemAccessVector*)(B + N/2 + idx);
    MemAccessVector b2 = *(MemAccessVector*)(B + M/2 * N +  idx);
    MemAccessVector b3 = *(MemAccessVector*)(B + M/2 *N + N/2 + idx);

    uint write_idx = m * N/2 + n;
    
    MemAccessVector b10;
    sumHalf8(reinterpret_cast<half2*>(&b1), reinterpret_cast<half2*>(&b0),
             reinterpret_cast<half2*>(&b10), -1.0f);
    MemAccessVector b310;
    sumHalf8(reinterpret_cast<half2*>(&b3), reinterpret_cast<half2*>(&b10),
             reinterpret_cast<half2*>(&b310), -1.0f);
    
    *(MemAccessVector*)(B_out + MmaStrassen::GetSchedule1(schedule, 0)*sizeB/4 + write_idx) = b0;

    *(MemAccessVector*)(B_out + MmaStrassen::GetSchedule1(schedule, 1)*sizeB/4 + write_idx) = b2;
    *(MemAccessVector*)(B_out + MmaStrassen::GetSchedule1(schedule, 2)*sizeB/4 + write_idx) = b310;

    MemAccessVector b31;
    sumHalf8(reinterpret_cast<half2*>(&b3), reinterpret_cast<half2*>(&b1),
              reinterpret_cast<half2*>(&b31), -1.0f);
    *(MemAccessVector*)(B_out + MmaStrassen::GetSchedule1(schedule, 3)*sizeB/4 + write_idx) = b31;

    *(MemAccessVector*)(B_out + MmaStrassen::GetSchedule1(schedule, 4)*sizeB/4 + write_idx) = b10;

    *(MemAccessVector*)(B_out + MmaStrassen::GetSchedule1(schedule, 5)*sizeB/4 + write_idx) = b3;

    MemAccessVector b32;
    sumHalf8(reinterpret_cast<half2*>(&b3), reinterpret_cast<half2*>(&b2),
             reinterpret_cast<half2*>(&b32), -1.0f);
    MemAccessVector b3210;
    sumHalf8(reinterpret_cast<half2*>(&b32), reinterpret_cast<half2*>(&b10),
             reinterpret_cast<half2*>(&b3210), -1.0f);
    *(MemAccessVector*)(B_out + MmaStrassen::GetSchedule1(schedule, 6)*sizeB/4 + write_idx) = b3210;
  }
  }
}

template<typename ElementA, int THREADS_Y, int THREADS_X, int THREAD_TILE_M, int THREAD_TILE_N>
__launch_bounds__(THREADS_Y*THREADS_X, 8)
__global__ void presumA(ElementA* A, ElementA* A_out, int M, int N) {
  int bid_n = blockIdx.x * THREADS_X * THREAD_TILE_N;
  int bid_m = blockIdx.y * THREADS_Y * THREAD_TILE_M;

  if (bid_m >= M) return;

  typedef float4 MemAccessVector;

  const uint MemAccessVectorElems = sizeof(float4)/sizeof(half);
  MemAccessVector a0[THREAD_TILE_M];
  MemAccessVector a1[THREAD_TILE_M];
  MemAccessVector a2[THREAD_TILE_M];
  MemAccessVector a3[THREAD_TILE_M];

  // #pragma unroll THREAD_TILE_M
  for (int m_ = 0; m_ < THREAD_TILE_M; m_ += 1) {
    uint n = bid_n + threadIdx.x * MemAccessVectorElems;
    uint m = bid_m + threadIdx.y * THREAD_TILE_M + THREADS_Y * m_;
    uint idx = m * N + n;
    const uint sizeA = M*N;
    a0[m_] = *(MemAccessVector*)&A[idx];
    a1[m_] = *(MemAccessVector*)&A[N/2 + idx];
    a2[m_] = *(MemAccessVector*)&A[M/2 * N +  idx];
    a3[m_] = *(MemAccessVector*)&A[M/2 *N + N/2 + idx];
  }

  MemAccessVector a23[THREAD_TILE_M];
  MemAccessVector a230[THREAD_TILE_M];
  MemAccessVector a02[THREAD_TILE_M];
  MemAccessVector a1230[THREAD_TILE_M];

  // #pragma unroll THREAD_TILE_M
  for (int m_ = 0; m_ < THREAD_TILE_M; m_ += 1) {
    sumHalf8(reinterpret_cast<half2*>(&a2[m_]),
             reinterpret_cast<half2*>(&a3[m_]),
             reinterpret_cast<half2*>(&a23[m_]), 1.0f);
    
    sumHalf8(reinterpret_cast<half2*>(&a23[m_]),
             reinterpret_cast<half2*>(&a0[m_]),
             reinterpret_cast<half2*>(&a230[m_]), -1.0f);
    
    sumHalf8(reinterpret_cast<half2*>(&a0[m_]),
             reinterpret_cast<half2*>(&a2[m_]),
             reinterpret_cast<half2*>(&a02[m_]), -1.0f);
    
    sumHalf8(reinterpret_cast<half2*>(&a1[m_]),
             reinterpret_cast<half2*>(&a230[m_]),
             reinterpret_cast<half2*>(&a1230[m_]), -1.0f);
  }

  #pragma unroll THREAD_TILE_M
  for (int m_ = 0; m_ < THREAD_TILE_M; m_ += 1) {
    uint n = bid_n + threadIdx.x * MemAccessVectorElems;
    uint m = bid_m + threadIdx.y * THREAD_TILE_M + THREADS_Y * m_;
    uint idx = m * N + n;
    const uint sizeA = M*N;
    uint write_idx = m * N/2 + n;
    uint write_idx2 = m * N + n;

    // A_out[0*sizeA/4 + write_idx] = a0;
    // A_out[1*sizeA/4 + write_idx] = a1;
    *(MemAccessVector*)&A_out[2*sizeA/4 + write_idx2] = a230[m_];
    *(MemAccessVector*)&A_out[2*sizeA/4 + write_idx2 + N/2] = a02[m_];
    *(MemAccessVector*)&A_out[4*sizeA/4 + write_idx] = a23[m_];
    *(MemAccessVector*)&A_out[5*sizeA/4 + write_idx] = a1230[m_];
    // A_out[6*sizeA/4 + write_idx] = a3;
  }
}

template <
    typename StrassenMiGroup,
    typename ScheduleStrassenGroups,
    /// Element type for A matrix operand
    typename ElementA_,
    /// Layout type for A matrix operand
    typename LayoutA_,
    /// Element type for B matrix operand
    typename ElementB_,
    /// Layout type for B matrix operand
    typename LayoutB_,
    /// Element type for C and D matrix operands
    typename ElementC_,
    /// Layout type for C and D matrix operands
    typename LayoutC_,
    /// Element type for internal accumulation
    typename ElementAccumulator_,
    /// Operator class tag
    typename OperatorClass_,
    /// Tag indicating architecture to tune for
    typename ArchTag_,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape_,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape_,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape_,
    /// Epilogue output operator
    typename EpilogueOutputOp_,
    typename InterimEpilogueOp_,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle_,
    /// Number of stages used in the pipelined mainloop
    int Stages,
    /// Access granularity of A matrix in units of elements
    int AlignmentA,
    /// Access granularity of B matrix in units of elements
    int AlignmentB,
    /// If true, kernel supports split-K with serial reduction
    bool SplitKSerial,
    bool SubGemmParallel,
    /// Operation performed by GEMM
    typename Operator_,
    /// Gather operand A by using an index array
    bool GatherA,
    /// Gather operand B by using an index array
    bool GatherB,
    /// Scatter result D by using an index array
    bool ScatterD,
    /// Permute result D
    typename PermuteDLayout>
class StrassenGemm<StrassenType::NormalGlobalPreSumLevel1, StrassenMiGroup, ScheduleStrassenGroups, ElementA_, LayoutA_, ElementB_, LayoutB_, ElementC_, LayoutC_, ElementAccumulator_, OperatorClass_, ArchTag_, ThreadblockShape_, WarpShape_, InstructionShape_, EpilogueOutputOp_, InterimEpilogueOp_, ThreadblockSwizzle_, Stages, AlignmentA, AlignmentB, SplitKSerial, SubGemmParallel, Operator_, GatherA, GatherB, ScatterD, PermuteDLayout> {
 public:
  using ElementA = ElementA_;
  using LayoutA = LayoutA_;
  using TensorRefA = TensorRef<ElementA const, LayoutA>;
  using ElementB = ElementB_;
  using LayoutB = LayoutB_;
  using TensorRefB = TensorRef<ElementB const, LayoutB>;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  using TensorRefC = TensorRef<ElementC const, LayoutC>;
  using TensorRefD = TensorRef<ElementC, LayoutC>;
  using ElementAccumulator = ElementAccumulator_;
  using OperatorClass = OperatorClass_;
  using ArchTag = ArchTag_;
  using ThreadblockShape = ThreadblockShape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  using EpilogueOutputOp = EpilogueOutputOp_;
  using InterimEpilogueOp = InterimEpilogueOp_;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  using Operator = Operator_;
  static int const kStages = Stages;
  static int const kAlignmentA = AlignmentA;
  static int const kAlignmentB = AlignmentB;
  static int const kAlignmentC = EpilogueOutputOp::kCount;
  static bool const kSplitKSerial = SplitKSerial;
  static ComplexTransform const kTransformA = ComplexTransform::kNone;
  static ComplexTransform const kTransformB = ComplexTransform::kNone;

  /// Define the kernel
  template<MmaStrassen::Type MmaKind, typename MmaStrassenConsts>
  using GemmKernel_ = typename kernel::DefaultStrassenGemm<
    ElementA,
    LayoutA,
    kAlignmentA,
    ElementB,
    LayoutB,
    kAlignmentB,
    ElementC,
    LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    MmaKind,
    typename StrassenMiGroup::Group0,
    MmaStrassen::Consts<1,1,1,1>,
    EpilogueOutputOp,
    InterimEpilogueOp,
    ThreadblockSwizzle,
    kStages,
    kSplitKSerial,SubGemmParallel,
    Operator,
    SharedMemoryClearOption::kNone,
    GatherA,
    GatherB,
    ScatterD,
    PermuteDLayout
  >;
  using GemmKernelM0 = typename GemmKernel_<MmaStrassen::Type::GlobalPreSumLevel1_M0, MmaStrassen::Consts<1,1,1,1>>::GemmKernel;
  using GemmKernelM1 = typename GemmKernel_<MmaStrassen::Type::GlobalPreSumLevel1_M1, MmaStrassen::Consts<1,1,1,1>>::GemmKernel;
  using GemmKernelM2 = typename GemmKernel_<MmaStrassen::Type::GlobalPreSumLevel1_M2, MmaStrassen::Consts<1,1,1,1>>::GemmKernel;
  using GemmKernelM3 = typename GemmKernel_<MmaStrassen::Type::GlobalPreSumLevel1_M3, MmaStrassen::Consts<1,1,1,1>>::GemmKernel;
  using GemmKernelM4 = typename GemmKernel_<MmaStrassen::Type::GlobalPreSumLevel1_M4, MmaStrassen::Consts<1,1,1,1>>::GemmKernel;
  using GemmKernelM5 = typename GemmKernel_<MmaStrassen::Type::GlobalPreSumLevel1_M5, MmaStrassen::Consts<1,1,1,1>>::GemmKernel;
  using GemmKernelM6 = typename GemmKernel_<MmaStrassen::Type::GlobalPreSumLevel1_M6, MmaStrassen::Consts<1,1,1,1>>::GemmKernel;

  /// Argument structure
  struct Arguments {

    //
    // Data members
    //

    GemmCoord problem_size;
    TensorRef<ElementA const, LayoutA> ref_A;
    TensorRef<ElementB const, LayoutB> ref_B;
    TensorRef<ElementC const, LayoutC> ref_C;
    TensorRef<ElementC, LayoutC> ref_D;
    typename EpilogueOutputOp::Params epilogue;
    int split_k_slices;
    // For gather+scatter operations
    int const *gather_A_indices;
    int const *gather_B_indices;
    int const *scatter_D_indices;

    //
    // Methods
    //

    /// Default ctor
    CUTLASS_HOST_DEVICE
    Arguments(): problem_size(0, 0, 0), split_k_slices(1) {

    }

    /// Constructs an Arguments structure
    CUTLASS_HOST_DEVICE
    Arguments(
      GemmCoord problem_size_,
      TensorRef<ElementA const, LayoutA> ref_A_,
      TensorRef<ElementB const, LayoutB> ref_B_,
      TensorRef<ElementC const, LayoutC> ref_C_,
      TensorRef<ElementC, LayoutC> ref_D_,
      typename EpilogueOutputOp::Params epilogue_ =
        typename EpilogueOutputOp::Params(),
      int split_k_slices = 1,
      int const *gather_A_indices_ = nullptr,
      int const *gather_B_indices_ = nullptr,
      int const *scatter_D_indices_ = nullptr
    ):
      problem_size(problem_size_),
      ref_A(ref_A_),
      ref_B(ref_B_),
      ref_C(ref_C_),
      ref_D(ref_D_),
      epilogue(epilogue_),
      split_k_slices(split_k_slices),
      gather_A_indices(gather_A_indices_),
      gather_B_indices(gather_B_indices_),
      scatter_D_indices(scatter_D_indices_) {

    }
  };

private:

  /// Kernel parameters object
  typename GemmKernelM0::Params paramsM0_;
  typename GemmKernelM1::Params paramsM1_;
  typename GemmKernelM2::Params paramsM2_;
  typename GemmKernelM3::Params paramsM3_;
  typename GemmKernelM4::Params paramsM4_;
  typename GemmKernelM5::Params paramsM5_;
  typename GemmKernelM6::Params paramsM6_;

public:

  /// Constructs the GEMM.
  StrassenGemm() {
    /*int kM = GemmKernel_::Mma_::MmaCore::LaneMmaShape::kM;
    int kN = GemmKernel_::Mma_::MmaCore::LaneMmaShape::kN;
    printf("361: LaneMmaShape %d %d \n", kM, kN);
    using WarpCount = typename GemmKernel_::Mma_::MmaCore::WarpCount;
    printf("363: WarpCount %d %d\n", WarpCount::kM, WarpCount::kN);

    int WarpNumThreadsM = GemmKernel_::Mma_::MmaCore::WarpNumThreadsM;
    int WarpNumThreadsN = GemmKernel_::Mma_::MmaCore::WarpNumThreadsN;
    printf("367: %d %d\n", WarpNumThreadsM, WarpNumThreadsN);

    using WarpShape = typename GemmKernel_::Mma_::MmaCore::MmaWarpSimt::Shape;
    using PolicyWarpShape = typename GemmKernel_::Mma_::MmaCore::MmaWarpSimt::Policy::WarpShape;
    printf("368: %d %d; %d %d\n", WarpShape::kM, WarpShape::kN, PolicyWarpShape::kRow, PolicyWarpShape::kColumn);

    int MmaCorekThreads = GemmKernel_::Mma_::MmaCore::kThreads;
    printf("MmaCore::kThreads %d\n", MmaCorekThreads);

    using ThreadShape = typename GemmKernel_::Mma_::MmaCore::MmaWarpSimt::ThreadMma::Shape;
    printf("374: %d %d\n", ThreadShape::kM, ThreadShape::kN);

    using MmaCoreShape = typename GemmKernel_::Mma_::MmaCore::Shape;
    printf("377: %d %d %d\n", MmaCoreShape::kM, MmaCoreShape::kN, MmaCoreShape::kK);

    using ThreadMapA = typename GemmKernel_::Mma_::MmaCore::IteratorThreadMapA;
    printf("380: %d %d\n", ThreadMapA::Iterations::kCount, ThreadMapA::kElementsPerAccess);
    printf("381: %d %d %d\n", ThreadMapA::kThreads, ThreadMapA::Detail::ShapeVec::kContiguous, ThreadMapA::Detail::ShapeVec::kStrided);

    uint WarpCountkCount = GemmKernel::WarpCount::kCount;
    printf("384: WarpCountkCount %d\n", WarpCountkCount);*/
  }

  /// Determines whether the GEMM can execute the given problem.
  static Status can_implement(Arguments const &args) {

    if (!kSplitKSerial && args.split_k_slices > 1) {
      return Status::kErrorInvalidProblem;
    }

    Status status = GemmKernelM0::can_implement(
      args.problem_size,
      args.ref_A.non_const_ref(),
      args.ref_B.non_const_ref(),
      args.ref_C.non_const_ref(),
      args.ref_D
    );

    if (status != Status::kSuccess) {
      return status;
    }

    return Status::kSuccess;
  }

  /// Gets the workspace size
  static size_t get_workspace_size(Arguments const &args) {
    printf("1643\n");
    size_t bytes = 0;

    // Determine grid shape
    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord tiled_shape = threadblock_swizzle.get_tiled_shape(
      args.problem_size,
      {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
      args.split_k_slices);

    //For A and B pre-sums
    bytes += sizeof(int) * 7 * (
                                size_t(args.problem_size.m()/2) * size_t(args.problem_size.k()/2) +
                                size_t(args.problem_size.n()/2) * size_t(args.problem_size.k()/2)
                               );
    //For M0 to M6
    bytes += sizeof(int) * 7 * (size_t(args.problem_size.m()/2) * size_t(args.problem_size.n()/2));

    //For semaphores for presum sync
    bytes += sizeof(int) * 7 * (size_t(tiled_shape.m()) * size_t(tiled_shape.n()));

    //For semaphores for postsum sync
    bytes += sizeof(int) * 7 * (size_t(tiled_shape.m()) * size_t(tiled_shape.n()));

    if (kSplitKSerial && args.split_k_slices > 1) {
      //For splitk semaphores
      bytes += sizeof(int) * 7 * size_t(tiled_shape.m()) * size_t(tiled_shape.n());
    }

    return bytes;
  }

  /// Initializes GEMM state from arguments.
  Status initialize(Arguments const &args, void *workspace = nullptr, cudaStream_t stream = nullptr) {

    // Determine grid shape
    ThreadblockSwizzle threadblock_swizzle;
    auto new_problem_size = GemmCoord{args.problem_size.m()/2, args.problem_size.n()/2, args.problem_size.k()};
    cutlass::gemm::GemmCoord grid_shape = threadblock_swizzle.get_tiled_shape(
      new_problem_size,
      {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
      args.split_k_slices);

    cutlass::gemm::GemmCoord orig_grid_shape = threadblock_swizzle.get_tiled_shape(
      args.problem_size,
      {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
      args.split_k_slices);

    if (!workspace) {
      return Status::kErrorWorkspaceNull;
    }

    size_t bytes = get_workspace_size(args);

    cudaError_t result = cudaMemsetAsync(workspace, 0, bytes, stream);

    if (result != cudaSuccess) {
      return Status::kErrorInternal;
    }

    if (SubGemmParallel || kSplitKSerial) {
      if (SubGemmParallel || args.split_k_slices > 1) {
        if (!workspace) {
          return Status::kErrorWorkspaceNull;
        }

        size_t bytes = get_workspace_size(args);

        cudaError_t result = cudaMemsetAsync(workspace, 0, bytes, stream);

        if (result != cudaSuccess) {
          return Status::kErrorInternal;
        }
      }
    }
    else {

      if (args.split_k_slices > 1) {
        return Status::kErrorInvalidProblem;
      }
    }

    // Initialize the Params structure
    paramsM0_ = typename GemmKernelM0::Params{
      args.problem_size,
      grid_shape,
      args.ref_A.non_const_ref(),
      args.ref_B.non_const_ref(),
      args.ref_C.non_const_ref(),
      args.ref_D,
      args.epilogue,
      static_cast<int *>(workspace),// + (SubGemmParallel? 0 : 0)*(orig_grid_shape.m() * orig_grid_shape.n()),
      args.gather_A_indices,
      args.gather_B_indices,
      args.scatter_D_indices
    };

    paramsM1_ = typename GemmKernelM1::Params{
      args.problem_size,
      grid_shape,
      args.ref_A.non_const_ref(),
      args.ref_B.non_const_ref(),
      args.ref_C.non_const_ref(),
      args.ref_D,
      args.epilogue,
      static_cast<int *>(workspace),// + (SubGemmParallel? 0 : 1) * (orig_grid_shape.m() * orig_grid_shape.n()),
      args.gather_A_indices,
      args.gather_B_indices,
      args.scatter_D_indices
    };

    paramsM2_ = typename GemmKernelM2::Params{
      args.problem_size,
      grid_shape,
      args.ref_A.non_const_ref(),
      args.ref_B.non_const_ref(),
      args.ref_C.non_const_ref(),
      args.ref_D,
      args.epilogue,
      static_cast<int *>(workspace),// + (SubGemmParallel? 0 : 2)*(orig_grid_shape.m() * orig_grid_shape.n()),
      args.gather_A_indices,
      args.gather_B_indices,
      args.scatter_D_indices
    };

    paramsM3_ = typename GemmKernelM3::Params{
      args.problem_size,
      grid_shape,
      args.ref_A.non_const_ref(),
      args.ref_B.non_const_ref(),
      args.ref_C.non_const_ref(),
      args.ref_D,
      args.epilogue,
      static_cast<int *>(workspace),// + (SubGemmParallel? 0 : 3)*(orig_grid_shape.m() * orig_grid_shape.n()),
      args.gather_A_indices,
      args.gather_B_indices,
      args.scatter_D_indices
    };

    paramsM4_ = typename GemmKernelM4::Params{
      args.problem_size,
      grid_shape,
      args.ref_A.non_const_ref(),
      args.ref_B.non_const_ref(),
      args.ref_C.non_const_ref(),
      args.ref_D,
      args.epilogue,
      static_cast<int *>(workspace),// + (SubGemmParallel? 0: 4)*(orig_grid_shape.m() * orig_grid_shape.n()),
      args.gather_A_indices,
      args.gather_B_indices,
      args.scatter_D_indices
    };

    paramsM5_ = typename GemmKernelM5::Params{
      args.problem_size,
      grid_shape,
      args.ref_A.non_const_ref(),
      args.ref_B.non_const_ref(),
      args.ref_C.non_const_ref(),
      args.ref_D,
      args.epilogue,
      static_cast<int *>(workspace),// + (SubGemmParallel? 0 : 5) *(orig_grid_shape.m() * orig_grid_shape.n()),
      args.gather_A_indices,
      args.gather_B_indices,
      args.scatter_D_indices
    };

    paramsM6_ = typename GemmKernelM6::Params{
      args.problem_size,
      grid_shape,
      args.ref_A.non_const_ref(),
      args.ref_B.non_const_ref(),
      args.ref_C.non_const_ref(),
      args.ref_D,
      args.epilogue,
      static_cast<int *>(workspace),// + (SubGemmParallel? 0 : 6) *(orig_grid_shape.m() * orig_grid_shape.n()),
      args.gather_A_indices,
      args.gather_B_indices,
      args.scatter_D_indices
    };

    return Status::kSuccess;
  }

  /// Lightweight update given a subset of arguments
  Status update(Arguments const &args, void *workspace = nullptr) {
    assert(false);
    if (kSplitKSerial && args.split_k_slices > 1) {
      if (!workspace) {
        return Status::kErrorWorkspaceNull;
      }
    }

    paramsM0_.ref_A.reset(args.ref_A.non_const_ref().data());
    paramsM0_.ref_B.reset(args.ref_B.non_const_ref().data());
    paramsM0_.ref_C.reset(args.ref_C.non_const_ref().data());
    paramsM0_.ref_D.reset(args.ref_D.data());
    paramsM0_.output_op = args.epilogue;
    paramsM0_.semaphore = static_cast<int *>(workspace);


    paramsM1_.ref_A.reset(args.ref_A.non_const_ref().data());
    paramsM1_.ref_B.reset(args.ref_B.non_const_ref().data());
    paramsM1_.ref_C.reset(args.ref_C.non_const_ref().data());
    paramsM1_.ref_D.reset(args.ref_D.data());
    paramsM1_.output_op = args.epilogue;
    paramsM1_.semaphore = static_cast<int *>(workspace);


    paramsM2_.ref_A.reset(args.ref_A.non_const_ref().data());
    paramsM2_.ref_B.reset(args.ref_B.non_const_ref().data());
    paramsM2_.ref_C.reset(args.ref_C.non_const_ref().data());
    paramsM2_.ref_D.reset(args.ref_D.data());
    paramsM2_.output_op = args.epilogue;
    paramsM2_.semaphore = static_cast<int *>(workspace);


    paramsM3_.ref_A.reset(args.ref_A.non_const_ref().data());
    paramsM3_.ref_B.reset(args.ref_B.non_const_ref().data());
    paramsM3_.ref_C.reset(args.ref_C.non_const_ref().data());
    paramsM3_.ref_D.reset(args.ref_D.data());
    paramsM3_.output_op = args.epilogue;
    paramsM3_.semaphore = static_cast<int *>(workspace);

    paramsM4_.ref_A.reset(args.ref_A.non_const_ref().data());
    paramsM4_.ref_B.reset(args.ref_B.non_const_ref().data());
    paramsM4_.ref_C.reset(args.ref_C.non_const_ref().data());
    paramsM4_.ref_D.reset(args.ref_D.data());
    paramsM4_.output_op = args.epilogue;
    paramsM4_.semaphore = static_cast<int *>(workspace);

    paramsM5_.ref_A.reset(args.ref_A.non_const_ref().data());
    paramsM5_.ref_B.reset(args.ref_B.non_const_ref().data());
    paramsM5_.ref_C.reset(args.ref_C.non_const_ref().data());
    paramsM5_.ref_D.reset(args.ref_D.data());
    paramsM5_.output_op = args.epilogue;
    paramsM5_.semaphore = static_cast<int *>(workspace);

    paramsM6_.ref_A.reset(args.ref_A.non_const_ref().data());
    paramsM6_.ref_B.reset(args.ref_B.non_const_ref().data());
    paramsM6_.ref_C.reset(args.ref_C.non_const_ref().data());
    paramsM6_.ref_D.reset(args.ref_D.data());
    paramsM6_.output_op = args.epilogue;
    paramsM6_.semaphore = static_cast<int *>(workspace);

    return Status::kSuccess;
  }

  // template<typename GemmKernel>
  // cudaError_t run(typename GemmKernel::Params& params, cudaStream_t stream = nullptr) {
  //   ThreadblockSwizzle threadblock_swizzle;

  //   dim3 grid = threadblock_swizzle.get_grid_shape(params.grid_tiled_shape);
  //   dim3 block(GemmKernel::kThreadCount, 1, 1);
  //   params.run += 1;
  //   cudaError_t result;
  //   int smem_size = int(sizeof(typename GemmKernel::SharedStorage));
  //   // grid.x = 2*grid.x; grid.y = 2*grid.y;
  //   // int epilogue_size = int(sizeof(typename GemmKernel::Epilogue::SharedStorage));
  //   // std::cout << "504 " << smem_size << " " << GemmKernel::kThreadCount << " " << grid.x << " " << grid.y << " " << grid.z << " " << epilogue_size <<std::endl;
  //   if (smem_size >= (48 << 10)) {
  //     result = cudaFuncSetAttribute(Kernel<GemmKernel>,
  //                                   cudaFuncAttributeMaxDynamicSharedMemorySize,
  //                                   smem_size);

  //     if (result != cudaSuccess) {
  //       return result;
  //     }
  //   }

  //   cutlass::Kernel<GemmKernel><<<grid, block, smem_size, stream>>>(params);
  //   result = cudaGetLastError();
  //   if (paramsM0_.problem_size.m() <= 4096)
  //     result = cudaStreamSynchronize(stream);
  //   return result;
  // }

  template<typename FusedOperator1, typename FusedOperator2, typename FusedOperator3,
           typename FusedOperator4, typename FusedOperator5, typename FusedOperator6,
           typename FusedOperator7>
  cudaError_t run(typename FusedOperator1::Operator1::Params& params1,
                  typename FusedOperator2::Operator1::Params& params2,
                  typename FusedOperator3::Operator1::Params& params3,
                  typename FusedOperator4::Operator1::Params& params4,
                  typename FusedOperator5::Operator1::Params& params5,
                  typename FusedOperator6::Operator1::Params& params6,
                  typename FusedOperator7::Operator1::Params& params7,
                  cudaStream_t stream = nullptr) {
    ThreadblockSwizzle threadblock_swizzle;

    dim3 grid = threadblock_swizzle.get_grid_shape(params1.grid_tiled_shape);
    dim3 block(FusedOperator1::Operator1::kThreadCount, 1, 1);
    params1.run += 1; params2.run += 1; params3.run += 1; params4.run += 1; params5.run += 1;
    params6.run += 1; params7.run += 1;
    cudaError_t result;
    int smem_size = int(sizeof(typename FusedOperator1::Operator1::SharedStorage));
    // int epilogue_size = int(sizeof(typename GemmKernel::Epilogue::SharedStorage));
    dim3 origGrid = grid;
    grid.z = 7*grid.z;
    // std::cout << "504 " << smem_size << " " << GemmKernel::kThreadCount << " " << grid.x << " " << grid.y << " " << grid.z << " " << epilogue_size <<std::endl;
    if (smem_size >= (48 << 10)) {
        result = cudaFuncSetAttribute(Kernel7<FusedOperator1, FusedOperator2, FusedOperator3, FusedOperator4, FusedOperator5, FusedOperator6, FusedOperator7>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      smem_size);

      if (result != cudaSuccess) {
        return result;
      }
    }

    cutlass::Kernel7<FusedOperator1, FusedOperator2, FusedOperator3, FusedOperator4, FusedOperator5, FusedOperator6, FusedOperator7><<<grid, block, smem_size, stream>>>(params1, params2, params3, params4, params5, params6, params7, origGrid);
    result = cudaGetLastError();
    return result;
  }

  template<typename FusedOperator1, typename FusedOperator2, typename FusedOperator3,
           typename FusedOperator4, typename FusedOperator5>
  cudaError_t run(typename FusedOperator1::Operator1::Params& params1,
                  typename FusedOperator2::Operator1::Params& params2,
                  typename FusedOperator3::Operator1::Params& params3,
                  typename FusedOperator4::Operator1::Params& params4,
                  typename FusedOperator5::Operator1::Params& params5,
                  cudaStream_t stream = nullptr) {
    ThreadblockSwizzle threadblock_swizzle;

    dim3 grid = threadblock_swizzle.get_grid_shape(params1.grid_tiled_shape);
    dim3 block(FusedOperator1::Operator1::kThreadCount, 1, 1);
    params1.run += 1; params2.run += 1; params3.run += 1; params4.run += 1; params5.run += 1;
    cudaError_t result;
    int smem_size = int(sizeof(typename FusedOperator1::Operator1::SharedStorage));
    // int epilogue_size = int(sizeof(typename GemmKernel::Epilogue::SharedStorage));
    dim3 origGrid = grid;
    grid.z = 5*grid.z;
    // std::cout << "504 " << smem_size << " " << GemmKernel::kThreadCount << " " << grid.x << " " << grid.y << " " << grid.z << " " << epilogue_size <<std::endl;
    if (smem_size >= (48 << 10)) {
        result = cudaFuncSetAttribute(Kernel5<FusedOperator1, FusedOperator2, FusedOperator3, FusedOperator4, FusedOperator5>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      smem_size);

      if (result != cudaSuccess) {
        return result;
      }
    }

    cutlass::Kernel5<FusedOperator1, FusedOperator2, FusedOperator3, FusedOperator4, FusedOperator5><<<grid, block, smem_size, stream>>>(params1, params2, params3, params4, params5, origGrid);
    result = cudaGetLastError();
    return result;
  }

  template<typename FusedOperator1, typename FusedOperator2, typename FusedOperator3, typename FusedOperator4>
  cudaError_t run(typename FusedOperator1::Operator1::Params& params1,
                  typename FusedOperator2::Operator1::Params& params2,
                  typename FusedOperator3::Operator1::Params& params3,
                  typename FusedOperator4::Operator1::Params& params4,
                  cudaStream_t stream = nullptr) {
    ThreadblockSwizzle threadblock_swizzle;

    dim3 grid = threadblock_swizzle.get_grid_shape(params1.grid_tiled_shape);
    dim3 block(FusedOperator1::Operator1::kThreadCount, 1, 1);
    params1.run += 1; params2.run += 1; params3.run += 1; params4.run += 1;
    cudaError_t result;
    int smem_size = int(sizeof(typename FusedOperator1::Operator1::SharedStorage));
    // int epilogue_size = int(sizeof(typename GemmKernel::Epilogue::SharedStorage));
    dim3 origGrid = grid;
    grid.z = 4*grid.z;
    // std::cout << "504 " << smem_size << " " << GemmKernel::kThreadCount << " " << grid.x << " " << grid.y << " " << grid.z << " " << epilogue_size <<std::endl;
    if (smem_size >= (48 << 10)) {
        result = cudaFuncSetAttribute(Kernel4<FusedOperator1, FusedOperator2, FusedOperator3, FusedOperator4>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      smem_size);

      if (result != cudaSuccess) {
        return result;
      }
    }

    cutlass::Kernel4<FusedOperator1, FusedOperator2, FusedOperator3, FusedOperator4><<<grid, block, smem_size, stream>>>(params1, params2, params3, params4, origGrid);
    result = cudaGetLastError();
    return result;
  }

  template<typename FusedOperator1, typename FusedOperator2, typename FusedOperator3>
  cudaError_t run(typename FusedOperator1::Operator1::Params& params1,
                  typename FusedOperator2::Operator1::Params& params2,
                  typename FusedOperator3::Operator1::Params& params3,
                  cudaStream_t stream = nullptr) {
    ThreadblockSwizzle threadblock_swizzle;

    dim3 grid = threadblock_swizzle.get_grid_shape(params1.grid_tiled_shape);
    dim3 block(FusedOperator1::Operator1::kThreadCount, 1, 1);
    params1.run += 1; params2.run += 1; params3.run += 1;
    cudaError_t result;
    int smem_size = int(sizeof(typename FusedOperator1::Operator1::SharedStorage));
    // int epilogue_size = int(sizeof(typename GemmKernel::Epilogue::SharedStorage));
    dim3 origGrid = grid;
    grid.z = 3*grid.z;
    // std::cout << "504 " << smem_size << " " << GemmKernel::kThreadCount << " " << grid.x << " " << grid.y << " " << grid.z << " " << epilogue_size <<std::endl;
    if (smem_size >= (48 << 10)) {
        result = cudaFuncSetAttribute(Kernel3<FusedOperator1, FusedOperator2, FusedOperator3>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      smem_size);

      if (result != cudaSuccess) {
        return result;
      }
    }

    cutlass::Kernel3<FusedOperator1, FusedOperator2, FusedOperator3><<<grid, block, smem_size, stream>>>(params1, params2, params3, origGrid);
    result = cudaGetLastError();
    return result;
  }

  template<typename FusedOperator1, typename FusedOperator2>
  cudaError_t run(typename FusedOperator1::Operator1::Params& params1,
                  typename FusedOperator2::Operator1::Params& params2,
                  cudaStream_t stream = nullptr) {
    ThreadblockSwizzle threadblock_swizzle;

    dim3 grid = threadblock_swizzle.get_grid_shape(params1.grid_tiled_shape);
    dim3 block(FusedOperator1::Operator1::kThreadCount, 1, 1);
    params1.run += 1; params2.run += 1;
    cudaError_t result;
    int smem_size = int(sizeof(typename FusedOperator1::Operator1::SharedStorage));
    // int epilogue_size = int(sizeof(typename GemmKernel::Epilogue::SharedStorage));
    dim3 origGrid = grid;
    grid.z = 2*grid.z;
    // std::cout << "504 " << smem_size << " " << grid.x << " " << grid.y << " " << grid.z << " " << std::endl;
    if (smem_size >= (48 << 10)) {
        result = cudaFuncSetAttribute(Kernel2<FusedOperator1, FusedOperator2>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      smem_size);

      if (result != cudaSuccess) {
        return result;
      }
    }

    cutlass::Kernel2<FusedOperator1, FusedOperator2><<<grid, block, smem_size, stream>>>(params1, params2, origGrid);
    result = cudaGetLastError();
    return result;
  }

  template<typename FusedOperator>
  cudaError_t run(typename FusedOperator::Operator1::Params& params1,
                  cudaStream_t stream = nullptr) {
    ThreadblockSwizzle threadblock_swizzle;

    dim3 grid = threadblock_swizzle.get_grid_shape(params1.grid_tiled_shape);
    dim3 block(FusedOperator::Operator1::kThreadCount, 1, 1);
    params1.run += 1;
    cudaError_t result;
    dim3 origGrid = grid;
    grid.z = 1*grid.z;
    int smem_size = int(sizeof(typename FusedOperator::Operator1::SharedStorage));
    // int epilogue_size = int(sizeof(typename FusedOperator::Epilogue::SharedStorage));
    // std::cout << "504 " << smem_size << " " << FusedOperator::Operator1::kThreadCount << " " << grid.x << " " << grid.y << " " << grid.z <<std::endl;
    if (smem_size >= (48 << 10)) {
        result = cudaFuncSetAttribute(Kernel1<FusedOperator>,
                                      cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      smem_size);

      if (result != cudaSuccess) {
        return result;
      }
    }

    cutlass::Kernel1<FusedOperator><<<grid, block, smem_size, stream>>>(params1, origGrid);
    result = cudaGetLastError();
    return result;
  }

  /// Runs the kernel using initialized state.
  Status run(cudaStream_t* streams = nullptr, int num_streams = 0) {
    cudaError_t result;

    if (SubGemmParallel) {
      cudaStream_t defaultStreams[7] = {0};
      if (streams == nullptr || num_streams == 0) {
        streams = defaultStreams;
      } else {
        assert (num_streams >= 7);
      }
      // result = run<GemmKernelM0>(paramsM0_, streams[0]);
      // if (result != cudaSuccess)
      // {printf("Error at %d: %s\n", __LINE__, cudaGetErrorString(result)); return Status::kErrorInternal;}

      // result = run<GemmKernelM1>(paramsM1_, streams[1]);
      // if (result != cudaSuccess)
      // {printf("Error at %d: %s\n", __LINE__, cudaGetErrorString(result)); return Status::kErrorInternal;}

      // result = run<GemmKernelM2>(paramsM2_, streams[2]);
      // if (result != cudaSuccess)
      // {printf("Error at %d: %s\n", __LINE__, cudaGetErrorString(result)); return Status::kErrorInternal;}

      // result = run<GemmKernelM3>(paramsM3_, streams[3]);
      // if (result != cudaSuccess)
      // {printf("Error at %d: %s\n", __LINE__, cudaGetErrorString(result)); return Status::kErrorInternal;}

      // result = run<GemmKernelM4>(paramsM4_, streams[4]);
      // if (result != cudaSuccess)
      // {printf("Error at %d: %s\n", __LINE__, cudaGetErrorString(result)); return Status::kErrorInternal;}

      // result = run<GemmKernelM5>(paramsM5_, streams[5]);
      // if (result != cudaSuccess)
      // {printf("Error at %d: %s\n", __LINE__, cudaGetErrorString(result)); return Status::kErrorInternal;}

      // result = run<GemmKernelM6>(paramsM6_, streams[6]);
      // if (result != cudaSuccess)
      // {printf("Error at %d: %s\n", __LINE__, cudaGetErrorString(result)); return Status::kErrorInternal;}
    } else {
      cudaStream_t stream1, stream2, stream3;
      if (streams == nullptr || num_streams == 0) {
        stream1 = 0;
        stream2 = 0;
        stream3 = 0;
      } else {
        stream1 = streams[0];
        stream2 = streams[1];
        stream3 = streams[2];
      }
      constexpr uint64_t schedule = MmaStrassen::Schedule230F15FF46F();

      if (paramsM0_.run < 1) {
        const int THREAD_TILE_N = 2;
        const int THREAD_TILE_M = 2;
        const int MemAccessVectorElems = 8;
        dim3 block = {16,16};
        dim3 grid = {paramsM0_.problem_size.n()/(2*THREAD_TILE_N*MemAccessVectorElems*block.x),
                     paramsM0_.problem_size.k()/(2*THREAD_TILE_M*block.y)};

        const size_t sizeA = paramsM0_.problem_size.m() * paramsM0_.problem_size.k();

        ElementA* A_out = (ElementA*)paramsM0_.semaphore;
        ElementB* B_out = (ElementB*)(((char*)paramsM0_.semaphore) + (7 * sizeA/4 * sizeof(ElementA)));

        presumB<ElementB, THREAD_TILE_M, THREAD_TILE_N><<<grid, block, 0, stream1>>>(paramsM0_.ref_B.data(), B_out, paramsM0_.problem_size.k(), paramsM0_.problem_size.n(), schedule);
        result = cudaGetLastError();
        result = cudaStreamSynchronize(stream1);
        // std::cout << 2412 << " " << paramsM0_.run << std::endl;
        if (result != cudaSuccess)
        {printf("Error at %d: %s\n", __LINE__, cudaGetErrorString(result)); return Status::kErrorInternal;}

// #ifndef PRESUM_OVERLAPPED
// #endif
      }

  

    if (false) { //paramsM0_.problem_size.m() > 10*1024
      // result = run3<10*1024+1, GemmKernelM2, GemmKernelM3, GemmKernelM0>(paramsM2_, paramsM3_, paramsM0_, stream1);
      // if (result != cudaSuccess)
      // {printf("Error at %d: %s\n", __LINE__, cudaGetErrorString(result)); return Status::kErrorInternal;}

      // result = run<10*1024+1, GemmKernelM1, GemmKernelM5>(paramsM1_, paramsM5_, stream2);
      // if (result != cudaSuccess)
      // {printf("Error at %d: %s\n", __LINE__, cudaGetErrorString(result)); return Status::kErrorInternal;}

      // result = run<10*1024+1, GemmKernelM4, GemmKernelM6>(paramsM4_, paramsM6_, stream1);
      // if (result != cudaSuccess)
      // {printf("Error at %d: %s\n", __LINE__, cudaGetErrorString(result)); return Status::kErrorInternal;}
    } //else if (paramsM0_.problem_size.m() == 10*1024) {
      // result = run3<10*1024, GemmKernelM2, GemmKernelM3, GemmKernelM0>
      //          (paramsM2_, paramsM3_, paramsM0_, stream1);
      // if (result != cudaSuccess)
      // {printf("Error at %d: %s\n", __LINE__, cudaGetErrorString(result)); return Status::kErrorInternal;}

      // result = run2x2<10*1024, GemmKernelM1, GemmKernelM5, GemmKernelM4, GemmKernelM6>
      //          (paramsM1_, paramsM5_, paramsM4_, paramsM6_, stream2);
      // if (result != cudaSuccess)
      // {printf("Error at %d: %s\n", __LINE__, cudaGetErrorString(result)); return Status::kErrorInternal;}
     else if (true || paramsM0_.problem_size.m() < 10*1024) {
      const uint64_t PresumM0 = (1 << MmaStrassen::GlobalPreSumLevel1_M2) | (1 << MmaStrassen::GlobalPreSumLevel1_M3) | (1 << MmaStrassen::GlobalPreSumLevel1_M4) | (1 << MmaStrassen::GlobalPreSumLevel1_M5); 
      const uint64_t PresumM1 = 0;
      // if (paramsM0_.run <= 1 && PresumM0 == 0) {
      if (PresumM0 == 0) {
        const int THREAD_TILE_N = 8;
        const int THREAD_TILE_M = 1;
        const int THREADS_X = 256;
        const int THREADS_Y = 1;
        const dim3 block = {THREADS_X, THREADS_Y};
        const dim3 grid = {paramsM0_.problem_size.k()/(2*block.x*THREAD_TILE_N),
                      paramsM0_.problem_size.m()/(2*block.y*THREAD_TILE_M)};

        ElementA* A_out = (ElementA*)paramsM0_.semaphore;

        presumA<ElementA, THREADS_Y, THREADS_X, THREAD_TILE_M, THREAD_TILE_N><<<grid, block, 0, stream1>>>(paramsM0_.ref_A.data(), A_out, paramsM0_.problem_size.m(), paramsM0_.problem_size.k());
        // result = cudaGetLastError();
        // result = cudaStreamSynchronize(stream1);
        // if (result != cudaSuccess)
        // {printf("Error at %d: %s\n", __LINE__, cudaGetErrorString(result)); return Status::kErrorInternal;}
      }

      const bool DoGlobalSync = false;

      {
        //TODO: Concat Ai|Ai+1 through columns when fused
        if (false) {
          //Work with splitk
          //4k x 4k x 4k
          //{M2},{M3},{M6},{M4},{M5}
          const uint64_t ReadWriteM0 = SetReadWriteM(MmaStrassen::RWGlobalContig, MmaStrassen::GlobalPreSumLevel1_M0, false, DoGlobalSync);
                                      //  SetReadWriteM(MmaStrassen::RWCombineMMALoop, MmaStrassen::GlobalPreSumLevel1_M1, false) |
                                      //  MmaStrassen::SetRWOutputC(0);
          const uint64_t ReadWriteM1 = SetReadWriteM(MmaStrassen::RWGlobalContig, MmaStrassen::GlobalPreSumLevel1_M0, false, DoGlobalSync) |
                                       MmaStrassen::SetRWOutputC(0);
          //V0 = M2+M0
          const uint64_t ReadWriteM2 = SetReadWriteM(MmaStrassen::RWGlobalContig, MmaStrassen::GlobalPreSumLevel1_M0, false, DoGlobalSync) |
                                       SetReadWriteM(MmaStrassen::RWGlobalContig, MmaStrassen::GlobalPreSumLevel1_M2, false, DoGlobalSync);

          //V1 = V0+M3
          const uint64_t ReadWriteM3 = SetReadWriteM(MmaStrassen::RWGlobalContig, MmaStrassen::GlobalPreSumLevel1_M2, false, DoGlobalSync) |
                                       SetReadWriteM(MmaStrassen::RWGlobalContig, MmaStrassen::GlobalPreSumLevel1_M3, false, DoGlobalSync);

          //C2 = V1-M6
          const uint64_t ReadWriteM6 = SetReadWriteM(MmaStrassen::RWGlobalContig, MmaStrassen::GlobalPreSumLevel1_M3, false, DoGlobalSync) |
                                       SetReadWriteM(MmaStrassen::RWUseSign,      MmaStrassen::GlobalPreSumLevel1_M6, true)  |
                                       MmaStrassen::SetRWOutputC(2);
          //C3 = V1 + M4
          const uint64_t ReadWriteM4 = SetReadWriteM(MmaStrassen::RWGlobalContig, MmaStrassen::GlobalPreSumLevel1_M3, false, DoGlobalSync) |
                                       SetReadWriteM(MmaStrassen::RWGlobalContig, MmaStrassen::GlobalPreSumLevel1_M4, false, DoGlobalSync) |
                                       MmaStrassen::SetRWOutputC(3);
          //C1 = V0 + M4 + M5
          const uint64_t ReadWriteM5 = SetReadWriteM(MmaStrassen::RWGlobalContig, MmaStrassen::GlobalPreSumLevel1_M2, false, DoGlobalSync) |
                                       SetReadWriteM(MmaStrassen::RWGlobalContig, MmaStrassen::GlobalPreSumLevel1_M4, false, DoGlobalSync) |
                                       MmaStrassen::SetRWOutputC(1);

          using Op0 = FusedOperator1<GemmKernelM0, schedule, PresumM0, ReadWriteM0>;
          using Op1 = FusedOperator1<GemmKernelM1, schedule, PresumM1, ReadWriteM1>;
          using Op2 = FusedOperator1<GemmKernelM2, schedule, 0, ReadWriteM2>;

          using Op3 = FusedOperator1<GemmKernelM3, schedule, 0, ReadWriteM3>;
          using Op4 = FusedOperator1<GemmKernelM4, schedule, 0, ReadWriteM4>;
          using Op5 = FusedOperator1<GemmKernelM5, schedule, 0, ReadWriteM5>;
          using Op6 = FusedOperator1<GemmKernelM6, schedule, 0, ReadWriteM6>;
          // result = run<Op0, Op1, Op2, Op3>(paramsM0_, paramsM1_, paramsM2_, paramsM3_, stream1);
          // result = run<Op6, Op4, Op5>(paramsM6_, paramsM4_, paramsM5_, stream2);
          if (result != cudaSuccess)
          {printf("Error at %d: %s\n", __LINE__, cudaGetErrorString(result)); return Status::kErrorInternal;}

          return Status::kSuccess;
        }
        if (false) {
          //Work with splitk
          //4k x 4k x 4k
          //{M2,M3,M6,M4,M5}
          const uint64_t ReadWriteM0 = SetReadWriteM(MmaStrassen::RWGlobalContig, MmaStrassen::GlobalPreSumLevel1_M0, false, DoGlobalSync);
                                      //  SetReadWriteM(MmaStrassen::RWCombineMMALoop, MmaStrassen::GlobalPreSumLevel1_M1, false) |
                                      //  MmaStrassen::SetRWOutputC(0);
          const uint64_t ReadWriteM1 = SetReadWriteM(MmaStrassen::RWGlobalContig, MmaStrassen::GlobalPreSumLevel1_M0, false, DoGlobalSync) |
                                       MmaStrassen::SetRWOutputC(0);
          using Op0 = FusedOperator1<GemmKernelM0, schedule,
                                     PresumM0, ReadWriteM0>;
          using Op1 = FusedOperator1<GemmKernelM1, schedule,
                                     PresumM1, ReadWriteM1>;
          // result = run<Op0, Op1> (paramsM0_, paramsM1_, stream2);
          if (result != cudaSuccess) {printf("Error at %d: %s\n", __LINE__, cudaGetErrorString(result)); return Status::kErrorInternal;}

          const uint64_t ReadWriteM2 = SetReadWriteM(MmaStrassen::RWSharedAddEnd, MmaStrassen::GlobalPreSumLevel1_M0, false, DoGlobalSync) |
                                       SetReadWriteM(MmaStrassen::RWGlobalContigAndShared, MmaStrassen::GlobalPreSumLevel1_M2, false, DoGlobalSync);

          const uint64_t ReadWriteM3 = SetReadWriteM(MmaStrassen::RWAccumInShared, MmaStrassen::GlobalPreSumLevel1_M2, false) |
                                       SetReadWriteM(MmaStrassen::RWGlobalContig, MmaStrassen::GlobalPreSumLevel1_M3, false, DoGlobalSync);

          const uint64_t ReadWriteM6 = //SetReadWriteM(MmaStrassen::RWGlobalContig, MmaStrassen::GlobalPreSumLevel1_M0, false) |
                                      //SetReadWriteM(MmaStrassen::RWGlobalContig, MmaStrassen::GlobalPreSumLevel1_M2, false) |
                                      SetReadWriteM(MmaStrassen::RWSharedAddEnd, MmaStrassen::GlobalPreSumLevel1_M3, false, DoGlobalSync) |
                                      SetReadWriteM(MmaStrassen::RWUseSign, MmaStrassen::GlobalPreSumLevel1_M6, true)  |
                                      MmaStrassen::SetRWOutputC(2);

          const uint64_t ReadWriteM4 = //SetReadWriteM(MmaStrassen::RWGlobalContig, MmaStrassen::GlobalPreSumLevel1_M0, false) |
                                      //SetReadWriteM(MmaStrassen::RWGlobalContig, MmaStrassen::GlobalPreSumLevel1_M2, false) |
                                      SetReadWriteM(MmaStrassen::RWAccumInSharedInEpilogue, MmaStrassen::GlobalPreSumLevel1_M3, false) |
                                      SetReadWriteM(MmaStrassen::RWGlobalContig, MmaStrassen::GlobalPreSumLevel1_M4, false, DoGlobalSync) |
                                      MmaStrassen::SetRWOutputC(3);

          const uint64_t ReadWriteM5 = //SetReadWriteM(MmaStrassen::RWGlobalContig, MmaStrassen::GlobalPreSumLevel1_M0, false) |
                                      SetReadWriteM(MmaStrassen::RWSharedAddEnd, MmaStrassen::GlobalPreSumLevel1_M2, false, DoGlobalSync) |
                                      SetReadWriteM(MmaStrassen::RWGlobalContig, MmaStrassen::GlobalPreSumLevel1_M4, false, DoGlobalSync) |
                                      //SetReadWriteM(MmaStrassen::RWGlobalContig, MmaStrassen::GlobalPreSumLevel1_M5, false) |
                                      MmaStrassen::SetRWOutputC(1);
          using Op2 = FusedOperator2<GemmKernelM2, GemmKernelM3, schedule, 0, 0, ReadWriteM2, ReadWriteM3>;
          using Op3 = FusedOperator2<GemmKernelM6, GemmKernelM4, schedule, 0, 0, ReadWriteM6, ReadWriteM4>;
          using Op4 = FusedOperator1<GemmKernelM5,               schedule,    0, ReadWriteM5>;
          // result = run<Op2, Op3, Op4>(paramsM2_, paramsM6_, paramsM5_, stream3);
          if (result != cudaSuccess)
          {printf("Error at %d: %s\n", __LINE__, cudaGetErrorString(result)); return Status::kErrorInternal;}

          return Status::kSuccess;
        }
        const uint64_t ReadWriteM0 = SetReadWriteM(MmaStrassen::RWGlobalContig, MmaStrassen::GlobalPreSumLevel1_M0, false, DoGlobalSync) |
                                     SetReadWriteM(MmaStrassen::RWCombineMMALoop, MmaStrassen::GlobalPreSumLevel1_M1, false, false) | 
                                     MmaStrassen::SetRWOutputC(0);
        const uint64_t ReadWriteM1 = SetReadWriteM(MmaStrassen::RWAccumInShared, MmaStrassen::GlobalPreSumLevel1_M0, false, false) |
                                     MmaStrassen::SetRWOutputC(0);
        if (PresumM0 != 0) {
          if (true) {
            assert(paramsM0_.grid_tiled_shape.k() == 1);//Do not work with split k
            using Op1 = FusedOperator1<GemmKernelM0, schedule, PresumM0, ReadWriteM0>;
            result = run<Op1> (paramsM0_, stream1);
          }
        } else {
          const uint64_t ReadWriteM0 = SetReadWriteM(MmaStrassen::RWGlobalContigAndShared, MmaStrassen::GlobalPreSumLevel1_M0, false, DoGlobalSync);
          const uint64_t ReadWriteM1 = SetReadWriteM(MmaStrassen::RWAccumInShared, MmaStrassen::GlobalPreSumLevel1_M0, false, false) |
                                      MmaStrassen::SetRWOutputC(0);
          using Op1 = FusedOperator2<GemmKernelM0, GemmKernelM1, schedule, PresumM0, 0, ReadWriteM0, ReadWriteM1>;
          // result = run<Op1> (paramsM0_, stream2);
        }
        if (result != cudaSuccess)
        {printf("Error at %d: %s\n", __LINE__, cudaGetErrorString(result)); return Status::kErrorInternal;}

        if (false) {
          //5k x 5k x 5k
          //{M2,M3,M6,M4,M5}
          const uint64_t ReadWriteM2 = SetReadWriteM(MmaStrassen::RWSharedAddEnd, MmaStrassen::GlobalPreSumLevel1_M0, false, DoGlobalSync) |
                                      SetReadWriteM(MmaStrassen::RWGlobalContigAndShared, MmaStrassen::GlobalPreSumLevel1_M2, false);
                                      
          const uint64_t ReadWriteM3 = SetReadWriteM(MmaStrassen::RWAccumInShared, MmaStrassen::GlobalPreSumLevel1_M2, false) |
                                      SetReadWriteM(MmaStrassen::RWShared, MmaStrassen::GlobalPreSumLevel1_M3, false);

          const uint64_t ReadWriteM6 = //SetReadWriteM(MmaStrassen::RWGlobalContig, MmaStrassen::GlobalPreSumLevel1_M0, false) |
                                      //SetReadWriteM(MmaStrassen::RWGlobalContig, MmaStrassen::GlobalPreSumLevel1_M2, false) |
                                      SetReadWriteM(MmaStrassen::RWAccumInShared, MmaStrassen::GlobalPreSumLevel1_M3, false) |
                                      SetReadWriteM(MmaStrassen::RWUseSign, MmaStrassen::GlobalPreSumLevel1_M6, true)  |
                                      MmaStrassen::SetRWOutputC(2);

          const uint64_t ReadWriteM4 = //SetReadWriteM(MmaStrassen::RWGlobalContig, MmaStrassen::GlobalPreSumLevel1_M0, false) |
                                      //SetReadWriteM(MmaStrassen::RWGlobalContig, MmaStrassen::GlobalPreSumLevel1_M2, false) |
                                      SetReadWriteM(MmaStrassen::RWAccumInSharedInEpilogue, MmaStrassen::GlobalPreSumLevel1_M3, false) |
                                      SetReadWriteM(MmaStrassen::RWSharedInEpilogue, MmaStrassen::GlobalPreSumLevel1_M4, false) |
                                      MmaStrassen::SetRWOutputC(3);

          const uint64_t ReadWriteM5 = //SetReadWriteM(MmaStrassen::RWGlobalContig, MmaStrassen::GlobalPreSumLevel1_M0, false) |
                                      SetReadWriteM(MmaStrassen::RWGlobalContig, MmaStrassen::GlobalPreSumLevel1_M2, false) |
                                      SetReadWriteM(MmaStrassen::RWAccumInShared, MmaStrassen::GlobalPreSumLevel1_M4, false) |
                                      //SetReadWriteM(MmaStrassen::RWGlobalContig, MmaStrassen::GlobalPreSumLevel1_M5, false) |
                                      MmaStrassen::SetRWOutputC(1);
          using Op = FusedOperator5<GemmKernelM2, GemmKernelM3, GemmKernelM6, GemmKernelM4, GemmKernelM5,
                                    schedule,
                                    0, 0, 0, 0, 0,
                                    ReadWriteM2, ReadWriteM3, ReadWriteM6, ReadWriteM4, ReadWriteM5>;
          // result = run<Op>(paramsM2_, stream2);
          if (result != cudaSuccess)
          {printf("Error at %d: %s\n", __LINE__, cudaGetErrorString(result)); return Status::kErrorInternal;}
        }
        
        if (true) {
          //6k x 6k x 6k and more
          //{M2,M3,M6,M4}, {M5}
          const uint64_t ReadWriteM2 = SetReadWriteM(MmaStrassen::RWSharedAddEnd, MmaStrassen::GlobalPreSumLevel1_M0, false, DoGlobalSync) |
                                      SetReadWriteM(MmaStrassen::RWGlobalContigAndShared, MmaStrassen::GlobalPreSumLevel1_M2, false, DoGlobalSync); //TODO: Accum share through registers 
                                      
          const uint64_t ReadWriteM3 = SetReadWriteM(MmaStrassen::RWAccumInShared, MmaStrassen::GlobalPreSumLevel1_M2, false, false) |
                                      SetReadWriteM(MmaStrassen::RWShared, MmaStrassen::GlobalPreSumLevel1_M3, false, false);
                                      //TODO: Accum share through registers

          const uint64_t ReadWriteM6 = //SetReadWriteM(MmaStrassen::RWGlobalContig, MmaStrassen::GlobalPreSumLevel1_M0, false) |
                                      //SetReadWriteM(MmaStrassen::RWGlobalContig, MmaStrassen::GlobalPreSumLevel1_M2, false) |
                                      SetReadWriteM(MmaStrassen::RWAccumInShared, MmaStrassen::GlobalPreSumLevel1_M3, false, false) |
                                      SetReadWriteM(MmaStrassen::RWUseSign, MmaStrassen::GlobalPreSumLevel1_M6, true, false)  |
                                      MmaStrassen::SetRWOutputC(2);

          const uint64_t ReadWriteM4 = //SetReadWriteM(MmaStrassen::RWGlobalContig, MmaStrassen::GlobalPreSumLevel1_M0, false) |
                                      //SetReadWriteM(MmaStrassen::RWGlobalContig, MmaStrassen::GlobalPreSumLevel1_M2, false) |
                                      SetReadWriteM(MmaStrassen::RWAccumInSharedInEpilogue, MmaStrassen::GlobalPreSumLevel1_M3, false, false) |
                                      SetReadWriteM(MmaStrassen::RWGlobalContig, MmaStrassen::GlobalPreSumLevel1_M4, false, DoGlobalSync) |
                                      MmaStrassen::SetRWOutputC(3);

          const uint64_t ReadWriteM5 = //SetReadWriteM(MmaStrassen::RWGlobalContig, MmaStrassen::GlobalPreSumLevel1_M0, false) |
                                      SetReadWriteM(MmaStrassen::RWSharedAddEnd, MmaStrassen::GlobalPreSumLevel1_M2, false, DoGlobalSync) |
                                      SetReadWriteM(MmaStrassen::RWGlobalContig, MmaStrassen::GlobalPreSumLevel1_M4, false, DoGlobalSync) |
                                      //SetReadWriteM(MmaStrassen::RWGlobalContig, MmaStrassen::GlobalPreSumLevel1_M5, false) |
                                      MmaStrassen::SetRWOutputC(1);

          using Op1 = FusedOperator4<GemmKernelM2, GemmKernelM3, GemmKernelM6, GemmKernelM4, schedule,
                                     0, 0, 0, 0,
                                     ReadWriteM2, ReadWriteM3, ReadWriteM6, ReadWriteM4>;
          using Op2 = FusedOperator1<GemmKernelM5, schedule, 0, ReadWriteM5>;
          result = run<Op1, Op2>(paramsM2_, paramsM5_, stream3);
          if (result != cudaSuccess)
          {printf("Error at %d: %s\n", __LINE__, cudaGetErrorString(result)); return Status::kErrorInternal;}
        }
        // result = run5x2<GemmKernelM2, GemmKernelM3, GemmKernelM0,  GemmKernelM4, GemmKernelM6,
        //                 schedule,
        //                 PresumM2, PresumM3, ReadWriteM2, ReadWriteM3, ReadWriteM0, ReadWriteM4, ReadWriteM6>
        //   (paramsM2_, paramsM3_, paramsM0_, paramsM4_, paramsM6_, stream1);
        // if (result != cudaSuccess)
        // {printf("Error at %d: %s\n", __LINE__, cudaGetErrorString(result)); return Status::kErrorInternal;}

        if (false) {
          //tried with ReadWrite = RWOutputC(0)
          //tried but slow {M2,M3}, {M6,M4,M5}
          //tried but slow {M2,M3,M4}, {M5,M6}
          //tried but ?? {M2,M3},{M4,M5},{M6} 8k->3.5, 13k->14.65,  
          //tried but ?? {M2,M3},{M4},{M5},{M6} 8k-> 3.495, 13k->14.65
          //Looks like last grid should only have 1 sub-gemmm
          //For 8k x 8k x 8k, second kernel runs 1% slower when PresumM0M1 kernel is enabled than without presum.
          // const uint64_t ReadWriteM2 = SetReadWriteM(MmaStrassen::RWSharedAddEnd, MmaStrassen::GlobalPreSumLevel1_M0, false) |
          //                             SetReadWriteM(MmaStrassen::RWGlobalContigAndShared, MmaStrassen::GlobalPreSumLevel1_M2, false);
                                      
          // const uint64_t ReadWriteM3 = SetReadWriteM(MmaStrassen::RWAccumInShared, MmaStrassen::GlobalPreSumLevel1_M2, false) |
          //                             SetReadWriteM(MmaStrassen::RWGlobalContigAndShared, MmaStrassen::GlobalPreSumLevel1_M3, false);

          // const uint64_t ReadWriteM6 = //SetReadWriteM(MmaStrassen::RWGlobalContig, MmaStrassen::GlobalPreSumLevel1_M0, false) |
          //                             //SetReadWriteM(MmaStrassen::RWGlobalContig, MmaStrassen::GlobalPreSumLevel1_M2, false) |
          //                             // SetReadWriteM(MmaStrassen::RWAccumInShared, MmaStrassen::GlobalPreSumLevel1_M3, false) |
          //                             // SetReadWriteM(MmaStrassen::RWUseSign, MmaStrassen::GlobalPreSumLevel1_M6, true)  |
          //                             MmaStrassen::SetRWOutputC(2);

          // const uint64_t ReadWriteM4 = //SetReadWriteM(MmaStrassen::RWGlobalContig, MmaStrassen::GlobalPreSumLevel1_M0, false) |
          //                             //SetReadWriteM(MmaStrassen::RWGlobalContig, MmaStrassen::GlobalPreSumLevel1_M2, false) |
          //                             // SetReadWriteM(MmaStrassen::RWSharedAddEnd, MmaStrassen::GlobalPreSumLevel1_M3, false) |
          //                             // SetReadWriteM(MmaStrassen::RWSharedInEpilogue, MmaStrassen::GlobalPreSumLevel1_M4, false) |
          //                             MmaStrassen::SetRWOutputC(3);

          // const uint64_t ReadWriteM5 = //SetReadWriteM(MmaStrassen::RWGlobalContig, MmaStrassen::GlobalPreSumLevel1_M0, false) |
          //                             // SetReadWriteM(MmaStrassen::RWAccumInShared, MmaStrassen::GlobalPreSumLevel1_M2, false) |
          //                             // SetReadWriteM(MmaStrassen::RWAccumInShared, MmaStrassen::GlobalPreSumLevel1_M4, false) |
          //                             //SetReadWriteM(MmaStrassen::RWGlobalContig, MmaStrassen::GlobalPreSumLevel1_M5, false) |
          //                             MmaStrassen::SetRWOutputC(1);

          // using Op2 = FusedOperator2<GemmKernelM2, GemmKernelM3, schedule,
          //                            0, 0,
          //                            ReadWriteM2, ReadWriteM3>;
          // using Op3 = FusedOperator2<GemmKernelM4, GemmKernelM5, schedule,
          //                            0, 0,
          //                            ReadWriteM4, ReadWriteM5>;
          // using Op4 = FusedOperator1<GemmKernelM6, schedule,
          //                            0,
          //                            ReadWriteM6>;
          // // using Op4 = FusedOperator1<GemmKernelM6, schedule, 0, ReadWriteM6>;

          // // result = run<Op2, Op3, Op4>(paramsM2_, paramsM4_, paramsM6_, stream2);
          // if (result != cudaSuccess)
          // {printf("Error at %d: %s\n", __LINE__, cudaGetErrorString(result)); return Status::kErrorInternal;}
        }
      }

      // result = run2x1<GemmKernelM1, GemmKernelM5, schedule, 0, 0, ReadWriteM1, ReadWriteM5>(paramsM1_, paramsM5_, stream2);
      // if (result != cudaSuccess)
      // {printf("Error at %d: %s\n", __LINE__, cudaGetErrorString(result)); return Status::kErrorInternal;}

      // result = run5x2<GemmKernelM2, GemmKernelM3, GemmKernelM0,  GemmKernelM4, GemmKernelM6,
      //                 schedule,
      //                 PresumM2, PresumM3, ReadWriteM2, ReadWriteM3, ReadWriteM0, ReadWriteM4, ReadWriteM6>
      //   (paramsM2_, paramsM3_, paramsM0_, paramsM4_, paramsM6_, stream1);
      // if (result != cudaSuccess)
      // {printf("Error at %d: %s\n", __LINE__, cudaGetErrorString(result)); return Status::kErrorInternal;}
      
      
      // result = run2x2<10*1024, GemmKernelM1, GemmKernelM5, GemmKernelM4, GemmKernelM6>
      //          (paramsM1_, paramsM5_, paramsM4_, paramsM6_, stream2);
      // if (result != cudaSuccess)
      // {printf("Error at %d: %s\n", __LINE__, cudaGetErrorString(result)); return Status::kErrorInternal;}
     }
    }

    return Status::kSuccess;
  }

  /// Runs the kernel using initialized state.
  // Status operator()(cudaStream_t stream = nullptr) {
  //   return run(stream);
  // }

  /// Runs the kernel using initialized state.
  Status operator()(
    Arguments const &args,
    void *workspace = nullptr,
    cudaStream_t* streams = nullptr,
    int num_streams = 0) {
    cudaStream_t init_stream;

    if (streams == nullptr || num_streams == 0) {
      init_stream = nullptr;
    } else {
      init_stream = streams[0];
    }

    Status status = initialize(args, workspace, init_stream);

    if (status == Status::kSuccess) {
      status = run(streams, num_streams);
    }

    return status;
  }
};


////////////////////////////////////////////////////////////////////////////////

template <
    typename StrassenMiGroup,
    typename ScheduleStrassenGroups,
    /// Element type for A matrix operand
    typename ElementA_,
    /// Layout type for A matrix operand
    typename LayoutA_,
    /// Element type for B matrix operand
    typename ElementB_,
    /// Layout type for B matrix operand
    typename LayoutB_,
    /// Element type for C and D matrix operands
    typename ElementC_,
    /// Layout type for C and D matrix operands
    typename LayoutC_,
    /// Element type for internal accumulation
    typename ElementAccumulator_,
    /// Operator class tag
    typename OperatorClass_,
    /// Tag indicating architecture to tune for
    typename ArchTag_,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape_,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape_,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape_,
    /// Epilogue output operator
    typename EpilogueOutputOp_,
    typename InterimEpilogueOp_,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle_,
    /// Number of stages used in the pipelined mainloop
    int Stages,
    /// Access granularity of A matrix in units of elements
    int AlignmentA,
    /// Access granularity of B matrix in units of elements
    int AlignmentB,
    /// If true, kernel supports split-K with serial reduction
    bool SplitKSerial,
    bool SubGemmParallel,
    /// Operation performed by GEMM
    typename Operator_,
    /// Gather operand A by using an index array
    bool GatherA,
    /// Gather operand B by using an index array
    bool GatherB,
    /// Scatter result D by using an index array
    bool ScatterD,
    /// Permute result D
    typename PermuteDLayout>
class StrassenGemm<StrassenType::MatrixGlobalLevel1, StrassenMiGroup, ScheduleStrassenGroups, ElementA_, LayoutA_, ElementB_, LayoutB_, ElementC_, LayoutC_, ElementAccumulator_, OperatorClass_, ArchTag_, ThreadblockShape_, WarpShape_, InstructionShape_, EpilogueOutputOp_, InterimEpilogueOp_, ThreadblockSwizzle_, Stages, AlignmentA, AlignmentB, SplitKSerial, SubGemmParallel, Operator_, GatherA, GatherB, ScatterD, PermuteDLayout> {
 public:
  using ElementA = ElementA_;
  using LayoutA = LayoutA_;
  using TensorRefA = TensorRef<ElementA const, LayoutA>;
  using ElementB = ElementB_;
  using LayoutB = LayoutB_;
  using TensorRefB = TensorRef<ElementB const, LayoutB>;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  using TensorRefC = TensorRef<ElementC const, LayoutC>;
  using TensorRefD = TensorRef<ElementC, LayoutC>;
  using ElementAccumulator = ElementAccumulator_;
  using OperatorClass = OperatorClass_;
  using ArchTag = ArchTag_;
  using ThreadblockShape = ThreadblockShape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  using EpilogueOutputOp = EpilogueOutputOp_;
  using InterimEpilogueOp = InterimEpilogueOp_;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  using Operator = Operator_;
  static int const kStages = Stages;
  static int const kAlignmentA = AlignmentA;
  static int const kAlignmentB = AlignmentB;
  static int const kAlignmentC = EpilogueOutputOp::kCount;
  static bool const kSplitKSerial = SplitKSerial;
  static ComplexTransform const kTransformA = ComplexTransform::kNone;
  static ComplexTransform const kTransformB = ComplexTransform::kNone;

  /// Define the kernel
  template<MmaStrassen::Type MmaKind, typename MmaStrassenConsts>
  using GemmKernel_ = typename kernel::DefaultStrassenGemm<
    ElementA,
    LayoutA,
    kAlignmentA,
    ElementB,
    LayoutB,
    kAlignmentB,
    ElementC,
    LayoutC,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    MmaKind,
    typename StrassenMiGroup::Group0,
    MmaStrassen::Consts<1,1,1,1>,
    EpilogueOutputOp,
    InterimEpilogueOp,
    ThreadblockSwizzle,
    kStages,
    kSplitKSerial,SubGemmParallel,
    Operator,
    SharedMemoryClearOption::kNone,
    GatherA,
    GatherB,
    ScatterD,
    PermuteDLayout
  >;
  using GemmKernelM0 = typename GemmKernel_<MmaStrassen::Type::MatrixGlobalLevel1_M0, MmaStrassen::Consts<1,1,1,1>>::GemmKernel;
  using GemmKernelM1 = typename GemmKernel_<MmaStrassen::Type::MatrixGlobalLevel1_M1, MmaStrassen::Consts<1,1,1,1>>::GemmKernel;
  using GemmKernelM2 = typename GemmKernel_<MmaStrassen::Type::MatrixGlobalLevel1_M2, MmaStrassen::Consts<1,1,1,1>>::GemmKernel;
  using GemmKernelM3 = typename GemmKernel_<MmaStrassen::Type::MatrixGlobalLevel1_M3, MmaStrassen::Consts<1,1,1,1>>::GemmKernel;
  using GemmKernelM4 = typename GemmKernel_<MmaStrassen::Type::MatrixGlobalLevel1_M4, MmaStrassen::Consts<1,1,1,1>>::GemmKernel;
  using GemmKernelM5 = typename GemmKernel_<MmaStrassen::Type::MatrixGlobalLevel1_M5, MmaStrassen::Consts<1,1,1,1>>::GemmKernel;
  using GemmKernelM6 = typename GemmKernel_<MmaStrassen::Type::MatrixGlobalLevel1_M6, MmaStrassen::Consts<1,1,1,1>>::GemmKernel;

  /// Argument structure
  struct Arguments {

    //
    // Data members
    //

    GemmCoord problem_size;
    TensorRef<ElementA const, LayoutA> ref_A;
    TensorRef<ElementB const, LayoutB> ref_B;
    TensorRef<ElementC const, LayoutC> ref_C;
    TensorRef<ElementC, LayoutC> ref_D;
    typename EpilogueOutputOp::Params epilogue;
    int split_k_slices;
    // For gather+scatter operations
    int const *gather_A_indices;
    int const *gather_B_indices;
    int const *scatter_D_indices;

    //
    // Methods
    //

    /// Default ctor
    CUTLASS_HOST_DEVICE
    Arguments(): problem_size(0, 0, 0), split_k_slices(1) {

    }

    /// Constructs an Arguments structure
    CUTLASS_HOST_DEVICE
    Arguments(
      GemmCoord problem_size_,
      TensorRef<ElementA const, LayoutA> ref_A_,
      TensorRef<ElementB const, LayoutB> ref_B_,
      TensorRef<ElementC const, LayoutC> ref_C_,
      TensorRef<ElementC, LayoutC> ref_D_,
      typename EpilogueOutputOp::Params epilogue_ =
        typename EpilogueOutputOp::Params(),
      int split_k_slices = 1,
      int const *gather_A_indices_ = nullptr,
      int const *gather_B_indices_ = nullptr,
      int const *scatter_D_indices_ = nullptr
    ):
      problem_size(problem_size_),
      ref_A(ref_A_),
      ref_B(ref_B_),
      ref_C(ref_C_),
      ref_D(ref_D_),
      epilogue(epilogue_),
      split_k_slices(split_k_slices),
      gather_A_indices(gather_A_indices_),
      gather_B_indices(gather_B_indices_),
      scatter_D_indices(scatter_D_indices_) {

    }
  };

private:

  /// Kernel parameters object
  typename GemmKernelM0::Params paramsM0_;
  typename GemmKernelM1::Params paramsM1_;
  typename GemmKernelM2::Params paramsM2_;
  typename GemmKernelM3::Params paramsM3_;
  typename GemmKernelM4::Params paramsM4_;
  typename GemmKernelM5::Params paramsM5_;
  typename GemmKernelM6::Params paramsM6_;

public:

  /// Constructs the GEMM.
  StrassenGemm() {
    /*int kM = GemmKernel_::Mma_::MmaCore::LaneMmaShape::kM;
    int kN = GemmKernel_::Mma_::MmaCore::LaneMmaShape::kN;
    printf("361: LaneMmaShape %d %d \n", kM, kN);
    using WarpCount = typename GemmKernel_::Mma_::MmaCore::WarpCount;
    printf("363: WarpCount %d %d\n", WarpCount::kM, WarpCount::kN);

    int WarpNumThreadsM = GemmKernel_::Mma_::MmaCore::WarpNumThreadsM;
    int WarpNumThreadsN = GemmKernel_::Mma_::MmaCore::WarpNumThreadsN;
    printf("367: %d %d\n", WarpNumThreadsM, WarpNumThreadsN);

    using WarpShape = typename GemmKernel_::Mma_::MmaCore::MmaWarpSimt::Shape;
    using PolicyWarpShape = typename GemmKernel_::Mma_::MmaCore::MmaWarpSimt::Policy::WarpShape;
    printf("368: %d %d; %d %d\n", WarpShape::kM, WarpShape::kN, PolicyWarpShape::kRow, PolicyWarpShape::kColumn);

    int MmaCorekThreads = GemmKernel_::Mma_::MmaCore::kThreads;
    printf("MmaCore::kThreads %d\n", MmaCorekThreads);

    using ThreadShape = typename GemmKernel_::Mma_::MmaCore::MmaWarpSimt::ThreadMma::Shape;
    printf("374: %d %d\n", ThreadShape::kM, ThreadShape::kN);

    using MmaCoreShape = typename GemmKernel_::Mma_::MmaCore::Shape;
    printf("377: %d %d %d\n", MmaCoreShape::kM, MmaCoreShape::kN, MmaCoreShape::kK);

    using ThreadMapA = typename GemmKernel_::Mma_::MmaCore::IteratorThreadMapA;
    printf("380: %d %d\n", ThreadMapA::Iterations::kCount, ThreadMapA::kElementsPerAccess);
    printf("381: %d %d %d\n", ThreadMapA::kThreads, ThreadMapA::Detail::ShapeVec::kContiguous, ThreadMapA::Detail::ShapeVec::kStrided);

    uint WarpCountkCount = GemmKernel::WarpCount::kCount;
    printf("384: WarpCountkCount %d\n", WarpCountkCount);*/
  }

  /// Determines whether the GEMM can execute the given problem.
  static Status can_implement(Arguments const &args) {

    if (!kSplitKSerial && args.split_k_slices > 1) {
      return Status::kErrorInvalidProblem;
    }

    Status status = GemmKernelM0::can_implement(
      args.problem_size,
      args.ref_A.non_const_ref(),
      args.ref_B.non_const_ref(),
      args.ref_C.non_const_ref(),
      args.ref_D
    );

    if (status != Status::kSuccess) {
      return status;
    }

    return Status::kSuccess;
  }

  /// Gets the workspace size
  static size_t get_workspace_size(Arguments const &args) {

    size_t bytes = 0;

    // Determine grid shape
    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord tiled_shape = threadblock_swizzle.get_tiled_shape(
      args.problem_size,
      {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
      args.split_k_slices);

    if (SubGemmParallel || (kSplitKSerial && args.split_k_slices > 1)) {
      bytes += sizeof(int) * size_t(tiled_shape.m()) * size_t(tiled_shape.n()) * (SubGemmParallel ? 1 : 7);
    }

    return bytes;
  }

  /// Initializes GEMM state from arguments.
  Status initialize(Arguments const &args, void *workspace = nullptr, cudaStream_t stream = nullptr) {

    // Determine grid shape
    ThreadblockSwizzle threadblock_swizzle;
    auto new_problem_size = GemmCoord{args.problem_size.m()/2, args.problem_size.n()/2, args.problem_size.k()/2};
    cutlass::gemm::GemmCoord grid_shape = threadblock_swizzle.get_tiled_shape(
      new_problem_size,
      {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
      args.split_k_slices);

    cutlass::gemm::GemmCoord orig_grid_shape = threadblock_swizzle.get_tiled_shape(
      args.problem_size,
      {ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK},
      args.split_k_slices);

    if (SubGemmParallel || kSplitKSerial) {
      if (SubGemmParallel || args.split_k_slices > 1) {
        if (!workspace) {
          return Status::kErrorWorkspaceNull;
        }

        size_t bytes = get_workspace_size(args);

        cudaError_t result = cudaMemsetAsync(workspace, 0, bytes, stream);

        if (result != cudaSuccess) {
          return Status::kErrorInternal;
        }
      }
    }
    else {

      if (args.split_k_slices > 1) {
        return Status::kErrorInvalidProblem;
      }
    }

    uint new_problem_elems = new_problem_size.m() * new_problem_size.n();

    // Initialize the Params structure
    paramsM0_ = typename GemmKernelM0::Params{
      args.problem_size,
      grid_shape,
      args.ref_A.non_const_ref(),
      args.ref_B.non_const_ref(),
      args.ref_C.non_const_ref(),
      args.ref_D,
      args.epilogue,
      static_cast<int *>(workspace) + (SubGemmParallel? 0 : 0)*(orig_grid_shape.m() * orig_grid_shape.n()),
      args.gather_A_indices,
      args.gather_B_indices,
      args.scatter_D_indices
    };

    paramsM1_ = typename GemmKernelM1::Params{
      args.problem_size,
      grid_shape,
      args.ref_A.non_const_ref().add_coord_offset({1*new_problem_size.m(), 0}),
      args.ref_B.non_const_ref().add_coord_offset({1*new_problem_size.k(), 0}),
      args.ref_C.non_const_ref(),
      args.ref_D,
      args.epilogue,
      static_cast<int *>(workspace) + (SubGemmParallel? 0 : 1) * (orig_grid_shape.m() * orig_grid_shape.n()),
      args.gather_A_indices,
      args.gather_B_indices,
      args.scatter_D_indices
    };

    paramsM2_ = typename GemmKernelM2::Params{
      args.problem_size,
      grid_shape,
      args.ref_A.non_const_ref().add_coord_offset({2*new_problem_size.m(), 0}),
      args.ref_B.non_const_ref().add_coord_offset({2*new_problem_size.k(), 0}),
      args.ref_C.non_const_ref(),
      args.ref_D,
      args.epilogue,
      static_cast<int *>(workspace) + (SubGemmParallel? 0 : 2)*(orig_grid_shape.m() * orig_grid_shape.n()),
      args.gather_A_indices,
      args.gather_B_indices,
      args.scatter_D_indices
    };

    paramsM3_ = typename GemmKernelM3::Params{
      args.problem_size,
      grid_shape,
      args.ref_A.non_const_ref().add_coord_offset({3*new_problem_size.m(), 0}),
      args.ref_B.non_const_ref().add_coord_offset({3*new_problem_size.k(), 0}),
      args.ref_C.non_const_ref(),
      args.ref_D,
      args.epilogue,
      static_cast<int *>(workspace) + (SubGemmParallel? 0 : 3)*(orig_grid_shape.m() * orig_grid_shape.n()),
      args.gather_A_indices,
      args.gather_B_indices,
      args.scatter_D_indices
    };

    paramsM4_ = typename GemmKernelM4::Params{
      args.problem_size,
      grid_shape,
      args.ref_A.non_const_ref().add_coord_offset({4*new_problem_size.m(), 0}),
      args.ref_B.non_const_ref().add_coord_offset({4*new_problem_size.k(), 0}),
      args.ref_C.non_const_ref(),
      args.ref_D,
      args.epilogue,
      static_cast<int *>(workspace) + (SubGemmParallel? 0: 4)*(orig_grid_shape.m() * orig_grid_shape.n()),
      args.gather_A_indices,
      args.gather_B_indices,
      args.scatter_D_indices
    };

    paramsM5_ = typename GemmKernelM5::Params{
      args.problem_size,
      grid_shape,
      args.ref_A.non_const_ref().add_coord_offset({5*new_problem_size.m(), 0}),
      args.ref_B.non_const_ref().add_coord_offset({5*new_problem_size.k(), 0}),
      args.ref_C.non_const_ref(),
      args.ref_D,
      args.epilogue,
      static_cast<int *>(workspace) + (SubGemmParallel? 0 : 5) *(orig_grid_shape.m() * orig_grid_shape.n()),
      args.gather_A_indices,
      args.gather_B_indices,
      args.scatter_D_indices
    };

    paramsM6_ = typename GemmKernelM6::Params{
      args.problem_size,
      grid_shape,
      args.ref_A.non_const_ref().add_coord_offset({6*new_problem_size.m(), 0}),
      args.ref_B.non_const_ref().add_coord_offset({6*new_problem_size.k(), 0}),
      args.ref_C.non_const_ref(),
      args.ref_D,
      args.epilogue,
      static_cast<int *>(workspace) + (SubGemmParallel? 0 : 6) *(orig_grid_shape.m() * orig_grid_shape.n()),
      args.gather_A_indices,
      args.gather_B_indices,
      args.scatter_D_indices
    };

    return Status::kSuccess;
  }

  /// Lightweight update given a subset of arguments
  Status update(Arguments const &args, void *workspace = nullptr) {
    assert(false);
    if (kSplitKSerial && args.split_k_slices > 1) {
      if (!workspace) {
        return Status::kErrorWorkspaceNull;
      }
    }

    paramsM0_.ref_A.reset(args.ref_A.non_const_ref().data());
    paramsM0_.ref_B.reset(args.ref_B.non_const_ref().data());
    paramsM0_.ref_C.reset(args.ref_C.non_const_ref().data());
    paramsM0_.ref_D.reset(args.ref_D.data());
    paramsM0_.output_op = args.epilogue;
    paramsM0_.semaphore = static_cast<int *>(workspace);


    paramsM1_.ref_A.reset(args.ref_A.non_const_ref().data());
    paramsM1_.ref_B.reset(args.ref_B.non_const_ref().data());
    paramsM1_.ref_C.reset(args.ref_C.non_const_ref().data());
    paramsM1_.ref_D.reset(args.ref_D.data());
    paramsM1_.output_op = args.epilogue;
    paramsM1_.semaphore = static_cast<int *>(workspace);


    paramsM2_.ref_A.reset(args.ref_A.non_const_ref().data());
    paramsM2_.ref_B.reset(args.ref_B.non_const_ref().data());
    paramsM2_.ref_C.reset(args.ref_C.non_const_ref().data());
    paramsM2_.ref_D.reset(args.ref_D.data());
    paramsM2_.output_op = args.epilogue;
    paramsM2_.semaphore = static_cast<int *>(workspace);


    paramsM3_.ref_A.reset(args.ref_A.non_const_ref().data());
    paramsM3_.ref_B.reset(args.ref_B.non_const_ref().data());
    paramsM3_.ref_C.reset(args.ref_C.non_const_ref().data());
    paramsM3_.ref_D.reset(args.ref_D.data());
    paramsM3_.output_op = args.epilogue;
    paramsM3_.semaphore = static_cast<int *>(workspace);

    paramsM4_.ref_A.reset(args.ref_A.non_const_ref().data());
    paramsM4_.ref_B.reset(args.ref_B.non_const_ref().data());
    paramsM4_.ref_C.reset(args.ref_C.non_const_ref().data());
    paramsM4_.ref_D.reset(args.ref_D.data());
    paramsM4_.output_op = args.epilogue;
    paramsM4_.semaphore = static_cast<int *>(workspace);

    paramsM5_.ref_A.reset(args.ref_A.non_const_ref().data());
    paramsM5_.ref_B.reset(args.ref_B.non_const_ref().data());
    paramsM5_.ref_C.reset(args.ref_C.non_const_ref().data());
    paramsM5_.ref_D.reset(args.ref_D.data());
    paramsM5_.output_op = args.epilogue;
    paramsM5_.semaphore = static_cast<int *>(workspace);

    paramsM6_.ref_A.reset(args.ref_A.non_const_ref().data());
    paramsM6_.ref_B.reset(args.ref_B.non_const_ref().data());
    paramsM6_.ref_C.reset(args.ref_C.non_const_ref().data());
    paramsM6_.ref_D.reset(args.ref_D.data());
    paramsM6_.output_op = args.epilogue;
    paramsM6_.semaphore = static_cast<int *>(workspace);

    return Status::kSuccess;
  }

  template<typename GemmKernel>
  cudaError_t run(typename GemmKernel::Params& params, cudaStream_t stream = nullptr) {
    ThreadblockSwizzle threadblock_swizzle;

    dim3 grid = threadblock_swizzle.get_grid_shape(params.grid_tiled_shape);

    dim3 block(GemmKernel::kThreadCount, 1, 1);

    cudaError_t result;
    int smem_size = int(sizeof(typename GemmKernel::SharedStorage));
    // std::cout << "504 " << smem_size << " " << GemmKernel::kThreadCount << " " << grid.x << " " << grid.y << " " << grid.z << std::endl;
    if (smem_size >= (48 << 10)) {
      result = cudaFuncSetAttribute(Kernel<GemmKernel>,
                                    cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    smem_size);

      if (result != cudaSuccess) {
        return result;
      }
    }

    cutlass::Kernel<GemmKernel><<<grid, block, smem_size, stream>>>(params);
    result = cudaGetLastError();
    result = cudaStreamSynchronize(stream);
    return result;
  }

  /// Runs the kernel using initialized state.
  Status run(cudaStream_t* streams = nullptr, int num_streams = 0) {
    cudaError_t result;

    if (SubGemmParallel) {
      cudaStream_t defaultStreams[7] = {0};
      if (streams == nullptr || num_streams == 0) {
        streams = defaultStreams;
      } else {
        assert (num_streams >= 7);
      }
      // result = run<GemmKernelM0>(paramsM0_, streams[0]);
      // if (result != cudaSuccess)
      // {printf("Error at %d: %s\n", __LINE__, cudaGetErrorString(result)); return Status::kErrorInternal;}

      // result = run<GemmKernelM1>(paramsM1_, streams[1]);
      // if (result != cudaSuccess)
      // {printf("Error at %d: %s\n", __LINE__, cudaGetErrorString(result)); return Status::kErrorInternal;}

      // result = run<GemmKernelM2>(paramsM2_, streams[2]);
      // if (result != cudaSuccess)
      // {printf("Error at %d: %s\n", __LINE__, cudaGetErrorString(result)); return Status::kErrorInternal;}

      // result = run<GemmKernelM3>(paramsM3_, streams[3]);
      // if (result != cudaSuccess)
      // {printf("Error at %d: %s\n", __LINE__, cudaGetErrorString(result)); return Status::kErrorInternal;}

      // result = run<GemmKernelM4>(paramsM4_, streams[4]);
      // if (result != cudaSuccess)
      // {printf("Error at %d: %s\n", __LINE__, cudaGetErrorString(result)); return Status::kErrorInternal;}

      // result = run<GemmKernelM5>(paramsM5_, streams[5]);
      // if (result != cudaSuccess)
      // {printf("Error at %d: %s\n", __LINE__, cudaGetErrorString(result)); return Status::kErrorInternal;}

      // result = run<GemmKernelM6>(paramsM6_, streams[6]);
      // if (result != cudaSuccess)
      // {printf("Error at %d: %s\n", __LINE__, cudaGetErrorString(result)); return Status::kErrorInternal;}
    } else {
      cudaStream_t stream1, stream2;
      if (streams == nullptr || num_streams == 0) {
        stream1 = 0;
        stream2 = 0;
      } else {
        stream1 = streams[0];
        stream2 = streams[1];
      }
      result = run<GemmKernelM1>(paramsM1_, stream1);
      if (result != cudaSuccess)
      {printf("Error at %d: %s\n", __LINE__, cudaGetErrorString(result)); return Status::kErrorInternal;}

      result = run<GemmKernelM4>(paramsM4_, stream2);
      if (result != cudaSuccess)
      {printf("Error at %d: %s\n", __LINE__, cudaGetErrorString(result)); return Status::kErrorInternal;}

      result = run<GemmKernelM5>(paramsM5_, stream1);
      if (result != cudaSuccess)
      {printf("Error at %d: %s\n", __LINE__, cudaGetErrorString(result)); return Status::kErrorInternal;}

      result = run<GemmKernelM6>(paramsM6_, stream2);
      if (result != cudaSuccess)
      {printf("Error at %d: %s\n", __LINE__, cudaGetErrorString(result)); return Status::kErrorInternal;}

      // if (stream2 != stream1) {
      //   result = cudaStreamSynchronize(stream1);
      //   if (result != cudaSuccess)
      //   {printf("Error at %d: %s\n", __LINE__, cudaGetErrorString(result)); return Status::kErrorInternal;}
      //   result = cudaStreamSynchronize(stream2);
      //   if (result != cudaSuccess)
      //   {printf("Error at %d: %s\n", __LINE__, cudaGetErrorString(result)); return Status::kErrorInternal;}
      // }

      result = run<GemmKernelM2>(paramsM2_, stream1);
      if (result != cudaSuccess)
      {printf("Error at %d: %s\n", __LINE__, cudaGetErrorString(result)); return Status::kErrorInternal;}

      result = run<GemmKernelM3>(paramsM3_, stream2);
      if (result != cudaSuccess)
      {printf("Error at %d: %s\n", __LINE__, cudaGetErrorString(result)); return Status::kErrorInternal;}

      // if (stream2 != stream1) {
      //   result = cudaStreamSynchronize(stream1);
      //   if (result != cudaSuccess)
      //   {printf("Error at %d: %s\n", __LINE__, cudaGetErrorString(result)); return Status::kErrorInternal;}
      //   result = cudaStreamSynchronize(stream2);
      //   if (result != cudaSuccess)
      //   {printf("Error at %d: %s\n", __LINE__, cudaGetErrorString(result)); return Status::kErrorInternal;}
      // }

      result = run<GemmKernelM0>(paramsM0_, stream1);
      if (result != cudaSuccess)
      {printf("Error at %d: %s\n", __LINE__, cudaGetErrorString(result)); return Status::kErrorInternal;}
    }

    return Status::kSuccess;
  }

  /// Runs the kernel using initialized state.
  // Status operator()(cudaStream_t stream = nullptr) {
  //   return run(stream);
  // }

  /// Runs the kernel using initialized state.
  Status operator()(
    Arguments const &args,
    void *workspace = nullptr,
    cudaStream_t* streams = nullptr,
    int num_streams = 0) {
    cudaStream_t init_stream;

    if (streams == nullptr || num_streams == 0) {
      init_stream = nullptr;
    } else {
      init_stream = streams[0];
    }

    Status status = initialize(args, workspace, init_stream);

    if (status == Status::kSuccess) {
      status = run(streams, num_streams);
    }

    return status;
  }
};

////////////////////////////////////////////////////////////////////////////////
/// Partial specialization for column-major output exchanges problem size and operand.
/*
template <
    StrassenType::ENUM StrassenKind,
    /// Element type for A matrix operand
    typename ElementA_,
    /// Layout type for A matrix operand
    typename LayoutA_,
    /// Element type for B matrix operand
    typename ElementB_,
    /// Layout type for B matrix operand
    typename LayoutB_,
    /// Element type for C and D matrix operands
    typename ElementC_,
    /// Element type for internal accumulation
    typename ElementAccumulator_,
    /// Operator class tag
    typename OperatorClass_,
    /// Tag indicating architecture to tune for
    typename ArchTag_,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape_,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape_,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape_,
    /// Epilogue output operator
    typename EpilogueOutputOp_,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle_,
    /// Number of stages used in the pipelined mainloop
    int Stages,
    /// Access granularity of A matrix in units of elements
    int AlignmentA,
    /// Access granularity of B matrix in units of elements
    int AlignmentB,
    /// If true, kernel supports split-K as a serial reduction
    bool SplitKSerial,
    /// Operation performed by GEMM
    typename Operator_,
    /// Gather operand A by using an index array
    bool GatherA,
    /// Gather operand B by using an index array
    bool GatherB,
    /// Scatter result D by using an index array
    bool ScatterD,
    /// Permute result D
    typename PermuteDLayout
>
class StrassenGemm<StrassenKind, ElementA_, LayoutA_, ElementB_, LayoutB_, ElementC_,
           layout::ColumnMajor,  // partially specialized on LayoutC
           ElementAccumulator_, OperatorClass_, ArchTag_, ThreadblockShape_,
           WarpShape_, InstructionShape_, EpilogueOutputOp_,
           ThreadblockSwizzle_, Stages, AlignmentA, AlignmentB, SplitKSerial,
           Operator_, GatherA, GatherB, ScatterD, PermuteDLayout> {
 public:

  using ElementA = ElementA_;
  using LayoutA = LayoutA_;
  using TensorRefA = TensorRef<ElementA const, LayoutA>;
  using ElementB = ElementB_;
  using LayoutB = LayoutB_;
  using TensorRefB = TensorRef<ElementB const, LayoutB>;
  using ElementC = ElementC_;
  using LayoutC = layout::ColumnMajor;
  using TensorRefC = TensorRef<ElementC const, LayoutC>;
  using TensorRefD = TensorRef<ElementC, LayoutC>;
  using ElementAccumulator = ElementAccumulator_;
  using OperatorClass = OperatorClass_;
  using ArchTag = ArchTag_;
  using ThreadblockShape = ThreadblockShape_;
  using WarpShape = WarpShape_;
  using InstructionShape = InstructionShape_;
  using EpilogueOutputOp = EpilogueOutputOp_;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  using Operator = Operator_;
  static int const kStages = Stages;
  static int const kAlignmentA = AlignmentA;
  static int const kAlignmentB = AlignmentB;
  static ComplexTransform const kTransformA = ComplexTransform::kNone;
  static ComplexTransform const kTransformB = ComplexTransform::kNone;
  static bool const kSplitKSerial = SplitKSerial;

  using UnderlyingOperator = StrassenGemm<
    StrassenKind,
    ElementB,
    typename layout::LayoutTranspose<LayoutB>::type,
    ElementA,
    typename layout::LayoutTranspose<LayoutA>::type,
    ElementC,
    layout::RowMajor,
    ElementAccumulator,
    OperatorClass,
    ArchTag,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    EpilogueOutputOp,
    ThreadblockSwizzle,
    Stages,
    kAlignmentB,
    kAlignmentA,
    SplitKSerial,
    Operator,
    GatherB,
    GatherA,
    ScatterD,
    PermuteDLayout
  >;

  using UnderlyingArguments = typename UnderlyingOperator::Arguments;
  using GemmKernel = typename UnderlyingOperator::GemmKernel;
  static int const kAlignmentC = UnderlyingOperator::kAlignmentC;

  /// Argument structure
  struct Arguments {

    //
    // Data members
    //

    GemmCoord problem_size;
    TensorRef<ElementA const, LayoutA> ref_A;
    TensorRef<ElementB const, LayoutB> ref_B;
    TensorRef<ElementC const, LayoutC> ref_C;
    TensorRef<ElementC, LayoutC> ref_D;
    typename EpilogueOutputOp::Params epilogue;
    int split_k_slices;
    // For gather+scatter operations
    int *gather_A_indices;
    int *gather_B_indices;
    int *scatter_D_indices;

    //
    // Methods
    //

    /// Default ctor
    CUTLASS_HOST_DEVICE
    Arguments() { }

    /// Constructs an Arguments structure
    CUTLASS_HOST_DEVICE
    Arguments(
      GemmCoord problem_size_,
      TensorRef<ElementA const, LayoutA> ref_A_,
      TensorRef<ElementB const, LayoutB> ref_B_,
      TensorRef<ElementC const, LayoutC> ref_C_,
      TensorRef<ElementC, LayoutC> ref_D_,
      typename EpilogueOutputOp::Params epilogue_ =
        typename EpilogueOutputOp::Params(),
      int split_k_slices = 1,
      int *gather_A_indices_ = nullptr,
      int *gather_B_indices_ = nullptr,
      int *scatter_D_indices_ = nullptr
    ):
      problem_size(problem_size_),
      ref_A(ref_A_),
      ref_B(ref_B_),
      ref_C(ref_C_),
      ref_D(ref_D_),
      epilogue(epilogue_),
      split_k_slices(split_k_slices),
      gather_A_indices(gather_A_indices_),
      gather_B_indices(gather_B_indices_),
      scatter_D_indices(scatter_D_indices_) { }
  };

private:

  UnderlyingOperator underlying_operator_;

public:

  /// Constructs the GEMM.
  StrassenGemm() { }

  /// Helper to construct a transposed equivalent for the underying GEMM operator
  static UnderlyingArguments to_underlying_arguments(Arguments const &args) {
    return UnderlyingArguments(
      {args.problem_size.n(), args.problem_size.m(), args.problem_size.k()},
      {args.ref_B.data(), args.ref_B.stride(0)},
      {args.ref_A.data(), args.ref_A.stride(0)},
      {args.ref_C.data(), args.ref_C.stride(0)},
      {args.ref_D.data(), args.ref_D.stride(0)},
      args.epilogue,
      args.split_k_slices,
      args.gather_B_indices,
      args.gather_A_indices,
      args.scatter_D_indices
    );
  }

  /// Determines whether the GEMM can execute the given problem.
  static Status can_implement(Arguments const &args) {

    return UnderlyingOperator::can_implement(to_underlying_arguments(args));
  }

  /// Gets the workspace size
  static size_t get_workspace_size(Arguments const &args) {

    return UnderlyingOperator::get_workspace_size(to_underlying_arguments(args));
  }

  /// Initializes GEMM state from arguments.
  Status initialize(Arguments const &args, void *workspace = nullptr, cudaStream_t stream = nullptr) {

    return underlying_operator_.initialize(to_underlying_arguments(args), workspace);
  }

  /// Lightweight update given a subset of arguments
  Status update(Arguments const &args, void *workspace = nullptr) {

    return underlying_operator_.update(to_underlying_arguments(args), workspace);
  }

  /// Runs the kernel using initialized state.
  Status run(cudaStream_t stream = nullptr) {

    return underlying_operator_.run(stream);
  }

  /// Runs the kernel using initialized state.
  Status operator()(cudaStream_t stream = nullptr) {
    return run(stream);
  }

  /// Runs the kernel using initialized state.
  Status operator()(
    Arguments const &args,
    void *workspace = nullptr,
    cudaStream_t stream = nullptr) {

    Status status = initialize(args, workspace, stream);

    if (status == Status::kSuccess) {
      status = run(stream);
    }

    return status;
  }
};
*/
////////////////////////////////////////////////////////////////////////////////

} // namespace device
} // namespace gemm
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
