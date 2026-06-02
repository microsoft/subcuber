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
    \brief 
      Default kernel-level GEMM definitions combine threadblock-scoped matrix multiply-add with
      the appropriate threadblock-scoped epilogue.
  
      Note, CUTLASS epilogues universally target row-major outputs. Column-major outputs are
      accommodated by exchanging A and B operands and assuming transposed layouts. Partial
      specializations here choose 'device::GemmTransposed' to implement this functionality.
*/

#pragma once

#include "cutlass/cutlass.h"

#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"
#include "cutlass/arch/wmma.h"

#include "cutlass/epilogue/threadblock/epilogue.h"
#include "cutlass/epilogue/thread/linear_combination.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/strassen_gemm.h"
#include "cutlass/gemm/kernel/strassen_presum_gemm.h"
#include "cutlass/gemm/kernel/gemm_pipelined.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm75.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm70.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm80.h"
#include "cutlass/gemm/threadblock/default_strassen_mma.h"
#include "cutlass/gemm/threadblock/default_mma_core_simt.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"

#include "cutlass/epilogue/threadblock/default_strassen_epilogue_tensor_op.h"
#include "cutlass/epilogue/threadblock/default_epilogue_tensor_op.h"
#include "cutlass/epilogue/threadblock/default_epilogue_volta_tensor_op.h"
#include "cutlass/epilogue/threadblock/default_epilogue_simt.h"
#include "cutlass/epilogue/threadblock/default_strassen_epilogue_simt.h"
#include "cutlass/epilogue/threadblock/default_strassen_interim_epilogue_simt.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"

#include "cutlass/epilogue/threadblock/default_strassen_interim_epilogue_tensor_op.h"

#include "cutlass/layout/permute.h"

#if defined(CUTLASS_ARCH_WMMA_ENABLED)
#include "cutlass/epilogue/threadblock/default_epilogue_wmma_tensor_op.h"
#endif //CUTLASS_ARCH_WMMA_ENABLED

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

////////////////////////////////////////////////////////////////////////////////

template <
    /// Element type for A matrix operand
    typename ElementA_,
    /// Layout type for A matrix operand
    typename LayoutA_,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Element type for B matrix operand
    typename ElementB_,
    /// Layout type for B matrix operand
    typename LayoutB_,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for C and D matrix operands
    typename ElementC_,
    /// Layout type for C and D matrix operands
    typename LayoutC_,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Operator class tag
    typename OperatorClass,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape,
    MmaStrassen::Type StrassenKind,
    typename StrassenMiGroup,
    typename StrassenConsts, 
    // (concept : MmaStrassen::MmaStrassenConsts)
    /// Epilogue output operator
    typename EpilogueOutputOp,
    typename InterimEpilogueOp,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle,
    /// Number of stages used in the pipelined mainloop
    int Stages,
    /// If true, kernel is configured to support serial reduction in the
    /// epilogue
    bool SplitKSerial,
    bool SubGemmParallel,
    /// Operation performed by GEMM
    typename Operator,
    /// Use zfill or predicate for out-of-bound cp.async
    SharedMemoryClearOption SharedMemoryClear = SharedMemoryClearOption::kNone,
    /// Gather operand A by using an index array
    bool GatherA = false,
    /// Gather operand B by using an index array
    bool GatherB = false,
    /// Scatter result D by using an index array
    bool ScatterD = false,
    /// Permute result D
    typename PermuteDLayout = layout::NoPermute,
    /// Permute operand A
    typename PermuteALayout = layout::NoPermute,
    /// Permute operand B
    typename PermuteBLayout = layout::NoPermute,
    ///
    typename Enable = void
>
struct DefaultStrassenGemm;

////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for Hopper Architecture
template <
    /// Element type for A matrix operand
    typename ElementA,
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Element type for B matrix operand
    typename ElementB,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of A matrix in units of elements
    int kAlignmentB,
    /// Element type for C and D matrix operands
    typename ElementC,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape,
    MmaStrassen::Type StrassenKind,
    typename StrassenMiGroup,
    typename StrassenConsts,
    /// Epilogue output operator
    typename EpilogueOutputOp,
    typename InterimEpilogueOp,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle,
    /// Number of stages used in the pipelined mainloop
    int Stages,
    /// If true, kernel is configured to support serial reduction in the
    /// epilogue
    bool SplitKSerial,
    bool SubGemmParallel,
    /// Operation performed by GEMM
    typename Operator,
    /// Use zfill or predicate for out-of-bound cp.async
    SharedMemoryClearOption SharedMemoryClear,
    /// Gather operand A by using an index array
    bool GatherA,
    /// Gather operand B by using an index array
    bool GatherB,
    /// Scatter result D by using an index array
    bool ScatterD,
    /// Permute result D
    typename PermuteDLayout,
    /// Permute operand A
    typename PermuteALayout,
    /// Permute operand B
    typename PermuteBLayout
>
struct DefaultStrassenGemm<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB, ElementC,
                   layout::RowMajor, ElementAccumulator, arch::OpClassTensorOp,
                   arch::Sm90, ThreadblockShape, WarpShape, InstructionShape, StrassenKind, StrassenMiGroup,
                   StrassenConsts, EpilogueOutputOp, InterimEpilogueOp, ThreadblockSwizzle, Stages, SplitKSerial, SubGemmParallel,
                   Operator, SharedMemoryClear, GatherA, GatherB, ScatterD,
                   PermuteDLayout, PermuteALayout, PermuteBLayout> {
  /// Define the threadblock-scoped matrix multiply-accumulate
  using Mma = typename cutlass::gemm::threadblock::DefaultStrassenMma<
      ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB,
      ElementAccumulator, layout::RowMajor, arch::OpClassTensorOp, arch::Sm90,
      ThreadblockShape, WarpShape, InstructionShape, StrassenKind, StrassenMiGroup, StrassenConsts, Stages,
      Operator, InterimEpilogueOp, false, SharedMemoryClear, GatherA, GatherB, 
      PermuteALayout, PermuteBLayout>::ThreadblockMma;

  static const int kPartitionsK = ThreadblockShape::kK / WarpShape::kK;

  /// Define the epilogue
  using Epilogue =
      typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
          ThreadblockShape, typename Mma::Operator, kPartitionsK, EpilogueOutputOp,
          EpilogueOutputOp::kCount, ScatterD, PermuteDLayout>::Epilogue;

  /// Define the kernel-level GEMM operator.
  using GemmKernel = kernel::StrassenGemm<StrassenMiGroup, Mma, Epilogue, Epilogue, ThreadblockSwizzle, SplitKSerial, SubGemmParallel>;
};

////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for Ada Architecture
template <
    /// Element type for A matrix operand
    typename ElementA,
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Element type for B matrix operand
    typename ElementB,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of A matrix in units of elements
    int kAlignmentB,
    /// Element type for C and D matrix operands
    typename ElementC,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape,
    MmaStrassen::Type StrassenKind,
    typename StrassenMiGroup,
    typename StrassenConsts,
    /// Epilogue output operator
    typename EpilogueOutputOp,
    typename InterimEpilogueOp,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle,
    /// Number of stages used in the pipelined mainloop
    int Stages,
    /// If true, kernel is configured to support serial reduction in the
    /// epilogue
    bool SplitKSerial,
    bool SubGemmParallel,
    /// Operation performed by GEMM
    typename Operator,
    /// Use zfill or predicate for out-of-bound cp.async
    SharedMemoryClearOption SharedMemoryClear,
    /// Gather operand A by using an index array
    bool GatherA,
    /// Gather operand B by using an index array
    bool GatherB,
    /// Scatter result D by using an index array
    bool ScatterD,
    /// Permute result D
    typename PermuteDLayout,
    /// Permute operand A
    typename PermuteALayout,
    /// Permute operand B
    typename PermuteBLayout
>
struct DefaultStrassenGemm<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB, ElementC,
                   layout::RowMajor, ElementAccumulator, arch::OpClassTensorOp,
                   arch::Sm89, ThreadblockShape, WarpShape, InstructionShape, StrassenKind, StrassenMiGroup, StrassenConsts,
                   EpilogueOutputOp, InterimEpilogueOp, ThreadblockSwizzle, Stages, SplitKSerial, SubGemmParallel,
                   Operator, SharedMemoryClear, GatherA, GatherB, ScatterD, 
                   PermuteDLayout, PermuteALayout, PermuteBLayout> {
  /// Define the threadblock-scoped matrix multiply-accumulate
  using Mma = typename cutlass::gemm::threadblock::DefaultStrassenMma<
      ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB,
      ElementAccumulator, layout::RowMajor, arch::OpClassTensorOp, arch::Sm89,
      ThreadblockShape, WarpShape, InstructionShape, StrassenKind, StrassenMiGroup, StrassenConsts, Stages,
      Operator, InterimEpilogueOp, false, SharedMemoryClear, GatherA, GatherB,
      PermuteALayout, PermuteBLayout>::ThreadblockMma;

  static const int kPartitionsK = ThreadblockShape::kK / WarpShape::kK;

  /// Define the epilogue
  using Epilogue =
      typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
          ThreadblockShape, typename Mma::Operator, kPartitionsK, EpilogueOutputOp,
          EpilogueOutputOp::kCount, ScatterD, PermuteDLayout>::Epilogue;

  /// Define the kernel-level GEMM operator.
  using GemmKernel = kernel::StrassenGemm<StrassenMiGroup, Mma, Epilogue, Epilogue, ThreadblockSwizzle, SplitKSerial, SubGemmParallel>;
};

////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for Ampere Architecture
template <
    /// Element type for A matrix operand
    typename ElementA,
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Element type for B matrix operand
    typename ElementB,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of A matrix in units of elements
    int kAlignmentB,
    /// Element type for C and D matrix operands
    typename ElementC,
    /// Layout type for C and D matrix operand
    typename LayoutC,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape,
    MmaStrassen::Type StrassenKind,
    typename StrassenMiGroup,
    typename StrassenConsts, 
    /// Epilogue output operator
    typename EpilogueOutputOp,
    typename InterimEpilogueOp,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle,
    /// Number of stages used in the pipelined mainloop
    int Stages,
    /// If true, kernel is configured to support serial reduction in the
    /// epilogue
    bool SplitKSerial,
    bool SubGemmParallel,
    /// Operation performed by GEMM
    typename Operator,
    /// Use zfill or predicate for out-of-bound cp.async
    SharedMemoryClearOption SharedMemoryClear,
    /// Gather operand A by using an index array
    bool GatherA,
    /// Gather operand B by using an index array
    bool GatherB,
    /// Scatter result D by using an index array
    bool ScatterD,
    /// Permute result D
    typename PermuteDLayout,
    /// Permute operand A
    typename PermuteALayout,
    /// Permute operand B
    typename PermuteBLayout
>
struct DefaultStrassenGemm<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB, ElementC,
                   LayoutC, ElementAccumulator, arch::OpClassTensorOp,
                   arch::Sm80, ThreadblockShape, WarpShape, InstructionShape, StrassenKind, StrassenMiGroup, StrassenConsts,
                   EpilogueOutputOp, InterimEpilogueOp, ThreadblockSwizzle, Stages, SplitKSerial, SubGemmParallel,
                   Operator, SharedMemoryClear, GatherA, GatherB, ScatterD,
                   PermuteDLayout, PermuteALayout, PermuteBLayout> {

  static_assert((platform::is_same<LayoutC, layout::RowMajor>::value
             || platform::is_same<LayoutC, layout::AffineRankN<2>>::value),
             "Epilogue in the kernel level must be row major");
  // Using this specialization
  /// Define the threadblock-scoped matrix multiply-accumulate
  using Mma = typename cutlass::gemm::threadblock::DefaultStrassenMma<
      ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB,
      ElementAccumulator, LayoutC, arch::OpClassTensorOp, arch::Sm80,
      ThreadblockShape, WarpShape, InstructionShape,
      StrassenKind, StrassenMiGroup, StrassenConsts, Stages,
      Operator, InterimEpilogueOp, false, SharedMemoryClear, GatherA, GatherB,
      PermuteALayout, PermuteBLayout>::ThreadblockMma;

  static const int kPartitionsK = ThreadblockShape::kK / WarpShape::kK;

  /// Define the epilogue
  using InterimEpilogue = typename cutlass::epilogue::threadblock::DefaultStrassenInterimEpilogueTensorOp<
    ThreadblockShape,
    typename Mma::StrassenShape,
    typename Mma::Operator,
    kPartitionsK,
    InterimEpilogueOp,
    InterimEpilogueOp::kCount,
    Mma::NumThreads
    >::InterimEpilogue;

  using RegularEpilogue =
      typename cutlass::epilogue::threadblock::DefaultStrassenEpilogueTensorOp<
        ThreadblockShape, typename Mma::StrassenShape, StrassenKind, typename Mma::Operator, kPartitionsK, EpilogueOutputOp,
        EpilogueOutputOp::kCount, typename InterimEpilogue::OutputTileIterator, StrassenMiGroup, ScatterD, PermuteDLayout>::Epilogue;

  using Affine2Epilogue =
      typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOpAffineRankN<
          2, ThreadblockShape, typename Mma::Operator, kPartitionsK, EpilogueOutputOp,
          EpilogueOutputOp::kCount>::Epilogue;

  using Epilogue = typename platform::conditional<platform::is_same<LayoutC, layout::RowMajor>::value,
                                                  RegularEpilogue,
                                                  Affine2Epilogue>::type;

  /// Define the kernel-level GEMM operator.
  using GemmKernel1 = kernel::StrassenGemm<StrassenMiGroup, Mma, Epilogue, InterimEpilogue, ThreadblockSwizzle, SplitKSerial, SubGemmParallel>;
  using GemmKernel = GemmKernel1;

//   using GemmKernel2 = kernel::StrassenPreSumGemm<Mma, Epilogue, ThreadblockSwizzle, SplitKSerial, SubGemmParallel>;
//   static const bool IsGlobalPreSumStrassenMatrixMma = 
//                                           Mma::MmaStrassenKind == MmaStrassen::Type::GlobalPreSumLevel1_M0 || 
//                                           Mma::MmaStrassenKind == MmaStrassen::Type::GlobalPreSumLevel1_M1 ||
//                                           Mma::MmaStrassenKind == MmaStrassen::Type::GlobalPreSumLevel1_M2 ||
//                                           Mma::MmaStrassenKind == MmaStrassen::Type::GlobalPreSumLevel1_M3 ||
//                                           Mma::MmaStrassenKind == MmaStrassen::Type::GlobalPreSumLevel1_M4 ||
//                                           Mma::MmaStrassenKind == MmaStrassen::Type::GlobalPreSumLevel1_M5 ||
//                                           Mma::MmaStrassenKind == MmaStrassen::Type::GlobalPreSumLevel1_M6;

//   using GemmKernel = typename platform::conditional<!IsGlobalPreSumStrassenMatrixMma, GemmKernel1, GemmKernel2>::type;
};

////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for Turing Architecture
template <
  /// Element type for A matrix operand
  typename ElementA,
  /// Layout type for A matrix operand
  typename LayoutA,
  /// Access granularity of A matrix in units of elements
  int kAlignmentA,
  /// Element type for B matrix operand
  typename ElementB,
  /// Layout type for B matrix operand
  typename LayoutB,
  /// Access granularity of B matrix in units of elements
  int kAlignmentB,
  /// Element type for C and D matrix operands
  typename ElementC,
  /// Element type for internal accumulation
  typename ElementAccumulator,
  /// Threadblock-level tile size (concept: GemmShape)
  typename ThreadblockShape,
  /// Warp-level tile size (concept: GemmShape)
  typename WarpShape,
  /// Warp-level tile size (concept: GemmShape)
  typename InstructionShape,
  MmaStrassen::Type StrassenKind,
  typename StrassenMiGroup,
  typename StrassenConsts,
  /// Epilogue output operator
  typename EpilogueOutputOp,
  typename InterimEpilogueOp,
  /// Threadblock-level swizzling operator
  typename ThreadblockSwizzle,
  /// If true, kernel is configured to support serial reduction in the epilogue
  bool SplitKSerial,
  bool SubGemmParallel,
  /// Operation performed by GEMM
  typename Operator,
  /// Use zfill or predicate for out-of-bound cp.async
  SharedMemoryClearOption SharedMemoryClear,
  /// Gather operand A by using an index array
  bool GatherA,
  /// Gather operand B by using an index array
  bool GatherB,
  /// Scatter result D by using an index array
  bool ScatterD,
  /// Permute result D
  typename PermuteDLayout,
  /// Permute operand A
  typename PermuteALayout,
  /// Permute operand B
  typename PermuteBLayout
>
struct DefaultStrassenGemm<
  ElementA, LayoutA, kAlignmentA,
  ElementB, LayoutB, kAlignmentB,
  ElementC, layout::RowMajor,
  ElementAccumulator,
  arch::OpClassTensorOp,
  arch::Sm75,
  ThreadblockShape,
  WarpShape,
  InstructionShape,
  StrassenKind,
  StrassenMiGroup,
  StrassenConsts,
  EpilogueOutputOp,
  InterimEpilogueOp,
  ThreadblockSwizzle,
  2,
  SplitKSerial, SubGemmParallel,
  Operator,
  SharedMemoryClear,
  GatherA,
  GatherB,
  ScatterD,
  PermuteDLayout,
  PermuteALayout,
  PermuteBLayout
> {

  /// Define the threadblock-scoped matrix multiply-accumulate
  using Mma = typename cutlass::gemm::threadblock::DefaultStrassenMma<
    ElementA,
    LayoutA,
    kAlignmentA,
    ElementB,
    LayoutB,
    kAlignmentB,
    ElementAccumulator,
    layout::RowMajor,
    arch::OpClassTensorOp,
    arch::Sm75,
    ThreadblockShape,
    WarpShape,
    InstructionShape,
    StrassenKind, StrassenMiGroup, StrassenConsts,
    2,
    Operator,
    InterimEpilogueOp,
    false,
    SharedMemoryClear,
    GatherA,
    GatherB,
    PermuteALayout,
    PermuteBLayout
  >::ThreadblockMma;

  static const int kPartitionsK = ThreadblockShape::kK / WarpShape::kK;

  /// Define the epilogue
  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
    ThreadblockShape,
    typename Mma::Operator,
    kPartitionsK,
    EpilogueOutputOp,
    EpilogueOutputOp::kCount,
    ScatterD,
    PermuteDLayout
  >::Epilogue;

  using InterimEpilogue = typename cutlass::epilogue::threadblock::DefaultStrassenInterimEpilogueSimt<
      ThreadblockShape,
      typename Mma::StrassenShape,
      typename Mma::Operator,
      InterimEpilogueOp,
      InterimEpilogueOp::kCount,
      Mma::NumThreads
      >::InterimEpilogue;

  /// Define the kernel-level GEMM operator.
  using GemmKernel = kernel::StrassenGemm<StrassenMiGroup, Mma, Epilogue, InterimEpilogue, ThreadblockSwizzle, SplitKSerial, SubGemmParallel>;
};

////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for Ampere Integer Matrix Multiply Interleaved layout
template <
    /// Element type for A matrix operand
    typename ElementA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Element type for B matrix operand
    typename ElementB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for C and D matrix operands
    typename ElementC,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape,
    MmaStrassen::Type StrassenKind,
    typename StrassenMiGroup,
    typename StrassenConsts,
    /// Epilogue output operator
    typename EpilogueOutputOp,
    typename InterimEpilogueOp,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle,
    /// Number of stages used in the pipelined mainloop
    int Stages,
    /// Number of Interleaved k
    int InterleavedK,
    /// If true, kernel is configured to support serial reduction in the
    /// epilogue
    bool SplitKSerial,
    bool SubGemmParallel,
    /// Operation performed by GEMM
    typename Operator,
    /// Use zfill or predicate for out-of-bound cp.async
    SharedMemoryClearOption SharedMemoryClear>
struct DefaultStrassenGemm<
    ElementA, layout::ColumnMajorInterleaved<InterleavedK>, kAlignmentA,
    ElementB, layout::RowMajorInterleaved<InterleavedK>, kAlignmentB, ElementC,
    layout::ColumnMajorInterleaved<InterleavedK>, int32_t,
    arch::OpClassTensorOp, arch::Sm80, ThreadblockShape, WarpShape,
    InstructionShape, StrassenKind, StrassenMiGroup, StrassenConsts, EpilogueOutputOp,
    InterimEpilogueOp, ThreadblockSwizzle, Stages,
    SplitKSerial, SubGemmParallel, Operator, SharedMemoryClear, false, false, false> {

  using LayoutA = layout::ColumnMajorInterleaved<InterleavedK>;
  using LayoutB = layout::RowMajorInterleaved<InterleavedK>;
  using LayoutC = layout::ColumnMajorInterleaved<InterleavedK>;

  using ElementAccumulator = int32_t;

  /// Define the threadblock-scoped matrix multiply-accumulate
  using Mma = typename cutlass::gemm::threadblock::DefaultStrassenMma<
      ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB,
      ElementAccumulator, LayoutC, arch::OpClassTensorOp, arch::Sm80,
      ThreadblockShape, WarpShape, InstructionShape, StrassenKind, StrassenMiGroup, StrassenConsts, Stages, Operator,
      InterimEpilogueOp, true, SharedMemoryClear>::ThreadblockMma;

  static const int kPartitionsK = ThreadblockShape::kK / WarpShape::kK;

  /// Define the epilogue
  using Epilogue = typename cutlass::epilogue::threadblock::
      DefaultInterleavedEpilogueTensorOp<
          ThreadblockShape, typename Mma::Operator, kPartitionsK, EpilogueOutputOp,
          64 / sizeof_bits<ElementC>::value, InterleavedK>::Epilogue;

  using InterimEpilogue = typename cutlass::epilogue::threadblock::DefaultStrassenInterimEpilogueSimt<
      ThreadblockShape,
      typename Mma::StrassenShape,
      typename Mma::Operator,
      InterimEpilogueOp,
      64 / sizeof_bits<ElementC>::value,
      Mma::NumThreads
      >::InterimEpilogue;
  
  /// Define the kernel-level GEMM operator.
  using GemmKernel = kernel::StrassenGemm<StrassenMiGroup, Mma, Epilogue, InterimEpilogue, ThreadblockSwizzle, SplitKSerial, SubGemmParallel>;
};

////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for Turing Integer Matrix Multiply Interleaved layout
template <
    /// Element type for A matrix operand
    typename ElementA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Element type for B matrix operand
    typename ElementB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for C and D matrix operands
    typename ElementC,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape,
    MmaStrassen::Type StrassenKind,
    typename StrassenMiGroup,
    typename StrassenConsts,
    /// Epilogue output operator
    typename EpilogueOutputOp,
    typename InterimEpilogueOp,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle,
    /// Number of Interleaved k
    int InterleavedK,
    /// If true, kernel is configured to support serial reduction in the
    /// epilogue
    bool SplitKSerial,
    bool SubGemmParallel,
    /// Operation performed by GEMM
    typename Operator,
    /// Use zfill or predicate for out-of-bound cp.async
    SharedMemoryClearOption SharedMemoryClear>
struct DefaultStrassenGemm<ElementA, layout::ColumnMajorInterleaved<InterleavedK>,
                   kAlignmentA, ElementB,
                   layout::RowMajorInterleaved<InterleavedK>, kAlignmentB,
                   ElementC, layout::ColumnMajorInterleaved<InterleavedK>,
                   int32_t, arch::OpClassTensorOp, arch::Sm75, ThreadblockShape,
                   WarpShape, InstructionShape, StrassenKind, StrassenMiGroup, StrassenConsts, EpilogueOutputOp,
                   InterimEpilogueOp, ThreadblockSwizzle, 2, SplitKSerial, SubGemmParallel, Operator, SharedMemoryClear,
                   false, false, false> {

  using LayoutA = layout::ColumnMajorInterleaved<InterleavedK>;
  using LayoutB = layout::RowMajorInterleaved<InterleavedK>;
  using LayoutC = layout::ColumnMajorInterleaved<InterleavedK>;

  using ElementAccumulator = int32_t;

  /// Define the threadblock-scoped matrix multiply-accumulate
  using Mma = typename cutlass::gemm::threadblock::DefaultStrassenMma<
      ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB, ElementAccumulator, LayoutC,
      arch::OpClassTensorOp, arch::Sm75, ThreadblockShape, WarpShape,
      InstructionShape, StrassenKind, StrassenMiGroup, StrassenConsts, 2, Operator, InterimEpilogueOp, true>::ThreadblockMma;

  static const int kPartitionsK = ThreadblockShape::kK / WarpShape::kK;

  /// Define the epilogue
  using Epilogue = typename cutlass::epilogue::threadblock::
      DefaultInterleavedEpilogueTensorOp<
          ThreadblockShape, typename Mma::Operator, kPartitionsK, EpilogueOutputOp,
          64 / sizeof_bits<ElementC>::value, InterleavedK>::Epilogue;

  using InterimEpilogue = typename cutlass::epilogue::threadblock::DefaultStrassenInterimEpilogueSimt<
      ThreadblockShape,
      typename Mma::StrassenShape,
      typename Mma::Operator,
      InterimEpilogueOp,
      64 / sizeof_bits<ElementC>::value,
      Mma::NumThreads
      >::InterimEpilogue;
  /// Define the kernel-level GEMM operator.
  using GemmKernel = kernel::StrassenGemm<StrassenMiGroup, Mma, Epilogue, InterimEpilogue, ThreadblockSwizzle, SplitKSerial, SubGemmParallel>;
};

////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for Volta architecture
template <
  /// Element type for A matrix operand
  typename ElementA,
  /// Layout type for A matrix operand
  typename LayoutA,
  /// Access granularity of A matrix in units of elements
  int kAlignmentA,
  /// Element type for B matrix operand
  typename ElementB,
  /// Layout type for B matrix operand
  typename LayoutB,
  /// Access granularity of B matrix in units of elements
  int kAlignmentB,
  /// Element type for C and D matrix operands
  typename ElementC,
  /// Element type for internal accumulation
  typename ElementAccumulator,
  /// Threadblock-level tile size (concept: GemmShape)
  typename ThreadblockShape,
  /// Warp-level tile size (concept: GemmShape)
  typename WarpShape,
  /// Epilogue output operator
  typename EpilogueOutputOp,
  typename InterimEpilogueOp,
  MmaStrassen::Type StrassenKind,
  typename StrassenMiGroup,
  typename StrassenConsts,
  /// Threadblock-level swizzling operator
  typename ThreadblockSwizzle,
  /// If true, kernel is configured to support serial reduction in the epilogue
  bool SplitKSerial,
  bool SubGemmParallel,
  /// Operation performed by GEMM
  typename Operator,
  /// Use zfill or predicate for out-of-bound cp.async
  SharedMemoryClearOption SharedMemoryClear,
  /// Gather operand A by using an index array
  bool GatherA,
  /// Gather operand B by using an index array
  bool GatherB,
  /// Scatter result D by using an index array
  bool ScatterD,
  /// Permute result D
  typename PermuteDLayout,
  /// Permute operand A
  typename PermuteALayout,
  /// Permute operand B
  typename PermuteBLayout
>
struct DefaultStrassenGemm<
  ElementA, LayoutA, kAlignmentA,
  ElementB, LayoutB, kAlignmentB,
  ElementC, layout::RowMajor,
  ElementAccumulator,
  arch::OpClassTensorOp,
  arch::Sm70,
  ThreadblockShape,
  WarpShape,
  GemmShape<8, 8, 4>,
  StrassenKind, StrassenMiGroup, StrassenConsts,
  EpilogueOutputOp,
  InterimEpilogueOp,
  ThreadblockSwizzle,
  2,
  SplitKSerial, SubGemmParallel,
  Operator,
  SharedMemoryClear,
  GatherA,
  GatherB,
  ScatterD,
  PermuteDLayout,
  PermuteALayout,
  PermuteBLayout
> {

  /// Define the threadblock-scoped matrix multiply-accumulate
  using Mma = typename cutlass::gemm::threadblock::DefaultStrassenMma<
    ElementA,
    LayoutA,
    kAlignmentA,
    ElementB,
    LayoutB,
    kAlignmentB,
    ElementAccumulator,
    layout::RowMajor,
    arch::OpClassTensorOp,
    arch::Sm70,
    ThreadblockShape,
    WarpShape,
    GemmShape<8, 8, 4>,
    StrassenKind, StrassenMiGroup, StrassenConsts, 2,
    Operator,
    InterimEpilogueOp,
    false,
    SharedMemoryClear,
    GatherA,
    GatherB,
    PermuteALayout,
    PermuteBLayout
  >::ThreadblockMma;

  static const int kPartitionsK = ThreadblockShape::kK / WarpShape::kK;

  /// Define the epilogue
  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueVoltaTensorOp<
    ThreadblockShape,
    typename Mma::Operator,
    kPartitionsK,
    EpilogueOutputOp,
    EpilogueOutputOp::kCount,
    ScatterD,
    PermuteDLayout
  >::Epilogue;

  using InterimEpilogue = typename cutlass::epilogue::threadblock::DefaultStrassenInterimEpilogueSimt<
      ThreadblockShape,
      typename Mma::StrassenShape,
      typename Mma::Operator,
      InterimEpilogueOp,
      InterimEpilogueOp::kCount,
      Mma::NumThreads
      >::InterimEpilogue;
  /// Define the kernel-level GEMM operator.
  using GemmKernel = kernel::StrassenGemm<StrassenMiGroup, Mma, Epilogue, InterimEpilogue, ThreadblockSwizzle, SplitKSerial, SubGemmParallel>;
};

////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for SIMT
template <
    /// Element type for A matrix operand
    typename ElementA,
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Element type for B matrix operand
    typename ElementB,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of A matrix in units of elements
    int kAlignmentB,
    /// Element type for C and D matrix operands
    typename ElementC,
    /// Layout type for C and D matrix operand
    typename LayoutC,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    MmaStrassen::Type StrassenKind,
    typename StrassenMiGroup,
    typename StrassenConsts,
    /// Epilogue output operator
    typename EpilogueOutputOp,
    typename InterimEpilogueOp,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle,
    /// If true, kernel is configured to support serial reduction in the epilogue
    bool SplitKSerial,
    bool SubGemmParallel,
    /// Operation performed by GEMM
    typename Operator,
    /// Use zfill or predicate for out-of-bound cp.async
    SharedMemoryClearOption SharedMemoryClear,
    /// Gather operand A by using an index array
    bool GatherA,
    /// Gather operand B by using an index array
    bool GatherB,
    /// Scatter result D by using an index array
    bool ScatterD,
    /// Permute result D
    typename PermuteDLayout,
    /// Permute operand A
    typename PermuteALayout,
    /// Permute operand B
    typename PermuteBLayout
  >
struct DefaultStrassenGemm<
    ElementA,
    LayoutA,
    kAlignmentA,
    ElementB,
    LayoutB,
    kAlignmentB,
    ElementC,
    LayoutC,
    ElementAccumulator,
    arch::OpClassSimt,
    ArchTag,
    ThreadblockShape,
    WarpShape,
    GemmShape<1, 1, 1>,
    StrassenKind, StrassenMiGroup, StrassenConsts,
    EpilogueOutputOp,
    InterimEpilogueOp,
    ThreadblockSwizzle,
    2, //(StrassenKind == MmaStrassen::Type::Normal) ? 2 : 1,
    SplitKSerial, SubGemmParallel,
    Operator,
    SharedMemoryClear,
    GatherA,
    GatherB,
    ScatterD,
    PermuteDLayout,
    PermuteALayout,
    PermuteBLayout,
    typename platform::enable_if< ! platform::is_same<ArchTag, arch::Sm80>::value >::type > {

  static_assert((platform::is_same<LayoutC, layout::RowMajor>::value
             || platform::is_same<LayoutC, layout::AffineRankN<2>>::value),
             "Epilogue in the kernel level must be row major");

  /// Define the threadblock-scoped matrix multiply-accumulate
  using Mma = typename cutlass::gemm::threadblock::DefaultStrassenMma<
      ElementA,
      LayoutA,
      kAlignmentA,
      ElementB,
      LayoutB,
      kAlignmentB,
      ElementAccumulator,
      LayoutC,
      arch::OpClassSimt,
      arch::Sm50,
      GemmShape<ThreadblockShape::kM, ThreadblockShape::kN, ThreadblockShape::kK>,
      WarpShape,
      GemmShape<1, 1, 1>,
      StrassenKind, StrassenMiGroup, StrassenConsts,
      2,// (StrassenKind == MmaStrassen::Type::Normal) ? 2 : 1,
      Operator,
      InterimEpilogueOp,
      false,
      SharedMemoryClear,
      GatherA,
      GatherB,
      PermuteALayout,
      PermuteBLayout>::ThreadblockMma;

  static int const kEpilogueElementsPerAccess = EpilogueOutputOp::kCount;
  static_assert(kEpilogueElementsPerAccess == 1, "simt epilogue must operate on scalars");

  /// Define the epilogue
  using InterimEpilogue = typename cutlass::epilogue::threadblock::DefaultStrassenInterimEpilogueSimt<
      ThreadblockShape,
      typename Mma::StrassenShape,
      typename Mma::Operator,
      InterimEpilogueOp,
      InterimEpilogueOp::kCount,
      Mma::NumThreads
      >::InterimEpilogue;

  using RegularEpilogue = typename cutlass::epilogue::threadblock::DefaultStrassenEpilogueSimt<
      ThreadblockShape,
      typename Mma::StrassenShape,
      typename Mma::Operator,
      EpilogueOutputOp,
      kEpilogueElementsPerAccess,
      typename InterimEpilogue::OutputTileIterator,
      StrassenMiGroup,
      ScatterD,
      PermuteDLayout
      >::Epilogue;

  using Affine2Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueSimtAffineRankN<
      2,
      ThreadblockShape,
      typename Mma::Operator,
      EpilogueOutputOp,
      kEpilogueElementsPerAccess
      >::Epilogue;

  using Epilogue = RegularEpilogue ;//typename platform::conditional<platform::is_same<LayoutC, layout::RowMajor>::value,
                                                  // RegularEpilogue,
                                                  // Affine2Epilogue>::type;

  /// Define the kernel-level GEMM operator.
  using GemmKernel = kernel::StrassenGemm<StrassenMiGroup, Mma, Epilogue, InterimEpilogue, ThreadblockSwizzle, SplitKSerial, SubGemmParallel>;
};

////////////////////////////////////////////////////////////////////////////////

/// Partial specialization for Ampere
template <
    /// Element type for A matrix operand
    typename ElementA,
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Element type for B matrix operand
    typename ElementB,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of A matrix in units of elements
    int kAlignmentB,
    /// Element type for C and D matrix operands
    typename ElementC,
    /// Layout type for C and D matrix operand
    typename LayoutC,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    MmaStrassen::Type StrassenKind,
    typename StrassenMiGroup,
    typename StrassenConsts,
    /// Epilogue output operator
    typename EpilogueOutputOp,
    typename InterimEpilogueOp,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle,
    /// Number of stages
    int Stages,
    /// If true, kernel is configured to support serial reduction in the epilogue
    bool SplitKSerial,
    bool SubGemmParallel,
    /// Operation performed by GEMM
    typename Operator,
    /// Use zfill or predicate for out-of-bound cp.async
    SharedMemoryClearOption SharedMemoryClear,
    /// Gather operand A by using an index array
    bool GatherA,
    /// Gather operand B by using an index array
    bool GatherB,
    /// Scatter result D by using an index array
    bool ScatterD,
    /// Permute result D
    typename PermuteDLayout,
    /// Permute operand A
    typename PermuteALayout,
    /// Permute operand B
    typename PermuteBLayout
>
struct DefaultStrassenGemm<ElementA,
                   LayoutA,
                   kAlignmentA,
                   ElementB,
                   LayoutB,
                   kAlignmentB,
                   ElementC,
                   LayoutC,
                   ElementAccumulator,
                   arch::OpClassSimt,
                   arch::Sm80,
                   ThreadblockShape,
                   WarpShape,
                   GemmShape<1, 1, 1>,
                   StrassenKind, StrassenMiGroup, StrassenConsts,
                   EpilogueOutputOp,
                   InterimEpilogueOp,
                   ThreadblockSwizzle,
                   Stages,
                   SplitKSerial, SubGemmParallel,
                   Operator,
                   SharedMemoryClear,
                   GatherA,
                   GatherB,
                   ScatterD,
                   PermuteDLayout,
                   PermuteALayout,
                   PermuteBLayout> {

  static_assert((platform::is_same<LayoutC, layout::RowMajor>::value
             || platform::is_same<LayoutC, layout::AffineRankN<2>>::value),
             "Epilogue in the kernel level must be row major");

  /// Define the threadblock-scoped matrix multiply-accumulate
  using Mma = typename cutlass::gemm::threadblock::DefaultStrassenMma<
      ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB,
      ElementAccumulator, LayoutC, arch::OpClassSimt, arch::Sm80,
      ThreadblockShape, WarpShape, GemmShape<1, 1, 1>, StrassenKind, StrassenMiGroup, StrassenConsts, Stages,
      Operator, InterimEpilogueOp, false, SharedMemoryClear, GatherA, GatherB,
      PermuteALayout, PermuteBLayout>::ThreadblockMma;
  static int const kEpilogueElementsPerAccess = EpilogueOutputOp::kCount;
  static_assert(kEpilogueElementsPerAccess == 1, "simt epilogue must operate on scalars");

  using InterimEpilogue = typename cutlass::epilogue::threadblock::DefaultStrassenInterimEpilogueSimt<
      ThreadblockShape,
      typename Mma::StrassenShape,
      typename Mma::Operator,
      InterimEpilogueOp,
      InterimEpilogueOp::kCount,
      Mma::NumThreads
      >::InterimEpilogue;

  /// Define the epilogue
  using RegularEpilogue = typename cutlass::epilogue::threadblock::DefaultStrassenEpilogueSimt<
      ThreadblockShape,
      typename Mma::StrassenShape,
      typename Mma::Operator,
      EpilogueOutputOp,
      kEpilogueElementsPerAccess,
      typename InterimEpilogue::OutputTileIterator,
      StrassenMiGroup,
      ScatterD,
      PermuteDLayout
      >::Epilogue;

  using Affine2Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueSimtAffineRankN<
      2,
      ThreadblockShape,
      typename Mma::Operator,
      EpilogueOutputOp,
      kEpilogueElementsPerAccess
      >::Epilogue;

  using Epilogue = RegularEpilogue; 
    // typename platform::conditional<platform::is_same<LayoutC, layout::RowMajor>::value,
    //                                               RegularEpilogue,
    //                                               Affine2Epilogue>::type;

  /// Define the kernel-level GEMM operator.
  using GemmKernel = kernel::StrassenGemm<StrassenMiGroup, Mma, Epilogue, InterimEpilogue, ThreadblockSwizzle, SplitKSerial, SubGemmParallel>; 
};

////////////////////////////////////////////////////////////////////////////////
/// Partial specialization for SIMT DP4A

template <
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of A matrix in units of elements
    int kAlignmentB,
    /// Layout type for C matrix operand
    typename LayoutC,
    /// Element type for C and D matrix operands
    typename ElementC,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    MmaStrassen::Type StrassenKind,
    typename StrassenMiGroup,
    typename StrassenConsts,
    /// Epilogue output operator
    typename EpilogueOutputOp,
    typename InterimEpilogueOp,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle,
    /// If true, kernel is configured to support serial reduction in the
    /// epilogue
    bool SplitKSerial,
    bool SubGemmParallel,
    /// Operation performed by GEMM
    typename Operator,
    /// Use zfill or predicate for out-of-bound cp.async
    SharedMemoryClearOption SharedMemoryClear
>
struct DefaultStrassenGemm<int8_t, LayoutA, kAlignmentA, int8_t, LayoutB, kAlignmentB,
                   ElementC, LayoutC, ElementAccumulator, arch::OpClassSimt,
                   ArchTag, ThreadblockShape, WarpShape, GemmShape<1, 1, 4>, StrassenKind, StrassenMiGroup, StrassenConsts,
                   EpilogueOutputOp, InterimEpilogueOp, ThreadblockSwizzle, 2, SplitKSerial, SubGemmParallel,
                   Operator, SharedMemoryClear, false, false, false,
                   layout::NoPermute, layout::NoPermute> {
  using InstructionShape = GemmShape<1, 1, 4>;
  using ElementA = int8_t;
  using ElementB = int8_t;

  using OperatorClass =  arch::OpClassSimt;
  /// Define the threadblock-scoped matrix multiply-accumulate
  using Mma = typename cutlass::gemm::threadblock::DefaultStrassenMma<
      ElementA,
      LayoutA,
      kAlignmentA,
      ElementB,
      LayoutB,
      kAlignmentB,
      ElementAccumulator,
      LayoutC,
      arch::OpClassSimt,
      arch::Sm50,
      ThreadblockShape,
      WarpShape,
      InstructionShape,
      StrassenKind, StrassenMiGroup, StrassenConsts,
      2,
      Operator,
      InterimEpilogueOp
      >::ThreadblockMma;

  static int const kEpilogueElementsPerAccess = EpilogueOutputOp::kCount;
  static_assert(kEpilogueElementsPerAccess == 1, "simt epilogue must operate on scalars");

  /// Define the epilogue
  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueSimt<
      ThreadblockShape,
      typename Mma::Operator,
      EpilogueOutputOp,
      kEpilogueElementsPerAccess
      >::Epilogue;

  /// Define the kernel-level GEMM operator.
  using GemmKernel = kernel::StrassenGemm<StrassenMiGroup, Mma, Epilogue, Epilogue, ThreadblockSwizzle, SplitKSerial, SubGemmParallel>;
};

#if defined(CUTLASS_ARCH_WMMA_ENABLED)
////////////////////////////////////////////////////////////////////////////////
/// Partial specialization for Wmma Gemm Kernel
template <
    ///< Element type for A matrix operand
    typename ElementA,
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Element type for B matrix operand
    typename ElementB,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of A matrix in units of elements
    int kAlignmentB,
    /// Element type for C and D matrix operands
    typename ElementC,
    /// Layout type for C and D matrix operands
    typename LayoutC,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Warp-level tile size (concept: GemmShape)
    typename InstructionShape,
    MmaStrassen::Type StrassenKind,
    typename StrassenMiGroup,
    typename StrassenConsts,
    /// Epilogue output operator
    typename EpilogueOutputOp,
    typename InterimEpilogueOp,
    /// Threadblock-level swizzling operator
    typename ThreadblockSwizzle,
    /// Number of stages used in the pipelined mainloop
    int Stages,
    /// If true, kernel is configured to support serial reduction in the
    /// epilogue
    bool SplitKSerial,
    bool SubGemmParallel,
    /// Operation performed by GEMM
    typename Operator,
    /// Use zfill or predicate for out-of-bound cp.async
    SharedMemoryClearOption SharedMemoryClear
> 
struct DefaultStrassenGemm<
  ElementA, LayoutA, kAlignmentA, 
  ElementB, LayoutB, kAlignmentB, 
  ElementC, LayoutC, 
  ElementAccumulator, 
  arch::OpClassWmmaTensorOp,
  ArchTag, 
  ThreadblockShape, WarpShape, InstructionShape, StrassenKind, StrassenMiGroup, StrassenConsts,
  EpilogueOutputOp,
  InterimEpilogueOp,
  ThreadblockSwizzle, 
  Stages, 
  SplitKSerial, SubGemmParallel,
  Operator,
  SharedMemoryClear,
  false,
  false,
  false,
  layout::NoPermute,
  layout::NoPermute
> {
  /// Define the threadblock-scoped matrix multiply-accumulate
  using Mma = typename cutlass::gemm::threadblock::DefaultStrassenMma<
      ElementA, LayoutA, kAlignmentA,
      ElementB, LayoutB, kAlignmentB,
      ElementAccumulator, LayoutC, 
      arch::OpClassWmmaTensorOp, 
      ArchTag,
      ThreadblockShape, 
      WarpShape, 
      InstructionShape, 
      StrassenKind, StrassenMiGroup, StrassenConsts,
      Stages,
      Operator, InterimEpilogueOp>::ThreadblockMma;

  static const int kPartitionsK = ThreadblockShape::kK / WarpShape::kK;

  /// Define the epilogue 
  using Epilogue = typename cutlass::epilogue::threadblock::DefaultEpilogueWmmaTensorOp<
      ThreadblockShape,
      typename Mma::Operator, 
      kPartitionsK, 
      EpilogueOutputOp,
      EpilogueOutputOp::kCount
  >::Epilogue;

  using InterimEpilogue = typename cutlass::epilogue::threadblock::DefaultStrassenInterimEpilogueSimt<
      ThreadblockShape,
      typename Mma::StrassenShape,
      typename Mma::Operator,
      InterimEpilogueOp,
      InterimEpilogueOp::kCount,
      Mma::NumThreads
      >::InterimEpilogue;
  /// Define the kernel-level GEMM operator.
  using GemmKernel = kernel::StrassenGemm<StrassenMiGroup, Mma, Epilogue, InterimEpilogue, ThreadblockSwizzle, SplitKSerial, SubGemmParallel>;
};
////////////////////////////////////////////////////////////////////////////////

#endif //CUTLASS_ARCH_WMMA_ENABLED

////////////////////////////////////////////////////////////////////////////////

}  // namespace kernel
}  // namespace gemm
}  // namespace cutlass
