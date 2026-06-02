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
#include "cutlass/arch/wmma.h"

#include "cutlass/layout/matrix.h"
#include "cutlass/layout/permute.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator.h"
#include "cutlass/transform/threadblock/predicated_tile_iterator_2dthreadtile.h"

#include "cutlass/transform/threadblock/strassen_predicated_tile_access_iterator.h"
#include "cutlass/transform/threadblock/strassen_predicated_tile_iterator.h"

#include "cutlass/gemm/threadblock/strassen_mma_pipelined.h"
#include "cutlass/gemm/threadblock/strassen_mma_multistage.h"
#include "cutlass/gemm/threadblock/global_strassen_mma_pipelined.h"
#include "cutlass/gemm/threadblock/global_strassen_mma_multistage.h"
#include "cutlass/gemm/threadblock/global_strassen_matrix_mma_multistage.h"
#include "cutlass/gemm/threadblock/global_strassen_presum_mma_multistage.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/threadblock/default_strassen_mma_core_simt.h"
// #include "cutlass/gemm/threadblock/default_mma_core_sm70.h"
// #include "cutlass/gemm/threadblock/default_mma_core_sm75.h"
#include "cutlass/gemm/threadblock/default_strassen_mma_core_sm80.h"

// #if defined(CUTLASS_ARCH_WMMA_ENABLED)
// #include "cutlass/gemm/threadblock/default_mma_core_wmma.h"
// #endif //CUTLASS_ARCH_WMMA_ENABLED

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace threadblock {

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
    /// Element type for internal accumulation
    typename ElementAccumulator_,
    /// Layout type for C and D matrix operands
    typename LayoutC_,
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
    MmaStrassen::Type MmaStrassenKind,
    typename StrassenMiGroup,
    typename MmaStrassenConsts,
    /// Number of stages used in the pipelined mainloop
    int Stages,
    /// Operation perfomed by GEMM
    typename Operator,
    typename InterimEpilogueOp,
    /// Store the accumulators in row major or column major.  Row major is used
    /// when output layout is interleaved.
    bool AccumulatorsInRowMajor = false,
    /// Use zfill or predicate for out-of-bound cp.async
    SharedMemoryClearOption SharedMemoryClear = SharedMemoryClearOption::kNone,
    /// Gather operand A by using an index array
    bool GatherA = false,
    /// Gather operand B by using an index array
    bool GatherB = false,
    /// Permute operand A
    typename PermuteALayout = layout::NoPermute,
    /// Permute operand B
    typename PermuteBLayout = layout::NoPermute
    >
struct DefaultStrassenMma;

////////////////////////////////////////////////////////////////////////////////

/// Specialization for row-major output (OperatorClass Simt)
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
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Layout type for C and D matrix operand
    typename LayoutC,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape,
    MmaStrassen::Type MmaStrassenKind,
    typename StrassenMiGroup,
    typename MmaStrassenConsts,
    /// Operation performed by GEMM
    typename Operator,
    typename InterimEpilogueOp,
    /// Gather operand A by using an index array
    bool GatherA,
    /// Gather operand B by using an index array
    bool GatherB,
    /// Permute operand A
    typename PermuteALayout,
    /// Permute operand B
    typename PermuteBLayout
    >
struct DefaultStrassenMma<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB,
                  kAlignmentB, ElementAccumulator, LayoutC,
                  arch::OpClassSimt, ArchTag, ThreadblockShape, WarpShape,
                  InstructionShape, MmaStrassenKind, StrassenMiGroup, MmaStrassenConsts, 2, Operator, 
                  InterimEpilogueOp, false, SharedMemoryClearOption::kNone,
                  GatherA, GatherB, PermuteALayout, PermuteBLayout> {

  static_assert(platform::is_same<LayoutC, layout::RowMajor>::value
             || platform::is_same<LayoutC, layout::AffineRankN<2>>::value,
             "simt epilogue must be row major");
  /// Number of warps present
  using StrassenShape = GemmShape<ThreadblockShape::kM/StrassenMiGroup::TileMDivisor(),
                                  ThreadblockShape::kN/StrassenMiGroup::TileNDivisor(),
                                  ThreadblockShape::kK>;
  using StrassenWarpShape = GemmShape<WarpShape::kM/StrassenMiGroup::TileMDivisor(),
                                      WarpShape::kN/StrassenMiGroup::TileNDivisor(),
                                      WarpShape::kK>;

  //Using this specialization
  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultStrassenMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, StrassenShape, StrassenWarpShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementAccumulator, LayoutC,
      arch::OpClassSimt, 2, Operator>;

  // Define iterators over tiles from the A operand
  using IteratorA =
      cutlass::transform::threadblock::StrassenPredicatedTileIterator<
          cutlass::MatrixShape<StrassenShape::kM, StrassenShape::kK>,
          ElementA, LayoutA, 1, typename MmaCore::IteratorThreadMapA, kAlignmentA,
          GatherA, PermuteALayout>;

  // Define iterators over tiles from the B operand
  using IteratorB =
      cutlass::transform::threadblock::StrassenPredicatedTileIterator<
          cutlass::MatrixShape<StrassenShape::kK, StrassenShape::kN>,
          ElementB, LayoutB, 0, typename MmaCore::IteratorThreadMapB, kAlignmentB,
          GatherB, PermuteBLayout>;

  // Define the threadblock-scoped pipelined matrix multiply
  using ThreadblockMma1 = cutlass::gemm::threadblock::StrassenMmaPipeline<
      typename MmaCore::Shape, StrassenShape, IteratorA, typename MmaCore::SmemIteratorA,
      IteratorB, typename MmaCore::SmemIteratorB, ElementAccumulator,
      LayoutC, typename MmaCore::MmaPolicy, MmaStrassenKind, StrassenMiGroup>;
  
  using ThreadblockMma2 = cutlass::gemm::threadblock::GlobalStrassenMmaPipeline<
      typename MmaCore::Shape, StrassenShape, IteratorA, typename MmaCore::SmemIteratorA,
      IteratorB, typename MmaCore::SmemIteratorB, ElementAccumulator,
      LayoutC, typename MmaCore::MmaPolicy, MmaStrassenKind, StrassenMiGroup, MmaStrassenConsts>;

  using ThreadblockMma = typename platform::conditional<MmaStrassenKind == MmaStrassen::Type::StrassenWinograd,
                                                        ThreadblockMma1,
                                                        ThreadblockMma2>::type;
};

////////////////////////////////////////////////////////////////////////////////

/// Specialization for row-major output (OperatorClass TensorOp)
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
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape,
    MmaStrassen::Type MmaStrassenKind,
    typename StrassenMiGroup,
    typename MmaStrassenConsts,
    /// Operation performed by GEMM
    typename Operator,
    typename InterimEpilogueOp,
    /// Use zfill or predicate for out-of-bound cp.async
    SharedMemoryClearOption SharedMemoryClear,
    /// Gather operand A by using an index array
    bool GatherA,
    /// Gather operand B by using an index array
    bool GatherB,
    /// Permute operand A
    typename PermuteALayout,
    /// Permute operand B
    typename PermuteBLayout
    >
struct DefaultStrassenMma<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB,
                  kAlignmentB, ElementAccumulator, layout::RowMajor,
                  arch::OpClassTensorOp, ArchTag, ThreadblockShape, WarpShape,
                  InstructionShape, MmaStrassenKind,  StrassenMiGroup, MmaStrassenConsts, 2, Operator,
                  InterimEpilogueOp, false, SharedMemoryClear,
                  GatherA, GatherB, PermuteALayout, PermuteBLayout> {
  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementAccumulator, layout::RowMajor,
      arch::OpClassTensorOp, 2, Operator>;

  // Define iterators over tiles from the A operand
  using IteratorA =
      cutlass::transform::threadblock::PredicatedTileIterator<
          cutlass::MatrixShape<MmaCore::Shape::kM, MmaCore::Shape::kK>,
          ElementA, LayoutA, 1, typename MmaCore::IteratorThreadMapA, kAlignmentA,
          GatherA, PermuteALayout>;

  // Define iterators over tiles from the B operand
  using IteratorB =
      cutlass::transform::threadblock::PredicatedTileIterator<
          cutlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kN>,
          ElementB, LayoutB, 0, typename MmaCore::IteratorThreadMapB, kAlignmentB,
          GatherB, PermuteBLayout>;

  // Define the threadblock-scoped pipelined matrix multiply
  using ThreadblockMma = cutlass::gemm::threadblock::MmaPipelined<
      typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
      IteratorB, typename MmaCore::SmemIteratorB, ElementAccumulator,
      layout::RowMajor, typename MmaCore::MmaPolicy>;
};

////////////////////////////////////////////////////////////////////////////////
/// Specialization for row-major output (OperatorClass TensorOp)
template <
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape,
    MmaStrassen::Type MmaStrassenKind,
    typename StrassenMiGroup,
    typename MmaStrassenConsts,
    /// Operation performed by GEMM
    typename Operator,
    typename InterimEpilogueOp,
    /// Gather operand A by using an index array
    bool GatherA,
    /// Gather operand B by using an index array
    bool GatherB,
    /// Permute operand A
    typename PermuteALayout,
    /// Permute operand B
    typename PermuteBLayout
    >
struct DefaultStrassenMma<float, LayoutA, kAlignmentA, float, LayoutB,
                  kAlignmentB, float, layout::RowMajor,
                  arch::OpClassTensorOp, ArchTag, ThreadblockShape, WarpShape,
                  InstructionShape, MmaStrassenKind,  StrassenMiGroup, MmaStrassenConsts, 2, Operator, 
                  InterimEpilogueOp, false, SharedMemoryClearOption::kNone,
                  GatherA, GatherB, PermuteALayout, PermuteBLayout> {
  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, float, LayoutA, float,
      LayoutB, float, layout::RowMajor, arch::OpClassTensorOp, 2,
      arch::OpMultiplyAddFastF16>;

  // Define iterators over tiles from the A operand
  using IteratorA =
      cutlass::transform::threadblock::PredicatedTileIterator<
          cutlass::MatrixShape<MmaCore::Shape::kM, MmaCore::Shape::kK>,
          float, LayoutA, 1, typename MmaCore::IteratorThreadMapA, kAlignmentA,
          GatherA, PermuteALayout>;

  // Define iterators over tiles from the B operand
  using IteratorB =
      cutlass::transform::threadblock::PredicatedTileIterator<
          cutlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kN>,
          float, LayoutB, 0, typename MmaCore::IteratorThreadMapB, kAlignmentB,
          GatherB, PermuteBLayout>;

  // Define the threadblock-scoped pipelined matrix multiply
  using ThreadblockMma = cutlass::gemm::threadblock::MmaPipelined<
      typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
      IteratorB, typename MmaCore::SmemIteratorB, float,
      layout::RowMajor, typename MmaCore::MmaPolicy>;
};

////////////////////////////////////////////////////////////////////////////////

/// Specialization for column-major-interleaved output
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
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Tag indicating architecture to tune for
    typename OperatorClass,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape,
    MmaStrassen::Type MmaStrassenKind,
    typename StrassenMiGroup,
    typename MmaStrassenConsts,
    /// Operation performed by GEMM
    typename Operator,
    typename InterimEpilogueOp,
    /// Number of Interleaved K
    int InterleavedK>
struct DefaultStrassenMma<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB,
                  kAlignmentB, ElementAccumulator,
                  layout::ColumnMajorInterleaved<InterleavedK>, OperatorClass,
                  ArchTag, ThreadblockShape, WarpShape, InstructionShape, MmaStrassenKind, StrassenMiGroup, MmaStrassenConsts, 2,
                  Operator, InterimEpilogueOp, true, SharedMemoryClearOption::kNone, false, false,
                  layout::NoPermute, layout::NoPermute> {
  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementAccumulator,
      layout::ColumnMajorInterleaved<InterleavedK>, OperatorClass, 2, Operator,
      true>;

  static_assert(kAlignmentA == 128 / sizeof_bits<ElementA>::value, 
    "Alignment must match thread data map's vector length");

  static_assert(kAlignmentB ==128 / sizeof_bits<ElementB>::value,
    "Alignment must match thread data map's vector length");

  // Define iterators over tiles from the A operand
  using IteratorA = cutlass::transform::threadblock::PredicatedTileIterator<
      cutlass::MatrixShape<MmaCore::Shape::kM, MmaCore::Shape::kK>, ElementA,
      LayoutA, 1, typename MmaCore::IteratorThreadMapA>;

  // Define iterators over tiles from the B operand
  using IteratorB = cutlass::transform::threadblock::PredicatedTileIterator<
      cutlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kN>, ElementB,
      LayoutB, 0, typename MmaCore::IteratorThreadMapB>;

  // Define the threadblock-scoped pipelined matrix multiply
  using ThreadblockMma = cutlass::gemm::threadblock::MmaPipelined<
      typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
      IteratorB, typename MmaCore::SmemIteratorB, ElementAccumulator,
      layout::ColumnMajorInterleaved<InterleavedK>,
      typename MmaCore::MmaPolicy>;
};

////////////////////////////////////////////////////////////////////////////////

/// Specialization for row-major output
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
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Layout type for C and D matrix operand
    typename LayoutC,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape,
    MmaStrassen::Type MmaStrassenKind,
    typename StrassenMiGroup,
    typename MmaStrassenConsts,
    /// Number of stages used in the multistage mainloop
    int Stages,
    /// Operation perfomed by GEMM
    typename Operator,
    typename InterimEpilogueOp,
    /// Gather operand A by using an index array
    bool GatherA,
    /// Gather operand B by using an index array
    bool GatherB,
    /// Permute operand A
    typename PermuteALayout,
    /// Permute operand B
    typename PermuteBLayout
    >
struct DefaultStrassenMma<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB,
                  kAlignmentB, ElementAccumulator, LayoutC,
                  arch::OpClassSimt, ArchTag, ThreadblockShape, WarpShape,
                  InstructionShape, MmaStrassenKind,  StrassenMiGroup, MmaStrassenConsts, Stages, Operator,
                  InterimEpilogueOp, false, SharedMemoryClearOption::kNone,
                  GatherA, GatherB, PermuteALayout, PermuteBLayout> {

  static_assert(platform::is_same<LayoutC, layout::RowMajor>::value
             || platform::is_same<LayoutC, layout::AffineRankN<2>>::value,
             "simt epilogue must be row major");

  using StrassenShape = GemmShape<ThreadblockShape::kM/StrassenMiGroup::TileMDivisor(),
                                  ThreadblockShape::kN/StrassenMiGroup::TileNDivisor(),
                                  ThreadblockShape::kK>;
  using StrassenWarpShape = GemmShape<WarpShape::kM/StrassenMiGroup::TileMDivisor(),
                                      WarpShape::kN/StrassenMiGroup::TileNDivisor(),
                                      WarpShape::kK>;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultStrassenMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, StrassenShape, StrassenWarpShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementAccumulator, LayoutC, arch::OpClassSimt,
      Stages, Operator>;

  // Define iterators over tiles from the A operand
  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using AccessTypeA = cutlass::Array<ElementA, kAlignmentA>;
  using IteratorA =
      cutlass::transform::threadblock::StrassenPredicatedTileAccessIterator<
          cutlass::MatrixShape<StrassenShape::kM, StrassenShape::kK>,
          ElementA, LayoutA, 1, ThreadMapA, AccessTypeA, GatherA, PermuteALayout>;

  // Define iterators over tiles from the B operand
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using AccessTypeB = cutlass::Array<ElementB, kAlignmentB>;
  using IteratorB =
      cutlass::transform::threadblock::StrassenPredicatedTileAccessIterator<
          cutlass::MatrixShape<StrassenShape::kK, StrassenShape::kN>,
          ElementB, LayoutB, 0, ThreadMapB, AccessTypeB, GatherB, PermuteBLayout>;

  // Define the threadblock-scoped multistage matrix multiply
  using ThreadblockMma1 = cutlass::gemm::threadblock::StrassenMmaMultistage<
      typename MmaCore::Shape, StrassenShape, IteratorA, typename MmaCore::SmemIteratorA,
      MmaCore::kCacheOpA, IteratorB, typename MmaCore::SmemIteratorB, 
      MmaCore::kCacheOpB, ElementAccumulator,
      LayoutC, typename MmaCore::MmaPolicy, MmaStrassenKind, StrassenMiGroup, Stages,
      InterimEpilogueOp>;

  // Define the threadblock-scoped multistage matrix multiply
  using ThreadblockMma2 = cutlass::gemm::threadblock::GlobalStrassenMmaMultistage<
      typename MmaCore::Shape, StrassenShape, IteratorA, typename MmaCore::SmemIteratorA,
      MmaCore::kCacheOpA, IteratorB, typename MmaCore::SmemIteratorB,
      MmaCore::kCacheOpB, ElementAccumulator, LayoutC,
      typename MmaCore::MmaPolicy, MmaStrassenKind, StrassenMiGroup, Stages>;

  using ThreadblockMma = typename platform::conditional<MmaStrassenKind == MmaStrassen::Type::StrassenWinograd,
                                                        ThreadblockMma1,
                                                        ThreadblockMma2>::type;
};

////////////////////////////////////////////////////////////////////////////////

/// Specialization for row-major output (OperatorClass TensorOp)
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
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Layout type for C and D matrix operand
    typename LayoutC,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape,
    MmaStrassen::Type MmaStrassenKind,
    typename StrassenMiGroup,
    typename MmaStrassenConsts,
    /// Number of stages used in the multistage mainloop
    int Stages,
    /// Operation perfomed by GEMM
    typename Operator,
    typename InterimEpilogueOp,
    /// Use zfill or predicate for out-of-bound cp.async
    SharedMemoryClearOption SharedMemoryClear,
    /// Gather operand A by using an index array
    bool GatherA,
    /// Gather operand B by using an index array
    bool GatherB,
    /// Permute operand A
    typename PermuteALayout,
    /// Permute operand B
    typename PermuteBLayout
    >
struct DefaultStrassenMma<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB,
                  kAlignmentB, ElementAccumulator, LayoutC,
                  arch::OpClassTensorOp, ArchTag, ThreadblockShape, WarpShape,
                  InstructionShape, MmaStrassenKind, StrassenMiGroup, MmaStrassenConsts, Stages, Operator, 
                  InterimEpilogueOp, false, SharedMemoryClear,
                  GatherA, GatherB, PermuteALayout, PermuteBLayout> {

  static_assert(platform::is_same<LayoutC, layout::RowMajor>::value
             || platform::is_same<LayoutC, layout::AffineRankN<2>>::value,
             "simt epilogue must be row major");
  // using this specialization for half f16
  using StrassenShape =  typename platform::conditional<MmaStrassenKind == MmaStrassen::Normal || MmaStrassenKind == MmaStrassen::Compressed,
                                               GemmShape<ThreadblockShape::kM/2, ThreadblockShape::kN/2, ThreadblockShape::kK>,
                                               ThreadblockShape>::type;
  using StrassenWarpShape = typename platform::conditional<MmaStrassenKind == MmaStrassen::Normal || MmaStrassenKind == MmaStrassen::Compressed,
                                                  GemmShape<WarpShape::kM/2, WarpShape::kN/2, WarpShape::kK>,
                                                  WarpShape>::type;

  static cutlass::arch::CacheOperation::Kind const CacheOpA =
      ((sizeof_bits<ElementA>::value * kAlignmentA) == 128)
          ? cutlass::arch::CacheOperation::Global
          : cutlass::arch::CacheOperation::Always;

  static cutlass::arch::CacheOperation::Kind const CacheOpB =
      ((sizeof_bits<ElementB>::value * kAlignmentB) == 128)
          ? cutlass::arch::CacheOperation::Global
          : cutlass::arch::CacheOperation::Always;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultStrassenMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, StrassenShape, StrassenWarpShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementAccumulator, LayoutC, arch::OpClassTensorOp,
      Stages, Operator, false, CacheOpA, CacheOpB>;

  // Define iterators over tiles from the A operand
  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using AccessTypeA = cutlass::Array<ElementA, kAlignmentA>;
  using IteratorA =
      cutlass::transform::threadblock::StrassenPredicatedTileAccessIterator<
          cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
          ElementA, LayoutA, 1, ThreadMapA, AccessTypeA, GatherA, PermuteALayout>;

  // Define iterators over tiles from the B operand
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using AccessTypeB = cutlass::Array<ElementB, kAlignmentB>;
  using IteratorB =
      cutlass::transform::threadblock::StrassenPredicatedTileAccessIterator<
          cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
          ElementB, LayoutB, 0, ThreadMapB, AccessTypeB, GatherB, PermuteBLayout>;
  
  using ThreadblockMma1 = cutlass::gemm::threadblock::StrassenMmaMultistage<
      typename MmaCore::Shape, StrassenShape, IteratorA, typename MmaCore::SmemIteratorA,
      MmaCore::kCacheOpA, IteratorB, typename MmaCore::SmemIteratorB, 
      MmaCore::kCacheOpB, ElementAccumulator,
      LayoutC, typename MmaCore::MmaPolicy, MmaStrassenKind, StrassenMiGroup, Stages,
      InterimEpilogueOp>;
  using ThreadblockMma = ThreadblockMma1;

  // Define the threadblock-scoped multistage matrix multiply
//   using ThreadblockMma2 = cutlass::gemm::threadblock::GlobalStrassenMmaMultistage<
//       typename MmaCore::Shape, StrassenShape, IteratorA, typename MmaCore::SmemIteratorA,
//       MmaCore::kCacheOpA, IteratorB, typename MmaCore::SmemIteratorB,
//       MmaCore::kCacheOpB, ElementAccumulator, LayoutC,
//       typename MmaCore::MmaPolicy, MmaStrassenKind, StrassenMiGroup, Stages, SharedMemoryClear>;

//   using ThreadblockMma3 = cutlass::gemm::threadblock::GlobalStrassenMatrixMmaMultistage<
//       typename MmaCore::Shape, StrassenShape, IteratorA, typename MmaCore::SmemIteratorA,
//       MmaCore::kCacheOpA, IteratorB, typename MmaCore::SmemIteratorB,
//       MmaCore::kCacheOpB, ElementAccumulator, LayoutC,
//       typename MmaCore::MmaPolicy, MmaStrassenKind, MmaStrassenConsts, Stages, SharedMemoryClear>;

//   using ThreadblockMma4 = cutlass::gemm::threadblock::GlobalStrassenPresumMmaMultistage<
//       typename MmaCore::Shape, StrassenShape, IteratorA, typename MmaCore::SmemIteratorA,
//       MmaCore::kCacheOpA, IteratorB, typename MmaCore::SmemIteratorB,
//       MmaCore::kCacheOpB, ElementAccumulator, LayoutC,
//       typename MmaCore::MmaPolicy, MmaStrassenKind, MmaStrassenConsts, Stages, SharedMemoryClear>;

//   static bool const IsGlobalStrassenMma = MmaStrassenKind == MmaStrassen::Type::GlobalLevel1_M0 || 
//                                           MmaStrassenKind == MmaStrassen::Type::GlobalLevel1_M1 ||
//                                           MmaStrassenKind == MmaStrassen::Type::GlobalLevel1_M2 ||
//                                           MmaStrassenKind == MmaStrassen::Type::GlobalLevel1_M3 ||
//                                           MmaStrassenKind == MmaStrassen::Type::GlobalLevel1_M4 ||
//                                           MmaStrassenKind == MmaStrassen::Type::GlobalLevel1_M5 ||
//                                           MmaStrassenKind == MmaStrassen::Type::GlobalLevel1_M6;

//   static bool const IsGlobalPreSumStrassenMma = MmaStrassenKind == MmaStrassen::Type::GlobalPreSumLevel1_M0 || 
//                                                 MmaStrassenKind == MmaStrassen::Type::GlobalPreSumLevel1_M1 ||
//                                                 MmaStrassenKind == MmaStrassen::Type::GlobalPreSumLevel1_M2 ||
//                                                 MmaStrassenKind == MmaStrassen::Type::GlobalPreSumLevel1_M3 ||
//                                                 MmaStrassenKind == MmaStrassen::Type::GlobalPreSumLevel1_M4 ||
//                                                 MmaStrassenKind == MmaStrassen::Type::GlobalPreSumLevel1_M5 ||
//                                                 MmaStrassenKind == MmaStrassen::Type::GlobalPreSumLevel1_M6;

//   using ThreadblockMma = typename platform::conditional<MmaStrassenKind == MmaStrassen::Type::Normal || MmaStrassenKind == MmaStrassen::Type::Compressed, ThreadblockMma1,
//                                                         typename platform::conditional<IsGlobalStrassenMma, ThreadblockMma2,
//                                                                                        typename platform::conditional<IsGlobalPreSumStrassenMma, ThreadblockMma4, 
//                                                                                        ThreadblockMma3>::type>::type>::type;
};

////////////////////////////////////////////////////////////////////////////////

/// Specialization for column-major-interleaved output
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
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Tag indicating architecture to tune for
    typename OperatorClass,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape,
    MmaStrassen::Type MmaStrassenKind, 
    typename StrassenMiGroup,
    typename MmaStrassenConsts,
    /// Number of stages used in the multistage mainloop
    int Stages,
    /// Operation performed by GEMM
    typename Operator,
    typename InterimEpilogueOp,
    /// Number of Interleaved K
    int InterleavedK>
struct DefaultStrassenMma<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB,
                  kAlignmentB, ElementAccumulator,
                  layout::ColumnMajorInterleaved<InterleavedK>, OperatorClass,
                  ArchTag, ThreadblockShape, WarpShape, InstructionShape,
                  MmaStrassenKind, StrassenMiGroup, MmaStrassenConsts, Stages, Operator,
                  InterimEpilogueOp, true, SharedMemoryClearOption::kNone,
                  false, false, layout::NoPermute, layout::NoPermute> {
  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementAccumulator,
      layout::ColumnMajorInterleaved<InterleavedK>, OperatorClass, Stages,
      Operator, true>;

  // Define iterators over tiles from the A operand
  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using AccessTypeA = cutlass::Array<ElementA, kAlignmentA>;
  using IteratorA =
      cutlass::transform::threadblock::PredicatedTileAccessIterator<
          cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
          ElementA, LayoutA, 1, ThreadMapA, AccessTypeA>;

  // Define iterators over tiles from the B operand
  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using AccessTypeB = cutlass::Array<ElementB, kAlignmentB>;
  using IteratorB =
      cutlass::transform::threadblock::PredicatedTileAccessIterator<
          cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
          ElementB, LayoutB, 0, ThreadMapB, AccessTypeB>;

  // Define the threadblock-scoped multistage matrix multiply
  using ThreadblockMma = cutlass::gemm::threadblock::MmaMultistage<
      typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
      MmaCore::kCacheOpA, IteratorB, typename MmaCore::SmemIteratorB,
      MmaCore::kCacheOpB, ElementAccumulator, layout::RowMajor,
      typename MmaCore::MmaPolicy, Stages>;
};

////////////////////////////////////////////////////////////////////////////////

/// Specialization for SIMT IDP4A Kernels
template <
    /// Layout type for A matrix operand
    typename LayoutA,
    /// Access granularity of A matrix in units of elements
    int kAlignmentA,
    /// Layout type for B matrix operand
    typename LayoutB,
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    MmaStrassen::Type MmaStrassenKind,  
    typename StrassenMiGroup,
    typename MmaStrassenConsts,
    /// Operation performed by GEMM
    typename Operator,
    typename InterimEpilogueOp,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape>
struct DefaultStrassenMma<int8_t, LayoutA, kAlignmentA, int8_t, LayoutB, kAlignmentB,
                  ElementAccumulator, layout::RowMajor, arch::OpClassSimt,
                  ArchTag, ThreadblockShape, WarpShape, GemmShape<1, 1, 4>, MmaStrassenKind, StrassenMiGroup, MmaStrassenConsts, 2,
                  Operator, InterimEpilogueOp, false, SharedMemoryClearOption::kNone,
                  false, false, layout::NoPermute, layout::NoPermute> {
  using InstructionShape = GemmShape<1, 1, 4>;
  using ElementA = int8_t;
  using ElementB = int8_t;
  using OperatorClass =  arch::OpClassSimt;

  static const bool transposeA = platform::is_same< LayoutA, layout::ColumnMajor >::value;
  static const bool transposeB = platform::is_same< LayoutB, layout::RowMajor >::value;

  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementAccumulator, layout::RowMajor,
      OperatorClass, 2, Operator>;

  // Define iterators over tiles from the A operand
  using IteratorA =
      cutlass::transform::threadblock::PredicatedTileIterator2dThreadTile<
          cutlass::MatrixShape<MmaCore::Shape::kM, MmaCore::Shape::kK>,
          ElementA, LayoutA, 1, typename MmaCore::IteratorThreadMapA, transposeA>;

  // Define iterators over tiles from the B operand
  using IteratorB =
      cutlass::transform::threadblock::PredicatedTileIterator2dThreadTile<
          cutlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kN>,
          ElementB, LayoutB, 0, typename MmaCore::IteratorThreadMapB, transposeB>;

  // Define the threadblock-scoped pipelined matrix multiply
  using ThreadblockMma = cutlass::gemm::threadblock::MmaPipelined<
      typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
      IteratorB, typename MmaCore::SmemIteratorB, ElementAccumulator,
      layout::RowMajor, typename MmaCore::MmaPolicy>;
};

////////////////////////////////////////////////////////////////////////////////

#if defined(CUTLASS_ARCH_WMMA_ENABLED)
/// Specialization for Wmma TensorOp operator with 2 staged pipeline
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
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Layout type for C and D matrix operands
    typename LayoutC,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape,
    MmaStrassen::Type MmaStrassenKind,
    typename StrassenMiGroup,
    typename MmaStrassenConsts,
    /// Operation performed by GEMM
    typename Operator,
    typename InterimEpilogueOp>
struct DefaultStrassenMma<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB,
                  kAlignmentB, ElementAccumulator, LayoutC,
                  arch::OpClassWmmaTensorOp, ArchTag, ThreadblockShape, WarpShape,
                  InstructionShape, MmaStrassenKind, StrassenMiGroup, MmaStrassenConsts, 2, Operator,
                  InterimEpilogueOp, false, SharedMemoryClearOption::kNone,
                  false, false, layout::NoPermute, layout::NoPermute> {
  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementAccumulator, LayoutC,
      arch::OpClassWmmaTensorOp, 2, Operator>;

  // Define iterators over tiles from the A operand
  using IteratorA =
      cutlass::transform::threadblock::PredicatedTileIterator<
          cutlass::MatrixShape<MmaCore::Shape::kM, MmaCore::Shape::kK>,
          ElementA, LayoutA, 1, typename MmaCore::IteratorThreadMapA, kAlignmentA>;

  // Define iterators over tiles from the B operand
  using IteratorB =
      cutlass::transform::threadblock::PredicatedTileIterator<
          cutlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kN>,
          ElementB, LayoutB, 0, typename MmaCore::IteratorThreadMapB, kAlignmentB>;

  // Define the threadblock-scoped pipelined matrix multiply
  using ThreadblockMma = cutlass::gemm::threadblock::MmaPipelined<
      typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
      IteratorB, typename MmaCore::SmemIteratorB, ElementAccumulator,
      LayoutC, typename MmaCore::MmaPolicy>;
};

////////////////////////////////////////////////////////////////////////////////

/// Specialization for Wmma TensorOp operator with 1 staged pipeline
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
    /// Access granularity of B matrix in units of elements
    int kAlignmentB,
    /// Element type for internal accumulation
    typename ElementAccumulator,
    /// Layout type for C and D matrix operands
    typename LayoutC,
    /// Tag indicating architecture to tune for
    typename ArchTag,
    /// Threadblock-level tile size (concept: GemmShape)
    typename ThreadblockShape,
    /// Warp-level tile size (concept: GemmShape)
    typename WarpShape,
    /// Instruction-level tile size (concept: GemmShape)
    typename InstructionShape,
    MmaStrassen::Type MmaStrassenKind,
    typename StrassenMiGroup,
    typename MmaStrassenConsts,
    /// Operation performed by GEMM
    typename Operator,
    typename InterimEpilogueOp>
struct DefaultStrassenMma<ElementA, LayoutA, kAlignmentA, ElementB, LayoutB,
                  kAlignmentB, ElementAccumulator, LayoutC,
                  arch::OpClassWmmaTensorOp, ArchTag, ThreadblockShape, WarpShape,
                  InstructionShape, MmaStrassenKind, StrassenMiGroup, MmaStrassenConsts, 1, Operator,
                  InterimEpilogueOp, false, SharedMemoryClearOption::kNone,
                  false, false, layout::NoPermute, layout::NoPermute> {
  // Define the MmaCore components
  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementAccumulator, LayoutC,
      arch::OpClassWmmaTensorOp, 1, Operator>; 

  // Define iterators over tiles from the A operand
  using IteratorA =
      cutlass::transform::threadblock::PredicatedTileIterator<
          cutlass::MatrixShape<MmaCore::Shape::kM, MmaCore::Shape::kK>,
          ElementA, LayoutA, 1, typename MmaCore::IteratorThreadMapA, kAlignmentA>;

  // Define iterators over tiles from the B operand
  using IteratorB =
      cutlass::transform::threadblock::PredicatedTileIterator<
          cutlass::MatrixShape<MmaCore::Shape::kK, MmaCore::Shape::kN>,
          ElementB, LayoutB, 0, typename MmaCore::IteratorThreadMapB, kAlignmentB>;

  // Define the threadblock-scoped singlestage matrix multiply
  using ThreadblockMma = cutlass::gemm::threadblock::MmaSingleStage<
      typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
      IteratorB, typename MmaCore::SmemIteratorB, ElementAccumulator,
      LayoutC, typename MmaCore::MmaPolicy>;
};

////////////////////////////////////////////////////////////////////////////////
#endif //CUTLASS_ARCH_WMMA_ENABLED

} // namespace threadblock
} // namespace gemm
} // namespace cutlass 

////////////////////////////////////////////////////////////////////////////////
