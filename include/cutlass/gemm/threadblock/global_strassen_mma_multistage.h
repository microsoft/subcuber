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
    \brief Template for a double-buffered threadblock-scoped GEMM kernel.
*/

#pragma once


#include "cutlass/aligned_buffer.h"
#include "cutlass/arch/memory.h"
#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_types.h"
#include "cutlass/coord.h"

#include "cutlass/gemm/threadblock/mma_base.h"
#include "cutlass/gemm/threadblock/strassen_mma_multistage.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Structure to compute the matrix product targeting CUDA cores and SIMT math
/// instructions.
template <
    /// Size of the Gemm problem - concept: gemm::GemmShape<>
    typename Shape_,
    typename StrassenShape_,
    /// Iterates over tiles of A operand in global memory
    //  (concept: ReadableTileIterator | ForwardTileIterator |
    //  MaskedTileIterator)
    typename IteratorA_,
    /// Iterates over tiles of A operand in shared memory
    /// (concept: WriteableTileIterator | RandomAccessTileIterator)
    typename SmemIteratorA_,
    /// Cache operation for operand A
    cutlass::arch::CacheOperation::Kind CacheOpA,
    /// Iterates over tiles of B operand in global memory
    //  (concept: ReadableTileIterator | ForwardTileIterator |
    //  MaskedTileIterator)
    typename IteratorB_,
    /// Iterates over tiles of B operand in shared memory
    /// (concept: WriteableTileIterator | RandomAccessTileIterator)
    typename SmemIteratorB_,
    /// Cache operation for operand B
    cutlass::arch::CacheOperation::Kind CacheOpB,
    /// Data type of accumulator matrix
    typename ElementC_,
    /// Data type of accumulator matrix
    typename LayoutC_,
    /// Policy describing tuning details (concept: MmaPolicy)
    typename Policy_,
    MmaStrassen::Type MmaStrassenKind,
    typename StrassenMiGroup_,
    /// Number of stages,
    int Stages,
    /// Use zfill or predicate for out-of-bound cp.async
    SharedMemoryClearOption SharedMemoryClear = SharedMemoryClearOption::kNone,
    /// Used for partial specialization
    typename Enable = bool>
class GlobalStrassenMmaMultistage : 
  public GlobalStrassenMmaBase<Shape_, StrassenShape_, Policy_, Stages, MmaStrassenKind, StrassenMiGroup_> {
public:
  ///< Base class
  using Base = GlobalStrassenMmaBase<Shape_, StrassenShape_, Policy_, Stages, MmaStrassenKind, StrassenMiGroup_>;
  using StrassenMiGroup = StrassenMiGroup_;
  ///< Size of the Gemm problem - concept: gemm::GemmShape<>
  using Shape = Shape_;
  ///< Iterates over tiles of A operand in global memory
  using IteratorA = IteratorA_;
  ///< Iterates over tiles of B operand in global memory
  using IteratorB = IteratorB_;
  ///< Data type of accumulator matrix
  using ElementC = ElementC_;
  ///< Layout of accumulator matrix
  using LayoutC = LayoutC_;
  ///< Policy describing tuning details
  using Policy = Policy_;

  using SmemIteratorA = SmemIteratorA_;
  using SmemIteratorB = SmemIteratorB_;

  static cutlass::arch::CacheOperation::Kind const kCacheOpA = CacheOpA;
  static cutlass::arch::CacheOperation::Kind const kCacheOpB = CacheOpB;

  //
  // Dependent types
  //

  /// Fragment of accumulator tile
  using FragmentC = typename Policy::Operator::FragmentC;

  /// Warp-level Mma
  using Operator = typename Policy::Operator;

  /// Minimum architecture is Sm80 to support cp.async
  using ArchTag = arch::Sm80;

  /// Complex transform on A operand
  static ComplexTransform const kTransformA = Operator::kTransformA;

  /// Complex transform on B operand
  static ComplexTransform const kTransformB = Operator::kTransformB;

  /// Internal structure exposed for introspection.
  struct Detail {

    /// Number of cp.async instructions to load one stage of operand A
    static int const AsyncCopyIterationsPerStageA =
        IteratorA::ThreadMap::Iterations::kCount;

    /// Number of cp.async instructions to load one stage of operand B
    static int const AsyncCopyIterationsPerStageB =
        IteratorB::ThreadMap::Iterations::kCount;

    /// Number of stages
    static int const kStages = Stages;

    /// Number of cp.async instructions to load on group of operand A
    static int const kAccessesPerGroupA =
        (AsyncCopyIterationsPerStageA + Base::kWarpGemmIterations - 1) / Base::kWarpGemmIterations;

    /// Number of cp.async instructions to load on group of operand B
    static int const kAccessesPerGroupB =
        (AsyncCopyIterationsPerStageB + Base::kWarpGemmIterations - 1) / Base::kWarpGemmIterations;

    // Optional staged-accumulation (e.g., tf32x3 kernels) for improved numerical
    // accuracy, where each mainloop iteration first accumulates into a temporary
    // set of freshly-cleared accumulators, which are subsequently added to the
    // final accumulator set.
    static bool const kStagedAccumulation = arch::detail::UseStagedAccumulation<Operator>::value;
  };
  static const int NumThreads = Base::WarpCount::kCount * 32;
  using PresumShape = GemmShape<Shape::kM, Shape::kN, 1>;
  static const int PresumAccessPerIter = 1;//TODO: change for half
  using PresumGlobalIteratorA = PresumDetail::GlobalIterator<typename IteratorA::Element, NumThreads,  
                                                            PresumShape, Array<float, 4>, false, false>;
  using PresumGlobalIteratorB =PresumGlobalIteratorA;
  using PresumSharedIterator = PresumDetail::SharedIterator<typename IteratorA::Element, Array<float, 4>, NumThreads,
                                                            1,
                                                            1,
                                                            Stages, PresumAccessPerIter>;
  static const uint kPresumComputeIterations = 0;
  using PresumShapeA = GemmShape<Shape::kM, Shape::kN, 1>;
  using PresumShapeB = GemmShape<Shape::kM, Shape::kN, 1>;
  using PresumVecType = Array<typename IteratorA::Element, 16/sizeof(typename IteratorA::Element)>;
  using PresumComputeType = ElementC;
  using PresumComputeVecType = Array<PresumComputeType, 16/sizeof(PresumComputeType)>;
  using PresumComputeToIOType = NumericArrayConverter<typename IteratorA::Element, PresumComputeType, PresumVecType::kElements>;
  using PresumIOToComputeType = NumericArrayConverter<PresumComputeType, typename IteratorA::Element, PresumVecType::kElements>;
  using SharedPostsumOpIterator = cutlass::epilogue::threadblock::StrassenInterimEpilogueSharedTileIterator<
    StrassenShape_,
    float,
    NumThreads,
    4
  >;

  using PostsumSrcTileIterator = cutlass::epilogue::threadblock::StrassenInterimEpilogueTileIterator<
    StrassenShape_,
    float,
    float,
    NumThreads,
    4
  >;

  PresumSharedIterator sharedPreSums;
  typename IteratorA::Element* shared_operand_presum;

 private:


  // Structure encapsulating pipeline state live from one iteration to the next
  struct PipeState {

    using WarpLoadedFragmentA = typename Operator::FragmentA;
    using WarpLoadedFragmentB = typename Operator::FragmentB;
    using WarpTransformedFragmentA = typename Operator::TransformedFragmentA;
    using WarpTransformedFragmentB = typename Operator::TransformedFragmentB;

    /// Temporary accumulator to facilitate staged-accumulation
    FragmentC tmp_accum_;

    /// Pair of A fragments used to overlap shared memory loads and math instructions
    WarpLoadedFragmentA warp_loaded_frag_A_[2];
    WarpLoadedFragmentA warp_loaded_frag_A1_[2];
    WarpTransformedFragmentA warp_transformed_frag_A_[2];

    /// Pair of B fragments used to overlap shared memory loads and math instructions
    WarpLoadedFragmentB warp_loaded_frag_B_[2];
    WarpLoadedFragmentB warp_loaded_frag_B1_[2];
    WarpTransformedFragmentB warp_transformed_frag_B_[2];
  };


 private:

  //
  // Data members
  //

  /// Warp-level MMA operator
  Operator warp_mma_;

  /// Iterator to write threadblock-scoped tile of A operand to shared memory
  SmemIteratorA smem_iterator_A_;
  SmemIteratorA smem_iterator_A1_;

  SmemIteratorA read_smem_iterator_A_;
  SmemIteratorA read_smem_iterator_A1_;

  /// Iterator to write threadblock-scoped tile of B operand to shared memory
  SmemIteratorB smem_iterator_B_;
  SmemIteratorB smem_iterator_B1_;

  SmemIteratorB read_smem_iterator_B_;
  SmemIteratorB read_smem_iterator_B1_;

  /// Shared memory write stage index
  int smem_write_stage_idx_;

  /// Shared memory read stage index
  int smem_read_stage_idx_;


public:
  int presum_a_log_tile_multiplier;
  int presum_b_log_tile_multiplier;

  /// Construct from tensor references
  CUTLASS_DEVICE
  GlobalStrassenMmaMultistage(
      ///< Shared storage needed for internal use by threadblock-scoped GEMM
      typename Base::SharedStorage &shared_storage,
      ///< ID within the threadblock
      int thread_idx,
      ///< ID of warp
      int warp_idx,
      ///< ID of each thread within a warp
      int lane_idx,
      SharedPostsumOpIterator& shared_postsum_op_iter
    ):
      Base(shared_storage, thread_idx, warp_idx, lane_idx),
      smem_iterator_A_(shared_storage.operand_A_ref(), thread_idx),
      smem_iterator_B_(shared_storage.operand_B_ref(), thread_idx),
      smem_iterator_A1_(shared_storage.operand_A1_ref(), thread_idx),
      smem_iterator_B1_(shared_storage.operand_B1_ref(), thread_idx),
      read_smem_iterator_A_(smem_iterator_A_), read_smem_iterator_A1_(smem_iterator_A1_),
      read_smem_iterator_B_(smem_iterator_B_), read_smem_iterator_B1_(smem_iterator_B1_),
      smem_write_stage_idx_(0),
      smem_read_stage_idx_(0),
      sharedPreSums(nullptr, thread_idx)
  {
    // Compute warp location within threadblock tile by mapping the warp_id to
    // three coordinates:
    //   _m: the warp's position within the threadblock along the M dimension
    //   _n: the warp's position within the threadblock along the N dimension
    //   _k: the warp's position within the threadblock along the K dimension

    int warp_idx_mn = warp_idx % (Base::WarpCount::kM * Base::WarpCount::kN);
    int warp_idx_k = warp_idx / (Base::WarpCount::kM * Base::WarpCount::kN);

    int warp_idx_m = warp_idx_mn % Base::WarpCount::kM;
    int warp_idx_n = warp_idx_mn / Base::WarpCount::kM;

    // Add per-warp offsets in units of warp-level tiles
    this->warp_tile_iterator_A_.add_tile_offset(
        {warp_idx_m, Base::kWarpGemmIterations * warp_idx_k});
    this->warp_tile_iterator_B_.add_tile_offset(
        {Base::kWarpGemmIterations * warp_idx_k, warp_idx_n});
  }

  /// Advance shared memory read-iterators to the next stage
  CUTLASS_DEVICE
  void advance_smem_read_stage()
  {
    ++smem_read_stage_idx_;

    if (smem_read_stage_idx_ == Base::kStages) {
      // Wrap back around to the 'start' of the circular buffer in shared memory
      this->warp_tile_iterator_A_.add_tile_offset({0, -Base::kStages * Policy::kPartitionsK * Base::kWarpGemmIterations});
      this->warp_tile_iterator_B_.add_tile_offset({-Base::kStages * Policy::kPartitionsK * Base::kWarpGemmIterations, 0});

      smem_read_stage_idx_ = 0;
    }
  }

  CUTLASS_DEVICE
  void set_iteration_indexA(IteratorA& it0, IteratorA& it3, int val) {
    if (StrassenMiGroup::hasM0() ||
        StrassenMiGroup::hasM1() ||
        StrassenMiGroup::hasM2() ||
        StrassenMiGroup::hasM4() ||
        StrassenMiGroup::hasM5() ||
        StrassenMiGroup::hasM6())
      it0.set_iteration_index(val);
      
    if (StrassenMiGroup::hasM3())
      it3.set_iteration_index(val);
  }

  CUTLASS_DEVICE
  void set_iteration_indexB(IteratorB& it0, IteratorB& it3, int val) {
    if (StrassenMiGroup::hasM0() ||
        StrassenMiGroup::hasM1() ||
        StrassenMiGroup::hasM2() ||
        StrassenMiGroup::hasM3() ||
        StrassenMiGroup::hasM5() ||
        StrassenMiGroup::hasM6())
      it0.set_iteration_index(val);

    if (StrassenMiGroup::hasM4())
      it3.set_iteration_index(val);
  }

  CUTLASS_DEVICE
  void clear_maskA(IteratorA& it0, IteratorA& it3, bool val) {
    if (StrassenMiGroup::hasM0() ||
        StrassenMiGroup::hasM1() ||
        StrassenMiGroup::hasM2() ||
        StrassenMiGroup::hasM4() ||
        StrassenMiGroup::hasM5() ||
        StrassenMiGroup::hasM6())
      it0.clear_mask(val);

    if (StrassenMiGroup::hasM3())
      it3.clear_mask(val);
  }

  CUTLASS_DEVICE
  void clear_maskB(IteratorB& it0, IteratorB& it3, bool val) {
    if (StrassenMiGroup::hasM0() ||
        StrassenMiGroup::hasM1() ||
        StrassenMiGroup::hasM2() ||
        StrassenMiGroup::hasM3() ||
        StrassenMiGroup::hasM5() ||
        StrassenMiGroup::hasM6())
      it0.clear_mask(val);

    if (StrassenMiGroup::hasM4())
      it3.clear_mask(val);
  }

  CUTLASS_DEVICE
  void add_tile_offsetA(IteratorA& it0, IteratorA& it3, int x, int y) {
    if (StrassenMiGroup::hasM0() ||
        StrassenMiGroup::hasM1() ||
        StrassenMiGroup::hasM2() ||
        StrassenMiGroup::hasM4() ||
        StrassenMiGroup::hasM5() ||
        StrassenMiGroup::hasM6())
        it0.add_tile_offset({x, y});

    if (StrassenMiGroup::hasM3())
      it3.add_tile_offset({x, y});
  }

  CUTLASS_DEVICE
  void add_tile_offsetB(IteratorB& it0, IteratorB& it3, int x, int y) {
    if (StrassenMiGroup::hasM0() ||
        StrassenMiGroup::hasM1() ||
        StrassenMiGroup::hasM2() ||
        StrassenMiGroup::hasM3() ||
        StrassenMiGroup::hasM5() ||
        StrassenMiGroup::hasM6())
        it0.add_tile_offset({x, y});

    if (StrassenMiGroup::hasM4())
      it3.add_tile_offset({x, y});
  }

  /// Advance global memory read-iterators and shared memory write-iterators to the stage
  CUTLASS_DEVICE
  void advance_smem_write_stage(
    IteratorA& iterator_A0,      ///< [in|out] iterator over A operand in global memory
    IteratorA& iterator_A3,
    IteratorB& iterator_B0,     ///< [in|out] iterator over B operand in global memory
    IteratorB& iterator_B3)
  {
    // Advance global iterators
    add_tile_offsetA(iterator_A0, iterator_A3, 0, 1);
    add_tile_offsetB(iterator_B0, iterator_B3, 1, 0);

    // Advance shared iterators
    smem_iterator_A_.add_tile_offset({0, 1});
    smem_iterator_B_.add_tile_offset({1, 0});
    smem_iterator_A1_.add_tile_offset({0, 1});
    smem_iterator_B1_.add_tile_offset({1, 0});

    // Increment shared memory write stage index
    ++smem_write_stage_idx_;

    if (smem_write_stage_idx_ == Base::kStages) {
      // Wrap back around to the 'start' of the circular buffer in shared memory
      smem_iterator_A_.add_tile_offset({0, -Base::kStages});
      smem_iterator_B_.add_tile_offset({-Base::kStages, 0});
      smem_iterator_A1_.add_tile_offset({0, -Base::kStages});
      smem_iterator_B1_.add_tile_offset({-Base::kStages, 0});
      smem_write_stage_idx_ = 0;
    }
  }

  CUTLASS_DEVICE
  void copy_tiles_and_advance(IteratorA& iterator_A0,      ///< [in|out] iterator over A operand in global memory
      IteratorA& iterator_A3,
      IteratorB& iterator_B0,     ///< [in|out] iterator over B operand in global memory
      IteratorB& iterator_B3,
      int group_start_A = 0, int group_start_B = 0) {
    set_iteration_indexA(iterator_A0, iterator_A3, group_start_A * IteratorA::kAccessesPerVector);

    this->smem_iterator_A_.set_iteration_index(group_start_A);
    this->smem_iterator_A1_.set_iteration_index(group_start_A);

    set_iteration_indexB(iterator_B0, iterator_B3, group_start_B * IteratorB::kAccessesPerVector);

    this->smem_iterator_B_.set_iteration_index(group_start_B);
    this->smem_iterator_B1_.set_iteration_index(group_start_B);

    prefetch_load<Detail::kAccessesPerGroupA, Detail::kAccessesPerGroupB>(iterator_A0, iterator_A3,
                  iterator_B0, iterator_B3,
                  group_start_A, group_start_B);
  }

  template<cutlass::arch::CacheOperation::Kind CacheOp, typename Iterator, typename SmemIterator>
  CUTLASS_DEVICE
  void cp_async_iterator(Iterator& iterator, SmemIterator& smem_iterator, uint AsyncCopyIterationsPerStage, int loopRange,
  int group_start = 0) {
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < loopRange; ++j) {
      if (group_start + j < AsyncCopyIterationsPerStage) {
        typename Iterator::AccessType *dst_ptr =
            reinterpret_cast<typename Iterator::AccessType *>(
                smem_iterator.get());

        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < Iterator::kAccessesPerVector; ++v) {
          int const kSrcBytes =
              sizeof_bits<typename Iterator::Element>::value *
              Iterator::ThreadMap::kElementsPerAccess /
              Iterator::kAccessesPerVector / 8;

          int src_bytes = (iterator.valid() ? kSrcBytes : 0);

          if (SharedMemoryClear == SharedMemoryClearOption::kZfill) {
            cutlass::arch::cp_async_zfill<kSrcBytes, CacheOp>(
              dst_ptr + v, iterator.get(), iterator.valid());
          } else {
            cutlass::arch::cp_async<kSrcBytes, CacheOp>(
              dst_ptr + v, iterator.get(), iterator.valid());
          }

          ++iterator;
        }

        ++smem_iterator;
      }
    }
  }

  template<uint AsyncCopyIterationsPerStage, uint LoopRange, cutlass::arch::CacheOperation::Kind CacheOp, typename Iterator, typename SmemIterator>
  CUTLASS_DEVICE
  void cp_async_iterator_parts(int part1, int part2, Iterator& iterator, SmemIterator& smem_iterator_, SmemIterator& smem_iterator_1, int group_start = 0) {
    // typename Iterator::AccessType* part2Ptrs[LoopRange]; 

    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < LoopRange; ++j) {
      if (group_start + j < AsyncCopyIterationsPerStage) {
        typename Iterator::AccessType *dst_ptr_ =
            reinterpret_cast<typename Iterator::AccessType *>(
                smem_iterator_.get());

        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < Iterator::kAccessesPerVector; ++v) {
          int const kSrcBytes =
              sizeof_bits<typename Iterator::Element>::value *
              Iterator::ThreadMap::kElementsPerAccess /
              Iterator::kAccessesPerVector / 8;

          // part2Ptrs[j]  = iterator.get(part2);
          if (SharedMemoryClear == SharedMemoryClearOption::kZfill) {
            cutlass::arch::cp_async_zfill<kSrcBytes, CacheOp>(
              dst_ptr_  + v, iterator.get(0), iterator.valid());
          } else {
            cutlass::arch::cp_async<kSrcBytes, CacheOp>(
              dst_ptr_  + v, iterator.get(0), iterator.valid());
          }

          ++iterator;
        }

        ++smem_iterator_;
      }
    }

    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < LoopRange; ++j) {
      if (group_start + j < AsyncCopyIterationsPerStage) {
        typename Iterator::AccessType *dst_ptr_1 =
            reinterpret_cast<typename Iterator::AccessType *>(
                smem_iterator_1.get());

        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < Iterator::kAccessesPerVector; ++v) {
          int const kSrcBytes =
              sizeof_bits<typename Iterator::Element>::value *
              Iterator::ThreadMap::kElementsPerAccess /
              Iterator::kAccessesPerVector / 8;

          if (SharedMemoryClear == SharedMemoryClearOption::kZfill) {
            cutlass::arch::cp_async_zfill<kSrcBytes, CacheOp>(
              dst_ptr_1 + v, iterator.get(1), iterator.valid(1));
          } else {
            cutlass::arch::cp_async<kSrcBytes, CacheOp>(
              dst_ptr_1 + v, iterator.get(1), iterator.valid(1));
          }
          iterator.inc(1);
        }
        ++smem_iterator_1;
      }
    }
  }

  template<uint LoopRangeA, uint LoopRangeB>
  CUTLASS_DEVICE
  void prefetch_load(IteratorA& iterator_A0,      ///< [in|out] iterator over A operand in global memory
                     IteratorA& iterator_A3,
                     IteratorB& iterator_B0,     ///< [in|out] iterator over B operand in global memory
                     IteratorB& iterator_B3,
                     int group_start_A = 0, int group_start_B = 0) {
    uint loop_rangeB = LoopRangeB;
    uint loop_rangeA = LoopRangeA;
    if (StrassenMiGroup::hasM0()) {
      // Async Copy for operand A
      cp_async_iterator_parts<Detail::AsyncCopyIterationsPerStageA, LoopRangeA, kCacheOpA>(0, 1, iterator_A0, this->smem_iterator_A_, this->smem_iterator_A1_, group_start_A);

      // Async Copy for operand B
      cp_async_iterator_parts<Detail::AsyncCopyIterationsPerStageB, LoopRangeB, kCacheOpB>(0, 1, iterator_B0, this->smem_iterator_B_, this->smem_iterator_B1_, group_start_B);
    } else if (StrassenMiGroup::hasM1()) {
      // Async Copy for operand A
      cp_async_iterator_parts<Detail::AsyncCopyIterationsPerStageA, LoopRangeA, kCacheOpA>(0, 1, iterator_A0, this->smem_iterator_A_, this->smem_iterator_A1_, group_start_A);

      // Async Copy for operand B
      cp_async_iterator<kCacheOpB>(iterator_B0, this->smem_iterator_B_,  Detail::AsyncCopyIterationsPerStageB, loop_rangeB, group_start_B);
    } else if (StrassenMiGroup::hasM2()) {
      // Async Copy for operand A
      cp_async_iterator<kCacheOpA>(iterator_A0, this->smem_iterator_A_,  Detail::AsyncCopyIterationsPerStageA, loop_rangeA, group_start_A);

      // Async Copy for operand B
      cp_async_iterator_parts<Detail::AsyncCopyIterationsPerStageB, LoopRangeB, kCacheOpB>(0, 1, iterator_B0, this->smem_iterator_B_, this->smem_iterator_B1_, group_start_B);
    } else if (StrassenMiGroup::hasM3()) {
      // Async Copy for operand A
      cp_async_iterator<kCacheOpA>(iterator_A3, this->smem_iterator_A_,  Detail::AsyncCopyIterationsPerStageA, loop_rangeA, group_start_A);

      // Async Copy for operand B
      cp_async_iterator_parts<Detail::AsyncCopyIterationsPerStageB, LoopRangeB, kCacheOpB>(0, 1, iterator_B0, this->smem_iterator_B_, this->smem_iterator_B1_, group_start_B);
    } else if (StrassenMiGroup::hasM4()) {
      // Async Copy for operand A
      cp_async_iterator_parts<Detail::AsyncCopyIterationsPerStageA, LoopRangeA, kCacheOpA>(0, 1, iterator_A0, this->smem_iterator_A_, this->smem_iterator_A1_, group_start_A);

      // Async Copy for operand B
      cp_async_iterator<kCacheOpB>(iterator_B3, this->smem_iterator_B_,  Detail::AsyncCopyIterationsPerStageB, loop_rangeB, group_start_B);
    } else if (StrassenMiGroup::hasM5()) {
      // Async Copy for operand A
      cp_async_iterator_parts<Detail::AsyncCopyIterationsPerStageA, LoopRangeA, kCacheOpA>(0, 1, iterator_A0, this->smem_iterator_A_, this->smem_iterator_A1_, group_start_A);

      // Async Copy for operand B
      cp_async_iterator_parts<Detail::AsyncCopyIterationsPerStageB, LoopRangeB, kCacheOpB>(0, 1, iterator_B0, this->smem_iterator_B_, this->smem_iterator_B1_, group_start_B);
    } else if (StrassenMiGroup::hasM6()) {
      // Async Copy for operand A
      cp_async_iterator_parts<Detail::AsyncCopyIterationsPerStageA, LoopRangeA, kCacheOpA>(0, 1, iterator_A0, this->smem_iterator_A_, this->smem_iterator_A1_, group_start_A);

      // Async Copy for operand B
      cp_async_iterator_parts<Detail::AsyncCopyIterationsPerStageB, LoopRangeB, kCacheOpB>(0, 1, iterator_B0, this->smem_iterator_B_, this->smem_iterator_B1_, group_start_B);
    }
  }

  /// GEMM prologue.  Bootstrap the global->shared memory pipeline by fetching
  /// the global fragments needed by the first kStages-1 threadblock mainloop iterations
  CUTLASS_DEVICE
  void prologue(
    IteratorA& iterator_A0,      ///< [in|out] iterator over A operand in global memory
    IteratorA& iterator_A3,
    IteratorB& iterator_B0,     ///< [in|out] iterator over B operand in global memory
    IteratorB& iterator_B3,
    int &gemm_k_iterations)     ///< [in|out] number of threadblock mainloop iterations remaining
  {
    // Issue several complete stages
    CUTLASS_PRAGMA_UNROLL
    for (int stage = 0; stage < Base::kStages - 1; ++stage, --gemm_k_iterations) {
      
      // Disable global fetching if done with global fetch iterations
      clear_maskA(iterator_A0, iterator_A3, gemm_k_iterations == 0);
      clear_maskB(iterator_B0, iterator_B3, gemm_k_iterations == 0);
      set_iteration_indexB(iterator_B0, iterator_B3, 0);
      set_iteration_indexA(iterator_A0, iterator_A3, 0);

      this->smem_iterator_A_.set_iteration_index(0);
      this->smem_iterator_B_.set_iteration_index(0);
      this->smem_iterator_A1_.set_iteration_index(0);
      this->smem_iterator_B1_.set_iteration_index(0);

      prefetch_load<Detail::AsyncCopyIterationsPerStageA, Detail::AsyncCopyIterationsPerStageB>
                    (iterator_A0, iterator_A3,
                     iterator_B0, iterator_B3);

      // Move to the next write stage
      advance_smem_write_stage(iterator_A0, iterator_A3,
                               iterator_B0, iterator_B3);

      // Defines the boundary of a stage of cp.async.
      cutlass::arch::cp_async_fence();
    }

    // Optionally clear the remaining stages of SMEM. This is a functional requirement for
    // some kernels so that all accumulator elements outside the GEMM footprint are zero.
    if (false && SharedMemoryClear == SharedMemoryClearOption::kClearLastStage) {

      /// Iterator to write threadblock-scoped tile of A operand to shared memory
      SmemIteratorA last_smem_iterator_A(this->smem_iterator_A_);
      typename IteratorA::AccessType zero_A;

      zero_A.clear();
      last_smem_iterator_A.set_iteration_index(0);

      // Async Copy for operand A
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < Detail::AsyncCopyIterationsPerStageA; ++j) {

        typename IteratorA::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorA::AccessType *>(
                last_smem_iterator_A.get());

        *dst_ptr = zero_A;

        ++last_smem_iterator_A;
      }

      /// Iterator to write threadblock-scoped tile of B operand to shared memory
      SmemIteratorB last_smem_iterator_B(this->smem_iterator_B_);
      typename IteratorB::AccessType zero_B;

      zero_B.clear();
      last_smem_iterator_B.set_iteration_index(0);

      // Async Copy for operand B
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < Detail::AsyncCopyIterationsPerStageB; ++j) {

        typename IteratorB::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorB::AccessType *>(
                last_smem_iterator_B.get());

        *dst_ptr = zero_B;

        ++last_smem_iterator_B;
      }
    }
  }

  template<typename Fragment>
  CUTLASS_DEVICE
  void sum(Fragment& frag_op1, Fragment& frag_op2, Fragment& frag_out) {
    #pragma unroll
    for (int i = 0; i < frag_op1.size(); i++) {
      frag_out[i] = frag_op1[i] + frag_op2[i];
    }
  }

  template<typename Fragment>
  CUTLASS_DEVICE
  void subtract(Fragment& frag_op1, Fragment& frag_op2, Fragment& frag_out) {
    #pragma unroll
    for (int i = 0; i < frag_op1.size(); i++) {
      frag_out[i] = frag_op1[i] - frag_op2[i];
    }
  }

  /// Wait until we have at least one completed global fetch stage
  CUTLASS_DEVICE
  void gmem_wait(int gemm_k_iterations, int read_stage)
  {
    // Wait until we have at least one committed global fetch stage. (#uncommitted = Base::kStages - 1 - #committed)
    cutlass::arch::cp_async_wait<Base::kStages - 2>();

    // if (!(gemm_k_iterations - 1 > (-Base::kStages + 1))) return;
        
    bool addA = StrassenMiGroup::hasM0() ||
        StrassenMiGroup::hasM1() ||
        StrassenMiGroup::hasM4() ||
        StrassenMiGroup::hasM5() ||
        StrassenMiGroup::hasM6();
    if (addA) {
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < Detail::AsyncCopyIterationsPerStageA; ++j) {
        typename IteratorA::AccessType __restrict__  *ptrA =
          reinterpret_cast<typename IteratorA::AccessType *>(
            read_smem_iterator_A_.get());

        typename IteratorA::AccessType __restrict__  *ptrA1 =
          reinterpret_cast<typename IteratorA::AccessType *>(
            read_smem_iterator_A1_.get());

        typename IteratorA::AccessType *writeA = ptrA;

        using ElementA = typename IteratorA::Element;

        // CUTLASS_PRAGMA_UNROLL
        // for (int v = 0; v < IteratorA::kAccessesPerVector; ++v) {
          auto e1 = *ptrA;
          auto e2 = *ptrA1;
                 if (StrassenMiGroup::hasM6()) {
            *(writeA) = e1 - e2;
          } else if (StrassenMiGroup::hasM5()) {
            *(writeA) = e1 - e2;
          } else {
            *(writeA) = e1 + e2;
          }
        // }

        ++read_smem_iterator_A_;
        ++read_smem_iterator_A1_;
      }
    }
    
    bool addB = StrassenMiGroup::hasM0() ||
        StrassenMiGroup::hasM2() ||
        StrassenMiGroup::hasM3() ||
        StrassenMiGroup::hasM5() ||
        StrassenMiGroup::hasM6();

    if (addB) {
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < Detail::AsyncCopyIterationsPerStageB; ++j) {
        typename IteratorB::AccessType __restrict__  *ptrB =
            reinterpret_cast<typename IteratorB::AccessType *>(
                read_smem_iterator_B_.get());

        typename IteratorB::AccessType __restrict__ *ptrB1 =
            reinterpret_cast<typename IteratorB::AccessType *>(
                read_smem_iterator_B1_.get());

        typename IteratorB::AccessType *writeB = ptrB;

        using ElementB = typename IteratorB::Element;

        // CUTLASS_PRAGMA_UNROLL
        // for (int v = 0; v < IteratorB::kAccessesPerVector; ++v) {
          auto e1 = *ptrB;
          auto e2 = *ptrB1;
          if (StrassenMiGroup::hasM2() ||
              StrassenMiGroup::hasM3()) {
            *(writeB) = e1 - e2;
          } else {
            *(writeB) = e1 + e2;
          }
        // }

        ++read_smem_iterator_B_;
        ++read_smem_iterator_B1_;
      }
    }
    
    // Advance shared iterators
    if (addA) {
      read_smem_iterator_A_.add_tile_offset({0, 1});
      read_smem_iterator_A1_.add_tile_offset({0, 1});
    }

    if (addB) {
      read_smem_iterator_B_.add_tile_offset({1, 0});
      read_smem_iterator_B1_.add_tile_offset({1, 0});
    }

    if (read_stage + 1 == Base::kStages) {
      // Wrap back around to the 'start' of the circular buffer in shared memory
      if (addA) {
        read_smem_iterator_A_.add_tile_offset({0, -Base::kStages});
        read_smem_iterator_A1_.add_tile_offset({0, -Base::kStages});
      }

      if (addB) {
        read_smem_iterator_B_.add_tile_offset({-Base::kStages, 0});
        read_smem_iterator_B1_.add_tile_offset({-Base::kStages, 0});
      }
    }
  }


  /// Perform a threadblock mainloop iteration of matrix multiply-accumulate
  CUTLASS_DEVICE
  void mac_loop_iter(
    PipeState &pipe_state,          ///< [in|out] loop-carried pipeline state
    FragmentC &accum,               ///< [in|out] destination accumulator tile
    IteratorA& iterator_A0,      ///< [in|out] iterator over A operand in global memory
    IteratorA& iterator_A3,
    IteratorB& iterator_B0,     ///< [in|out] iterator over B operand in global memory
    IteratorB& iterator_B3,
    int &gemm_k_iterations)         ///< [in|out] number of threadblock mainloop iterations remaining
  {
    // Unroll the warp-level MMA tiles of a threadblock's mainloop iteration
    CUTLASS_PRAGMA_UNROLL
    for (int warp_mma_k = 0; warp_mma_k < Base::kWarpGemmIterations; ++warp_mma_k) {
      // Load the next warp-tile's A fragment from shared memory
      this->warp_tile_iterator_A_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations);
      this->warp_tile_iterator_A_.load(pipe_state.warp_loaded_frag_A_[(warp_mma_k + 1) % 2]);
      ++this->warp_tile_iterator_A_;

      // Load the next warp-tile's B fragment from shared memory
      this->warp_tile_iterator_B_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations);
      this->warp_tile_iterator_B_.load(pipe_state.warp_loaded_frag_B_[(warp_mma_k + 1) % 2]);
      ++this->warp_tile_iterator_B_;

      // Except for the first warp-tile, all warp-tiles convert their incoming shared memory fragments as necessary
      if (warp_mma_k > 0) {
        warp_mma_.transform(
          pipe_state.warp_transformed_frag_A_[warp_mma_k % 2],
          pipe_state.warp_transformed_frag_B_[warp_mma_k % 2],
          pipe_state.warp_loaded_frag_A_[warp_mma_k % 2],
          pipe_state.warp_loaded_frag_B_[warp_mma_k % 2]);
      }

      // Execute the current warp-tile of MMA operations
      if (Detail::kStagedAccumulation) {
        warp_mma_(
          pipe_state.tmp_accum_,
          pipe_state.warp_transformed_frag_A_[warp_mma_k % 2],
          pipe_state.warp_transformed_frag_B_[warp_mma_k % 2],
          pipe_state.tmp_accum_
        );

        if (warp_mma_k == 0) {
          plus<FragmentC> plus_accum;
          accum = plus_accum(accum, pipe_state.tmp_accum_);
          pipe_state.tmp_accum_.clear();
        }
      } else {
        warp_mma_(
          accum,
          pipe_state.warp_transformed_frag_A_[warp_mma_k % 2],
          pipe_state.warp_transformed_frag_B_[warp_mma_k % 2],
          accum
        );
      }

      // Except for the last warp-tile, all warp-tiles issue their share of
      // global->shared fragment copies
      if (warp_mma_k < Base::kWarpGemmIterations - 1) {

        int group_start_iteration_A, group_start_iteration_B;
        group_start_iteration_A = warp_mma_k * Detail::kAccessesPerGroupA;
        group_start_iteration_B = warp_mma_k * Detail::kAccessesPerGroupB;

        copy_tiles_and_advance(
            iterator_A0, iterator_A3,
            iterator_B0, iterator_B3,
            group_start_iteration_A,
            group_start_iteration_B);
      }

      // The second-to-last warp-tile also:
      //   - performs the last warp-tile's share of global->shared fragment copies
      //   - moves to the next global fetch stage
      if (warp_mma_k + 2 == Base::kWarpGemmIterations) {

        // Performs the last warp-tile's share of global->shared fragment copies
        int group_start_iteration_A = (warp_mma_k + 1) * Detail::kAccessesPerGroupA;
        int group_start_iteration_B = (warp_mma_k + 1) * Detail::kAccessesPerGroupB;

        copy_tiles_and_advance(
          iterator_A0, iterator_A3,
          iterator_B0, iterator_B3,
          group_start_iteration_A,
          group_start_iteration_B);

        // Inserts a memory fence between stages of cp.async instructions.
        cutlass::arch::cp_async_fence();

        // Wait until we have at least one completed global fetch stage
        gmem_wait(gemm_k_iterations, (smem_read_stage_idx_ + 1));

        // Move to the next global fetch stage
        advance_smem_write_stage(iterator_A0, iterator_A3,
                                 iterator_B0, iterator_B3);
        advance_smem_read_stage();
        __syncthreads();

        // Disable global fetching when done with global fetch iterations
        --gemm_k_iterations;
        clear_maskA(iterator_A0, iterator_A3, gemm_k_iterations == 0);
        clear_maskB(iterator_B0, iterator_B3, gemm_k_iterations == 0);
      }

      // The last warp-tile also converts the shared memory fragments used by
      // the first warp-tile of the next iteration, if necessary (so we can
      // immediately start issuing MMA instructions at the top of the loop )
      if (warp_mma_k + 1 == Base::kWarpGemmIterations) {

        warp_mma_.transform(
          pipe_state.warp_transformed_frag_A_[(warp_mma_k + 1) % 2],
          pipe_state.warp_transformed_frag_B_[(warp_mma_k + 1) % 2],
          pipe_state.warp_loaded_frag_A_[(warp_mma_k + 1) % 2],
          pipe_state.warp_loaded_frag_B_[(warp_mma_k + 1) % 2]);
      }

    }

    //Printing something here will lead to a compiler bug
  }


  /// Perform the specified number of threadblock mainloop iterations of matrix
  /// multiply-accumulate.  Assumes prologue has been initiated.
  CUTLASS_DEVICE
  void gemm_iters(
      int gemm_k_iterations,        ///< number of threadblock mainloop iterations
      FragmentC &accum,             ///< [in|out] accumulator tile
      IteratorA& iterator_A0,      ///< [in|out] iterator over A operand in global memory
      IteratorA& iterator_A3,
      IteratorB& iterator_B0,     ///< [in|out] iterator over B operand in global memory
      IteratorB& iterator_B3)
  {
    PipeState pipe_state;

    // Disable global fetching if done with global fetch iterations
    clear_maskA(iterator_A0, iterator_A3, gemm_k_iterations == 0);
    clear_maskB(iterator_B0, iterator_B3, gemm_k_iterations == 0);

    // Load first warp-tile's A fragment from shared memory
    this->warp_tile_iterator_A_.set_kgroup_index(0);
    this->warp_tile_iterator_A_.load(pipe_state.warp_loaded_frag_A_[0]);
    ++this->warp_tile_iterator_A_;

    // Load first warp-tile's B fragment from shared memory
    this->warp_tile_iterator_B_.set_kgroup_index(0);
    this->warp_tile_iterator_B_.load(pipe_state.warp_loaded_frag_B_[0]);
    ++this->warp_tile_iterator_B_;

    // Transform, if necessary, the first warp-tile's shared memory fragments
    warp_mma_.transform(
      pipe_state.warp_transformed_frag_A_[0],
      pipe_state.warp_transformed_frag_B_[0],
      pipe_state.warp_loaded_frag_A_[0],
      pipe_state.warp_loaded_frag_B_[0]);

    if (Detail::kStagedAccumulation) {
      pipe_state.tmp_accum_.clear();
    }

    // Mainloop
    CUTLASS_GEMM_LOOP
    for (; gemm_k_iterations > (-Base::kStages + 1);) {
      mac_loop_iter(
        pipe_state,
        accum,
        iterator_A0, iterator_A3,
        iterator_B0, iterator_B3,
        gemm_k_iterations);
    }
  
    if (Detail::kStagedAccumulation) {
      plus<FragmentC> plus_accum;
      accum = plus_accum(accum, pipe_state.tmp_accum_);
    }

    // Commit and drain all pending and predicated cp.async pnz from the GEMM mainloop
    cutlass::arch::cp_async_fence();
    cutlass::arch::cp_async_wait<0>();
    __syncthreads();

    // if (threadIdx.x == 0 and StrassenMiGroup::hasM4()) {
    //   printf("832: %f\n", accum[0]);
    // }
  }


  /// Prepares the class for another prologue.
  CUTLASS_DEVICE
  void wind_down()
  {
    // Catch-up the smem-read iterator to the smem-write iterator (so this class can be reused for another tile's prologue)

    // First, increment remaining warp tiles to get to the next full stage.  (Ideally we would
    // just decrement one tile, but not all iterators implement --() decrement.)
    #pragma unroll
    for (int warp_mma_k = 1; warp_mma_k < Base::kWarpGemmIterations; ++warp_mma_k)
    {
      this->warp_tile_iterator_A_.set_kgroup_index(warp_mma_k);
      this->warp_tile_iterator_B_.set_kgroup_index(warp_mma_k);

      ++this->warp_tile_iterator_A_;
      ++this->warp_tile_iterator_B_;
    }
    smem_read_stage_idx_++;

    // Then wrap back two full stages (one for the tile advancing we just did, and one to catch the write iterators)
    static const int kStageIters = Policy::kPartitionsK * Base::kWarpGemmIterations;
    if (smem_read_stage_idx_ > 1)
    {
      this->warp_tile_iterator_A_.add_tile_offset({0, (-2 * kStageIters)});
      this->warp_tile_iterator_B_.add_tile_offset({(-2 * kStageIters), 0});
    }
    else
    {
      this->warp_tile_iterator_A_.add_tile_offset({0, ((Base::kStages - 2) * kStageIters)});
      this->warp_tile_iterator_B_.add_tile_offset({((Base::kStages - 2) * kStageIters), 0});
    }
    smem_read_stage_idx_ = smem_write_stage_idx_;
  }


  /// Perform a threadblock-scoped matrix multiply-accumulate
  CUTLASS_DEVICE
  void operator()(
    int gemm_k_iterations,                            ///< number of iterations of the mainloop
    FragmentC accumM[7],
    IteratorA iterator_A0,                             ///< iterator over A operand in global memory
    IteratorA iterator_A1,
    IteratorA iterator_A2,
    IteratorA iterator_A3,
    IteratorB iterator_B0,                             ///< iterator over B operand in global memory
    IteratorB iterator_B1,
    IteratorB iterator_B2,
    IteratorB iterator_B3,
    IteratorA iterator_PresumAs[],
    IteratorB iterator_PresumBs[],
    PresumGlobalIteratorA iter_PresumA,
    PresumGlobalIteratorB iter_PresumB,
    PresumGlobalIteratorA iter_PresumA_M,
    PresumGlobalIteratorB iter_PresumB_M,
    PostsumSrcTileIterator source_interim_iterator)
  {

    FragmentC& accum = (StrassenMiGroup::hasM0()) ? accumM[0] :
                       (StrassenMiGroup::hasM1()) ? accumM[1] :
                       (StrassenMiGroup::hasM2()) ? accumM[2] :
                       (StrassenMiGroup::hasM3()) ? accumM[3] :
                       (StrassenMiGroup::hasM4()) ? accumM[4] :
                       (StrassenMiGroup::hasM5()) ? accumM[5] : accumM[6];
                      //  (StrassenMiGroup::hasM6()) ? m6 : 


    // Prologue
    prologue(iterator_A0, iterator_A3,
             iterator_B0, iterator_B3,
             gemm_k_iterations);
    // Wait until we have at least one completed global fetch stage
    gmem_wait(gemm_k_iterations, 0);
    __syncthreads();
    // Perform accumulation in the 'd' output operand
    // Perform the MAC-iterations
    gemm_iters(gemm_k_iterations, accum, 
               iterator_A0, iterator_A3,
               iterator_B0, iterator_B3);

      // if (MmaStrassenKind - MmaStrassen::Type::GlobalLevel1_M0 == 0 &&
      //     threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
      //   printf("670 %d: %f\n",
      //       MmaStrassenKind - MmaStrassen::Type::GlobalLevel1_M0, accum[0]);
      // }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace gemm
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////