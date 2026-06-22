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

#include "cutlass/gemm/threadblock/mma_base.h"
#include "cutlass/gemm/threadblock/presum_detail.h"
#include "cutlass/epilogue/threadblock/strassen_interim_epilogue_tile_iterator.h"

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
    typename InterimEpilogueOp,
    /// Use zfill or predicate for out-of-bound cp.async
    SharedMemoryClearOption SharedMemoryClear = SharedMemoryClearOption::kNone,
    /// Used for partial specialization
    typename Enable = bool>
class StrassenMmaMultistage : 
  public StrassenMmaBase<Shape_, StrassenShape_, Policy_, Stages, MmaStrassenKind, StrassenMiGroup_> {
public:
  ///< Base class
  using StrassenMiGroup = StrassenMiGroup_;
  using Base = StrassenMmaBase<Shape_, StrassenShape_, Policy_, Stages, MmaStrassenKind, StrassenMiGroup_>;
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
  
  //Multiply Shape::kN by F for a load-balanced distribution when K < N or K > N.
  //When K >N, F = K>N. When K<N, F=K/N and this F should be tuned
  using PresumShapeA = GemmShape<Shape::kM, Shape::kN, 1>;
  using PresumShapeB = GemmShape<Shape::kM, Shape::kN, 1>;

  static const int PresumAccessPerIter = 1;//TODO: change for half
  using PresumVecTypeA = Array<typename IteratorA::Element, 16/sizeof(typename IteratorA::Element)>;
  using PresumVecTypeB = Array<typename IteratorB::Element, 16/sizeof(typename IteratorB::Element)>;
  using PresumComputeType = ElementC;
  using PresumComputeVecType = Array<PresumComputeType, 16/sizeof(PresumComputeType)>;
  using PresumComputeToIOType = NumericArrayConverter<typename IteratorA::Element, PresumComputeType, PresumVecTypeA::kElements>;
  using PresumIOToComputeType = NumericArrayConverter<PresumComputeType, typename IteratorA::Element, PresumVecTypeA::kElements>;

  using PresumGlobalIteratorA = PresumDetail::GlobalIterator<typename IteratorA::Element, NumThreads,  
                                                            PresumShapeA, PresumVecTypeA, false, false>;
  using PresumGlobalIteratorB = PresumDetail::GlobalIterator<typename IteratorB::Element, NumThreads,  
                                                            PresumShapeB, PresumVecTypeB, true, false>;

  using PresumSharedIterator = PresumDetail::SharedIterator<typename IteratorA::Element, PresumVecTypeA, NumThreads,
                                                            Base::SharedStorage::PresumBuffer::NumLoadsForPresum,
                                                            Base::SharedStorage::PresumBuffer::SingleStageSize,
                                                            Stages, PresumAccessPerIter>;
  static const uint kPresumComputeIterationsA = (Base::SharedStorage::PresumBuffer::SingleStageSize > 0) ?
                                                PresumShapeA::kCount/Base::SharedStorage::PresumBuffer::SingleStageSize : 1;
  static const uint kPresumComputeIterationsB = (Base::SharedStorage::PresumBuffer::SingleStageSize > 0) ?
                                                PresumShapeB::kCount/Base::SharedStorage::PresumBuffer::SingleStageSize : 1;

  using SharedPostsumOpIterator = cutlass::epilogue::threadblock::StrassenInterimEpilogueSharedTileIterator<
    StrassenShape_,
    typename InterimEpilogueOp::ElementOutput,
    NumThreads,
    InterimEpilogueOp::kCount
  >;

  using PostsumSrcTileIterator = cutlass::epilogue::threadblock::StrassenInterimEpilogueTileIterator<
    StrassenShape_,
    InterimEpilogueOp,
    typename InterimEpilogueOp::ElementOutput,
    NumThreads,
    InterimEpilogueOp::kCount
  >;

  SharedPostsumOpIterator shared_postsum_op_iter;

  PresumSharedIterator sharedPreSums;

 private:

  MmaStrassen::InputAccesses<MmaStrassen::APresums::ANumPresums> APresumStores = StrassenMiGroup::APresumStores();
  MmaStrassen::InputAccesses<MmaStrassen::BPresums::BNumPresums> BPresumStores = StrassenMiGroup::BPresumStores();
  MmaStrassen::InputAccesses<MmaStrassen::APresums::ANumPresums> APresumLoads = StrassenMiGroup::APresumLoads();
  MmaStrassen::InputAccesses<MmaStrassen::BPresums::BNumPresums> BPresumLoads = StrassenMiGroup::BPresumLoads();

  // Structure encapsulating pipeline state live from one iteration to the next
  struct PipeState {

    using WarpLoadedFragmentA = typename Operator::FragmentA;
    using WarpLoadedFragmentB = typename Operator::FragmentB;
    using WarpTransformedFragmentA = typename Operator::TransformedFragmentA;
    using WarpTransformedFragmentB = typename Operator::TransformedFragmentB;
    static const int WarpFrags = 2;

    /// Temporary accumulator to facilitate staged-accumulation
    FragmentC tmp_accum_;

    /// Pair of A fragments used to overlap shared memory loads and math instructions
    WarpLoadedFragmentA warp_loaded_frag_A_M_[7][WarpFrags];
    WarpTransformedFragmentA warp_transformed_frag_A_M_[7][WarpFrags];

    WarpLoadedFragmentA warp_loaded_frag_input_A_[WarpFrags];
    WarpTransformedFragmentA warp_transformed_frag_input_A_[WarpFrags];

    /// Pair of B fragments used to overlap shared memory loads and math instructions
    WarpLoadedFragmentB warp_loaded_frag_B_M_[7][WarpFrags];
    WarpTransformedFragmentB warp_transformed_frag_B_M_[7][WarpFrags];

    WarpLoadedFragmentB warp_loaded_frag_input_B_[WarpFrags];
    WarpTransformedFragmentB warp_transformed_frag_input_B_[WarpFrags];
  };


 private:

  //
  // Data members
  //

  /// Warp-level MMA operator
  Operator warp_mma_;

  /// Iterator to write threadblock-scoped tile of A operand to shared memory
  SmemIteratorA smem_iterator_A_[4];
  SmemIteratorA smem_iterator_presum_A_[4]; //A02, S1, S2, A1S2

  //Read for computing and write presum in shared memory
  SmemIteratorA read_smem_iterator_A_[4];
  SmemIteratorA read_smem_iterator_presum_A_[4]; //A02, S1, S2, A1S2

  SmemIteratorA smem_iterator_A_M_[7];

  SmemIteratorA smem_iterator_input_A_;

  /// Iterator to write threadblock-scoped tile of B operand to shared memory
  SmemIteratorB smem_iterator_B_[4];
  SmemIteratorB smem_iterator_presum_B_[4]; //B31, B10, S3, S3B2

  //Read for computing and write presum in shared memory
  SmemIteratorB read_smem_iterator_B_[4];
  SmemIteratorB read_smem_iterator_presum_B_[4]; ////B31, B10, S3, S3B2

  SmemIteratorB smem_iterator_B_M_[7];

  SmemIteratorB smem_iterator_input_B_;

  /// Shared memory write stage index
  int smem_write_stage_idx_;

  /// Shared memory read stage index
  int smem_read_stage_idx_;

  
public:
  typename IteratorA::Element* shared_operand_presum;

  /// Construct from tensor references
  CUTLASS_DEVICE
  StrassenMmaMultistage(
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
      smem_iterator_A_{
        SmemIteratorA(shared_storage.operand_A0_ref(), thread_idx),
        SmemIteratorA(shared_storage.operand_A1_ref(), thread_idx),
        SmemIteratorA(shared_storage.operand_A2_ref(), thread_idx),
        SmemIteratorA(shared_storage.operand_A3_ref(), thread_idx)
      },

      smem_iterator_presum_A_{
        SmemIteratorA(shared_storage.operand_A02_ref(), thread_idx),
        SmemIteratorA(shared_storage.operand_S1_ref(), thread_idx),
        SmemIteratorA(shared_storage.operand_S2_ref(), thread_idx),
        SmemIteratorA(shared_storage.operand_A1S2_ref(), thread_idx)
      },

      read_smem_iterator_A_{
        SmemIteratorA(shared_storage.operand_A0_ref(), thread_idx),
        SmemIteratorA(shared_storage.operand_A1_ref(), thread_idx),
        SmemIteratorA(shared_storage.operand_A2_ref(), thread_idx),
        SmemIteratorA(shared_storage.operand_A3_ref(), thread_idx)
      },

      read_smem_iterator_presum_A_{
        SmemIteratorA(shared_storage.operand_A02_ref(), thread_idx),
        SmemIteratorA(shared_storage.operand_S1_ref(), thread_idx),
        SmemIteratorA(shared_storage.operand_S2_ref(), thread_idx),
        SmemIteratorA(shared_storage.operand_A1S2_ref(), thread_idx),
      },

      smem_iterator_input_A_(shared_storage.operand_input_A_ref(), thread_idx),

      smem_iterator_B_{
        SmemIteratorB(shared_storage.operand_B0_ref(), thread_idx),
        SmemIteratorB(shared_storage.operand_B1_ref(), thread_idx),
        SmemIteratorB(shared_storage.operand_B2_ref(), thread_idx),
        SmemIteratorB(shared_storage.operand_B3_ref(), thread_idx)
      },

      smem_iterator_presum_B_{
        SmemIteratorB(shared_storage.operand_B31_ref(), thread_idx),
        SmemIteratorB(shared_storage.operand_B10_ref(), thread_idx),
        SmemIteratorB(shared_storage.operand_S3_ref(), thread_idx),
        SmemIteratorB(shared_storage.operand_S3B2_ref(), thread_idx)
      },

      read_smem_iterator_B_{
        SmemIteratorB(shared_storage.operand_B0_ref(), thread_idx),
        SmemIteratorB(shared_storage.operand_B1_ref(), thread_idx),
        SmemIteratorB(shared_storage.operand_B2_ref(), thread_idx),
        SmemIteratorB(shared_storage.operand_B3_ref(), thread_idx)
      },

      read_smem_iterator_presum_B_{
        SmemIteratorB(shared_storage.operand_B31_ref(), thread_idx),
        SmemIteratorB(shared_storage.operand_B10_ref(), thread_idx),
        SmemIteratorB(shared_storage.operand_S3_ref(), thread_idx),
        SmemIteratorB(shared_storage.operand_S3B2_ref(), thread_idx)
      },

      smem_iterator_input_B_(shared_storage.operand_input_B_ref(), thread_idx),

      smem_iterator_A_M_{
        SmemIteratorA(shared_storage.operand_A_M0_ref(), thread_idx),
        SmemIteratorA(shared_storage.operand_A_M1_ref(), thread_idx),
        SmemIteratorA(shared_storage.operand_A_M2_ref(), thread_idx),
        SmemIteratorA(shared_storage.operand_A_M3_ref(), thread_idx),
        SmemIteratorA(shared_storage.operand_A_M4_ref(), thread_idx),
        SmemIteratorA(shared_storage.operand_A_M5_ref(), thread_idx),
        SmemIteratorA(shared_storage.operand_A_M6_ref(), thread_idx)
      },

      smem_iterator_B_M_{
        SmemIteratorB(shared_storage.operand_B_M0_ref(), thread_idx),
        SmemIteratorB(shared_storage.operand_B_M1_ref(), thread_idx),
        SmemIteratorB(shared_storage.operand_B_M2_ref(), thread_idx),
        SmemIteratorB(shared_storage.operand_B_M3_ref(), thread_idx),
        SmemIteratorB(shared_storage.operand_B_M4_ref(), thread_idx),
        SmemIteratorB(shared_storage.operand_B_M5_ref(), thread_idx),
        SmemIteratorB(shared_storage.operand_B_M6_ref(), thread_idx)
      },

      sharedPreSums(shared_storage.operand_presum(0), thread_idx),
      shared_operand_presum(shared_storage.operand_presum(0)),
      shared_postsum_op_iter(shared_postsum_op_iter),

      smem_write_stage_idx_(0),
      smem_read_stage_idx_(0)
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
    if (Base::RequireNoLocalPresum()) {
      this->warp_tile_iterator_input_A_.add_tile_offset({warp_idx_m, Base::kWarpGemmIterations * warp_idx_k});
      this->warp_tile_iterator_input_B_.add_tile_offset({Base::kWarpGemmIterations * warp_idx_k, warp_idx_n});
    } else {
      #pragma unroll 7
      for (int m = 0; m < 7; m++) {
        if (StrassenMiGroup::hasMi(m)) {
          this->warp_tile_iterator_A_M_[m].add_tile_offset({warp_idx_m, Base::kWarpGemmIterations * warp_idx_k});
          this->warp_tile_iterator_B_M_[m].add_tile_offset({Base::kWarpGemmIterations * warp_idx_k, warp_idx_n});
        }
      }
    }

    // if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
    //   if (hasM0()) printf("351 %d %d\n", StrassenMiGroup_::APresumLoads().index(0), StrassenMiGroup_::APresumStores(true).index(StrassenMiGroup_::APresums::A0)); 
    // }
  }

  /// Advance shared memory read-iterators to the next stage
  CUTLASS_DEVICE
  void advance_smem_read_stage()
  {
    ++smem_read_stage_idx_;

    if (smem_read_stage_idx_ == Base::kStages) {
      // Wrap back around to the 'start' of the circular buffer in shared memory
      // wrap read stage
      if (Base::RequireNoLocalPresum()) {
        this->warp_tile_iterator_input_A_.add_tile_offset(
          {0, -Base::kStages * Policy::kPartitionsK * Base::kWarpGemmIterations});
        this->warp_tile_iterator_input_B_.add_tile_offset(
          {-Base::kStages * Policy::kPartitionsK * Base::kWarpGemmIterations, 0});
      } else {
        #pragma unroll 7
        for (int m = 0; m < 7; m++) {
          if (StrassenMiGroup::hasMi(m)) {
            this->warp_tile_iterator_A_M_[m].add_tile_offset(
              {0, -Base::kStages * Policy::kPartitionsK * Base::kWarpGemmIterations});
            this->warp_tile_iterator_B_M_[m].add_tile_offset(
              {-Base::kStages * Policy::kPartitionsK * Base::kWarpGemmIterations, 0});
          }
        }
      }

      // smem_iterator_A03_.add_tile_offset({0, -Base::kStages});
      // smem_iterator_A23_.add_tile_offset({0, -Base::kStages});
      // // smem_iterator_A01_.add_tile_offset({0, -Base::kStages});
      // // smem_iterator_A20_.add_tile_offset({0, -Base::kStages});
      // smem_iterator_A13_.add_tile_offset({0, -Base::kStages});

      // smem_iterator_B03_.add_tile_offset({-Base::kStages, 0});
      // smem_iterator_B13_.add_tile_offset({-Base::kStages, 0});
      // // smem_iterator_B20_.add_tile_offset({-Base::kStages, 0});
      // // smem_iterator_B01_.add_tile_offset({-Base::kStages, 0});
      // smem_iterator_B23_.add_tile_offset({-Base::kStages, 0});

      smem_read_stage_idx_ = 0;
    }
  }

  /// Advance global memory read-iterators and shared memory write-iterators to the stage
  CUTLASS_DEVICE
  void advance_smem_write_stage(
    IteratorA iterator_Ais[4], IteratorB iterator_Bis[4],
    IteratorA iterator_PresumAs[4], IteratorB iterator_PresumBs[4],
    IteratorA& iterator_input_A, IteratorB& iterator_input_B)
  {
    // Advance global and shared iterators
    if (Base::RequireNoLocalPresum()) {
      iterator_input_A.add_tile_offset({0, 1});
      smem_iterator_input_A_.add_tile_offset({0, 1});

      iterator_input_B.add_tile_offset({1, 0});
      smem_iterator_input_B_.add_tile_offset({1, 0});
    } else {
      #pragma unroll 4
      for (int i = 0; i < 4; i++) {
        if (APresumLoads.hasAccess(StrassenMiGroup::APresums::A0, i)) {
          iterator_Ais[i].add_tile_offset({0, 1});
          smem_iterator_A_[i].add_tile_offset({0, 1});
        }
        if (BPresumLoads.hasAccess(StrassenMiGroup::BPresums::B0, i)) {
          iterator_Bis[i].add_tile_offset({1, 0});
          smem_iterator_B_[i].add_tile_offset({1, 0});
        }
      }

      #pragma unroll 4
      for (int i = 0; i < 4; i++) {
        if (APresumLoads.hasAccess(StrassenMiGroup::APresums::A02, i)) {
          iterator_PresumAs[i].add_tile_offset({0, 1});
          smem_iterator_presum_A_[i].add_tile_offset({0, 1});
        }
      }

      #pragma unroll 4
      for (int i = 0; i < 4; i++) {
        if (BPresumLoads.hasAccess(StrassenMiGroup::BPresums::B31, i)) {
          iterator_PresumBs[i].add_tile_offset({1, 0});
          smem_iterator_presum_B_[i].add_tile_offset({1, 0});
        }
      }
    }

    // Increment shared memory write stage index
    ++smem_write_stage_idx_;

    if (smem_write_stage_idx_ == Base::kStages) {
      // Wrap back around to the 'start' of the circular buffer in shared memory
      if (Base::RequireNoLocalPresum()) {
        smem_iterator_input_A_.add_tile_offset({0, -Base::kStages});
        smem_iterator_input_B_.add_tile_offset({-Base::kStages, 0});
      } else {
        #pragma unroll 4
        for (int i = 0; i < 4; i++) {
          if (APresumLoads.hasAccess(StrassenMiGroup::APresums::A0, i))
            smem_iterator_A_[i].add_tile_offset({0, -Base::kStages});
          if (BPresumLoads.hasAccess(StrassenMiGroup::BPresums::B0, i))
            smem_iterator_B_[i].add_tile_offset({-Base::kStages, 0});
        }
        
        #pragma unroll 4
        for (int i = 0; i < 4; i++) {
          if (APresumLoads.hasAccess(StrassenMiGroup::APresums::A02, i))
            smem_iterator_presum_A_[i].add_tile_offset({0, -Base::kStages});
          if (BPresumLoads.hasAccess(StrassenMiGroup::BPresums::B31, i))
            smem_iterator_presum_B_[i].add_tile_offset({-Base::kStages, 0});
        }
      }

      smem_write_stage_idx_ = 0;
    }
  }

  template<typename Iterator, typename SmemIterator>
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
            cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpA>(
              dst_ptr + v, iterator.get(), iterator.valid());
          } else {
            cutlass::arch::cp_async<kSrcBytes, kCacheOpA>(
              dst_ptr + v, iterator.get(), iterator.valid());
          }

          ++iterator;
        }

        ++smem_iterator;
      }
    }
  }

  CUTLASS_DEVICE
  void copy_tiles_and_advance(IteratorA iterator_Ais[4], IteratorB iterator_Bis[4],
                              IteratorA iterator_PresumAs[], IteratorB iterator_PresumBs[],
                              IteratorA& iterator_input_A, IteratorB& iterator_input_B,
                              int group_start_A = 0, int group_start_B = 0) {
    if (Base::RequireNoLocalPresum()) {
      iterator_input_A.set_iteration_index(group_start_A *
                                           IteratorA::kAccessesPerVector);
      this->smem_iterator_input_A_.set_iteration_index(group_start_A);
      cp_async_iterator(iterator_input_A, this->smem_iterator_input_A_,
                        Detail::AsyncCopyIterationsPerStageA, Detail::kAccessesPerGroupA, group_start_A);
      
      iterator_input_B.set_iteration_index(group_start_B *
                                              IteratorB::kAccessesPerVector);
      this->smem_iterator_input_B_.set_iteration_index(group_start_B);
      cp_async_iterator(iterator_input_B, this->smem_iterator_input_B_,
                      Detail::AsyncCopyIterationsPerStageB, Detail::kAccessesPerGroupB, group_start_B);
    } else {
      #pragma unroll 4
      for (int i = 0; i < 4; i++) {
        if (APresumLoads.hasAccess(StrassenMiGroup::APresums::A0, i)) {
          iterator_Ais[i].set_iteration_index(group_start_A *
                                              IteratorA::kAccessesPerVector);
          this->smem_iterator_A_[i].set_iteration_index(group_start_A);
          cp_async_iterator(iterator_Ais[i], this->smem_iterator_A_[i],
                            Detail::AsyncCopyIterationsPerStageA, Detail::kAccessesPerGroupA, group_start_A);
        }
        if (BPresumLoads.hasAccess(StrassenMiGroup::BPresums::B0, i)) {
          iterator_Bis[i].set_iteration_index(group_start_B *
                                              IteratorB::kAccessesPerVector);
          this->smem_iterator_B_[i].set_iteration_index(group_start_B);
          cp_async_iterator(iterator_Bis[i], this->smem_iterator_B_[i],
                          Detail::AsyncCopyIterationsPerStageB, Detail::kAccessesPerGroupB, group_start_B);
        }
      }

      #pragma unroll 4
      for (int i = 0; i < 4; i++) {
        if (APresumLoads.hasAccess(StrassenMiGroup::APresums::A02, i)) {
          iterator_PresumAs[i].set_iteration_index(group_start_A *
                                                  IteratorA::kAccessesPerVector);
          this->smem_iterator_presum_A_[i].set_iteration_index(group_start_A);
          cp_async_iterator(iterator_PresumAs[i], this->smem_iterator_presum_A_[i],
                            Detail::AsyncCopyIterationsPerStageA, Detail::kAccessesPerGroupA, group_start_A);
        }
      }

      #pragma unroll 4
      for (int i = 0; i < 4; i++) {
        if (BPresumLoads.hasAccess(StrassenMiGroup::BPresums::B31, i)) {
          iterator_PresumBs[i].set_iteration_index(group_start_B *
                                                  IteratorB::kAccessesPerVector);
          this->smem_iterator_presum_B_[i].set_iteration_index(group_start_B);
          cp_async_iterator(iterator_PresumBs[i], this->smem_iterator_presum_B_[i],
                            Detail::AsyncCopyIterationsPerStageB, Detail::kAccessesPerGroupB, group_start_B);
        }
      }
    }
  }

  /// GEMM prologue.  Bootstrap the global->shared memory pipeline by fetching
  /// the global fragments needed by the first kStages-1 threadblock mainloop iterations
  CUTLASS_DEVICE
  void prologue(
    IteratorA iterator_Ais[4], IteratorB iterator_Bis[4], ///< [in|out] iterator over A operand in global memory
    IteratorA iterator_PresumAs[4], IteratorB iterator_PresumBs[4],
    IteratorA& iterator_input_A, IteratorB& iterator_input_B,
    int &gemm_k_iterations,///< [in|out] number of threadblock mainloop iterations remaining
    PresumGlobalIteratorA& iter_PresumA, PresumGlobalIteratorB& iter_PresumB,
    PresumGlobalIteratorA& iter_PresumA_M, PresumGlobalIteratorB& iter_PresumB_M,
    PostsumSrcTileIterator source_interim_iterator)
  {
    // Issue several complete stages
    CUTLASS_PRAGMA_UNROLL
    for (int stage = 0; stage < Base::kStages - 1; ++stage, --gemm_k_iterations) {

      // Disable global fetching if done with global fetch iterations
      if (Base::RequireNoLocalPresum()) {
        iterator_input_A.clear_mask(gemm_k_iterations == 0);
        iterator_input_A.set_iteration_index(0);
        this->smem_iterator_input_A_.set_iteration_index(0);
        cp_async_iterator(iterator_input_A, this->smem_iterator_input_A_, Detail::AsyncCopyIterationsPerStageA, Detail::AsyncCopyIterationsPerStageA, 0);

        iterator_input_B.clear_mask(gemm_k_iterations == 0);
        iterator_input_B.set_iteration_index(0);
        this->smem_iterator_input_B_.set_iteration_index(0);
        cp_async_iterator(iterator_input_B, this->smem_iterator_input_B_, Detail::AsyncCopyIterationsPerStageB, Detail::AsyncCopyIterationsPerStageB, 0);

      } else {
        #pragma unroll 4
        for (int i = 0; i < 4; i++) {
          if (APresumLoads.hasAccess(StrassenMiGroup::APresums::A0, i)) {
            iterator_Ais[i].clear_mask(gemm_k_iterations == 0);
            iterator_Ais[i].set_iteration_index(0);
            this->smem_iterator_A_[i].set_iteration_index(0);
            cp_async_iterator(iterator_Ais[i], this->smem_iterator_A_[i], Detail::AsyncCopyIterationsPerStageA, Detail::AsyncCopyIterationsPerStageA, 0);
          }
        }

        #pragma unroll 4
        for (int i = 0; i < 4; i++) {
          if (APresumLoads.hasAccess(StrassenMiGroup::APresums::A02, i)) {
            iterator_PresumAs[i].clear_mask(gemm_k_iterations == 0);
            iterator_PresumAs[i].set_iteration_index(0);
            this->smem_iterator_presum_A_[i].set_iteration_index(0);
            cp_async_iterator(iterator_PresumAs[i], this->smem_iterator_presum_A_[i], Detail::AsyncCopyIterationsPerStageA, Detail::AsyncCopyIterationsPerStageA, 0);
          }
        }

        #pragma unroll 4
        for (int i = 0; i < 4; i++) {
          if (BPresumLoads.hasAccess(StrassenMiGroup::BPresums::B0, i)) {
            iterator_Bis[i].clear_mask(gemm_k_iterations == 0);
            iterator_Bis[i].set_iteration_index(0);
            this->smem_iterator_B_[i].set_iteration_index(0);
            cp_async_iterator(iterator_Bis[i], this->smem_iterator_B_[i], Detail::AsyncCopyIterationsPerStageB, Detail::AsyncCopyIterationsPerStageB, 0);
          }
        }

        #pragma unroll 4
        for (int i = 0; i < 4; i++) {
          if (BPresumLoads.hasAccess(StrassenMiGroup::BPresums::B31, i)) {
            iterator_PresumBs[i].clear_mask(gemm_k_iterations == 0);
            iterator_PresumBs[i].set_iteration_index(0);
            this->smem_iterator_presum_B_[i].set_iteration_index(0);
            cp_async_iterator(iterator_PresumBs[i], this->smem_iterator_presum_B_[i], Detail::AsyncCopyIterationsPerStageB, Detail::AsyncCopyIterationsPerStageB, 0);
          }
        }
      }

      auto presumAComputeLoads = StrassenMiGroup::AllPresums::APresumComputeLoads();
      auto presumBComputeLoads = StrassenMiGroup::AllPresums::BPresumComputeLoads();

      if (presumAComputeLoads.numAccess() > 0) {
        // if (StrassenMiGroup::hasM0() && StrassenMiGroup::Level == 1 &&
        //     threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0)
        //     printf("1632 %p %p %p %p\n", iter_PresumA.get(0), iter_PresumA.get(1), iter_PresumA.get(2), iter_PresumA.get(3));
        //Do loads needed for Computing Global Presums
        if (presumAComputeLoads.hasAccess(MmaStrassen::APresums::A0)) {
          PresumDetail::cp_async_presum(sharedPreSums.get(presumAComputeLoads.index(MmaStrassen::APresums::A0), stage, 0),
                                        iter_PresumA.get(0), iter_PresumA_M.validTB());
        }
        if (presumAComputeLoads.hasAccess(MmaStrassen::APresums::A1)) {
          PresumDetail::cp_async_presum(sharedPreSums.get(presumAComputeLoads.index(MmaStrassen::APresums::A1), stage, 0),
                                        iter_PresumA.get(1), iter_PresumA_M.validTB());
        }
        if (presumAComputeLoads.hasAccess(MmaStrassen::APresums::A2)) {
          PresumDetail::cp_async_presum(sharedPreSums.get(presumAComputeLoads.index(MmaStrassen::APresums::A2), stage, 0),
                                        iter_PresumA.get(2), iter_PresumA_M.validTB());
        }
        if (presumAComputeLoads.hasAccess(MmaStrassen::APresums::A3)) {
          PresumDetail::cp_async_presum(sharedPreSums.get(presumAComputeLoads.index(MmaStrassen::APresums::A3), stage, 0),
                                        iter_PresumA.get(3), iter_PresumA_M.validTB());
        }
        iter_PresumA.inc();
      } else if (presumBComputeLoads.numAccess() > 0) {
        if (presumBComputeLoads.hasAccess(MmaStrassen::BPresums::B0)) {
          PresumDetail::cp_async_presum(sharedPreSums.get(presumBComputeLoads.index(MmaStrassen::BPresums::B0), stage, 0),
                                        iter_PresumB.get(0), iter_PresumB_M.validTB());
        }
        if (presumBComputeLoads.hasAccess(MmaStrassen::BPresums::B1)) {
          PresumDetail::cp_async_presum(sharedPreSums.get(presumBComputeLoads.index(MmaStrassen::BPresums::B1), stage, 0),
                                        iter_PresumB.get(1), iter_PresumB_M.validTB());
        }
        if (presumBComputeLoads.hasAccess(MmaStrassen::BPresums::B2)) {
          PresumDetail::cp_async_presum(sharedPreSums.get(presumBComputeLoads.index(MmaStrassen::BPresums::B2), stage, 0),
                                        iter_PresumB.get(2), iter_PresumB_M.validTB());
        }
        if (presumBComputeLoads.hasAccess(MmaStrassen::BPresums::B3)) {
          PresumDetail::cp_async_presum(sharedPreSums.get(presumBComputeLoads.index(MmaStrassen::BPresums::B3), stage, 0),
                                        iter_PresumB.get(3), iter_PresumB_M.validTB());
        }
        iter_PresumB.inc();
      }

      if (source_interim_iterator.mask()) {
        for (int access = 0; access < 4; access++) {//TODO: why 4
          PresumDetail::cp_async_presum(shared_postsum_op_iter.smem_pointer(stage*4+access),
                                        source_interim_iterator.pointer(stage*4+access), true);
          // ++source_interim_iterator;
          // ++shared_postsum_op_iter;
        }
      }

      // Move to the next write stage
      advance_smem_write_stage(iterator_Ais, iterator_Bis,
                               iterator_PresumAs, iterator_PresumBs,
                               iterator_input_A, iterator_input_B);

      // Defines the boundary of a stage of cp.async.
      cutlass::arch::cp_async_fence();
    }

    if (source_interim_iterator.mask()) {
      for (int access = 0; access < 4; access++) {//TODO: why 4
        PresumDetail::cp_async_presum(shared_postsum_op_iter.smem_pointer(3*4+access),
                                      source_interim_iterator.pointer(3*4+access), true);
        // ++source_interim_iterator;
        // ++shared_postsum_op_iter;
      }
    }

    // Optionally clear the remaining stages of SMEM. This is a functional requirement for
    // some kernels so that all accumulator elements outside the GEMM footprint are zero.
    // if (SharedMemoryClear == SharedMemoryClearOption::kClearLastStage) {

    //   /// Iterator to write threadblock-scoped tile of A operand to shared memory
    //   SmemIteratorA last_smem_iterator_A(this->smem_iterator_A_);
    //   typename IteratorA::AccessType zero_A;

    //   zero_A.clear();
    //   last_smem_iterator_A.set_iteration_index(0);

    //   // Async Copy for operand A
    //   CUTLASS_PRAGMA_UNROLL
    //   for (int j = 0; j < Detail::AsyncCopyIterationsPerStageA; ++j) {

    //     typename IteratorA::AccessType *dst_ptr =
    //         reinterpret_cast<typename IteratorA::AccessType *>(
    //             last_smem_iterator_A.get());

    //     *dst_ptr = zero_A;

    //     ++last_smem_iterator_A;
    //   }

    //   /// Iterator to write threadblock-scoped tile of B operand to shared memory
    //   SmemIteratorB last_smem_iterator_B(this->smem_iterator_B_);
    //   typename IteratorB::AccessType zero_B;

    //   zero_B.clear();
    //   last_smem_iterator_B.set_iteration_index(0);

    //   // Async Copy for operand B
    //   CUTLASS_PRAGMA_UNROLL
    //   for (int j = 0; j < Detail::AsyncCopyIterationsPerStageB; ++j) {

    //     typename IteratorB::AccessType *dst_ptr =
    //         reinterpret_cast<typename IteratorB::AccessType *>(
    //             last_smem_iterator_B.get());

    //     *dst_ptr = zero_B;

    //     ++last_smem_iterator_B;
    //   }
    // }
  }



  /// Wait until we have at least one completed global fetch stage
  CUTLASS_DEVICE
  void gmem_wait(int gemm_k_iterations, int read_stage)
  {
    // Wait until we have at least one committed global fetch stage. (#uncommitted = Base::kStages - 1 - #committed)
    cutlass::arch::cp_async_wait<Base::kStages - 2>();
    if (Base::RequireNoLocalPresum()) return;

    if (sizeof(typename IteratorA::Element) == 2)
      return;
    else {
      // if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0)
      //   //FIXME: this return is needed because compiler generates this code when AllPresums are done
      //   printf("fix %s: %d\n", __FILE__, __LINE__);
    }
    // if (!(gemm_k_iterations > (-Base::kStages + 1))) return ;
    //////////////
    SmemIteratorA& read_smem_iterator_A0_1_ = read_smem_iterator_A_[0];
    // read_smem_iterator_A0_1_.add_tile_offset({0, read_stage});
    SmemIteratorA& read_smem_iterator_A1_1_ = read_smem_iterator_A_[1];
    // read_smem_iterator_A1_1_.add_tile_offset({0, read_stage});
    SmemIteratorA& read_smem_iterator_A2_1_ = read_smem_iterator_A_[2];
    // read_smem_iterator_A2_1_.add_tile_offset({0, read_stage});
    SmemIteratorA& read_smem_iterator_A3_1_ = read_smem_iterator_A_[3];
    // read_smem_iterator_A3_1_.add_tile_offset({0, read_stage});

    // SmemIteratorA& smem_iterator_S2_1_ = smem_iterator_S2_;
    // // smem_iterator_A03_1_.add_tile_offset({0, read_stage});
    // SmemIteratorA& smem_iterator_A02_1_ = smem_iterator_A02_;
    // // smem_iterator_A23_1_.add_tile_offset({0, Base::kStages});
    // // smem_iterator_A23_1_.add_tile_offset({0, read_stage});
    // SmemIteratorA& smem_iterator_S1_1_ = smem_iterator_S1_;
    // SmemIteratorA& smem_iterator_A1S2_1_ = smem_iterator_A1S2_;
    // // smem_iterator_A13_1_.add_tile_offset({0, read_stage});

    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Detail::AsyncCopyIterationsPerStageA; ++j) {
      typename IteratorA::AccessType __restrict__*  a0Ld =
          reinterpret_cast<typename IteratorA::AccessType *>(
              read_smem_iterator_A0_1_.get());
      typename IteratorA::AccessType __restrict__*  a1Ld =
          reinterpret_cast<typename IteratorA::AccessType *>(
              read_smem_iterator_A1_1_.get());
      typename IteratorA::AccessType __restrict__*  a2Ld =
          reinterpret_cast<typename IteratorA::AccessType *>(
              read_smem_iterator_A2_1_.get());
      typename IteratorA::AccessType __restrict__*  a3Ld =
          reinterpret_cast<typename IteratorA::AccessType *>(
              read_smem_iterator_A3_1_.get());
      typename IteratorA::AccessType __restrict__* a02Ld =
          reinterpret_cast<typename IteratorA::AccessType *>(
            read_smem_iterator_presum_A_[0].get());
      typename IteratorA::AccessType __restrict__* s1Ld =
          reinterpret_cast<typename IteratorA::AccessType *>(
            read_smem_iterator_presum_A_[1].get());
      typename IteratorA::AccessType __restrict__* s2Ld =
          reinterpret_cast<typename IteratorA::AccessType *>(
            read_smem_iterator_presum_A_[2].get());
      typename IteratorA::AccessType __restrict__* a1s2Ld =
          reinterpret_cast<typename IteratorA::AccessType *>(
            read_smem_iterator_presum_A_[3].get());

      typename IteratorA::AccessType __restrict__  *s2St =
        reinterpret_cast<typename IteratorA::AccessType *> (
          smem_iterator_A_M_[2].get());
      typename IteratorA::AccessType __restrict__  *a02St =
        reinterpret_cast<typename IteratorA::AccessType *> (
          smem_iterator_A_M_[3].get());
      typename IteratorA::AccessType __restrict__  *s1St =
        reinterpret_cast<typename IteratorA::AccessType *> (
          smem_iterator_A_M_[4].get());
      typename IteratorA::AccessType __restrict__  *a1s2St =
        reinterpret_cast<typename IteratorA::AccessType *> (
          smem_iterator_A_M_[5].get());

      typename IteratorA::AccessType a0, a1, a2, a3;

      if (APresumLoads.hasAccess(StrassenMiGroup::APresums::A0)) {
        a0 = *a0Ld;
        ++read_smem_iterator_A0_1_;
      }
      if (APresumLoads.hasAccess(StrassenMiGroup::APresums::A1)) {
        a1 = *a1Ld;
        ++read_smem_iterator_A1_1_;
      }
      if (APresumLoads.hasAccess(StrassenMiGroup::APresums::A2)) {
        a2 = *a2Ld;
        ++read_smem_iterator_A2_1_;
      }
      if (APresumLoads.hasAccess(StrassenMiGroup::APresums::A3)) {
        a3 = *a3Ld;
        ++read_smem_iterator_A3_1_;
      }

      typename IteratorA::AccessType a02, s1, s2, a1s2;

      if (APresumLoads.hasAccess(StrassenMiGroup::APresums::A02)) {
        a02 = *a02Ld;
        ++read_smem_iterator_presum_A_[0];
      } else {
        a02 = a0 - a2;
      }

      if (APresumLoads.hasAccess(StrassenMiGroup::APresums::S1)) {
        s1 = *s1Ld;
        ++read_smem_iterator_presum_A_[1];
      } else {
        s1 = a2+a3;
      }

      if (APresumLoads.hasAccess(StrassenMiGroup::APresums::S2)) {
        s2 = *s2Ld;
        ++read_smem_iterator_presum_A_[2];
      } else {
        s2 = s1 - a0;
      }

      if (APresumLoads.hasAccess(StrassenMiGroup::APresums::A1S2)) {
        a1s2 = *a1s2Ld;
        ++read_smem_iterator_presum_A_[3];
      } else {
        a1s2 = a1 - s2;
      }

      if (StrassenMiGroup::hasM3()) {
        // FIXME: If we enable this check then perf decreases
        // if (!APresumLoads.hasAccess(StrassenMiGroup::APresums::A02))
          *a02St = a02;
        ++smem_iterator_A_M_[3];
      }
      if (StrassenMiGroup::hasM2()) {
        // if (!APresumLoads.hasAccess(StrassenMiGroup::APresums::S2))
          *s2St = s2;
        ++smem_iterator_A_M_[2];
      }
      if (StrassenMiGroup::hasM4()) {
        // if (!APresumLoads.hasAccess(StrassenMiGroup::APresums::S1))
          *s1St = s1;
        ++smem_iterator_A_M_[4];
      }
      if (StrassenMiGroup::hasM5()) {
        // if (!APresumLoads.hasAccess(StrassenMiGroup::APresums::A1S2))
          *a1s2St = a1s2;
        ++smem_iterator_A_M_[5];
      }
    }

    SmemIteratorB& read_smem_iterator_B0_1_ = read_smem_iterator_B_[0];
    // read_smem_iterator_B0_1_.add_tile_offset({read_stage, 0});
    SmemIteratorB& read_smem_iterator_B1_1_ = read_smem_iterator_B_[1];
    // read_smem_iterator_B1_1_.add_tile_offset({read_stage, 0});
    SmemIteratorB& read_smem_iterator_B2_1_ = read_smem_iterator_B_[2];
    // read_smem_iterator_B2_1_.add_tile_offset({read_stage, 0});
    SmemIteratorB& read_smem_iterator_B3_1_ = read_smem_iterator_B_[3];
    // read_smem_iterator_B3_1_.add_tile_offset({read_stage, 0});

    // SmemIteratorB& smem_iterator_B10_1_ = smem_iterator_B10_;
    // // smem_iterator_B03_1_.add_tile_offset({read_stage, 0});
    // SmemIteratorB& smem_iterator_S3_1_ = smem_iterator_S3_;
    // // smem_iterator_B13_1_.add_tile_offset({read_stage, 0});
    // SmemIteratorB& smem_iterator_B31_1_ = smem_iterator_B31_;
    // SmemIteratorB& smem_iterator_S3B2_1_ = smem_iterator_S3B2_;
    // // smem_iterator_B23_1_.add_tile_offset({read_stage, 0});

    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Detail::AsyncCopyIterationsPerStageB; ++j) {
      typename IteratorB::AccessType __restrict__* b0Ld =
          reinterpret_cast<typename IteratorB::AccessType *>(
              read_smem_iterator_B0_1_.get());
      typename IteratorB::AccessType __restrict__* b1Ld =
          reinterpret_cast<typename IteratorB::AccessType *>(
              read_smem_iterator_B1_1_.get());
      typename IteratorB::AccessType __restrict__* b2Ld =
          reinterpret_cast<typename IteratorB::AccessType *>(
              read_smem_iterator_B2_1_.get());
      typename IteratorB::AccessType __restrict__* b3Ld =
          reinterpret_cast<typename IteratorB::AccessType *>(
              read_smem_iterator_B3_1_.get());
      typename IteratorB::AccessType* __restrict__ b10Ld = 
          reinterpret_cast<typename IteratorB::AccessType *>(
              read_smem_iterator_presum_B_[1].get());
      typename IteratorB::AccessType* __restrict__ s3Ld = 
          reinterpret_cast<typename IteratorB::AccessType *>(
              read_smem_iterator_presum_B_[2].get());
      typename IteratorB::AccessType* __restrict__ b31Ld = 
          reinterpret_cast<typename IteratorB::AccessType *>(
              read_smem_iterator_presum_B_[0].get());
      typename IteratorB::AccessType* __restrict__ s3b2Ld = 
          reinterpret_cast<typename IteratorB::AccessType *>(
              read_smem_iterator_presum_B_[3].get());

      typename IteratorB::AccessType __restrict__ *b10St =
          reinterpret_cast<typename IteratorB::AccessType *>(
              smem_iterator_B_M_[4].get());
      typename IteratorB::AccessType __restrict__ *s3St =
          reinterpret_cast<typename IteratorB::AccessType *>(
              smem_iterator_B_M_[2].get());
      typename IteratorB::AccessType __restrict__ *b31St =
          reinterpret_cast<typename IteratorB::AccessType *>(
              smem_iterator_B_M_[3].get());
      typename IteratorB::AccessType __restrict__ *s3b2St =
          reinterpret_cast<typename IteratorB::AccessType *>(
              smem_iterator_B_M_[6].get());
      
      typename IteratorB::AccessType b0, b1, b2, b3;

      if (BPresumLoads.hasAccess(StrassenMiGroup::BPresums::B0)) {
        b0 = *b0Ld;
        ++read_smem_iterator_B0_1_;
      }
      if (BPresumLoads.hasAccess(StrassenMiGroup::BPresums::B1)) {
        b1 = *b1Ld;
        ++read_smem_iterator_B1_1_;
      }
      if (BPresumLoads.hasAccess(StrassenMiGroup::BPresums::B2)) {
        b2 = *b2Ld;
        ++read_smem_iterator_B2_1_;
      }
      if (BPresumLoads.hasAccess(StrassenMiGroup::BPresums::B3)) {
        b3 = *b3Ld;
        ++read_smem_iterator_B3_1_;
      }

      typename IteratorB::AccessType b10, s3, b31, s3b2;

      if (BPresumLoads.hasAccess(StrassenMiGroup::BPresums::B31)) {
        b31 = *b31Ld;
        ++read_smem_iterator_presum_B_[0];
      } else {
        b31 = b3 - b1;
      }

      if (BPresumLoads.hasAccess(StrassenMiGroup::BPresums::B10)) {
        b10 = *b10Ld;
        ++read_smem_iterator_presum_B_[1];
      } else {
        b10 = b1 - b0;
      }

      if (BPresumLoads.hasAccess(StrassenMiGroup::BPresums::S3)) {
        s3 = *s3Ld;
        ++read_smem_iterator_presum_B_[2];
      } else {
        s3 = b31 + b0;
      }

      if (BPresumLoads.hasAccess(StrassenMiGroup::BPresums::S3B2)) {
        s3b2 = *s3b2Ld;
        ++read_smem_iterator_presum_B_[3];
      } else {
        s3b2 = s3 - b2;
      }

      if (StrassenMiGroup::hasM2()) {
        // if (!BPresumLoads.hasAccess(StrassenMiGroup::BPresums::S3))
        *s3St = s3;
        ++smem_iterator_B_M_[2];        
      }
      if (StrassenMiGroup::hasM3()) {
        // if (!BPresumLoads.hasAccess(StrassenMiGroup::BPresums::B31))
          *b31St = b31;
        ++smem_iterator_B_M_[3];
      }
      if (StrassenMiGroup::hasM4()) {
        // if (!BPresumLoads.hasAccess(StrassenMiGroup::BPresums::B10))
          *b10St = b10;
        ++smem_iterator_B_M_[4];
      }
      if (StrassenMiGroup::hasM6()) {
        // if (!BPresumLoads.hasAccess(StrassenMiGroup::BPresums::S3B2))
          *s3b2St = s3b2;
        ++smem_iterator_B_M_[6];
      }
    }

    // __syncthreads();
//     if (threadIdx.x == 0) {
//       typename IteratorB::AccessType __restrict__  *b3 = 
//         reinterpret_cast<typename IteratorB::AccessType *> (
//           read_smem_iterator_B3_.get());
//       for (int i = 0; i < 64*8; i++) {
//         printf("%d , %d, %d,%d: %f\n", gemm_k_iterations, i, i/128, i%128, b3[i][0]);
//       }
//     }

    #pragma unroll 4
    for (int i = 0; i < 4; i++) {
      if (APresumLoads.hasAccess(StrassenMiGroup::APresums::A0, i))
        read_smem_iterator_A_[i].add_tile_offset({0, 1});
    }

    #pragma unroll 4
    for (int i = 0; i < 4; i++) {
      if (APresumLoads.hasAccess(StrassenMiGroup::APresums::A02, i))
        read_smem_iterator_presum_A_[i].add_tile_offset({0, 1});
    }

    #pragma unroll 4
    for (int i = 0; i < 4; i++) {
      if (BPresumLoads.hasAccess(StrassenMiGroup::BPresums::B0, i))
        read_smem_iterator_B_[i].add_tile_offset({1, 0});
    }
    
    #pragma unroll 4
    for (int i = 0; i < 4; i++) {
      if (BPresumLoads.hasAccess(StrassenMiGroup::BPresums::B31, i))
        read_smem_iterator_presum_B_[i].add_tile_offset({1, 0});
    }

    #pragma unroll 7
    for (int m = 0; m < 7; m++) {
      if (StrassenMiGroup::hasMi(m)) {
        smem_iterator_A_M_[m].add_tile_offset({0, 1});
        smem_iterator_B_M_[m].add_tile_offset({1, 0});
      }
    }

    if (read_stage + 1 == Base::kStages) {
      #pragma unroll 4
      for (int i = 0; i < 4; i++) {
        if (APresumLoads.hasAccess(StrassenMiGroup::APresums::A0, i))
          read_smem_iterator_A_[i].add_tile_offset({0, -Base::kStages});
      }

      #pragma unroll 4
      for (int i = 0; i < 4; i++) {
        if (APresumLoads.hasAccess(StrassenMiGroup::APresums::A02, i))
          read_smem_iterator_presum_A_[i].add_tile_offset({0, -Base::kStages});
      }

      #pragma unroll 4
      for (int i = 0; i < 4; i++) {
        if (BPresumLoads.hasAccess(StrassenMiGroup::BPresums::B0, i))
          read_smem_iterator_B_[i].add_tile_offset({-Base::kStages, 0});
      }

      #pragma unroll 4
      for (int i = 0; i < 4; i++) {
        if (BPresumLoads.hasAccess(StrassenMiGroup::BPresums::B31, i))
          read_smem_iterator_presum_B_[i].add_tile_offset({-Base::kStages, 0});
      }

      #pragma unroll 7
      for (int m = 0; m < 7; m++) {
        if (StrassenMiGroup::hasMi(m)) {
          smem_iterator_A_M_[m].add_tile_offset({0, -Base::kStages});
          smem_iterator_B_M_[m].add_tile_offset({-Base::kStages, 0});
        }
      }
    }
  }

  template<typename WarpTileIterator, typename WarpLoadedFragment>
  CUTLASS_DEVICE
  void warp_tile_iterator_load(WarpTileIterator& warp_tile_iterator, uint warp_mma_k, WarpLoadedFragment& warp_loaded_frag) {
    warp_tile_iterator.set_kgroup_index((warp_mma_k) % Base::kWarpGemmIterations);
    warp_tile_iterator.load(warp_loaded_frag);
    ++warp_tile_iterator;
  }

  /// Perform a threadblock mainloop iteration of matrix multiply-accumulate
  CUTLASS_DEVICE
  void mac_loop_iter(
    PipeState &pipe_state,          ///< [in|out] loop-carried pipeline state
    FragmentC accumM[7],       ///< [in|out] destination accumulator tile
    IteratorA iterator_Ais[4], ///< [in|out] iterator over A operand in global memory
    IteratorB iterator_Bis[4], ///< [in|out] iterator over B operand in global memory
    IteratorA iterator_PresumAs[],
    IteratorB iterator_PresumBs[],
    IteratorA &iterator_input_A,
    IteratorB &iterator_input_B,
    int &gemm_k_iterations,         ///< [in|out] number of threadblock mainloop iterations remaining
    bool gemm_k_iter_decr)
  {
    // Unroll the warp-level MMA tiles of a threadblock's mainloop iteration
    CUTLASS_PRAGMA_UNROLL
    for (int warp_mma_k = 0; warp_mma_k < Base::kWarpGemmIterations; ++warp_mma_k) {
      if (Base::RequireNoLocalPresum()) {
        // Load the next warp-tile's A fragment from shared memory
        warp_tile_iterator_load(this->warp_tile_iterator_input_A_, warp_mma_k+1,
                                pipe_state.warp_loaded_frag_input_A_[(warp_mma_k + 1) % PipeState::WarpFrags]);
        // Load the next warp-tile's B fragment from shared memory
        warp_tile_iterator_load(this->warp_tile_iterator_input_B_, warp_mma_k+1,
                                pipe_state.warp_loaded_frag_input_B_[(warp_mma_k + 1) % PipeState::WarpFrags]);
        // if (StrassenMiGroup::hasM1() && StrassenMiGroup::Level == 1 && warp_mma_k == 0 &&
        //   threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0)
        //   printf("121 %f %f\n", float(pipe_state.warp_loaded_frag_input_A_[(warp_mma_k + 1) % PipeState::WarpFrags][0]),
        //                         float(pipe_state.warp_loaded_frag_input_B_[(warp_mma_k + 1) % PipeState::WarpFrags][0]));
      } else {
        #pragma unroll 7
        for (int m = 0; m < 7; m++) {
          if (StrassenMiGroup::hasMi(m)) {
            // Load the next warp-tile's A fragment from shared memory
            warp_tile_iterator_load(this->warp_tile_iterator_A_M_[m], warp_mma_k+1,
                                    pipe_state.warp_loaded_frag_A_M_[m][(warp_mma_k + 1) % PipeState::WarpFrags]);
            // Load the next warp-tile's B fragment from shared memory
            warp_tile_iterator_load(this->warp_tile_iterator_B_M_[m], warp_mma_k+1,
                                    pipe_state.warp_loaded_frag_B_M_[m][(warp_mma_k + 1) % PipeState::WarpFrags]);
          }
        }
      }

      // Except for the first warp-tile, all warp-tiles convert their incoming shared memory fragments as necessary
      if (warp_mma_k > 0) {
        if (Base::RequireNoLocalPresum()) {
          warp_mma_.transform(
            pipe_state.warp_transformed_frag_input_A_[warp_mma_k % PipeState::WarpFrags],
            pipe_state.warp_transformed_frag_input_B_[warp_mma_k % PipeState::WarpFrags],
            pipe_state.warp_loaded_frag_input_A_[warp_mma_k % PipeState::WarpFrags],
            pipe_state.warp_loaded_frag_input_B_[warp_mma_k % PipeState::WarpFrags]);
        } else {
          #pragma unroll 7
          for (int m = 0; m < 7; m++) {
            if (StrassenMiGroup::hasMi(m)) {
              warp_mma_.transform(
                pipe_state.warp_transformed_frag_A_M_[m][warp_mma_k % PipeState::WarpFrags],
                pipe_state.warp_transformed_frag_B_M_[m][warp_mma_k % PipeState::WarpFrags],
                pipe_state.warp_loaded_frag_A_M_[m][warp_mma_k % PipeState::WarpFrags],
                pipe_state.warp_loaded_frag_B_M_[m][warp_mma_k % PipeState::WarpFrags]);
            }
          }
        }
      }

      // Execute the current warp-tile of MMA operations
      if (Detail::kStagedAccumulation) {
        // warp_mma_(
        //   pipe_state.tmp_accum_,
        //   pipe_state.warp_transformed_frag_A_[warp_mma_k % PipeState::WarpFrags],
        //   pipe_state.warp_transformed_frag_B_[warp_mma_k % PipeState::WarpFrags],
        //   pipe_state.tmp_accum_
        // );

        // if (warp_mma_k == 0) {
        //   plus<FragmentC> plus_accum;
        //   accum = plus_accum(accum, pipe_state.tmp_accum_);
        //   pipe_state.tmp_accum_.clear();
        // }
      } else {
        // if (threadIdx.x == 0 and blockIdx.x == 0 and blockIdx.y == 0)
        // printf("970: %ld\n", m0.size());
        if (Base::RequireNoLocalPresum()) {
          const uint m = StrassenMiGroup::getMi();
          warp_mma_(
            accumM[m],
            pipe_state.warp_transformed_frag_input_A_[warp_mma_k % PipeState::WarpFrags],
            pipe_state.warp_transformed_frag_input_B_[warp_mma_k % PipeState::WarpFrags],
            accumM[m]
          );
        } else {
          #pragma unroll 7
          for (int m = 0; m < 7; m++) {
            if (StrassenMiGroup::hasMi(m)) {
              warp_mma_(
                accumM[m],
                pipe_state.warp_transformed_frag_A_M_[m][warp_mma_k % PipeState::WarpFrags],
                pipe_state.warp_transformed_frag_B_M_[m][warp_mma_k % PipeState::WarpFrags],
                accumM[m]
              );
            }
          }
        }
      }

      // Except for the last warp-tile, all warp-tiles issue their share of
      // global->shared fragment copies
      if (warp_mma_k < Base::kWarpGemmIterations - 1) {

        int group_start_iteration_A, group_start_iteration_B;
        group_start_iteration_A = warp_mma_k * Detail::kAccessesPerGroupA;
        group_start_iteration_B = warp_mma_k * Detail::kAccessesPerGroupB;

        copy_tiles_and_advance(
            iterator_Ais, iterator_Bis,
            iterator_PresumAs, iterator_PresumBs,
            iterator_input_A, iterator_input_B,
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
          iterator_Ais, iterator_Bis,
          iterator_PresumAs, iterator_PresumBs,
          iterator_input_A, iterator_input_B,
          group_start_iteration_A,
          group_start_iteration_B);

        // Inserts a memory fence between stages of cp.async instructions.
        cutlass::arch::cp_async_fence();

        // Wait until we have at least one completed global fetch stage
        gmem_wait(gemm_k_iterations - 1, (smem_read_stage_idx_+1) % Base::kStages);

        // Move to the next global fetch stage
        advance_smem_write_stage(iterator_Ais, iterator_Bis,
                                 iterator_PresumAs, iterator_PresumBs,
                                 iterator_input_A, iterator_input_B);
        advance_smem_read_stage();
        __syncthreads();
        
        if (gemm_k_iter_decr) {
          // Disable global fetching when done with global fetch iterations
          --gemm_k_iterations;
          if (Base::RequireNoLocalPresum()) {
            iterator_input_A.clear_mask(gemm_k_iterations == 0);
            iterator_input_B.clear_mask(gemm_k_iterations == 0);
          } else {
            #pragma unroll 4
            for (int i = 0; i < 4; i++) {
              if (APresumLoads.hasAccess(StrassenMiGroup::APresums::A0, i))
                iterator_Ais[i].clear_mask(gemm_k_iterations == 0);
              if (BPresumLoads.hasAccess(StrassenMiGroup::BPresums::B0, i))
                iterator_Bis[i].clear_mask(gemm_k_iterations == 0);
            }

            #pragma unroll 4
            for (int i = 0; i < 4; i++) {
              if (APresumLoads.hasAccess(StrassenMiGroup::APresums::A02, i))
                iterator_PresumAs[i].clear_mask(gemm_k_iterations == 0);
              if (BPresumLoads.hasAccess(StrassenMiGroup::BPresums::B31, i))
                iterator_PresumBs[i].clear_mask(gemm_k_iterations == 0);
            }
          }
        }
      }

      // The last warp-tile also converts the shared memory fragments used by
      // the first warp-tile of the next iteration, if necessary (so we can
      // immediately start issuing MMA instructions at the top of the loop )
      if (warp_mma_k + 1 == Base::kWarpGemmIterations) {
        if (Base::RequireNoLocalPresum()) {
          warp_mma_.transform(
            pipe_state.warp_transformed_frag_input_A_[(warp_mma_k+1) % PipeState::WarpFrags],
            pipe_state.warp_transformed_frag_input_B_[(warp_mma_k+1) % PipeState::WarpFrags],
            pipe_state.warp_loaded_frag_input_A_[(warp_mma_k+1) % PipeState::WarpFrags],
            pipe_state.warp_loaded_frag_input_B_[(warp_mma_k+1) % PipeState::WarpFrags]);
        } else {
          #pragma unroll 7
          for (int m = 0; m < 7; m++) {
            if (StrassenMiGroup::hasMi(m)) {
              warp_mma_.transform(
                pipe_state.warp_transformed_frag_A_M_[m][(warp_mma_k+1) % PipeState::WarpFrags],
                pipe_state.warp_transformed_frag_B_M_[m][(warp_mma_k+1) % PipeState::WarpFrags],
                pipe_state.warp_loaded_frag_A_M_[m][(warp_mma_k+1) % PipeState::WarpFrags],
                pipe_state.warp_loaded_frag_B_M_[m][(warp_mma_k+1) % PipeState::WarpFrags]);
            }
          }
        }
      }
    }
  }

  // CUTLASS_DEVICE
  // static PresumComputeVecType frag_presum_compute_type(PresumVecType& src) {
  //   if (not std::is_same<typename IteratorA::Element, half>::value) {
  //     PresumComputeVecType trg;
  //     for (int i = 0; i < 8; i += 1) {
  //       auto f2 = ElementC(src[i]);
  //       trg[i] = f2;
  //     }
  //     return trg;
  //   } else {
  //     PresumComputeVecType trg;
  //     for (int i = 0; i < 8; i += 2) {
  //       auto f2 = __half22float2(half2{(half)src[i], (half)src[i+1]});
  //       trg[i+0] =  f2.x; trg[i+1] = f2.y;
  //     }

  //     return trg;
  //   }
  // }

  // CUTLASS_DEVICE
  // static PresumVecType frag_presum_vec_type(PresumComputeVecType& src) {
  //   if (not std::is_same<typename IteratorA::Element, half>::value) {
  //     PresumComputeVecType trg;
  //     for (int i = 0; i < 8; i += 1) {
  //       auto f2 = typename IteratorA::Element(src[i]);
  //       trg[i] = f2;
  //     }
  //     return trg;
  //   } else {
  //     PresumVecType trg;
  //     for (int i = 0; i < 8; i+=2) {
  //       auto h2 = __float22half2_rn(float2{(ElementC)src[i+0], (ElementC)src[i+1]});
  //       trg[i/2] = h2;
  //     }
  //     return trg;
  //   }
  // }

  int presum_a_log_tile_multiplier = 0;
  int presum_b_log_tile_multiplier = 0;

  /// Perform the specified number of threadblock mainloop iterations of matrix
  /// multiply-accumulate.  Assumes prologue has been initiated.
  CUTLASS_DEVICE
  void gemm_iters(
    int gemm_k_iterations,        ///< number of threadblock mainloop iterations
    FragmentC accumM[7],
    IteratorA iterator_Ais[4],        ///< [in|out] iterator over A operand in global memory
    IteratorB iterator_Bis[4],        ///< [in|out] iterator over B operand in global memory
    IteratorA iterator_PresumAs[4],
    IteratorB iterator_PresumBs[4],
    IteratorA &iterator_input_A,
    IteratorB &iterator_input_B,
    PresumGlobalIteratorA& iter_PresumA, PresumGlobalIteratorB& iter_PresumB,
    PresumGlobalIteratorA& iter_PresumA_M, PresumGlobalIteratorB& iter_PresumB_M)
  {
    PipeState pipe_state;

    // Disable global fetching if done with global fetch iterations
    if (Base::RequireNoLocalPresum()) {
      iterator_input_A.clear_mask(gemm_k_iterations == 0);
      iterator_input_B.clear_mask(gemm_k_iterations == 0);
    } else {
      #pragma unroll 4
      for (int i = 0; i < 4; i++) {
        if (APresumLoads.hasAccess(StrassenMiGroup::APresums::A0, i))
          iterator_Ais[i].clear_mask(gemm_k_iterations == 0);
      }

      #pragma unroll 4
      for (int i = 0; i < 4; i++) {
        if (APresumLoads.hasAccess(StrassenMiGroup::APresums::A02, i))
          iterator_PresumAs[i].clear_mask(gemm_k_iterations == 0);
      }

      #pragma unroll 4
      for (int i = 0; i < 4; i++) {
        if (BPresumLoads.hasAccess(StrassenMiGroup::BPresums::B0, i))
          iterator_Bis[i].clear_mask(gemm_k_iterations == 0);
      }

      #pragma unroll 4
      for (int i = 0; i < 4; i++) {
        if (BPresumLoads.hasAccess(StrassenMiGroup::BPresums::B31, i))
          iterator_PresumBs[i].clear_mask(gemm_k_iterations == 0);
      }
    }

    if (Base::RequireNoLocalPresum()) {
      // Load the next warp-tile's A and B fragment from shared memory
      warp_tile_iterator_load(this->warp_tile_iterator_input_A_, 0,
                              pipe_state.warp_loaded_frag_input_A_[0]);
      warp_tile_iterator_load(this->warp_tile_iterator_input_B_, 0,
                              pipe_state.warp_loaded_frag_input_B_[0]);
      // Transform, if necessary, the first warp-tile's shared memory fragments
      warp_mma_.transform(
        pipe_state.warp_transformed_frag_input_A_[0],
        pipe_state.warp_transformed_frag_input_B_[0],
        pipe_state.warp_loaded_frag_input_A_[0],
        pipe_state.warp_loaded_frag_input_B_[0]);
    } else {
      #pragma unroll 7
      for (int m = 0; m < 7; m++) {
        if (StrassenMiGroup::hasMi(m)) {
          // Load the next warp-tile's A and B fragment from shared memory
          warp_tile_iterator_load(this->warp_tile_iterator_A_M_[m], 0,
                                  pipe_state.warp_loaded_frag_A_M_[m][0]);
          warp_tile_iterator_load(this->warp_tile_iterator_B_M_[m], 0,
                                  pipe_state.warp_loaded_frag_B_M_[m][0]);
          // Transform, if necessary, the first warp-tile's shared memory fragments
          warp_mma_.transform(
            pipe_state.warp_transformed_frag_A_M_[m][0],
            pipe_state.warp_transformed_frag_B_M_[m][0],
            pipe_state.warp_loaded_frag_A_M_[m][0],
            pipe_state.warp_loaded_frag_B_M_[m][0]);
        }
      }
    }

    // if (Detail::kStagedAccumulation) {
    //   pipe_state.tmp_accum_.clear();
    // }

    auto presumAComputeLoads = StrassenMiGroup::AllPresums::APresumComputeLoads();
    auto presumBComputeLoads = StrassenMiGroup::AllPresums::BPresumComputeLoads();
    const bool need_presum_A = presumAComputeLoads.numAccess() > 0;
    const bool need_presum_B = presumBComputeLoads.numAccess() > 0;
    const int presumComputeIterationsA = kPresumComputeIterationsA * (1<<presum_a_log_tile_multiplier);
    const int presumComputeIterationsB = kPresumComputeIterationsB * (1<<presum_b_log_tile_multiplier);

    int presum_tile = 0;
    // Mainloop
    if (need_presum_A || need_presum_B || StrassenMiGroup::FusedOrContinueMMA() == 1) {
      // int presum_a_or_b = 0; //0 for A and 1 for B
      if (gemm_k_iterations >= presumComputeIterationsA*need_presum_A + presumComputeIterationsB*need_presum_B) {
        int presumIter = 0;
        // int half_gemm_k_iterations = gemm_k_iterations;

        // int total_gemm_k_iterations = (gemm_k_iterations+Base::kStages - 1)*2 - Base::kStages+1;
        int total_gemm_k_iterations = gemm_k_iterations;
        // gemm_k_iterations = total_gemm_k_iterations;
        // if (StrassenMiGroup::hasM1() && StrassenMiGroup::Level == 1 &&
            // threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0)
            // printf("1666 %d %d %d\n", gemm_k_iterations, need_presum_B, presumComputeIterationsB);
        CUTLASS_GEMM_LOOP
        for (; gemm_k_iterations > (-Base::kStages + 1);) {
          //TODO: Fix for when there is only B
          if (presumIter < presumComputeIterationsA*1 - (Base::kStages - 1)) {
            presum_tile = (presumIter + Base::kStages - 1)/kPresumComputeIterationsA;
            //Improve presumA increment  4589f932b056b234808486d76a06839792b7c55e.
            //This provides slight improvement in 13kx13kx13k
            iter_PresumA.reset(presum_tile);
            iter_PresumA.set_iteration(presumIter+Base::kStages-1 - presum_tile*kPresumComputeIterationsA);
            uint presum_write_stage = (presumIter + Base::kStages - 1)% Base::kStages;
            if (presumAComputeLoads.hasAccess(MmaStrassen::APresums::A0)) {
              PresumDetail::cp_async_presum(sharedPreSums.get(presumAComputeLoads.index(MmaStrassen::APresums::A0), presum_write_stage, 0),
                                            iter_PresumA.get(0), iter_PresumA_M.validTB() && iter_PresumA_M.valid());
            }
            if (presumAComputeLoads.hasAccess(MmaStrassen::APresums::A1)) {
              PresumDetail::cp_async_presum(sharedPreSums.get(presumAComputeLoads.index(MmaStrassen::APresums::A1), presum_write_stage, 0),
                                            iter_PresumA.get(1), iter_PresumA_M.validTB() && iter_PresumA_M.valid());
            }
            if (presumAComputeLoads.hasAccess(MmaStrassen::APresums::A2)) {
              PresumDetail::cp_async_presum(sharedPreSums.get(presumAComputeLoads.index(MmaStrassen::APresums::A2), presum_write_stage, 0),
                                            iter_PresumA.get(2), iter_PresumA_M.validTB() && iter_PresumA_M.valid());
            }
            if (presumAComputeLoads.hasAccess(MmaStrassen::APresums::A3)) {
              PresumDetail::cp_async_presum(sharedPreSums.get(presumAComputeLoads.index(MmaStrassen::APresums::A3), presum_write_stage, 0),
                                            iter_PresumA.get(3), iter_PresumA_M.validTB() && iter_PresumA_M.valid());
            }

            // iter_PresumA.inc();
          } else if (need_presum_B and 
                     presumIter < presumComputeIterationsA + presumComputeIterationsB - (Base::kStages - 1)) {
            // iter_PresumB.reset();
            // iter_PresumB.row += (presumIter + Base::kStages - 1 - kPresumComputeIterations) * iter_PresumB_M.row_increment();
            uint presum_write_stage = (presumIter + Base::kStages - 1)% Base::kStages;
            presum_tile = (presumIter - presumComputeIterationsA + Base::kStages - 1)/kPresumComputeIterationsB;
            iter_PresumB.reset(presum_tile);
            iter_PresumB.set_iteration(presumIter - presumComputeIterationsA - presum_tile*kPresumComputeIterationsB + Base::kStages-1);
            if (presumBComputeLoads.hasAccess(MmaStrassen::BPresums::B0)) {
              PresumDetail::cp_async_presum(sharedPreSums.get(presumBComputeLoads.index(MmaStrassen::BPresums::B0), presum_write_stage, 0),
                                            iter_PresumB.get(0), iter_PresumB_M.validTB() && iter_PresumB_M.valid());
            }
            if (presumBComputeLoads.hasAccess(MmaStrassen::BPresums::B1)) {
              PresumDetail::cp_async_presum(sharedPreSums.get(presumBComputeLoads.index(MmaStrassen::BPresums::B1), presum_write_stage, 0),
                                            iter_PresumB.get(1), iter_PresumB_M.validTB() && iter_PresumB_M.valid());
            }
            if (presumBComputeLoads.hasAccess(MmaStrassen::BPresums::B2)) {
              PresumDetail::cp_async_presum(sharedPreSums.get(presumBComputeLoads.index(MmaStrassen::BPresums::B2), presum_write_stage, 0),
                                            iter_PresumB.get(2), iter_PresumB_M.validTB() && iter_PresumB_M.valid());
            }
            if (presumBComputeLoads.hasAccess(MmaStrassen::BPresums::B3)) {
              PresumDetail::cp_async_presum(sharedPreSums.get(presumBComputeLoads.index(MmaStrassen::BPresums::B3), presum_write_stage, 0),
                                            iter_PresumB.get(3), iter_PresumB_M.validTB() && iter_PresumB_M.valid());
            }
            iter_PresumB.inc();
          }

          //TODO: consider s1,s2,a02, etc. to compute global presum

          mac_loop_iter(
            pipe_state, accumM,
            iterator_Ais, iterator_Bis,
            iterator_PresumAs, iterator_PresumBs, 
            iterator_input_A, iterator_input_B,
            gemm_k_iterations, StrassenMiGroup::FusedOrContinueMMA() != 1);

            if (StrassenMiGroup::FusedOrContinueMMA() == 1)
              --gemm_k_iterations;
          
          if (presumIter < presumComputeIterationsA) {
            //This code above mac_loop_iter gives some improvement.
            //Changes done after commit: 853df006e0f2bfad3313460b2fcfdabb15d31067
            uint presum_read_stage = presumIter % Base::kStages;
            presum_tile = presumIter/kPresumComputeIterationsA;
            iter_PresumA_M.reset(presum_tile);
            iter_PresumA_M.set_iteration(presumIter - presum_tile * kPresumComputeIterationsA);
            if (iter_PresumA_M.validTB() && iter_PresumA_M.valid())
            for (int v = 0; v < 1; v += 1) {
              // iter_PresumA_M.reset();
              // iter_PresumA_M.row += presumIter * iter_PresumA_M.row_increment();

              PresumVecTypeA a0; a0.clear();
              PresumVecTypeA a1; a1.clear();
              PresumVecTypeA a2; a2.clear();
              PresumVecTypeA a3; a3.clear();

              if (presumAComputeLoads.hasAccess(MmaStrassen::APresums::A0))
                a0 = PresumDetail::shared_load_128b<PresumVecTypeA>(sharedPreSums.get(presumAComputeLoads.index(MmaStrassen::APresums::A0), presum_read_stage, 0));
              if (presumAComputeLoads.hasAccess(MmaStrassen::APresums::A1))
                a1 = PresumDetail::shared_load_128b<PresumVecTypeA>(sharedPreSums.get(presumAComputeLoads.index(MmaStrassen::APresums::A1), presum_read_stage, 0));
              if (presumAComputeLoads.hasAccess(MmaStrassen::APresums::A2))
                a2 = PresumDetail::shared_load_128b<PresumVecTypeA>(sharedPreSums.get(presumAComputeLoads.index(MmaStrassen::APresums::A2), presum_read_stage, 0));
              if (presumAComputeLoads.hasAccess(MmaStrassen::APresums::A3))
                a3 = PresumDetail::shared_load_128b<PresumVecTypeA>(sharedPreSums.get(presumAComputeLoads.index(MmaStrassen::APresums::A3), presum_read_stage, 0));

              PresumIOToComputeType presum_io_to_compute_type;
              PresumComputeToIOType presum_compute_to_io_type;

              auto s1   = presum_io_to_compute_type(a2) + presum_io_to_compute_type(a3);
              auto s2   = s1 - presum_io_to_compute_type(a0);
              auto a02  = presum_io_to_compute_type(a0) - presum_io_to_compute_type(a2);
              auto a1s2 = presum_io_to_compute_type(a1) - s2;
              // if (StrassenMiGroup::hasM0() && StrassenMiGroup::Level == 2 &&
              //     threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0)
              //     printf("1623 %d ; %f %f %f %f\n", iter_PresumA_M.stride, float(a0[0]), float(a1[0]), float(a2[0]), float(a3[0]));
              // auto s1   = a2 + a3;
              // auto s2   = s1 - a0;
              // auto a02  = a0 - a2;
              // auto a1s2 = a1 - s2;

              using AllPresums = typename StrassenMiGroup::AllPresums;

              if (AllPresums::doesComputeA(MmaStrassen::APresums::A02)) {
                arch::global_store<PresumVecTypeA, sizeof(PresumVecTypeA)>(presum_compute_to_io_type(a02), iter_PresumA_M.get(AllPresums::indexAPresum(MmaStrassen::APresums::A02)),
                                                                         iter_PresumA_M.validTB() && iter_PresumA_M.valid());
              }
              if (AllPresums::doesComputeA(MmaStrassen::APresums::S1)) {
                arch::global_store<PresumVecTypeA, sizeof(PresumVecTypeA)>(presum_compute_to_io_type(s1), iter_PresumA_M.get(AllPresums::indexAPresum(MmaStrassen::APresums::S1)),
                                                                         iter_PresumA_M.validTB() && iter_PresumA_M.valid());
              }
              if (AllPresums::doesComputeA(MmaStrassen::APresums::S2)) {
                arch::global_store<PresumVecTypeA, sizeof(PresumVecTypeA)>(presum_compute_to_io_type(s2), iter_PresumA_M.get(AllPresums::indexAPresum(MmaStrassen::APresums::S2)),
                                                                         iter_PresumA_M.validTB() && iter_PresumA_M.valid());
              }
              if (AllPresums::doesComputeA(MmaStrassen::APresums::A1S2)) {
                arch::global_store<PresumVecTypeA, sizeof(PresumVecTypeA)>(presum_compute_to_io_type(a1s2), iter_PresumA_M.get(AllPresums::indexAPresum(MmaStrassen::APresums::A1S2)),
                                                                         iter_PresumA_M.validTB() && iter_PresumA_M.valid());
              }

              iter_PresumA_M.inc();
            }
          } else if (need_presum_B and presumIter < presumComputeIterationsA + presumComputeIterationsB) {
            uint presum_read_stage = presumIter % Base::kStages;
            presum_tile = (presumIter - presumComputeIterationsA)/kPresumComputeIterationsB;
            iter_PresumB_M.reset(presum_tile);
            iter_PresumB_M.set_iteration(presumIter - presumComputeIterationsA - presum_tile * kPresumComputeIterationsB);
            if (iter_PresumB_M.validTB() and iter_PresumB_M.valid())
            for (int v = 0; v < 1; v += 1) {
              // iter_PresumB_M.reset();
              // iter_PresumB_M.row += (presumIter - kPresumComputeIterations) * iter_PresumB_M.row_increment();

              PresumVecTypeB b0; b0.clear();
              PresumVecTypeB b1; b1.clear();
              PresumVecTypeB b2; b2.clear();
              PresumVecTypeB b3; b3.clear();

              if (presumBComputeLoads.hasAccess(MmaStrassen::BPresums::B0))
                // PresumDetail::shared_load_128b(&b0, sharedPreSums.get(presumBComputeLoads.index(MmaStrassen::BPresums::B0), presum_read_stage, 0));
                b0 = PresumDetail::shared_load_128b<PresumVecTypeB>(sharedPreSums.get(presumBComputeLoads.index(MmaStrassen::BPresums::B0), presum_read_stage, 0));
              if (presumBComputeLoads.hasAccess(MmaStrassen::BPresums::B1))
                b1 = PresumDetail::shared_load_128b<PresumVecTypeB>(sharedPreSums.get(presumBComputeLoads.index(MmaStrassen::BPresums::B1), presum_read_stage, 0));
              if (presumBComputeLoads.hasAccess(MmaStrassen::BPresums::B2))
                b2 = PresumDetail::shared_load_128b<PresumVecTypeB>(sharedPreSums.get(presumBComputeLoads.index(MmaStrassen::BPresums::B2), presum_read_stage, 0));
              if (presumBComputeLoads.hasAccess(MmaStrassen::BPresums::B3))
                b3 = PresumDetail::shared_load_128b<PresumVecTypeB>(sharedPreSums.get(presumBComputeLoads.index(MmaStrassen::BPresums::B3), presum_read_stage, 0));

              PresumIOToComputeType presum_io_to_compute_type;
              PresumComputeToIOType presum_compute_to_io_type;

              auto b10  = presum_io_to_compute_type(b1) - presum_io_to_compute_type(b0);
              auto b31  = presum_io_to_compute_type(b3) - presum_io_to_compute_type(b1);
              auto s3   = b31 + presum_io_to_compute_type(b0);
              auto s3b2 = s3 - presum_io_to_compute_type(b2);

              using AllPresums = typename StrassenMiGroup::AllPresums;
                // if (StrassenMiGroup::hasM0() && StrassenMiGroup::Level == 2 &&
                //     iter_PresumB_M.tb_offset.column() == 0 && iter_PresumB_M.col == 0 && iter_PresumB_M.tb_offset.row() + iter_PresumB_M.row == 512)
                //     printf("1632 %p %d ; %d %d; %f %f %f %f\n", iter_PresumB_M.get(AllPresums::indexBPresum(MmaStrassen::BPresums::B31)),
                //     iter_PresumB_M.tb_offset.row() + iter_PresumB_M.row, iter_PresumB_M.stride, iter_PresumB_M.VectorLoadElems,
                //     float(b10[0]), float(b31[0]), float(s3[0]), float(s3b2[0]));
                  // printf("1632 %p\n", iter_PresumB_M.get(AllPresums::indexBPresum(MmaStrassen::BPresums::B31)));
              if (AllPresums::doesComputeB(MmaStrassen::BPresums::B10)) {
                arch::global_store<PresumVecTypeB, sizeof(PresumVecTypeB)>(presum_compute_to_io_type(b10), iter_PresumB_M.get(AllPresums::indexBPresum(MmaStrassen::BPresums::B10)),
                                                                         iter_PresumB_M.validTB() && iter_PresumB_M.valid());
              }
              if (AllPresums::doesComputeB(MmaStrassen::BPresums::B31)) {
                arch::global_store<PresumVecTypeB, sizeof(PresumVecTypeB)>(presum_compute_to_io_type(b31), iter_PresumB_M.get(AllPresums::indexBPresum(MmaStrassen::BPresums::B31)),
                                                                         iter_PresumB_M.validTB() && iter_PresumB_M.valid());
              }
              if (AllPresums::doesComputeB(MmaStrassen::BPresums::S3)) {
                arch::global_store<PresumVecTypeB, sizeof(PresumVecTypeB)>(presum_compute_to_io_type(s3), iter_PresumB_M.get(AllPresums::indexBPresum(MmaStrassen::BPresums::S3)),
                                                                         iter_PresumB_M.validTB() && iter_PresumB_M.valid());
              }
              if (AllPresums::doesComputeB(MmaStrassen::BPresums::S3B2)) {
                arch::global_store<PresumVecTypeB, sizeof(PresumVecTypeB)>(presum_compute_to_io_type(s3b2), iter_PresumB_M.get(AllPresums::indexBPresum(MmaStrassen::BPresums::S3B2)),
                                                                         iter_PresumB_M.validTB() && iter_PresumB_M.valid());
              }

              iter_PresumB_M.inc();
            }
          }
          presumIter++;
        }

        if (StrassenMiGroup::FusedOrContinueMMA() == 1) {
          if(true) {
            float4* ptr = (float4*)shared_operand_presum;
            // unsigned smem_int_ptr = cutlass::arch::cutlass_get_smem_pointer(ptr);

            for (int i = 0; i < 128; i+=8) {
              float* x = (float*)(&accumM[0][i]);
              half2 hx[4];
              for (int j = 0; j < 8; j+=2) {
                hx[j/2] = __float22half2_rn(float2{x[j],x[j+1]});
              }
              // ptr[(threadIdx.x + (i/8) * NumThreads)] = *(float4*)hx;
              arch::global_store<float4, sizeof(float4)>(*(float4*)hx, &ptr[(threadIdx.x + (i/8) * NumThreads)], true);
            }
          }

          gemm_k_iterations = total_gemm_k_iterations;

          CUTLASS_GEMM_LOOP
          for (; gemm_k_iterations > (-Base::kStages + 1);) {
            mac_loop_iter(
              pipe_state, accumM,
              iterator_Ais, iterator_Bis,
              iterator_PresumAs, iterator_PresumBs,
              iterator_input_A, iterator_input_B,
              gemm_k_iterations, true);
          }
        }
      } else __builtin_unreachable();
    } else {
      bool continueMMA = false; 
      if (continueMMA) {
        //Also requires removing "gemm_k_iterations = gemm_k_iterations/2" in strassen_gemm_kernel.h
        int half_gemm_k_iterations = (gemm_k_iterations + Base::kStages - 1)/2;

        // int total_gemm_k_iterations = gemm_k_iterations*2 + Base::kStages - 1;
        // int total_gemm_k_iterations = gemm_k_iterations;
        // gemm_k_iterations = total_gemm_k_iterations;

        CUTLASS_GEMM_LOOP
        for (; half_gemm_k_iterations > 0;) {

          mac_loop_iter(
            pipe_state, accumM,
            iterator_Ais, iterator_Bis,
            iterator_PresumAs, iterator_PresumBs,
            iterator_input_A, iterator_input_B,
            gemm_k_iterations, true);
          // --gemm_k_iterations;
          --half_gemm_k_iterations;


        }

                  //if (half_gemm_k_iterations == 0) {
          if (true) {
            float4* ptr = (float4*)shared_operand_presum;
            unsigned smem_int_ptr = cutlass::arch::cutlass_get_smem_pointer(ptr);
            
            for (int i = 0; i < 128; i += 8) {
              float* x = (float*)(&accumM[0][i]);
              half2 hx[4];
              for (int j = 0; j < 8; j+=2) {
                hx[j/2] = __float22half2_rn(float2{x[j],x[j+1]});
              }

              arch::shared_store<16>(smem_int_ptr + (threadIdx.x + (i/8) * 256)*16, hx);
            }
          }
          // if(half_gemm_k_iterations == -Base::kStages - 1) {
          // if (true) {
          //   float4* ptr = (float4*)shared_operand_presum;
          //   int i = 0;
          //   while (i < 128) {
          //     float* x = (float*)(&accumM[0][i]);
          //     half2 hx[4];
          //     for (int j = 0; j < 8; j+=2) {
          //       hx[j/2] = __float22half2_rn(float2{x[j],x[j+1]});
          //     }

          //     ptr[threadIdx.x + (i/8) * 256] = *(float4*)&hx;
          //     i += 8;
          //   }
          // }

        // gemm_k_iterations = total_gemm_k_iterations;

        CUTLASS_GEMM_LOOP
        for (; gemm_k_iterations > (-Base::kStages + 1);) {
          mac_loop_iter(
            pipe_state, accumM,
            iterator_Ais, iterator_Bis,
            iterator_PresumAs, iterator_PresumBs,
            iterator_input_A, iterator_input_B,
            gemm_k_iterations, true);
        }
      } else {
        CUTLASS_GEMM_LOOP
        for (; gemm_k_iterations > (-Base::kStages + 1);) {
          mac_loop_iter(
            pipe_state, accumM,
            iterator_Ais, iterator_Bis,
            iterator_PresumAs, iterator_PresumBs,
            iterator_input_A, iterator_input_B,
            gemm_k_iterations, true);
        }
      }
    }

    // if (Detail::kStagedAccumulation) {
    //   plus<FragmentC> plus_accum;
    //   accum = plus_accum(accum, pipe_state.tmp_accum_);
    // }

    // Commit and drain all pending and predicated cp.async pnz from the GEMM mainloop
    cutlass::arch::cp_async_fence();
    cutlass::arch::cp_async_wait<0>();
    // __syncthreads(); We do not need this syncthreads

  }


  /// Prepares the class for another prologue.
  CUTLASS_DEVICE
  void wind_down()
  {
    // // Catch-up the smem-read iterator to the smem-write iterator (so this class can be reused for another tile's prologue)

    // // First, increment remaining warp tiles to get to the next full stage.  (Ideally we would
    // // just decrement one tile, but not all iterators implement --() decrement.)
    // #pragma unroll
    // for (int warp_mma_k = 1; warp_mma_k < Base::kWarpGemmIterations; ++warp_mma_k)
    // {
    //   this->warp_tile_iterator_A_.set_kgroup_index(warp_mma_k);
    //   this->warp_tile_iterator_B_.set_kgroup_index(warp_mma_k);

    //   ++this->warp_tile_iterator_A_;
    //   ++this->warp_tile_iterator_B_;
    // }
    // smem_read_stage_idx_++;

    // // Then wrap back two full stages (one for the tile advancing we just did, and one to catch the write iterators)
    // static const int kStageIters = Policy::kPartitionsK * Base::kWarpGemmIterations;
    // if (smem_read_stage_idx_ > 1)
    // {
    //   this->warp_tile_iterator_A_.add_tile_offset({0, (-2 * kStageIters)});
    //   this->warp_tile_iterator_B_.add_tile_offset({(-2 * kStageIters), 0});
    // }
    // else
    // {
    //   this->warp_tile_iterator_A_.add_tile_offset({0, ((Base::kStages - 2) * kStageIters)});
    //   this->warp_tile_iterator_B_.add_tile_offset({((Base::kStages - 2) * kStageIters), 0});
    // }
    // smem_read_stage_idx_ = smem_write_stage_idx_;
  }


  CUTLASS_DEVICE
  IteratorA get_iterator_input_A(IteratorA iterator_Ais[], IteratorA iterator_presumAs[]) {
    if (StrassenMiGroup::numMs() > 1 || !Base::RequireNoLocalPresum())
      return iterator_Ais[0];
    auto ALoads = StrassenMiGroup_::APresumLoads();
    int input_idx = ALoads.get_first_access_idx();
    if (input_idx <= MmaStrassen::APresums::A3) {
      return iterator_Ais[input_idx];
    }
    return iterator_presumAs[input_idx - (MmaStrassen::APresums::A3 + 1)];    
  }

  CUTLASS_DEVICE
  IteratorB get_iterator_input_B(IteratorB iterator_Bis[], IteratorB iterator_presumBs[]) {
    if (StrassenMiGroup::numMs() > 1 || !Base::RequireNoLocalPresum())
      return iterator_Bis[0];
    auto BLoads = StrassenMiGroup_::BPresumLoads();
    int input_idx = BLoads.get_first_access_idx();
    if (input_idx <= MmaStrassen::BPresums::B3) {
      return iterator_Bis[input_idx];
    }
    return iterator_presumBs[input_idx - (MmaStrassen::BPresums::B3 + 1)];    
  }

  /// Perform a threadblock-scoped matrix multiply-accumulate
  CUTLASS_DEVICE
  void operator()(
      ///< problem size of GEMM
      int gemm_k_iterations,
      ///< destination accumulator tile
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
      PostsumSrcTileIterator source_interim_iterator) {

    // Prologue (start fetching iterations of global fragments into shared memory)
    IteratorA iterator_Ais[] = {iterator_A0, iterator_A1, iterator_A2, iterator_A3};
    IteratorB iterator_Bis[] = {iterator_B0, iterator_B1, iterator_B2, iterator_B3};
    IteratorA iterator_input_A = get_iterator_input_A(iterator_Ais, iterator_PresumAs);
    IteratorB iterator_input_B = get_iterator_input_B(iterator_Bis, iterator_PresumBs);
    // if (StrassenMiGroup::hasM1() && StrassenMiGroup::Level == 1 &&
    //               threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
    //                 float* ptr = (float*)iterator_input_B.get(0);
    //               // printf("1632 %f %f %f %f\n", float(b10[0]), float(b31[0]), float(s3[0]), float(s3b2[0]));
    //               printf("1936 %p %p %f %f\n", iterator_input_B.get(0), &ptr[0], ptr[1*1024], ptr[256*1024]);
    //               }

    prologue(iterator_Ais, iterator_Bis,
             iterator_PresumAs, iterator_PresumBs,
             iterator_input_A, iterator_input_B,
             gemm_k_iterations,
             iter_PresumA, iter_PresumB, iter_PresumA_M, iter_PresumB_M,
             source_interim_iterator);

    // Wait until we have at least one completed global fetch stage
    gmem_wait(gemm_k_iterations, 0);
    __syncthreads();

    // Initialize destination accumulators with source accumulators
    // Perform the MAC-iterations
    gemm_iters(gemm_k_iterations,
               accumM, 
               iterator_Ais, iterator_Bis,
               iterator_PresumAs, iterator_PresumBs,
              iterator_input_A, iterator_input_B,
               iter_PresumA, iter_PresumB, iter_PresumA_M, iter_PresumB_M);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace gemm
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////

