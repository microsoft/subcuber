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

#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cutlass/aligned_buffer.h"
#include "cutlass/numeric_conversion.h"

#include "cutlass/numeric_types.h"
#include "cutlass/matrix_shape.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/threadblock/strassen_mma_base.h"
#include "cutlass/gemm/threadblock/presum_detail.h"
#include "cutlass/epilogue/threadblock/strassen_interim_epilogue_tile_iterator.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Structure to compute the matrix product targeting CUDA cores and SIMT math instructions.
template <
  /// Size of the Gemm problem - concept: gemm::GemmShape<>
  typename Shape_,
  typename StrassenShape_,
  /// Iterates over tiles of A operand in global memory 
  //  (concept: ReadableTileIterator | ForwardTileIterator | MaskedTileIterator)
  typename IteratorA_,
  /// Iterates over tiles of A operand in shared memory
  /// (concept: WriteableTileIterator | RandomAccessTileIterator)
  typename SmemIteratorA_,
  /// Iterates over tiles of B operand in global memory
  //  (concept: ReadableTileIterator | ForwardTileIterator | MaskedTileIterator)
  typename IteratorB_,
  /// Iterates over tiles of B operand in shared memory
  /// (concept: WriteableTileIterator | RandomAccessTileIterator)
  typename SmemIteratorB_,
  /// Data type of accumulator matrix
  typename ElementC_,
  /// Data type of accumulator matrix
  typename LayoutC_,
  /// Policy describing tuning details (concept: MmaPolicy)
  typename Policy_,
  MmaStrassen::Type MmaStrassenKind,
  typename StrassenMiGroup_,
  /// Transformation applied to A operand
  typename TransformA_ = NumericArrayConverter<
    typename SmemIteratorA_::Element, 
    typename IteratorA_::Element, 
    IteratorA_::Fragment::kElements>,
  ///
  /// Transformation applied to B operand
  typename TransformB_ = NumericArrayConverter<
    typename SmemIteratorB_::Element, 
    typename IteratorB_::Element, 
    IteratorB_::Fragment::kElements>,
  /// Used for partial specialization
  typename Enable = bool
>
class StrassenMmaPipeline://<Shape_, StrassenShape_,IteratorA_, SmemIteratorA_, IteratorB_, SmemIteratorB_, ElementC_, LayoutC_, Policy_, MmaStrassenKind, StrassenMiGroup_, TransformA_, TransformB_, Enable> : 
  public StrassenMmaBase<Shape_, StrassenShape_, Policy_, 2, MmaStrassenKind, StrassenMiGroup_> {
public:
  ///< Base class
  using Base = StrassenMmaBase<Shape_, StrassenShape_, Policy_, 2, MmaStrassenKind, StrassenMiGroup_>;
  using StrassenMiGroup = StrassenMiGroup_;
  using Shape = Shape_;             ///< Size of the Gemm problem - concept: gemm::GemmShape<>
  using StrassenShape = StrassenShape_;
  using IteratorA = IteratorA_;     ///< Iterates over tiles of A operand in global memory
  using IteratorB = IteratorB_;     ///< Iterates over tiles of B operand in global memory
  using ElementC = ElementC_;       ///< Data type of accumulator matrix
  using LayoutC = LayoutC_;         ///< Layout of accumulator matrix
  using Policy = Policy_;           ///< Policy describing tuning details

  using SmemIteratorA = SmemIteratorA_;
  using SmemIteratorB = SmemIteratorB_;

  using TransformA = TransformA_;
  using TransformB = TransformB_;

  //
  // Dependent types
  //

  /// Fragment of operand A loaded from global memory
  using FragmentA = typename IteratorA::Fragment;

  /// Fragment of operand B loaded from global memory
  using FragmentB = typename IteratorB::Fragment;

  /// Fragment of accumulator tile
  using FragmentC = typename Policy::Operator::FragmentC;

  /// Warp-level Mma
  using Operator = typename Policy::Operator;

  /// Obtain the arch tag from the warp-level operator
  using ArchTag = typename Policy::Operator::ArchTag;

  /// Complex transform on A operand
  static ComplexTransform const kTransformA = Operator::kTransformA;

  /// Complex transform on B operand
  static ComplexTransform const kTransformB = Operator::kTransformB;

  // staticaly assert kStages for StrassenMmaPipeline is two (Double-buffered pipeline)
  static_assert((Base::kStages==2), "StrassenMmaPipeline requires kStages set to value 2");
  int presum_a_log_tile_multiplier;
  int presum_b_log_tile_multiplier;
  static const int Stages = 2;
  static const int NumThreads = Base::WarpCount::kCount * 32;
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
                                                            PresumShapeB, PresumVecTypeA, true, false>;

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
    float, //typename InterimEpilogueOp::ElementOutput,
    NumThreads,
    4//InterimEpilogueOp::kCount
  >;

  using PostsumSrcTileIterator = cutlass::epilogue::threadblock::StrassenInterimEpilogueTileIterator<
    StrassenShape_,
    float,
    float, //typename InterimEpilogueOp::ElementOutput,
    NumThreads,
    4 //InterimEpilogueOp::kCount
  >;

  SharedPostsumOpIterator shared_postsum_op_iter;

  PresumSharedIterator sharedPreSums;

protected:

  //
  // Data members
  //

  /// Warp-level MMA operator
  Operator warp_mma;

  /// Iterator to write threadblock-scoped tile of A operand to shared memory
  SmemIteratorA smem_iterator_A_M_[7];
  SmemIteratorA smem_iterator_input_A_;

  /// Iterator to write threadblock-scoped tile of B operand to shared memory
  SmemIteratorB smem_iterator_B_M_[7];
  SmemIteratorB smem_iterator_input_B_;

  ///< transformation applied to A fragment
  TransformA transform_A_;

  ///< transformation applied to B fragment
  TransformB transform_B_;

  /// Shared memory write stage index
  int smem_write_stage_idx;

public:
  typename Base::SharedStorage& shared_storage;
  typename IteratorA::Element* shared_operand_presum;

  /// Construct from tensor references
  CUTLASS_DEVICE
  StrassenMmaPipeline(
    typename Base::SharedStorage &shared_storage,       ///< Shared storage needed for internal use by threadblock-scoped GEMM
    int thread_idx,                                     ///< ID within the threadblock
    int warp_idx,                                       ///< ID of warp
    int lane_idx,                                       ///< ID of each thread within a warp
    SharedPostsumOpIterator& shared_postsum_op_iter,
    TransformA transform_A = TransformA(),              ///< transformation applied to A fragment
    TransformB transform_B = TransformB()               ///< transformation applied to B fragment
  ):
    shared_storage(shared_storage),
    Base(shared_storage, thread_idx, warp_idx, lane_idx),

    smem_iterator_A_M_ {
      SmemIteratorA(shared_storage.operand_A_M0_ref(), thread_idx),
      SmemIteratorA(shared_storage.operand_A_M1_ref(), thread_idx),
      SmemIteratorA(shared_storage.operand_A_M2_ref(), thread_idx),
      SmemIteratorA(shared_storage.operand_A_M3_ref(), thread_idx),
      SmemIteratorA(shared_storage.operand_A_M4_ref(), thread_idx),
      SmemIteratorA(shared_storage.operand_A_M5_ref(), thread_idx),
      SmemIteratorA(shared_storage.operand_A_M6_ref(), thread_idx),
    },

    smem_iterator_B_M_ {
      SmemIteratorB(shared_storage.operand_B_M0_ref(), thread_idx),
      SmemIteratorB(shared_storage.operand_B_M1_ref(), thread_idx),
      SmemIteratorB(shared_storage.operand_B_M2_ref(), thread_idx),
      SmemIteratorB(shared_storage.operand_B_M3_ref(), thread_idx),
      SmemIteratorB(shared_storage.operand_B_M4_ref(), thread_idx),
      SmemIteratorB(shared_storage.operand_B_M5_ref(), thread_idx),
      SmemIteratorB(shared_storage.operand_B_M6_ref(), thread_idx),
    },
    
    smem_iterator_input_A_(shared_storage.operand_input_A_ref(), thread_idx),
    smem_iterator_input_B_(shared_storage.operand_input_B_ref(), thread_idx),

    transform_A_(transform_A),
    transform_B_(transform_B),
    smem_write_stage_idx(0),
    sharedPreSums(nullptr, thread_idx),
    shared_operand_presum(shared_storage.operand_presum(0)),
    shared_postsum_op_iter(shared_postsum_op_iter)
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
    // if (threadIdx.x == 0 && blockIdx.x * blockIdx.y == 0)
    //   printf("Base::WarpCount::kM %d\n", Base::WarpCount::kM);
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
  }


  /// Advance shared memory write-iterators to the next stage
  CUTLASS_DEVICE
  void advance_smem_write_stage()
  {
    if (Base::RequireNoLocalPresum()) {
      ++this->smem_iterator_input_A_;
      ++this->smem_iterator_input_B_;
    } else {
      #pragma unroll 7
      for (int m = 0; m < 7; m++) {
        if (StrassenMiGroup::hasMi(m)) {
          ++this->smem_iterator_A_M_[m];
          ++this->smem_iterator_B_M_[m];
        }
      }
    }

    // Add negative offsets to return iterators to the 'start' of the circular buffer in shared memory
    if (smem_write_stage_idx == 1) {
      if (Base::RequireNoLocalPresum()) {
        this->smem_iterator_input_A_.add_tile_offset({0, -Base::kStages});
        this->smem_iterator_input_B_.add_tile_offset({-Base::kStages, 0});
      } else {
        #pragma unroll 7
        for (int m = 0; m < 7; m++) {
          if (StrassenMiGroup::hasMi(m)) {
            this->smem_iterator_A_M_[m].add_tile_offset({0, -Base::kStages});
            this->smem_iterator_B_M_[m].add_tile_offset({-Base::kStages, 0});
          }
        }
      }
    }

    smem_write_stage_idx ^= 1;
  }

  /// Advance shared memory read- and write-iterators to the next stage
  CUTLASS_DEVICE
  void advance_smem_stages()
  {
    if (Base::RequireNoLocalPresum()) {
      ++this->smem_iterator_input_A_;
      ++this->smem_iterator_input_B_;
    } else {
      #pragma unroll 7
      for (int m = 0; m < 7; m++) {
        if (StrassenMiGroup::hasMi(m)) {
          ++this->smem_iterator_A_M_[m];
          ++this->smem_iterator_B_M_[m];
        }
      }
    }

    // Add negative offsets to return iterators to the 'start' of the circular buffer in shared memory
    if (smem_write_stage_idx == 1) {
      if (Base::RequireNoLocalPresum()) {
        this->smem_iterator_input_A_.add_tile_offset({0, -Base::kStages});
        this->smem_iterator_input_B_.add_tile_offset({-Base::kStages, 0});
      } else {
        #pragma unroll 7
        for (int m = 0; m < 7; m++) {
          if (StrassenMiGroup::hasMi(m)) {
            this->smem_iterator_A_M_[m].add_tile_offset({0, -Base::kStages});
            this->smem_iterator_B_M_[m].add_tile_offset({-Base::kStages, 0});
          }
        }
      }
    } else {
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
    }

    smem_write_stage_idx ^= 1;
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

  template<typename Fragment, typename Element>
  CUTLASS_DEVICE
  void storeToMSh(Fragment& m, Element* data) {
    for (int i = 0; i < m.size(); i++) {
      data[i * blockDim.x + threadIdx.x] = m[i];
    }
  }

  template<typename Fragment, typename Element>
  CUTLASS_DEVICE
  void loadFromMSh(Fragment& m, Element* data) {
    for (int i = 0; i < m.size(); i++) {
      m[i] = data[i * blockDim.x + threadIdx.x];
    }
  }


  template<typename Element>
  CUTLASS_DEVICE
  void zeroMsh(Element* data, size_t sz) {
    for (int i = threadIdx.x; i < sz; i += blockDim.x) {
      data[i] = 0;
    }
  }

  CUTLASS_DEVICE
  void strassen_A(FragmentA& tb_frag_A0, FragmentA& tb_frag_A1, FragmentA& tb_frag_A2, FragmentA& tb_frag_A3,
                  FragmentA& tb_frag_S2, FragmentA& tb_frag_A02, FragmentA& tb_frag_S1, FragmentA& tb_frag_A1S2) {
    sum     (tb_frag_A2, tb_frag_A3, tb_frag_S1);
    subtract(tb_frag_S1, tb_frag_A0, tb_frag_S2);
    subtract(tb_frag_A0, tb_frag_A2, tb_frag_A02);
    subtract(tb_frag_A1, tb_frag_S2, tb_frag_A1S2);
  }

  CUTLASS_DEVICE
  void strassen_B(FragmentB& tb_frag_B0, FragmentB& tb_frag_B1, FragmentB& tb_frag_B2, FragmentB& tb_frag_B3,
                  FragmentB& tb_frag_S3, FragmentB& tb_frag_B31, FragmentB& tb_frag_B10, FragmentB& tb_frag_S3B2) {
    subtract(tb_frag_B1, tb_frag_B0, tb_frag_B10);
    subtract(tb_frag_B3, tb_frag_B10, tb_frag_S3);
    subtract(tb_frag_B3, tb_frag_B1, tb_frag_B31);
    subtract(tb_frag_S3, tb_frag_B2, tb_frag_S3B2);
  }

  /// GEMM prologue.  Bootstrap the global->shared memory pipeline by fetching
  /// the global fragments needed by the first kStages-1 threadblock mainloop iterations
  CUTLASS_DEVICE
  void prologue(
    IteratorA &iterator_A0,      ///< [in|out] iterator over A operand in global memory
    IteratorA &iterator_A1,
    IteratorA &iterator_A2,      ///< [in|out] iterator over A operand in global memory
    IteratorA &iterator_A3,
    IteratorB &iterator_B0,      ///< [in|out] iterator over B operand in global memory
    IteratorB &iterator_B1,
    IteratorB &iterator_B2,
    IteratorB &iterator_B3,
    IteratorA &iterator_input_A,
    IteratorB &iterator_input_B,
    int &gemm_k_iterations)     ///< [in|out] number of threadblock mainloop iterations remaining
  {
    // The last kblock is loaded in the prolog

    // Load A fragment from global A
    FragmentA tb_frag_input_A;
    FragmentA tb_frag_A0; tb_frag_A0.clear();
    FragmentA tb_frag_A1; tb_frag_A1.clear();
    FragmentA tb_frag_A2; tb_frag_A2.clear();
    FragmentA tb_frag_A3; tb_frag_A3.clear();
    FragmentA tb_frag_S2, tb_frag_A02, tb_frag_S1, tb_frag_A1S2;

    auto APresumLoads = StrassenMiGroup::APresumLoads();
    if (Base::RequireNoLocalPresum()) {
      iterator_input_A.load(tb_frag_input_A);
      ++iterator_input_A;
    } else {
      if (APresumLoads.hasAccess(0)) {
        iterator_A0.load(tb_frag_A0);
        ++iterator_A0;
      }

      if (APresumLoads.hasAccess(1)) {
        iterator_A1.load(tb_frag_A1);
        ++iterator_A1;
      }

      if (APresumLoads.hasAccess(2)) {
        iterator_A2.load(tb_frag_A2);
        ++iterator_A2;
      }

      if (APresumLoads.hasAccess(3)) {
        iterator_A3.load(tb_frag_A3);
        ++iterator_A3;
      }

      strassen_A(tb_frag_A0, tb_frag_A1, tb_frag_A2, tb_frag_A3,
                 tb_frag_S2, tb_frag_A02, tb_frag_S1, tb_frag_A1S2);
    }
    // Store A and B fragments to shared
    auto APresumStores = StrassenMiGroup::APresumStores();
    if (Base::RequireNoLocalPresum()) {
      this->smem_iterator_input_A_.store(transform_A_(tb_frag_input_A));
    } else {
      if (StrassenMiGroup::hasM0()) {
        this->smem_iterator_A_M_[0].store(transform_A_(tb_frag_A0));
      }
      if (StrassenMiGroup::hasM1()) {
        this->smem_iterator_A_M_[1].store(transform_A_(tb_frag_A1));
      }
      if (StrassenMiGroup::hasM2()) {
        this->smem_iterator_A_M_[2].store(transform_A_(tb_frag_S2));
      }
      if (StrassenMiGroup::hasM3()) {
        this->smem_iterator_A_M_[3].store(transform_A_(tb_frag_A02));
      }
      if (StrassenMiGroup::hasM4()) {
        this->smem_iterator_A_M_[4].store(transform_A_(tb_frag_S1));
      }
      if (StrassenMiGroup::hasM5()) {
        this->smem_iterator_A_M_[5].store(transform_A_(tb_frag_A1S2));
      }
      if (StrassenMiGroup::hasM6()) {
        this->smem_iterator_A_M_[6].store(transform_A_(tb_frag_A3));
      }
    }

    // Load B fragment from global B
    FragmentB tb_frag_input_B;
    FragmentB tb_frag_B0, tb_frag_B1, tb_frag_B2, tb_frag_B3;
    tb_frag_B0.clear(); tb_frag_B1.clear(); tb_frag_B2.clear(); tb_frag_B3.clear();
    
    if (Base::RequireNoLocalPresum()) {
      iterator_input_B.load(tb_frag_input_B);
      ++iterator_input_B;
      this->smem_iterator_input_B_.store(transform_B_(tb_frag_input_B));
    } else {
      auto BPresumLoads = StrassenMiGroup::BPresumLoads();

      if (BPresumLoads.hasAccess(0)) {
        iterator_B0.load(tb_frag_B0);
        ++iterator_B0;
      }

      if (BPresumLoads.hasAccess(1)) {
        iterator_B1.load(tb_frag_B1);
        ++iterator_B1;
      }

      if (BPresumLoads.hasAccess(2)) {
        iterator_B2.load(tb_frag_B2);
        ++iterator_B2;
      }

      if (BPresumLoads.hasAccess(3)) {
        iterator_B3.load(tb_frag_B3);
        ++iterator_B3;
      }
      
      FragmentB tb_frag_S3, tb_frag_B31, tb_frag_B10, tb_frag_S3B2;

      strassen_B(tb_frag_B0, tb_frag_B1, tb_frag_B2, tb_frag_B3,
                tb_frag_S3, tb_frag_B31, tb_frag_B10, tb_frag_S3B2);
      
      auto BPresumStores = StrassenMiGroup::BPresumStores();
      
      if (StrassenMiGroup::hasM0()) {
        this->smem_iterator_B_M_[0].store(transform_B_(tb_frag_B0));
      }
      if (StrassenMiGroup::hasM1()) {
        this->smem_iterator_B_M_[1].store(transform_B_(tb_frag_B2));
      }
      if (StrassenMiGroup::hasM2()) {
        this->smem_iterator_B_M_[2].store(transform_B_(tb_frag_S3));
      }
      if (StrassenMiGroup::hasM3()) {
        this->smem_iterator_B_M_[3].store(transform_B_(tb_frag_B31));
      }
      if (StrassenMiGroup::hasM4()) {
        this->smem_iterator_B_M_[4].store(transform_B_(tb_frag_B10));
      }
      if (StrassenMiGroup::hasM5()) {
        this->smem_iterator_B_M_[5].store(transform_B_(tb_frag_B3));
      }
      if (StrassenMiGroup::hasM6()) {
        this->smem_iterator_B_M_[6].store(transform_B_(tb_frag_S3B2));
      }
    }

    // Advance write stage
    advance_smem_write_stage();
  }

  /// Wait until we have at least one completed global fetch stage
  CUTLASS_DEVICE
  void gmem_wait()
  {
    __syncthreads();
  }


  /// Perform the specified number of threadblock mainloop iterations of matrix
  /// multiply-accumulate.  Assumes prologue has been initiated.
  CUTLASS_DEVICE
  void gemm_iters(
    // int part,
    int gemm_k_iterations,        ///< number of threadblock mainloop iterations
    FragmentC accumM[7],
    IteratorA &iterator_A0,        ///< [in|out] iterator over A operand in global memory
    IteratorA &iterator_A1,        ///< [in|out] iterator over A operand in global memory
    IteratorA &iterator_A2,        ///< [in|out] iterator over A operand in global memory
    IteratorA &iterator_A3,        ///< [in|out] iterator over A operand in global memory
    IteratorB &iterator_B0,        ///< [in|out] iterator over B operand in global memory
    IteratorB &iterator_B1,
    IteratorB &iterator_B2,
    IteratorB &iterator_B3,
    IteratorA &iterator_input_A,
    IteratorB &iterator_input_B)
  {
    using WarpFragmentA = typename Operator::FragmentA;
    using WarpFragmentB = typename Operator::FragmentB;

    // Pair of fragments used to overlap shared memory loads and math instructions
    WarpFragmentA warp_frag_A_M[7][2];
    WarpFragmentB warp_frag_B_M[7][2];
    WarpFragmentA warp_frag_input_A[2];
    WarpFragmentB warp_frag_input_B[2];

    auto APresumStores = StrassenMiGroup::APresumStores();
    auto BPresumStores = StrassenMiGroup::BPresumStores();

    // for (int i = 0; i < shared_storage.operand_B0.size(); i++) {
    //   if (threadIdx.x == 0)
    //     printf("i %d ; %f %f\n", i, shared_storage.operand_A0.data()[i], shared_storage.operand_B0.data()[i]);
    // }
    // Load A fragment from shared A
    if (Base::RequireNoLocalPresum()) {
      this->warp_tile_iterator_input_A_.set_kgroup_index(0);
      this->warp_tile_iterator_input_A_.load(warp_frag_input_A[0]);
      ++this->warp_tile_iterator_input_A_;
      this->warp_tile_iterator_input_B_.set_kgroup_index(0);
      this->warp_tile_iterator_input_B_.load(warp_frag_input_B[0]);
      ++this->warp_tile_iterator_input_B_;
    } else {
      #pragma unroll 7
      for (int m = 0; m < 7; m++) {
        if (StrassenMiGroup::hasMi(m)) {
          this->warp_tile_iterator_A_M_[m].set_kgroup_index(0);
          this->warp_tile_iterator_A_M_[m].load(warp_frag_A_M[m][0]);
          ++this->warp_tile_iterator_A_M_[m];
          this->warp_tile_iterator_B_M_[m].set_kgroup_index(0);
          this->warp_tile_iterator_B_M_[m].load(warp_frag_B_M[m][0]);
          ++this->warp_tile_iterator_B_M_[m];
        }
      }
    }

    // Pair of fragments used to overlap global memory loads and math instructions;
    FragmentA tb_frag_A0;
    FragmentA tb_frag_A1;
    FragmentA tb_frag_A2;
    FragmentA tb_frag_A3;
    FragmentA tb_frag_input_A;

    FragmentB tb_frag_B0;
    FragmentB tb_frag_B1;
    FragmentB tb_frag_B2;
    FragmentB tb_frag_B3;
    FragmentB tb_frag_input_B;

    // Avoid reading out of bounds
    iterator_A0.clear_mask(gemm_k_iterations <= 1);
    iterator_A1.clear_mask(gemm_k_iterations <= 1);
    iterator_A2.clear_mask(gemm_k_iterations <= 1);
    iterator_A3.clear_mask(gemm_k_iterations <= 1);
    iterator_input_A.clear_mask(gemm_k_iterations <= 1);

    iterator_B0.clear_mask(gemm_k_iterations <= 1);
    iterator_B1.clear_mask(gemm_k_iterations <= 1);
    iterator_B2.clear_mask(gemm_k_iterations <= 1);
    iterator_B3.clear_mask(gemm_k_iterations <= 1);
    iterator_input_B.clear_mask(gemm_k_iterations <= 1);

    // if (threadIdx.x == 0) printf("warp_frag_A_M0 %ld warp_frag_B_M0 %ld\n", warp_frag_A_M0[0].size(), warp_frag_B_M0[0].size());
    //
    // Mainloop
    //
    // if (threadIdx.x==0) printf("318 gemm_k_iterations %d Base::kWarpGemmIterations %d\n", gemm_k_iterations, Base::kWarpGemmIterations);
    // Note: The main loop does not support Base::kWarpGemmIterations == 2.
    CUTLASS_GEMM_LOOP
    for (; gemm_k_iterations > 0; --gemm_k_iterations) {
      //
      // Loop over GEMM K dimension
      //

      CUTLASS_PRAGMA_UNROLL
      for (int warp_mma_k = 0; warp_mma_k < Base::kWarpGemmIterations; ++warp_mma_k) {

        // Load warp-level tiles from shared memory, wrapping to k offset if this is the last group
        // as the case may be.

        if (warp_mma_k == Base::kWarpGemmIterations - 1) {

          // Write fragments to shared memory

          if (Base::RequireNoLocalPresum()) {
            this->smem_iterator_input_A_.store(transform_A_(tb_frag_input_A));
            this->smem_iterator_input_B_.store(transform_B_(tb_frag_input_B));
          } else {
            FragmentA tb_frag_S2, tb_frag_A02, tb_frag_S1, tb_frag_A1S2;

            strassen_A(tb_frag_A0, tb_frag_A1, tb_frag_A2, tb_frag_A3,
                      tb_frag_S2, tb_frag_A02, tb_frag_S1, tb_frag_A1S2);
            
            auto APresumStores = StrassenMiGroup::APresumStores();

            if (StrassenMiGroup::hasM0()) {
              this->smem_iterator_A_M_[0].store(transform_A_(tb_frag_A0));
            }
            if (StrassenMiGroup::hasM1()) {
              this->smem_iterator_A_M_[1].store(transform_A_(tb_frag_A1));
            }
            if (StrassenMiGroup::hasM2()) {
              this->smem_iterator_A_M_[2].store(transform_A_(tb_frag_S2));
            }
            if (StrassenMiGroup::hasM3()) {
              this->smem_iterator_A_M_[3].store(transform_A_(tb_frag_A02));
            }
            if (StrassenMiGroup::hasM4()) {
              this->smem_iterator_A_M_[4].store(transform_A_(tb_frag_S1));
            }
            if (StrassenMiGroup::hasM5()) {
              this->smem_iterator_A_M_[5].store(transform_A_(tb_frag_A1S2));
            }
            if (StrassenMiGroup::hasM6()) {
              this->smem_iterator_A_M_[6].store(transform_A_(tb_frag_A3));
            }


            FragmentB tb_frag_S3, tb_frag_B31, tb_frag_B10, tb_frag_S3B2;

            strassen_B(tb_frag_B0, tb_frag_B1, tb_frag_B2, tb_frag_B3,
                      tb_frag_S3, tb_frag_B31, tb_frag_B10, tb_frag_S3B2);
            
            auto BPresumStores = StrassenMiGroup::BPresumStores();
      
            if (StrassenMiGroup::hasM0()) {
              this->smem_iterator_B_M_[0].store(transform_B_(tb_frag_B0));
            }
            if (StrassenMiGroup::hasM1()) {
              this->smem_iterator_B_M_[1].store(transform_B_(tb_frag_B2));
            }
            if (StrassenMiGroup::hasM2()) {
              this->smem_iterator_B_M_[2].store(transform_B_(tb_frag_S3));
            }
            if (StrassenMiGroup::hasM3()) {
              this->smem_iterator_B_M_[3].store(transform_B_(tb_frag_B31));
            }
            if (StrassenMiGroup::hasM4()) {
              this->smem_iterator_B_M_[4].store(transform_B_(tb_frag_B10));
            }
            if (StrassenMiGroup::hasM5()) {
              this->smem_iterator_B_M_[5].store(transform_B_(tb_frag_B3));
            }
            if (StrassenMiGroup::hasM6()) {
              this->smem_iterator_B_M_[6].store(transform_B_(tb_frag_S3B2));
            }
          }

          // Wait until we have at least one completed global fetch stage
          gmem_wait();

          // Advance smem read and write stages
          advance_smem_stages();
        }

        if (Base::RequireNoLocalPresum()) {
          this->warp_tile_iterator_input_A_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations);
          this->warp_tile_iterator_input_B_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations);
          this->warp_tile_iterator_input_A_.load(warp_frag_input_A[(warp_mma_k + 1) % 2]);
          this->warp_tile_iterator_input_B_.load(warp_frag_input_B[(warp_mma_k + 1) % 2]);
          ++this->warp_tile_iterator_input_A_;
          ++this->warp_tile_iterator_input_B_;
        } else {
          #pragma unroll 7
          for (int m = 0; m < 7; m++) {
            if (StrassenMiGroup::hasMi(m)) {
              this->warp_tile_iterator_A_M_[m].set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations);
              this->warp_tile_iterator_B_M_[m].set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations);
              this->warp_tile_iterator_A_M_[m].load(warp_frag_A_M[m][(warp_mma_k + 1) % 2]);
              this->warp_tile_iterator_B_M_[m].load(warp_frag_B_M[m][(warp_mma_k + 1) % 2]);
              ++this->warp_tile_iterator_A_M_[m];
              ++this->warp_tile_iterator_B_M_[m];
            }
          }
        }

        if (warp_mma_k == 0) {
          if (Base::RequireNoLocalPresum()) {
            iterator_input_A.load(tb_frag_input_A);
            iterator_input_B.load(tb_frag_input_B);
            ++iterator_input_A;
            ++iterator_input_B;
            iterator_input_A.clear_mask(gemm_k_iterations <= 2);
            iterator_input_B.clear_mask(gemm_k_iterations <= 2);
          } else {
            auto APresumLoads = StrassenMiGroup::APresumLoads();
            if (APresumLoads.hasAccess(0)) {
              iterator_A0.load(tb_frag_A0);
              ++iterator_A0;
            }

            if (APresumLoads.hasAccess(1)) {
              iterator_A1.load(tb_frag_A1);
              ++iterator_A1;
            }

            if (APresumLoads.hasAccess(2)) {
              iterator_A2.load(tb_frag_A2);
              ++iterator_A2;
            }

            if (APresumLoads.hasAccess(3)) {
              iterator_A3.load(tb_frag_A3);
              ++iterator_A3;
            }

            // Load fragment from global B
            auto BPresumLoads = StrassenMiGroup::BPresumLoads();

            if (BPresumLoads.hasAccess(0)) {
              iterator_B0.load(tb_frag_B0);
              ++iterator_B0;
            }

            if (BPresumLoads.hasAccess(1)) {
              iterator_B1.load(tb_frag_B1);
              ++iterator_B1;
            }

            if (BPresumLoads.hasAccess(2)) {
              iterator_B2.load(tb_frag_B2);
              ++iterator_B2;
            }

            if (BPresumLoads.hasAccess(3)) {
              iterator_B3.load(tb_frag_B3);
              ++iterator_B3;
            }

            // Avoid reading out of bounds if this was the last loop iteration
            iterator_A0.clear_mask(gemm_k_iterations <= 2);
            iterator_A1.clear_mask(gemm_k_iterations <= 2);
            iterator_A2.clear_mask(gemm_k_iterations <= 2);
            iterator_A3.clear_mask(gemm_k_iterations <= 2);

            iterator_B0.clear_mask(gemm_k_iterations <= 2);
            iterator_B1.clear_mask(gemm_k_iterations <= 2);
            iterator_B2.clear_mask(gemm_k_iterations <= 2);
            iterator_B3.clear_mask(gemm_k_iterations <= 2);
          }
        }

        if (Base::RequireNoLocalPresum()) {
          warp_mma(accumM[StrassenMiGroup::getMi()],
              warp_frag_input_A[warp_mma_k % 2],
              warp_frag_input_B[warp_mma_k % 2],
              accumM[StrassenMiGroup::getMi()]);
        } else {
          #pragma unroll 7
          for (int m = 0; m < 7; m++) {
            if (StrassenMiGroup::hasMi(m)) {
              warp_mma(accumM[m],
              warp_frag_A_M[m][warp_mma_k % 2],
              warp_frag_B_M[m][warp_mma_k % 2],
              accumM[m]);
            }
          }
        }
      }

    }

    if (threadIdx.x == 0) {
      // for (int i = 0; i < 1; i++) {
      //   printf("m0: %d : %f\n", i, accumM[0][i]);
      // }

      // for (int i = 0; i < m3.size(); i++) {
      //   printf("m3: %d : %f\n", i, m3[i]);
      // }

      // for (int i = 0; i < m4.size(); i++) {
      //   printf("m4: %d : %f\n", i, m4[i]);
      // }

      // for (int i = 0; i < m6.size(); i++) {
      //   printf("m6: %d : %f\n", i, m6[i]);
      // }
    }

    // if (threadIdx.x == 0) {
    //   for (int i = 0; i < accum00.size(); i++) {
    //     printf("497: %d, %d : %f \n", threadIdx.x, i, accum00[i]);
    //   }
    // }
    // __syncthreads();
  }


  /// Prepares the class for another prologue.
  CUTLASS_DEVICE
  void wind_down()
  {
    // assert(false);
    // // First, increment remaining warp tiles to catch it up with the write stage.
    // #pragma unroll
    // for (int warp_mma_k = 1; warp_mma_k < Base::kWarpGemmIterations; ++warp_mma_k)
    // {
    //   this->warp_tile_iterator_A_M_[0].set_kgroup_index(warp_mma_k);
    //   this->warp_tile_iterator_A_M_[2].set_kgroup_index(warp_mma_k);
    //   this->warp_tile_iterator_B_M_[0].set_kgroup_index(warp_mma_k);
    //   this->warp_tile_iterator_B_M_[1].set_kgroup_index(warp_mma_k);

    //   ++this->warp_tile_iterator_A_M_[0];
    //   ++this->warp_tile_iterator_A_M_[2];
    //   ++this->warp_tile_iterator_B_M_[0];
    //   ++this->warp_tile_iterator_B_M_[1];
    // }

    // // If we bumped the read iterators to the end of the circular buffer, wrap them around to
    // // align them with the write iterators
    // if (smem_write_stage_idx == 0)
    // {
    //   this->warp_tile_iterator_A_M_[0].add_tile_offset(
    //     {0, -Base::kStages * Policy::kPartitionsK * Base::kWarpGemmIterations});
    //   this->warp_tile_iterator_A_M_[2].add_tile_offset(
    //     {0, -Base::kStages * Policy::kPartitionsK * Base::kWarpGemmIterations});
    //   this->warp_tile_iterator_B_M_[0].add_tile_offset(
    //     {-Base::kStages * Policy::kPartitionsK * Base::kWarpGemmIterations, 0});
    //   this->warp_tile_iterator_B_M_[1].add_tile_offset(
    //     {-Base::kStages * Policy::kPartitionsK * Base::kWarpGemmIterations, 0});
    // }
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
    IteratorA iterator_Ais[] = {iterator_A0, iterator_A1, iterator_A2, iterator_A3};
    IteratorA iterator_input_A = get_iterator_input_A(iterator_Ais, iterator_PresumAs);
    IteratorB iterator_Bis[] = {iterator_B0, iterator_B1, iterator_B2, iterator_B3};
    IteratorB iterator_input_B = get_iterator_input_B(iterator_Bis, iterator_PresumBs);

    // Prologue
    prologue(iterator_A0, iterator_A1, iterator_A2, iterator_A3, 
             iterator_B0, iterator_B1, iterator_B2, iterator_B3,
             iterator_input_A, iterator_input_B,
             gemm_k_iterations);

    // Wait until we have at least one completed global fetch stage
    gmem_wait();

    // // // Perform accumulation in the 'd' output operand

    // // // // Perform the MAC-iterations
    gemm_iters(gemm_k_iterations,
               accumM, 
               iterator_A0, iterator_A1, iterator_A2, iterator_A3,
               iterator_B0, iterator_B1, iterator_B2, iterator_B3,
               iterator_input_A, iterator_input_B);
  }

};

/////////////////////////////////////////////////////////////////////////////////////////////////

#include "cutlass/gemm/threadblock/strassen_mma_pipelined_compressed.h"

} // namespace threadblock
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
