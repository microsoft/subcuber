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
#include "cutlass/gemm/threadblock/global_strassen_mma_base.h"

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
  typename MmaStrassenConsts,
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
class GlobalStrassenMmaPipeline: 
  public GlobalStrassenMmaBase<Shape_, StrassenShape_, Policy_, 2, MmaStrassenKind, StrassenMiGroup_, MmaStrassenConsts> {
public:
  ///< Base class
  using Base = GlobalStrassenMmaBase<Shape_, StrassenShape_, Policy_, 2, MmaStrassenKind, StrassenMiGroup_, MmaStrassenConsts>;

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

  // staticaly assert kStages for GlobalStrassenMmaPipeline is two (Double-buffered pipeline)
  static_assert((Base::kStages==2), "GlobalStrassenMmaPipeline requires kStages set to value 2");

protected:

  //
  // Data members
  //

  /// Warp-level MMA operator
  Operator warp_mma;

  /// Iterator to write threadblock-scoped tile of A operand to shared memory
  SmemIteratorA smem_iterator_A_;

  /// Iterator to write threadblock-scoped tile of B operand to shared memory
  SmemIteratorB smem_iterator_B_;

  ///< transformation applied to A fragment
  TransformA transform_A_;

  ///< transformation applied to B fragment
  TransformB transform_B_;

  /// Shared memory write stage index
  int smem_write_stage_idx;

public:

  /// Construct from tensor references
  CUTLASS_DEVICE
  GlobalStrassenMmaPipeline(
    typename Base::SharedStorage &shared_storage,       ///< Shared storage needed for internal use by threadblock-scoped GEMM
    int thread_idx,                                     ///< ID within the threadblock
    int warp_idx,                                       ///< ID of warp
    int lane_idx,                                       ///< ID of each thread within a warp
    TransformA transform_A = TransformA(),              ///< transformation applied to A fragment
    TransformB transform_B = TransformB()               ///< transformation applied to B fragment
  ):
    Base(shared_storage, thread_idx, warp_idx, lane_idx),
    smem_iterator_A_(shared_storage.operand_A_ref(), thread_idx),
    smem_iterator_B_(shared_storage.operand_B_ref(), thread_idx),
    transform_A_(transform_A),
    transform_B_(transform_B),
    smem_write_stage_idx(0)
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
    this->warp_tile_iterator_A_.add_tile_offset({warp_idx_m, Base::kWarpGemmIterations * warp_idx_k});
    this->warp_tile_iterator_B_.add_tile_offset({Base::kWarpGemmIterations * warp_idx_k, warp_idx_n});
  }


  /// Advance shared memory write-iterators to the next stage
  CUTLASS_DEVICE
  void advance_smem_write_stage()
  {
    ++this->smem_iterator_A_;
    ++this->smem_iterator_B_;

    // Add negative offsets to return iterators to the 'start' of the circular buffer in shared memory
    if (smem_write_stage_idx == 1) {
      this->smem_iterator_A_.add_tile_offset({0, -Base::kStages});
      this->smem_iterator_B_.add_tile_offset({-Base::kStages, 0});
    }

    smem_write_stage_idx ^= 1;
  }

  /// Advance shared memory read- and write-iterators to the next stage
  CUTLASS_DEVICE
  void advance_smem_stages()
  {
    ++this->smem_iterator_A_;
    ++this->smem_iterator_B_;

    // Add negative offsets to return iterators to the 'start' of the circular buffer in shared memory
    if (smem_write_stage_idx == 1) {
      // wrap write stage
      this->smem_iterator_A_.add_tile_offset({0, -Base::kStages});
      this->smem_iterator_B_.add_tile_offset({-Base::kStages, 0});
    }
    else
    {
      // wrap read stage
      this->warp_tile_iterator_A_.add_tile_offset(
        {0, -Base::kStages * Policy::kPartitionsK * Base::kWarpGemmIterations});
      this->warp_tile_iterator_B_.add_tile_offset(
        {-Base::kStages * Policy::kPartitionsK * Base::kWarpGemmIterations, 0});
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

  CUTLASS_DEVICE
  void memory_load(IteratorA& iterator_A0,      ///< [in|out] iterator over A operand in global memory
                  IteratorA& iterator_A1,
                  IteratorA& iterator_A2,
                  IteratorA& iterator_A3,
                  IteratorB& iterator_B0,     ///< [in|out] iterator over B operand in global memory
                  IteratorB& iterator_B1,
                  IteratorB& iterator_B2,
                  IteratorB& iterator_B3,
                  FragmentA& tb_frag_A,
                  FragmentB& tb_frag_B) {
    if (StrassenMiGroup_::hasM0()) {
      // Load A fragment from global A
      FragmentA tb_frag_A0;
      tb_frag_A0.clear();
      iterator_A0.load(tb_frag_A0);
      ++iterator_A0;

      FragmentA tb_frag_A3;
      tb_frag_A3.clear();
      iterator_A3.load(tb_frag_A3);
      ++iterator_A3;

      sum(tb_frag_A0, tb_frag_A3, tb_frag_A);

      // Load B fragment from global B
      FragmentB tb_frag_B0;
      tb_frag_B0.clear();
      iterator_B0.load(tb_frag_B0);
      ++iterator_B0;

      FragmentB tb_frag_B3;
      tb_frag_B3.clear();
      iterator_B3.load(tb_frag_B3);
      ++iterator_B3;

      sum(tb_frag_B0, tb_frag_B3, tb_frag_B);
    } else if (StrassenMiGroup_::hasM1()) {
      // Load A fragment from global A
      FragmentA tb_frag_A2;
      tb_frag_A2.clear();
      iterator_A2.load(tb_frag_A2);
      ++iterator_A2;

      FragmentA tb_frag_A3;
      tb_frag_A3.clear();
      iterator_A3.load(tb_frag_A3);
      ++iterator_A3;

      sum(tb_frag_A2, tb_frag_A3, tb_frag_A);

      // Load B fragment from global B
      iterator_B0.load(tb_frag_B);
      ++iterator_B0;
    } else if (StrassenMiGroup_::hasM2()) {
      // Load A fragment from global A
      iterator_A0.load(tb_frag_A);
      ++iterator_A0;

      // Load B fragment from global B
      FragmentB tb_frag_B1;
      tb_frag_B1.clear();
      iterator_B1.load(tb_frag_B1);
      ++iterator_B1;

      FragmentB tb_frag_B3;
      tb_frag_B3.clear();
      iterator_B3.load(tb_frag_B3);
      ++iterator_B3;

      subtract(tb_frag_B1, tb_frag_B3, tb_frag_B);
    } else if (StrassenMiGroup_::hasM3()) {
      // Load A fragment from global A
      iterator_A3.load(tb_frag_A);
      ++iterator_A3;

      // Load B fragment from global B
      FragmentB tb_frag_B0;
      tb_frag_B0.clear();
      iterator_B0.load(tb_frag_B0);
      ++iterator_B0;

      FragmentB tb_frag_B2;
      tb_frag_B2.clear();
      iterator_B2.load(tb_frag_B2);
      ++iterator_B2;

      subtract(tb_frag_B2, tb_frag_B0, tb_frag_B);
    } else if (StrassenMiGroup_::hasM4()) {
      // Load A fragment from global A
      FragmentA tb_frag_A0;
      tb_frag_A0.clear();
      iterator_A0.load(tb_frag_A0);
      ++iterator_A0;

      FragmentA tb_frag_A1;
      tb_frag_A1.clear();
      iterator_A1.load(tb_frag_A1);
      ++iterator_A1;

      sum(tb_frag_A0, tb_frag_A1, tb_frag_A);

      // Load B fragment from global B
      iterator_B3.load(tb_frag_B);
      ++iterator_B3;
    } else if (StrassenMiGroup_::hasM5()) {
      // Load A fragment from global A
      FragmentA tb_frag_A0;
      tb_frag_A0.clear();
      iterator_A0.load(tb_frag_A0);
      ++iterator_A0;

      FragmentA tb_frag_A2;
      tb_frag_A2.clear();
      iterator_A2.load(tb_frag_A2);
      ++iterator_A2;

      subtract(tb_frag_A2, tb_frag_A0, tb_frag_A);

      // Load B fragment from global B
      FragmentB tb_frag_B0;
      tb_frag_B0.clear();
      iterator_B0.load(tb_frag_B0);
      ++iterator_B0;

      FragmentB tb_frag_B1;
      tb_frag_B1.clear();
      iterator_B1.load(tb_frag_B1);
      ++iterator_B1;

      sum(tb_frag_B0, tb_frag_B1, tb_frag_B);
    } else if (StrassenMiGroup_::hasM6()) {
      // Load A fragment from global A
      FragmentA tb_frag_A1;
      tb_frag_A1.clear();
      iterator_A1.load(tb_frag_A1);
      ++iterator_A1;

      FragmentA tb_frag_A3;
      tb_frag_A3.clear();
      iterator_A3.load(tb_frag_A3);
      ++iterator_A3;

      subtract(tb_frag_A1, tb_frag_A3, tb_frag_A);

      // Load B fragment from global B
      FragmentB tb_frag_B2;
      tb_frag_B2.clear();
      iterator_B2.load(tb_frag_B2);
      ++iterator_B2;

      FragmentB tb_frag_B3;
      tb_frag_B3.clear();
      iterator_B3.load(tb_frag_B3);
      ++iterator_B3;

      sum(tb_frag_B2, tb_frag_B3, tb_frag_B);
    }
  }

  CUTLASS_DEVICE
  void pre_sum(FragmentA& tb_frag1_A, FragmentA& tb_frag2_A,
                    FragmentA& tb_frag_A,
                    FragmentB& tb_frag1_B, FragmentB& tb_frag2_B,
                    FragmentB& tb_frag_B) {
    if (StrassenMiGroup_::hasM0()) {
      sum(tb_frag1_A, tb_frag2_A, tb_frag_A);
      sum(tb_frag1_B, tb_frag2_B, tb_frag_B);
    } else if (StrassenMiGroup_::hasM1()) {
      // Load A fragment from global A
      sum(tb_frag1_A, tb_frag2_A, tb_frag_A);

      // Load B fragment from global B
      tb_frag_B = tb_frag1_B;
    } else if (StrassenMiGroup_::hasM2()) {
      // Load A fragment from global A
      tb_frag_A = tb_frag1_A;

      subtract(tb_frag1_B, tb_frag2_B, tb_frag_B);
    } else if (StrassenMiGroup_::hasM3()) {
      // Load A fragment from global A
      tb_frag_A = tb_frag1_A;

      subtract(tb_frag2_B, tb_frag1_B, tb_frag_B);
    } else if (StrassenMiGroup_::hasM4()) {
      sum(tb_frag1_A, tb_frag2_A, tb_frag_A);

      tb_frag_B = tb_frag1_B;
    } else if (StrassenMiGroup_::hasM5()) {
      subtract(tb_frag2_A, tb_frag1_A, tb_frag_A);

      sum(tb_frag1_B, tb_frag2_B, tb_frag_B);
    } else if (StrassenMiGroup_::hasM6()) {
      subtract(tb_frag1_A, tb_frag2_A, tb_frag_A);
      sum(tb_frag1_B, tb_frag2_B, tb_frag_B);
    }
  }

  CUTLASS_DEVICE
  void prefetch_load(IteratorA& iterator_A0,      ///< [in|out] iterator over A operand in global memory
                  IteratorA& iterator_A1,
                  IteratorA& iterator_A2,
                  IteratorA& iterator_A3,
                  IteratorB& iterator_B0,     ///< [in|out] iterator over B operand in global memory
                  IteratorB& iterator_B1,
                  IteratorB& iterator_B2,
                  IteratorB& iterator_B3,
                  FragmentA& tb_frag1_A, FragmentA& tb_frag2_A,
                  FragmentB& tb_frag1_B, FragmentB& tb_frag2_B) {
    if (StrassenMiGroup_::hasM0()) {
      // Load A fragment from global A
      FragmentA tb_frag_A0;
      tb_frag_A0.clear();
      iterator_A0.load(tb_frag_A0);
      ++iterator_A0;

      FragmentA tb_frag_A3;
      tb_frag_A3.clear();
      iterator_A3.load(tb_frag_A3);
      ++iterator_A3;

      tb_frag1_A = tb_frag_A0;
      tb_frag2_A = tb_frag_A3;

      // Load B fragment from global B
      FragmentB tb_frag_B0;
      tb_frag_B0.clear();
      iterator_B0.load(tb_frag_B0);
      ++iterator_B0;

      FragmentB tb_frag_B3;
      tb_frag_B3.clear();
      iterator_B3.load(tb_frag_B3);
      ++iterator_B3;

      tb_frag1_B = tb_frag_B0;
      tb_frag2_B = tb_frag_B3;

    } else if (StrassenMiGroup_::hasM1()) {
      // Load A fragment from global A
      FragmentA tb_frag_A2;
      tb_frag_A2.clear();
      iterator_A2.load(tb_frag_A2);
      ++iterator_A2;

      FragmentA tb_frag_A3;
      tb_frag_A3.clear();
      iterator_A3.load(tb_frag_A3);
      ++iterator_A3;

      tb_frag1_A = tb_frag_A2;
      tb_frag2_A = tb_frag_A3;

      // Load B fragment from global B
      iterator_B0.load(tb_frag1_B);
      ++iterator_B0;
    } else if (StrassenMiGroup_::hasM2()) {
      // Load A fragment from global A
      iterator_A0.load(tb_frag1_A);
      ++iterator_A0;

      // Load B fragment from global B
      tb_frag1_B.clear();
      iterator_B1.load(tb_frag1_B);
      ++iterator_B1;

      tb_frag2_B.clear();
      iterator_B3.load(tb_frag2_B);
      ++iterator_B3;
    } else if (StrassenMiGroup_::hasM3()) {
      // Load A fragment from global A
      iterator_A3.load(tb_frag1_A);
      ++iterator_A3;

      // Load B fragment from global B
      iterator_B0.load(tb_frag1_B);
      ++iterator_B0;

      iterator_B2.load(tb_frag2_B);
      ++iterator_B2;
    } else if (StrassenMiGroup_::hasM4()) {
      // Load A fragment from global A
      iterator_A0.load(tb_frag1_A);
      ++iterator_A0;

      iterator_A1.load(tb_frag2_A);
      ++iterator_A1;

      // Load B fragment from global B
      iterator_B3.load(tb_frag1_B);
      ++iterator_B3;
    } else if (StrassenMiGroup_::hasM5()) {
      // Load A fragment from global A
      iterator_A0.load(tb_frag1_A);
      ++iterator_A0;

      iterator_A2.load(tb_frag2_A);
      ++iterator_A2;

      // Load B fragment from global B
      iterator_B0.load(tb_frag1_B);
      ++iterator_B0;

      iterator_B1.load(tb_frag2_B);
      ++iterator_B1;
    } else if (StrassenMiGroup_::hasM6()) {
      // Load A fragment from global A
      iterator_A1.load(tb_frag1_A);
      ++iterator_A1;

      iterator_A3.load(tb_frag2_A);
      ++iterator_A3;

      // Load B fragment from global B
      iterator_B2.load(tb_frag1_B);
      ++iterator_B2;

      iterator_B3.load(tb_frag2_B);
      ++iterator_B3;
    }
  }

  /// GEMM prologue.  Bootstrap the global->shared memory pipeline by fetching
  /// the global fragments needed by the first kStages-1 threadblock mainloop iterations
  CUTLASS_DEVICE
  void prologue(
    IteratorA& iterator_A0,      ///< [in|out] iterator over A operand in global memory
    IteratorA& iterator_A1,
    IteratorA& iterator_A2,
    IteratorA& iterator_A3,
    IteratorB& iterator_B0,     ///< [in|out] iterator over B operand in global memory
    IteratorB& iterator_B1,
    IteratorB& iterator_B2,
    IteratorB& iterator_B3,
    int &gemm_k_iterations)     ///< [in|out] number of threadblock mainloop iterations remaining
  {
    // The last kblock is loaded in the prolog

    FragmentA tb_frag_A;
    tb_frag_A.clear();
    FragmentB tb_frag_B;
    tb_frag_B.clear();

    memory_load(iterator_A0, iterator_A1, iterator_A2, iterator_A3, 
                iterator_B0, iterator_B1, iterator_B2, iterator_B3, tb_frag_A, tb_frag_B);
    
    // Store A and B fragments to shared
    this->smem_iterator_A_.store(transform_A_(tb_frag_A));
    this->smem_iterator_B_.store(transform_B_(tb_frag_B));

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
    int gemm_k_iterations,        ///< number of threadblock mainloop iterations
    FragmentC &accum,             ///< [in|out] accumulator tile
    IteratorA& iterator_A0,      ///< [in|out] iterator over A operand in global memory
    IteratorA& iterator_A1,
    IteratorA& iterator_A2,
    IteratorA& iterator_A3,
    IteratorB& iterator_B0,     ///< [in|out] iterator over B operand in global memory
    IteratorB& iterator_B1,
    IteratorB& iterator_B2,
    IteratorB& iterator_B3)        ///< [in|out] iterator over B operand in global memory
  {
    using WarpFragmentA = typename Operator::FragmentA;
    using WarpFragmentB = typename Operator::FragmentB;

    // Pair of fragments used to overlap shared memory loads and math instructions
    WarpFragmentA warp_frag_A[2];
    WarpFragmentB warp_frag_B[2];

    // Load A fragment from shared A
    this->warp_tile_iterator_A_.set_kgroup_index(0);
    this->warp_tile_iterator_A_.load(warp_frag_A[0]);
    ++this->warp_tile_iterator_A_;

    // Load B fragment from shared B
    this->warp_tile_iterator_B_.set_kgroup_index(0);
    this->warp_tile_iterator_B_.load(warp_frag_B[0]);
    ++this->warp_tile_iterator_B_;

    // Pair of fragments used to overlap global memory loads and math instructions;
    FragmentA tb_frag1_A, tb_frag2_A, tb_frag_A;
    FragmentB tb_frag1_B, tb_frag2_B, tb_frag_B;

    // Avoid reading out of bounds
    iterator_A0.clear_mask(gemm_k_iterations <= 1);
    iterator_A1.clear_mask(gemm_k_iterations <= 1);
    iterator_A2.clear_mask(gemm_k_iterations <= 1);
    iterator_A3.clear_mask(gemm_k_iterations <= 1);

    iterator_B0.clear_mask(gemm_k_iterations <= 1);
    iterator_B1.clear_mask(gemm_k_iterations <= 1);
    iterator_B2.clear_mask(gemm_k_iterations <= 1);
    iterator_B3.clear_mask(gemm_k_iterations <= 1);

    //
    // Mainloop
    //

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
          
          pre_sum(tb_frag1_A, tb_frag2_A, tb_frag_A, tb_frag1_B, tb_frag2_B, tb_frag_B);

          // Write fragments to shared memory
          this->smem_iterator_A_.store(transform_A_(tb_frag_A));

          this->smem_iterator_B_.store(transform_B_(tb_frag_B));

          // Wait until we have at least one completed global fetch stage
          gmem_wait();

          // Advance smem read and write stages
          advance_smem_stages();
        }

        this->warp_tile_iterator_A_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations);
        this->warp_tile_iterator_B_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations);

        this->warp_tile_iterator_A_.load(warp_frag_A[(warp_mma_k + 1) % 2]);
        this->warp_tile_iterator_B_.load(warp_frag_B[(warp_mma_k + 1) % 2]);

        ++this->warp_tile_iterator_A_;
        ++this->warp_tile_iterator_B_;

        if (warp_mma_k == 0) {
          // Load fragment from global A
          // tb_frag1_A.clear(); tb_frag2_A.clear();

          // // Load fragment from global B
          // tb_frag1_B.clear(); tb_frag2_B.clear();

          prefetch_load(iterator_A0, iterator_A1, iterator_A2, iterator_A3, 
                      iterator_B0, iterator_B1, iterator_B2, iterator_B3,
                      tb_frag1_A, tb_frag2_A, tb_frag1_B, tb_frag2_B);

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
        // if (MmaStrassenKind == MmaStrassen::GlobalLevel1_M0 && threadIdx.x == 0 && warp_mma_k == 0) {
        //   for (int i = 0; i < warp_frag_B[warp_mma_k % 2].size(); i++) {
        //     printf("573 : %f %f\n",warp_frag_A[warp_mma_k % 2][i], warp_frag_B[warp_mma_k % 2][i]);
        //   }
        // }
        warp_mma(
          accum,
          warp_frag_A[warp_mma_k % 2],
          warp_frag_B[warp_mma_k % 2],
          accum);
      }
    }
  }


  /// Prepares the class for another prologue.
  CUTLASS_DEVICE
  void wind_down()
  {
    // First, increment remaining warp tiles to catch it up with the write stage.
    #pragma unroll
    for (int warp_mma_k = 1; warp_mma_k < Base::kWarpGemmIterations; ++warp_mma_k)
    {
      this->warp_tile_iterator_A_.set_kgroup_index(warp_mma_k);
      this->warp_tile_iterator_B_.set_kgroup_index(warp_mma_k);

      ++this->warp_tile_iterator_A_;
      ++this->warp_tile_iterator_B_;
    }

    // If we bumped the read iterators to the end of the circular buffer, wrap them around to
    // align them with the write iterators
    if (smem_write_stage_idx == 0)
    {
      this->warp_tile_iterator_A_.add_tile_offset(
        {0, -Base::kStages * Policy::kPartitionsK * Base::kWarpGemmIterations});
      this->warp_tile_iterator_B_.add_tile_offset(
        {-Base::kStages * Policy::kPartitionsK * Base::kWarpGemmIterations, 0});
    }
  }

  /// Perform a threadblock-scoped matrix multiply-accumulate
  CUTLASS_DEVICE
  void operator()(
    int gemm_k_iterations,                            ///< number of iterations of the mainloop
    FragmentC &m0, FragmentC &m1, 
    FragmentC &m2, FragmentC &m3, 
    FragmentC &m4, FragmentC &m5, 
    FragmentC &m6,
    IteratorA iterator_A0,                             ///< iterator over A operand in global memory
    IteratorA iterator_A1,
    IteratorA iterator_A2,
    IteratorA iterator_A3,
    IteratorB iterator_B0,                             ///< iterator over B operand in global memory
    IteratorB iterator_B1,
    IteratorB iterator_B2,
    IteratorB iterator_B3)
  {
    FragmentC& accum = (StrassenMiGroup_::hasM0()) ? m0 :
                       (StrassenMiGroup_::hasM1()) ? m1 :
                       (StrassenMiGroup_::hasM2()) ? m2 :
                       (StrassenMiGroup_::hasM3()) ? m3 :
                       (StrassenMiGroup_::hasM4()) ? m4 :
                       (StrassenMiGroup_::hasM5()) ? m5 : m6;
                      //  (StrassenMiGroup_::hasM6()) ? m6 : 


    // Prologue
    prologue(iterator_A0, iterator_A1, iterator_A2, iterator_A3,
             iterator_B0, iterator_B1, iterator_B2, iterator_B3,
             gemm_k_iterations);

    // // Wait until we have at least one completed global fetch stage
    gmem_wait();

    // // Perform accumulation in the 'd' output operand
    // // Perform the MAC-iterations
    gemm_iters(gemm_k_iterations, accum, 
               iterator_A0, iterator_A1, iterator_A2, iterator_A3,
               iterator_B0, iterator_B1, iterator_B2, iterator_B3);
  }

};

/////////////////////////////////////////////////////////////////////////////////////////////////
} // namespace threadblock
} // namespace gemm
} // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
