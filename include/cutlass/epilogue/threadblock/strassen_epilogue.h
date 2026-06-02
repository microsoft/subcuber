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
  \brief Epilogue for threadblock scoped GEMMs using Tensor Ops.

  The epilogue rearranges the result of a matrix product through shared memory to match canonical
  tensor layouts in global memory. Epilogues support conversion and reduction operations.

  The shared memory resource is time-sliced across warps.
*/

#pragma once

#if defined(__CUDACC_RTC__)
#include <cuda/std/cassert>
#else
#include <assert.h>
#endif

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/array.h"
#include "cutlass/layout/vector.h"
#include "cutlass/layout/tensor.h"
#include "cutlass/tensor_coord.h"
#include "cutlass/aligned_buffer.h"
#include "cutlass/functional.h"

#include "cutlass/gemm/gemm.h"

#include "cutlass/transform/pitch_linear_thread_map.h"
#include "cutlass/transform/threadblock/regular_tile_iterator.h"

#include "cutlass/epilogue/threadblock/epilogue_base.h"
#include "cutlass/epilogue/threadblock/epilogue_base_streamk.h"
#include "cutlass/epilogue/threadblock/predicated_tile_iterator.h"

#include "cutlass/gemm/threadblock/strassen_mma_base.h"
using namespace MmaStrassen;

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace threadblock {


////////////////////////////////////////////////////////////////////////////////

/// Epilogue operator
template <
  typename Shape_,                          ///< Shape of threadblock tile (concept: GemmShape)
  typename WarpMmaOperator_,                ///< Warp-level MMA operator (concept: gemm::warp::MmaTensorOp)
  int PartitionsK,                          ///< Number of partitions of the K dimension
  typename OutputTileIterator_,             ///< Tile iterator reading and writing output tensors
  typename SourceInterimIterator_,
  typename AccumulatorFragmentIterator_,    ///< Fragment iterator selecting accumulators
  typename WarpTileIterator_,               ///< Warp-scoped tile iterator writing accumulators to SMEM
  typename SharedLoadIterator_,             ///< Threadblock-scoped tile iterator loading from SMEM
  typename OutputOp_,                       ///< Output operator
  typename Padding_,                        ///< Padding added to SMEM allocation to avoid bank conflicts (concept: MatrixShape)
  typename StrassenMiGroup_,
  int FragmentsPerPartition = 1,            ///< Used to coarsten the epilogue granularity
  int IterationsUnroll =                    ///< Used to reduce binary size when epilogue op is large
    (!IsEpilogueFunctorHeavy<OutputOp_>::value)
>
class StrassenEpilogue :
  public EpilogueBase<
    Shape_,
    typename WarpMmaOperator_::StrassenShape,
    PartitionsK,
    AccumulatorFragmentIterator_,
    WarpTileIterator_,
    Padding_,
    FragmentsPerPartition>,
  public EpilogueBaseStreamK<
    Shape_,
    PartitionsK,
    WarpMmaOperator_,
    AccumulatorFragmentIterator_>
{

public:
  using Base = EpilogueBase<
    Shape_,
    typename WarpMmaOperator_::StrassenShape,
    PartitionsK,
    AccumulatorFragmentIterator_,
    WarpTileIterator_,
    Padding_,
    FragmentsPerPartition>;

  using BaseStreamK = EpilogueBaseStreamK<
    Shape_,
    PartitionsK,
    WarpMmaOperator_,
    AccumulatorFragmentIterator_>;

  using Shape = Shape_;
  using WarpMmaOperator = WarpMmaOperator_;
  static int const kPartitionsK = PartitionsK;
  using OutputTileIterator = OutputTileIterator_;
  using SourceInterimIterator = SourceInterimIterator_;
  using AccumulatorFragmentIterator = AccumulatorFragmentIterator_;
  using WarpTileIterator = WarpTileIterator_;
  using SharedLoadIterator = SharedLoadIterator_;
  using OutputOp = OutputOp_;
  using Padding = Padding_;
  using StrassenMiGroup = StrassenMiGroup_;
  using Layout = layout::RowMajor;
  using LongIndex = typename Layout::LongIndex;

  /// Number of warps per block
  using WarpCount = typename Base::WarpCount;

  /// Number of threads per block
  static int const kBlockThreads = 32 * WarpCount::kCount;

  /// Per-thread accumulator tile type
  using AccumulatorTile = typename Base::AccumulatorTile;

  /// Numerical accumulation element type
  using ElementAccumulator = typename WarpMmaOperator::ElementC;

  /// Fragment type used by the accumulator tile's fragment iterator
  using AccumulatorFragment = typename AccumulatorFragmentIterator::Fragment;

  /// Output element
  using ElementOutput = typename OutputTileIterator::Element;

  /// Output access size
  static int const kElementsPerAccess = OutputTileIterator::kElementsPerAccess;

  /// Tensor reference to destination tensor
  using TensorRef = typename OutputTileIterator::TensorRef;

  /// Tensor reference to sync tensor
  using SyncTensorRef = typename cutlass::TensorRef<int, cutlass::layout::PackedVectorLayout>;

  /// Const tensor reference to source tensor
  using ConstTensorRef = typename OutputTileIterator::ConstTensorRef;

  /// Vector type used by the global output iterator
  using OutputAccessType = Array<
    typename OutputTileIterator::Element, OutputTileIterator::kElementsPerAccess>;

  /// Vector type used by the shared output iterator
  using AccumulatorAccessType = Array<typename WarpTileIterator::Element, OutputTileIterator::kElementsPerAccess>;

  static int constexpr kSmemTiles = Base::kFragmentsPerIteration > 1 ? Base::kFragmentsPerIteration : kPartitionsK;

  static int constexpr kSmemPointerOffset = Base::SharedStorage::StorageShape::kCount / kSmemTiles;


public:

  static_assert(SharedLoadIterator::Fragment::kElements == OutputTileIterator::Fragment::kElements,
    "Mismatch between shared load iterator and output tile iterator.");

  static_assert(OutputTileIterator::kElementsPerAccess, "OutputTileIterator::kElementsPerAccess must not be zero.");

  static_assert(!(OutputTileIterator::Fragment::kElements % OutputTileIterator::kElementsPerAccess), 
    "Divisibility");

  static_assert(kPartitionsK == 1 || Base::kFragmentsPerIteration == 1, "One of these must be exactly 1.");


public:

  /// Aspect for when epilogue source is not needed
  struct SourceAspectNotNeeded
  {
    /// Constructor
    CUTLASS_DEVICE
    SourceAspectNotNeeded()
    {}

    // No-op
    CUTLASS_DEVICE
    void load() { }

    /// Invoke the output functor over each vector of output
    CUTLASS_DEVICE
    void apply_output_operator(
      typename OutputTileIterator::Fragment &output_fragment,
      OutputOp const &output_op,
      typename SharedLoadIterator::Fragment const &aligned_accum_fragment)
    {
      OutputAccessType *output_frag_ptr =
        reinterpret_cast<OutputAccessType *>(&output_fragment);

      AccumulatorAccessType const *compute_frag_ptr =
        reinterpret_cast<AccumulatorAccessType const *>(&aligned_accum_fragment);

      int const kOutputOpIterations =
        OutputTileIterator::Fragment::kElements / OutputTileIterator::kElementsPerAccess;

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kOutputOpIterations; ++i)
      {
        // Call the output operator
        output_frag_ptr[i] = output_op(compute_frag_ptr[i]);
      }
    }

    CUTLASS_DEVICE
    void apply_output_operator(
      typename OutputTileIterator::Fragment &output_fragment,
      typename OutputTileIterator::Fragment const &aligned_accum_fragment)
    {
      output_fragment = aligned_accum_fragment;
    }
  };


  /// Aspect for when epilogue source is needed
  template<bool AccumIsNegative = false>
  struct SourceAspectNeeded
  {
    OutputTileIterator source_iterator;

    typename OutputTileIterator::Fragment source_fragment;

    /// Invoke the output functor over each vector of output
    CUTLASS_DEVICE
    static void apply_output_operator(
      typename OutputTileIterator::Fragment &output_fragment,
      OutputOp const &output_op,
      typename SharedLoadIterator::Fragment const &aligned_accum_fragment,
      typename OutputTileIterator::Fragment const &source_fragment)
    {
      OutputAccessType *output_frag_ptr =
        reinterpret_cast<OutputAccessType *>(&output_fragment);

      AccumulatorAccessType const *compute_frag_ptr =
        reinterpret_cast<AccumulatorAccessType const *>(&aligned_accum_fragment);

      OutputAccessType const *source_frag_ptr =
        reinterpret_cast<OutputAccessType const *>(&source_fragment);

      int const kOutputOpIterations =
        OutputTileIterator::Fragment::kElements / OutputTileIterator::kElementsPerAccess;

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kOutputOpIterations; ++i)
      {
        auto new_frag = multiplies<AccumulatorAccessType>()(compute_frag_ptr[i], (AccumIsNegative ? -1 : 1));

        // Call the output operator

        output_frag_ptr[i] = output_op(new_frag, source_frag_ptr[i]);
      }
    }

    /// Constructor
    CUTLASS_DEVICE
    SourceAspectNeeded(OutputTileIterator source_iterator) :
      source_iterator(source_iterator)
    {
      source_fragment.clear();
    }

    // Load addend source fragment from global memory
    CUTLASS_DEVICE
    void load() {
      source_iterator.load(source_fragment);
      ++source_iterator;
    }

    /// Invoke the output functor over each vector of output
    CUTLASS_DEVICE
    void apply_output_operator(
      typename OutputTileIterator::Fragment &output_fragment,
      OutputOp const &output_op,
      typename SharedLoadIterator::Fragment const &aligned_accum_fragment)
    {
      apply_output_operator(output_fragment, output_op, aligned_accum_fragment, source_fragment);
    }

    CUTLASS_DEVICE
    void apply_output_operator(
      typename OutputTileIterator::Fragment &output_fragment,
      typename OutputTileIterator::Fragment const &aligned_accum_fragment)
    {
      output_fragment = plus<typename OutputTileIterator::Fragment>()(source_fragment, aligned_accum_fragment);
    }
  };


  template<bool isAccumNegative, bool IsNeeded>
  struct SourceAspectNeededWithMatrixOffset;

   /// Aspect for when epilogue source is needed
  template<bool isAccumNegative>
  struct SourceAspectNeededWithMatrixOffset<isAccumNegative, true>
  {
    OutputTileIterator source_iterator;

    typename OutputTileIterator::Fragment source_fragment1;
    typename OutputTileIterator::Fragment source_fragment2;

    /// Invoke the output functor over each vector of output
    CUTLASS_DEVICE
    static void apply_output_operator(
      typename OutputTileIterator::Fragment &output_fragment1,
      typename OutputTileIterator::Fragment &output_fragment2,
      OutputOp const &output_op,
      typename SharedLoadIterator::Fragment const &aligned_accum_fragment,
      typename OutputTileIterator::Fragment const &source_fragment1,
      typename OutputTileIterator::Fragment const &source_fragment2)
    {
      OutputAccessType *output_frag_ptr1 =
        reinterpret_cast<OutputAccessType *>(&output_fragment1);
      
      OutputAccessType *output_frag_ptr2 =
        reinterpret_cast<OutputAccessType *>(&output_fragment2);

      AccumulatorAccessType const *compute_frag_ptr =
        reinterpret_cast<AccumulatorAccessType const *>(&aligned_accum_fragment);

      OutputAccessType const *source_frag_ptr1 =
        reinterpret_cast<OutputAccessType const *>(&source_fragment1);

      OutputAccessType const *source_frag_ptr2 =
        reinterpret_cast<OutputAccessType const *>(&source_fragment2);

      int const kOutputOpIterations =
        OutputTileIterator::Fragment::kElements / OutputTileIterator::kElementsPerAccess;

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kOutputOpIterations; ++i)
      {
        // Call the output operator
        auto new_frag = multiplies<AccumulatorAccessType>()(compute_frag_ptr[i], (isAccumNegative ? -1 : 1));
        output_frag_ptr1[i] = output_op(compute_frag_ptr[i], source_frag_ptr1[i]);
        output_frag_ptr2[i] = output_op(new_frag, source_frag_ptr2[i]);
      }
    }

    /// Constructor
    CUTLASS_DEVICE
    SourceAspectNeededWithMatrixOffset(OutputTileIterator source_iterator) :
      source_iterator(source_iterator)
    {
      source_fragment1.clear();
      source_fragment2.clear();
    }

    // Load addend source fragment from global memory
    CUTLASS_DEVICE
    void load() {
      source_iterator.load(source_fragment1, source_fragment2);
      ++source_iterator;
    }

    /// Invoke the output functor over each vector of output
    CUTLASS_DEVICE
    void apply_output_operator(
      typename OutputTileIterator::Fragment &output_fragment1,
      typename OutputTileIterator::Fragment &output_fragment2,
      OutputOp const &output_op,
      typename SharedLoadIterator::Fragment const &aligned_accum_fragment)
    {
      apply_output_operator(output_fragment1, output_fragment2, output_op, 
                            aligned_accum_fragment, source_fragment1, source_fragment2);
    }
  };

     /// Aspect for when epilogue source is needed
  template<bool isAccumNegative>
  struct SourceAspectNeededWithMatrixOffset<isAccumNegative, false>
  {
    /// Invoke the output functor over each vector of output
    CUTLASS_DEVICE
    static void apply_output_operator(
      typename OutputTileIterator::Fragment &output_fragment1,
      typename OutputTileIterator::Fragment &output_fragment2,
      OutputOp const &output_op,
      typename SharedLoadIterator::Fragment const &aligned_accum_fragment)
    {
      OutputAccessType *output_frag_ptr1 =
        reinterpret_cast<OutputAccessType *>(&output_fragment1);
      
      OutputAccessType *output_frag_ptr2 =
        reinterpret_cast<OutputAccessType *>(&output_fragment2);

      AccumulatorAccessType const *compute_frag_ptr =
        reinterpret_cast<AccumulatorAccessType const *>(&aligned_accum_fragment);

      OutputAccessType const *source_frag_ptr1 =
        reinterpret_cast<OutputAccessType const *>(&output_fragment1);

      OutputAccessType const *source_frag_ptr2 =
        reinterpret_cast<OutputAccessType const *>(&output_fragment2);

      int const kOutputOpIterations =
        OutputTileIterator::Fragment::kElements / OutputTileIterator::kElementsPerAccess;

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kOutputOpIterations; ++i)
      {
        // Call the output operator
        output_frag_ptr1[i] = plus<AccumulatorAccessType>()(compute_frag_ptr[i], source_frag_ptr1[i]);
        auto new_frag = multiplies<AccumulatorAccessType>()(compute_frag_ptr[i], (isAccumNegative ? -1 : 1));
        output_frag_ptr2[i] = plus<AccumulatorAccessType>()(new_frag, source_frag_ptr2[i]);
      }
    }

    /// Constructor
    CUTLASS_DEVICE
    SourceAspectNeededWithMatrixOffset()
    {}

    // Load addend source fragment from global memory
    CUTLASS_DEVICE
    void load() {}
  };
private:

  /// Loads fragment from shared memory aligned with output tensor
  SharedLoadIterator shared_load_iterator_;

  /// Thread index in the threadblock
  int thread_idx;

  /// Warp index in the threadblock
  int warp_idx;

public:

  /// Constructor
  CUTLASS_DEVICE
  StrassenEpilogue(
      typename Base::SharedStorage &shared_storage,   ///< Shared storage object
      int thread_idx,                                 ///< ID of a thread within the threadblock
      int warp_idx,                                   ///< ID of warp within threadblock
      int lane_idx)                                   ///< Id of thread within warp
  :
      Base(shared_storage, thread_idx, warp_idx, lane_idx),
      BaseStreamK(thread_idx),
      shared_load_iterator_(shared_storage.reference(), thread_idx),
      thread_idx(thread_idx),
      warp_idx(warp_idx)
  {
    // if (threadIdx.x == 32)
    //   printf("320: %p\n", this->warp_tile_iterator_.pointer_);
  }


  /// Aggregates the accumulator sets shared by peer blocks in the global workspace,
  /// performing epilogue computations, writing to output
  CUTLASS_DEVICE
  void reduce(
      int peer_idx_begin,
      int peer_idx_end,
      int reduce_fragment_idx,
      void *element_workspace,
      OutputOp const &output_op,                      ///< Output operator
      OutputTileIterator destination_iterator,        ///< Tile iterator for destination
      OutputTileIterator source_iterator)             ///< Threadblock tile coordinate in GEMM (in units of threadblock tiles)
  {
    // Reduce peer accumulator fragments into one fragment
    AccumulatorFragment accum_fragment;
    BaseStreamK::reduce(accum_fragment, peer_idx_begin, peer_idx_end, reduce_fragment_idx, element_workspace);

    // Store fragment to shared memory
    this->warp_tile_iterator_.store(accum_fragment);

    __syncthreads();

    // Initialize/load source-fragment data
    typename OutputTileIterator::Fragment source_fragment;
    source_fragment.clear();

    if (output_op.is_source_needed())
    {
      source_iterator += reduce_fragment_idx;
      source_iterator.load(source_fragment);
    }

    // Load fragment from shared memory
    typename SharedLoadIterator::Fragment aligned_accum_fragment;
    shared_load_iterator_.load(aligned_accum_fragment);

    // Add fragments shared by other k partitions
    if (kPartitionsK > 1)
    {
      plus <typename SharedLoadIterator::Fragment> add_fragments;

      CUTLASS_PRAGMA_UNROLL
      for ( int i = 1; i < kPartitionsK; ++i) {
        typename SharedLoadIterator::Fragment aligned_addend_fragment;
        shared_load_iterator_.add_pointer_offset(kSmemPointerOffset);
        shared_load_iterator_.load(aligned_addend_fragment);
        aligned_accum_fragment = add_fragments(aligned_accum_fragment, aligned_addend_fragment);
      }
    }

    // Compute the output result
    typename OutputTileIterator::Fragment output_fragment;

    // Apply the output operator
    SourceAspectNeeded<false>::apply_output_operator(
        output_fragment,
        output_op,
        aligned_accum_fragment,
        source_fragment);

    // Store the final result
    destination_iterator += reduce_fragment_idx;
    destination_iterator.store(output_fragment);
  }


  /// Perform the epilogue computations and stream the result to global memory.
  CUTLASS_DEVICE
  void operator()(
    OutputOp const &output_op,                      ///< Output operator
    OutputTileIterator destination_iterator,        ///< Tile iterator for destination
    AccumulatorTile const &accumulators)            ///< Complete warp-level accumulator tile
  {
    operator()(output_op, destination_iterator, accumulators, SourceAspectNotNeeded());
  }


  /// Perform the epilogue computations and stream the result to global memory.  Implements
  /// two alternative codepaths, depending on whether the output op requires addend data to be loaded.
  CUTLASS_DEVICE
  void operator()(
    OutputOp &output_op,                      ///< Output operator
    OutputTileIterator destination_iterator,        ///< Tile iterator for destination
    AccumulatorTile const &accumulators,            ///< Complete warp-level accumulator tile
    OutputTileIterator source_iterator,            ///< Tile iterator for addend source
    const PostsumOp& postsum_global_dest,
    const PostsumOp& postsum_src,
    OutputTileIterator destination_iterator2,
    OutputTileIterator source_iterator2) {
    
    if (postsum_global_dest.is_mem_global() &&
        postsum_global_dest.is_layout_final()) {
      bool do_second_load = StrassenMiGroup::Level1Idx == 5 && StrassenMiGroup::Level == 1 && source_iterator2.base_pointer() != nullptr;//TODO: Level should be 2 not 1.
      bool do_second_store = StrassenMiGroup::Level1Idx == 4 && StrassenMiGroup::Level == 1 && destination_iterator2.base_pointer() != nullptr;
      if (postsum_src.is_mem_global() && postsum_src.is_layout_final()) { // output_op.is_source_needed()
        operator()(output_op, destination_iterator, destination_iterator2, accumulators, SourceAspectNeeded<false>(source_iterator),
                   SourceAspectNeeded<false>(source_iterator2), postsum_src, do_second_load, do_second_store);
      } else if (output_op.is_source_needed()) {
        operator()(output_op, destination_iterator, destination_iterator2, accumulators, SourceAspectNeeded<false>(source_iterator),
                   SourceAspectNeeded<false>(source_iterator2), postsum_src, do_second_load, do_second_store);
      } else {
        operator()(output_op, destination_iterator, destination_iterator2, accumulators, SourceAspectNotNeeded(),
                   SourceAspectNotNeeded(), postsum_src, do_second_load, do_second_store);
      }
    }
  }


  /// Perform the epilogue computations and stream the result to global memory.  Implements a
  /// single codepath, regardless of whether the output op requires addend data to be loaded
  CUTLASS_DEVICE
  void unified(
    OutputOp const &output_op,                      ///< Output operator
    OutputTileIterator destination_iterator,        ///< Tile iterator for destination
    AccumulatorTile const &accumulators,            ///< Complete warp-level accumulator tile
    OutputTileIterator source_iterator )            ///< Tile iterator for addend source
  {
    if (!output_op.is_source_needed())
    {
      source_iterator.clear_mask();
      __syncthreads();  // Dummy (CUDA 11.0)
    }

    operator()(output_op, destination_iterator, accumulators, SourceAspectNeeded<false>(source_iterator));
  }

  template<class Seq>
  struct acc2smem;

  template <size_t... Seq>
  struct acc2smem<cutlass::index_sequence<Seq...>> {
    template<int Advance>
    CUTLASS_DEVICE
    static void helper(AccumulatorFragmentIterator accum_fragment_iterator,
                      WarpTileIterator &warp_tile_iterator) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < Advance; i++) {
        ++accum_fragment_iterator;
      }

      typename AccumulatorFragmentIterator::Fragment accum_fragment;

      accum_fragment_iterator.load(accum_fragment);
      ++accum_fragment_iterator;
      warp_tile_iterator.store(accum_fragment);
    }

    CUTLASS_DEVICE
    static void push(size_t pos,
                    AccumulatorFragmentIterator const &iterator_begin,
                    WarpTileIterator &warp_tile_iterator) {
      int dummy[] = {(pos == Seq) && (helper<Seq>(iterator_begin, warp_tile_iterator), 0)...};
    }

    //Compute D0
    template<int Advance>
    CUTLASS_DEVICE
    static void helperD0(AccumulatorFragmentIterator m0,
                       AccumulatorFragmentIterator m3,
                       AccumulatorFragmentIterator m4,
                       AccumulatorFragmentIterator m6,
                       WarpTileIterator &warp_tile_iterator) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < Advance; i++) {
        ++m0; ++m3; ++m4; ++m6;
      }

      typename AccumulatorFragmentIterator::Fragment m0_frag;
      typename AccumulatorFragmentIterator::Fragment m3_frag;
      typename AccumulatorFragmentIterator::Fragment m4_frag;
      typename AccumulatorFragmentIterator::Fragment m6_frag;

      m0.load(m0_frag);
      m3.load(m3_frag);
      m4.load(m4_frag);
      m6.load(m6_frag);

      ++m0; ++m3; ++m4; ++m6;

      typename AccumulatorFragmentIterator::Fragment accum_fragment;

      for (int i = 0; i < accum_fragment.size(); i++) {
        accum_fragment[i] = m0_frag[i] + m3_frag[i] - m4_frag[i] + m6_frag[i];
      }

      warp_tile_iterator.store(accum_fragment);
    }

    CUTLASS_DEVICE
    static void pushD0(size_t pos,
                    AccumulatorFragmentIterator const &m0,
                    AccumulatorFragmentIterator const &m3,
                    AccumulatorFragmentIterator const &m4,
                    AccumulatorFragmentIterator const &m6,
                    WarpTileIterator &warp_tile_iterator) {
      int dummy[] = {(pos == Seq) && (helperD0<Seq>(m0, m3, m4, m6, warp_tile_iterator), 0)...};
    }

    //Compute D1
    template<int Advance>
    CUTLASS_DEVICE
    static void helperD1(AccumulatorFragmentIterator m2,
                         AccumulatorFragmentIterator m4,
                         WarpTileIterator &warp_tile_iterator) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < Advance; i++) {
        ++m2; ++m4;
      }

      typename AccumulatorFragmentIterator::Fragment m2_frag;
      typename AccumulatorFragmentIterator::Fragment m4_frag;

      m2.load(m2_frag);
      m4.load(m4_frag);
      
      ++m2; ++m4;

      typename AccumulatorFragmentIterator::Fragment accum_fragment;

      for (int i = 0; i < accum_fragment.size(); i++) {
        accum_fragment[i] = m2_frag[i] + m4_frag[i];
      }

      warp_tile_iterator.store(accum_fragment);
    }

    CUTLASS_DEVICE
    static void pushD1(size_t pos,
                    AccumulatorFragmentIterator const &m2,
                    AccumulatorFragmentIterator const &m4,
                    WarpTileIterator &warp_tile_iterator) {
      int dummy[] = {(pos == Seq) && (helperD1<Seq>(m2, m4, warp_tile_iterator), 0)...};
    }

    //Compute D2
    template<int Advance>
    CUTLASS_DEVICE
    static void helperD2(AccumulatorFragmentIterator m1,
                         AccumulatorFragmentIterator m3,
                         WarpTileIterator &warp_tile_iterator) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < Advance; i++) {
        ++m1; ++m3;
      }

      typename AccumulatorFragmentIterator::Fragment m1_frag;
      typename AccumulatorFragmentIterator::Fragment m3_frag;

      m1.load(m1_frag);
      m3.load(m3_frag);
      
      ++m1; ++m3;

      typename AccumulatorFragmentIterator::Fragment accum_fragment;

      for (int i = 0; i < accum_fragment.size(); i++) {
        accum_fragment[i] = m1_frag[i] + m3_frag[i];
      }

      warp_tile_iterator.store(accum_fragment);
    }

    CUTLASS_DEVICE
    static void pushD2(size_t pos,
                    AccumulatorFragmentIterator const &m1,
                    AccumulatorFragmentIterator const &m3,
                    WarpTileIterator &warp_tile_iterator) {
      int dummy[] = {(pos == Seq) && (helperD2<Seq>(m1, m3, warp_tile_iterator), 0)...};
    }

    //Compute D3
    template<int Advance>
    CUTLASS_DEVICE
    static void helperD3(AccumulatorFragmentIterator m0,
                         AccumulatorFragmentIterator m1,
                          AccumulatorFragmentIterator m2,
                          AccumulatorFragmentIterator m5,
                         WarpTileIterator &warp_tile_iterator) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < Advance; i++) {
        ++m0; ++m1; ++m2; ++m5;
      }

      typename AccumulatorFragmentIterator::Fragment m0_frag;
      typename AccumulatorFragmentIterator::Fragment m1_frag;
      typename AccumulatorFragmentIterator::Fragment m2_frag;
      typename AccumulatorFragmentIterator::Fragment m5_frag;

      m0.load(m0_frag);
      m1.load(m1_frag);
      m2.load(m2_frag);
      m5.load(m5_frag);

      ++m0; ++m1; ++m2; ++m5;

      typename AccumulatorFragmentIterator::Fragment accum_fragment;

      for (int i = 0; i < accum_fragment.size(); i++) {
        accum_fragment[i] = m0_frag[i] - m1_frag[i] + m2_frag[i] + m5_frag[i];
      }

      warp_tile_iterator.store(accum_fragment);
    }

    CUTLASS_DEVICE
    static void pushD3(size_t pos,
                        AccumulatorFragmentIterator const &m0,
                    AccumulatorFragmentIterator const &m1,
                    AccumulatorFragmentIterator const &m2,
                        AccumulatorFragmentIterator const &m5,
                    WarpTileIterator &warp_tile_iterator) {
      int dummy[] = {(pos == Seq) && (helperD3<Seq>(m0, m1, m2, m5, warp_tile_iterator), 0)...};
    }
  };

  /// Streams the result to global memory
  template <typename SourceAspect>
  CUTLASS_DEVICE
  void operator()(
    OutputOp const &output_op,                      ///< Output operator
    OutputTileIterator destination_iterator,        ///< Tile iterator for destination
    OutputTileIterator destination_iterator2,
    AccumulatorTile const &accumulators,            ///< Complete warp-level accumulator tile
    SourceAspect source, SourceAspect source2,
    const PostsumOp& postsum_src, bool do_second_load, bool do_second_store)
  {
    // if (outputLayout == LayoutInterim && sourceLayout == LayoutFinal)
    //   return;

    // if (outputLayout == LayoutInterim) {
    //   if (sourceLayout == LayoutInterim) {

    //   }

    //   ElementOutput* destPtr = destination_iterator.pointer;
    //   OutputAccessType elems = *(OutputAccessType*)shptr;

    //   for (uint i = 0; i < accumulators.size(); i++) {
    //     destPtr[]
    //   }
    //   return;
    // }

    // Iterator over warp-level accumulator fragment
    AccumulatorFragmentIterator accum_fragment_iterator(accumulators);
    // typename SourceInterimIterator::Fragment *update_frag_ptr = reinterpret_cast<typename SourceInterimIterator::Fragment *> (&accumulators);

    //
    // Iterate over accumulator tile
    //

    #pragma unroll(IterationsUnroll ? OutputTileIterator::kIterations : 1)
    for (int iter = 0; iter < OutputTileIterator::kIterations; ++iter)
    {
      //
      // Load the source
      //

      source.load();
      if (do_second_load) source2.load();

      //
      // Convert and store fragment
      //
      #if 0
      //FIXME: This code fixes stack size of interim_epilogue.addSource but 
      //this performs a little worst than interim_epilogue.addSource.
      //Also this mapping is not correct.
      if (source_interim_iterator.mask()) {
        const uint UpdateIters = sizeof(typename OutputTileIterator::Fragment)/sizeof(typename SourceInterimIterator::Fragment);
        for (int ii = 0; ii < UpdateIters; ii++) {
          typename SourceInterimIterator::Fragment& update_frag = update_frag_ptr[0];
          typename SourceInterimIterator::Fragment source_fragment;
          source_interim_iterator.load(source_fragment);
          update_frag = update_frag + source_fragment;
          ++update_frag_ptr;
          ++source_interim_iterator;
        }
      }
      #endif

      __syncthreads();

      acc2smem<cutlass::make_index_sequence<OutputTileIterator::kIterations>>::push(
        iter, accum_fragment_iterator, this->warp_tile_iterator_);

      __syncthreads();

      // if (threadIdx.x == 0 && iter == 1) {
      //   auto f = this->warp_tile_iterator_.pointer_;
      //   size_t s = 256;////WarpTileIterator::Shape::kMN;
      //   for (size_t i = 0; i < s; i++) {
      //     printf("%ld; %p, %p, %f\n", i, &f[i], &f[i][0], f[i][0]);
      //   }
      // }
      // __syncthreads();
      //
      // Load fragments from shared memory
      //

      typename SharedLoadIterator::Fragment aligned_accum_fragment[kPartitionsK];
      shared_load_iterator_.load(aligned_accum_fragment[0]);
      typename OutputTileIterator::Fragment output_fragment2;
      SourceAspectNotNeeded source_not_needed;
      if (do_second_store) {
        source_not_needed.apply_output_operator(output_fragment2, output_op, aligned_accum_fragment[0]);
        // if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0)
        //   printf("875 %f %f\n", float(output_fragment2[0]), float(aligned_accum_fragment[0][0]));
        destination_iterator2.store(output_fragment2);
      }

      if (kPartitionsK > 1) {
        plus <typename SharedLoadIterator::Fragment> add_fragments;

        CUTLASS_PRAGMA_UNROLL
        for ( int i = 1; i < kPartitionsK; ++i) {
          shared_load_iterator_.add_pointer_offset(kSmemPointerOffset);
          shared_load_iterator_.load(aligned_accum_fragment[i]);
          aligned_accum_fragment[0] = add_fragments(aligned_accum_fragment[0], aligned_accum_fragment[i]);
        }

        shared_load_iterator_.add_pointer_offset((1 - kPartitionsK) * kSmemPointerOffset);
      }

      //
      // Compute the output result
      //

      typename OutputTileIterator::Fragment output_fragment;
      source.apply_output_operator(output_fragment, output_op, aligned_accum_fragment[0]);
      
      if (do_second_load) {
        source2.apply_output_operator(output_fragment, output_fragment);
        // output_fragment = output_fragment + output_fragment2;
      }

      //
      // Store the final result
      //

      destination_iterator.store(output_fragment);
      ++destination_iterator; ++destination_iterator2;
    }
  }
  

  CUTLASS_DEVICE
  void update(
    OutputOp const &output_op,                      ///< Output operator
    OutputTileIterator destination_iterator,        ///< Tile iterator for destination
    AccumulatorTile const &accumulators,            ///< Complete warp-level accumulator tile
    OutputTileIterator source_iterator)            ///< Tile iterator for addend source
  {
    if (output_op.is_source_needed())
    {
      update(output_op, destination_iterator, accumulators, SourceAspectNeeded<false>(source_iterator));
    }
    else
    {
      update(output_op, destination_iterator, accumulators, SourceAspectNotNeeded());
    }
  }

    /// Streams the result to global memory
  template <typename SourceAspect>
  CUTLASS_DEVICE
  void update(
    OutputOp const &output_op,                      ///< Output operator
    OutputTileIterator destination_iterator,        ///< Tile iterator for destination
    AccumulatorTile const &accumulators,            ///< Complete warp-level accumulator tile
    SourceAspect source)
  {
    // Iterator over warp-level accumulator fragment
    AccumulatorFragmentIterator accum_fragment_iterator(accumulators);

    //
    // Iterate over accumulator tile
    //
    #pragma unroll(IterationsUnroll ? OutputTileIterator::kIterations : 1)
    for (int iter = 0; iter < OutputTileIterator::kIterations; ++iter)
    {
      //
      // Load the source
      //

        source.load();
      //
      // Convert and store fragment
      //

      __syncthreads();

      acc2smem<cutlass::make_index_sequence<OutputTileIterator::kIterations>>::push(
        iter, accum_fragment_iterator, this->warp_tile_iterator_);

      __syncthreads();

      // if (threadIdx.x == 0 && iter == 1) {
      //   auto f = this->warp_tile_iterator_.pointer_;
      //   size_t s = 256;////WarpTileIterator::Shape::kMN;
      //   for (size_t i = 0; i < s; i++) {
      //     printf("%ld; %p, %p, %f\n", i, &f[i], &f[i][0], f[i][0]);
      //   }
      // }
      // __syncthreads();
      //
      // Load fragments from shared memory
      //

      typename SharedLoadIterator::Fragment aligned_accum_fragment[kPartitionsK];
      shared_load_iterator_.load(aligned_accum_fragment[0]);

      if (kPartitionsK > 1) {
        plus <typename SharedLoadIterator::Fragment> add_fragments;

        CUTLASS_PRAGMA_UNROLL
        for ( int i = 1; i < kPartitionsK; ++i) {
          shared_load_iterator_.add_pointer_offset(kSmemPointerOffset);
          shared_load_iterator_.load(aligned_accum_fragment[i]);
          aligned_accum_fragment[0] = add_fragments(aligned_accum_fragment[0], aligned_accum_fragment[i]);
        }

        shared_load_iterator_.add_pointer_offset((1 - kPartitionsK) * kSmemPointerOffset);
      }

      //
      // Compute the output result
      //

      typename OutputTileIterator::Fragment output_fragment;
      destination_iterator.load(output_fragment);
      {
        // OutputAccessType* output_frag_ptr = reinterpret_cast<OutputAccessType *>(&output_fragment);

        // AccumulatorAccessType const *compute_frag_ptr =
        // reinterpret_cast<AccumulatorAccessType const *>(&aligned_accum_fragment[i]);
        
        int const kOutputOpIterations =
        OutputTileIterator::Fragment::kElements / OutputTileIterator::kElementsPerAccess;

        // CUTLASS_PRAGMA_UNROLL
        // for (int i = 0; i < kOutputOpIterations; ++i) {
        //   output_frag_ptr[i] = plus<AccumulatorAccessType>()(compute_frag_ptr[i], output_frag_ptr[i]);
        // }

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < kOutputOpIterations; ++i) {
          output_fragment[i] = aligned_accum_fragment[0][i] + output_fragment[i];
        }
      }
      // source.apply_output_operator(output_fragment, output_op, aligned_accum_fragment[0]);

      //
      // Store the final result
      //

      destination_iterator.store(output_fragment);
      ++destination_iterator;
    }
  }

  /// Perform the epilogue computations and stream the result to global memory.  Implements
  /// two alternative codepaths, depending on whether the output op requires addend data to be loaded.
  CUTLASS_DEVICE
  void updateSum(
    OutputOp const &output_op,                      ///< Output operator
    OutputTileIterator destination_iterator_D0D3,        ///< Tile iterator for destination
    AccumulatorTile const &m0,            ///< Complete warp-level accumulator tile
    OutputTileIterator source_iterator_C0C3)            ///< Tile iterator for addend source
  {
    if (output_op.is_source_needed())
    {
      updateSum(output_op, destination_iterator_D0D3, m0, 
      SourceAspectNeededWithMatrixOffset<false, true>(source_iterator_C0C3));
    }
    else
    {
      updateSum(output_op, destination_iterator_D0D3, m0, 
      SourceAspectNeededWithMatrixOffset<false, false>());
    }
  }

  /// Streams the result to global memory
  template <typename SourceAspect>
  CUTLASS_DEVICE
  void updateSum(
    OutputOp const &output_op,                      ///< Output operator
    OutputTileIterator destination_iterator_D0D3,        ///< Tile iterator for destination
    AccumulatorTile const &m0,            ///< Complete warp-level accumulator tile
    SourceAspect source_C0C3)
  {
    // Iterator over warp-level accumulator fragment
    AccumulatorFragmentIterator m0_iterator(m0);

    //
    // Iterate over accumulator tile
    //
    // if (threadIdx.x == 0) printf("480: OutputTileIterator::kIterations %d\n", OutputTileIterator::kIterations);
    #pragma unroll(IterationsUnroll ? OutputTileIterator::kIterations : 1)
    for (int iter = 0; iter < OutputTileIterator::kIterations; ++iter)
    {
      //
      // Load the source
      //

        source_C0C3.load();
      //
      // Convert and store fragment
      //
      __syncthreads();

      acc2smem<cutlass::make_index_sequence<OutputTileIterator::kIterations>>::push(
        iter, m0_iterator, this->warp_tile_iterator_);

      __syncthreads();

      // if (threadIdx.x == 0 && iter == 1) {
      //   auto f = this->warp_tile_iterator_.pointer_;
      //   size_t s = 256;////WarpTileIterator::Shape::kMN;
      //   for (size_t i = 0; i < s; i++) {
      //     printf("%ld; %p, %p, %f\n", i, &f[i], &f[i][0], f[i][0]);
      //   }
      // }
      // __syncthreads();
      //
      // Load fragments from shared memory
      //

      typename SharedLoadIterator::Fragment aligned_accum_fragment[kPartitionsK];
      shared_load_iterator_.load(aligned_accum_fragment[0]);

      // if (kPartitionsK > 1) {
      //   plus <typename SharedLoadIterator::Fragment> add_fragments;

      //   CUTLASS_PRAGMA_UNROLL
      //   for ( int i = 1; i < kPartitionsK; ++i) {
      //     shared_load_iterator_.add_pointer_offset(kSmemPointerOffset);
      //     shared_load_iterator_.load(aligned_accum_fragment[i]);
      //     aligned_accum_fragment[0] = add_fragments(aligned_accum_fragment[0], aligned_accum_fragment[i]);
      //   }

      //   shared_load_iterator_.add_pointer_offset((1 - kPartitionsK) * kSmemPointerOffset);
      // }

      // //
      // // Compute the output result
      // //

      typename OutputTileIterator::Fragment output_fragment_D0;
      typename OutputTileIterator::Fragment output_fragment_D3;
      destination_iterator_D0D3.load(output_fragment_D0, output_fragment_D3);
      source_C0C3.apply_output_operator(output_fragment_D0, output_fragment_D3, 
                                        output_op, aligned_accum_fragment[0]);

      // //
      // // Store the final result
      // //
      destination_iterator_D0D3.store(output_fragment_D0, output_fragment_D3);
      ++destination_iterator_D0D3;
    }
  }

  CUTLASS_DEVICE
  void updateDiff(
    OutputOp const &output_op,                      ///< Output operator
    OutputTileIterator destination_iterator_D13,        ///< Tile iterator for destination
    AccumulatorTile const &m2,            ///< Complete warp-level accumulator tile
    OutputTileIterator source_iterator_C13)            ///< Tile iterator for addend source
  {
    if (output_op.is_source_needed())
    {
      updateDiff(output_op, destination_iterator_D13, m2, 
      SourceAspectNeededWithMatrixOffset<true, true>(source_iterator_C13));
    }
    else
    {
      updateDiff(output_op, destination_iterator_D13, m2,
      SourceAspectNeededWithMatrixOffset<true, false>());
    }
  }

  /// Streams the result to global memory
  template <typename SourceAspect>
  CUTLASS_DEVICE
  void updateDiff(
    OutputOp const &output_op,                      ///< Output operator
    OutputTileIterator destination_iterator_D13,        ///< Tile iterator for destination
    AccumulatorTile const &m2,            ///< Complete warp-level accumulator tile
    SourceAspect source_C13)
  {
    // Iterator over warp-level accumulator fragment
    AccumulatorFragmentIterator m2_iterator(m2);

    //
    // Iterate over accumulator tile
    //
    // if (threadIdx.x == 0) printf("480: OutputTileIterator::kIterations %d\n", OutputTileIterator::kIterations);
    #pragma unroll(IterationsUnroll ? OutputTileIterator::kIterations : 1)
    for (int iter = 0; iter < OutputTileIterator::kIterations; ++iter)
    {
      //
      // Load the source
      //

        source_C13.load();
      //
      // Convert and store fragment
      //

      __syncthreads();

      acc2smem<cutlass::make_index_sequence<OutputTileIterator::kIterations>>::push(
        iter, m2_iterator, this->warp_tile_iterator_);

      __syncthreads();

      // if (threadIdx.x == 0 && iter == 1) {
      //   auto f = this->warp_tile_iterator_.pointer_;
      //   size_t s = 256;////WarpTileIterator::Shape::kMN;
      //   for (size_t i = 0; i < s; i++) {
      //     printf("%ld; %p, %p, %f\n", i, &f[i], &f[i][0], f[i][0]);
      //   }
      // }
      // __syncthreads();
      //
      // Load fragments from shared memory
      //

      typename SharedLoadIterator::Fragment aligned_accum_fragment[kPartitionsK];
      shared_load_iterator_.load(aligned_accum_fragment[0]);

      // if (kPartitionsK > 1) {
      //   plus <typename SharedLoadIterator::Fragment> add_fragments;

      //   CUTLASS_PRAGMA_UNROLL
      //   for ( int i = 1; i < kPartitionsK; ++i) {
      //     shared_load_iterator_.add_pointer_offset(kSmemPointerOffset);
      //     shared_load_iterator_.load(aligned_accum_fragment[i]);
      //     aligned_accum_fragment[0] = add_fragments(aligned_accum_fragment[0], aligned_accum_fragment[i]);
      //   }

      //   shared_load_iterator_.add_pointer_offset((1 - kPartitionsK) * kSmemPointerOffset);
      // }

      // //
      // // Compute the output result
      // //

      typename OutputTileIterator::Fragment output_fragment_D1, output_fragment_D3;
      destination_iterator_D13.load(output_fragment_D1, output_fragment_D3);
      source_C13.apply_output_operator(output_fragment_D1, output_fragment_D3, output_op, aligned_accum_fragment[0]);
      // //
      // // Store the final result
      // //

      destination_iterator_D13.store(output_fragment_D1, output_fragment_D3);
      ++destination_iterator_D13;
    }
  }
};

////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace epilogue
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
