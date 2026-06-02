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
  typename SharedTileIterator_,
  typename AccumulatorFragmentIterator_,    ///< Fragment iterator selecting accumulators
  typename WarpTileIterator_,               ///< Warp-scoped tile iterator writing accumulators to SMEM
  typename OutputOp_,                       ///< Output operator
  typename Padding_,                        ///< Padding added to SMEM allocation to avoid bank conflicts (concept: MatrixShape)
  int FragmentsPerPartition = 1,            ///< Used to coarsten the epilogue granularity
  int IterationsUnroll =                    ///< Used to reduce binary size when epilogue op is large
    (!IsEpilogueFunctorHeavy<OutputOp_>::value)
>
class StrassenInterimEpilogue :
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
  using SharedTileIterator = SharedTileIterator_;
  using AccumulatorFragmentIterator = AccumulatorFragmentIterator_;
  using WarpTileIterator = WarpTileIterator_;
  using OutputOp = OutputOp_;
  using Padding = Padding_;
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

  /// Const tensor reference to source tensor
  using ConstTensorRef = typename OutputTileIterator::ConstTensorRef;

  /// Vector type used by the global output iterator
  using OutputAccessType = Array<
    typename OutputTileIterator::Element, OutputTileIterator::kElementsPerAccess>;

  /// Vector type used by the shared output iterator
  using AccumulatorAccessType = Array<typename WarpTileIterator::Element, OutputTileIterator::kElementsPerAccess>;


public:

  static_assert(OutputTileIterator::kElementsPerAccess, "OutputTileIterator::kElementsPerAccess must not be zero.");

  static_assert(!(OutputTileIterator::Fragment::kElements % OutputTileIterator::kElementsPerAccess), 
    "Divisibility");

private:
  /// Thread index in the threadblock
  int thread_idx;

  /// Warp index in the threadblock
  int warp_idx;

public:

  /// Constructor
  CUTLASS_DEVICE
  StrassenInterimEpilogue(
      typename Base::SharedStorage &shared_storage,   ///< Shared storage object
      int thread_idx,                                 ///< ID of a thread within the threadblock
      int warp_idx,                                   ///< ID of warp within threadblock
      int lane_idx)                                   ///< Id of thread within warp
  :
      Base(shared_storage, thread_idx, warp_idx, lane_idx),
      BaseStreamK(thread_idx),
      thread_idx(thread_idx),
      warp_idx(warp_idx)
  {
    // if (threadIdx.x == 32)
    //   printf("320: %p\n", this->warp_tile_iterator_.pointer_);
  }

  /// Perform the epilogue computations and stream the result to global and/or shared memory.
  /// Streams the result to global memory
  CUTLASS_DEVICE
  void operator()(
    // OutputOp const &output_op,                      ///< Output operator
    AccumulatorTile const &accumulators,            ///< Complete warp-level accumulator tile
    OutputTileIterator destination_iterator,        ///< Tile iterator for destination
    OutputTileIterator source_iterator,
    SharedTileIterator shared_tile_iterator,
    const PostsumOp& global_dest, const PostsumOp& shared_dest,
    PostsumOp sources[4])
  {
    // Iterator over warp-level accumulator fragment
    // AccumulatorFragmentIterator accum_fragment_iterator(accumulators);

    //
    // Iterate over accumulator tile
    //

    AccumulatorAccessType const *compute_frag_ptr = reinterpret_cast<AccumulatorAccessType const *> (&accumulators);
    NumericArrayConverter<ElementAccumulator, ElementOutput, 
            OutputOp::kCount, OutputOp::kRound> source_converter;
    NumericArrayConverter<ElementOutput, ElementAccumulator, 
            OutputOp::kCount, OutputOp::kRound> compute_frag_converter;

    #pragma unroll(IterationsUnroll ? OutputTileIterator::kIterations : 1)
    for (int iter = 0; iter < OutputTileIterator::kIterations; ++iter) {
      AccumulatorAccessType compute_frag = compute_frag_ptr[0];

      for (int i = 0; i < 4; i++) {
        if (sources[i].valid()) {
          if (sources[i].is_mem_global() && 
              sources[i].is_layout_interim()) {
            typename OutputTileIterator::Fragment source_fragment;
            source_iterator.load(source_fragment, sources[i].get_op());
            compute_frag = compute_frag + source_converter(source_fragment);
          }

          if (sources[i].is_mem_shared() && 
              sources[i].is_layout_interim()) {
            typename SharedTileIterator::Fragment shared_fragment;
            shared_tile_iterator.load(shared_fragment);
            compute_frag = compute_frag + source_converter(shared_fragment);
          }
        }
      }

      ++source_iterator;

      auto frag_store = compute_frag_converter(compute_frag);
      if (shared_dest.valid() && shared_dest.is_layout_interim()) {
        shared_tile_iterator.store(frag_store);
      }

      ++shared_tile_iterator;

      if (global_dest.valid() &&
          global_dest.is_layout_interim()) {
        destination_iterator.store(frag_store);
        ++destination_iterator;
      }
      ++compute_frag_ptr;
    }
  }



  /// Perform the epilogue computations and stream the result to global and/or shared memory.
  /// Streams the result to global memory
  CUTLASS_DEVICE
  void shared_to_global(
    // OutputOp const &output_op,                      ///< Output operator
    OutputTileIterator destination_iterator,        ///< Tile iterator for destination
    SharedTileIterator shared_tile_iterator)
  {
    // Iterator over warp-level accumulator fragment
    // AccumulatorFragmentIterator accum_fragment_iterator(accumulators);

    //
    // Iterate over accumulator tile
    //

    NumericArrayConverter<ElementAccumulator, ElementOutput, 
            OutputOp::kCount, OutputOp::kRound> source_converter;
    NumericArrayConverter<ElementOutput, ElementAccumulator, 
            OutputOp::kCount, OutputOp::kRound> compute_frag_converter;

    #pragma unroll(IterationsUnroll ? OutputTileIterator::kIterations : 1)
    for (int iter = 0; iter < OutputTileIterator::kIterations; ++iter) {

        typename SharedTileIterator::Fragment shared_fragment;
        shared_tile_iterator.load(shared_fragment);
      

  

      ++shared_tile_iterator;
      
        destination_iterator.store(shared_fragment);
        ++destination_iterator;
    }
  }

  CUTLASS_DEVICE
  void add_source(
    AccumulatorTile const &src_accum,        ///< Complete warp-level accumulator tile
    AccumulatorTile &dst_accum,
    OutputTileIterator source_iterator,
    SharedTileIterator shared_interim_iterator,
    PostsumOp sources[4]) {
    
    AccumulatorAccessType const *src_compute_frag_ptr = reinterpret_cast<AccumulatorAccessType const *> (&src_accum);
    AccumulatorAccessType *dst_compute_frag_ptr = reinterpret_cast<AccumulatorAccessType *> (&dst_accum);
    NumericArrayConverter<ElementAccumulator, ElementOutput, 
            OutputOp::kCount, OutputOp::kRound> source_converter;

    #pragma unroll(IterationsUnroll ? OutputTileIterator::kIterations : 1)
    for (int iter = 0; iter < OutputTileIterator::kIterations; ++iter) {
      typename OutputTileIterator::Fragment source_fragment;
      AccumulatorAccessType src_compute_frag = src_compute_frag_ptr[0];
      AccumulatorAccessType& dst_compute_frag = dst_compute_frag_ptr[0];

      #pragma unroll 4
      for (int i = 0; i < 4; i++) {
        if (sources[i].valid() && sources[i].is_layout_interim()) {
          source_fragment.clear();
          if (sources[i].is_mem_global()) {
            source_iterator.load(source_fragment, sources[i].get_op());
            src_compute_frag = src_compute_frag + source_converter(source_fragment);
          } else if (sources[i].is_mem_shared()) {
            shared_interim_iterator.load(source_fragment);
            src_compute_frag = src_compute_frag + source_converter(source_fragment);
          }
        }
      }

      dst_compute_frag = src_compute_frag;

      ++source_iterator;
      ++shared_interim_iterator;

      ++src_compute_frag_ptr;
      ++dst_compute_frag_ptr;
    }
  }
};

////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace epilogue
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
