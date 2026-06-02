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

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace threadblock {


////////////////////////////////////////////////////////////////////////////////

/// Epilogue operator
template <
  typename Shape_,                          ///< Shape of threadblock tile (concept: GemmShape)
  typename StrassenShape_,
  typename WarpMmaOperator_,                ///< Warp-level MMA operator (concept: gemm::warp::MmaTensorOp)
  int PartitionsK,                          ///< Number of partitions of the K dimension
  typename OutputTileIterator_,             ///< Tile iterator reading and writing output tensors
  typename AccumulatorFragmentIterator_,    ///< Fragment iterator selecting accumulators
  typename WarpTileIterator_,               ///< Warp-scoped tile iterator writing accumulators to SMEM
  typename SharedLoadIterator_,             ///< Threadblock-scoped tile iterator loading from SMEM
  typename OutputOp_,                       ///< Output operator
  typename Padding_,                        ///< Padding added to SMEM allocation to avoid bank conflicts (concept: MatrixShape)
  int FragmentsPerPartition = 1,            ///< Used to coarsten the epilogue granularity
  int IterationsUnroll =                    ///< Used to reduce binary size when epilogue op is large
    (!IsEpilogueFunctorHeavy<OutputOp_>::value)
>
class StrassenMatrixEpilogue :
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
  using AccumulatorFragmentIterator = AccumulatorFragmentIterator_;
  using WarpTileIterator = WarpTileIterator_;
  using SharedLoadIterator = SharedLoadIterator_;
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

  struct SourceAspectNotNeededFinalUpdate
  {
    typename OutputTileIterator::Fragment source_fragment[4];

    /// Constructor
    CUTLASS_DEVICE
    SourceAspectNotNeededFinalUpdate()
    {}
    
    using ElementCompute = typename OutputOp::ElementCompute;
    using ElementSource = typename OutputOp::ElementSource;

    NumericArrayConverter<ElementCompute, ElementSource, 
      OutputOp::kCount, OutputOp::kRound> source_converter;

    // No-op
    CUTLASS_DEVICE
    void load(OutputTileIterator source_iterator, uint64_t size) {
      source_iterator.load(source_fragment[0]);
      source_iterator.load_with_byte_offset(source_fragment[1], 1*size);
      source_iterator.load_with_byte_offset(source_fragment[2], 2*size);
      source_iterator.load_with_byte_offset(source_fragment[3], 3*size);
    }

    /// Invoke the output functor over each vector of output
    template<int d>
    CUTLASS_DEVICE
    void apply_output_operator_(
      typename OutputTileIterator::Fragment &output_fragment,
      OutputOp const &output_op,
      typename SharedLoadIterator::Fragment &aligned_accum_fragment)
    {
      OutputAccessType *output_frag_ptr =
        reinterpret_cast<OutputAccessType *>(&output_fragment);

      AccumulatorAccessType *compute_frag_ptr =
        reinterpret_cast<AccumulatorAccessType *>(&aligned_accum_fragment);

      OutputAccessType const *source_frag_ptr_0 =
        reinterpret_cast<OutputAccessType const *>(&source_fragment[0]);

      OutputAccessType const *source_frag_ptr_1 =
        reinterpret_cast<OutputAccessType const *>(&source_fragment[1]);
      
      OutputAccessType const *source_frag_ptr_2 =
        reinterpret_cast<OutputAccessType const *>(&source_fragment[2]);
      
      OutputAccessType const *source_frag_ptr_3 =
        reinterpret_cast<OutputAccessType const *>(&source_fragment[3]);

      int const kOutputOpIterations =
        OutputTileIterator::Fragment::kElements / OutputTileIterator::kElementsPerAccess;
      
      
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kOutputOpIterations; ++i)
      {
        auto new_frag = compute_frag_ptr[i];

        if (d == 0) {
          new_frag = new_frag + new_frag + source_converter(source_frag_ptr_0[i]) + source_converter(source_frag_ptr_3[i]);
        } else if (d == 1) {
          new_frag = new_frag + source_converter(source_frag_ptr_2[i]) + source_converter(source_frag_ptr_3[i]);
        } else if (d == 2) {
          new_frag = new_frag + source_converter(source_frag_ptr_0[i]);
        } else if (d == 3) {
          new_frag = new_frag + source_converter(source_frag_ptr_3[i]);
        } else if (d == 4) {
          new_frag = new_frag + source_converter(source_frag_ptr_0[i]) + source_converter(source_frag_ptr_1[i]);
        } else if (d == 5) {
          new_frag = source_converter(source_frag_ptr_2[i]) - source_converter(source_frag_ptr_0[i]) - new_frag;
        } else if (d == 6) {
          new_frag = source_converter(source_frag_ptr_1[i]) - source_converter(source_frag_ptr_3[i]) - new_frag;
        }

        // Call the output operator
        output_frag_ptr[i] = output_op(new_frag);
      }
    }

    CUTLASS_DEVICE
    void apply_output_operator(
      typename OutputTileIterator::Fragment *output_fragment,
      OutputOp const &output_op,
      typename SharedLoadIterator::Fragment &aligned_accum_fragment)
    {
      apply_output_operator_<0>(output_fragment[0], output_op, aligned_accum_fragment);
      apply_output_operator_<1>(output_fragment[1], output_op, aligned_accum_fragment);
      apply_output_operator_<2>(output_fragment[2], output_op, aligned_accum_fragment);
      apply_output_operator_<3>(output_fragment[3], output_op, aligned_accum_fragment);
      apply_output_operator_<4>(output_fragment[4], output_op, aligned_accum_fragment);
      apply_output_operator_<5>(output_fragment[5], output_op, aligned_accum_fragment);
      apply_output_operator_<6>(output_fragment[6], output_op, aligned_accum_fragment);
    }
  };

  /// Aspect for when epilogue source is not needed
  template<int c0 = 0, int c1 = 0, int c2 = 0, int c3 = 0, int c4 = 0, int c5 = 0, int c6 = 0>
  struct SourceAspectNotNeeded
  {
    typename OutputTileIterator::Fragment source_fragment[7];

    /// Constructor
    CUTLASS_DEVICE
    SourceAspectNotNeeded()
    {}

    // No-op
    CUTLASS_DEVICE
    void load7(OutputTileIterator& source_iterator, uint64_t size,
               uint32_t loadC) {
      if (c0 != 0 && (loadC & (1U << 0)) == (1U <<0)) {
        source_iterator.load(source_fragment[0]);
      }
      if (c1 != 0 && (loadC & (1U << 1)) == (1U <<1))
        source_iterator.load_with_byte_offset(source_fragment[1], 1*size);
      if (c2 != 0 && (loadC & (1U << 2)) == (1U <<2)) {
        source_iterator.load_with_byte_offset(source_fragment[2], 2*size);
      }
      if (c3 != 0 && (loadC & (1U << 3)) == (1U <<3)) {
        source_iterator.load_with_byte_offset(source_fragment[3], 3*size);
      }
      if (c4 != 0 && (loadC & (1U << 4)) == (1U <<4)) {
        source_iterator.load_with_byte_offset(source_fragment[4], 4*size);
      }
      if (c5 != 0 && (loadC & (1U << 5)) == (1U <<5))
        source_iterator.load_with_byte_offset(source_fragment[5], 5*size);
      if (c6 != 0 && (loadC & (1U << 6)) == (1U <<6))
        source_iterator.load_with_byte_offset(source_fragment[6], 6*size);
      
      ++source_iterator;
    }

    CUTLASS_DEVICE
    void loadLinear(OutputTileIterator& source_iterator, uint64_t size,
                    uint32_t loadC, uint linearIdx, uint iter) {
      ElementOutput* Mptr = (ElementOutput*)source_iterator.base_pointer();

      if (c0 != 0 && (loadC & (1U << 0)) == (1U <<0)) {
        source_iterator.load(source_fragment[0]);
      }
      if (c1 != 0 && (loadC & (1U << 1)) == (1U <<1))
        source_iterator.load_with_byte_offset(source_fragment[1], 1*size);
      if (c2 != 0 && (loadC & (1U << 2)) == (1U <<2)) {
        source_iterator.load_linear_with_offset(source_fragment[2], 2*size/2, linearIdx, iter);
      }
      if (c3 != 0 && (loadC & (1U << 3)) == (1U <<3)) {
        source_iterator.load_with_byte_offset(source_fragment[3], 3*size);
      }
      if (c4 != 0 && (loadC & (1U << 4)) == (1U <<4)) {
        source_iterator.load_with_byte_offset(source_fragment[4], 4*size);
      }
      if (c5 != 0 && (loadC & (1U << 5)) == (1U <<5))
        source_iterator.load_with_byte_offset(source_fragment[5], 5*size);
      if (c6 != 0 && (loadC & (1U << 6)) == (1U <<6))
        source_iterator.load_with_byte_offset(source_fragment[6], 6*size);
      
      // ++source_iterator;
    }

    CUTLASS_DEVICE
    void load(OutputTileIterator source_iterator) {
      // source_iterator.load(source_fragment[0]);
    }

    CUTLASS_DEVICE
    void zero() {
      for (int i = 0; i < 7; i++) {
        for (int j = 0; j < source_fragment[i].size(); j++) {
          source_fragment[i][j] = 0;
        }
      }
    }

    CUTLASS_DEVICE
    void load4(OutputTileIterator source_iterator, MatrixCoord shape, uint32_t loadC) {
      if (c0 != 0 && (loadC & 1) == 1 << 0)
        source_iterator.load(source_fragment[0]);
      if (c1 != 0 && (loadC & (1 << 2)) == 1 << 2)
        source_iterator.load_with_byte_offset(source_fragment[1], shape.column()/2* sizeof(OutputTileIterator::Element));
      if (c2 != 0 && (loadC & (1 << 3)) == 1 << 3)
        source_iterator.load_with_byte_offset(source_fragment[2], shape.row()/2 * shape.column()* sizeof(OutputTileIterator::Element));
      if (c3 != 0 && (loadC & (1 << 4)) == 1 << 4)
        source_iterator.load_with_byte_offset(source_fragment[3], (shape.row()/2 * shape.column() + shape.column()/2)* sizeof(OutputTileIterator::Element));
      ++source_iterator;
    }

    /// Invoke the output functor over each vector of output

    template<int c>
    CUTLASS_DEVICE
    void apply_output_operator(
      typename OutputTileIterator::Fragment &output_fragment, //half
      OutputOp const &output_op,
      typename SharedLoadIterator::Fragment &aligned_accum_fragment, //float
      typename OutputTileIterator::Fragment &source_fragment, //half
      bool sourceLoaded)
    {
      OutputAccessType *output_frag_ptr =
        reinterpret_cast<OutputAccessType *>(&output_fragment);

      AccumulatorAccessType *compute_frag_ptr =
        reinterpret_cast<AccumulatorAccessType *>(&aligned_accum_fragment);

      OutputAccessType const *source_frag_ptr =
        reinterpret_cast<OutputAccessType const *>(&source_fragment);

      int const kOutputOpIterations =
        OutputTileIterator::Fragment::kElements / OutputTileIterator::kElementsPerAccess;
      using ElementCompute = typename OutputOp::ElementCompute;
      using ElementSource = typename OutputOp::ElementSource;

      NumericArrayConverter<ElementCompute, ElementSource, 
      OutputOp::kCount, OutputOp::kRound> source_converter;

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kOutputOpIterations; ++i)
      {
        auto new_frag = compute_frag_ptr[i];

        CUTLASS_PRAGMA_UNROLL
        for (int j = 0; j < compute_frag_ptr[i].size(); j++) {
          new_frag[j] = new_frag[j] * (float)c;
        }

        if (sourceLoaded) {
          new_frag = new_frag + source_converter(source_frag_ptr[i]);
        }
        // Call the output operator
        output_frag_ptr[i] = output_op(new_frag);
      }
    }

    CUTLASS_DEVICE
    void apply_output_operator(
      typename OutputTileIterator::Fragment *output_fragment,
      OutputOp const &output_op,
      typename SharedLoadIterator::Fragment &aligned_accum_fragment,
      uint32_t loadC)
    {
      if (c0 != 0)
        apply_output_operator<c0>(output_fragment[0], output_op,
                                  aligned_accum_fragment, source_fragment[0], 
                                  (loadC & (1 << 0)) == (1 << 0));
      if (c1 != 0)
        apply_output_operator<c1>(output_fragment[1], output_op,
                                  aligned_accum_fragment, source_fragment[1],
                                  (loadC & (1 << 1)) == (1<<1));
      if (c2 != 0)
        apply_output_operator<c2>(output_fragment[2], output_op,
                                  aligned_accum_fragment, source_fragment[2],
                                  (loadC & (1 << 2)) == (1<<2));
      if (c3 != 0)
        apply_output_operator<c3>(output_fragment[3], output_op,
                                  aligned_accum_fragment, source_fragment[3],
                                  (loadC & (1 << 3)) == (1<<3));
      if (c4 != 0)
        apply_output_operator<c4>(output_fragment[4], output_op,
                                  aligned_accum_fragment, source_fragment[4],
                                  (loadC & (1 << 4)) == (1<<4));
      if (c5 != 0)
        apply_output_operator<c5>(output_fragment[5], output_op,
                                  aligned_accum_fragment, source_fragment[5],
                                  (loadC & (1 << 5)) == (1<<5));
      if (c6 != 0)
        apply_output_operator<c6>(output_fragment[6], output_op,
                                  aligned_accum_fragment, source_fragment[6],
                                  (loadC & (1 << 6)) == (1<<6));
    }

    CUTLASS_DEVICE
    void copy_output(
      typename OutputTileIterator::Fragment &output_fragment,
      OutputOp const &output_op,
      typename SharedLoadIterator::Fragment &aligned_accum_fragment)
    {
      OutputAccessType *output_frag_ptr =
        reinterpret_cast<OutputAccessType *>(&output_fragment);

      AccumulatorAccessType *compute_frag_ptr =
        reinterpret_cast<AccumulatorAccessType *>(&aligned_accum_fragment);

      int const kOutputOpIterations =
        OutputTileIterator::Fragment::kElements / OutputTileIterator::kElementsPerAccess;
      using ElementCompute = typename OutputOp::ElementCompute;
      using ElementSource = typename OutputOp::ElementSource;

      NumericArrayConverter<ElementCompute, ElementSource, 
      OutputOp::kCount, OutputOp::kRound> source_converter;

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kOutputOpIterations; ++i)
      {
        // Call the output operator
        output_frag_ptr[i] = output_op(compute_frag_ptr[i]);
      }
    }


    CUTLASS_DEVICE
    void addM(typename OutputTileIterator::Fragment &output, typename OutputTileIterator::Fragment &in1, typename OutputTileIterator::Fragment &in2,
              int32_t m1, int32_t m2)
    {
      int const kOutputOpIterations =
        OutputTileIterator::Fragment::kElements / OutputTileIterator::kElementsPerAccess;
      OutputAccessType *out_ptr =
        reinterpret_cast<OutputAccessType *>(&output);
      OutputAccessType const *in1_ptr =
        reinterpret_cast<OutputAccessType const *>(&in1);
      OutputAccessType const *in2_ptr =
        reinterpret_cast<OutputAccessType const *>(&in2);
      
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < kOutputOpIterations; j++) {
        if (m1 == 1 and m2 == 1) {
          out_ptr[j] = in1_ptr[j] + in2_ptr[j]; 
        } else if (m1 == -1 and m2 == 1) {
          out_ptr[j] = in2_ptr[j] - in1_ptr[j];
        } else if (m1 == 1 and m2 == -1) {
          out_ptr[j] = in1_ptr[j] - in2_ptr[j];
        }
      }
    }


    CUTLASS_DEVICE
    void addFinalM(typename SharedLoadIterator::Fragment &output, typename OutputTileIterator::Fragment &in)
    {
      int const kOutputOpIterations =
        OutputTileIterator::Fragment::kElements / OutputTileIterator::kElementsPerAccess;
      AccumulatorAccessType*out_ptr =
        reinterpret_cast<AccumulatorAccessType *>(&output);
      OutputAccessType const *in1_ptr =
        reinterpret_cast<OutputAccessType const *>(&in);

      using ElementCompute = typename OutputOp::ElementCompute;
      using ElementSource = typename OutputOp::ElementSource;
      NumericArrayConverter<ElementCompute, ElementSource, 
        OutputOp::kCount, OutputOp::kRound> source_converter;

      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < kOutputOpIterations; j++) {
        out_ptr[j] = out_ptr[j] + source_converter(in1_ptr[j]);
      }
    }
    
    CUTLASS_DEVICE
    void addMs(
      typename OutputTileIterator::Fragment &source_output_fragment,
      typename SharedLoadIterator::Fragment &accum_fragment,
      uint OutputC)
    {
      int const kOutputOpIterations =
        OutputTileIterator::Fragment::kElements / OutputTileIterator::kElementsPerAccess;
        
             if (OutputC == 0) {
        addM(source_output_fragment, source_fragment[0],     source_fragment[3], 1,  1);
        addM(source_output_fragment, source_output_fragment, source_fragment[4], 1, -1);
      } else if (OutputC == 1) {
        OutputAccessType *output_frag_ptr =
        reinterpret_cast<OutputAccessType *>(&source_output_fragment);
        OutputAccessType const *input_frag_ptr =
        reinterpret_cast<OutputAccessType const *>(&source_fragment[2]);
        for (uint i = 0; i < kOutputOpIterations; i++) {
          output_frag_ptr[i] = input_frag_ptr[i];
        }
      } else if (OutputC == 2) {
      OutputAccessType *output_frag_ptr =
        reinterpret_cast<OutputAccessType *>(&source_output_fragment);
        OutputAccessType const *input_frag_ptr =
        reinterpret_cast<OutputAccessType const *>(&source_fragment[3]);
        for (uint i = 0; i < kOutputOpIterations; i++) {
          output_frag_ptr[i] = input_frag_ptr[i];
        }
      } else if (OutputC == 3) {
        addM(source_output_fragment, source_fragment[0], source_fragment[1], 1, -1);
        addM(source_output_fragment, source_output_fragment, source_fragment[2], 1, 1);
        addFinalM(accum_fragment, source_output_fragment);
      }
    }

    // No-op
    template<typename PresumSharedIter>
    CUTLASS_DEVICE
    void loadAndAddMs(OutputTileIterator& source_iterator, 
                      typename SharedLoadIterator::Fragment &accum_fragment,
                      uint64_t size, uint32_t outputC,
                      PresumSharedIter& postSumSharedIter) {
      if (outputC == 0) {
        typename OutputTileIterator::Fragment add_M_fragment;
        add_M_fragment.clear();

        // source_iterator.load_with_byte_offset(source_fragment[0], 0*size);
        // source_iterator.load_with_byte_offset(source_fragment[3], 3*size);
        // source_iterator.load_with_byte_offset(source_fragment[4], 4*size);
        // postSumSharedIter.template load<OutputAccessType>(add_M_fragment);
        // addM(add_M_fragment, source_fragment[0], source_fragment[3], 1, 1);
        // addM(add_M_fragment, add_M_fragment, source_fragment[4], 1, -1);
        // addFinalM(accum_fragment, add_M_fragment);
        // ++postSumSharedIter;
      }

      if (outputC == 1) {
        source_iterator.load_with_byte_offset(source_fragment[2], 2*size);
        addFinalM(accum_fragment, source_fragment[2]);
      }

      if (outputC == 2) {
        source_iterator.load_with_byte_offset(source_fragment[3], 3*size);
        addFinalM(accum_fragment, source_fragment[3]);
      }

      if (outputC == 3) {
        typename OutputTileIterator::Fragment add_M_fragment;
        add_M_fragment.clear();

        source_iterator.load_with_byte_offset(source_fragment[0], 0*size);
        source_iterator.load_with_byte_offset(source_fragment[1], 1*size);
        source_iterator.load_with_byte_offset(source_fragment[2], 2*size);

        addM(add_M_fragment, source_fragment[0], source_fragment[1], 1, -1);
        addM(add_M_fragment, add_M_fragment, source_fragment[2], 1, 1);
        addFinalM(accum_fragment, add_M_fragment);
      }
      
      ++source_iterator;
    }
  };


  /// Aspect for when epilogue source is needed
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
        // Call the output operator
        output_frag_ptr[i] = output_op(compute_frag_ptr[i], source_frag_ptr[i]);
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
  StrassenMatrixEpilogue(
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
    SourceAspectNeeded::apply_output_operator(
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
    // operator()(output_op, destination_iterator, accumulators, SourceAspectNotNeeded());
  }


  /// Perform the epilogue computations and stream the result to global memory.  Implements
  /// two alternative codepaths, depending on whether the output op requires addend data to be loaded.
  CUTLASS_DEVICE
  void operator()(
    OutputOp const &output_op,                      ///< Output operator
    OutputTileIterator destination_iterator,        ///< Tile iterator for destination
    AccumulatorTile const &accumulators,            ///< Complete warp-level accumulator tile
    OutputTileIterator source_iterator)            ///< Tile iterator for addend source
  {
    if (output_op.is_source_needed())
    {
      // operator()<c1, c2, c3, c4, c5, c6, c7>(output_op, destination_iterator, accumulators, SourceAspectNeeded(source_iterator), shape);
    }
    else
    {
      // operator()(output_op, destination_iterator, accumulators, SourceAspectNotNeeded());
    }
  }

  template<MmaStrassen::Type OutputM, typename OutputAccessType>
  CUTLASS_DEVICE
  bool processRWMi(MmaStrassen::Type mi, OutputAccessType& sourceElems, ElementOutput* ptr,
                   uint64_t ReadWriteMi, uint sizeM, MmaStrassen::ReadWriteM RWType) {
    OutputAccessType elemsMi; elemsMi.clear();
    if (mi != OutputM &&
        MmaStrassen::GetRWBit(ReadWriteMi, mi) == RWType) {
      if (RWType == MmaStrassen::RWAccumInSharedInEpilogue)
        elemsMi = *(OutputAccessType*)ptr;
      else
        cutlass::arch::global_load<OutputAccessType,
                                  sizeof(OutputAccessType),
                                  arch::CacheOperation::LastUse>
                            (elemsMi, ptr + (mi - MmaStrassen::GlobalPreSumLevel1_M0)*sizeM, true);
      if (MmaStrassen::GetRWMulSign(ReadWriteMi, mi)) {
        sourceElems = sourceElems - elemsMi;
      } else {
        sourceElems = sourceElems + elemsMi;
      }

      return true;
    }

    return false;
  }

  template<MmaStrassen::Type OutputM, typename OutputAccessType>
  CUTLASS_DEVICE
  bool postsum(OutputAccessType& sourceElems, ElementOutput* ptr,
               uint64_t ReadWriteMi, uint sizeM,
              MmaStrassen::ReadWriteM RWType) {
    bool addAllMs = false;
  
    if (processRWMi<OutputM>(MmaStrassen::GlobalPreSumLevel1_M0, sourceElems, ptr,
                             ReadWriteMi, sizeM, RWType)) {
      // if (OutputM - MmaStrassen::GlobalPreSumLevel1_M0 == 5 && threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 6) {
      //   printf("843 %f\n", (float)sourceElems[0]);
      // }
      addAllMs = true;
    }
    if (processRWMi<OutputM>(MmaStrassen::GlobalPreSumLevel1_M1, sourceElems, ptr,
                             ReadWriteMi, sizeM, RWType)) {
      // if (OutputM - MmaStrassen::GlobalPreSumLevel1_M0 == 5 && threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 6) {
      //   printf("847 %p : %f\n", ptr, (float)sourceElems[0]);
      // }
      addAllMs = true;
    }
    if (processRWMi<OutputM>(MmaStrassen::GlobalPreSumLevel1_M2, sourceElems, ptr,
                             ReadWriteMi, sizeM, RWType)) {
      addAllMs = true;
    }
    if (processRWMi<OutputM>(MmaStrassen::GlobalPreSumLevel1_M3, sourceElems, ptr,
                             ReadWriteMi, sizeM, RWType)) {
      addAllMs = true;
    }
    if (processRWMi<OutputM>(MmaStrassen::GlobalPreSumLevel1_M4, sourceElems, ptr,
                             ReadWriteMi, sizeM, RWType)) {
      addAllMs = true;
    }
    if (processRWMi<OutputM>(MmaStrassen::GlobalPreSumLevel1_M5, sourceElems, ptr,
                             ReadWriteMi, sizeM, RWType)) {
      addAllMs = true;
    }

    if (processRWMi<OutputM>(MmaStrassen::GlobalPreSumLevel1_M6, sourceElems, ptr,
                             ReadWriteMi, sizeM, RWType)) {
      addAllMs = true;
    }
    
    return addAllMs;
    uint outputC = 0;
    if (outputC == 0) {
      OutputAccessType elems0, elems3, elems4;
      elems0.clear(); elems3.clear(); elems4.clear();
      if (MmaStrassen::GetRWBit(ReadWriteMi, MmaStrassen::GlobalPreSumLevel1_M0) ==
          RWType) {
        cutlass::arch::global_load<OutputAccessType,
                                  sizeof(OutputAccessType),
                                  arch::CacheOperation::LastUse>
                            (elems0, ptr + 0*sizeM, true);
        addAllMs = true;
      }
      if (MmaStrassen::GetRWBit(ReadWriteMi, MmaStrassen::GlobalPreSumLevel1_M3) ==
          RWType) {
        cutlass::arch::global_load<OutputAccessType,
                                  sizeof(OutputAccessType),
                                  arch::CacheOperation::LastUse>
                            (elems3, ptr + 3*sizeM, true);
        addAllMs = true;
      }
      if (MmaStrassen::GetRWBit(ReadWriteMi, MmaStrassen::GlobalPreSumLevel1_M4) ==
          RWType) {
        cutlass::arch::global_load<OutputAccessType,
                                  sizeof(OutputAccessType),
                                  arch::CacheOperation::LastUse>
                            (elems4, ptr + 4*sizeM, true);
        addAllMs = true;
      }

      sourceElems = elems0+elems3-elems4;
    } else if (outputC == 1) {
      OutputAccessType elems2; elems2.clear();
      if (MmaStrassen::GetRWBit(ReadWriteMi, MmaStrassen::GlobalPreSumLevel1_M2) ==
          RWType) {
        cutlass::arch::global_load<OutputAccessType,
                                  sizeof(OutputAccessType),
                                  arch::CacheOperation::LastUse>
                            (elems2, ptr + 2*sizeM, true);
        addAllMs = true;
        sourceElems = elems2;
      }
    } else if (outputC == 2) {
      OutputAccessType elems2; elems2.clear();
      if (MmaStrassen::GetRWBit(ReadWriteMi, MmaStrassen::GlobalPreSumLevel1_M3) ==
          RWType) {
        cutlass::arch::global_load<OutputAccessType,
                                  sizeof(OutputAccessType),
                                  arch::CacheOperation::LastUse>
                            (elems2, ptr + 3*sizeM, true);
        addAllMs = true;
        sourceElems = elems2;
      }
    } else if (outputC == 3) {
      OutputAccessType elems0, elems1, elems2;
      elems0.clear(); elems2.clear(); elems1.clear();
      if (MmaStrassen::GetRWBit(ReadWriteMi, MmaStrassen::GlobalPreSumLevel1_M0) ==
          RWType) {
        cutlass::arch::global_load<OutputAccessType,
                                  sizeof(OutputAccessType),
                                  arch::CacheOperation::LastUse>
                            (elems0, ptr + 0*sizeM, true);
        addAllMs = true;
      }
      if (MmaStrassen::GetRWBit(ReadWriteMi, MmaStrassen::GlobalPreSumLevel1_M1) ==
          RWType) {
        cutlass::arch::global_load<OutputAccessType,
                                  sizeof(OutputAccessType),
                                  arch::CacheOperation::LastUse>
                            (elems1, ptr + 1*sizeM, true);
        addAllMs = true;
      }
      if (MmaStrassen::GetRWBit(ReadWriteMi, MmaStrassen::GlobalPreSumLevel1_M2) ==
          RWType) {
        cutlass::arch::global_load<OutputAccessType,
                                  sizeof(OutputAccessType),
                                  arch::CacheOperation::LastUse>
                            (elems2, ptr + 2*sizeM, true);
        addAllMs = true;
      }
      sourceElems = elems0-elems1+elems2;
    }

    return addAllMs;
  }

  template<uint BLOCK_DIM,
           typename MmaShape,
           MmaStrassen::Type MmaStrassenKind, typename PreSumSharedIter, typename Mma>
  CUTLASS_DEVICE
  void update(
    OutputOp const &output_op,                      ///< Output operator
    OutputTileIterator destination_iterator,        ///< Tile iterator for destination
    AccumulatorTile &accumulators,            ///< Complete warp-level accumulator tile
    OutputTileIterator source_iterator,
    MatrixCoord shape,
    bool withoutShMem,
    PreSumSharedIter postSumSharedIter,
    uint block_idx,
    Mma& mma,
    uint64_t ReadWriteMi,
    bool SplitKFirstTB,
    bool SplitKLastTB,
    bool IsSplitKSerial
    )            ///< Tile iterator for addend source
  {
    if (withoutShMem) {
      size_t size = shape.row() * shape.column();
      ElementOutput* ptr = (ElementOutput*)destination_iterator.base_pointer();
      
      // uint linearTBIdx = tb_tile_idx.y * actualGrid.x + tb_tile_idx.x;
      
      uint linearIdx   = block_idx * MmaShape::kM * MmaShape::kN + threadIdx.x * 8;
      ptr = ptr + linearIdx;
      AccumulatorFragmentIterator accum_fragment_iterator(accumulators);
      ElementOutput* shptr = postSumSharedIter.pointer;
      shptr = shptr + threadIdx.x * 8;

      using ElementCompute = typename OutputOp::ElementCompute;
      using ElementSource = typename OutputOp::ElementSource;

      NumericArrayConverter<ElementCompute, ElementSource, 
        OutputOp::kCount, OutputOp::kRound> source_converter;

      if (MmaStrassen::GetRWBit(ReadWriteMi, MmaStrassenKind) ==
          MmaStrassen::ReadWriteM::RWGlobalContig
                                  or
          MmaStrassen::GetRWBit(ReadWriteMi, MmaStrassenKind) ==
          MmaStrassen::ReadWriteM::RWGlobalContigAndShared
                                  or
          (MmaStrassen::GetRWBit(ReadWriteMi, MmaStrassenKind) ==
           MmaStrassen::ReadWriteM::RWShared && IsSplitKSerial)
        ) {
        AccumulatorAccessType *compute_frag_ptr = 
          reinterpret_cast<AccumulatorAccessType *> (&accumulators);
        bool storeToGlobal = 
          MmaStrassen::GetRWBit(ReadWriteMi, MmaStrassenKind) == MmaStrassen::ReadWriteM::RWGlobalContig
                                  or
          MmaStrassen::GetRWBit(ReadWriteMi, MmaStrassenKind) == MmaStrassen::ReadWriteM::RWGlobalContigAndShared;
        bool storeToShared = 
          MmaStrassen::GetRWBit(ReadWriteMi, MmaStrassenKind) == MmaStrassen::ReadWriteM::RWShared
                                  or
          MmaStrassen::GetRWBit(ReadWriteMi, MmaStrassenKind) == MmaStrassen::ReadWriteM::RWGlobalContigAndShared;

        if (!SplitKFirstTB && IsSplitKSerial) {
          ElementOutput* shptr2 = shptr;
          ElementOutput* stPtr = ptr;// + i*BLOCK_DIM;
          for(uint i = 0; i < accumulators.size(); i += 8) {
            int mi = MmaStrassenKind - MmaStrassen::GlobalPreSumLevel1_M0;
            AccumulatorAccessType compute_frag = compute_frag_ptr[i/8];
            OutputAccessType elems2; elems2.clear();
            cutlass::arch::global_load<OutputAccessType,
                                      sizeof(OutputAccessType)>
                                  (elems2, stPtr + mi*size, true);
            compute_frag = compute_frag + source_converter(elems2);
            compute_frag_ptr[i/8] = compute_frag;

            stPtr += 8 * BLOCK_DIM;
            shptr2 += 8 * BLOCK_DIM;
          }
        }

        if (SplitKLastTB && IsSplitKSerial) {
          bool neg;
          if (mma.storeMInSh(&neg) != -1) {
            ElementOutput* shptr2 = shptr;
            for(uint i = 0; i < accumulators.size(); i += 8) {
              OutputAccessType prevM = *(OutputAccessType*)(shptr2);
              bool negAccum = MmaStrassen::GetRWMulSign(ReadWriteMi, MmaStrassenKind);
              float mul = neg ? -1 : 1;
              float mulAccum = negAccum ? -1 : 1;
              compute_frag_ptr[i/8] = mulAccum * compute_frag_ptr[i/8] + mul * source_converter(prevM);
              shptr2 += 8 * BLOCK_DIM;
            }
          }
        }

        ElementOutput* stPtr = ptr;
        for(uint i = 0; i < accumulators.size(); i += 8) {
          // + i*BLOCK_DIM;
          int mi = MmaStrassenKind - MmaStrassen::GlobalPreSumLevel1_M0;
          OutputAccessType elemsi;
          elemsi.clear();
          bool addAllMs = (SplitKLastTB and !storeToGlobal) ? postsum<MmaStrassenKind>(elemsi, stPtr, ReadWriteMi, size,
                                                                    MmaStrassen::ReadWriteM::RWGlobalContig) :
                                                                    false;
          AccumulatorAccessType frag = compute_frag_ptr[i/8];
          if (SplitKLastTB) {
            if (IsSplitKSerial &&
                GetRWBit(ReadWriteMi, MmaStrassenKind) == MmaStrassen::RWShared) {
              *(OutputAccessType*)(shptr) = output_op(frag);
            }
            if (addAllMs)
              frag = frag + source_converter(elemsi);
            if (GetRWBit(ReadWriteMi, MmaStrassenKind) == MmaStrassen::RWGlobalContigAndShared) {
              *(OutputAccessType*)(shptr) = output_op(frag);
            }
          }
          if ((IsSplitKSerial && !SplitKFirstTB) or storeToGlobal)
            cutlass::arch::global_store<OutputAccessType,
                                        sizeof(OutputAccessType)>
                                    (output_op(frag), stPtr + mi*size, true);
          stPtr += 8 * BLOCK_DIM;
          shptr += 8 * BLOCK_DIM;
        }
      } else if (MmaStrassen::GetRWBit(ReadWriteMi, MmaStrassenKind) ==
                 MmaStrassen::ReadWriteM::RWGlobalTensorOpMap) {
        #pragma unroll 8
        // for(uint i = 0; i < accumulators.size(); i += 16) {
        while(accum_fragment_iterator.index_ < 8) {
          ElementOutput* stPtr = ptr;// + i*BLOCK_DIM;

          typename AccumulatorFragmentIterator::Fragment frag;
          accum_fragment_iterator.load(frag);
          
          const AccumulatorAccessType *frag_ptr = 
            reinterpret_cast<const AccumulatorAccessType *> (&frag);
          
          auto elems0 = output_op(frag_ptr[0]);
          auto elems1 = output_op(frag_ptr[1]);

          int mi = MmaStrassenKind - MmaStrassen::GlobalPreSumLevel1_M0;
          cutlass::arch::global_store<OutputAccessType,
                                      sizeof(OutputAccessType)>
                                (elems0, stPtr + mi*size, true);
          cutlass::arch::global_store<OutputAccessType,
                                      sizeof(OutputAccessType)>
                                (elems1, stPtr + mi*size + 8*BLOCK_DIM, true);
          ++accum_fragment_iterator;
          ptr += 16 * BLOCK_DIM;
        }
      }
    }
    else {
      // operator()<c1, c2, c3, c4, c5, c6, c7>(output_op, destination_iterator, accumulators, 
      // SourceAspectNotNeeded<c1, c2, c3, c4, c5, c6, c7>(), shape, withoutShMem);
    }
  }

  template<int c0 = 0, int c1 = 0, int c2 = 0, int c3 = 0>
  CUTLASS_DEVICE
  void update4(
    OutputOp const &output_op,                      ///< Output operator
    OutputTileIterator destination_iterator,        ///< Tile iterator for destination
    AccumulatorTile const &accumulators,            ///< Complete warp-level accumulator tile
    OutputTileIterator source_iterator,
    MatrixCoord shape={0,0},
    uint32_t loadC = ~0)            ///< Tile iterator for addend source
  {
    // Iterator over warp-level accumulator fragment
    AccumulatorFragmentIterator accum_fragment_iterator(accumulators);
    SourceAspectNotNeeded<c0, c1, c2, c3, 0,0,0> source;

    //
    // Iterate over accumulator tile
    //
    #pragma unroll(IterationsUnroll ? OutputTileIterator::kIterations : 1)
    for (int iter = 0; iter < OutputTileIterator::kIterations; ++iter)
    {
      //
      // Load the source
      //
      source.load4(destination_iterator, shape, loadC);
      //
      // Convert and store fragment
      //

      __syncthreads();

      acc2smem<cutlass::make_index_sequence<OutputTileIterator::kIterations>>::push(
        iter, accum_fragment_iterator, this->warp_tile_iterator_);

      __syncthreads();

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

      typename OutputTileIterator::Fragment output_fragment[7];

      source.apply_output_operator(output_fragment, output_op, 
                                   aligned_accum_fragment[0], loadC);
      //
      // Store the final result
      //

      if (c0 != 0) {
        destination_iterator.store(output_fragment[0]);
      }
      if (c1 != 0) {
        destination_iterator.store_with_byte_offset(output_fragment[1], shape.column()/2 * sizeof(OutputTileIterator::Element));
      }
      if (c2 != 0) {
        destination_iterator.store_with_byte_offset(output_fragment[2], shape.row()/2 * shape.column()* sizeof(OutputTileIterator::Element));
      }
      if (c3 != 0) {
        destination_iterator.store_with_byte_offset(output_fragment[3], (shape.row()/2 * shape.column() + shape.column()/2)* sizeof(OutputTileIterator::Element));
      }
      ++destination_iterator;
    }
  }

  template<uint BLOCK_DIM, typename MmaShape, MmaStrassen::Type OutputM, typename PreSumSharedIter, typename Mma>
  CUTLASS_DEVICE
  void updateDandM(
    OutputOp const &output_op,                      ///< Output operator
    OutputTileIterator destination_iterator,        ///< Tile iterator for destination
    OutputTileIterator destination_iterator_M,
    AccumulatorTile &accumulators,            ///< Complete warp-level accumulator tile
    OutputTileIterator source_iterator_M,
    uint outputC_,
    PreSumSharedIter postSumSharedIter,
    dim3 actualBlockIdx,
    dim3 actualGrid,
    uint block_idx,
    Mma& mma,
    MatrixCoord Mshape,
    MatrixCoord Cshape,
    uint64_t ReadWriteMi,
    bool SplitKFirstTB,
    bool SplitKLastTB,
    bool IsSplitKSerial)            ///< Tile iterator for addend source
  {
    // Iterator over warp-level accumulator fragment
    AccumulatorFragmentIterator accum_fragment_iterator(accumulators);
    SourceAspectNotNeeded source;
    size_t sizeM = size_t(Mshape.row()) * size_t(Mshape.column());

    //
    // Iterate over accumulator tile
    //

    if (OutputM == MmaStrassen::GlobalPreSumLevel1_M0) {
      ElementOutput* ptr = (ElementOutput*)destination_iterator_M.base_pointer();
      AccumulatorAccessType *compute_frag_ptr = reinterpret_cast<AccumulatorAccessType *> (&accumulators);

      uint linearIdx   = block_idx * MmaShape::kM * MmaShape::kN + threadIdx.x * 8;
      ptr = ptr + linearIdx;
      ElementOutput* shptr = postSumSharedIter.pointer;
      shptr = shptr + threadIdx.x * 8;

      for(uint i = 0; i < accumulators.size(); i += 8) {
        ElementOutput* stPtr = ptr;// + i*BLOCK_DIM;
        OutputAccessType elems = *(OutputAccessType*)shptr;

        if (!SplitKFirstTB) {
          OutputAccessType elems2;
          cutlass::arch::global_load<OutputAccessType,
                                    sizeof(OutputAccessType)>
                              (elems2, stPtr + (OutputM - MmaStrassen::GlobalPreSumLevel1_M0)*sizeM, true);
          elems = elems + elems2;  
        }
        cutlass::arch::global_store<OutputAccessType,
                                    sizeof(OutputAccessType)>
                              (elems, stPtr + (OutputM - MmaStrassen::GlobalPreSumLevel1_M0)*sizeM, true);
        ptr += 8 * BLOCK_DIM;
        shptr += 8 * BLOCK_DIM;
      }
    } else if (ReadWriteMi != 0) {
      ElementOutput* ptr = (ElementOutput*)destination_iterator_M.base_pointer();
      AccumulatorAccessType *compute_frag_ptr = reinterpret_cast<AccumulatorAccessType *> (&accumulators);

      uint linearIdx   = block_idx * MmaShape::kM * MmaShape::kN + threadIdx.x * 8;
      ptr = ptr + linearIdx;
      ElementOutput* shptr = postSumSharedIter.pointer;
      shptr = shptr + threadIdx.x * 8;

      if (!SplitKFirstTB or !SplitKLastTB) {
        ElementOutput* stPtr = ptr;// + i*BLOCK_DIM;
        for(uint i = 0; i < accumulators.size(); i += 8) {
          using ElementCompute = typename OutputOp::ElementCompute;
          using ElementSource = typename OutputOp::ElementSource;

          NumericArrayConverter<ElementCompute, ElementSource, 
            OutputOp::kCount, OutputOp::kRound> source_converter;


          if (!SplitKFirstTB) {
            OutputAccessType elems2; elems2.clear();
            cutlass::arch::global_load<OutputAccessType,
                                      sizeof(OutputAccessType)>
                                  (elems2, stPtr + (OutputM - MmaStrassen::GlobalPreSumLevel1_M0)*sizeM, true);

            compute_frag_ptr[i/8] = compute_frag_ptr[i/8] + source_converter(elems2);
          }

          if (!SplitKLastTB) {
            auto elems = output_op(compute_frag_ptr[i/8]);

            cutlass::arch::global_store<OutputAccessType,
                                        sizeof(OutputAccessType)>
                                  (elems, stPtr + (OutputM - MmaStrassen::GlobalPreSumLevel1_M0)*sizeM, true);
          }

          stPtr += BLOCK_DIM*8;
        }
      }

      if (SplitKLastTB && IsSplitKSerial) {
        bool storeAccumToShared = GetRWBit(ReadWriteMi, OutputM) == MmaStrassen::RWShared or
                                  GetRWBit(ReadWriteMi, OutputM) == MmaStrassen::RWGlobalContigAndShared;
        if (storeAccumToShared or mma.storeMInSh() != -1) {
          float4* ptr = (float4*)mma.sharedPostSumM.pointer;
          int i = 0;
          while (i < accumulators.size()) {
            float* x = (float*)(&accumulators[i]);
            float4 prevM = ptr[threadIdx.x + (i/8) * BLOCK_DIM];
            bool neg;
            bool negAccum = MmaStrassen::GetRWMulSign(ReadWriteMi, OutputM);
            if (mma.storeMInSh(&neg) != -1) {
              half2* prevMh = (half2*)&prevM;
              for (int j = 0; j < 8; j += 2) {
                float2 xx = __half22float2(prevMh[j/2]);
                float mul = neg ? -1 : 1;
                float mulAccum = negAccum ? -1 : 1;
                x[j]   = mulAccum * x[j]   + mul * xx.x;
                x[j+1] = mulAccum * x[j+1] + mul * xx.y;
              }
            }

            if (storeAccumToShared) {
              half hx[8] = {0};
              for (int j = 0; j < 8; j++) {
                hx[j] = (half)x[j];
              }

              ptr[threadIdx.x + (i/8) * BLOCK_DIM] = *(float4*)&hx;
            }
            i += 8;
          }
        }
      }

      for(uint i = 0; i < accumulators.size(); i += 8) {
        using ElementCompute = typename OutputOp::ElementCompute;
        using ElementSource = typename OutputOp::ElementSource;

        NumericArrayConverter<ElementCompute, ElementSource, 
          OutputOp::kCount, OutputOp::kRound> source_converter;

        ElementOutput* stPtr = ptr;// + i*BLOCK_DIM;

        auto elems = output_op(compute_frag_ptr[i/8]);

        if (SplitKLastTB and
            (MmaStrassen::GetRWBit(ReadWriteMi, OutputM) == MmaStrassen::ReadWriteM::RWGlobalContig or
             MmaStrassen::GetRWBit(ReadWriteMi, OutputM) == MmaStrassen::ReadWriteM::RWGlobalContigAndShared)) {
          cutlass::arch::global_store<OutputAccessType,
                                      sizeof(OutputAccessType)>
                                (elems, stPtr + (OutputM - MmaStrassen::GlobalPreSumLevel1_M0)*sizeM, true);
        }
        
        OutputAccessType elemsi;
        elemsi.clear();

        bool addAllMs = (SplitKLastTB) ?
                         postsum<OutputM>(elemsi, stPtr, ReadWriteMi, sizeM,
                                          MmaStrassen::ReadWriteM::RWGlobalContig) :
                         false;
        bool addAllSharedMs = (SplitKLastTB) ?
                               postsum<OutputM>(elemsi, shptr, ReadWriteMi, sizeM,
                                                MmaStrassen::ReadWriteM::RWAccumInSharedInEpilogue) :
                               false;

        if (SplitKLastTB &&
            MmaStrassen::GetRWBit(ReadWriteMi, OutputM) == MmaStrassen::RWSharedInEpilogue) {
          *(OutputAccessType*)(shptr) = elems;
        }
        
        if (addAllMs || addAllSharedMs) {
          float sign = 1;
          if (MmaStrassen::GetRWMulSign(ReadWriteMi, OutputM)) sign = -1;
          compute_frag_ptr[i/8] = sign * compute_frag_ptr[i/8] + source_converter(elemsi);
        }

        ptr += 8 * BLOCK_DIM;
        shptr += 8 * BLOCK_DIM;
      }
    }

    accum_fragment_iterator = AccumulatorFragmentIterator(accumulators);
    AccumulatorFragmentIterator accum_fragment_iterator1(accumulators);

    ElementOutput* ptr = (ElementOutput*)destination_iterator_M.base_pointer();

    uint linearIdx   = block_idx * MmaShape::kM * MmaShape::kN + threadIdx.x * 8;
    ptr = ptr + linearIdx;

    if (SplitKLastTB) {
      uint outputC = MmaStrassen::GetRWOutputC(ReadWriteMi);

      #pragma unroll(IterationsUnroll ? OutputTileIterator::kIterations : 1)
      for (int iter = 0; iter < OutputTileIterator::kIterations; ++iter)
      {
        //
        // Load the source
        //

        //
        // Convert and store fragment
        //
        typename AccumulatorFragmentIterator::Fragment addFragment; addFragment.clear();
        bool addGlobalTensorOpM = false;
        if (ReadWriteMi != 0) {
          ElementOutput* stPtr = ptr + iter*16*BLOCK_DIM;
          AccumulatorAccessType *add_frag_ptr = reinterpret_cast<AccumulatorAccessType *> (&addFragment);

          if (MmaStrassen::GetRWBit(ReadWriteMi, OutputM) == MmaStrassen::ReadWriteM::RWGlobalTensorOpMap) {
            typename AccumulatorFragmentIterator::Fragment frag;
            accum_fragment_iterator1.load(frag);

            const AccumulatorAccessType *frag_ptr = reinterpret_cast<const AccumulatorAccessType *> (&frag);
            auto elems0 = output_op(frag_ptr[0]);
            auto elems1 = output_op(frag_ptr[1]);

            cutlass::arch::global_store<OutputAccessType,
                                        sizeof(OutputAccessType)>
                                  (elems0, stPtr + (OutputM - MmaStrassen::GlobalPreSumLevel1_M0)*sizeM, true);
            cutlass::arch::global_store<OutputAccessType,
                                        sizeof(OutputAccessType)>
                                  (elems1, stPtr + (OutputM - MmaStrassen::GlobalPreSumLevel1_M0)*sizeM + 8*BLOCK_DIM, true);
            ++accum_fragment_iterator1;
          }
          
          OutputAccessType elemsi[2];
          elemsi[0].clear(); elemsi[1].clear();

          #pragma unroll 2
          for (int ee = 0; ee < 2; ee++) {
            addGlobalTensorOpM = postsum<OutputM>(elemsi[ee], stPtr + ee*8*BLOCK_DIM,
                                                  ReadWriteMi, sizeM,
                                                  MmaStrassen::ReadWriteM::RWGlobalTensorOpMap);
          }

          for (int ee = 0; ee < 2; ee++) {
            using ElementCompute = typename OutputOp::ElementCompute;
            using ElementSource = typename OutputOp::ElementSource;

            NumericArrayConverter<ElementCompute, ElementSource, 
              OutputOp::kCount, OutputOp::kRound> source_converter;
            if (addGlobalTensorOpM) {
              add_frag_ptr[ee] = source_converter(elemsi[ee]);
            }
          }

          // ptr += 16 * BLOCK_DIM;
        }

        __syncthreads();

        if (addGlobalTensorOpM)
          acc2smem<cutlass::make_index_sequence<OutputTileIterator::kIterations>>::push(
            iter, accum_fragment_iterator, this->warp_tile_iterator_, addFragment);
        else
          acc2smem<cutlass::make_index_sequence<OutputTileIterator::kIterations>>::push(
            iter, accum_fragment_iterator, this->warp_tile_iterator_);

        __syncthreads();

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


        //
        // Store the final result
        //

        typename OutputTileIterator::Fragment add_M_fragment;
        add_M_fragment.clear();

        if ((MmaStrassen::GetRWBit(ReadWriteMi, OutputM) == MmaStrassen::ReadWriteM::RWGlobalTensorOpMap && 
             MmaStrassen::GetRWMulSign(ReadWriteMi, OutputM))) {
              //TODO: Fix for RWUseSign
          source.apply_output_operator<-1>(output_fragment, output_op,
                                          aligned_accum_fragment[0],
                                          add_M_fragment, false);
        } else {
          source.apply_output_operator<1>(output_fragment, output_op,
                                          aligned_accum_fragment[0],
                                          add_M_fragment, false);
        }

        if (outputC == 0) {
          destination_iterator.store(output_fragment);
        }
        if (outputC == 1) {
          destination_iterator.store_with_byte_offset(output_fragment, Cshape.column()/2 * sizeof(OutputTileIterator::Element));
        }
        if (outputC == 2) {
          destination_iterator.store_with_byte_offset(output_fragment, Cshape.row()/2 * Cshape.column()* sizeof(OutputTileIterator::Element));
        }
        if (outputC == 3) {
          destination_iterator.store_with_byte_offset(output_fragment, (Cshape.row()/2 * Cshape.column() + Cshape.column()/2)* sizeof(OutputTileIterator::Element));
        }

        //
        // Store the final result
        //
        ++destination_iterator;
      }
    }
  }

  // CUTLASS_DEVICE
  // void finalUpdate(
  //   OutputOp const &output_op,                      ///< Output operator
  //   OutputTileIterator destination_iterator,        ///< Tile iterator for destination
  //   AccumulatorTile const &accumulators,            ///< Complete warp-level accumulator tile
  //   OutputTileIterator source_iterator,
  //   MatrixCoord shape={0,0})            ///< Tile iterator for addend source
  // {
  //   if (output_op.is_source_needed())
  //   {
  //     // operator()<c1, c2, c3, c4, c5, c6, c7>(output_op, destination_iterator, accumulators, SourceAspectNeeded(source_iterator), shape);
  //   }
  //   else
  //   {
  //     finalUpdate(output_op, destination_iterator, accumulators, shape);
  //   }
  // }

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

    // operator()(output_op, destination_iterator, accumulators, SourceAspectNeeded(source_iterator));
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

    template<int Advance>
    CUTLASS_DEVICE
    static void helper(AccumulatorFragmentIterator accum_fragment_iterator,
                      WarpTileIterator &warp_tile_iterator,
                      typename AccumulatorFragmentIterator::Fragment addFrag) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < Advance; i++) {
        ++accum_fragment_iterator;
      }

      typename AccumulatorFragmentIterator::Fragment accum_fragment;

      accum_fragment_iterator.load(accum_fragment);
      accum_fragment = accum_fragment + addFrag;
      ++accum_fragment_iterator;
      warp_tile_iterator.store(accum_fragment);
    }

    CUTLASS_DEVICE
    static void push(size_t pos,
                    AccumulatorFragmentIterator const &iterator_begin,
                    WarpTileIterator &warp_tile_iterator,
                    typename AccumulatorFragmentIterator::Fragment addFrag) {
      int dummy[] = {(pos == Seq) && (helper<Seq>(iterator_begin, warp_tile_iterator, addFrag), 0)...};
    }
  };


  /// Streams the result to global memory
  template <int c0, int c1, int c2, int c3, int c4,
            int c5, int c6, typename SourceAspect>
  CUTLASS_DEVICE
  void operator()(
    OutputOp const &output_op,                      ///< Output operator
    OutputTileIterator destination_iterator,        ///< Tile iterator for destination
    AccumulatorTile &accumulators,            ///< Complete warp-level accumulator tile
    SourceAspect source,
    MatrixCoord shape, bool withoutSharedMem)
  {
    // Iterator over warp-level accumulator fragment
    AccumulatorFragmentIterator accum_fragment_iterator(accumulators);

    size_t size = shape.row() * shape.column() * sizeof(OutputTileIterator::Element);
    //
    // Iterate over accumulator tile
    //
    #pragma unroll(IterationsUnroll ? OutputTileIterator::kIterations : 1)
    for (int iter = 0; iter < OutputTileIterator::kIterations; ++iter)
    {
      //
      // Load the source
      //

      // source.load(destination_iterator);
      //
      // Convert and store fragment
      //

      if (!withoutSharedMem) {
        __syncthreads();

        acc2smem<cutlass::make_index_sequence<OutputTileIterator::kIterations>>::push(
          iter, accum_fragment_iterator, this->warp_tile_iterator_);

        __syncthreads();
      }

      //
      // Load fragments from shared memory
      //

      typename SharedLoadIterator::Fragment aligned_accum_fragment[kPartitionsK];
      
      if (withoutSharedMem) {
        accum_fragment_iterator.load(aligned_accum_fragment[0]);
        ++accum_fragment_iterator;
      } else {
        shared_load_iterator_.load(aligned_accum_fragment[0]);
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

      typename OutputTileIterator::Fragment output_fragment[7];

      source.apply_output_operator(output_fragment, output_op, 
                                   aligned_accum_fragment[0],
                                   0);
      //
      // Store the final result
      //

      if (c0 != 0) {
        destination_iterator.store                 (output_fragment[0]);
      }
      if (c1 != 0) {
        destination_iterator.store_with_byte_offset(output_fragment[1], 1 * size);
      }
      if (c2 != 0) {
        destination_iterator.store_with_byte_offset(output_fragment[2], 2 * size);
      }
      if (c3 != 0) {
        destination_iterator.store_with_byte_offset(output_fragment[3], 3 * size);
      }
      if (c4 != 0) {
        destination_iterator.store_with_byte_offset(output_fragment[4], 4 * size);
      }
      if (c5 != 0) {
        destination_iterator.store_with_byte_offset(output_fragment[5], 5 * size);
      }
      if (c6 != 0) {
        destination_iterator.store_with_byte_offset(output_fragment[6], 6 * size);
      }
      ++destination_iterator;
    }
  }

  CUTLASS_DEVICE
  void finalUpdate(
    OutputOp const &output_op,                      ///< Output operator
    OutputTileIterator destination_iterator,        ///< Tile iterator for destination
    AccumulatorTile const &accumulators,            ///< Complete warp-level accumulator tile
    OutputTileIterator source_iterator,
    MatrixCoord shape,
    cutlass::half_t* d_ptr)
  {
    // Iterator over warp-level accumulator fragment
    AccumulatorFragmentIterator accum_fragment_iterator(accumulators);

    const size_t size = shape.row() * shape.column() *sizeof(OutputTileIterator::Element);
    SourceAspectNotNeededFinalUpdate source;
    //
    // Iterate over accumulator tile
    //
    #pragma unroll(IterationsUnroll ? OutputTileIterator::kIterations : 1)
    for (int iter = 0; iter < OutputTileIterator::kIterations; ++iter)
    {
      //
      // Load the source
      //

      source.load(destination_iterator, size);
      //
      // Convert and store fragment
      //

      __syncthreads();

      acc2smem<cutlass::make_index_sequence<OutputTileIterator::kIterations>>::push(
        iter, accum_fragment_iterator, this->warp_tile_iterator_);

      __syncthreads();

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

      //
      // Compute the output result
      //

      typename OutputTileIterator::Fragment output_fragment[7];

      source.apply_output_operator(output_fragment, output_op, 
                                   aligned_accum_fragment[0]);


      //
      // Store the final result
      //

      destination_iterator.store(output_fragment[0]);
      destination_iterator.store_with_byte_offset(output_fragment[1], 1 * size);
      destination_iterator.store_with_byte_offset(output_fragment[2], 2 * size);
      destination_iterator.store_with_byte_offset(output_fragment[3], 3 * size);
      destination_iterator.store_with_byte_offset(output_fragment[4], 4 * size);
      destination_iterator.store_with_byte_offset(output_fragment[5], 5 * size);
      destination_iterator.store_with_byte_offset(output_fragment[6], 6 * size);
      ++destination_iterator;
    }
  }
};

////////////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace epilogue
} // namespace cutlass

////////////////////////////////////////////////////////////////////////////////
