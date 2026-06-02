#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/array.h"

#include "cutlass/platform/platform.h"

#include "cutlass/gemm/gemm.h"

#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/thread/linear_combination_clamp.h"
#include "cutlass/epilogue/thread/linear_combination_relu.h"
#include "cutlass/epilogue/thread/linear_combination_relu0.h"
#include "cutlass/epilogue/thread/linear_combination_gelu.h"
#include "cutlass/epilogue/thread/linear_combination_sigmoid.h"
#include "cutlass/epilogue/thread/linear_combination_hardswish.h"
#include "cutlass/epilogue/thread/linear_combination_planar_complex.h"

#include "cutlass/epilogue/thread/conversion_op.h"
#include "cutlass/epilogue/thread/reduction_op.h"

#include "cutlass/transform/threadblock/regular_tile_iterator_pitch_linear.h"

#include "cutlass/epilogue/warp/strassen_fragment_iterator_tensor_op.h"
#include "cutlass/epilogue/warp/fragment_iterator_complex_tensor_op.h"
#include "cutlass/epilogue/warp/tile_iterator_tensor_op.h"
#include "cutlass/epilogue/warp/tile_iterator_tensor_op_mixed.h"
#include "cutlass/epilogue/threadblock/default_thread_map_tensor_op.h"
#include "cutlass/epilogue/threadblock/predicated_tile_iterator.h"
#include "cutlass/epilogue/threadblock/strassen_predicated_tile_iterator.h"
#include "cutlass/epilogue/threadblock/predicated_tile_iterator_conv.h"
#include "cutlass/epilogue/threadblock/predicated_tile_iterator_strided_dgrad.h"
#include "cutlass/epilogue/threadblock/predicated_tile_iterator_affine.h"
#include "cutlass/epilogue/threadblock/shared_load_iterator.h"
#include "cutlass/epilogue/threadblock/shared_load_iterator_mixed.h"

#include "cutlass/epilogue/threadblock/default_epilogue_tensor_op.h"

#include "cutlass/epilogue/threadblock/strassen_epilogue.h"

#include "cutlass/epilogue/threadblock/strassen_matrix_epilogue.h"
#include "cutlass/epilogue/threadblock/interleaved_epilogue.h"

#include "cutlass/layout/permute.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace epilogue {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

/// Defines sensible defaults for epilogues for TensorOps.
template <
  typename Shape_,
  typename StrassenShape_,
  typename WarpMmaTensorOp_,
  int PartitionsK,
  typename OutputOp_,
  int ElementsPerAccess,
  int NumThreads
>
struct DefaultStrassenInterimEpilogueTensorOp {

  using Shape = Shape_;
  using StrassenShape = StrassenShape_;
  using WarpMmaTensorOp = WarpMmaTensorOp_;
  using OutputOp = OutputOp_;
  static int const kElementsPerAccess = ElementsPerAccess;
  static int const kPartitionsK = PartitionsK;

  using ElementOutput = typename OutputOp::ElementOutput;
  using LayoutC = typename WarpMmaTensorOp::LayoutC;
  using ElementAccumulator = typename WarpMmaTensorOp::ElementC;

  //
  // Thread map
  //

  using OutputTileIterator = cutlass::epilogue::threadblock::StrassenInterimEpilogueTileIterator<
    StrassenShape,
    OutputOp,
    ElementOutput,
    NumThreads,
    kElementsPerAccess
  >;

  static bool const UseCUDAStore = platform::is_same<ElementOutput, double>::value;

  using SharedTileIterator = cutlass::epilogue::threadblock::StrassenInterimEpilogueSharedTileIterator<
    StrassenShape,
    ElementOutput,
    NumThreads,
    kElementsPerAccess
  >;

  using AccumulatorFragmentIterator = typename platform::conditional<is_complex<ElementOutput>::value,
                                    cutlass::epilogue::warp::FragmentIteratorComplexTensorOp<
                                        typename WarpMmaTensorOp::StrassenShape,
                                        typename WarpMmaTensorOp::Policy::Operator::Shape,
                                        typename WarpMmaTensorOp::Policy::Operator::ElementC,
                                        typename WarpMmaTensorOp::Policy::Operator::FragmentC,
                                        LayoutC>,
                                    cutlass::epilogue::warp::StrassenFragmentIteratorTensorOp<
                                        typename WarpMmaTensorOp::StrassenShape,
                                        typename WarpMmaTensorOp::Policy::Operator::Shape,
                                        typename WarpMmaTensorOp::Policy::Operator::ElementC,
                                        typename WarpMmaTensorOp::Policy::Operator::FragmentC,
                                        LayoutC> >::type;

  using OutputTileThreadMap = typename cutlass::epilogue::threadblock::DefaultThreadMapTensorOp<
    StrassenShape,
    typename WarpMmaTensorOp::StrassenShape,
    kPartitionsK,
    ElementOutput,
    kElementsPerAccess
  >::Type;

  /// Support several implementations depending on structure of epilogue
  using DefaultIterators = detail::DefaultIteratorsTensorOp<
    ElementOutput,
    ElementAccumulator,
    kElementsPerAccess,
    StrassenShape,
    typename WarpMmaTensorOp::StrassenShape,
    typename WarpMmaTensorOp::Policy::Operator::Shape,
    typename OutputTileThreadMap::CompactedThreadMap
  >;

  using WarpTileIterator = typename DefaultIterators::WarpTileIterator;
  using SharedLoadIterator = typename DefaultIterators::SharedLoadIterator;

  /// Hard-coded padding elements added 
  using Padding = cutlass::MatrixShape<0, 64 / sizeof_bits<ElementAccumulator>::value * 4>;
  static int const kFragmentsPerIteration = (kPartitionsK == 1 ? DefaultIterators::kFragmentsPerIteration : 1);

  //
  // Define the epilogue
  //
  using InterimEpilogue = cutlass::epilogue::threadblock::StrassenInterimEpilogue<
    StrassenShape,
    WarpMmaTensorOp,
    kPartitionsK,
    OutputTileIterator,
    SharedTileIterator,
    AccumulatorFragmentIterator,
    WarpTileIterator,
    OutputOp,
    Padding,
    kFragmentsPerIteration
  >;
};

}
}
}