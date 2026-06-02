#pragma once

#include "cutlass/transform/threadblock/predicated_tile_iterator.h"

/*template <typename Shape, typename Element, typename Layout, int AdvanceRank,
          typename ThreadMap, typename AccessType, bool Gather = false,
          typename PermuteLayout = layout::NoPermute>
class StrassenPredicatedTileIterator;

template <typename Shape_, typename Element_, int AdvanceRank,
          typename ThreadMap_, typename AccessType_, bool Gather,
          typename PermuteLayout>
class StrassenPredicatedTileIterator<Shape_, Element_, layout::PitchLinear,
                                          AdvanceRank, ThreadMap_, AccessType_, Gather,
                                          PermuteLayout> : 
  public PredicatedTileAccessIterator<Shape_, Element_, layout::PitchLinear, AdvanceRank, ThreadMap_, AccessType_, Gather, PermuteLayout> 
{

  using Base = PredicatedTileAccessIterator<Shape_, Element_, layout::PitchLinear, AdvanceRank, ThreadMap_, AccessType_, Gather, PermuteLayout>;
  /// Default constructor
  StrassenPredicatedTileIterator() = default;

  /// Constructs a TileIterator from its precomputed state, threadblock offset,
  /// and thread ID
  CUTLASS_HOST_DEVICE
  StrassenPredicatedTileIterator(
      /// Precomputed parameters object
      Params const &params,
      /// Pointer to start of tensor
      Pointer pointer,
      /// Extent of tensor
      TensorCoord extent,
      /// ID of each participating thread
      int thread_id,
      /// Initial offset of threadblock
      TensorCoord const &threadblock_offset,
      /// Gather indices
      int const *indices = nullptr)
      : Base(params, pointer, extent, thread_id, threadblock_offset, indices) {}

  /// Construct a PredicatedTileAccessIterator with zero threadblock offset
  CUTLASS_HOST_DEVICE
  StrassenPredicatedTileIterator(
      /// Precomputed parameters object
      Params const &params,
      /// Pointer to start of tensor
      Pointer pointer,
      /// Extent of tensor
      TensorCoord extent,
      ///< ID of each participating thread
      int thread_id)
      : Base(params, pointer, extent, thread_id, make_Coord(0, 0)) {}
};*/


namespace cutlass {
namespace transform {
namespace threadblock {
/// Specialization of PredicatedTileAccessIterator for row-major data.
///
/// Satisfies: ForwardTileIteratorConcept |
///            ReadableContiguousTileIteratorConcept |
///            WriteableContiguousTileIteratorConcept |
///            MaskedTileIteratorConcept
///
template <typename Shape_, typename Element_, typename Layout_, int AdvanceRank_,
          typename ThreadMap_, int AccessSize_  = ThreadMap_::kElementsPerAccess, 
          bool Gather_ = false, typename PermuteLayout_ = layout::NoPermute>
class StrassenPredicatedTileIterator {
 public:
  using UnderlyingIterator = PredicatedTileIterator
      <Shape_, Element_, Layout_, AdvanceRank_, ThreadMap_, AccessSize_, Gather_,
      PermuteLayout_>;

  using Layout = typename UnderlyingIterator::Layout;
  using TensorCoord = typename UnderlyingIterator::TensorCoord;
  using Pointer = typename UnderlyingIterator::Pointer;
  using LongIndex = typename UnderlyingIterator::LongIndex;
  using AccessType = typename UnderlyingIterator::AccessType;
  using TensorRef = typename UnderlyingIterator::TensorRef;
  using ThreadMap = typename UnderlyingIterator::ThreadMap;
  using Element = typename UnderlyingIterator::Element;
  using Fragment = typename UnderlyingIterator::Fragment;
  using Index = typename Layout::Index;

  static int const kAccessesPerVector = UnderlyingIterator::kAccessesPerVector;

  /// Predicate vector stores mask to guard accesses
  using Mask = typename UnderlyingIterator::Mask;

  /// Parameters object is precomputed state and is host-constructible
  class Params {
   private:
    friend StrassenPredicatedTileIterator;
    // friend UnderlyingIterator;

    /// Parameters object
    typename UnderlyingIterator::Params params_;

   public:

    /// Default constructor
    Params() = default;

    /// Construct the Params object given a pitch-linear tensor's layout
    CUTLASS_HOST_DEVICE
    Params(Layout const &layout)
        : params_(layout){};

    /// Construct the Params object given a pitch-linear tensor's layout
    CUTLASS_HOST_DEVICE
    Params(typename UnderlyingIterator::Params const &base) 
        : params_(base) {}
  };

 private:
  //
  // Data members
  //

  /// Underlying pitch-linear tile iterator
  UnderlyingIterator iterator_;
  uint64_t offset_var;

 public:

  /// Default constructor
  StrassenPredicatedTileIterator() = default;

  /// Constructs a TileIterator from its precomputed state, threadblock offset,
  /// and thread ID

  CUTLASS_HOST_DEVICE
  StrassenPredicatedTileIterator(
      ///< Precomputed parameters object
      Params const &params,
      ///< Pointer to start of tensor
      Pointer pointer,
      ///< Extent of tensor
      TensorCoord extent,
      ///< ID of each participating thread
      int thread_id,
      ///< Initial offset of threadblock
      TensorCoord const &threadblock_offset_base,
      /// Gather indices
      int const *indices = nullptr)
      : iterator_(params.params_, pointer, extent, thread_id, threadblock_offset_base, indices)
        {
          offset_var = 0;
        }

  CUTLASS_HOST_DEVICE
  StrassenPredicatedTileIterator(
      ///< Precomputed parameters object
      Params const &params,
      ///< Pointer to start of tensor
      Pointer pointer,
      ///< Extent of tensor
      TensorCoord extent,
      ///< ID of each participating thread
      int thread_id,
      ///< Initial offset of threadblock
      TensorCoord const &threadblock_offset_base,
      TensorCoord const &matrix_offset0,
      TensorCoord const &matrix_offset1,
      TensorCoord const &matrix_offset2,
      /// Gather indices
      int const *indices = nullptr)
      : iterator_(params.params_, pointer, extent, thread_id, threadblock_offset_base, indices) {
        // offset_var = ((long)matrix_offset.row()) * extent.column() + (long)matrix_offset.column();
        // if (threadIdx.x == 0) printf("%ld\n", offset[1]);
        // UnderlyingIterator iterator_2(params.params_, pointer, extent, thread_id, threadblock_offset_2, indices);
        // offset[2] = iterator_2.get() - iterator_.get();
        // UnderlyingIterator iterator_3(params.params_, pointer, extent, thread_id, threadblock_offset_3, indices);
        // offset[3] = iterator_3.get() - iterator_.get();
      }

  /// Construct a PredicatedTileAccessIterator with zero threadblock offset
  CUTLASS_HOST_DEVICE
  StrassenPredicatedTileIterator(
      Params const &params,  ///< Precomputed parameters object
      Pointer pointer,       ///< Pointer to start of tensor
      TensorCoord extent,    ///< Extent of tensor
      int thread_id          ///< ID of each participating thread
      )
      : StrassenPredicatedTileIterator(params, pointer, extent, thread_id,
                                     make_Coord(0, 0)) {}

    /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the iterator's
  /// internal pointer is reverted to the first "steady state" tile. Subsequent calls
  /// are lightweight and must only update the internal pointer.
  CUTLASS_HOST_DEVICE
  StrassenPredicatedTileIterator &operator++() {
    ++iterator_;
    return *this;
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the iterator's
  /// internal pointer is reverted to the first "steady state" tile. Subsequent calls
  /// are lightweight and must only update the internal pointer.
  CUTLASS_HOST_DEVICE
  StrassenPredicatedTileIterator operator++(int) {
    StrassenPredicatedTileIterator self(*this);
    operator++();
    return self;
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void clear_mask(bool enable = true) {
    iterator_.clear_mask(enable);
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void enable_mask() {
    iterator_.enable_mask();
  }

  /// Sets the predicate mask, overriding value stored in predicate iterator
  CUTLASS_HOST_DEVICE
  void set_mask(Mask const &mask) {
    iterator_.set_mask(mask);
  }

  /// Gets the mask
  CUTLASS_HOST_DEVICE
  void get_mask(Mask &mask) {
    iterator_.get_mask(mask);
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load_with_pointer_offset(Fragment &frag, Index pointer_offset) {
    iterator_.load_with_pointer_offset(frag, pointer_offset);
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load_with_byte_offset(Fragment &frag, LongIndex byte_offset) {
    iterator_.load_with_byte_offset(frag, byte_offset);
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load(Fragment &frag) {
    load_with_pointer_offset(frag, 0);
  }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store_with_pointer_offset(Fragment const &frag, Index pointer_offset) {
    iterator_.store_with_pointer_offset(frag, pointer_offset);
  }
  
  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store_with_byte_offset(Fragment const &frag, LongIndex byte_offset) {
    iterator_.store_with_byte_offset(frag, byte_offset);
  }

  /// Store a fragment to memory
  CUTLASS_DEVICE
  void store(Fragment const &frag) {
    store_with_pointer_offset(frag, 0);
  }
};
}
}
}