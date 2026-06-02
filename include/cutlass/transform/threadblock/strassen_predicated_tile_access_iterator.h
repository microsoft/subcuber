#pragma once

#include "cutlass/transform/threadblock/predicated_tile_access_iterator.h"

/*template <typename Shape, typename Element, typename Layout, int AdvanceRank,
          typename ThreadMap, typename AccessType, bool Gather = false,
          typename PermuteLayout = layout::NoPermute>
class StrassenPredicatedTileAccessIterator;

template <typename Shape_, typename Element_, int AdvanceRank,
          typename ThreadMap_, typename AccessType_, bool Gather,
          typename PermuteLayout>
class StrassenPredicatedTileAccessIterator<Shape_, Element_, layout::PitchLinear,
                                          AdvanceRank, ThreadMap_, AccessType_, Gather,
                                          PermuteLayout> : 
  public PredicatedTileAccessIterator<Shape_, Element_, layout::PitchLinear, AdvanceRank, ThreadMap_, AccessType_, Gather, PermuteLayout> 
{

  using Base = PredicatedTileAccessIterator<Shape_, Element_, layout::PitchLinear, AdvanceRank, ThreadMap_, AccessType_, Gather, PermuteLayout>;
  /// Default constructor
  StrassenPredicatedTileAccessIterator() = default;

  /// Constructs a TileIterator from its precomputed state, threadblock offset,
  /// and thread ID
  CUTLASS_HOST_DEVICE
  StrassenPredicatedTileAccessIterator(
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
  StrassenPredicatedTileAccessIterator(
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
          typename ThreadMap_, typename AccessType_, bool Gather_ = false,
          typename PermuteLayout_ = layout::NoPermute>
class StrassenPredicatedTileAccessIterator {
 public:
  using UnderlyingIterator = PredicatedTileAccessIterator
      <Shape_, Element_, Layout_, AdvanceRank_, ThreadMap_, AccessType_, Gather_,
      PermuteLayout_>;

  using Layout = typename UnderlyingIterator::Layout;
  using TensorCoord = typename UnderlyingIterator::TensorCoord;
  using Pointer = typename UnderlyingIterator::Pointer;
  using LongIndex = typename UnderlyingIterator::LongIndex;
  using AccessType = typename UnderlyingIterator::AccessType;
  using TensorRef = typename UnderlyingIterator::TensorRef;
  using ThreadMap = typename UnderlyingIterator::ThreadMap;
  using Element = typename UnderlyingIterator::Element;

  static int const kAccessesPerVector = UnderlyingIterator::kAccessesPerVector;

  /// Predicate vector stores mask to guard accesses
  using Mask = typename UnderlyingIterator::Mask;

  /// Parameters object is precomputed state and is host-constructible
  class Params {
   private:
    friend StrassenPredicatedTileAccessIterator;
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
  UnderlyingIterator iterators_[4];
  uint64_t offset_var;

 public:

  /// Default constructor
  StrassenPredicatedTileAccessIterator() = default;

  /// Constructs a TileIterator from its precomputed state, threadblock offset,
  /// and thread ID

  CUTLASS_HOST_DEVICE
  StrassenPredicatedTileAccessIterator(
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
      : iterators_{UnderlyingIterator(params.params_, pointer, extent, thread_id, threadblock_offset_base, indices),
                   UnderlyingIterator(params.params_, pointer, extent, thread_id, threadblock_offset_base, indices),
                   UnderlyingIterator(params.params_, pointer, extent, thread_id, threadblock_offset_base, indices),
                   UnderlyingIterator(params.params_, pointer, extent, thread_id, threadblock_offset_base, indices)}
        {
          offset_var = 0;
        }

  CUTLASS_HOST_DEVICE
  StrassenPredicatedTileAccessIterator(
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
      TensorCoord const matrix_offset0 = {0,0},
      TensorCoord const matrix_offset1 = {0,0},
      TensorCoord const matrix_offset2 = {0,0},
      // TensorCoord const &threadblock_offset_2,
      // TensorCoord const &threadblock_offset_3,
      /// Gather indices
      int const *indices = nullptr)
      : 
        iterators_{UnderlyingIterator(params.params_, pointer, extent, thread_id, threadblock_offset_base, indices),
                   UnderlyingIterator(params.params_, pointer, extent, thread_id, threadblock_offset_base + matrix_offset0, indices),
                   UnderlyingIterator(params.params_, pointer, extent, thread_id, threadblock_offset_base + matrix_offset1, indices),
                   UnderlyingIterator(params.params_, pointer, extent, thread_id, threadblock_offset_base + matrix_offset2, indices)}
      {
        // if (threadIdx.x == 0) printf("%ld\n", offset[1]);
        // UnderlyingIterator iterators_2(params.params_, pointer, extent, thread_id, threadblock_offset_2, indices);
        // offset[2] = iterators_2.get() - iterators_.get();
        // UnderlyingIterator iterators_3(params.params_, pointer, extent, thread_id, threadblock_offset_3, indices);
        // offset[3] = iterators_3.get() - iterators_.get();
      }

  /// Construct a PredicatedTileAccessIterator with zero threadblock offset
  CUTLASS_HOST_DEVICE
  StrassenPredicatedTileAccessIterator(
      Params const &params,  ///< Precomputed parameters object
      Pointer pointer,       ///< Pointer to start of tensor
      TensorCoord extent,    ///< Extent of tensor
      int thread_id          ///< ID of each participating thread
      )
      : StrassenPredicatedTileAccessIterator(params, pointer, extent, thread_id,
                                     make_Coord(0, 0)) {}

  /// Overrides the internal iteration index
  CUTLASS_HOST_DEVICE
  void set_iteration_index(int index) {
    for (int i = 0; i < 4; i++) {
      iterators_[i].set_iteration_index(index);
    }
  }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    for (int i = 0; i < 4; i++) {
      iterators_[i].add_pointer_offset(pointer_offset);
    }
  }

  /// Advances an iterator along logical dimensions of matrix in units of whole
  /// tiles
  CUTLASS_HOST_DEVICE
  void add_tile_offset(TensorCoord const &tile_offset) {
    for (int i = 0; i < 4; i++) {
      iterators_[i].add_tile_offset(tile_offset);
    }
  }

  /// Returns a pointer
  CUTLASS_HOST_DEVICE
  AccessType *get(uint part = 0) const {
    return reinterpret_cast<AccessType *>(iterators_[part].get());
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  StrassenPredicatedTileAccessIterator &operator++() {
    ++iterators_[0];
    return *this;
  }

  CUTLASS_HOST_DEVICE
  StrassenPredicatedTileAccessIterator &inc(int part) {
    ++iterators_[part];
    return *this;
  }

  /// Advances to the next tile in memory.
  ///
  /// The first time this method is called, predicates are updated, and the
  /// iterator's internal pointer is reverted to the first "steady state" tile.
  /// Subsequent calls are lightweight and must only update the internal
  /// pointer.
  CUTLASS_HOST_DEVICE
  StrassenPredicatedTileAccessIterator operator++(int) {
    StrassenPredicatedTileAccessIterator self(*this);
    operator++();
    return self;
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void clear_mask(bool enable = true) { 
    for (int i = 0; i < 4; i++) {
      iterators_[i].clear_mask(enable);
    }
  }

  /// Clears the predicate set efficiently
  CUTLASS_HOST_DEVICE
  void enable_mask() {
    for (int i = 0; i < 4; i++) {
      iterators_[i].enable_mask();
    }
  }

  /// Sets the predicate mask, overriding value stored in predicate iterator
  CUTLASS_HOST_DEVICE
  void set_mask(Mask const &mask) { 
    for (int i = 0; i < 4; i++) {
      iterators_[i].set_mask(mask);
    }
  }

  /// Gets the mask
  CUTLASS_HOST_DEVICE
  void get_mask(Mask &mask) {
    for (int i = 0; i < 4; i++) {
      iterators_[i].get_mask(mask);
    }
  }

  /// Returns whether access is valid or not
  CUTLASS_HOST_DEVICE
  bool valid(int part = 0) {
    return iterators_[part].valid();
  }
};
}
}
}