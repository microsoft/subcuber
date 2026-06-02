#pragma once

#include "cutlass/transform/threadblock/regular_tile_access_iterator.h"

namespace cutlass {
namespace transform {
namespace threadblock {
template <typename Shape_, typename Element_, typename Layout_, int AdvanceRank_,
          typename ThreadMap_,
          int Alignment_ =
              sizeof_bits<Element_>::value* ThreadMap_::kElementsPerAccess / 8>
class StrassenRegularTileAccessIterator {
/// Underlying iterator type
  using UnderlyingIterator = RegularTileAccessIterator<
      Shape_, Element_,
      Layout_,
      AdvanceRank_, 
      ThreadMap_>;

  using Shape = typename UnderlyingIterator::Shape;
  using Element = typename UnderlyingIterator::Element;
  using Layout = typename UnderlyingIterator::Layout;
  static int const kAdvanceRank = UnderlyingIterator::kAdvanceRank;
  static int const kAlignment = UnderlyingIterator::kAlignment;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;

  using TensorRef = TensorRef<Element, Layout>;
  using TensorCoord = typename Layout::TensorCoord;

  using ThreadMap = typename UnderlyingIterator::ThreadMap;

  using AccessType = typename UnderlyingIterator::AccessType;

 private:

  /// Underlying iterator
  UnderlyingIterator iterator_;

 public:
  /// Construct a TileIterator with zero threadblock offset
  CUTLASS_HOST_DEVICE
  StrassenRegularTileAccessIterator(TensorRef ref,  ///< Pointer to start of tensor
                            int thread_id   ///< ID of each participating thread
                            )
      : iterator_({ref.data(), ref.stride()}, thread_id) {}

  /// Overrides the internal iteration index
  CUTLASS_HOST_DEVICE
  void set_iteration_index(int index) { iterator_.set_iteration_index(index); }

  /// Adds a pointer offset in units of Element
  CUTLASS_HOST_DEVICE
  void add_pointer_offset(LongIndex pointer_offset) {
    iterator_.add_pointer_offset(pointer_offset);
  }

  /// Returns a pointer
  CUTLASS_HOST_DEVICE
  AccessType *get() const {
    return reinterpret_cast<AccessType *>(iterator_.get());
  }

  /// Adds a tile offset
  CUTLASS_DEVICE
  void add_tile_offset(TensorCoord const &coord) {
    iterator_.add_tile_offset({coord.row(), coord.column()});
  }

  /// Advances to the next tile in memory.
  CUTLASS_HOST_DEVICE
  StrassenRegularTileAccessIterator &operator++() {
    ++iterator_;
    return *this;
  }

  /// Advances to the next tile in memory.
  CUTLASS_HOST_DEVICE
  StrassenRegularTileAccessIterator operator++(int) {
    StrassenRegularTileAccessIterator prev(*this);
    ++iterator_;

    return prev;
  }
};
}
}
}