#include "cutlass/epilogue/threadblock/predicated_tile_iterator.h"
#include "cutlass/gemm/threadblock/strassen_mma_base.h"

#pragma once

namespace cutlass {
namespace epilogue {
namespace threadblock {
template <
  typename Shape_,           ///< Thread block shape
  typename EpilogueOp_,
  typename Element_,         ///< Element data type
  uint Threads_,
  uint ElementsPerAccess_
>
class StrassenInterimEpilogueTileIterator {
public:
  using Shape = Shape_;

  using Element = Element_;

  static const uint kThreads = Threads_;
 
  using Layout = layout::RowMajor;
  using TensorRef = TensorRef<Element, Layout>;
  using ConstTensorRef = typename TensorRef::ConstTensorRef;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;
  using TensorCoord = MatrixCoord;

  static int const kElementsPerAccess = ElementsPerAccess_;
  static int const kIterations = (Shape::kM*Shape::kN)/(Threads_*kElementsPerAccess);

  /// Fragment object
  using Fragment = Array<
    Element, kElementsPerAccess
    >;

  /// Memory access size
  using AccessType = AlignedArray<Element, kElementsPerAccess>;

  bool mask_;

public:

  //
  // Data members
  //

  Element *base_pointer_;
  char* byte_pointer_;

  /// Extent of the matrix tile in rows
  Index extent_row_;

  /// Extent of the matrix tile in rows
  Index extent_column_;
  
  TensorCoord thread_block_offset_;
  /// A thread's starting row position (assuming steady-state predicates have been computed)
  Index thread_start_row_;

  /// A thread's starting column
  Index thread_start_column_;

  Index next_offset = 0;
  Index extendSize;

  MmaStrassen::MemLayout mem_layout_;
public:

  //
  // Methods
  //
  TensorCoord initial_thread_offset;

  /// Constructor
  CUTLASS_DEVICE
  StrassenInterimEpilogueTileIterator(
    Element *base_pointer,
    TensorCoord extent,
    int thread_idx,
    TensorCoord threadblock_offset,
    MmaStrassen::MemLayout mem_layout
  ): 
    base_pointer_(base_pointer), thread_block_offset_(threadblock_offset), mem_layout_(mem_layout)
  {
    reset(extent, thread_idx, threadblock_offset);
  }

  CUTLASS_DEVICE
  void reset(TensorCoord extent, uint thread_idx, TensorCoord threadblock_offset) {
    // thread_start_column_ = threadblock_offset.column() + (thread_idx * kElementsPerAccess)%Shape::kN;
    // thread_start_row_    = threadblock_offset.row() + (thread_idx * kElementsPerAccess)/Shape::kN;

    // extent_row_ = extent.row();
    // extent_column_ = extent.column();
    // byte_pointer_ = (char*)(base_pointer_ + (thread_start_row_*extent_column_ +
    //                                          thread_start_column_));
    // mask_ = true;

    uint thread_start_ = 0;

    // if (mem_layout_ == MemLayout::LayoutInterim1D) {
      thread_start_ = (threadblock_offset.row() + threadblock_offset.column())*Shape::kM*Shape::kN + (thread_idx * kElementsPerAccess);

      extent_row_ = extent.row();
      extent_column_ = extent.column();
    // }

    byte_pointer_ = (char*)(base_pointer_ + thread_start_);
    mask_ = true;
  }

  CUTLASS_DEVICE
  TensorCoord thread_block_offset() {return thread_block_offset_;}

  CUTLASS_DEVICE
  Element* base_pointer() {return base_pointer_;}

  CUTLASS_DEVICE
  Element* pointer()      {return (Element*)byte_pointer_;}

  /// Adds a pointer offset in units of Element
  // CUTLASS_HOST_DEVICE
  // void add_pointer_offset(LongIndex pointer_offset) {
  //   store_byte_pointer_ += pointer_offset * sizeof_bits<Element>::value / 8;
  //   byte_pointer_ += pointer_offset * sizeof_bits<Element>::value / 8;
  // }

  CUTLASS_DEVICE
  void store(Fragment& frag, int part = 0) {
    char* ptr = byte_pointer_ + part * (extent_row_ * extent_column_) * sizeof(Element);
    cutlass::arch::global_store<Fragment, sizeof(Fragment)>(frag, ptr, true);
  }

  CUTLASS_DEVICE
  void load(Fragment& frag, int part = 0) {
    char* ptr = byte_pointer_ + part * (extent_row_ * extent_column_) * sizeof(Element);
    cutlass::arch::global_load<Fragment, sizeof(Fragment)>(frag, ptr, true);
  }

  CUTLASS_HOST_DEVICE
  char* pointer(int i) {
    return byte_pointer_ + i*(kThreads*kElementsPerAccess) * sizeof(Element);
  }

  /// Advances to the next position to load or store
  CUTLASS_HOST_DEVICE
  StrassenInterimEpilogueTileIterator &operator++() {
    if (kThreads >= Shape::kN/kElementsPerAccess) {
      // byte_pointer_ += (kThreads*kElementsPerAccess)/(Shape::kN) * sizeof(Element) * extent_column_;
      byte_pointer_ += (kThreads*kElementsPerAccess) * sizeof(Element);
    } else {
      //FIXME: Below code path is not probably correct
      // thread_start_column_ += kThreads*kElementsPerAccess;
    }
    return *this;
  }

  CUTLASS_DEVICE
  bool mask() {return mask_;}

  CUTLASS_DEVICE
  bool valid() {return mask_;}

  CUTLASS_DEVICE
  void clear_mask(bool c) {
    if (c) mask_ = false;
  }
};

template <
  typename Shape_,           ///< Thread block shape
  typename Element_,         ///< Element data type
  uint Threads_,
  uint ElementsPerAccess_
>
class StrassenInterimEpilogueSharedTileIterator {
public:
  using Shape = Shape_;

  using Element = Element_;

  static const uint kThreads = Threads_;
 
  using Layout = layout::RowMajor;
  using TensorRef = TensorRef<Element, Layout>;
  using ConstTensorRef = typename TensorRef::ConstTensorRef;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;
  using TensorCoord = MatrixCoord;

  static int const kElementsPerAccess = ElementsPerAccess_;
  static int const kIterations = (Shape::kM*Shape::kN)/(Threads_*kElementsPerAccess);

  /// Fragment object
  using Fragment = Array<
    Element, kElementsPerAccess
    >;

  /// Memory access size
  using AccessType = AlignedArray<Element, kElementsPerAccess>;

  bool mask_;

public:

  //
  // Data members
  //

  Element *base_pointer_;
  char* byte_pointer_;

  /// Extent of the matrix tile in rows
  Index extent_row_;

  /// Extent of the matrix tile in rows
  Index extent_column_;
  
  /// A thread's starting row position (assuming steady-state predicates have been computed)
  Index thread_start_row_;

  /// A thread's starting column
  Index thread_start_column_;

  Index next_offset = 0;
  Index extendSize;

  int thread_idx;

public:

  //
  // Methods
  //
  TensorCoord initial_thread_offset;

  /// Constructor
  CUTLASS_DEVICE
  StrassenInterimEpilogueSharedTileIterator(
    Element *pointer,
    int thread_idx
  ): 
    base_pointer_(pointer), thread_idx(thread_idx)
  {
    reset();
  }

  CUTLASS_DEVICE
  void reset() {
    thread_start_column_ = (thread_idx * kElementsPerAccess)%Shape::kN;
    thread_start_row_    = (thread_idx * kElementsPerAccess)/Shape::kN;

    extent_row_ = Shape::kM;
    extent_column_ = Shape::kN;
    byte_pointer_ = (char*)(base_pointer_ + (thread_start_row_*extent_column_ +
                                             thread_start_column_));
    mask_ = true;
  }

  CUTLASS_DEVICE
  Element* base_pointer() {return base_pointer_;}

  CUTLASS_DEVICE
  char* smem_pointer(int i)      {
    return byte_pointer_ + i*(kThreads*kElementsPerAccess)/(Shape::kN) * sizeof(Element) * extent_column_;
  }

  /// Adds a pointer offset in units of Element
  // CUTLASS_HOST_DEVICE
  // void add_pointer_offset(LongIndex pointer_offset) {
  //   store_byte_pointer_ += pointer_offset * sizeof_bits<Element>::value / 8;
  //   byte_pointer_ += pointer_offset * sizeof_bits<Element>::value / 8;
  // }

  CUTLASS_DEVICE
  void store(Fragment& frag) {
    *(Fragment*)byte_pointer_ = frag;
    // cutlass::arch::global_store<Fragment, sizeof(Fragment)>(frag, byte_pointer_, true);
  }

  CUTLASS_DEVICE
  void load(Fragment& frag) {
    frag = *(Fragment*)byte_pointer_;
    // cutlass::arch::global_load<Fragment, sizeof(Fragment)>(frag, byte_pointer_, true);
  }

  CUTLASS_HOST_DEVICE
  void set_iteration(int i) {
    byte_pointer_ += i*(kThreads*kElementsPerAccess)/(Shape::kN) * sizeof(Element) * extent_column_;
  }

  /// Advances to the next position to load or store
  CUTLASS_HOST_DEVICE
  StrassenInterimEpilogueSharedTileIterator &operator++() {
    if (kThreads >= Shape::kN/kElementsPerAccess) {
      byte_pointer_ += (kThreads*kElementsPerAccess)/(Shape::kN) * sizeof(Element) * extent_column_;
    } else {
      //FIXME: Below code path is not probably correct
      // thread_start_column_ += kThreads*kElementsPerAccess;
    }
    return *this;
  }

  CUTLASS_DEVICE
  bool mask() {return mask_;}

  CUTLASS_DEVICE
  void clear_mask(bool c) {
    if (c) mask_ = false;
  }
};

}
}
}