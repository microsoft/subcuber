#include "cutlass/epilogue/threadblock/predicated_tile_iterator.h"

#pragma once

namespace cutlass {
namespace epilogue {
namespace threadblock {
  template <
  typename ThreadMap_,       ///< Thread map (conept: OutputTileThreadMap)
  typename Element_,         ///< Element data type
  bool ScatterD = false,     ///< Scatter D operand or not
  typename PermuteDLayout = layout::NoPermute, ///< Permute D operand or not
  bool UseCUDAStore = false
>
class StrassenPredicatedTileIterator {
public:
  using ThreadMap = ThreadMap_;
  using Shape = typename ThreadMap::Shape;

  using Element = Element_;

  using Layout = layout::RowMajor;
  using TensorRef = TensorRef<Element, Layout>;
  using ConstTensorRef = typename TensorRef::ConstTensorRef;

  using Index = typename Layout::Index;
  using LongIndex = typename Layout::LongIndex;
  using TensorCoord = MatrixCoord;

  static int const kElementsPerAccess = ThreadMap::kElementsPerAccess;
  static int const kThreads = ThreadMap::kThreads;
  static int const kIterations = ThreadMap::Count::kTile;

  static bool constexpr PermuteD = !layout::is_trivial_permute<PermuteDLayout>;

  static_assert( ThreadMap::Iterations::kRow > 0,"ThreadMap::Iterations::kRow must be > 0");
  static_assert( ThreadMap::Iterations::kGroup > 0,"ThreadMap::Iterations::kGroup must be > 0");
  static_assert( ThreadMap::Iterations::kCluster > 0,"ThreadMap::Iterations::kCluster must be > 0");
  static_assert( ThreadMap::Iterations::kColumn > 0,"ThreadMap::Iterations::kColumn must be > 0");

  /// Fragment object
  using Fragment = Array<
    Element,
    ThreadMap::Iterations::kColumn *
    ThreadMap::Iterations::kRow *
    ThreadMap::Iterations::kGroup *
    ThreadMap::Iterations::kCluster * ThreadMap::kElementsPerAccess>;

  using HalfFragment = Array<
    Element,
    (ThreadMap::Iterations::kColumn *
    ThreadMap::Iterations::kRow *
    ThreadMap::Iterations::kGroup *
    ThreadMap::Iterations::kCluster * ThreadMap::kElementsPerAccess)/2>;

  /// Memory access size
  using AccessType = AlignedArray<Element, ThreadMap::kElementsPerAccess>;

  //
  // Parameters struct
  //

  /// Uses a non-template class
  struct Params : PredicatedTileIteratorParams {
    using Base = PredicatedTileIteratorParams;

    CUTLASS_HOST_DEVICE
    Params() { }

    CUTLASS_HOST_DEVICE
    Params(Layout const &layout):
      PredicatedTileIteratorParams(
        layout.stride(0) * int(sizeof(AccessType)) / kElementsPerAccess,
        make_OutputTileThreadMapDesc<ThreadMap>()
      ) 
    { }

    CUTLASS_HOST_DEVICE
    Params(Layout const &layout,
           // Not needed.  Added to be compatible with strided conv epilogue.
           conv::Conv2dProblemSize const &problem_size):
      Params(layout)
    { }

    CUTLASS_HOST_DEVICE
    Params(Layout const &layout,
           // Not needed.  Added to be compatible with strided conv epilogue.
           conv::Conv3dProblemSize const &problem_size):
      Params(layout)
    { }

    CUTLASS_HOST_DEVICE
    Params(Base const &base) : 
      Base(base) { }
  };

  /// Mask object
  struct Mask {

    static int const kCount = ThreadMap::Iterations::kColumn;

    /// Predicate state
    bool predicates[kCount];

    //
    // Mask
    //
    CUTLASS_HOST_DEVICE
    Mask() {
      enable();
    }

    ///< Efficiently disables all accesses guarded by mask
    CUTLASS_HOST_DEVICE void clear() {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kCount; ++i) {
        predicates[i] = false;
      }
    }

    ///< CUTLASS_HOST_DEVICE enables all accesses guarded by mask
    CUTLASS_DEVICE void enable() {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < kCount; ++i) {
        predicates[i] = true;
      }
    }
  };

public:

  //
  // Data members
  //

  /// Parameters structure containing reference and precomputed state.
  PredicatedTileIteratorParams params_;

  Element *base_pointer_;
  /// Byte-level pointer. This pointer is usually for both load() and store(), unless PermuteD is performed. When having PermuteD, byte_pointer_ is only for load().
  uint8_t *byte_pointer_;
  uint8_t *initial_byte_pointer_;
  /// Byte-level pointer for store(). Due to PermuteD Op, store_byte_pointer_ may be with different address computation compared to byte_pointer_.
  uint8_t *store_byte_pointer_;

  /// Array of boolean values to contain steady-state predicates
  Mask mask_;

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
  /// Internal state counter
  int state_[3];

  /// Scatter indices
  int const *indices_;

  /// PermuteDLayout
  PermuteDLayout permute_layout_;

  //
  // Static asserts about internal strides
  //

  static_assert(sizeof(extent_row_) == 4, "Expected 32b extents");
  static_assert(sizeof(thread_start_row_) == 4, "Expected 32b extents");
  static_assert(sizeof(PredicatedTileIteratorParams::stride) == 8, "Expected 64b strides");

private:

  //
  // Methods
  //

public:

  //
  // Methods
  //
  TensorCoord initial_thread_offset;

  /// Constructor
  CUTLASS_DEVICE
  StrassenPredicatedTileIterator(
    PredicatedTileIteratorParams const & params,
    Element *pointer,
    TensorCoord extent,
    int thread_idx,
    TensorCoord threadblock_offset = TensorCoord(),
    TensorCoord matrix_offset = TensorCoord(),
    int const *indices = nullptr
  ): 
    params_(params), base_pointer_(pointer), indices_(indices), thread_block_offset_(threadblock_offset),
    permute_layout_(PitchLinearCoord(extent.column(), extent.row()), params_.stride * kElementsPerAccess / sizeof(AccessType))
  {
    next_offset = matrix_offset.row() * extent.column() + matrix_offset.column();
    TensorCoord thread_offset = ThreadMap::initial_offset(thread_idx) + threadblock_offset;
    initial_thread_offset = thread_offset;
    extendSize = extent.column() * extent.row() * sizeof(Element);
    extent_row_ = extent.row();
    extent_column_ = extent.column();

    thread_start_row_ = thread_offset.row();
    thread_start_column_ = thread_offset.column();

    // Initialize predicates
    CUTLASS_PRAGMA_UNROLL
    for (int c = 0; c < ThreadMap::Iterations::kColumn; ++c) {

      mask_.predicates[c] = ((thread_offset.column()
        + ThreadMap::Delta::kColumn * c) < extent.column());
    }

    // Null pointer performs no accesses
    if (!pointer) {
      mask_.clear();
    }

    if (ScatterD && !indices) {
      mask_.clear();
    }
    // __syncthreads();
    // if (threadIdx.x % 16 == 0 && blockIdx.x == 0 && blockIdx.y == 0)
    //   printf("225: %d : %d %d : %p %ld\n", threadIdx.x, thread_offset.row(), thread_offset.column(), pointer, params_.stride);
    // __syncthreads();
    // Initialize byte_pointer_
    byte_pointer_ = reinterpret_cast<uint8_t *>(pointer) +
      LongIndex(thread_offset.row()) * LongIndex(params_.stride) +
      LongIndex(thread_offset.column()) * sizeof(AccessType) / kElementsPerAccess;
    initial_byte_pointer_ = byte_pointer_;
    if (ScatterD) {
      byte_pointer_ = reinterpret_cast<uint8_t *>(pointer) +
        LongIndex(thread_offset.column()) * sizeof(AccessType) / kElementsPerAccess;
    }

    // store_byte_pointer_ is set to be the same with byte_pointer_ unless PermuteD is used.
    store_byte_pointer_ = PermuteD ? reinterpret_cast<uint8_t *>(pointer) : byte_pointer_;

    // Initialize internal state counter
    state_[0] = state_[1] = state_[2] = 0;
  }

  CUTLASS_DEVICE
  void reset() {state_[0] = state_[1] = state_[2] = 0;
    thread_start_row_ = initial_thread_offset.row();
    thread_start_column_ = initial_thread_offset.column();
    byte_pointer_ = initial_byte_pointer_;
  }

  CUTLASS_DEVICE
  StrassenPredicatedTileIterator(
    PredicatedTileIteratorParams const & params,
    Element *pointer,
    TensorCoord extent,
    int thread_idx,
    TensorCoord threadblock_offset = TensorCoord(),
    int const *indices = nullptr
  ): StrassenPredicatedTileIterator(params, pointer, extent, thread_idx, threadblock_offset, {0,0}, indices) {} 

  CUTLASS_DEVICE
  TensorCoord thread_block_offset() {return thread_block_offset_;}
  CUTLASS_DEVICE
  Element* base_pointer() {return base_pointer_;}

  // /// Adds a pointer offset in units of Element
  // CUTLASS_HOST_DEVICE
  // void add_pointer_offset(LongIndex pointer_offset) {
  //   store_byte_pointer_ += pointer_offset * sizeof_bits<Element>::value / 8;
  //   byte_pointer_ += pointer_offset * sizeof_bits<Element>::value / 8;
  // }

  CUTLASS_DEVICE
  void load_linear_with_offset(Fragment &frag, int64_t offset, uint linearIdx, uint iter) const {
    AccessType *frag_ptr = reinterpret_cast<AccessType *>(&frag);
    Element* ptr = base_pointer_ + offset + linearIdx + iter * 8 * 256;
    cutlass::arch::global_load<AccessType, sizeof(AccessType)>(frag_ptr[0], (void*)ptr, true);
    cutlass::arch::global_load<AccessType, sizeof(AccessType)>(frag_ptr[1], (void*)(ptr + 8*256), true);
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load_with_byte_offset(Fragment &frag, Fragment &frag2, int64_t byte_offset) const {

    uint8_t *byte_pointer = byte_pointer_;
    AccessType *frag_ptr = reinterpret_cast<AccessType *>(&frag);
    AccessType *frag2_ptr = reinterpret_cast<AccessType *>(&frag2);
  
    CUTLASS_PRAGMA_UNROLL
    for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster; ++cluster) {

      CUTLASS_PRAGMA_UNROLL
      for (int group = 0; group < ThreadMap::Iterations::kGroup; ++group) {

        CUTLASS_PRAGMA_UNROLL
        for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {

          int frag_row_idx =
            (row + ThreadMap::Iterations::kRow * (group + ThreadMap::Iterations::kGroup * cluster));

          int row_offset = row * ThreadMap::Delta::kRow 
            + group * ThreadMap::Delta::kGroup 
            + cluster * ThreadMap::Delta::kCluster;

          bool row_guard = ((row_offset + thread_start_row_) < extent_row_);

          AccessType *memory_pointer = reinterpret_cast<AccessType *>(byte_pointer + byte_offset);

          if (ScatterD && row_guard) {
            assert(indices_);

            memory_pointer = reinterpret_cast<AccessType *>(byte_pointer + byte_offset +
              LongIndex(indices_[row_offset + thread_start_row_]) * LongIndex(params_.stride));
          }

          CUTLASS_PRAGMA_UNROLL
          for (int column = 0; column < ThreadMap::Iterations::kColumn; ++column) {

            bool guard = row_guard && mask_.predicates[column];

            cutlass::arch::global_load<
              AccessType,
              sizeof(AccessType)
            >(
                frag_ptr[frag_row_idx * ThreadMap::Iterations::kColumn +
                         column],
                (void *)&memory_pointer[column * ThreadMap::Delta::kColumn /
                                        kElementsPerAccess],
                guard);
            // __syncthreads();
            // if (threadIdx.x % 16 == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
            //   printf("323 %d, %d : %p %p\n", threadIdx.x, row, byte_pointer_, &memory_pointer[column * ThreadMap::Delta::kColumn /
            //                             kElementsPerAccess]);
            // }
            // __syncthreads();
            if (next_offset != 0) {
              cutlass::arch::global_load<
                AccessType,
                sizeof(AccessType)
              >(
                  frag2_ptr[frag_row_idx * ThreadMap::Iterations::kColumn +
                            column],
                  (void *)&memory_pointer[column * ThreadMap::Delta::kColumn /
                                          kElementsPerAccess + next_offset],
                  guard);
            }
          }

          if (row + 1 < ThreadMap::Iterations::kRow) {
            if (!ScatterD) {
              byte_pointer += params_.increment_row;
            }
          }
        }

        if (group + 1 < ThreadMap::Iterations::kGroup) {
          byte_pointer += params_.increment_group;
        }
      }

      if (cluster + 1 < ThreadMap::Iterations::kCluster) {
        byte_pointer += params_.increment_cluster;
      }
    }
  }

  CUTLASS_DEVICE
  void cp_async_presum(void *smem_ptr, void const *global_ptr, bool pred_guard = true) const {
    #if CUDA_CP_ASYNC_ACTIVATED
      unsigned smem_int_ptr = cutlass::arch::cutlass_get_smem_pointer(smem_ptr);
      asm volatile(
          "{\n"
          //"  .reg .pred p;\n"
          //"  setp.ne.b32 p, %0, 0;\n"
          "  cp.async.cg.shared.global [%1], [%2], %3;\n"
          "}\n" ::"r"((int)pred_guard),
          "r"(smem_int_ptr), "l"(global_ptr), "n"(16));
    #endif
  }

  // Loads a fragment from memory
  CUTLASS_DEVICE
  void load_in_shmem(void* ptr1, void* ptr2, int Mi) const {

    uint8_t *byte_pointer = byte_pointer_;
    uint byte_offset = Mi * extendSize;

    CUTLASS_PRAGMA_UNROLL
    for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster; ++cluster) {

      CUTLASS_PRAGMA_UNROLL
      for (int group = 0; group < ThreadMap::Iterations::kGroup; ++group) {

        CUTLASS_PRAGMA_UNROLL
        for (int row = 0; row < ThreadMap::Iterations::kRow; ++row)
        {

          int frag_row_idx =
            (ThreadMap::Iterations::kRow * (group + ThreadMap::Iterations::kGroup * cluster));

          int row_offset = row * ThreadMap::Delta::kRow 
            + group * ThreadMap::Delta::kGroup 
            + cluster * ThreadMap::Delta::kCluster;

          bool row_guard = ((row_offset + thread_start_row_) < extent_row_);

          AccessType *memory_pointer = reinterpret_cast<AccessType *>(byte_pointer + byte_offset);

          if (ScatterD && row_guard) {
            assert(indices_);

            memory_pointer = reinterpret_cast<AccessType *>(byte_pointer + byte_offset +
              LongIndex(indices_[row_offset + thread_start_row_]) * LongIndex(params_.stride));
          }

          CUTLASS_PRAGMA_UNROLL
          for (int column = 0; column < ThreadMap::Iterations::kColumn; ++column) {

            bool guard = row_guard && mask_.predicates[column];

            void* stptr = (row == 0) ? ptr1 : (row == 1?ptr2:NULL);
            cp_async_presum(
                stptr,
                (void *)&memory_pointer[column * ThreadMap::Delta::kColumn /
                                        kElementsPerAccess],
                guard);
          }

          if (row + 1 < ThreadMap::Iterations::kRow) {
            if (!ScatterD) {
              byte_pointer += params_.increment_row;
            }
          }
        }

        if (group + 1 < ThreadMap::Iterations::kGroup) {
          byte_pointer += params_.increment_group;
        }
      }

      if (cluster + 1 < ThreadMap::Iterations::kCluster) {
        byte_pointer += params_.increment_cluster;
      }
    }
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load(Fragment &frag, Fragment &frag2) const {

    load_with_byte_offset(frag, frag2, 0);
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load_with_byte_offset(Fragment &frag, uint64_t size) const {
    Fragment frag2;
    load_with_byte_offset(frag, frag2, size);
  }
  
    /// Loads a fragment from memory
  CUTLASS_DEVICE
  void load(Fragment &frag) const {
    Fragment frag2;
    load_with_byte_offset(frag, frag2, 0);
  }

  /// Stores a fragment to memory
  CUTLASS_DEVICE
  void store_with_byte_offset(Fragment const &frag, Fragment const &frag2, int64_t byte_offset) const {
    uint8_t *byte_pointer = store_byte_pointer_;
    AccessType const *frag_ptr = reinterpret_cast<AccessType const *>(&frag);
    AccessType const *frag2_ptr = reinterpret_cast<AccessType const *>(&frag2);

    // if (threadIdx.x == 0) {
    //   printf("ThreadMap %d %d %d %d\n", ThreadMap::Iterations::kCluster, 
    //   ThreadMap::Iterations::kGroup, ThreadMap::Iterations::kRow, ThreadMap::Iterations::kColumn);
    //   printf("increment_ %ld\n", params_.increment_group);
    //   printf("byte_pointer %p\n", byte_pointer);
    // }

    // if (thread_start_row_ == 32) {
    //   printf("395: tid %d %f\n", threadIdx.x, frag[0]);
    // }
    CUTLASS_PRAGMA_UNROLL
    for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster; ++cluster) {

      CUTLASS_PRAGMA_UNROLL
      for (int group = 0; group < ThreadMap::Iterations::kGroup; ++group) {

        CUTLASS_PRAGMA_UNROLL
        for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {

          int frag_row_idx =
            (row + ThreadMap::Iterations::kRow * (group + ThreadMap::Iterations::kGroup * cluster));

          int row_offset = row * ThreadMap::Delta::kRow
            + group * ThreadMap::Delta::kGroup
            + cluster * ThreadMap::Delta::kCluster;

          bool row_guard = ((row_offset + thread_start_row_) < extent_row_);

          AccessType *memory_pointer = reinterpret_cast<AccessType *>(byte_pointer + byte_offset);

          if (ScatterD && row_guard) {
            assert(indices_);

            memory_pointer = reinterpret_cast<AccessType *>(byte_pointer + byte_offset +
              LongIndex(indices_[row_offset + thread_start_row_]) * LongIndex(params_.stride));
          }
          // __syncthreads();
          // if (threadIdx.x%32 == 0 || threadIdx.x%32 == 31) {
          //     printf("428: %d: %p\n", threadIdx.x, memory_pointer);
          //   }
          // __syncthreads();
          CUTLASS_PRAGMA_UNROLL
          for (int column = 0; column < ThreadMap::Iterations::kColumn; ++column) {
            bool guard = row_guard && mask_.predicates[column];
            
            if (PermuteD) {

              int col_offset = column * ThreadMap::Delta::kColumn;

              int col = col_offset + thread_start_column_;
              int row = row_offset + thread_start_row_;

              // Locate memory_pointer
              memory_pointer = reinterpret_cast<AccessType *>(byte_pointer + byte_offset
                 + permute_layout_(PitchLinearCoord(col, row)) * sizeof(AccessType) / kElementsPerAccess);
            }

            if (UseCUDAStore) {
              if (guard) {
                memory_pointer[0] =
                    frag_ptr[frag_row_idx * ThreadMap::Iterations::kColumn + column];
                
                if (next_offset != 0) {
                  memory_pointer[next_offset] =
                    frag2_ptr[frag_row_idx * ThreadMap::Iterations::kColumn + column];
                }
              }
            } else {
              cutlass::arch::global_store<AccessType, sizeof(AccessType)>(
                  frag_ptr[frag_row_idx * ThreadMap::Iterations::kColumn + column],
                  (void *)&memory_pointer[0],
                  guard);
              
              if (next_offset != 0) {
                cutlass::arch::global_store<AccessType, sizeof(AccessType)>(
                    frag2_ptr[frag_row_idx * ThreadMap::Iterations::kColumn + column],
                    (void *)&memory_pointer[next_offset],
                    guard);
              }
            }

            if (!PermuteD) {
              memory_pointer += (ThreadMap::Delta::kColumn / kElementsPerAccess);
            }
          }

          if (row + 1 < ThreadMap::Iterations::kRow) {
            if (!ScatterD && !PermuteD) {
              byte_pointer += params_.increment_row;
            }
          }
        }

        if (group + 1 < ThreadMap::Iterations::kGroup) {
          if (!ScatterD && !PermuteD) {
            byte_pointer += params_.increment_group;
            // if (threadIdx.x == 0) {
            //   printf("476 %d: %p\n", threadIdx.x, byte_pointer);
            // }
          }
        }
      }

      if (cluster + 1 < ThreadMap::Iterations::kCluster) {
        if (!ScatterD && !PermuteD) {
          byte_pointer += params_.increment_cluster;
        }
      }
    }
  }

  /// Stores a fragment to memory
  CUTLASS_DEVICE
  void store(Fragment const &frag, Fragment const &frag2) const {

    store_with_byte_offset(frag, frag2, 0);
  }

  CUTLASS_DEVICE
  void store_with_byte_offset(Fragment const &frag, uint64_t byte_offset) const {
    Fragment frag2;
    store_with_byte_offset(frag, frag2, byte_offset);
  }

  CUTLASS_DEVICE
  void store(Fragment const &frag) const {
    Fragment frag2;
    store_with_byte_offset(frag, frag2, 0);
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void downsample_load_with_byte_offset(Fragment &frag, int64_t byte_offset, int convolution_P, int convolution_Q, int add_P, int add_Q, int problem_N) const {

    uint8_t *byte_pointer = byte_pointer_;
    AccessType *frag_ptr = reinterpret_cast<AccessType *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster; ++cluster) {

      CUTLASS_PRAGMA_UNROLL
      for (int group = 0; group < ThreadMap::Iterations::kGroup; ++group) {

        CUTLASS_PRAGMA_UNROLL
        for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {

          int frag_row_idx = 
            (row + ThreadMap::Iterations::kRow * (group + ThreadMap::Iterations::kGroup * cluster));

          int row_offset = row * ThreadMap::Delta::kRow 
            + group * ThreadMap::Delta::kGroup 
            + cluster * ThreadMap::Delta::kCluster;

          bool row_guard = ((row_offset + thread_start_row_) < extent_row_);

          int output_row = row_offset + thread_start_row_;
          int output_N = output_row / (convolution_P * convolution_Q);
          int output_PQ = output_row % (convolution_P * convolution_Q);
          int output_P = output_PQ / convolution_Q;
          int output_Q = output_PQ % convolution_Q;

          int input_row = output_N * 2 * convolution_P * 2 * convolution_Q +
            (2 * output_P + add_P) * 2 * convolution_Q + 2 * output_Q + add_Q;

          int64_t byte_offset = (input_row-output_row)*problem_N*sizeof(float);

          AccessType *memory_pointer = reinterpret_cast<AccessType *>(byte_pointer + byte_offset);

          CUTLASS_PRAGMA_UNROLL
          for (int column = 0; column < ThreadMap::Iterations::kColumn; ++column) {

            bool guard = row_guard && mask_.predicates[column];

            cutlass::arch::global_load<
              AccessType, 
              sizeof(AccessType)
            >(
                frag_ptr[frag_row_idx * ThreadMap::Iterations::kColumn +
                         column],
                (void *)&memory_pointer[column * ThreadMap::Delta::kColumn /
                                        kElementsPerAccess],
                guard);
          }


          if (row + 1 < ThreadMap::Iterations::kRow) {
            byte_pointer += params_.increment_row;
          }
        }

        if (group + 1 < ThreadMap::Iterations::kGroup) {
          byte_pointer += params_.increment_group;
        }
      }

      if (cluster + 1 < ThreadMap::Iterations::kCluster) {
        byte_pointer += params_.increment_cluster;
      }
    }
  }

  /// Loads a fragment from memory
  CUTLASS_DEVICE
  void upsample_load_with_byte_offset(Fragment &frag, int64_t byte_offset, int convolution_P, int convolution_Q, int add_P, int add_Q, int problem_N) const {

    uint8_t *byte_pointer = byte_pointer_;
    AccessType *frag_ptr = reinterpret_cast<AccessType *>(&frag);

    CUTLASS_PRAGMA_UNROLL
    for (int cluster = 0; cluster < ThreadMap::Iterations::kCluster; ++cluster) {

      CUTLASS_PRAGMA_UNROLL
      for (int group = 0; group < ThreadMap::Iterations::kGroup; ++group) {

        CUTLASS_PRAGMA_UNROLL
        for (int row = 0; row < ThreadMap::Iterations::kRow; ++row) {

          int frag_row_idx = 
            (row + ThreadMap::Iterations::kRow * (group + ThreadMap::Iterations::kGroup * cluster));

          int row_offset = row * ThreadMap::Delta::kRow 
            + group * ThreadMap::Delta::kGroup 
            + cluster * ThreadMap::Delta::kCluster;

          bool row_guard = ((row_offset + thread_start_row_) < extent_row_);

          int output_row = row_offset + thread_start_row_;
          int output_N = output_row / (convolution_P * convolution_Q);
          int output_PQ = output_row % (convolution_P * convolution_Q);
          int output_P = output_PQ / convolution_Q;
          int output_Q = output_PQ % convolution_Q;
          int row_add_P = add_P;
          int row_add_Q = add_Q;
	  if (output_P > convolution_P - 2) row_add_P = 0;
	  if (output_Q > convolution_Q - 2) row_add_Q = 0;

          int input_row = output_N * (convolution_P/2) * (convolution_Q/2) +
            ((output_P + row_add_P)/2) * (convolution_Q/2) + (output_Q + row_add_Q)/2;

          int64_t byte_offset = (input_row-output_row)*problem_N*sizeof(float);

          AccessType *memory_pointer = reinterpret_cast<AccessType *>(byte_pointer + byte_offset);

          CUTLASS_PRAGMA_UNROLL
          for (int column = 0; column < ThreadMap::Iterations::kColumn; ++column) {

            bool guard = row_guard && mask_.predicates[column];

            cutlass::arch::global_load<
              AccessType, 
              sizeof(AccessType)
            >(
                frag_ptr[frag_row_idx * ThreadMap::Iterations::kColumn +
                         column],
                (void *)&memory_pointer[column * ThreadMap::Delta::kColumn /
                                        kElementsPerAccess],
                guard);
          }

          if (row + 1 < ThreadMap::Iterations::kRow) {
            byte_pointer += params_.increment_row;
          }
        }

        if (group + 1 < ThreadMap::Iterations::kGroup) {
          byte_pointer += params_.increment_group;
        }
      }

      if (cluster + 1 < ThreadMap::Iterations::kCluster) {
        byte_pointer += params_.increment_cluster;
      }
    }
  }

  CUTLASS_DEVICE
  MatrixCoord thread_start() const {
    return MatrixCoord(thread_start_row_, thread_start_column_);
  }

  /// Need to get the thread start row from the tile iterator
  CUTLASS_DEVICE
  int32_t thread_start_row() const {
    return thread_start_row_;
  }

  /// Need to get the thread start row from the tile iterator
  CUTLASS_DEVICE
  int32_t thread_start_column() const {
    return thread_start_column_;
  }

  /// Extent of the matrix in rows
  CUTLASS_DEVICE
  Index extent_row() const {
    return extent_row_;
  }

  /// Extent of the matrix in columns
  CUTLASS_DEVICE
  Index extent_column() const {
    return extent_column_;
  }

  /// Advances to the next position to load or store
  CUTLASS_HOST_DEVICE
  StrassenPredicatedTileIterator &operator++() {
    ++state_[0];

    if (!ScatterD) {
      byte_pointer_ += params_.advance_row;
    }

    if (!ScatterD && !PermuteD) {
      store_byte_pointer_ += params_.advance_row;
    }

    if (state_[0] == ThreadMap::Count::kRow) {

      state_[0] = 0;
      ++state_[1];

      if (!ScatterD) {
        byte_pointer_ += params_.advance_group;
      }

      if (!ScatterD && !PermuteD) {
        store_byte_pointer_ += params_.advance_group;
      }

      thread_start_row_ += (ThreadMap::Shape::kGroup - 1) *
        ThreadMap::Shape::kRow * ThreadMap::Count::kRow;

      if (state_[1] == ThreadMap::Count::kGroup) {

        state_[1] = 0;
        ++state_[2];

        if (!ScatterD) {
          byte_pointer_ += params_.advance_cluster;
        }

        if (!ScatterD && !PermuteD) {
          store_byte_pointer_ += params_.advance_cluster;
        }

        thread_start_row_ += ThreadMap::Count::kGroup *
          ThreadMap::Shape::kGroup * ThreadMap::Count::kRow * ThreadMap::Shape::kRow;

        if (state_[2] == ThreadMap::Count::kCluster) {
          state_[2] = 0;

          if (!ScatterD) {
            byte_pointer_ += params_.advance_tile;
          }

          if (!ScatterD && !PermuteD) {
            store_byte_pointer_ += params_.advance_tile;
          }

          thread_start_row_ += ThreadMap::Shape::kGroup * ThreadMap::Shape::kRow
            * ThreadMap::Shape::kCluster * ThreadMap::Shape::kTile;
        }
      }
    }
    return *this;
  }

  /// Advances a number of positions to load or store
  CUTLASS_HOST_DEVICE
  StrassenPredicatedTileIterator &operator+=(int increment)
  {
    if (increment == 0) return *this;
    // Row
    state_[0] += increment;
    int increment_row = state_[0] / ThreadMap::Count::kRow;
    state_[0] = state_[0] % ThreadMap::Count::kRow;

    byte_pointer_ += (params_.advance_row * increment);
    store_byte_pointer_ += (params_.advance_row * increment);
    thread_start_row_ += (ThreadMap::Shape::kRow * increment);

    // Group
    state_[1] += increment_row;
    int increment_group = state_[1] / ThreadMap::Count::kGroup;
    state_[1] = state_[1] % ThreadMap::Count::kGroup;

    byte_pointer_ += (params_.advance_group * increment_row);
    store_byte_pointer_ += (params_.advance_group * increment_row);
    thread_start_row_ +=
        (ThreadMap::Shape::kGroup - 1) *
        ThreadMap::Shape::kRow *
        ThreadMap::Count::kRow *
        increment_row;


    // Cluster
    state_[2] += increment_group;
    int increment_cluster = state_[2] / ThreadMap::Count::kCluster;
    state_[2] = state_[2] % ThreadMap::Count::kCluster;

    byte_pointer_ += (params_.advance_cluster * increment_group);
    store_byte_pointer_ += (params_.advance_cluster * increment_group);
    thread_start_row_ +=
        ThreadMap::Count::kGroup *
        ThreadMap::Shape::kGroup *
        ThreadMap::Count::kRow *
        ThreadMap::Shape::kRow *
        increment_group;

    // Tile
    byte_pointer_ += (params_.advance_tile * increment_cluster);
    store_byte_pointer_ += (params_.advance_tile * increment_cluster);
    thread_start_row_ +=
        ThreadMap::Shape::kGroup *
        ThreadMap::Shape::kRow *
        ThreadMap::Shape::kCluster *
        ThreadMap::Shape::kTile *
        increment_cluster;

    return *this;
  }

  ///< Efficiently disables all accesses guarded by mask
  CUTLASS_DEVICE void clear_mask() {
    mask_.clear();
  }

  ///< Efficiently enables all accesses guarded by mask
  CUTLASS_DEVICE void enable_mask() {
    mask_.enable();
  }

  ///< Sets the mask
  CUTLASS_DEVICE void get_mask(Mask &mask) const {
    mask = mask_;
  }

  ///< Sets the mask
  CUTLASS_DEVICE void set_mask(Mask const &mask) {
    mask_ = mask;
  }
};

}
}
}