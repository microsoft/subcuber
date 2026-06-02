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


#include "cutlass/aligned_buffer.h"
#include "cutlass/arch/memory.h"
#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_types.h"
#include "cutlass/postsum_semaphore.h"

#include "cutlass/gemm/threadblock/global_strassen_matrix_mma_base.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace threadblock {

/// Copy

/// ld.shared - 128b
CUTLASS_DEVICE
void shared_load_128b(void *dst, void* ptr) {
  float4 *dst_u128 = reinterpret_cast<float4 *>(dst);
  // *dst_u128 = *(float4*)ptr;
  uint64_t addr64 = 0;
  // uint addr32;
  asm volatile("cvta.to.shared.u64 %0, %1;": "=l"(addr64) : "l"(ptr));
  // addr32 = (uint)addr64;
  asm volatile("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];\n"
    :
      "=f"(dst_u128->x),
      "=f"(dst_u128->y),
      "=f"(dst_u128->z),
      "=f"(dst_u128->w)
    : "l"(addr64));
}

CUTLASS_DEVICE
void shared_load_32b(void *dst, void* ptr) {
  float& dst_32b = *reinterpret_cast<float *>(dst);
  // *dst_u128 = *(float4*)ptr;
  uint64_t addr64 = 0;
  // uint addr32;
  asm volatile("cvta.to.shared.u64 %0, %1;": "=l"(addr64) : "l"(ptr));
  // addr32 = (uint)addr64;
  asm volatile("ld.shared.f32 %0, [%1];\n"
    :
      "=f"(dst_32b)
    : "l"(addr64));
}

CUTLASS_DEVICE
static Array<float, 8> frag_converter(Array<half2, 4>& src) {
  Array<float, 8> trg;
  for (int i = 0; i < 8; i+=2) {
    auto f2 = __half22float2(src[i/2]);
    trg[i+0] =  f2.x; trg[i+1] = f2.y;
  }

  return trg;
}

CUTLASS_DEVICE
static Array<half2, 4> frag_converter(Array<float, 8>& src) {
  Array<half2, 4> trg;
  for (int i = 0; i < 8; i+=2) {
    auto h2 = __float22half2_rn(float2{src[i+0], src[i+1]});
    trg[i/2] = h2;
  }
  return trg;
}

template<typename Element, uint BLOCK_DIM, typename MmaTileShape, typename VectorLoadType, bool LinearIndexing>
struct PresumOverlapIterator {
  int row;
  int col;
  Element* pointer;
  MatrixCoord extent;
  long nextOffset0, nextOffset1, nextOffset2, nextOffset3, nextOffset4;
  static const uint VectorLoadElems = sizeof(VectorLoadType)/sizeof(half);
  static const uint LdBytes = sizeof(VectorLoadType);
  static const uint kIterations = (MmaTileShape::kM * MmaTileShape::kN) / (BLOCK_DIM * VectorLoadElems);
  MatrixCoord tb_offset;
  bool isValidTB;
  bool isDest;
  uint linear_block_idx;

  CUTLASS_DEVICE
  PresumOverlapIterator(Element* pointer_, MatrixCoord extent_, MatrixCoord tb_offset_, uint linear_block_idx,
                        MatrixCoord nextOffset0_, uint threadId,
                        MatrixCoord nextOffset1_=MatrixCoord(0,0),
                        MatrixCoord nextOffset2_=MatrixCoord(0,0),
                        MatrixCoord nextOffset3_=MatrixCoord(0,0),
                        MatrixCoord nextOffset4_=MatrixCoord(0,0),
                        bool isDest = false) : 
    extent(extent_), tb_offset(tb_offset_), isDest(isDest), linear_block_idx(linear_block_idx)
  {
    nextOffset0 = nextOffset0_.row() * extent.column() + nextOffset0_.column();
    nextOffset1 = nextOffset1_.row() * extent.column() + nextOffset1_.column();
    nextOffset2 = nextOffset2_.row() * extent.column() + nextOffset2_.column();
    nextOffset3 = nextOffset3_.row() * extent.column() + nextOffset3_.column();
    nextOffset4 = nextOffset4_.row() * extent.column() + nextOffset4_.column();
    reset();
    pointer = pointer_;
    isValidTB = tb_offset.column() < extent.column();
  }

  CUTLASS_DEVICE
  void inc() {
    if (LinearIndexing) {
      col += VectorLoadElems * BLOCK_DIM;
    } else {
      if (BLOCK_DIM*VectorLoadElems >= MmaTileShape::kN) {
        row += (BLOCK_DIM*VectorLoadElems)/MmaTileShape::kN;
      } else {
        col += BLOCK_DIM*VectorLoadElems;
      }
    }
  }

  CUTLASS_DEVICE
  int increment() {
    if (LinearIndexing) {
      return VectorLoadElems * BLOCK_DIM;
    } else if (BLOCK_DIM*VectorLoadElems >= MmaTileShape::kN) {
      return (BLOCK_DIM*VectorLoadElems)/MmaTileShape::kN;
    } else {
      return BLOCK_DIM*VectorLoadElems;
    }
  }

  CUTLASS_DEVICE
  void reset() {
    if (LinearIndexing) {
      col = linear_block_idx;
      col = col * MmaTileShape::kM * MmaTileShape::kN + threadIdx.x * 8;
      row = 0;
    } else {
      // if (threadIdx.x == 0) printf("121 %d %d -> %d %d\n",
      //   blockIdx.x, blockIdx.y, tb_offset.row(), tb_offset.column());
      col = (threadIdx.x*VectorLoadElems) %  MmaTileShape::kN;
      row = (threadIdx.x*VectorLoadElems) / MmaTileShape::kN;
    }
  }

  CUTLASS_DEVICE
  bool valid() {
    if (LinearIndexing) return col < MmaTileShape::kM * MmaTileShape::kN;
    return row < MmaTileShape::kM;
  }
  
  CUTLASS_DEVICE
  bool validTB() {
    return isValidTB;
  }

  CUTLASS_DEVICE
  static void sumHalf8(half2* frag_op1, half2* frag_op2, half2* frag_out, uint N) {
    for (int i = 0; i < N; i++) {
      frag_out[i] = frag_op1[i] + frag_op2[i];
    }
  }

  
  CUTLASS_DEVICE
  static void diffHalf8(half2* frag_op1, half2* frag_op2, half2* frag_out, uint N) {
    for (int i = 0; i < N; i++) {
      frag_out[i] = frag_op1[i] - frag_op2[i];
    }
  }

  CUTLASS_DEVICE
  static VectorLoadType presum(VectorLoadType in1, VectorLoadType in2, bool add, bool sub) {
    VectorLoadType out;
    if (add)
      sumHalf8((half2*)&in1, (half2*)&in2, (half2*)&out, VectorLoadElems/2);
    if (sub)
      diffHalf8((half2*)&in1, (half2*)&in2, (half2*)&out, VectorLoadElems/2);
    return out;
  }

  template<typename Fragment>
  CUTLASS_DEVICE
  static void presum(Fragment& in1, Fragment& in2, bool add, bool sub) {
    if (add) {
      in1 = in1 + in2;
    }
    if (sub) {
      in1 = in1 - in2;
    }
    // for (int i = 0; i < in1.size(); i++) {
    //   if (add) {
    //     in1[i] = in1[i] + in2[i];
    //   }
    //   if (sub) {
    //     in1[i] = in1[i] - in2[i];
    //   }
    // }
  }

  template<typename Fragment1, typename Fragment2>
  CUTLASS_DEVICE
  static void presumWithConvert(Fragment1& acc, Fragment2& in, bool add, bool sub) {
    uint N = acc.size();
    // half* inptr = (half*)&in;
    for (int i = 0; i < N; i++) {
      if (add)
        acc[i] = acc[i] + (float)(in[i]);
      else if (sub)
        acc[i] = acc[i] - (float)(in[i]);
    }
  }

  template<typename Fragment1, typename Fragment2>
  CUTLASS_DEVICE
  static void presumWithConvert(Fragment1& acc, Fragment2& in1, Fragment2& in2, bool add, bool sub) {
    uint N = acc.size();
    // half* inptr = (half*)&in;
    for (int i = 0; i < N; i++) {
      float x = 0;
      if (i < in1.size()) x = (float)(in1[i]);
      else x = (float)(in2[i-in1.size()]);

      x     *= (sub) ? -1 : 1;
      acc[i] = acc[i] + x;
    }
  }

  CUTLASS_DEVICE
  VectorLoadType* get(int part, bool b = false) {
    if (LinearIndexing or not isDest) {
      auto pointer0 = pointer + ((LinearIndexing) ? col : ((tb_offset.row() + row) * extent.column() + (tb_offset.column() + col)));     
       if (LinearIndexing) {
        return (VectorLoadType*)(pointer0 + part * extent.row() * extent.column());
       } else {
        if (part == 0)      return (VectorLoadType*)(pointer0 + nextOffset0);
        else if (part == 1) return (VectorLoadType*)(pointer0 + nextOffset1);
        else if (part == 2) return (VectorLoadType*)(pointer0 + nextOffset2);
        else if (part == 3) return (VectorLoadType*)(pointer0 + nextOffset3);
        else if (part == 4) return (VectorLoadType*)(pointer0 + nextOffset4);

        return nullptr;
      }

      return nullptr;
    }

    auto pointer0 = pointer + (tb_offset.row() + row) * extent.column()*2 + (tb_offset.column() + col);
         if (part == 0) return (VectorLoadType*)(pointer0 + nextOffset0);
    else if (part == 1) return (VectorLoadType*)(pointer0 + nextOffset1);

    auto pointer1 = pointer + ((LinearIndexing) ? col : ((tb_offset.row() + row) * (extent.column()) + (tb_offset.column() + col)));
        //  if (part == 1) return (VectorLoadType*)(pointer1 + nextOffset1);
    if (part == 2) return (VectorLoadType*)(pointer1 + nextOffset2);
    else if (part == 3) return (VectorLoadType*)(pointer1 + nextOffset3);
    else if (part == 4) return (VectorLoadType*)(pointer1 + nextOffset4);
    return nullptr; //(VectorLoadType*)pointer0;
  }

  CUTLASS_DEVICE
  half2* get2(int part, bool b = false) {
    auto pointer0 = pointer + ((LinearIndexing) ? col : ((tb_offset.row() + row) * extent.column() + (tb_offset.column() + col)));
    if (part == 0) return (half2*)(pointer0 + nextOffset0);
    else if (part == 1) return (half2*)(pointer0 + nextOffset1);
    else if (part == 2) return (half2*)(pointer0 + nextOffset2);
    else if (part == 3) return (half2*)(pointer0 + nextOffset3);
    else if (part == 4) return (half2*)(pointer0 + nextOffset4);
    return nullptr;
  }

  CUTLASS_DEVICE
  void cp_async_presum(void *smem_ptr, void const *global_ptr, bool pred_guard = true) {
    #if CUDA_CP_ASYNC_ACTIVATED
      unsigned smem_int_ptr = cutlass::arch::cutlass_get_smem_pointer(smem_ptr);
      asm volatile(
          "{\n"
          "  .reg .pred p;\n"
          "  setp.ne.b32 p, %0, 0;\n"
          "  @p cp.async.cg.shared.global [%1], [%2], %3;\n"
          "}\n" ::"r"((int)pred_guard),
          "r"(smem_int_ptr), "l"(global_ptr), "n"(LdBytes));
    #endif
  }
};

template<typename Element, uint BLOCK_DIM, uint Size, uint Stages, uint PresumAccessPerIter, uint LdSz>
struct PresumSharedIterator {
  Element* pointer;
  uint threadId;

CUTLASS_DEVICE
  PresumSharedIterator(Element* pointer_, uint threadId_) {
    pointer = pointer_;
    threadId = threadId_;
  }
CUTLASS_DEVICE
  Element* get(int stage, int access = 0, int ldsz = LdSz) {
    auto p = pointer  + stage * Size +
             access   * BLOCK_DIM * (ldsz/sizeof(Element)) +
             threadId * (ldsz/sizeof(Element));
    
    return p;
  }

  CUTLASS_DEVICE
  Element* getLinear(int idx) {
    auto p = pointer  + idx +
             threadId * (LdSz/sizeof(Element));
    return p;
  }

  CUTLASS_DEVICE
  Element* get2(int stage, int mi, int part, int ldsz = LdSz) {
    auto p = pointer  + mi * Stages * Size + stage * Size + 
             part     * (ldsz/sizeof(Element)) * BLOCK_DIM +
             threadId * (ldsz/sizeof(Element));
    
    return p;
  }

CUTLASS_DEVICE
  void inc() {}
};

template<typename Element, uint BLOCK_DIM, typename Shape, uint LdSz>
struct PresumOutSharedIterator {
  Element* pointer;
  uint col, row;
  static const uint VectorLoadElems = sizeof(float4)/sizeof(half);

  CUTLASS_DEVICE
  PresumOutSharedIterator(Element* pointer_, uint threadId) {
    pointer = pointer_;
    col = (threadId*VectorLoadElems) %  Shape::kN;
    row = (threadId*VectorLoadElems) / Shape::kN;
  }

  CUTLASS_DEVICE
  Element* get(int row, int col) {
    auto p = pointer + row * Shape::kN + col;
    return p;
  }

  CUTLASS_DEVICE
  Element* rget(int row, int col) {
    auto p = pointer + Shape::kM * Shape::kN - (row * Shape::kN + col);
    return p;
  }

  template<typename AccessType, typename Fragment>
  CUTLASS_DEVICE
  void load(Fragment& frag) {
    AccessType *frag_ptr = reinterpret_cast<AccessType *>(&frag);

    frag_ptr[0] = *(AccessType*)rget(row, col);
    this->operator++();
    frag_ptr[1] = *(AccessType*)rget(row, col);
    this->operator++();
    
  }

  CUTLASS_DEVICE
  void operator++() {
    row += (BLOCK_DIM*VectorLoadElems)/Shape::kN;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

/// Structure to compute the matrix product targeting CUDA cores and SIMT math
/// instructions.
template <
    /// Size of the Gemm problem - concept: gemm::GemmShape<>
    typename Shape_,
    typename StrassenShape_,
    /// Iterates over tiles of A operand in global memory
    //  (concept: ReadableTileIterator | ForwardTileIterator |
    //  MaskedTileIterator)
    typename IteratorA_,
    /// Iterates over tiles of A operand in shared memory
    /// (concept: WriteableTileIterator | RandomAccessTileIterator)
    typename SmemIteratorA_,
    /// Cache operation for operand A
    cutlass::arch::CacheOperation::Kind CacheOpA,
    /// Iterates over tiles of B operand in global memory
    //  (concept: ReadableTileIterator | ForwardTileIterator |
    //  MaskedTileIterator)
    typename IteratorB_,
    /// Iterates over tiles of B operand in shared memory
    /// (concept: WriteableTileIterator | RandomAccessTileIterator)
    typename SmemIteratorB_,
    /// Cache operation for operand B
    cutlass::arch::CacheOperation::Kind CacheOpB,
    /// Data type of accumulator matrix
    typename ElementC_,
    /// Data type of accumulator matrix
    typename LayoutC_,
    /// Policy describing tuning details (concept: MmaPolicy)
    typename Policy_,
    /// Number of stages,
    MmaStrassen::Type MmaStrassenKind_,
    typename MmaStrassenConsts,
    int Stages,
    /// Use zfill or predicate for out-of-bound cp.async
    SharedMemoryClearOption SharedMemoryClear = SharedMemoryClearOption::kNone,
    /// Used for partial specialization
    typename Enable = bool>
class GlobalStrassenPresumMmaMultistage : 
  public GlobalStrassenMatrixMmaBase<StrassenShape_, Policy_, Stages> {
public:
  ///< Base class
  using Base = GlobalStrassenMatrixMmaBase<Shape_, Policy_, Stages>;
  static const auto MmaStrassenKind = MmaStrassenKind_;

  ///< Size of the Gemm problem - concept: gemm::GemmShape<>
  using Shape = Shape_;
  using StrassenShape = StrassenShape_;
  using PresumShape = GemmShape<Shape::kM, Shape::kN, 1>;

  ///< Iterates over tiles of A operand in global memory
  using IteratorA = IteratorA_;
  ///< Iterates over tiles of B operand in global memory
  using IteratorB = IteratorB_;
  ///< Data type of accumulator matrix
  using ElementC = ElementC_;
  ///< Layout of accumulator matrix
  using LayoutC = LayoutC_;
  ///< Policy describing tuning details
  using Policy = Policy_;

  using SmemIteratorA = SmemIteratorA_;
  using SmemIteratorB = SmemIteratorB_;
  static const int BLOCK_DIM = Base::WarpCount::kCount * 32;
  int SplitKLastTB = true;

  static cutlass::arch::CacheOperation::Kind const kCacheOpA = CacheOpA;
  static cutlass::arch::CacheOperation::Kind const kCacheOpB = CacheOpB;

  using PresumIteratorA    = PresumOverlapIterator<typename IteratorA::Element, BLOCK_DIM, PresumShape, float4, false>;
  using PresumIteratorSrcA = PresumOverlapIterator<typename IteratorA::Element, BLOCK_DIM, PresumShape, float4, false>;
  using PresumIteratorDstA = PresumOverlapIterator<typename IteratorA::Element, BLOCK_DIM, PresumShape, float4, false>;

  using PresumIteratorSrcB = PresumOverlapIterator<typename IteratorB::Element, BLOCK_DIM, PresumShape, float4, false>;
  using PresumIteratorDstB = PresumOverlapIterator<typename IteratorB::Element, BLOCK_DIM, PresumShape, float4, false>;

  using PresumIteratorSrcM = PresumOverlapIterator<typename IteratorA::Element, BLOCK_DIM, Shape, float4, true>;
  //
  // Dependent types
  //

  /// Fragment of accumulator tile
  using FragmentC = typename Policy::Operator::FragmentC;

  /// Warp-level Mma
  using Operator = typename Policy::Operator;

  /// Minimum architecture is Sm80 to support cp.async
  using ArchTag = arch::Sm80;

  /// Complex transform on A operand
  static ComplexTransform const kTransformA = Operator::kTransformA;

  /// Complex transform on B operand
  static ComplexTransform const kTransformB = Operator::kTransformB;

  /// Internal structure exposed for introspection.
  struct Detail {

    /// Number of cp.async instructions to load one stage of operand A
    static int const AsyncCopyIterationsPerStageA =
        IteratorA::ThreadMap::Iterations::kCount;

    /// Number of cp.async instructions to load one stage of operand B
    static int const AsyncCopyIterationsPerStageB =
        IteratorB::ThreadMap::Iterations::kCount;

    /// Number of stages
    static int const kStages = Stages;

    /// Number of cp.async instructions to load on group of operand A
    static int const kAccessesPerGroupA =
        (AsyncCopyIterationsPerStageA + Base::kWarpGemmIterations - 1) / Base::kWarpGemmIterations;

    /// Number of cp.async instructions to load on group of operand B
    static int const kAccessesPerGroupB =
        (AsyncCopyIterationsPerStageB + Base::kWarpGemmIterations - 1) / Base::kWarpGemmIterations;

    // Optional staged-accumulation (e.g., tf32x3 kernels) for improved numerical
    // accuracy, where each mainloop iteration first accumulates into a temporary
    // set of freshly-cleared accumulators, which are subsequently added to the
    // final accumulator set.
    static bool const kStagedAccumulation = arch::detail::UseStagedAccumulation<Operator>::value;
  };

 private:


  // Structure encapsulating pipeline state live from one iteration to the next
  struct PipeState {

    using WarpLoadedFragmentA = typename Operator::FragmentA;
    using WarpLoadedFragmentB = typename Operator::FragmentB;
    using WarpTransformedFragmentA = typename Operator::TransformedFragmentA;
    using WarpTransformedFragmentB = typename Operator::TransformedFragmentB;

    /// Temporary accumulator to facilitate staged-accumulation
    FragmentC tmp_accum_;

    /// Pair of A fragments used to overlap shared memory loads and math instructions
    WarpLoadedFragmentA warp_loaded_frag_A_[2];
    WarpTransformedFragmentA warp_transformed_frag_A_[2];

    /// Pair of B fragments used to overlap shared memory loads and math instructions
    WarpLoadedFragmentB warp_loaded_frag_B_[2];
    WarpTransformedFragmentB warp_transformed_frag_B_[2];
  };

  CUTLASS_DEVICE
  static constexpr bool addForNextM() {
    return MmaStrassenKind == MmaStrassen::Type::GlobalPreSumLevel1_M0;// || //A0+A3 For M0
          //  MmaStrassenKind == MmaStrassen::Type::GlobalPreSumLevel1_M1; // A2+A3 For M1
          //  MmaStrassenKind == MmaStrassen::Type::GlobalPreSumLevel1_M0;  //A0+A1 For M4
  }

  CUTLASS_DEVICE
  static constexpr bool subForNextM() {
    return MmaStrassenKind == MmaStrassen::Type::GlobalPreSumLevel1_M3 || //A2-A0 For M5
           MmaStrassenKind == MmaStrassen::Type::GlobalPreSumLevel1_M2; //A1-A3 For M6
  }

  CUTLASS_DEVICE
  static constexpr bool subForNextMOrder() {
    return MmaStrassenKind == MmaStrassen::Type::GlobalPreSumLevel1_M2; //A1-A3 For M6 (sharedA2 - sharedA1)
          //  MmaStrassenKind == MmaStrassen::Type::GlobalPreSumLevel1_M2; //A2-A0 for M5 (sharedA1 - sharedA2)
  }

  static constexpr uint PresumAccessPerIter = addForNextM() ? 1 : 4;
          // (MmaStrassenKind == MmaStrassen::Type::GlobalPreSumLevel1_M2 ||
          //  MmaStrassenKind == MmaStrassen::Type::GlobalPreSumLevel1_M3 ||
          //  MmaStrassenKind == MmaStrassen::Type::GlobalPreSumLevel1_M4 ||
          //  MmaStrassenKind == MmaStrassen::Type::GlobalPreSumLevel1_M1) ? 2 : 1;
 private:

  //
  // Data members
  //

  /// Warp-level MMA operator
  Operator warp_mma_;

  /// Iterator to write threadblock-scoped tile of A operand to shared memory
  SmemIteratorA smem_iterator_A_;

  /// Iterator to write threadblock-scoped tile of B operand to shared memory
  SmemIteratorB smem_iterator_B_;

  /// Shared memory write stage index
  int smem_write_stage_idx_;

  /// Shared memory read stage index
  int smem_read_stage_idx_;
public:
  PresumSharedIterator<typename IteratorA::Element, BLOCK_DIM,
                       BLOCK_DIM * 8 * PresumAccessPerIter, Stages, PresumAccessPerIter,
                       PresumIteratorSrcA::LdBytes> sharedPreSumA0;
  PresumSharedIterator<typename IteratorA::Element, BLOCK_DIM,
                       BLOCK_DIM * 8 * PresumAccessPerIter, Stages, PresumAccessPerIter,
                       PresumIteratorSrcA::LdBytes> sharedPreSumA1;
  PresumSharedIterator<typename IteratorA::Element, BLOCK_DIM,
                       BLOCK_DIM * 8 * PresumAccessPerIter, Stages, PresumAccessPerIter,
                       PresumIteratorSrcA::LdBytes> sharedPreSumA2;
  PresumSharedIterator<typename IteratorA::Element, BLOCK_DIM,
                       BLOCK_DIM * 8 * PresumAccessPerIter, Stages, PresumAccessPerIter,
                       PresumIteratorSrcA::LdBytes> sharedPreSumA3;
  PresumSharedIterator<typename IteratorA::Element, BLOCK_DIM,
                       BLOCK_DIM * 8 * PresumAccessPerIter, Stages, PresumAccessPerIter,
                       PresumIteratorSrcA::LdBytes> sharedPostSumM;

public:
  
  PresumOutSharedIterator<typename IteratorA::Element, BLOCK_DIM,
                          Shape,
                          PresumIteratorDstA::LdBytes> sharedPostSumC;

public:
  bool isStoreMInSh;
  uint64_t PresumM0OrM1;
  uint64_t PresumLoadAs;
  uint64_t ReadWriteMi;

  CUTLASS_DEVICE
  bool bitcheck(uint x, uint bit) {
    return (x & (1<<bit)) == (1<<bit);
  }
  /// Construct from tensor references
  CUTLASS_DEVICE
  GlobalStrassenPresumMmaMultistage(
      ///< Shared storage needed for internal use by threadblock-scoped GEMM
      typename Base::SharedStorage &shared_storage,
      ///< ID within the threadblock
      int thread_idx,
      ///< ID of warp
      int warp_idx,
      ///< ID of each thread within a warp
      int lane_idx,
      uint64_t PresumM0OrM1,
      uint64_t ReadWriteMi
    ):
      Base(shared_storage, thread_idx, warp_idx, lane_idx),
      smem_iterator_A_(shared_storage.operand_A_ref(), thread_idx),
      smem_iterator_B_(shared_storage.operand_B_ref(), thread_idx),
      sharedPreSumA0(shared_storage.operand_presumA1_ref(PresumAccessPerIter), thread_idx),
      sharedPreSumA1(shared_storage.operand_presumA2_ref(PresumAccessPerIter), thread_idx),
      sharedPreSumA2(shared_storage.operand_presumA3_ref(PresumAccessPerIter), thread_idx),
      sharedPreSumA3(shared_storage.operand_presumA4_ref(PresumAccessPerIter), thread_idx),
      sharedPostSumC(shared_storage.operand_presumA1_ref(PresumAccessPerIter), thread_idx),
      sharedPostSumM(shared_storage.operand_presumA1_ref(PresumAccessPerIter), thread_idx),
      // sharedPreSumOutA(shared_storage.operand_presumOutA_ref()),
      smem_write_stage_idx_(0),
      smem_read_stage_idx_(0), isStoreMInSh(true), PresumM0OrM1(PresumM0OrM1), ReadWriteMi(ReadWriteMi)
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
    this->warp_tile_iterator_A_.add_tile_offset(
        {warp_idx_m, Base::kWarpGemmIterations * warp_idx_k});
    this->warp_tile_iterator_B_.add_tile_offset(
        {Base::kWarpGemmIterations * warp_idx_k, warp_idx_n});;
    
    uint A0 = 1 << 0, A1 = 1 << 1, A2 = 1 << 2, A3 = 1 << 3;
    PresumLoadAs = 0;
    if (bitcheck(PresumM0OrM1, MmaStrassen::Type::GlobalPreSumLevel1_M2)) {
      PresumLoadAs = PresumLoadAs | A0 | A2 | A3;
    }
    if (bitcheck(PresumM0OrM1, MmaStrassen::Type::GlobalPreSumLevel1_M3)) {
      PresumLoadAs = PresumLoadAs | A0 | A2;
    }
    if (bitcheck(PresumM0OrM1, MmaStrassen::Type::GlobalPreSumLevel1_M4)) {
      PresumLoadAs = PresumLoadAs | A2 | A3;
    }
    if (bitcheck(PresumM0OrM1, MmaStrassen::Type::GlobalPreSumLevel1_M5)) {
      PresumLoadAs = PresumLoadAs | A0 | A1 | A2 | A3;
    }
  }

  /// Advance shared memory read-iterators to the next stage
  CUTLASS_DEVICE
  void advance_smem_read_stage()
  {
    ++smem_read_stage_idx_;

    if (smem_read_stage_idx_ == Base::kStages) {
      // Wrap back around to the 'start' of the circular buffer in shared memory
      this->warp_tile_iterator_A_.add_tile_offset({0, -Base::kStages * Policy::kPartitionsK * Base::kWarpGemmIterations});
      this->warp_tile_iterator_B_.add_tile_offset({-Base::kStages * Policy::kPartitionsK * Base::kWarpGemmIterations, 0});
      smem_read_stage_idx_ = 0;
    }
  }

  /// Advance global memory read-iterators and shared memory write-iterators to the stage
  CUTLASS_DEVICE
  void advance_smem_write_stage(
    IteratorA &iterator_A,
    IteratorB &iterator_B)
  {
    // Advance global iterators
    iterator_A.add_tile_offset({0, 1});
    iterator_B.add_tile_offset({1, 0});

    // Advance shared iterators
    smem_iterator_A_.add_tile_offset({0, 1});
    smem_iterator_B_.add_tile_offset({1, 0});

    // Increment shared memory write stage index
    ++smem_write_stage_idx_;

    if (smem_write_stage_idx_ == Base::kStages) {
      // Wrap back around to the 'start' of the circular buffer in shared memory
      smem_iterator_A_.add_tile_offset({0, -Base::kStages});
      smem_iterator_B_.add_tile_offset({-Base::kStages, 0});
      smem_write_stage_idx_ = 0;
    }
  }

  CUTLASS_DEVICE
  void copy_tiles_and_advance(IteratorA &iterator_A, IteratorB &iterator_B,
                              int group_start_A = 0, int group_start_B = 0) {
    iterator_A.set_iteration_index(group_start_A *
                                   IteratorA::kAccessesPerVector);
    this->smem_iterator_A_.set_iteration_index(group_start_A);

    // Async Copy for operand A
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Detail::kAccessesPerGroupA; ++j) {
      if (group_start_A + j < Detail::AsyncCopyIterationsPerStageA) {
        typename IteratorA::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorA::AccessType *>(
                this->smem_iterator_A_.get());

        int const kSrcBytes = sizeof_bits<typename IteratorA::Element>::value *
                              IteratorA::ThreadMap::kElementsPerAccess /
                              IteratorA::kAccessesPerVector / 8;

        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorA::kAccessesPerVector; ++v) {
          auto gmem_ptr = iterator_A.get();

          if (SharedMemoryClear == SharedMemoryClearOption::kZfill) {
            cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpA>(
                dst_ptr + v, gmem_ptr, iterator_A.valid());
          } else {
            cutlass::arch::cp_async<kSrcBytes, kCacheOpA>(
                dst_ptr + v, gmem_ptr, iterator_A.valid());
          }

          ++iterator_A;
        }

        ++this->smem_iterator_A_;
      }
    }

    iterator_B.set_iteration_index(group_start_B *
                                   IteratorB::kAccessesPerVector);
    this->smem_iterator_B_.set_iteration_index(group_start_B);

    // Async Copy for operand B
    CUTLASS_PRAGMA_UNROLL
    for (int j = 0; j < Detail::kAccessesPerGroupB; ++j) {
      if (group_start_B + j < Detail::AsyncCopyIterationsPerStageB) {
        typename IteratorB::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorB::AccessType *>(
                this->smem_iterator_B_.get());

        int const kSrcBytes = sizeof_bits<typename IteratorB::Element>::value *
                              IteratorB::ThreadMap::kElementsPerAccess /
                              IteratorB::kAccessesPerVector / 8;

        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorB::kAccessesPerVector; ++v) {
          auto gmem_ptr = iterator_B.get();

          if (SharedMemoryClear == SharedMemoryClearOption::kZfill) {
            cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpB>(
                dst_ptr + v, gmem_ptr, iterator_B.valid());
          } else {
            cutlass::arch::cp_async<kSrcBytes, kCacheOpB>(
                dst_ptr + v, gmem_ptr, iterator_B.valid());
          }

          ++iterator_B;
        }
        ++this->smem_iterator_B_;
      }
    }
  }


  CUTLASS_DEVICE
  int storeMInSh(bool* neg = nullptr, bool* global_sync = nullptr) {
    for (uint i = MmaStrassen::GlobalPreSumLevel1_M0; i <= MmaStrassen::GlobalPreSumLevel1_M6; i++) {
      if (i != MmaStrassenKind &&
          MmaStrassen::GetRWBit(ReadWriteMi, (MmaStrassen::Type)i) == MmaStrassen::RWSharedAddEnd) {
        if (neg)         *neg         = MmaStrassen::GetRWMulSign(ReadWriteMi, (MmaStrassen::Type)i);
        if (global_sync) *global_sync = MmaStrassen::GetRWGlobalSync(ReadWriteMi, (MmaStrassen::Type)i);
        return (i - MmaStrassen::GlobalPreSumLevel1_M0);
      }
    }
    return -1;
  }

  CUTLASS_DEVICE
  int computeForM(uint mi) {
    int found = 0;
    for (uint i = MmaStrassen::GlobalPreSumLevel1_M0; i <= MmaStrassen::GlobalPreSumLevel1_M6; i++) {
      if (i != MmaStrassenKind &&
          MmaStrassen::GetRWBit(ReadWriteMi, (MmaStrassen::Type)i) == MmaStrassen::RWSharedAddMiddle) {
        if (found == mi) return i - MmaStrassen::GlobalPreSumLevel1_M0;
        found++;
      }
    }
    return -1;
  }

  CUTLASS_DEVICE
  int combineMMALoop() {
    for (uint i = MmaStrassen::GlobalPreSumLevel1_M0; i <= MmaStrassen::GlobalPreSumLevel1_M6; i++) {
      if (i != MmaStrassenKind &&
          MmaStrassen::GetRWBit(ReadWriteMi, (MmaStrassen::Type)i) == MmaStrassen::RWCombineMMALoop) {
        return i - MmaStrassen::GlobalPreSumLevel1_M0;
      }
    }
    return -1;
  }

  // CUTLASS_DEVICE
  // bool computeForM(uint mi) {
  //   if (MmaStrassenKind == MmaStrassen::Type::GlobalPreSumLevel1_M6) {
  //     if (mi == 0) return GetRWBit(ReadWriteMi, MmaStrassen::Type::GlobalPreSumLevel1_M0) == MmaStrassen::RWSharedAddMiddle;
  //     if (mi == 1) return GetRWBit(ReadWriteMi, MmaStrassen::Type::GlobalPreSumLevel1_M3) == MmaStrassen::RWSharedAddMiddle;
  //     if (mi == 2) return GetRWBit(ReadWriteMi, MmaStrassen::Type::GlobalPreSumLevel1_M4) == MmaStrassen::RWSharedAddMiddle;
  //   }
    
  //   if (MmaStrassenKind == MmaStrassen::Type::GlobalPreSumLevel1_M5) {
  //     if (mi == 0) return GetRWBit(ReadWriteMi, MmaStrassen::Type::GlobalPreSumLevel1_M0) == MmaStrassen::RWSharedAddMiddle;
  //     if (mi == 2) return GetRWBit(ReadWriteMi, MmaStrassen::Type::GlobalPreSumLevel1_M1) == MmaStrassen::RWSharedAddMiddle;
  //     if (mi == 1) return GetRWBit(ReadWriteMi, MmaStrassen::Type::GlobalPreSumLevel1_M2) == MmaStrassen::RWSharedAddMiddle;
  //   }

  //   return false;
  // }

  /// GEMM prologue.  Bootstrap the global->shared memory pipeline by fetching
  /// the global fragments needed by the first kStages-1 threadblock mainloop iterations
  CUTLASS_DEVICE
  void prologue(
    IteratorA &iterator_A,      ///< [in|out] iterator over A operand in global memory
    PresumIteratorSrcA& iterator_A_src,
    PresumIteratorDstA& iterator_A_dst,
    IteratorB &iterator_B,      ///< [in|out] iterator over B operand in global memory
    PresumIteratorSrcB& iterator_B_src,
    PresumIteratorDstB& iterator_B_dst,
    PresumIteratorSrcM& iterator_M_src,
    int &gemm_k_iterations,
    int &presum_iterations)     ///< [in|out] number of threadblock mainloop iterations remaining
  {
    int storeMShIdx = 0;
    int computeForMShIdx = 0;

    // Issue several complete stages
    CUTLASS_PRAGMA_UNROLL
    for (int stage = 0; stage < Base::kStages - 1; ++stage, --gemm_k_iterations) {

      // Disable global fetching if done with global fetch iterations
      iterator_A.clear_mask(gemm_k_iterations == 0);
      iterator_B.clear_mask(gemm_k_iterations == 0);

      iterator_A.set_iteration_index(0);
      this->smem_iterator_A_.set_iteration_index(0);

      // Async Copy for operand A
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < Detail::AsyncCopyIterationsPerStageA; ++j) {
        typename IteratorA::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorA::AccessType *>(
                this->smem_iterator_A_.get());

        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorA::kAccessesPerVector; ++v) {
          int const kSrcBytes =
              sizeof_bits<typename IteratorA::Element>::value *
              IteratorA::ThreadMap::kElementsPerAccess /
              IteratorA::kAccessesPerVector / 8;

          int src_bytes = (iterator_A.valid() ? kSrcBytes : 0);

          cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpA>(
              dst_ptr + v, iterator_A.get(), iterator_A.valid());

          ++iterator_A;
        }

        ++this->smem_iterator_A_;
      }

      iterator_B.set_iteration_index(0);
      this->smem_iterator_B_.set_iteration_index(0);

      // Async Copy for operand B
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < Detail::AsyncCopyIterationsPerStageB; ++j) {
        typename IteratorB::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorB::AccessType *>(
                this->smem_iterator_B_.get());

        CUTLASS_PRAGMA_UNROLL
        for (int v = 0; v < IteratorB::kAccessesPerVector; ++v) {
          int const kSrcBytes =
              sizeof_bits<typename IteratorB::Element>::value *
              IteratorB::ThreadMap::kElementsPerAccess /
              IteratorB::kAccessesPerVector / 8;

          cutlass::arch::cp_async_zfill<kSrcBytes, kCacheOpB>(
              dst_ptr + v, iterator_B.get(), iterator_B.valid());

          ++iterator_B;
        }

        ++this->smem_iterator_B_;
      }

#ifdef PRESUM_OVERLAPPED
      if (addForNextM()) {
        for (int access = 0; access < PresumAccessPerIter; access++) {
          if (bitcheck(PresumLoadAs, 0)) {
            iterator_A_src.cp_async_presum(sharedPreSumA0.get(stage, access), iterator_A_src.get(0), iterator_A_dst.validTB());
          }
          if (bitcheck(PresumLoadAs, 1)) {
            iterator_A_src.cp_async_presum(sharedPreSumA1.get(stage, access), iterator_A_src.get(1), iterator_A_dst.validTB());
          }
          if (bitcheck(PresumLoadAs, 2)) {
            iterator_A_src.cp_async_presum(sharedPreSumA2.get(stage, access), iterator_A_src.get(2), iterator_A_dst.validTB());
          }
          if (bitcheck(PresumLoadAs, 3)) {
            iterator_A_src.cp_async_presum(sharedPreSumA3.get(stage, access), iterator_A_src.get(3), iterator_A_dst.validTB());
          }

          iterator_A_src.inc();
        }
        // --presum_iterations;
      }
      if (SplitKLastTB && storeMInSh() != -1) {
        for (int access = 0; access < 4; access++) {
          // uint idx = stage  * iterator_M_src.increment() * PresumAccessPerIter +
          //            access * iterator_M_src.increment();
          iterator_M_src.cp_async_presum(sharedPostSumM.getLinear(storeMShIdx),
                                         iterator_M_src.get(storeMInSh()), true);
          iterator_M_src.inc();
          storeMShIdx += BLOCK_DIM * 8;
        }
      }
      if (SplitKLastTB && computeForM(0) != -1) {
        for (int access = 0; access < 4; access++) {
          // uint idx = stage  * iterator_M_src.increment() * PresumAccessPerIter +
          //            access * iterator_M_src.increment();
          iterator_M_src.cp_async_presum(sharedPostSumM.getLinear(computeForMShIdx),
                                         iterator_M_src.get(0), true);
          iterator_M_src.inc();
          computeForMShIdx += BLOCK_DIM * 8;
        }
      }
      // if (computeForM()) {
      //   for (int access = 0; access < PresumAccessPerIter; access++) {
      //     if (computeForM(0)) {
      //       iterator_M_src.cp_async_presum(sharedPostSumM.get2(stage, 0, access),
      //                                     iterator_M_src.get(0), true);
      //     }
      //     if (computeForM(1)) {
      //       iterator_M_src.cp_async_presum(sharedPostSumM.get2(stage, 1, access),
      //                                     iterator_M_src.get(1), true);
      //     }
      //     // if (computeForM(2)) {
      //     //   iterator_M_src.cp_async_presum(sharedPostSumM.get2(stage, 2, 0),
      //     //                                  iterator_M_src.get(2), true);
      //     // }
      //     iterator_M_src.inc();
      //   }
      //   // iterator_M_src.load_in_shmem(sharedPreSumA2.get(stage), stage % 2, 3);
      //   // iterator_M_src.load_in_shmem(sharedPreSumA3.get(stage), stage % 2, 4);
      // }
#endif

      // Move to the next write stage
      advance_smem_write_stage(iterator_A, iterator_B);

      // Defines the boundary of a stage of cp.async.
      cutlass::arch::cp_async_fence();
    }

    if (SplitKLastTB && storeMInSh() != -1) {
      for (int access = 0; access < 4; access++) {
        // uint idx = stage  * iterator_M_src.increment() * PresumAccessPerIter +
        //            access * iterator_M_src.increment();
        iterator_M_src.cp_async_presum(sharedPostSumM.getLinear(storeMShIdx),
                                        iterator_M_src.get(storeMInSh()), true);
        iterator_M_src.inc();
        storeMShIdx += BLOCK_DIM * 8;
      }
    }

    if (SplitKLastTB && computeForM(0) != -1) {
      for (int access = 0; access < 4; access++) {
        // uint idx = stage  * iterator_M_src.increment() * PresumAccessPerIter +
        //            access * iterator_M_src.increment();
        iterator_M_src.cp_async_presum(sharedPostSumM.getLinear(computeForMShIdx),
                                        iterator_M_src.get(0), true);
        iterator_M_src.inc();
        computeForMShIdx += BLOCK_DIM * 8;
      }
    }

    // Optionally clear the remaining stages of SMEM. This is a functional requirement for
    // some kernels so that all accumulator elements outside the GEMM footprint are zero.
    if (SharedMemoryClear == SharedMemoryClearOption::kClearLastStage) {

      /// Iterator to write threadblock-scoped tile of A operand to shared memory
      SmemIteratorA last_smem_iterator_A(this->smem_iterator_A_);
      typename IteratorA::AccessType zero_A;

      zero_A.clear();
      last_smem_iterator_A.set_iteration_index(0);

      // Async Copy for operand A
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < Detail::AsyncCopyIterationsPerStageA; ++j) {

        typename IteratorA::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorA::AccessType *>(
                last_smem_iterator_A.get());

        *dst_ptr = zero_A;

        ++last_smem_iterator_A;
      }

      /// Iterator to write threadblock-scoped tile of B operand to shared memory
      SmemIteratorB last_smem_iterator_B(this->smem_iterator_B_);
      typename IteratorB::AccessType zero_B;

      zero_B.clear();
      last_smem_iterator_B.set_iteration_index(0);

      // Async Copy for operand B
      CUTLASS_PRAGMA_UNROLL
      for (int j = 0; j < Detail::AsyncCopyIterationsPerStageB; ++j) {

        typename IteratorB::AccessType *dst_ptr =
            reinterpret_cast<typename IteratorB::AccessType *>(
                last_smem_iterator_B.get());

        *dst_ptr = zero_B;

        ++last_smem_iterator_B;
      }
    }
  }


  /// Wait until we have at least one completed global fetch stage
  CUTLASS_DEVICE
  void gmem_wait(int read_idx, int presum_iterations, PresumIteratorDstA& iterator_A_dst)
  {
    // Wait until we have at least one committed global fetch stage. (#uncommitted = Base::kStages - 1 - #committed)
    cutlass::arch::cp_async_wait<Base::kStages - 2>();
    __syncthreads();
  }


  /// Perform a threadblock mainloop iteration of matrix multiply-accumulate
  CUTLASS_DEVICE
  void mac_loop_iter(
    PipeState &pipe_state,          ///< [in|out] loop-carried pipeline state
    FragmentC &accum,               ///< [in|out] destination accumulator tile
    IteratorA &iterator_A,          ///< [in|out] iterator over A operand in global memory
    PresumIteratorA& iterator_A_src,
    PresumIteratorA& iterator_A_dst,
    IteratorB &iterator_B,          ///< [in|out] iterator over B operand in global memory
    PresumIteratorSrcB& iterator_B_src,
    PresumIteratorDstB& iterator_B_dst,
    int &gemm_k_iterations,
    int &presum_iterations,
    bool gemm_k_iter_gt_0)         ///< [in|out] number of threadblock mainloop iterations remaining
  {  
    // Unroll the warp-level MMA tiles of a threadblock's mainloop iteration
    CUTLASS_PRAGMA_UNROLL
    for (int warp_mma_k = 0; warp_mma_k < Base::kWarpGemmIterations; ++warp_mma_k) {

      // Load the next warp-tile's A fragment from shared memory
      this->warp_tile_iterator_A_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations);
      this->warp_tile_iterator_A_.load(pipe_state.warp_loaded_frag_A_[(warp_mma_k + 1) % 2]);
      ++this->warp_tile_iterator_A_;

      // Load the next warp-tile's B fragment from shared memory
      this->warp_tile_iterator_B_.set_kgroup_index((warp_mma_k + 1) % Base::kWarpGemmIterations);
      this->warp_tile_iterator_B_.load(pipe_state.warp_loaded_frag_B_[(warp_mma_k + 1) % 2]);
      ++this->warp_tile_iterator_B_;

      // Except for the first warp-tile, all warp-tiles convert their incoming shared memory fragments as necessary
      if (warp_mma_k > 0) {
        warp_mma_.transform(
          pipe_state.warp_transformed_frag_A_[warp_mma_k % 2],
          pipe_state.warp_transformed_frag_B_[warp_mma_k % 2],
          pipe_state.warp_loaded_frag_A_[warp_mma_k % 2],
          pipe_state.warp_loaded_frag_B_[warp_mma_k % 2]);
      }

      // if (threadIdx.x == 0 && blockIdx.x == 1 && MmaStrassenKind == MmaStrassen::Type::GlobalPreSumLevel1_M0)
        // printf("533 %f %f %f\n", (float)pipe_state.warp_transformed_frag_A_[warp_mma_k % 2][0],
          // (float)pipe_state.warp_transformed_frag_B_[warp_mma_k % 2][0], (float)accum[0]);
      // Execute the current warp-tile of MMA operations
      if (Detail::kStagedAccumulation) {
        warp_mma_(
          pipe_state.tmp_accum_,
          pipe_state.warp_transformed_frag_A_[warp_mma_k % 2],
          pipe_state.warp_transformed_frag_B_[warp_mma_k % 2],
          pipe_state.tmp_accum_
        );

        if (warp_mma_k == 0) {
          plus<FragmentC> plus_accum;
          accum = plus_accum(accum, pipe_state.tmp_accum_);
          pipe_state.tmp_accum_.clear();
        }
      } else {
        warp_mma_(
          accum,
          pipe_state.warp_transformed_frag_A_[warp_mma_k % 2],
          pipe_state.warp_transformed_frag_B_[warp_mma_k % 2],
          accum
        );
      }

      // Except for the last warp-tile, all warp-tiles issue their share of
      // global->shared fragment copies
      if (warp_mma_k < Base::kWarpGemmIterations - 1) {

        int group_start_iteration_A, group_start_iteration_B;
        group_start_iteration_A = warp_mma_k * Detail::kAccessesPerGroupA;
        group_start_iteration_B = warp_mma_k * Detail::kAccessesPerGroupB;

        copy_tiles_and_advance(
            iterator_A,
            iterator_B,
            group_start_iteration_A,
            group_start_iteration_B);
      }

      // The second-to-last warp-tile also:
      //   - performs the last warp-tile's share of global->shared fragment copies
      //   - moves to the next global fetch stage
      if (warp_mma_k + 2 == Base::kWarpGemmIterations) {

        // Performs the last warp-tile's share of global->shared fragment copies
        int group_start_iteration_A = (warp_mma_k + 1) * Detail::kAccessesPerGroupA;
        int group_start_iteration_B = (warp_mma_k + 1) * Detail::kAccessesPerGroupB;

        copy_tiles_and_advance(
          iterator_A,
          iterator_B,
          group_start_iteration_A,
          group_start_iteration_B);

        // Inserts a memory fence between stages of cp.async instructions.
        cutlass::arch::cp_async_fence();

        // Move to the next global fetch stage
        advance_smem_write_stage(iterator_A, iterator_B);
        advance_smem_read_stage();
        // Wait until we have at least one completed global fetch stage
        gmem_wait(smem_read_stage_idx_, 0, iterator_B_dst);

        // Disable global fetching when done with global fetch iterations
        if (gemm_k_iter_gt_0) {}
        else {
          --gemm_k_iterations;
          iterator_A.clear_mask(gemm_k_iterations == 0);
          iterator_B.clear_mask(gemm_k_iterations == 0);
        }
      }

      // The last warp-tile also converts the shared memory fragments used by
      // the first warp-tile of the next iteration, if necessary (so we can
      // immediately start issuing MMA instructions at the top of the loop )
      if (warp_mma_k + 1 == Base::kWarpGemmIterations) {

        warp_mma_.transform(
          pipe_state.warp_transformed_frag_A_[(warp_mma_k + 1) % 2],
          pipe_state.warp_transformed_frag_B_[(warp_mma_k + 1) % 2],
          pipe_state.warp_loaded_frag_A_[(warp_mma_k + 1) % 2],
          pipe_state.warp_loaded_frag_B_[(warp_mma_k + 1) % 2]);
      }

    }
  }


  /// Perform the specified number of threadblock mainloop iterations of matrix
  /// multiply-accumulate.  Assumes prologue has been initiated.
  template<typename AccumFragIter>
  CUTLASS_DEVICE
  void gemm_iters(
      int gemm_k_iterations,        ///< number of threadblock mainloop iterations
      int &presum_iterations,
      FragmentC &accum,             ///< [in|out] accumulator tile
      IteratorA &iterator_A,        ///< [in|out] iterator over A operand in global memory
      PresumIteratorA& iterator_A_src,
      PresumIteratorA& iterator_A_dst,
      PresumIteratorA& iterator_D_src,
      IteratorB &iterator_B,
      PresumIteratorSrcB& iterator_B_src,
      PresumIteratorDstB& iterator_B_dst,
      PresumIteratorSrcM& iterator_M_src,
      PresumIteratorA& iterator_M_dst)        ///< [in|out] iterator over B operand in global memory
  {
    PipeState pipe_state;
    // Disable global fetching if done with global fetch iterations
    iterator_A.clear_mask(gemm_k_iterations == 0);
    iterator_B.clear_mask(gemm_k_iterations == 0);

    // Load first warp-tile's A fragment from shared memory
    this->warp_tile_iterator_A_.set_kgroup_index(0);
    this->warp_tile_iterator_A_.load(pipe_state.warp_loaded_frag_A_[0]);
    ++this->warp_tile_iterator_A_;

    // Load first warp-tile's B fragment from shared memory
    this->warp_tile_iterator_B_.set_kgroup_index(0);
    this->warp_tile_iterator_B_.load(pipe_state.warp_loaded_frag_B_[0]);
    ++this->warp_tile_iterator_B_;

    // Transform, if necessary, the first warp-tile's shared memory fragments
    warp_mma_.transform(
      pipe_state.warp_transformed_frag_A_[0],
      pipe_state.warp_transformed_frag_B_[0],
      pipe_state.warp_loaded_frag_A_[0],
      pipe_state.warp_loaded_frag_B_[0]);

    if (Detail::kStagedAccumulation) {
      pipe_state.tmp_accum_.clear();
    }
    
    iterator_A_dst.reset();
    iterator_A_src.reset();
    // iterator_M_src.reset();
    iterator_M_dst.reset();

    // total_gemm_k_iterations = 192-2;
          // gemm_k_iterations = 192-2;
    const uint MaxPresumIterations        = ((PresumShape::kCount)/(8*BLOCK_DIM))/PresumAccessPerIter;
    const uint MaxPresumIterationsPerLoop = ((PresumShape::kCount)/(8*BLOCK_DIM))/PresumAccessPerIter;
    uint RemainingIterations = 0;
    if (SplitKLastTB && (storeMInSh() != -1 || computeForM(0) != -1) &&
        MaxPresumIterations - Base::kStages > 0) {
      uint iters = (Shape::kM*Shape::kN)/(8*BLOCK_DIM);
      RemainingIterations = iters/PresumAccessPerIter - Base::kStages;
    }
    
    if (RemainingIterations > 0) {
    if (gemm_k_iterations >= RemainingIterations) {
      // Mainloop
      mac_loop_iter(
        pipe_state,
        accum,
        iterator_A, iterator_A_src, iterator_A_dst,
        iterator_B, iterator_B_src, iterator_B_dst,
        gemm_k_iterations,
        presum_iterations,
       true);

      gemm_k_iterations -= RemainingIterations;

      uint storeMShIdx = (RemainingIterations - 1 + Base::kStages)*PresumAccessPerIter*BLOCK_DIM*8;
      if (storeMInSh() != -1) {
        for (int access = 0; access < 4; access++) {
          // uint idx = stage  * iterator_M_src.increment() * PresumAccessPerIter +
          //            access * iterator_M_src.increment();
          iterator_M_src.cp_async_presum(sharedPostSumM.getLinear(storeMShIdx),
                                          iterator_M_src.get(storeMInSh()), true);
          iterator_M_src.inc();
          storeMShIdx += BLOCK_DIM * 8;
        }
      }
    } else __builtin_unreachable(); //helps in 8k x 8k x 8k
    }

    if (addForNextM() && PresumM0OrM1 != 0) {
    int i = 0;
    uint total_gemm_k_iterations = (combineMMALoop() != -1) ? gemm_k_iterations + Base::kStages - 1 : gemm_k_iterations;

    if (gemm_k_iterations >= MaxPresumIterations and \
        (gemm_k_iterations + Base::kStages - 1) % MaxPresumIterations == 0) {
        bool validTB = iterator_A_dst.validTB();
        //Should be here but was not due to some reason: CUTLASS_GEMM_LOOP
        for (; gemm_k_iterations > (-Base::kStages + 1);) {
          if (i < MaxPresumIterationsPerLoop - (Base::kStages - 1)) {
            uint stage = (i+Base::kStages-1)%Base::kStages;
            for (int access = 0; access < PresumAccessPerIter; access++) {
              iterator_A_src.reset();
              iterator_A_src.row += (i + Base::kStages-1) * iterator_A_src.increment() * PresumAccessPerIter +
                                    access * iterator_A_src.increment();
              if (bitcheck(PresumLoadAs, 0)) {
                iterator_A_src.cp_async_presum(sharedPreSumA0.get(stage, access), iterator_A_src.get(0), validTB);
              }
              if (bitcheck(PresumLoadAs, 1)) {
                iterator_A_src.cp_async_presum(sharedPreSumA1.get(stage, access), iterator_A_src.get(1), validTB);
              }
              if (bitcheck(PresumLoadAs, 2)) {
                iterator_A_src.cp_async_presum(sharedPreSumA2.get(stage, access), iterator_A_src.get(2), validTB);
              }
              if (bitcheck(PresumLoadAs, 3)) {
                iterator_A_src.cp_async_presum(sharedPreSumA3.get(stage, access), iterator_A_src.get(3), validTB);
              }
              // iterator_A_src.inc();
            }
          }
          mac_loop_iter(
            pipe_state,
            accum,
            iterator_A, iterator_A_src, iterator_A_dst,
            iterator_B, iterator_B_src, iterator_B_dst,
            gemm_k_iterations,
            presum_iterations, combineMMALoop() != -1);
          if (combineMMALoop() != -1) gemm_k_iterations--;
          if (i < MaxPresumIterationsPerLoop) {
            for (int access = 0; access < PresumAccessPerIter; access++) {
            using VecEl = Array<half2, 4>;
            // #pragma unroll 4
            for (int v = 0; v < sizeof(float4)/sizeof(VecEl); v += 1) {
              uint elems = sizeof(VecEl)/sizeof(half);
              iterator_A_dst.reset();
              iterator_A_dst.col = (threadIdx.x*elems) %  PresumShape::kN;
              iterator_A_dst.row = (threadIdx.x*elems) /  PresumShape::kN + i * iterator_A_dst.increment() * PresumAccessPerIter + (v *elems* BLOCK_DIM)/PresumShape::kN;
              //TODO: Using shared_load_128b increases reg usage a lot and provides very little perf improvement
              VecEl a0; shared_load_128b(&a0, (sharedPreSumA0.get((i)%Base::kStages, v, sizeof(VecEl))));
              VecEl a1; shared_load_128b(&a1, (sharedPreSumA1.get((i)%Base::kStages, v, sizeof(VecEl))));
              VecEl a2; shared_load_128b(&a2, (sharedPreSumA2.get((i)%Base::kStages, v, sizeof(VecEl))));
              VecEl a3; shared_load_128b(&a3, (sharedPreSumA3.get((i)%Base::kStages, v, sizeof(VecEl))));

              // VecEl a0 = *(VecEl*)(sharedPreSumA0.get((i)%Base::kStages, v, sizeof(VecEl)));
              // VecEl a1 = *(VecEl*)(sharedPreSumA1.get((i)%Base::kStages, v, sizeof(VecEl)));
              // VecEl a2 = *(VecEl*)(sharedPreSumA2.get((i)%Base::kStages, v, sizeof(VecEl)));
              // VecEl a3 = *(VecEl*)(sharedPreSumA3.get((i)%Base::kStages, v, sizeof(VecEl)));

              if (bitcheck(PresumM0OrM1, MmaStrassen::Type::GlobalPreSumLevel1_M2)) {
                auto a230 = frag_converter(a2) + frag_converter(a3) - frag_converter(a0);
                if (elems < 8)
                  *(VecEl*)(iterator_A_dst.get(0)) = frag_converter(a230);
                else
                  arch::global_store<VecEl, sizeof(VecEl)>
                  (frag_converter(a230), iterator_A_dst.get(0), validTB);
              }
              if (bitcheck(PresumM0OrM1, MmaStrassen::Type::GlobalPreSumLevel1_M3)) {
                auto a02 = frag_converter(a0) - frag_converter(a2);
                if (elems < 8)
                  *(VecEl*)(iterator_A_dst.get(1)) = frag_converter(a02);
                else
                  arch::global_store<VecEl, sizeof(VecEl)>
                  (frag_converter(a02), iterator_A_dst.get(1), validTB);
              }
              if (bitcheck(PresumM0OrM1, MmaStrassen::Type::GlobalPreSumLevel1_M4)) {
                auto a23 = frag_converter(a2) + frag_converter(a3);
                if (elems < 8)
                  *(VecEl*)(iterator_A_dst.get(2)) = frag_converter(a23);
                else
                  arch::global_store<VecEl, sizeof(VecEl)>
                  (frag_converter(a23), iterator_A_dst.get(2), validTB);
              }
              if (bitcheck(PresumM0OrM1, MmaStrassen::Type::GlobalPreSumLevel1_M5)) {
                auto a1230 =  frag_converter(a1)-(frag_converter(a2) + frag_converter(a3) - frag_converter(a0));
                if (elems < 8)
                  *(VecEl*)(iterator_A_dst.get(3)) = frag_converter(a1230);
                else
                  arch::global_store<VecEl, sizeof(VecEl)>
                  (frag_converter(a1230), iterator_A_dst.get(3), validTB);
              }
            }
          }}
          i++;
        }

        if (combineMMALoop() != -1) {
          if (true) {
            iterator_A_dst.reset();
            float4* ptr = (float4*)sharedPostSumM.pointer;
            int i = 0;
            while (i < accum.size()) {
              float* x = (float*)(&accum[i]);

              if (true) {
                half hx[8] = {0};
                for (int j = 0; j < 8; j++) {
                  hx[j] = (half)x[j];
                }

                ptr[threadIdx.x + (i/8) * BLOCK_DIM] = *(float4*)&hx;
              }
              i += 8;
            }
          }
          gemm_k_iterations = total_gemm_k_iterations - Base::kStages + 1;
          //Should be here but was not due to some reason: CUTLASS_GEMM_LOOP
          for (; gemm_k_iterations > (-Base::kStages + 1);) {
            mac_loop_iter(
              pipe_state,
              accum,
              iterator_A, iterator_A_src, iterator_A_dst,
              iterator_B, iterator_B_src, iterator_B_dst,
              gemm_k_iterations,
              presum_iterations, false);
          }
        }
      } else __builtin_unreachable();
    } else {
      // Mainloop
      //Should be here but was not due to some reason: CUTLASS_GEMM_LOOP
      for (; gemm_k_iterations > (-Base::kStages + 1);) {
        mac_loop_iter(
          pipe_state,
          accum,
          iterator_A, iterator_A_src, iterator_A_dst,
          iterator_B, iterator_B_src, iterator_B_dst,
          gemm_k_iterations,
          presum_iterations, false);
      }
    }

    if (Detail::kStagedAccumulation) {
      plus<FragmentC> plus_accum;
      accum = plus_accum(accum, pipe_state.tmp_accum_);
    }

    // Commit and drain all pending and predicated cp.async pnz from the GEMM mainloop
  cutlass::arch::cp_async_fence();
  cutlass::arch::cp_async_wait<0>();
  if (PresumM0OrM1 != 0)
    __syncthreads();
  }


  /// Prepares the class for another prologue.
  CUTLASS_DEVICE
  void wind_down()
  {
    // Catch-up the smem-read iterator to the smem-write iterator (so this class can be reused for another tile's prologue)

    // First, increment remaining warp tiles to get to the next full stage.  (Ideally we would
    // just decrement one tile, but not all iterators implement --() decrement.)
    #pragma unroll
    for (int warp_mma_k = 1; warp_mma_k < Base::kWarpGemmIterations; ++warp_mma_k)
    {
      this->warp_tile_iterator_A_.set_kgroup_index(warp_mma_k);
      this->warp_tile_iterator_B_.set_kgroup_index(warp_mma_k);

      ++this->warp_tile_iterator_A_;
      ++this->warp_tile_iterator_B_;
    }
    smem_read_stage_idx_++;

    // Then wrap back two full stages (one for the tile advancing we just did, and one to catch the write iterators)
    static const int kStageIters = Policy::kPartitionsK * Base::kWarpGemmIterations;
    if (smem_read_stage_idx_ > 1)
    {
      this->warp_tile_iterator_A_.add_tile_offset({0, (-2 * kStageIters)});
      this->warp_tile_iterator_B_.add_tile_offset({(-2 * kStageIters), 0});
    }
    else
    {
      this->warp_tile_iterator_A_.add_tile_offset({0, ((Base::kStages - 2) * kStageIters)});
      this->warp_tile_iterator_B_.add_tile_offset({((Base::kStages - 2) * kStageIters), 0});
    }
    smem_read_stage_idx_ = smem_write_stage_idx_;
  }


  /// Perform a threadblock-scoped matrix multiply-accumulate
  template<typename AccumFragIter>
  CUTLASS_DEVICE
  void operator()(
      ///< problem size of GEMM
      int gemm_k_iterations,
      ///< destination accumulator tile
      FragmentC &accum,
      ///< iterator over A operand in global memory
      IteratorA iterator_A,
      PresumIteratorA iterator_A_src,
      PresumIteratorA iterator_A_dst,
      PresumIteratorA iterator_D_src,
      ///< iterator over B operand in global memory
      IteratorB iterator_B,
      PresumIteratorSrcB iterator_B_src,
      PresumIteratorDstB iterator_B_dst,
      PresumIteratorSrcM iterator_M_src,
      PresumIteratorA iterator_M_dst,
      bool SplitKLastTB,
      bool SplitKSerial,
      PostsumSemaphore postsum_semaphore) {
    bool accumInShMem = false;
    bool negAccum = false;
    for (uint i = MmaStrassen::GlobalPreSumLevel1_M0; i <= MmaStrassen::GlobalPreSumLevel1_M6; i++) {
      if (i != MmaStrassenKind &&
          MmaStrassen::GetRWBit(ReadWriteMi, (MmaStrassen::Type)i) == MmaStrassen::RWAccumInShared) {
        accumInShMem = true;
        negAccum = MmaStrassen::GetRWMulSign(ReadWriteMi, (MmaStrassen::Type)i);
        break;
      }
    }
    this->SplitKLastTB = SplitKLastTB;
    accumInShMem = SplitKLastTB && accumInShMem;

    if (accumInShMem) {
      //TODO: Below leads to shared memory bank conflicts sometime
      iterator_A_dst.reset();
      float4* ptr = (float4*)sharedPostSumM.pointer;
      int i = 0;
      bool negMi = MmaStrassen::GetRWBit(ReadWriteMi, MmaStrassenKind) == MmaStrassen::RWUseSign &&
                   MmaStrassen::GetRWMulSign(ReadWriteMi, MmaStrassenKind);
      while (i < accum.size()) {
        float4 x = ptr[threadIdx.x + (i/8) * BLOCK_DIM];
        half* hx = (half*)&x;
        for (int j = 0; j < 8; j+=2) {
          float x = (negAccum) ? -1 : 1;
          if (negMi) x = -x;
          float2 fx = __half22float2(half2{hx[j], hx[j+1]});
          accum[i + j] = x * fx.x;
          accum[i + j + 1] = x * fx.y;
        }
        i += 8;
      }
    }

  #ifdef PRESUM_OVERLAPPED
    if (false && (addForNextM() || subForNextM())) {
      iterator_A_src.reset();
      iterator_A_dst.reset();

      while(iterator_A_src.valid()) {
        *iterator_A_dst.get(0) = iterator_A_dst.presum(*iterator_A_src.get(0),
                                                       *iterator_A_src.get(1),
                                                       addForNextM(),
                                                       subForNextM());
        iterator_A_src.inc();
        iterator_A_dst.inc();
      }
    }
  #endif
    int presum_iterations = 0;

    {
      bool global_sync = false;
      if (storeMInSh(nullptr, &global_sync) != -1) {
        if (global_sync) postsum_semaphore.wait(storeMInSh(), 0);
      }
     }

    // Prologue (start fetching iterations of global fragments into shared memory)
    prologue(iterator_A, iterator_A_src, iterator_A_dst, iterator_B, 
             iterator_B_src, iterator_B_dst, iterator_M_src, 
             gemm_k_iterations, presum_iterations);

    // Wait until we have at least one completed global fetch stage
    gmem_wait(0, presum_iterations, iterator_B_dst);

    // Perform the MAC-iterations
    gemm_iters<AccumFragIter>(gemm_k_iterations, presum_iterations, accum,
               iterator_A, iterator_A_src, iterator_A_dst,
               iterator_D_src, 
               iterator_B, iterator_B_src, iterator_B_dst,
               iterator_M_src, iterator_M_dst);

    if (accumInShMem &&
        MmaStrassen::GetRWBit(ReadWriteMi, MmaStrassenKind) == MmaStrassen::RWUseSign &&
        MmaStrassen::GetRWMulSign(ReadWriteMi, MmaStrassenKind)) {
      for (int i = 0; i < accum.size(); i++) accum[i] = -1 * accum[i];
    }

    bool storeAccumToShared = GetRWBit(ReadWriteMi, MmaStrassenKind) == MmaStrassen::RWShared or
                              GetRWBit(ReadWriteMi, MmaStrassenKind) == MmaStrassen::RWGlobalContigAndShared;

    if (!SplitKSerial && //Need to have this condition otherwise reg spills happen
        (storeAccumToShared or storeMInSh() != -1)) {
      iterator_A_dst.reset();
      float4* ptr = (float4*)sharedPostSumM.pointer;
      int i = 0;
      while (i < accum.size()) {
        float* x = (float*)(&accum[i]);
        float4 prevM = ptr[threadIdx.x + (i/8) * BLOCK_DIM];
        bool neg;
        bool negAccum = MmaStrassen::GetRWMulSign(ReadWriteMi, MmaStrassenKind);
        if (storeMInSh(&neg) != -1) {
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

    if (SplitKLastTB && computeForM(1) != -1) {
      iterator_A_dst.reset();
      float4* ptr = (float4*)sharedPostSumM.pointer;
      int i = 0;
      while (i < accum.size()) {
        float* x = (float*)(&accum[i]);
        float4 prevM = ptr[threadIdx.x + (i/8) * BLOCK_DIM];
        bool neg = false;
        half2* prevMh = (half2*)&prevM;
        for (int j = 0; j < 8; j += 2) {
          float2 xx = __half22float2(prevMh[j/2]);
          float mul = neg ? -1 : 1;
          x[j]   = x[j]   + mul * xx.x;
          x[j+1] = x[j+1] + mul * xx.y;
        }
        i += 8;
      }
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace gemm
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////

