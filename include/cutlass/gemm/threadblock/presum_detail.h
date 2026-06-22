#include<type_traits>

#include "cutlass/array.h"

#pragma once

namespace cutlass {
namespace gemm {
namespace threadblock {

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace PresumDetail {
template<class T>
CUTLASS_DEVICE
T shared_load_128b(void* ptr) {
  uint64_t addr64 = 0;
  float x, y, z, w;
  asm volatile("cvta.to.shared.u64 %0, %1;": "=l"(addr64) : "l"(ptr));
  asm volatile("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];\n"
    : "=f"(x), "=f"(y), "=f"(z), "=f"(w)
    : "l"(addr64));

  T dst;
  float4& packed = reinterpret_cast<float4&>(dst);
  packed.x = x;
  packed.y = y;
  packed.z = z;
  packed.w = w;
  return dst;
}

// ld.shared - 128b
CUTLASS_DEVICE
void shared_load_128b(void *dst, void* ptr) {
  float4 *dst_u128 = reinterpret_cast<float4 *>(dst);
  // *dst_u128 = *(float4*)ptr;
  uint64_t addr64 = 0;
  // // uint addr32;
  asm volatile("cvta.to.shared.u64 %0, %1;": "=l"(addr64) : "l"(ptr));
  // // addr32 = (uint)addr64;
  asm volatile("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];\n"
    :
      "=f"(dst_u128->x),
      "=f"(dst_u128->y),
      "=f"(dst_u128->z),
      "=f"(dst_u128->w)
    : "l"(addr64));
}

// ld.shared - 128b
CUTLASS_DEVICE
void shared_store_128b(void *dst, void* ptr) {
  float4 *dst_u128 = reinterpret_cast<float4 *>(dst);
  // *dst_u128 = *(float4*)ptr;
  // uint64_t addr64 = 0;
  // uint addr32;
  // asm volatile("cvta.to.shared.u64 %0, %1;": "=l"(addr64) : "l"(ptr));
  // addr32 = (uint)addr64;
  asm volatile("st.shared.v4.f32 [%4], {%0, %1, %2, %3};\n"
    :
      "=f"(dst_u128->x),
      "=f"(dst_u128->y),
      "=f"(dst_u128->z),
      "=f"(dst_u128->w)
    : "l"(ptr));
}

// CUTLASS_DEVICE
// void shared_load_32b(void *dst, void* ptr) {
//   float& dst_32b = *reinterpret_cast<float *>(dst);
//   // *dst_u128 = *(float4*)ptr;
//   uint64_t addr64 = 0;
//   // uint addr32;
//   asm volatile("cvta.to.shared.u64 %0, %1;": "=l"(addr64) : "l"(ptr));
//   // addr32 = (uint)addr64;
//   asm volatile("ld.shared.f32 %0, [%1];\n"
//     :
//       "=f"(dst_32b)
//     : "l"(addr64));
// }

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
        "r"(smem_int_ptr), "l"(global_ptr), "n"(16));
  #endif
}

template<typename Element, uint BLOCK_DIM,
         typename MmaTileShape, typename VectorLoadType,
         bool isAorB,//TODO: Change this to layout. false for A and true for B
         bool LinearIndexing>
struct GlobalIterator {
  int row;
  int col;
  Element* pointer;
  char* byte_pointer_;
  MatrixCoord extent;
  long nextOffset0, nextOffset1, nextOffset2, nextOffset3, nextOffset4;
  static const uint VectorLoadElems = sizeof(VectorLoadType)/sizeof(Element);
  static const uint LdBytes = sizeof(VectorLoadType);
  static const uint kIterations = (MmaTileShape::kM * MmaTileShape::kN) / (BLOCK_DIM * VectorLoadElems);
  MatrixCoord tb_offset;
  bool isValidTB;
  bool isDest;
  uint linear_block_idx;
  uint stride;
  uint threadId;

  CUTLASS_DEVICE
  GlobalIterator(Element* pointer_, uint stride_, MatrixCoord extent_, MatrixCoord tb_offset_,
                 uint linear_block_idx,
                 MatrixCoord nextOffset0_, uint threadId,
                 MatrixCoord nextOffset1_=MatrixCoord(0,0),
                 MatrixCoord nextOffset2_=MatrixCoord(0,0),
                 MatrixCoord nextOffset3_=MatrixCoord(0,0),
                 MatrixCoord nextOffset4_=MatrixCoord(0,0),
                 bool isDest = false) : 
    extent(extent_), stride(stride_), tb_offset(tb_offset_), isDest(isDest), linear_block_idx(linear_block_idx), threadId(threadId)
  {
    nextOffset0 = nextOffset0_.row() * stride + nextOffset0_.column();
    nextOffset1 = nextOffset1_.row() * stride + nextOffset1_.column();
    nextOffset2 = nextOffset2_.row() * stride + nextOffset2_.column();
    nextOffset3 = nextOffset3_.row() * stride + nextOffset3_.column();
    nextOffset4 = nextOffset4_.row() * stride + nextOffset4_.column();
    pointer = pointer_;

    reset();
    //TODO: compare only columns for A and only rows for B
    isValidTB = (!isAorB && tb_offset.column() < stride) || (isAorB && tb_offset.row() < extent.row());
  }

  CUTLASS_DEVICE
  void inc() {
    if (LinearIndexing) {
      col += VectorLoadElems * BLOCK_DIM;
    } else {
      if (BLOCK_DIM*VectorLoadElems >= MmaTileShape::kN) {
        row += (BLOCK_DIM*VectorLoadElems)/MmaTileShape::kN;
        byte_pointer_ += (BLOCK_DIM*VectorLoadElems)/MmaTileShape::kN*sizeof(Element)*stride;
      } else {
        col += BLOCK_DIM*VectorLoadElems;
      }
    }
  }

  CUTLASS_DEVICE
  void set_iteration(int iter) {
    byte_pointer_ += iter*(BLOCK_DIM*VectorLoadElems)/MmaTileShape::kN*sizeof(Element)*stride;
    row += iter*(BLOCK_DIM*VectorLoadElems)/MmaTileShape::kN;
  }

  CUTLASS_DEVICE
  int row_increment() {
    if (LinearIndexing) {
      return VectorLoadElems * BLOCK_DIM;
    } else if (BLOCK_DIM*VectorLoadElems >= MmaTileShape::kN) {
      return (BLOCK_DIM*VectorLoadElems)/MmaTileShape::kN;
    } else {
      return BLOCK_DIM*VectorLoadElems;
    }
  }

  CUTLASS_DEVICE
  void reset(int tile = 0) {
    // if (LinearIndexing) {
    //   col = linear_block_idx;
    //   col = col * MmaTileShape::kM * MmaTileShape::kN + threadIdx.x * 8;
    //   row = 0;
    // } else
    {
      col = ((!isAorB)?tile:0)*MmaTileShape::kN + (threadId*VectorLoadElems) %  MmaTileShape::kN;
      row = ((!isAorB)?0:tile)*MmaTileShape::kM + (threadId*VectorLoadElems) / MmaTileShape::kN;

      byte_pointer_ = (char*)(pointer + (tb_offset.row() + row) * stride +
                                         tb_offset.column() + col);
    }
  }

  CUTLASS_DEVICE
  bool valid() {
    if (LinearIndexing) return col < MmaTileShape::kM * MmaTileShape::kN;
    return (!isAorB) ? (tb_offset.column() + col < stride) : (tb_offset.row() + row < extent.row());
  }
  
  CUTLASS_DEVICE
  bool validTB() {
    return isValidTB;
  }

  CUTLASS_DEVICE
  VectorLoadType* get(int part, bool b = false) {
    // if (LinearIndexing or not isDest) {
      auto pointer0 = (Element*)byte_pointer_;//((LinearIndexing) ? col : ((tb_offset.row() + row) * extent.column() + (tb_offset.column() + col)));
      if (LinearIndexing) {
        return (VectorLoadType*)(pointer0 + part * extent.row() * stride);
       } else {
        if (part == 0)      return (VectorLoadType*)(pointer0 + nextOffset0);
        else if (part == 1) return (VectorLoadType*)(pointer0 + nextOffset1);
        else if (part == 2) return (VectorLoadType*)(pointer0 + nextOffset2);
        else if (part == 3) return (VectorLoadType*)(pointer0 + nextOffset3);
        else if (part == 4) return (VectorLoadType*)(pointer0 + nextOffset4);

        return nullptr;
      }

      return nullptr;
    // }

    // auto pointer0 = pointer + (tb_offset.row() + row) * extent.column()*2 + (tb_offset.column() + col);
    //      if (part == 0) return (VectorLoadType*)(pointer0 + nextOffset0);
    // else if (part == 1) return (VectorLoadType*)(pointer0 + nextOffset1);

    // auto pointer1 = pointer + ((LinearIndexing) ? col : ((tb_offset.row() + row) * (extent.column()) + (tb_offset.column() + col)));
    //     //  if (part == 1) return (VectorLoadType*)(pointer1 + nextOffset1);
    // if (part == 2) return (VectorLoadType*)(pointer1 + nextOffset2);
    // else if (part == 3) return (VectorLoadType*)(pointer1 + nextOffset3);
    // else if (part == 4) return (VectorLoadType*)(pointer1 + nextOffset4);
    // return nullptr; //(VectorLoadType*)pointer0;
  }
};

template<typename Element, typename VecLoadType_, uint BLOCK_DIM,
         uint NumLoads, uint Size, uint Stages, uint PresumAccessPerIter>
struct SharedIterator {
  Element* pointer;
  uint threadId;
  using VecLoadType = VecLoadType_;

  CUTLASS_DEVICE
  SharedIterator(Element* pointer_, uint threadId_) {
    threadId = threadId_;
    // uint64_t addr64 = 0;
    // asm volatile("cvta.to.shared.u64 %0, %1;": "=l"(addr64) : "l"(pointer_));
    // unsigned smem_int_ptr = cutlass::arch::cutlass_get_smem_pointer(pointer_);
    pointer = ((Element*)pointer_) + threadId_ * (sizeof(VecLoadType)/sizeof(Element));
  }

  CUTLASS_DEVICE
  Element* get(int load, int stage, int access = 0) {
    auto p = pointer  + load*Size*Stages + stage * Size +
             access   * BLOCK_DIM * (sizeof(VecLoadType)/sizeof(Element));
    
    return p;
  }

  CUTLASS_DEVICE
  Element* getLinear(int presum, int idx) {
    auto p = pointer  + idx +
             threadId * (sizeof(VecLoadType)/sizeof(Element));
    return p;
  }

  CUTLASS_DEVICE
  Element* get2(int stage, int mi, int part) {
    auto p = pointer  + mi * Stages * Size + stage * Size + 
             part     * (sizeof(VecLoadType)/sizeof(Element)) * BLOCK_DIM +
             threadId * (sizeof(VecLoadType)/sizeof(Element));
    
    return p;
  }

  CUTLASS_DEVICE
  void inc() {}
};
}
}
}
}