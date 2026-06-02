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
    \brief Template for a pipelined GEMM kernel. Does not compute batching or support split-K.
*/

#pragma once

#include "cutlass/cutlass.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/semaphore.h"
#include "cutlass/mutex.h"
#include "cutlass/arch/arch.h"
#include "cutlass/postsum_semaphore.h"

#include "cutlass/gemm/threadblock/strassen_mma_base.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename Mma_,                  ///! Threadblock-scoped matrix multiply-accumulate 
  typename Epilogue_,             ///! Epilogue
  typename ThreadblockSwizzle_,   ///! Threadblock swizzling function
  bool SplitKSerial,               ///! If true, code supporting split-K via serial reduction is enabled.
  bool SubGemmParallel
>
struct StrassenPreSumGemm {

  using Mma = Mma_;
  using Epilogue = Epilogue_;
  using OutputOp = typename Epilogue::OutputOp;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  static bool const kSplitKSerial = SplitKSerial;

  /// Warp count (concept: GemmShape)
  using WarpCount = typename Mma::WarpCount;
  static int const kThreadCount = 32 * WarpCount::kCount;

  /// Parameters structure
  struct Params {
    int run;
    cutlass::gemm::GemmCoord problem_size;
    cutlass::gemm::GemmCoord grid_tiled_shape;
    int swizzle_log_tile;
    typename Mma::IteratorA::Params params_A;
    typename Mma::IteratorA::TensorRef ref_A;
    typename Mma::IteratorB::Params params_B;
    typename Mma::IteratorB::TensorRef ref_B;
    typename Epilogue::OutputTileIterator::Params params_C;
    typename Epilogue::OutputTileIterator::TensorRef ref_C;
    typename Epilogue::OutputTileIterator::Params params_D;
    typename Epilogue::OutputTileIterator::TensorRef ref_D;
    typename Epilogue::OutputTileIterator::Params params_M;
    typename OutputOp::Params output_op;
    int SrcATilePerZ;
    int *semaphore;
    int gemm_k_size;
    // For gather+scatter operations
    int const *gather_A_indices;
    int const *gather_B_indices;
    int const *scatter_D_indices;

    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    Params(): swizzle_log_tile(0), semaphore(0), gemm_k_size(0), run(0) { }

    CUTLASS_HOST_DEVICE
    static typename Mma::IteratorA::TensorRef fixLayoutA(typename Mma::IteratorA::TensorRef ref_A) {
      if (Mma::MmaStrassenKind == MmaStrassen::Type::GlobalPreSumLevel1_M0) return ref_A;
      if (Mma::MmaStrassenKind == MmaStrassen::Type::GlobalPreSumLevel1_M1) return ref_A;
      if (Mma::MmaStrassenKind == MmaStrassen::Type::GlobalPreSumLevel1_M2) return ref_A;
      if (Mma::MmaStrassenKind == MmaStrassen::Type::GlobalPreSumLevel1_M3) return ref_A;
      if (Mma::MmaStrassenKind == MmaStrassen::Type::GlobalPreSumLevel1_M6) return ref_A;

      ref_A.layout().stride() /= typename Mma::IteratorA::TensorRef::Stride(2);
      return ref_A;
    } 

    CUTLASS_HOST_DEVICE
    static typename Mma::IteratorB::TensorRef fixLayoutB(typename Mma::IteratorB::TensorRef ref_B) {
      // if (Mma::MmaStrassenKind == MmaStrassen::Type::GlobalPreSumLevel1_M2) return ref_B;
      // if (Mma::MmaStrassenKind == MmaStrassen::Type::GlobalPreSumLevel1_M3) return ref_B;

      ref_B.layout().stride() /= typename Mma::IteratorB::TensorRef::Stride(2);
      return ref_B;
    }

    CUTLASS_HOST_DEVICE
    static typename Epilogue::OutputTileIterator::TensorRef::Layout layoutforM(typename Epilogue::OutputTileIterator::TensorRef ref_D) {
      typename Epilogue::OutputTileIterator::TensorRef::Layout layout = ref_D.layout();

      layout.stride() /= typename Epilogue::OutputTileIterator::TensorRef::Stride(2);
      return layout;
    }

    CUTLASS_HOST_DEVICE
    Params(
      cutlass::gemm::GemmCoord const & problem_size,
      cutlass::gemm::GemmCoord const & grid_tiled_shape,
      typename Mma::IteratorA::TensorRef ref_A,
      typename Mma::IteratorB::TensorRef ref_B,
      typename Epilogue::OutputTileIterator::TensorRef ref_C,
      typename Epilogue::OutputTileIterator::TensorRef ref_D,
      typename OutputOp::Params output_op = typename OutputOp::Params(),
      int *workspace = nullptr,
      int const *gather_A_indices = nullptr,
      int const *gather_B_indices = nullptr,
      int const *scatter_D_indices = nullptr,
      int run = 0
    ):
      problem_size(problem_size),
      grid_tiled_shape(grid_tiled_shape),
      swizzle_log_tile(ThreadblockSwizzle().get_log_tile(grid_tiled_shape)),
      params_A(fixLayoutA(ref_A).layout()),
      ref_A(ref_A),
      params_B(fixLayoutB(ref_B).layout()),
      ref_B(ref_B),
      params_C(ref_C.layout()),
      ref_C(ref_C),
      params_D(ref_D.layout()),
      ref_D(ref_D),
      params_M(layoutforM(ref_D)),
      output_op(output_op),
      gather_A_indices(gather_A_indices),
      gather_B_indices(gather_B_indices),
      scatter_D_indices(scatter_D_indices),
      run(run) {
      
      // printf("161 %ld %ld\n", params_D.stride, params_M.stride);
      int total_gemm_k_iterations = (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK;
      total_gemm_k_iterations = total_gemm_k_iterations/2;
      int gemm_k_iterations = (total_gemm_k_iterations + grid_tiled_shape.k() - 1) / grid_tiled_shape.k();
      gemm_k_size = gemm_k_iterations * Mma::Shape::kK;
      uint presum_A_tiles = (problem_size.k() / 2) / Mma::Shape::kN; //16*1024/2 / 256 = 32 ; 
      SrcATilePerZ = (presum_A_tiles + grid_tiled_shape.k() - 1)/grid_tiled_shape.k(); //32/2 = 16
      semaphore = workspace;
    }

    template<typename OtherParams>
    CUTLASS_HOST_DEVICE
    Params(const OtherParams& other) : Params(other.problem_size, other.grid_tiled_shape,
                                              other.ref_A, other.ref_B, other.ref_C, other.ref_D,
                                              other.output_op, other.semaphore, other.gather_A_indices,
                                              other.gather_B_indices, other.scatter_D_indices, other.run) {}
  };

  /// Shared memory storage structure
  union SharedStorage {
    typename Mma::SharedStorage main_loop;
    typename Epilogue::SharedStorage epilogue;
  };

  //
  // Methods
  //

  CUTLASS_HOST_DEVICE
  StrassenPreSumGemm() { } 

  /// Determines whether kernel satisfies alignment
  CUTLASS_HOST_DEVICE
  static Status can_implement(
    cutlass::gemm::GemmCoord const & problem_size,
    typename Mma::IteratorA::TensorRef ref_A,
    typename Mma::IteratorB::TensorRef ref_B,
    typename Epilogue::OutputTileIterator::TensorRef ref_C,
    typename Epilogue::OutputTileIterator::TensorRef ref_D) {

    static int const kAlignmentA = (platform::is_same<typename Mma::IteratorA::Layout,
                                                      layout::ColumnMajorInterleaved<32>>::value)
                                   ? 32
                                   : (platform::is_same<typename Mma::IteratorA::Layout,
                                                        layout::ColumnMajorInterleaved<64>>::value)
                                     ? 64
                                     : Mma::IteratorA::AccessType::kElements;
    static int const kAlignmentB =  (platform::is_same<typename Mma::IteratorB::Layout,
                                                       layout::RowMajorInterleaved<32>>::value)
                                   ? 32
                                   : (platform::is_same<typename Mma::IteratorB::Layout,
                                                        layout::RowMajorInterleaved<64>>::value)
                                     ? 64
                                     : Mma::IteratorB::AccessType::kElements;
    static int const kAlignmentC = (platform::is_same<typename Epilogue::OutputTileIterator::Layout,
                                                      layout::ColumnMajorInterleaved<32>>::value)
                                   ? 32
                                   : (platform::is_same<typename Epilogue::OutputTileIterator::Layout,
                                                        layout::ColumnMajorInterleaved<64>>::value)
                                     ? 64
                                     : Epilogue::OutputTileIterator::kElementsPerAccess;

    if (!TensorRef_aligned(ref_A, kAlignmentA)) {
      return Status::kErrorMisalignedOperand;
    }

    if (!TensorRef_aligned(ref_B, kAlignmentB)) {
      return Status::kErrorMisalignedOperand;
    }

    if (!TensorRef_aligned(ref_C, kAlignmentC)) {
      return Status::kErrorMisalignedOperand;
    }

    if (!TensorRef_aligned(ref_D, kAlignmentC)) {
      return Status::kErrorMisalignedOperand;
    }

    return Status::kSuccess;
  }

  
  template<typename Fragment>
  CUTLASS_DEVICE
  void sum(Fragment& frag_op1, Fragment& frag_op2, Fragment& frag_out) {
    for (int i = 0; i < frag_op1.size(); i++) {
      frag_out[i] = frag_op1[i] + frag_op2[i];
    }
  }

  template<typename Fragment>
  CUTLASS_DEVICE
  void subtract(Fragment& frag_op1, Fragment& frag_op2, Fragment& frag_out) {
    for (int i = 0; i < frag_op1.size(); i++) {
      frag_out[i] = frag_op1[i] - frag_op2[i];
    }
  }

  template<typename Fragment>
  CUTLASS_DEVICE
  void computeD0(Fragment& m0, Fragment& m3, Fragment& m4, Fragment& m6, Fragment& accum00) {
    #pragma unroll
    for (int i = 0; i < m3.size(); i++) {
      accum00[i] = m0[i] + m3[i] - m4[i] + m6[i];
    }
  }

  template<typename Fragment>
  CUTLASS_DEVICE
  void computeD1(Fragment& m2, Fragment& m4, Fragment& accum01) {
    #pragma unroll
    for (int i = 0; i < m2.size(); i++) {
      accum01[i] = m2[i] + m4[i];
    }
  }

  template<typename Fragment>
  CUTLASS_DEVICE
  void computeD2(Fragment& m1, Fragment& m3, Fragment& accum10) {
    #pragma unroll
    for (int i = 0; i < m1.size(); i++) {
      accum10[i] = m1[i] + m3[i];
    }
  }

  template<typename Fragment>
  CUTLASS_DEVICE
  void computeD3(Fragment& m0, Fragment& m1, Fragment& m2, Fragment& m5, Fragment& accum11) {
    #pragma unroll
    for (int i = 0; i < m0.size(); i++) {
      accum11[i] = m0[i] - m1[i] + m2[i] + m5[i];
    }
  }

  CUTLASS_DEVICE
  void sumHalf8(half2* frag_op1, half2* frag_op2, half2* frag_out, uint N) {
    for (int i = 0; i < N; i++) {
      frag_out[i] = frag_op1[i] + frag_op2[i];
    }
  }

  
  CUTLASS_DEVICE
  void diffHalf8(half2* frag_op1, half2* frag_op2, half2* frag_out, uint N) {
    for (int i = 0; i < N; i++) {
      frag_out[i] = frag_op1[i] - frag_op2[i];
    }
  }

  /// Executes one GEMM
  CUTLASS_DEVICE
  void operator()(Params const &params, SharedStorage &shared_storage, dim3 baseBlock, dim3 origGrid,
                  uint64_t Schedule,
                  uint64_t PresumM2OrM3 = 0, uint64_t ReadWriteMi = 0, uint64_t ReadCi = 0) {
    constexpr bool IsGlobalPreSumStrassenMatrixMma = Mma::MmaStrassenKind == MmaStrassen::Type::GlobalPreSumLevel1_M0 || 
                                                    Mma::MmaStrassenKind == MmaStrassen::Type::GlobalPreSumLevel1_M1 ||
                                                    Mma::MmaStrassenKind == MmaStrassen::Type::GlobalPreSumLevel1_M2 ||
                                                    Mma::MmaStrassenKind == MmaStrassen::Type::GlobalPreSumLevel1_M3 ||
                                                    Mma::MmaStrassenKind == MmaStrassen::Type::GlobalPreSumLevel1_M4 ||
                                                    Mma::MmaStrassenKind == MmaStrassen::Type::GlobalPreSumLevel1_M5 ||
                                                    Mma::MmaStrassenKind == MmaStrassen::Type::GlobalPreSumLevel1_M6;
    static_assert(IsGlobalPreSumStrassenMatrixMma, "");
    using ElementA = typename Mma::IteratorA::TensorRef::Element;
    using ElementB = typename Mma::IteratorB::TensorRef::Element;
    using ElementC = typename Epilogue::OutputTileIterator::TensorRef::Element;

    const int halfM = params.problem_size.m()/2;
    const int halfN = params.problem_size.n()/2;
    const int halfK = params.problem_size.k()/2;
    const size_t sizeA = params.problem_size.m() * params.problem_size.k();
    const size_t sizeB = params.problem_size.k() * params.problem_size.n();
    const size_t sizeC = params.problem_size.m() * params.problem_size.n();

    ElementA* __restrict__ A_workspace = (ElementA*)params.semaphore;
    ElementB* __restrict__ B_workspace = (ElementB*)(((char*)params.semaphore) + (7 * sizeA/4 * sizeof(ElementA)));
    ElementC* __restrict__ M_workspace = (ElementC*)(B_workspace + 7 * sizeB/4);
    
    const uint num_thread_blocks = params.grid_tiled_shape.m() * params.grid_tiled_shape.n();
    volatile int* __restrict__  presum_semaphores = (int*)(M_workspace + (7* halfM * halfN * sizeof(ElementC)));
    int* __restrict__  postsum_semaphores = (int*)(presum_semaphores + (7* num_thread_blocks * sizeof(int)));
    int* __restrict__  splitk_semaphores = (int*)(postsum_semaphores + (7* num_thread_blocks* sizeof(int))) +
                                                 (Mma::MmaStrassenKind - MmaStrassen::Type::GlobalPreSumLevel1_M0)*num_thread_blocks*sizeof(int);

    // Construct the semaphore.
    ThreadblockSwizzle threadblock_swizzle;

    dim3 swizzleBaseBlock = {baseBlock.y, baseBlock.x, baseBlock.z};
    if (not std::is_same<ThreadblockSwizzle, cutlass::gemm::threadblock::GemmHorizontalThreadblockSwizzle>::value) {
      swizzleBaseBlock = {baseBlock.x, baseBlock.y, baseBlock.z};
    }

    cutlass::gemm::GemmCoord threadblock_tile_offset =
        threadblock_swizzle.get_tile_offset(params.swizzle_log_tile, swizzleBaseBlock);
    int problem_size_k = params.problem_size.k(); 
    // min(
    //   params.problem_size.k(), 
    //   (threadblock_tile_offset.k() + 1) * params.gemm_k_size);

    // Early exit if CTA is out of range
    if (params.grid_tiled_shape.m() <= threadblock_tile_offset.m() ||
      params.grid_tiled_shape.n() <= threadblock_tile_offset.n()) {
      return;
    }
    // Problem size is a function of threadblock index in the K dimension
    const int gemm_k_size = params.gemm_k_size;

    int start_k = (kSplitKSerial && params.grid_tiled_shape.k() > 1) ?
                   (threadblock_tile_offset.k() * gemm_k_size) : 0;

    // Compute initial location in logical coordinates
    cutlass::MatrixCoord tb_offset_A0{
      threadblock_tile_offset.m() * Mma::Shape::kM,
      threadblock_tile_offset.k() * gemm_k_size,
    };

    cutlass::MatrixCoord tb_offset_A3{
      threadblock_tile_offset.m() * Mma::Shape::kM + halfM,
      threadblock_tile_offset.k() * gemm_k_size + problem_size_k/2,
    };

    cutlass::MatrixCoord tb_offset_B0{
      threadblock_tile_offset.k() * gemm_k_size,
      threadblock_tile_offset.n() * Mma::Shape::kN
    };
    cutlass::MatrixCoord tb_offset_B3{
      threadblock_tile_offset.k() * gemm_k_size + problem_size_k/2,
      threadblock_tile_offset.n() * Mma::Shape::kN + halfN
    };

    
    // Compute threadblock-scoped matrix multiply-add
    int gemm_k_iterations = (gemm_k_size + Mma::Shape::kK - 1)/Mma::Shape::kK;//(gemm_k_size - tb_offset_A0.column() + Mma::Shape::kK - 1) / Mma::Shape::kK;
    // gemm_k_iterations = gemm_k_iterations / 2;
    // Compute position within threadblock
    int thread_idx = threadIdx.x;
    threadblock_tile_offset =
        threadblock_swizzle.get_tile_offset(params.swizzle_log_tile, swizzleBaseBlock);

    //assume identity swizzle
    MatrixCoord threadblock_offset(
      threadblock_tile_offset.m() * Mma::Shape::kM,
      threadblock_tile_offset.n() * Mma::Shape::kN
    );

    int block_idx = threadblock_tile_offset.m() + threadblock_tile_offset.n() * params.grid_tiled_shape.m();

    PostsumSemaphore postsum_semaphore(postsum_semaphores + block_idx, num_thread_blocks, params.run, thread_idx);

    bool IsSplitKSerial = kSplitKSerial;// && params.grid_tiled_shape.k() > 1;


    // Construct iterators to A and B operands
    cutlass::MatrixCoord tb_offset_A_base = tb_offset_A0;
    cutlass::MatrixCoord tb_offset_B_base = tb_offset_B0;
    cutlass::MatrixCoord tb_offset_D_base = threadblock_offset;
    cutlass::MatrixCoord A_extent = {params.problem_size.m(), problem_size_k};
    cutlass::MatrixCoord B_extent = {problem_size_k, params.problem_size.n()};
    cutlass::MatrixCoord srcA_extent = params.problem_size.mk();
    int threadblock_src_A_offset_n = (!IsSplitKSerial) ? threadblock_tile_offset.n() * Mma::PresumShape::kN : 0;
    if (IsSplitKSerial) {
      if (params.problem_size.n() == problem_size_k) {
        //Lets keep PresumIterations as constant
        threadblock_src_A_offset_n = threadblock_tile_offset.n() * Mma::PresumShape::kN + 
                                     threadblock_tile_offset.k() * params.problem_size.k();
      } else {
        threadblock_src_A_offset_n = (threadblock_tile_offset.k() * params.SrcATilePerZ +
                                      threadblock_tile_offset.n())* Mma::PresumShape::kN;
      }
    }
    cutlass::MatrixCoord threadblock_src_A_offset = {
      threadblock_tile_offset.m() * Mma::PresumShape::kM,
      threadblock_src_A_offset_n
    };

    ElementA* __restrict__ ptrA_ = A_workspace;
    ElementB* __restrict__ ptrB_ = B_workspace;
    ElementA* __restrict__ ptrM_ = M_workspace;

    ElementA* __restrict__ srcA = params.ref_A.data();
    ElementA* __restrict__ dstA = A_workspace;
    cutlass::MatrixCoord srcA_next_offset0 = {0,0}, srcA_next_offset1 = {0,0}, srcA_next_offset2 = {0,0}, srcA_next_offset3 = {0,0};
    cutlass::MatrixCoord dstA_next_offset0 = {0,0}, dstA_next_offset1 = {0,0}, dstA_next_offset2 = {0,0}, dstA_next_offset3 = {0,0}, dstA_next_offset4={0,0};

    ElementB* srcB = params.ref_B.data();
    cutlass::MatrixCoord srcB_next_offset = {0,0};
    ElementB* dstB = B_workspace;

    ElementA* srcM = M_workspace;
    ElementA* srcD = params.ref_D.data();
    cutlass::MatrixCoord srcD_next_offset = {0,0};
    
    cutlass::MatrixCoord srcM_offset = {0,0};
    cutlass::MatrixCoord srcM_offset2 = {0,0}; 
    cutlass::MatrixCoord srcM_offset3 = {0,0};

    switch(Mma::MmaStrassenKind) {
      case MmaStrassen::Type::GlobalPreSumLevel1_M0: {
        tb_offset_A_base = tb_offset_A0;
        tb_offset_B_base = tb_offset_B0;
        ptrA_ = params.ref_A.data();
        ptrB_ = &B_workspace[MmaStrassen::GetSchedule1(Schedule, 0) * sizeB/4];
        // A_extent = A_extent/MatrixCoord{2,2};
        B_extent = B_extent/MatrixCoord{2,2};

        srcA = ptrA_;
        //A0
        srcA_next_offset0 = {0, 0};
        //A1
        srcA_next_offset1 = {0, halfK};
        //A2
        srcA_next_offset2 = {halfM, 0};
        //A3
        srcA_next_offset3 = {halfM, halfK};

        dstA              = &A_workspace[0];
        //M2
        dstA_next_offset0 = {2*halfM, 0};
        //M3
        dstA_next_offset1 = {2*halfM, halfK};
        //M4
        dstA_next_offset2 = {4*halfM, 0};
        //M5
        dstA_next_offset3 = {5*halfM, 0};

        // if (threadIdx.x == 0 && blockIdx.x == 0) {
        //   // for (int i = 0; i < 512*512; i++) {
        //   //   if (ptrB_[i] != 2.0f) {
        //   //     printf("%d %f\n", i, (float)ptrB_[i]);
        //   //     break;
        //   //   }
        //   // }
        //   for (int i = 0; i < 512*512; i++) {
        //     if (ptrA_[i] != 2.0f) {
        //       printf("%d %f\n", i, (float)ptrA_[i]);
        //       break;
        //     }
        //   }
        // }
        break;
      }

      case MmaStrassen::Type::GlobalPreSumLevel1_M1: {
        tb_offset_A_base = tb_offset_A0 + MatrixCoord{0, halfK};
        tb_offset_B_base = tb_offset_B0;
        ptrA_ = params.ref_A.data();
        ptrB_ = &B_workspace[MmaStrassen::GetSchedule1(Schedule, 1) * sizeB/4];
        // A_extent = A_extent/MatrixCoord{2,2};
        B_extent = B_extent/MatrixCoord{2,2};

        srcA = ptrA_;
        //A0
        srcA_next_offset0 = {0, 0};
        //A1
        srcA_next_offset1 = {0, halfK};
        //A2
        srcA_next_offset2 = {halfM, 0};
        //A3
        srcA_next_offset3 = {halfM, halfK};

        dstA              = &A_workspace[0];
        //M2
        dstA_next_offset0 = {2*halfM, 0};
        //M3
        dstA_next_offset1 = {2*halfM, halfK};
        //M4
        dstA_next_offset2 = {4*halfM, 0};
        //M5
        dstA_next_offset3 = {5*halfM, 0};
        break;
      }

      case MmaStrassen::Type::GlobalPreSumLevel1_M2: {
        tb_offset_A_base = tb_offset_A0;
        tb_offset_B_base = tb_offset_B0;
        ptrA_ = &A_workspace[2*sizeA/4];
        ptrB_ = &B_workspace[MmaStrassen::GetSchedule1(Schedule, 2) * sizeB/4];
        B_extent = B_extent/MatrixCoord{1,2};
        // A_extent = A_extent/MatrixCoord{2,2};

        //M2 updates C1 and C3
        srcM_offset = {0*halfM, 0};

        break;
      }

      case MmaStrassen::Type::GlobalPreSumLevel1_M3: {
        tb_offset_A_base = tb_offset_A0 + MatrixCoord{0,halfK};
        tb_offset_B_base = tb_offset_B0 + MatrixCoord{halfK,0};
        ptrA_ = &A_workspace[2*sizeA/4];
        ptrB_ = &B_workspace[MmaStrassen::GetSchedule1(Schedule, 2) * sizeB/4];

        // A_extent = A_extent/MatrixCoord{2,2};
        B_extent = B_extent/MatrixCoord{1,2};

        break;
      }

      case MmaStrassen::Type::GlobalPreSumLevel1_M4: {
        tb_offset_A_base = tb_offset_A0;
        tb_offset_B_base = tb_offset_B0;
        ptrA_ = &A_workspace[4*sizeA/4];
        ptrB_ = &B_workspace[MmaStrassen::GetSchedule1(Schedule, 4) * sizeB/4];
        A_extent = A_extent/MatrixCoord{2,2};
        B_extent = B_extent/MatrixCoord{2,2};

        srcM_offset  = {0*halfM, 0};
        srcM_offset2 = {2*halfM, 0};
        srcM_offset3 = {3*halfM, 0};
        break;
      }

      case MmaStrassen::Type::GlobalPreSumLevel1_M5: {
        tb_offset_A_base = tb_offset_A0;
        tb_offset_B_base = tb_offset_B0;
        ptrA_ = &A_workspace[5*sizeA/4];
        ptrB_ = &B_workspace[MmaStrassen::GetSchedule1(Schedule, 5) * sizeB/4];
        A_extent = A_extent/MatrixCoord{2,2};
        B_extent = B_extent/MatrixCoord{2,2};

        // //Presum B for M2
        // srcB = params.ref_B.data() + halfN;
        // srcB_next_offset = {halfK, 0};
        // dstB = &B_workspace[2*sizeB/4];
        srcM_offset = {0*halfM, 0};
        srcM_offset2 = {2*halfM, 0};
        srcM_offset3 = {4*halfM, 0};
        break;
      }

      case MmaStrassen::Type::GlobalPreSumLevel1_M6: {
        tb_offset_A_base = tb_offset_A0 + MatrixCoord{halfM, halfK};
        tb_offset_B_base = tb_offset_B0;
        ptrA_ = params.ref_A.data();
        ptrB_ = &B_workspace[MmaStrassen::GetSchedule1(Schedule, 6) * sizeB/4];
        // A_extent = A_extent/MatrixCoord{2,2};
        B_extent = B_extent/MatrixCoord{2,2};

        //Presum B for M3
        // srcB = params.ref_B.data() + halfK * params.problem_size.k();
        // srcB_next_offset = {-halfK, 0};
        // dstB = &B_workspace[3*sizeB/4];
        srcM_offset = {0,0};
        srcM_offset2 = {2*halfM, 0};
        srcM_offset3 = {3*halfM, 0};
        break;
      }
    }


    typename Mma::IteratorA iterator_A0(
      params.params_A,
      ptrA_,
      A_extent,
      thread_idx,
      tb_offset_A_base,
      params.gather_A_indices);

    typename Mma::PresumIteratorA iterator_A_src(
      srcA, srcA_extent, 
      threadblock_src_A_offset, block_idx, srcA_next_offset0,
      threadIdx.x, srcA_next_offset1, srcA_next_offset2, srcA_next_offset3);

    typename Mma::PresumIteratorA iterator_A_dst(
      dstA, {params.problem_size.m()/2,params.problem_size.k()/2}, 
      threadblock_src_A_offset, block_idx, dstA_next_offset0,
      threadIdx.x, dstA_next_offset1, dstA_next_offset2, dstA_next_offset3, dstA_next_offset4, true);

    typename Mma::PresumIteratorA iterator_M_dst(
      ptrM_ + 4*halfM*halfN, {halfM, halfN},
      threadblock_offset, block_idx, {0,0}, threadIdx.x);

    typename Mma::IteratorB iterator_B0(
      params.params_B,
      ptrB_,
      B_extent,
      thread_idx,
      tb_offset_B_base,
      params.gather_B_indices);

    typename Mma::PresumIteratorSrcB iterator_B_src(
      srcB, params.problem_size.kn(), 
      threadblock_offset, block_idx, srcB_next_offset,
      threadIdx.x);

    typename Mma::PresumIteratorDstB iterator_B_dst(
      dstB, {params.problem_size.k()/2,params.problem_size.n()/2}, 
      threadblock_offset, block_idx, {0,0},
      threadIdx.x);

    // Tile iterator writing to destination tensor.
    typename Mma::PresumIteratorA iterator_D_src(
      srcD,
      params.problem_size.mn(),
      tb_offset_D_base, block_idx, srcD_next_offset,
      threadIdx.x
    );

    // Broadcast the warp_id computed by lane 0 to ensure dependent code
    // is compiled as warp-uniform.
    int warp_idx = canonical_warp_idx_sync();
    int lane_idx = threadIdx.x % 32;

    //
    // Main loop
    //

    // Construct thread-scoped matrix multiply
    Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx, PresumM2OrM3, ReadWriteMi);

    typename Mma::FragmentC accum;
    accum.clear();

    typename Mma::PresumIteratorSrcM mma_iterator_srcM(
      srcM,
      {halfM, halfN},
      MatrixCoord {(int)blockIdx.y - (int)baseBlock.y, (int)blockIdx.x - (int)baseBlock.x},
      block_idx,
      srcM_offset, thread_idx, srcM_offset2, srcM_offset3
    );
    using AccumFragIter = typename Epilogue::AccumulatorFragmentIterator;
    bool SplitKFirstTB = (kSplitKSerial) ? // && params.grid_tiled_shape.k() > 1
                      threadblock_tile_offset.k() == 0 :
                      true;
    bool SplitKLastTB = (kSplitKSerial) ? // && params.grid_tiled_shape.k() > 1
                          params.grid_tiled_shape.k() == threadblock_tile_offset.k() + 1 :
                          true;
    if (!kSplitKSerial || gemm_k_iterations > 0) {
      // Compute threadblock-scoped matrix multiply-add
      mma.template operator()<AccumFragIter>(gemm_k_iterations,
          accum,
          iterator_A0, iterator_A_src, iterator_A_dst, iterator_D_src,
          iterator_B0, iterator_B_src, iterator_B_dst, 
          mma_iterator_srcM, iterator_M_dst,
          SplitKLastTB, IsSplitKSerial,
          postsum_semaphore);
    }
    //
    // Epilogue
    //

    OutputOp output_op(params.output_op);

    //
    // Masked tile iterators constructed from members
    //



    MatrixCoord threadblock_offset00 = threadblock_offset;
    MatrixCoord threadblock_offset01 = threadblock_offset + MatrixCoord(0, halfN);
    MatrixCoord threadblock_offset10 = threadblock_offset + MatrixCoord(halfM, 0);
    MatrixCoord threadblock_offset11 = threadblock_offset + MatrixCoord(halfM, halfN);

    // Tile iterator loading from source tensor.
    const GemmCoord sub_problem_size = {params.problem_size.m()/2, params.problem_size.n()/2, params.problem_size.k()/2};

    typename Epilogue::OutputTileIterator iterator_C0(
      params.params_C,
      params.ref_C.data(),
      params.problem_size.mn(),
      thread_idx,
      threadblock_offset00,
      params.scatter_D_indices
    );

    // Tile iterator writing to destination tensor.
    
    typename Epilogue::OutputTileIterator iterator_D0(
      params.params_D,
      params.ref_D.data(),
      params.problem_size.mn(),
      thread_idx,
      threadblock_offset00,
      params.scatter_D_indices
    );

    typename Epilogue::OutputTileIterator iterator_srcM(
      params.params_M,
      srcM,
      {halfM, halfN},
      thread_idx,
      threadblock_offset00,
      params.scatter_D_indices
    );

    typename Epilogue::OutputTileIterator iterator_M(
      params.params_M,
      ptrM_,
      {halfM, halfN},
      thread_idx,
      threadblock_offset00,
      params.scatter_D_indices
    );

    // if (!SubGemmParallel || Mma::MmaStrassenKind == MmaStrassen::Type::Normal || 
    //                         Mma::MmaStrassenKind == MmaStrassen::Type::Compressed) {
    if (!SubGemmParallel) {
      int block_idx = threadblock_tile_offset.m() + threadblock_tile_offset.n() * params.grid_tiled_shape.m();

      // Construct the semaphore.
      Semaphore semaphore(splitk_semaphores + block_idx, thread_idx);

      // If performing a reduction via split-K, fetch the initial synchronization
      if (kSplitKSerial && params.grid_tiled_shape.k() > 1) {
        
        // Fetch the synchronization lock initially but do not block.
        semaphore.fetch();

        // Indicate which position in a serial reduction the output operator is currently updating
        // if (Mma::MmaStrassenKind == MmaStrassen::Type::Normal || 
        //     Mma::MmaStrassenKind == MmaStrassen::Type::Compressed) {
          output_op.set_k_partition(threadblock_tile_offset.k(), params.grid_tiled_shape.k());
        // }
      }

      Epilogue epilogue(
        shared_storage.epilogue, 
        thread_idx, 
        warp_idx, 
        lane_idx);
      
      // Wait on the semaphore - this latency may have been covered by iterator construction
      if (kSplitKSerial && params.grid_tiled_shape.k() > 1) {
          
        // // For subsequent threadblocks, the source matrix is held in the 'D' tensor.
        // if (threadblock_tile_offset.k()) {
        //   iterator_C = iterator_D;
        // }

        semaphore.wait(threadblock_tile_offset.k());

      }

      // Wait on the semaphore - this latency may have been covered by iterator construction
      // Execute the epilogue operator to update the destination tensor.
      // if (Mma::MmaStrassenKind == MmaStrassen::Type::Normal ||Mma::MmaStrassenKind == MmaStrassen::Type::Compressed) {
        // typename Mma::FragmentC accums;
        // accums.clear();
        // computeD0(m0, m3, m4, m6, accums);

        // epilogue(output_op, iterator_D00, accums, iterator_C00);

        // computeD1(m2, m4, accums);

        // epilogue(output_op, iterator_D01, accums, iterator_C01);

        // computeD2(m1, m3, accums);

        // epilogue(output_op, iterator_D10, accums, iterator_C10);

        // computeD3(m0, m1, m2, m5, accums);

        // epilogue(output_op, iterator_D11, accums, iterator_C11);
      // } else 
      if (IsGlobalPreSumStrassenMatrixMma) {
        const MatrixCoord sizeM = {halfM, halfN};
        dim3 actualBlockIdx = {blockIdx.x - baseBlock.x, blockIdx.y - baseBlock.y};

        for (int i = 0; i < 7; i++) {
          MmaStrassen::Type ty = (MmaStrassen::Type)(i + MmaStrassen::GlobalPreSumLevel1_M0);
          if (ty != Mma::MmaStrassenKind &&
              MmaStrassen::GetRWBit(ReadWriteMi, ty) == MmaStrassen::ReadWriteM::RWGlobalContig &&
              MmaStrassen::GetRWGlobalSync(ReadWriteMi, ty)) {
            postsum_semaphore.wait(i, 0);
          }
        }


        if (not MmaStrassen::HasRWOutputC(ReadWriteMi)) {
          epilogue.update<kThreadCount, Mma::Shape, Mma::MmaStrassenKind>(output_op, iterator_M, accum,
            iterator_M, sizeM, true, mma.sharedPostSumC, block_idx, mma, ReadWriteMi, SplitKFirstTB, SplitKLastTB, IsSplitKSerial);
        } else {
          epilogue.updateDandM<kThreadCount, Mma::Shape, Mma::MmaStrassenKind>(output_op, iterator_D0, iterator_M, accum,
            iterator_srcM, 0, mma.sharedPostSumC,
            actualBlockIdx, origGrid, block_idx, mma,
            sizeM, params.problem_size.mn(),
            ReadWriteMi, SplitKFirstTB, SplitKLastTB,IsSplitKSerial);
        }

        if (MmaStrassen::GetRWGlobalSync(ReadWriteMi, Mma::MmaStrassenKind) &&
            (MmaStrassen::GetRWBit(ReadWriteMi, Mma::MmaStrassenKind) == MmaStrassen::ReadWriteM::RWGlobalContig or
             MmaStrassen::GetRWBit(ReadWriteMi, Mma::MmaStrassenKind) == MmaStrassen::ReadWriteM::RWGlobalContigAndShared)) {
          postsum_semaphore.release(Mma::MmaStrassenKind - MmaStrassen::GlobalPreSumLevel1_M0, 0, params.run);
        }

        if (kSplitKSerial && params.grid_tiled_shape.k() > 1) { 
          int lock = 0;
          if (params.grid_tiled_shape.k() == threadblock_tile_offset.k() + 1) {

            // The final threadblock resets the semaphore for subsequent grids.
            lock = 0;
          }
          else {
            // Otherwise, the semaphore is incremented
            lock = threadblock_tile_offset.k() + 1;
          }

          semaphore.release(lock);
        }
        // __threadfence();
        // if (threadIdx.x == 0)
        //   printf("802 %d %d - %d %d ; %d\n", blockIdx.x, blockIdx.y, baseBlock.x, baseBlock.y, origGrid.x);
      } else if (false /*IsGlobalStrassenMatrixMma*/) {
#ifdef STRASSEN_MATRIX_GLOBAL_LEVEL1
        const GemmCoord sub_problem_size = {params.problem_size.m()/2, params.problem_size.n()/2, params.problem_size.k()/2};
        const uint sub_size = sub_problem_size.m() * sub_problem_size.n();

        typename Epilogue::OutputTileIterator iterator_C0(
          params.params_C,
          params.ref_C.data() + 0 * sub_size,
          sub_problem_size.mn(),
          thread_idx,
          threadblock_offset00,
          params.scatter_D_indices
        );

        // Tile iterator writing to destination tensor.
        typename Epilogue::OutputTileIterator iterator_D0(
          params.params_D,
          params.ref_D.data() + 0 * sub_size,
          sub_problem_size.mn(),
          thread_idx,
          threadblock_offset00,
          params.scatter_D_indices
        );

        if (Mma::MmaStrassenKind == MmaStrassen::Type::MatrixGlobalLevel1_M0) {
          epilogue.finalUpdate(output_op, iterator_D0, m0, iterator_C0, MatrixCoord{sub_problem_size.mn()}, params.ref_D.data());
        } else if (Mma::MmaStrassenKind == MmaStrassen::Type::MatrixGlobalLevel1_M1) {
          epilogue.update<0, 0, 1, -1, 0, 0, 0>(output_op, iterator_D0, m1, iterator_C0, MatrixCoord{sub_problem_size.mn()});
        } else if (Mma::MmaStrassenKind == MmaStrassen::Type::MatrixGlobalLevel1_M2) {
          epilogue.update<0, 1, 0, 1, 0, 0, 0>(output_op, iterator_D0, m2, iterator_C0, MatrixCoord{sub_problem_size.mn()});
        } else if (Mma::MmaStrassenKind == MmaStrassen::Type::MatrixGlobalLevel1_M3) {
          epilogue.update<1, 0, 1, 0, 0, 0, 0>(output_op, iterator_D0, m3, iterator_C0, MatrixCoord{sub_problem_size.mn()});
        } else if (Mma::MmaStrassenKind == MmaStrassen::Type::MatrixGlobalLevel1_M4) {
          epilogue.update<-1, 1, 0, 0, 0, 0, 0>(output_op, iterator_D0, m4, iterator_C0, MatrixCoord{sub_problem_size.mn()});
        } else if (Mma::MmaStrassenKind == MmaStrassen::Type::MatrixGlobalLevel1_M5) {
          epilogue.update<0, 0, 0, 1, 0, 0, 0>(output_op, iterator_D0, m5, iterator_C0, MatrixCoord{sub_problem_size.mn()});
        } else if (Mma::MmaStrassenKind == MmaStrassen::Type::MatrixGlobalLevel1_M6) {
          epilogue.update<1, 0, 0, 0, 0, 0, 0>(output_op, iterator_D0, m6, iterator_C0, MatrixCoord{sub_problem_size.mn()});
        }
#endif
      }
      }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace gemm
} // namespace cutlass

