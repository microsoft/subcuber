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

#include <type_traits>

#include "cutlass/cutlass.h"

#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/semaphore.h"
#include "cutlass/postsum_semaphore.h"
#include "cutlass/mutex.h"
#include "cutlass/arch/arch.h"

#include "cutlass/gemm/threadblock/strassen_mma_base.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace kernel {

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
  typename StrassenMiGroup_,
  typename Mma_,                  ///! Threadblock-scoped matrix multiply-accumulate 
  typename Epilogue_,             ///! Epilogue
  typename InterimEpilogue_,
  typename ThreadblockSwizzle_,   ///! Threadblock swizzling function
  bool SplitKSerial,               ///! If true, code supporting split-K via serial reduction is enabled.
  bool SubGemmParallel
>
struct StrassenGemm {
  using StrassenMiGroup = StrassenMiGroup_;
  using Mma = Mma_;
  using Epilogue = Epilogue_;
  using InterimEpilogue = InterimEpilogue_;
  using OutputOp = typename Epilogue::OutputOp;
  using InterimOutputOp = typename InterimEpilogue::OutputOp;
  using ThreadblockSwizzle = ThreadblockSwizzle_;
  using ElementA = typename Mma::IteratorA::Element;
  using ElementB = typename Mma::IteratorB::Element;
  static bool const kSplitKSerial = SplitKSerial;

  /// Warp count (concept: GemmShape)
  using WarpCount = typename Mma::WarpCount;
  static int const kThreadCount = 32 * WarpCount::kCount;

  CUTLASS_HOST_DEVICE
  static int get_presum_log_multiplier(int k, int m_or_n) {
    //TODO: We want k/2 is a divisor of Number of threads (which is usually 256 for us)
    if (k <= m_or_n) return 0;
    int multiplier = (k + m_or_n - 1)/m_or_n;
    if (multiplier > 8) {
      return 4;
    } else if (multiplier > 4) {
      return 3; //multiply with 8
    } else if (multiplier > 2) {
      return 2; //multiply with 4
    } else if (multiplier > 1) {
      return 1; //multiply with 2
    }
    return 0;
  }

  /// Parameters structure
  struct Params {
    cutlass::gemm::GemmCoord problem_size;
    cutlass::gemm::GemmCoord grid_tiled_shape;
    int swizzle_log_tile;
    int presum_a_log_tile_multiplier;
    int presum_b_log_tile_multiplier;
    typename Mma::IteratorA::Params params_A;
    typename Mma::IteratorA::TensorRef ref_A;
    typename Mma::IteratorB::Params params_B;
    typename Mma::IteratorB::TensorRef ref_B;
    typename Mma::IteratorA::TensorRef::Layout layout_MA;
    typename Mma::IteratorB::TensorRef::Layout layout_MB;
    typename Mma::IteratorA::Params params_M_A;
    typename Mma::IteratorB::Params params_M_B;
    typename Epilogue::OutputTileIterator::Params params_C;
    typename Epilogue::OutputTileIterator::TensorRef ref_C;
    typename Epilogue::OutputTileIterator::Params params_D;
    typename Epilogue::OutputTileIterator::TensorRef ref_D;
    typename Epilogue::OutputTileIterator::Params params_C2;
    typename Epilogue::OutputTileIterator::TensorRef ref_C2;
    typename Epilogue::OutputTileIterator::Params params_D2;
    typename Epilogue::OutputTileIterator::TensorRef ref_D2;
    typename OutputOp::Params output_op;
    int *semaphore;
    typename Mma::IteratorA::Element* presum_m_a_workspace;
    typename Mma::IteratorB::Element* presum_m_b_workspace;
    typename Epilogue::OutputTileIterator::TensorRef::Element* postsum_m_workspace;
    int* postsum_semaphore;
    int gemm_k_size;
    int level_1_idx;
    // For gather+scatter operations
    int const *gather_A_indices;
    int const *gather_B_indices;
    int const *scatter_D_indices;
    int run;
    //
    // Methods
    //

    CUTLASS_HOST_DEVICE
    Params(): swizzle_log_tile(0), semaphore(0), gemm_k_size(0), run(0) { }

    template<typename TensorRef>
    CUTLASS_HOST_DEVICE
    static typename TensorRef::Layout layoutforM(TensorRef ref_A_or_B) {
      typename TensorRef::Layout layout = ref_A_or_B.layout();

      layout.stride() /= typename TensorRef::Stride(2);
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
      typename Mma::IteratorA::TensorRef::Layout layout_MA,
      typename Mma::IteratorB::TensorRef::Layout layout_MB,
      int level_1_idx,
      typename Epilogue::OutputTileIterator::TensorRef ref_C2,
      typename Epilogue::OutputTileIterator::TensorRef ref_D2,
      typename OutputOp::Params output_op = typename OutputOp::Params(),
      typename Mma::IteratorA::Element* presum_m_a_workspace = nullptr,
      typename Mma::IteratorB::Element* presum_m_b_workspace = nullptr,
      typename Epilogue::OutputTileIterator::TensorRef::Element* postsum_m_workspace = nullptr,
      int *postsum_semaphore = nullptr,
      int *sem_workspace = nullptr,
      int const *gather_A_indices = nullptr,
      int const *gather_B_indices = nullptr,
      int const *scatter_D_indices = nullptr
    ):
      problem_size(problem_size),
      grid_tiled_shape(grid_tiled_shape),
      swizzle_log_tile(ThreadblockSwizzle().get_log_tile(grid_tiled_shape)),
      params_A(ref_A.layout()),
      presum_a_log_tile_multiplier(get_presum_log_multiplier(problem_size.k(), problem_size.n())),
      presum_b_log_tile_multiplier(get_presum_log_multiplier(problem_size.k(), problem_size.m())),
      ref_A(ref_A),
      params_M_A(layout_MA),
      layout_MA(layout_MA),
      params_B(ref_B.layout()),
      ref_B(ref_B),
      params_M_B(layout_MB),
      layout_MB(layout_MB),
      level_1_idx(level_1_idx),
      params_C(ref_C.layout()),
      ref_C(ref_C),
      params_D(ref_D.layout()),
      ref_D(ref_D),
      params_C2(ref_C2.layout()),
      ref_C2(ref_C2),
      params_D2(ref_D2.layout()),
      ref_D2(ref_D2),
      output_op(output_op),
      gather_A_indices(gather_A_indices),
      gather_B_indices(gather_B_indices),
      scatter_D_indices(scatter_D_indices),
      run(0),
      presum_m_a_workspace((typename Mma::IteratorA::Element*)presum_m_a_workspace),
      presum_m_b_workspace((typename Mma::IteratorB::Element*)presum_m_b_workspace),
      postsum_m_workspace((typename Epilogue::OutputTileIterator::TensorRef::Element*)postsum_m_workspace),
      postsum_semaphore(postsum_semaphore),
      semaphore(sem_workspace) {
      int total_gemm_k_iterations = (problem_size.k() + Mma::Shape::kK - 1) / Mma::Shape::kK; // 256/8 = 32 || 16 ; 1152/8=144/2=72 ;
      total_gemm_k_iterations = total_gemm_k_iterations/2;
      int gemm_k_iterations = (total_gemm_k_iterations + grid_tiled_shape.k() - 1) / grid_tiled_shape.k(); //(32 + 2)/3 = 11 || (16+2)/3 = 6 ; (64+2)/3 = 22 ; (72+2)/3=24 ; 16/3=5
      // printf("gemm_k_iterations %d total_gemm_k_iterations %d\n", gemm_k_iterations, total_gemm_k_iterations);
      gemm_k_size = gemm_k_iterations * 2 * Mma::Shape::kK;//11*8 = 88 || 5*2*8 = 80 ; 22 * 2 * 8 = 352 ; 24*2*8=384
    }

    template<typename OtherParams>
    CUTLASS_HOST_DEVICE
    Params(const OtherParams& other) :
      Params(other.problem_size, other.grid_tiled_shape,
             other.ref_A, other.ref_B, other.ref_C, other.ref_D,
             other.layout_MA, other.layout_MB,
             other.level_1_idx,
             other.ref_C2, other.ref_D2,
             other.output_op, other.presum_m_a_workspace, other.presum_m_b_workspace,
             other.postsum_m_workspace, other.postsum_semaphore,
             other.semaphore, other.gather_A_indices,
             other.gather_B_indices, other.scatter_D_indices) {
              run = other.run;
    //TODO: Set semaphore
    //TODO: fix grid_tile_shape based on StrassenMiGroup::TileMDivisor of this kernel and OtherParmams' kernel.
    // static_assert(StrassenMiGroup::numMs() == 1, "");
    }

    CUTLASS_HOST_DEVICE
    typename Mma::IteratorA::Element* get_ptr_A() const {
      return ref_A.data();
    }

    CUTLASS_HOST_DEVICE
    typename Mma::IteratorB::Element* get_ptr_B() const {
      return ref_B.data();
    }

    CUTLASS_HOST_DEVICE
    int get_problem_shape_k() const {
      return problem_size.k();
    }

    CUTLASS_HOST_DEVICE
    int get_problem_shape_m() const {
      return problem_size.m();
    }

    CUTLASS_HOST_DEVICE
    int get_problem_shape_n() const {
      return problem_size.n();
    }

    CUTLASS_HOST_DEVICE
    int get_stride_A() const {
      return ref_A.stride(0);
    }

    CUTLASS_HOST_DEVICE
    int get_stride_B() const {
      return ref_B.stride(0);
    }

    CUTLASS_HOST_DEVICE
    int get_stride_MA() const {
      return layout_MA.stride(0);
    }

    CUTLASS_HOST_DEVICE
    int get_stride_MB() const {
      return layout_MB.stride(0);
    }

    CUTLASS_HOST_DEVICE
    int get_presum_log_tile_multiplier_a() const {
      return presum_a_log_tile_multiplier;
    }
  
    CUTLASS_HOST_DEVICE
    int get_presum_log_tile_multiplier_b() const {
      return presum_b_log_tile_multiplier;
    }
  };

  /// Shared memory storage structure
  union SharedStorage {
    struct {
      typename Mma::SharedStorage main_loop;
      union {
        AlignedBuffer<typename Epilogue::ElementOutput,
                      (StrassenMiGroup::CFusedSharedBuffs() > 0) ?
                        Mma::StrassenShape::kM*Mma::StrassenShape::kN : 1>
          fused_Cs[std::max(StrassenMiGroup::CFusedSharedBuffs(), 1)];
        
        AlignedBuffer<typename Epilogue::ElementOutput,
                      (StrassenMiGroup::HasGlobalAsyncLd()) ?
                        Mma::StrassenShape::kM*Mma::StrassenShape::kN : 1>
          postsum_C;
      };
    };
    typename Epilogue::SharedStorage epilogue;
  };

  //
  // Methods
  //

  CUTLASS_HOST_DEVICE
  StrassenGemm() { } 

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
  void computeD0(Fragment& m0, Fragment& m1, Fragment& accum00) {
    #pragma unroll
    for (int i = 0; i < m0.size(); i++) {
      accum00[i] = m0[i] + m1[i];
    }
  }

  template<typename Fragment>
  CUTLASS_DEVICE
  void computeD1(Fragment& m0, Fragment& m2, Fragment& m4, Fragment& m5, Fragment& accum01) {
    #pragma unroll
    for (int i = 0; i < m2.size(); i++) {
      accum01[i] = m0[i] + m2[i] + m4[i] + m5[i];
    }
  }

  template<typename Fragment>
  CUTLASS_DEVICE
  void computeD2(Fragment& m0, Fragment& m2, Fragment& m3, Fragment& m6, Fragment& accum10) {
    #pragma unroll
    for (int i = 0; i < m0.size(); i++) {
      accum10[i] = m0[i] + m2[i] + m3[i] - m6[i];
    }
  }

  template<typename Fragment>
  CUTLASS_DEVICE
  void computeD3(Fragment& m0, Fragment& m2, Fragment& m3, Fragment& m4, Fragment& accum11) {
    #pragma unroll
    for (int i = 0; i < m0.size(); i++) {
      accum11[i] = m0[i] + m2[i] + m3[i] + m4[i];
    }
  }

  /// Executes one GEMM
  CUTLASS_DEVICE
  void operator()(Params const &params, SharedStorage &shared_storage, char* accumStorePtr = nullptr,
                  dim3 baseBlock={0,0,0}, dim3 origGrid=gridDim,
                  uint64_t Schedule=0,
                  uint64_t PresumM2OrM3 = 0, uint64_t ReadWriteMi = 0, uint64_t ReadCi = 0) {
    if (StrassenMiGroup::isEmpty()) return;

    // Compute threadblock location
    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord threadblock_tile_offset =
        threadblock_swizzle.get_tile_offset(params.swizzle_log_tile, baseBlock);

    bool IsGlobalStrassenMma = Mma::MmaStrassenKind == MmaStrassen::Type::GlobalLevel1_M0 || 
                                 Mma::MmaStrassenKind == MmaStrassen::Type::GlobalLevel1_M1 ||
                                 Mma::MmaStrassenKind == MmaStrassen::Type::GlobalLevel1_M2 ||
                                 Mma::MmaStrassenKind == MmaStrassen::Type::GlobalLevel1_M3 ||
                                 Mma::MmaStrassenKind == MmaStrassen::Type::GlobalLevel1_M4 ||
                                 Mma::MmaStrassenKind == MmaStrassen::Type::GlobalLevel1_M5 ||
                                 Mma::MmaStrassenKind == MmaStrassen::Type::GlobalLevel1_M6;
    bool IsGlobalStrassenMatrixMma = Mma::MmaStrassenKind == MmaStrassen::Type::MatrixGlobalLevel1_M0 || 
                                       Mma::MmaStrassenKind == MmaStrassen::Type::MatrixGlobalLevel1_M1 ||
                                       Mma::MmaStrassenKind == MmaStrassen::Type::MatrixGlobalLevel1_M2 ||
                                       Mma::MmaStrassenKind == MmaStrassen::Type::MatrixGlobalLevel1_M3 ||
                                       Mma::MmaStrassenKind == MmaStrassen::Type::MatrixGlobalLevel1_M4 ||
                                       Mma::MmaStrassenKind == MmaStrassen::Type::MatrixGlobalLevel1_M5 ||
                                       Mma::MmaStrassenKind == MmaStrassen::Type::MatrixGlobalLevel1_M6;
    bool IsGlobalPreSumStrassenMatrixMma = Mma::MmaStrassenKind == MmaStrassen::Type::GlobalPreSumLevel1_M0 || 
                                          Mma::MmaStrassenKind == MmaStrassen::Type::GlobalPreSumLevel1_M1 ||
                                          Mma::MmaStrassenKind == MmaStrassen::Type::GlobalPreSumLevel1_M2 ||
                                          Mma::MmaStrassenKind == MmaStrassen::Type::GlobalPreSumLevel1_M3 ||
                                          Mma::MmaStrassenKind == MmaStrassen::Type::GlobalPreSumLevel1_M4 ||
                                          Mma::MmaStrassenKind == MmaStrassen::Type::GlobalPreSumLevel1_M5 ||
                                          Mma::MmaStrassenKind == MmaStrassen::Type::GlobalPreSumLevel1_M6;

    // Early exit if CTA is out of range
    if (params.grid_tiled_shape.m() <= threadblock_tile_offset.m() ||
      params.grid_tiled_shape.n() <= threadblock_tile_offset.n()) {

      return;
    }
    // if (threadIdx.x == 0) {
    //   printf("217: A %p B %p\n", params.ref_A.data(), params.ref_B.data());
    // }
    const bool threadBlockStrassen = StrassenMiGroup::hasAllM();
    const int halfM = (threadBlockStrassen) ? Mma::StrassenShape::kM : params.problem_size.m()/2;
    const int halfN = (threadBlockStrassen) ? Mma::StrassenShape::kN : params.problem_size.n()/2;
    const int halfK = params.problem_size.k()/2; //Only used for computing Presum

    // Problem size is a function of threadblock index in the K dimension
    const int gemm_k_size = params.gemm_k_size;

    int problem_size_k = (kSplitKSerial && params.grid_tiled_shape.k() > 1) ? 
      min(params.problem_size.k(),
      (threadblock_tile_offset.k() + 1) * gemm_k_size) :
      params.problem_size.k();

    int start_k = (kSplitKSerial && params.grid_tiled_shape.k() > 1) ? (threadblock_tile_offset.k() * gemm_k_size) : 0;

    // Compute initial location in logical coordinates
    cutlass::MatrixCoord tb_offset_A0{
      threadblock_tile_offset.m() * Mma::Shape::kM/StrassenMiGroup::TileOffsetMDivisor(),
      threadblock_tile_offset.k() * gemm_k_size,
    };
    cutlass::MatrixCoord tb_offset_A1{
      threadblock_tile_offset.m() * Mma::Shape::kM/StrassenMiGroup::TileOffsetMDivisor(),
      threadblock_tile_offset.k() * gemm_k_size + (problem_size_k - start_k)/2,
    };
    cutlass::MatrixCoord tb_offset_A2{
      threadblock_tile_offset.m() * Mma::Shape::kM/StrassenMiGroup::TileOffsetMDivisor()  + halfM,
      threadblock_tile_offset.k() * gemm_k_size,
    };
    cutlass::MatrixCoord tb_offset_A3{
      threadblock_tile_offset.m() * Mma::Shape::kM/StrassenMiGroup::TileOffsetMDivisor() + halfM,
      threadblock_tile_offset.k() * gemm_k_size + (problem_size_k - start_k)/2,
    };

    cutlass::MatrixCoord tb_offset_B0{
      threadblock_tile_offset.k() * gemm_k_size,
      threadblock_tile_offset.n() * Mma::Shape::kN/StrassenMiGroup::TileOffsetNDivisor()
    };
    cutlass::MatrixCoord tb_offset_B1{
      threadblock_tile_offset.k() * gemm_k_size,
      threadblock_tile_offset.n() * Mma::Shape::kN/StrassenMiGroup::TileOffsetNDivisor() + halfN
    };
    cutlass::MatrixCoord tb_offset_B2{
      threadblock_tile_offset.k() * gemm_k_size + (problem_size_k - start_k)/2,
      threadblock_tile_offset.n() * Mma::Shape::kN/StrassenMiGroup::TileOffsetNDivisor()
    };
    cutlass::MatrixCoord tb_offset_B3{
      threadblock_tile_offset.k() * gemm_k_size + (problem_size_k - start_k)/2,
      threadblock_tile_offset.n() * Mma::Shape::kN/StrassenMiGroup::TileOffsetNDivisor() + halfN
    };

    cutlass::MatrixCoord tb_offset_M_A{
      threadblock_tile_offset.m() * Mma::Shape::kM/StrassenMiGroup::TileOffsetMDivisor(),
      threadblock_tile_offset.k() * gemm_k_size,
    };

    cutlass::MatrixCoord tb_offset_M_B{
      threadblock_tile_offset.k() * gemm_k_size,
      threadblock_tile_offset.n() * Mma::Shape::kN/StrassenMiGroup::TileOffsetNDivisor(),
    };

    // Compute threadblock-scoped matrix multiply-add
    int gemm_k_iterations = (problem_size_k - tb_offset_A0.column() + Mma::Shape::kK - 1) / Mma::Shape::kK;
    gemm_k_iterations = gemm_k_iterations / 2;
    // if (StrassenMiGroup::hasM2() && threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0)
    //   printf("halfM %d halfN %d ; %d %d\n", halfM, halfN, halfK, gemm_k_iterations);

    // if (threadIdx.x == 0) printf("gemm_k_iterations %d problem_size_k %d params.gemm_k_size %d\n", 
    //   gemm_k_iterations, problem_size_k, params.gemm_k_size);

    // Compute position within threadblock
    int thread_idx = threadIdx.x;

    // Construct iterators to A and B operands
    cutlass::MatrixCoord tb_offset_A_base = tb_offset_A0;
    cutlass::MatrixCoord tb_offset_B_base = tb_offset_B0;
    cutlass::MatrixCoord matrix_offset = {0,0};
    cutlass::MatrixCoord B_offset = {0,0};
    if (!threadBlockStrassen && Mma::MmaStrassenKind == MmaStrassen::Strassen) { 
      if (StrassenMiGroup::hasM0()) {
        tb_offset_A_base = tb_offset_A0;
        matrix_offset = {halfM, (problem_size_k - start_k)/2};
        tb_offset_B_base = tb_offset_B0;
        B_offset = {(problem_size_k - start_k)/2, halfN};
      } else if (StrassenMiGroup::hasM1()) {
        tb_offset_A_base = tb_offset_A2;
        matrix_offset = {0, (problem_size_k - start_k)/2};
        tb_offset_B_base = tb_offset_B0;
        B_offset = {0,0};
      } else if (StrassenMiGroup::hasM2()) {
        tb_offset_A_base = tb_offset_A0;
        tb_offset_B_base = tb_offset_B1;
        matrix_offset = {0, (problem_size_k - start_k)/2};
        B_offset = {(problem_size_k - start_k)/2, 0};
      } else if (StrassenMiGroup::hasM3()) {
        tb_offset_A_base = tb_offset_A0;
        matrix_offset = {0, (problem_size_k - start_k)/2};
        tb_offset_B_base = tb_offset_B2;
        B_offset = {-(problem_size_k - start_k)/2, 0};
      } else if (StrassenMiGroup::hasM4()) {
        tb_offset_A_base = tb_offset_A0;
        matrix_offset = {0, (problem_size_k - start_k)/2};
      } else if (StrassenMiGroup::hasM5()) {
        tb_offset_A_base = tb_offset_A2;
        matrix_offset = {-halfM, 0};
        tb_offset_B_base = tb_offset_B0;
        B_offset = {0, halfN};
      } else if (StrassenMiGroup::hasM6()) {
        tb_offset_A_base = tb_offset_A1;
        matrix_offset = {halfM, 0};
        tb_offset_B_base = tb_offset_B2;
        B_offset = {0, halfN};
      }
    }



    const int divide = (IsGlobalStrassenMatrixMma || IsGlobalPreSumStrassenMatrixMma) ? 2 : 1;
    typename Mma::IteratorA iterator_A0(
      params.params_A,
      params.ref_A.data(),
      {params.problem_size.m()/divide, problem_size_k/divide},
      thread_idx,
      (Mma::Base::kStages == 2 || Mma::MmaStrassenKind == MmaStrassen::Type::StrassenWinograd) ? tb_offset_A0 : tb_offset_A_base,
      (Mma::Base::kStages == 2 || Mma::MmaStrassenKind == MmaStrassen::Type::StrassenWinograd) ? cutlass::MatrixCoord{0, problem_size_k/2} : matrix_offset,
      cutlass::MatrixCoord{halfM, 0}, cutlass::MatrixCoord{halfM, problem_size_k/2},
      params.gather_A_indices);

    typename Mma::IteratorA iterator_A1(
      params.params_A,
      params.ref_A.data(),
      {params.problem_size.m(), problem_size_k},
      thread_idx,
      tb_offset_A1,
      params.gather_A_indices);

    typename Mma::IteratorA iterator_A2(
      params.params_A,
      params.ref_A.data(),
      {params.problem_size.m(), problem_size_k},
      thread_idx,
      tb_offset_A2,
      params.gather_A_indices);

    typename Mma::IteratorA iterator_A3(
      params.params_A,
      params.ref_A.data(),
      {params.problem_size.m(), problem_size_k},
      thread_idx,
      tb_offset_A3,
      params.gather_A_indices);

    typename Mma::IteratorA iterator_PresumAs[] = {
      typename Mma::IteratorA {
        params.params_M_A,
        params.presum_m_a_workspace + StrassenMiGroup::AllPresums::indexAPresum(StrassenMiGroup::APresums::A02)*halfM*halfK,
        {halfM, halfK},
        thread_idx,
        tb_offset_M_A,
        params.gather_A_indices
      },
      typename Mma::IteratorA {
        params.params_M_A,
        params.presum_m_a_workspace + StrassenMiGroup::AllPresums::indexAPresum(StrassenMiGroup::APresums::S1)*halfM*halfK,
        {halfM, halfK},
        thread_idx,
        tb_offset_M_A,
        params.gather_A_indices
      },
      typename Mma::IteratorA {
        params.params_M_A,
        params.presum_m_a_workspace + StrassenMiGroup::AllPresums::indexAPresum(StrassenMiGroup::APresums::S2)*halfM*halfK,
        {halfM, halfK},
        thread_idx,
        tb_offset_M_A,
        params.gather_A_indices
      },
      typename Mma::IteratorA {
        params.params_M_A,
        params.presum_m_a_workspace + StrassenMiGroup::AllPresums::indexAPresum(StrassenMiGroup::APresums::A1S2)*halfM*halfK,
        {halfM, halfK},
        thread_idx,
        tb_offset_M_A,
        params.gather_A_indices
      }
    };

    typename Mma::IteratorB iterator_B0(
      params.params_B,
      params.ref_B.data(),
      {problem_size_k/divide, params.problem_size.n()/divide},
      thread_idx,
      (Mma::Base::kStages == 2 || Mma::MmaStrassenKind == MmaStrassen::Type::StrassenWinograd) ? tb_offset_B0 : tb_offset_B_base,
      (Mma::Base::kStages == 2 || Mma::MmaStrassenKind == MmaStrassen::Type::StrassenWinograd) ? cutlass::MatrixCoord{0,halfN} : B_offset,
      cutlass::MatrixCoord{problem_size_k/2,0}, cutlass::MatrixCoord{problem_size_k/2,halfN},
      params.gather_B_indices);

    typename Mma::IteratorB iterator_B1(
      params.params_B,
      params.ref_B.data(),
      {problem_size_k, params.problem_size.n()},
      thread_idx,
      tb_offset_B1,
      params.gather_B_indices);
    
    typename Mma::IteratorB iterator_B2(
      params.params_B,
      params.ref_B.data(),
      {problem_size_k, params.problem_size.n()},
      thread_idx,
      tb_offset_B2,
      params.gather_B_indices);
    
    typename Mma::IteratorB iterator_B3(
      params.params_B,
      params.ref_B.data(),
      {problem_size_k, params.problem_size.n()},
      thread_idx,
      tb_offset_B3,
      params.gather_B_indices);
    
    typename Mma::IteratorB iterator_PresumBs[] = {
      typename Mma::IteratorB {
        params.params_M_B,
        params.presum_m_b_workspace + StrassenMiGroup::AllPresums::indexBPresum(StrassenMiGroup::BPresums::B31)*halfK*halfN,
        {halfK, halfN},
        thread_idx,
        tb_offset_M_B,
        params.gather_B_indices
      },
      typename Mma::IteratorB {
        params.params_M_B,
        params.presum_m_b_workspace + StrassenMiGroup::AllPresums::indexBPresum(StrassenMiGroup::BPresums::B10)*halfK*halfN,
        {halfK, halfN},
        thread_idx,
        tb_offset_M_B,
        params.gather_B_indices
      },
      typename Mma::IteratorB {
        params.params_M_B,
        params.presum_m_b_workspace + StrassenMiGroup::AllPresums::indexBPresum(StrassenMiGroup::BPresums::S3)*halfK*halfN,
        {halfK, halfN},
        thread_idx,
        tb_offset_M_B,
        params.gather_B_indices
      },
      typename Mma::IteratorB {
        params.params_M_B,
        params.presum_m_b_workspace + StrassenMiGroup::AllPresums::indexBPresum(StrassenMiGroup::BPresums::S3B2)*halfK*halfN,
        {halfK, halfN},
        thread_idx,
        tb_offset_M_B,
        params.gather_B_indices
      }
    };

    int block_idx = threadblock_tile_offset.m() + threadblock_tile_offset.n() * params.grid_tiled_shape.m();

    typename Mma::PresumGlobalIteratorA iter_PresumAi(
      params.ref_A.data(), params.ref_A.stride(0),
      {params.problem_size.m(), params.problem_size.k()},
      {threadblock_tile_offset.m() * Mma::PresumShapeA::kM, threadblock_tile_offset.n() * Mma::PresumShapeA::kN * (1 << params.presum_a_log_tile_multiplier)},
      block_idx, {0, 0}, thread_idx, {0, halfK}, {halfM, 0}, {halfM, halfK}
    );

    typename Mma::PresumGlobalIteratorB iter_PresumBi(
      params.ref_B.data(), params.ref_B.stride(0),
      {params.problem_size.k(), params.problem_size.n()},
      {threadblock_tile_offset.m() * Mma::PresumShapeB::kM * (1 << params.presum_b_log_tile_multiplier), threadblock_tile_offset.n() * Mma::PresumShapeB::kN},
      block_idx, {0, 0}, thread_idx, {0, halfN}, {halfK, 0}, {halfK, halfN}
    );

    typename Mma::PresumGlobalIteratorA iter_PresumA_M(
      params.presum_m_a_workspace, params.layout_MA.stride(0),
      {halfM, halfK},
      {threadblock_tile_offset.m() * Mma::PresumShapeA::kM, threadblock_tile_offset.n() * Mma::PresumShapeA::kN * (1 << params.presum_a_log_tile_multiplier) +
                                                           ((kSplitKSerial) ? threadblock_tile_offset.k() * halfK : 0)},
      block_idx, {0, 0}, thread_idx, {1*halfM, 0}, {2*halfM, 0}, {3*halfM, 0}
    );

    typename Mma::PresumGlobalIteratorB iter_PresumB_M(
      params.presum_m_b_workspace, params.layout_MB.stride(0),
      {halfK, halfN},
      {threadblock_tile_offset.m() * Mma::PresumShapeB::kM  * (1 << params.presum_b_log_tile_multiplier), threadblock_tile_offset.n() * Mma::PresumShapeB::kN +
                                                           ((kSplitKSerial) ? threadblock_tile_offset.k() * halfK : 0)},
      block_idx, {0, 0}, thread_idx, {1*halfK, 0}, {2*halfK, 0}, {3*halfK, 0}
    );

    typename Mma::SharedPostsumOpIterator shared_postsum_op_iterator(
      shared_storage.postsum_C.data(),
      thread_idx
    );

    using RWCTypes = typename StrassenMiGroup::RWCTypes;

    auto postsum_async = RWCTypes::PostsumSrcGlobalAsync();
    typename Mma::PostsumSrcTileIterator source_interim_iterator(
      // params.ref_D.data(),
      &params.postsum_m_workspace[halfM*halfN*(postsum_async.get_op() == -1 ? 0: postsum_async.get_op())],
      {halfM, halfN},
      thread_idx,
      //TODO: LayoutInterim has 2-D and LayoutInterim1D has 1D
      // threadblock_offset + MatrixCoord(halfM*(read_c/2), halfN*(read_c%2))
      {threadblock_tile_offset.m() * params.grid_tiled_shape.n(), threadblock_tile_offset.n()},
      MmaStrassen::MemLayout::LayoutInterim1D
    );

    source_interim_iterator.clear_mask(!postsum_async.valid());

    // Broadcast the warp_id computed by lane 0 to ensure dependent code
    // is compiled as warp-uniform.
    int warp_idx = canonical_warp_idx_sync();
    int lane_idx = threadIdx.x % 32;

    //
    // Main loop
    //

    // Construct thread-scoped matrix multiply
    Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx, shared_postsum_op_iterator);
    mma.presum_a_log_tile_multiplier = params.presum_a_log_tile_multiplier;
    mma.presum_b_log_tile_multiplier = params.presum_b_log_tile_multiplier;

    typename Mma::FragmentC accumM[7];
    for (int i = 0; i < 7; i++) accumM[i].clear();

    using RWMTypes = typename StrassenMiGroup::RWMTypes;

    {
      int myRW = RWMTypes::MyVal;
      int sign = MmaStrassen::SIGN(myRW);
      myRW = MmaStrassen::ABS(myRW);

      if (myRW > 0) {
        switch(myRW) {
          case MmaStrassen::ContinueAccums: {
            typename Mma::FragmentC& accumStore = reinterpret_cast<typename Mma::FragmentC&>(*accumStorePtr);
            accumM[StrassenMiGroup::getMi()] = accumStore;
            break;
          }
        }
      }
    }
    //Only for F16
    mma.shared_operand_presum = &params.postsum_m_workspace[halfM*halfN*1 +
                                (threadblock_tile_offset.m() * params.grid_tiled_shape.n() +
                                 threadblock_tile_offset.n())*Mma::Shape::kM*Mma::Shape::kN];
    if (!kSplitKSerial || gemm_k_iterations > 0) {
      // Compute threadblock-scoped matrix multiply-add
      mma(gemm_k_iterations, accumM,
          iterator_A0, iterator_A1, iterator_A2, iterator_A3,
          iterator_B0, iterator_B1, iterator_B2, iterator_B3,
          iterator_PresumAs, iterator_PresumBs,
          iter_PresumAi, iter_PresumBi, iter_PresumA_M, iter_PresumB_M,
          source_interim_iterator);
    }
    //
    // Epilogue
    //

    OutputOp output_op(params.output_op);
    //
    // Masked tile iterators constructed from members
    //

    threadblock_tile_offset =
        threadblock_swizzle.get_tile_offset(params.swizzle_log_tile, baseBlock);

    //assume identity swizzle
    MatrixCoord threadblock_offset(
      threadblock_tile_offset.m() * Mma::Shape::kM/StrassenMiGroup::TileOffsetMDivisor(),
      threadblock_tile_offset.n() * Mma::Shape::kN/StrassenMiGroup::TileOffsetNDivisor()
    );

    MatrixCoord threadblock_offset00 = threadblock_offset;
    MatrixCoord threadblock_offset01 = threadblock_offset + MatrixCoord(0, halfN);
    MatrixCoord threadblock_offset10 = threadblock_offset + MatrixCoord(halfM, 0);
    MatrixCoord threadblock_offset11 = threadblock_offset + MatrixCoord(halfM, halfN);

    // Tile iterator loading from source tensor.
    typename Epilogue::OutputTileIterator iterator_C00(
      params.params_C,
      params.ref_C.data(),
      params.problem_size.mn(),
      thread_idx,
      threadblock_offset00,
      params.scatter_D_indices
    );

    // Tile iterator writing to destination tensor.
    typename Epilogue::OutputTileIterator iterator_D00(
      params.params_D,
      params.ref_D.data(),
      params.problem_size.mn(),
      thread_idx,
      threadblock_offset00,
      params.scatter_D_indices
    );

    typename Epilogue::OutputTileIterator iterator_C01(
      params.params_C,
      params.ref_C.data(),
      params.problem_size.mn(),
      thread_idx,
      threadblock_offset01,
      params.scatter_D_indices
    );

    // Tile iterator writing to destination tensor.
    typename Epilogue::OutputTileIterator iterator_D01(
      params.params_D,
      params.ref_D.data(),
      params.problem_size.mn(),
      thread_idx,
      threadblock_offset01,
      params.scatter_D_indices
    );

    typename Epilogue::OutputTileIterator iterator_C10(
      params.params_C,
      params.ref_C.data(),
      params.problem_size.mn(),
      thread_idx,
      threadblock_offset10,
      params.scatter_D_indices
    );

    // if (StrassenMiGroup::hasM1() && params.level_1_idx == 1 && StrassenMiGroup::Level == 1 && threadIdx.x == 0 &&
    //     threadblock_offset.row() == 256 && threadblock_offset.column() == 0)
    //   printf("M1 : %p %f %f\n", params.ref_C.data(), ((float*)params.ref_C.data())[0], accumM[1][0]);
    // if ((StrassenMiGroup::hasM1() || StrassenMiGroup::hasM0()) && StrassenMiGroup::Level == 1 && threadIdx.x == 0 &&
    //     (params.level_1_idx == 2 || params.level_1_idx == 3 || params.level_1_idx == 6) &&
    //     threadblock_offset.row() == 0 && threadblock_offset.column() == 0)
    //   printf("M %d %ld %ld : %f\n", StrassenMiGroup::getMi(),
    //          params.ref_A.layout().stride(0), params.ref_B.layout().stride(0), accumM[StrassenMiGroup::getMi()][0]);
    // if ((StrassenMiGroup::hasM5() || StrassenMiGroup::hasM2() || StrassenMiGroup::hasM4()) && params.level_1_idx == 1 && StrassenMiGroup::Level == 1 &&
    //     threadIdx.x == 0 && threadblock_offset.row() == 256 && threadblock_offset.column() == 0)
    //   printf("817 %d : %f %p %ld\n", StrassenMiGroup::getMi(), accumM[StrassenMiGroup::getMi()][0],
    //           params.ref_D.data(), params.ref_D.layout().stride()[0]);
    // Tile iterator writing to destination tensor.
    typename Epilogue::OutputTileIterator iterator_D10(
      params.params_D,
      params.ref_D.data(),
      params.problem_size.mn(),
      thread_idx,
      threadblock_offset10,
      params.scatter_D_indices
    );

    typename Epilogue::OutputTileIterator iterator_C11(
      params.params_C,
      params.ref_C.data(),
      params.problem_size.mn(),
      thread_idx,
      threadblock_offset11,
      params.scatter_D_indices
    );

    // Tile iterator writing to destination tensor.
    typename Epilogue::OutputTileIterator iterator_D11(
      params.params_D,
      params.ref_D.data(),
      params.problem_size.mn(),
      thread_idx,
      threadblock_offset11,
      params.scatter_D_indices
    );

    typename Epilogue::OutputTileIterator iterator_Ds[4] = {iterator_D00, iterator_D01, iterator_D10, iterator_D11};
    typename Epilogue::OutputTileIterator iterator_Cs[4] = {iterator_C00, iterator_C01, iterator_C10, iterator_C11};

    if (true) {
      int block_idx = threadblock_tile_offset.m() + threadblock_tile_offset.n() * params.grid_tiled_shape.m();

      // Construct the semaphore.
      Semaphore semaphore(params.semaphore + block_idx, thread_idx);
      PostsumSemaphore postsum_semaphore(params.postsum_semaphore + block_idx,
                                         params.grid_tiled_shape.m()*params.grid_tiled_shape.n(),
                                         params.run, thread_idx);

      // If performing a reduction via split-K, fetch the initial synchronization
      if (kSplitKSerial && params.grid_tiled_shape.k() > 1) {
        
        // Fetch the synchronization lock initially but do not block.
        semaphore.fetch();

        // Indicate which position in a serial reduction the output operator is currently updating
        if (threadBlockStrassen) {
          output_op.set_k_partition(threadblock_tile_offset.k(), params.grid_tiled_shape.k());
        }
      }

      Epilogue epilogue(
        shared_storage.epilogue,
        thread_idx, 
        warp_idx, 
        lane_idx);
      
      InterimEpilogue interim_epilogue(
        shared_storage.epilogue,
        thread_idx, 
        warp_idx, 
        lane_idx
      );
      // Wait on the semaphore - this latency may have been covered by iterator construction
      if (kSplitKSerial && params.grid_tiled_shape.k() > 1) {
          
        // For subsequent threadblocks, the source matrix is held in the 'D' tensor.
        if (threadblock_tile_offset.k() && threadBlockStrassen) {
          iterator_C00 = iterator_D00;
          iterator_C01 = iterator_D01;
          iterator_C10 = iterator_D10;
          iterator_C11 = iterator_D11;
        }
        // if (threadIdx.x == 0)
        //   printf("460: Mma %d %p %d %d %d\n", Mma::MmaStrassenKind, params.semaphore, block_idx, params.grid_tiled_shape.k(), semaphore.state);
        semaphore.wait(threadblock_tile_offset.k());
        // if (threadIdx.x == 0)
        //   printf("571: Mma %d (%p + %d) = %p ; %d\n", Mma::MmaStrassenKind, params.semaphore, block_idx, semaphore.lock, semaphore.state);
      }

      {
        int myRW = RWMTypes::MyVal;
        int sign = MmaStrassen::SIGN(myRW);
        myRW = MmaStrassen::ABS(myRW);

        if (myRW > 0) {
          switch(myRW) {
            case MmaStrassen::KeepAccums:{
              typename Mma::FragmentC& accumStore = reinterpret_cast<typename Mma::FragmentC&>(*accumStorePtr);
              accumStore = accumM[StrassenMiGroup::getMi()];
              break;
            }
            case MmaStrassen::ContinueAccums:
            {
              break;
            }
          }
        }
      }

      //   printf("937 M%d at %d : %p\n", StrassenMiGroup::getMi(), params.level_1_idx, 
      //          params.ref_C.data());

      // Execute the epilogue operator to update the destination tensor.
      if (StrassenMiGroup::hasAllM()) {
        typename Mma::FragmentC accums[4];
        // accums.clear();
        computeD0(accumM[0], accumM[1], accums[0]);
        computeD1(accumM[0], accumM[2], accumM[4], accumM[5], accums[1]);
        computeD2(accumM[0], accumM[2], accumM[3], accumM[6], accums[2]);
        computeD3(accumM[0], accumM[2], accumM[3], accumM[4], accums[3]);

        epilogue(output_op, iterator_D00, accums[0], iterator_C00,
                 MmaStrassen::PostsumOp(0, 1, MmaStrassen::MemGlobal, MmaStrassen::LayoutFinal),
                 MmaStrassen::PostsumOp(), iterator_C00, iterator_D00);

        epilogue(output_op, iterator_D01, accums[1], iterator_C01,
                 MmaStrassen::PostsumOp(1, 1, MmaStrassen::MemGlobal, MmaStrassen::LayoutFinal),
                 MmaStrassen::PostsumOp(), iterator_C00, iterator_D00);

        epilogue(output_op, iterator_D10, accums[2], iterator_C10,
                 MmaStrassen::PostsumOp(2, 1, MmaStrassen::MemGlobal, MmaStrassen::LayoutFinal),
                 MmaStrassen::PostsumOp(), iterator_C00, iterator_D00);

        epilogue(output_op, iterator_D11, accums[3], iterator_C11,
                 MmaStrassen::PostsumOp(3, 1, MmaStrassen::MemGlobal, MmaStrassen::LayoutFinal),
                 MmaStrassen::PostsumOp(), iterator_C00, iterator_D00);
      } else if (Mma::MmaStrassenKind == MmaStrassen::Type::StrassenWinograd) {
        if (StrassenMiGroup::numMs() > 1) {
          //FIXME: Below code lead to spills when numMs() == 1.
          //TODO: FusedMiGroup code do not work after commit 3328f2cbdd4a7d6134c33f0addb9a43b243d75a4 
          typename Mma::FragmentC accums;
          #pragma unroll 4
          for (int c = 0; c < 4; c++) {
            const MmaStrassen::PostsumOp postsum_global_dest = RWCTypes::PostsumGlobalDest(c);
            const MmaStrassen::PostsumOp postsum_shared_dest = RWCTypes::PostsumSharedDest(c);

            if (!postsum_shared_dest.valid() && !postsum_global_dest.valid()) continue;
            accums.clear();
            auto &iterator_C = iterator_C00;

            #pragma unroll 7
            for (int mi = 0; mi < 7; mi++) {
              if (!StrassenMiGroup::hasMi(mi)) continue;
              auto sign = RWCTypes::SignForCoWithMi(c, mi);
              if (sign == 1) {
                accums = accums + accumM[mi];
              } else if (sign == -1) {
                accums = accums - accumM[mi];
              }
            }

            int read_c = 0;
            MmaStrassen::PostsumOp postsum_src;
            #pragma unroll 4
            for (read_c = 0; read_c < 4; read_c++) {
              postsum_src = RWCTypes::PostsumSrcs(c, read_c);
              if (postsum_src.valid()) {
                break;
              }
            }

            //TODO: need to fix this for fusedMi cases
            // typename InterimEpilogue::OutputTileIterator sourceInterimIterator(
            //   params.ref_D.data(),
            //   params.problem_size.mn(),
            //   thread_idx,
            //   threadblock_offset + MatrixCoord(halfM*(read_c/2), halfN*(read_c%2))
            // );

            // typename InterimEpilogue::OutputTileIterator outputInterimIterator(
            //   params.ref_D.data(),
            //   params.problem_size.mn(),
            //   thread_idx,
            //   threadblock_offset + MatrixCoord(halfM*(c/2), halfN*(c%2))
            // );

            typename InterimEpilogue::SharedTileIterator sharedInterimIterator(
              shared_storage.fused_Cs[0].data(),
              thread_idx
            );

            if (postsum_global_dest.valid() &&
                postsum_global_dest.is_layout_interim()) {
              // interim_epilogue(accums, outputInterimIterator,
              //                  sourceInterimIterator, sharedInterimIterator,
              //                  postsum_global_dest, postsum_shared_dest,
              //                  postsum_src); 
            } else {
              // if (postsum_src.valid() && postsum_src.is_layout_interim()) {
              //   //TODO: This addSource probably need to be fused with in the Epilogue's loop
              //   interim_epilogue.addSource(accums, accums,
              //                              sourceInterimIterator,
              //                              sharedInterimIterator,
              //                              postsum_src);
              // }

              iterator_C = iterator_Cs[read_c];
              auto &iterator_D = iterator_Ds[c];
              if (postsum_global_dest.valid())
                epilogue(output_op, iterator_D, accums, iterator_C,
                         postsum_global_dest, postsum_src, iterator_D, iterator_C);
            }
          }
        } else {
          if (SubGemmParallel &&
              std::is_same<typename Mma::IteratorA::Element, cutlass::half_t>::value &&
              StrassenMiGroup::hasM0() && StrassenMiGroup::FusedOrContinueMMA() == 1) {
            // typename InterimEpilogue::OutputTileIterator output_interim_iterator(
            //   &params.postsum_m_workspace[halfM*halfN*1],
            //   {halfM, halfN},
            //   thread_idx,
            //   {threadblock_tile_offset.m() * params.grid_tiled_shape.n(), threadblock_tile_offset.n()},
            //   // + MatrixCoord(halfM*(0/2), halfN*(1%2))
            //   MmaStrassen::MemLayout::LayoutInterim1D
            // );

            // typename InterimEpilogue::SharedTileIterator sharedInterimIterator(
            //   mma.shared_operand_presum,
            //   thread_idx
            // );
            
            // interim_epilogue.shared_to_global(output_interim_iterator, sharedInterimIterator);
  
            postsum_semaphore.release(1, 0, params.run);
          }

          const bool use_output_by_index = not std::is_same<typename Mma::IteratorA::Element, cutlass::half_t>::value;
          //For FP32 use output by index because that avoids high stack size.
          //But for FP16 do not use by index as the compiler generate slower code for M5.
          if (SubGemmParallel) {
            bool has_src = false;
            #pragma unroll 4
            for (int c = 0; c < 4; c++) {
              MmaStrassen::PostsumOp postsum_srcs[4] = {MmaStrassen::PostsumOp(), MmaStrassen::PostsumOp(), MmaStrassen::PostsumOp(), MmaStrassen::PostsumOp()};

              uint mi = StrassenMiGroup::getMi();
              const MmaStrassen::PostsumOp postsum_global_dest = use_output_by_index ? RWCTypes::PostsumGlobalDestByOutputIndex(c) : RWCTypes::PostsumGlobalDest(c);
              const MmaStrassen::PostsumOp postsum_shared_dest = use_output_by_index ? RWCTypes::PostsumSharedDestByOutputIndex(c) : RWCTypes::PostsumSharedDest(c);
              int c_o = postsum_global_dest.get_op();
              auto misign = RWCTypes::SignForCoWithMi(c_o, mi);

              if (!postsum_shared_dest.valid() && !postsum_global_dest.valid()) continue;

              #pragma unroll 4
              for (int read_c = 0; read_c < 4; read_c++) {
                postsum_srcs[read_c] = use_output_by_index ?
                                      RWCTypes::PostsumSrcByOutputIndex(c, read_c) :
                                      RWCTypes::PostsumSrcs(c_o, read_c);
                if (SubGemmParallel &&
                    postsum_srcs[read_c].valid() && postsum_srcs[read_c].is_mem_global() &&
                    postsum_srcs[read_c].is_layout_interim()) {
                  has_src = true;
                  postsum_semaphore.wait(postsum_srcs[read_c].get_op(), 0, false);
                  // if ((postsum_srcs[read_c].get_op() == 1 || postsum_srcs[read_c].get_op() == 0) && threadIdx.x == 0 &&
                  //   (blockIdx.x == 0 || blockIdx.x == 1) && blockIdx.y == 0)
                  //   printf("1157 %d %d : %p %d\n", StrassenMiGroup::getMi(), postsum_srcs[read_c].get_op(), postsum_semaphore.lock_ptr(postsum_srcs[read_c].get_op(), 0), postsum_semaphore.state, postsum_semaphore.expectedVal);
                }
              }
            }

            if (has_src) __syncthreads();
          }

          #pragma unroll 4
          for (int c = 0; c < 4; c++) {
            auto &iterator_C = iterator_C00;
            MmaStrassen::PostsumOp postsum_srcs[4] = {MmaStrassen::PostsumOp(), MmaStrassen::PostsumOp(), MmaStrassen::PostsumOp(), MmaStrassen::PostsumOp()};

            uint mi = StrassenMiGroup::getMi();
            const MmaStrassen::PostsumOp postsum_global_dest = use_output_by_index ? RWCTypes::PostsumGlobalDestByOutputIndex(c) : RWCTypes::PostsumGlobalDest(c);
            const MmaStrassen::PostsumOp postsum_shared_dest = use_output_by_index ? RWCTypes::PostsumSharedDestByOutputIndex(c) : RWCTypes::PostsumSharedDest(c);
            int c_o = postsum_global_dest.get_op();
            auto misign = RWCTypes::SignForCoWithMi(c_o, mi);

            if (!postsum_shared_dest.valid() && !postsum_global_dest.valid()) continue;

            #pragma unroll 4
            for (int read_c = 0; read_c < 4; read_c++) {
              postsum_srcs[read_c] = use_output_by_index ?
                                     RWCTypes::PostsumSrcByOutputIndex(c, read_c) :
                                     RWCTypes::PostsumSrcs(c_o, read_c);
            }

            typename InterimEpilogue::OutputTileIterator source_interim_iterator(
              // params.ref_D.data(),
              params.postsum_m_workspace,
              {halfM, halfN},
              thread_idx,
              //TODO: LayoutInterim has 2-D and LayoutInterim1D has 1D
              // threadblock_offset + MatrixCoord(halfM*(read_c/2), halfN*(read_c%2))
              {threadblock_tile_offset.m() * params.grid_tiled_shape.n(), threadblock_tile_offset.n()},
              MmaStrassen::MemLayout::LayoutInterim1D
            );

            typename InterimEpilogue::OutputTileIterator output_interim_iterator(
              // params.ref_D.data(),
              &params.postsum_m_workspace[halfM*halfN*c_o],
              {halfM, halfN},
              thread_idx,
              // threadblock_offset + MatrixCoord(halfM*(c/2), halfN*(c%2))
              {threadblock_tile_offset.m() * params.grid_tiled_shape.n(), threadblock_tile_offset.n()},
              // +MatrixCoord(halfM*(c/2), halfN*(c%2))
              MmaStrassen::MemLayout::LayoutInterim1D
            );

            typename InterimEpilogue::SharedTileIterator shared_interim_iterator(
              shared_storage.fused_Cs[0].data(),
              thread_idx
            );

            if ((postsum_global_dest.valid() &&
                 postsum_global_dest.is_layout_interim()) ||
                postsum_shared_dest.valid()) {

              interim_epilogue(accumM[StrassenMiGroup::getMi()],
                               output_interim_iterator, source_interim_iterator,
                               shared_interim_iterator,
                               postsum_global_dest, postsum_shared_dest,
                               postsum_srcs);
              if (SubGemmParallel && postsum_global_dest.valid()) {
                postsum_semaphore.release(postsum_global_dest.get_op(), 0, params.run);
                // if ((postsum_global_dest.get_op() == 1 || postsum_global_dest.get_op() == 0) &&
                //     threadIdx.x == 0 && (blockIdx.x == 0 || blockIdx.x == 1) && blockIdx.y == 0)
                //   printf("1224 %d %d : %p %d\n", StrassenMiGroup::getMi(), postsum_global_dest.get_op(), postsum_semaphore.lock_ptr(postsum_global_dest.get_op(), 0), params.run+1);
              }
              continue;
            }

            typename Mma::FragmentC out_accum = accumM[StrassenMiGroup::getMi()];

            if (misign == -1) {
              for (int i = 0; i < accumM[StrassenMiGroup::getMi()].size(); i++)
                out_accum[i] = -1 * accumM[StrassenMiGroup::getMi()][i];
            }

            auto &iterator_D = iterator_Ds[c_o];

            //TODO: add source for postsum_srcs array[4] makes the code a little slow for fp16 commit: 3328f2cbdd4a7d6134c33f0addb9a43b243d75a4
            interim_epilogue.add_source(out_accum, out_accum,
                                        source_interim_iterator,
                                        shared_interim_iterator,
                                        postsum_srcs);

            if (postsum_global_dest.valid()) {
              iterator_C = iterator_Cs[c_o];
              typename Epilogue::OutputTileIterator iterator_C2 = typename Epilogue::OutputTileIterator(
                params.params_C2,
                params.ref_C2.data(),
                params.problem_size.mn(),
                thread_idx,
                threadblock_offset + MatrixCoord(c_o/2 * halfM, c_o%2 * halfN),
                params.scatter_D_indices
              );

              typename Epilogue::OutputTileIterator iterator_D2 = typename Epilogue::OutputTileIterator(
                params.params_D2,
                params.ref_D2.data(),
                params.problem_size.mn(),
                thread_idx,
                threadblock_offset + MatrixCoord(c_o/2 * halfM, c_o%2 * halfN),
                params.scatter_D_indices
              );
              int read = 0;
              for (read = 0; read < 4; read++) {
                if (postsum_srcs[read].valid() && postsum_srcs[read].is_layout_final()) {
                  break;
                }
              }

              MmaStrassen::PostsumOp postsum_src = (read < 4) ? postsum_srcs[read] : MmaStrassen::PostsumOp();

              //Consider the first postsum_src because this is indexed by output index
              // if (threadIdx.x == 0 && thread)
              int is_fp16 = std::is_same<typename Mma::IteratorA::Element, cutlass::half_t>::value? 1 : 0;
              if (params.ref_C.data() != nullptr && StrassenMiGroup::Level == 1 &&
                  (StrassenMiGroup::Level1Idx == 1 || StrassenMiGroup::Level1Idx == 2 || params.level_1_idx == 1 || params.level_1_idx == 2) &&
                  //TODO: This should be if StrassenMiGroup::continueMMA
                  ((StrassenMiGroup::FusedOrContinueMMA() == 1 && StrassenMiGroup::hasM0()) || 
                   (StrassenMiGroup::FusedOrContinueMMA() == 0 && StrassenMiGroup::hasM1()) ||
                   StrassenMiGroup::hasM5() || StrassenMiGroup::hasM6() || StrassenMiGroup::hasM4())) {

                 typename InterimEpilogue::OutputTileIterator source_interim_iterator(
                    // params.ref_D.data(),
                    //TODO: Fix this M0 ptr is different for half and float
                    &params.ref_C.data()[params.problem_size.m()*params.problem_size.n()*is_fp16],
                    {halfM, halfN},
                    thread_idx,
                    //TODO: LayoutInterim has 2-D and LayoutInterim1D has 1D
                    // threadblock_offset + MatrixCoord(halfM*(read_c/2), halfN*(read_c%2))
                    {((c_o/2) * params.grid_tiled_shape.m() + threadblock_tile_offset.m()) *
                                  (params.grid_tiled_shape.n() * (1 << StrassenMiGroup::Level)),
                     (c_o%2) * params.grid_tiled_shape.n() + threadblock_tile_offset.n()},
                    MmaStrassen::MemLayout::LayoutInterim1D
                  );
                MmaStrassen::PostsumOp postsum_srcs2[4] = {MmaStrassen::PostsumOp(0, 1, MmaStrassen::MemGlobal, MmaStrassen::LayoutInterim1D),
                                                           MmaStrassen::PostsumOp(),
                                                           MmaStrassen::PostsumOp(),
                                                           MmaStrassen::PostsumOp()};

                interim_epilogue.add_source(out_accum, out_accum,
                          source_interim_iterator,
                          shared_interim_iterator,
                          postsum_srcs2);
              }

              epilogue(output_op, iterator_D, out_accum, iterator_C,
                       postsum_global_dest, postsum_src,
                       iterator_D2, iterator_C2);
            }

            // if (StrassenMiGroup::hasM0()) {
              
            // }
            // else if (StrassenMiGroup::hasM1())
            //   epilogue(output_op, iterator_D, accumM[1], iterator_C);
            // else if (StrassenMiGroup::hasM2()) {
            //   epilogue(output_op, iterator_D, accumM[2], iterator_C);
            // }
            // else if (StrassenMiGroup::hasM3()) {
            //   epilogue(output_op, iterator_D, accumM[3], iterator_C);
            // }
            // else if (StrassenMiGroup::hasM4()) {
            //   epilogue(output_op, iterator_D, accumM[4], iterator_C);
            // }
            // else if (StrassenMiGroup::hasM5()) {
            //   epilogue(output_op, iterator_D, accumM[5], iterator_C);
            // }
            // else if (StrassenMiGroup::hasM6()) {
            //   if (misign == -1) for (int i = 0; i < accumM[6].size(); i++) accumM[6][i] = -1 * accumM[6][i];
            //   epilogue(output_op, iterator_D, accumM[6], iterator_C);
            // }
          }
        }
        // else if (StrassenMiGroup::hasM0()) {
        //   epilogue(output_op, iterator_D00, m0, iterator_C00); //C0=M0
        //   epilogue(output_op, iterator_D01, m0, iterator_C01); //C1=M0
        //   #if defined(MIN_LDS_NO_PRESUM) || defined(MIN_LDS_ONE_PRESUM)
        //   //TODO: fix these using presum/postsum
        //   epilogue(output_op, iterator_D10, m0, iterator_C10); //C0=M0
        //   epilogue(output_op, iterator_D11, m0, iterator_C11); //C1=M0
        //   #endif
        // } else if (StrassenMiGroup::hasM1()) {
        //   epilogue(output_op, iterator_D00, m1, iterator_C00); //C0=M0+M1
        // } else if (StrassenMiGroup::hasM2()) {
        //   epilogue(output_op, iterator_D01, m2, iterator_C01); //C1=M0+M2
        // } else if (StrassenMiGroup::hasM3()) {
        //   epilogue(output_op, iterator_D10, m3, iterator_C01); //C2=C1+M3 = M0+M2+M3
        // } else if (StrassenMiGroup::hasM4()) {
        //   epilogue(output_op, iterator_D11, m4, iterator_C10); //C3=C2+M4 = M0+M2+M3+M4
        //   epilogue(output_op, iterator_D01, m4, iterator_C01); //C1=C1+M4 = M0+M2+M4
        // } else if (StrassenMiGroup::hasM5()) {
        //   epilogue(output_op, iterator_D01, m5, iterator_C01); //C1=C1+M5 = M0+M2+M4+M5
        // } else if (StrassenMiGroup::hasM6()) {
        //   for (int i = 0; i < m6.size(); i++) m6[i] = -1 * m6[i];
        //   epilogue(output_op, iterator_D10, m6, iterator_C10); //C2=C2-M6 = M0+M2+M3-M6
        // }
      }  else if (Mma::MmaStrassenKind == MmaStrassen::Type::Strassen) {
      #if 0 //TODO
        auto global_dest = MmaStrassen::PostsumOp(0, 1, MmaStrassen::MemGlobal, MmaStrassen::LayoutFinal);
        auto postsum_src = MmaStrassen::PostsumOp(0, 1, MmaStrassen::MemGlobal, MmaStrassen::LayoutFinal);
        if (StrassenMiGroup::hasM0()) {
          epilogue(output_op, iterator_D00, accumM[0], iterator_C00,
                   global_dest, postsum_src);
          epilogue(output_op, iterator_D11, accumM[0], iterator_C11,
                   global_dest, postsum_src);
        } else if (StrassenMiGroup::hasM1()) {
          epilogue(output_op, iterator_D10, accumM[1], iterator_C10,
                   global_dest, postsum_src);
          for (int i = 0; i < accumM[1].size(); i++) accumM[1][i] = -1 * accumM[1][i];
          epilogue(output_op, iterator_D11, accumM[1], iterator_C11,
                   global_dest, postsum_src);
        } else if (StrassenMiGroup::hasM2()) {
          epilogue(output_op, iterator_D01, accumM[2], iterator_C01,
                   global_dest, postsum_src);
          epilogue(output_op, iterator_D11, accumM[2], iterator_C11,
                   global_dest, postsum_src);
        } else if (StrassenMiGroup::hasM3()) {
          epilogue(output_op, iterator_D00, accumM[3], iterator_C00,
                   global_dest, postsum_src);
          epilogue(output_op, iterator_D10, accumM[3], iterator_C10,
                   global_dest, postsum_src);
        } else if (StrassenMiGroup::hasM4()) {
          epilogue(output_op, iterator_D01, accumM[4], iterator_C01,
                   global_dest, postsum_src);
          for (int i = 0; i < accumM[4].size(); i++) accumM[4][i] = -1 * accumM[4][i];
          epilogue(output_op, iterator_D00, accumM[4], iterator_C00,
                   global_dest, postsum_src);
        } else if (StrassenMiGroup::hasM5()) {
          epilogue(output_op, iterator_D11, accumM[5], iterator_C11,
                   global_dest, postsum_src);
        } else if (StrassenMiGroup::hasM6()) {
          epilogue(output_op, iterator_D00, accumM[6], iterator_C00,
                   global_dest, postsum_src);
        }
      #endif
      } else if (IsGlobalStrassenMatrixMma) {
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
      
//       //
//       // Release the semaphore
//       //
// /*
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
//     } else {
// /*
//       uint half_tile_M = halfM/Mma::Shape::kM;
//       uint half_tile_N = halfN/Mma::Shape::kN;

//       auto threadblock_tile_offset00 = threadblock_tile_offset;
//       auto threadblock_tile_offset01 = threadblock_tile_offset + GemmCoord(0, half_tile_N, 0);
//       auto threadblock_tile_offset10 = threadblock_tile_offset + GemmCoord(half_tile_M, 0, 0);
//       auto threadblock_tile_offset11 = threadblock_tile_offset + GemmCoord(half_tile_M, half_tile_N, 0);

//       int block_idx00 = threadblock_tile_offset00.m() + 
//                         threadblock_tile_offset00.n() * (params.grid_tiled_shape.m()*2);
//       int block_idx01 = threadblock_tile_offset01.m() +
//                         threadblock_tile_offset01.n() * (params.grid_tiled_shape.m()*2);
//       int block_idx10 = threadblock_tile_offset10.m() +
//                         threadblock_tile_offset10.n() * (params.grid_tiled_shape.m()*2);
//       int block_idx11 = threadblock_tile_offset11.m() +
//                         threadblock_tile_offset11.n() * (params.grid_tiled_shape.m()*2);

//       // Construct the semaphore.
//       Mutex mutex00(params.semaphore + block_idx00, thread_idx);
//       Mutex mutex01(params.semaphore + block_idx01, thread_idx);
//       Mutex mutex10(params.semaphore + block_idx10, thread_idx);
//       Mutex mutex11(params.semaphore + block_idx11, thread_idx);

//       Epilogue epilogue(
//         shared_storage.epilogue, 
//         thread_idx, 
//         warp_idx, 
//         lane_idx);

//       switch (Mma::MmaStrassenKind) {
//         case MmaStrassen::Type::GlobalLevel1_M0: {
//           mutex00.lock();
//           epilogue.update(output_op, iterator_D00, m0, iterator_C00);
//           mutex00.release();
//           mutex11.lock();
//           epilogue.update(output_op, iterator_D11, m0, iterator_C11);
//           mutex11.release();
  
//           break;
//         }
//         case MmaStrassen::Type::GlobalLevel1_M1: {
//           mutex10.lock();
//           epilogue.update(output_op, iterator_D10, m1, iterator_C10);
//           mutex10.release();
//           mutex11.lock();
//           for (int i = 0; i < m1.size(); i++) m1[i] = -1 * m1[i];
//           epilogue.update(output_op, iterator_D11, m1, iterator_C11);
//           mutex11.release();
  
//           break;
//         }
//         case MmaStrassen::Type::GlobalLevel1_M2: {
//           mutex01.lock();
//           epilogue.update(output_op, iterator_D01, m2, iterator_C01);
//           mutex01.release();
//           mutex11.lock();
//           epilogue.update(output_op, iterator_D11, m2, iterator_C11);
//           mutex11.release();

//           break;
//         }
//         case MmaStrassen::Type::GlobalLevel1_M3: {
//           mutex00.lock();
//           epilogue.update(output_op, iterator_D00, m3, iterator_C00);
//           mutex00.release();
//           mutex10.lock();
//           epilogue.update(output_op, iterator_D10, m3, iterator_C10);
//           mutex10.release();

//           break;
//         }
//         case MmaStrassen::Type::GlobalLevel1_M4: {
//           mutex01.lock();
//           epilogue.update(output_op, iterator_D01, m4, iterator_C01);
//           mutex01.release();
//           mutex00.lock();
//           for (int i = 0; i < m4.size(); i++) m4[i] = -1 * m4[i];
//           epilogue.update(output_op, iterator_D00, m4, iterator_C00);
//           mutex00.release();

//           break;
//         }
//         case MmaStrassen::Type::GlobalLevel1_M5: {
//           mutex11.lock();
//           epilogue.update(output_op, iterator_D11, m5, iterator_C11);
//           mutex11.release();

//           break;
//         }
//         case MmaStrassen::Type::GlobalLevel1_M6: {
//           mutex00.lock();
//           epilogue.update(output_op, iterator_D00, m6, iterator_C00);
//           mutex00.release();

//           break;
//         }
//       }*/
    }
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace kernel
} // namespace gemm
} // namespace cutlass

