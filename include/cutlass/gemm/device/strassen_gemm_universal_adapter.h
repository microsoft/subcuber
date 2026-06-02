/***************************************************************************************************
 * Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
/*!
  \file
  \brief The universal GEMM accommodates serial reductions, parallel reductions, batched strided, and
    batched array variants.
*/

#pragma once

#include <type_traits>

// common
#include "cutlass/cutlass.h"
#include "cutlass/device_kernel.h"
#include "cutlass/strassen_device_kernel.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/detail/layout.hpp"
#include "cutlass/detail/mma.hpp"
#include "cutlass/cuda_host_adapter.hpp"

#include "cutlass/kernel_launch.h"
#if !defined(__CUDACC_RTC__)
#include "cutlass/cluster_launch.hpp"
#include "cutlass/trace.h"
#endif // !defined(__CUDACC_RTC__)

// 2.x
#include "cutlass/gemm/device/gemm_universal_base.h"
#include "cutlass/gemm/kernel/gemm_transpose_operands.h"
#include "cutlass/gemm/threadblock/threadblock_swizzle.h"
#include "cutlass/epilogue/threadblock/epilogue_with_visitor_callbacks.h"

// 3.x
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/device/strassen_decls.h"
#include "cutlass/strassen_presum_global_kernel.h"

#include "cutlass/epilogue/collective/collective_strassen_builder.hpp"
#include "cutlass/gemm/collective/collective_strassen_gemm_builder.hpp"
#include "cutlass/gemm/kernel/strassen_gemm_universal.hpp"

const static bool launch_parallel_kernels = true;

using namespace MmaStrassen;

////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::device {

////////////////////////////////////////////////////////////////////////////////

/*!
  StrassenGemmUniversalAdapter is a stateful, reusable GEMM handle built around a kernel
  of type cutlass::gemm::kernel::Gemm or cutlass::gemm::kernel::GemmUniversal.

  It manages the lifetime of the underlying `kernel::Params` struct, and exposes APIs
  to create it from the host facing arguments. For power users, new static methods
  are exposed in 3.x APIs that bypass the stateful methods or args->params lowering.

  It supports kernel types that implement both the 2.x and 3.0 APIs,
  however, this is done by specializing the implementation of StrassenGemmUniversalAdapter
  on the two kernel API types, and thus, StrassenGemmUniversalAdapter's behaviour might
  differ between the two specializations.
*/
template <typename ScheduleStrassenGroups, class GemmKernel_, class Enable = void>
class StrassenGemmUniversalAdapter;

////////////////////////////////////////////////////////////////////////////////
////////////////////////////// CUTLASS 3.x API /////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

namespace detail {

// Work-around for some DispatchPolicy types not having a Stages member.
// In that case, the Stages value is 0.  Most code should static_assert
// that the number of stages is valid.

// Whether DispatchPolicy::Stages is valid.
// It should also be convertible to int, but if not, that will show up
// as a build error when StrassenGemmUniversalAdapter attempts to assign it to kStages.
template <class DispatchPolicy, class Enable = void>
struct has_Stages : cute::false_type {};

template <class DispatchPolicy>
struct has_Stages<DispatchPolicy, cute::void_t<decltype(DispatchPolicy::Stages)>> : cute::true_type {};

template<class DispatchPolicy>
constexpr int stages_member(DispatchPolicy) {
  if constexpr (has_Stages<DispatchPolicy>::value) {
    return DispatchPolicy::Stages;
  }
  else {
    return 0;
  }
}

} // namespace detail


template<typename Elem>
static __global__ void presumcheck(uint R, uint C, Elem* A, Elem* presum) {
  int row = blockIdx.x;
  int col = threadIdx.x;
  // if (threadIdx.x == 0)
  // printf("%d %d: %f + %f = %f\n", row, threadIdx.x, A[(512+row) * 1024 + threadIdx.x], A[(512+row) * 1024 + 512+threadIdx.x], presum[row*512+threadIdx.x]);
  // if (row == 512 && presum[4096*4096+row*4096+threadIdx.x] != 2.0f)
  //   printf("63: %d %d: %f; %p\n", row, threadIdx.x,
  //         float(presum[4096*4096+row*4096+threadIdx.x]),
  //         &presum[4096*4096+row*4096+threadIdx.x]);
  
  R = R/2; C=C/2;
  for (int c = 0; c < C/1024; c++) {
    col = c*blockDim.x + threadIdx.x;
    //For B, set c == 0 && row < R. For A, set row == 0 && c < C
    if (presum[2*R*C+row*C+col] != Elem(1.0f)) //Elem(col%512 + col%512))
      printf("63: %d %d: %f; %p\n", row, col,
            float(presum[2*R*C+row*C+col]),
            &presum[2*R*C+row*C+col]);
  }
}

template<typename Elem>
static __global__ void postsumcheck(Elem* postsum) {
  int row = blockIdx.x;
  int R = 8*1024/2, C=8*1024/2;
  
  for (int c = 0; c < 4; c++) {
    uint col = c*blockDim.x + threadIdx.x;
    if (row < 8*1024/2 && col < 8*1024/2 && row == 0 && c == 0)// && float(postsum[0*R*C + row*C + col]) != 0.0f)
      printf("63: %d %d: M4 %f M2 %f\n", row, col,
             float(postsum[0*R*C + row*C + col]), float(postsum[1*R*C + row*C + col]));
            // &presum[row*512+threadIdx.x]);
  }
}

template<
         uint FixedPresumTileMultilplierLogA_ = UINT32_MAX,
         uint FixedPresumTileMultilplierLogB_ = UINT32_MAX,
         uint FixedPresumTileDividerLogA_ = UINT32_MAX,
         uint FixedPresumTileDividerLogB_ = UINT32_MAX>
struct PresumOpt {
  static constexpr uint FixedPresumTileMultilplierLogA = FixedPresumTileMultilplierLogA_;
  static constexpr uint FixedPresumTileMultilplierLogB = FixedPresumTileMultilplierLogB_;
  static constexpr uint FixedPresumTileDividerLogA = FixedPresumTileDividerLogA_;
  static constexpr uint FixedPresumTileDividerLogB = FixedPresumTileDividerLogB_;
};

template<typename StrassenGroups_,
         typename ElementA, typename LayoutA, typename ElementB, typename LayoutB,
         typename ElementC, typename LayoutC, typename ElementAccum, typename TileShape,
         typename ClusterShape, typename StageCount,
         typename PresumTileShapeA = void, typename PresumTileShapeB = void,
         typename PresumOpt_ = void>
class StrassenGemmKernels {
public:
  static const int AlignmentA  = 128 / cutlass::sizeof_bits<ElementA>::value;    // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)
  static const int AlignmentB  = 128 / cutlass::sizeof_bits<ElementB>::value;    // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)
  static const int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;    // Memory access granularity/alignment of C matrix in units of elements (up to 16 bytes)
  using StrassenGroups = StrassenGroups_;
  using PresumOpt = typename std::conditional<std::is_same<PresumOpt_, void>::value, cutlass::gemm::device::PresumOpt<>, PresumOpt_>::type;

  template<typename StrassenMiGroup, typename DefaultTileShape>
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveStrassenBuilder<
    StrassenMiGroup,
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    typename std::conditional<StrassenMiGroup::hasAnyM(), typename StrassenMiGroup::ThreadBlockShape, DefaultTileShape>::type,
    ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccum, ElementAccum,
    ElementC, LayoutC, AlignmentC,
    ElementC, LayoutC, AlignmentC,
    cutlass::epilogue::TmaWarpSpecialized,
    cutlass::epilogue::fusion::LinearCombination<
      cutlass::half_t,
      float,
      cutlass::half_t,
      float
    >
  >;

  template<typename StrassenMiGroup, typename DefaultTileShape>
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveStrassenBuilder<
    StrassenMiGroup,
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccum,
    typename std::conditional<StrassenMiGroup::hasAnyM(), typename StrassenMiGroup::ThreadBlockShape, DefaultTileShape>::type,
    ClusterShape,
    typename StrassenMiGroup::StageCountType,
    // cutlass::gemm::collective::KernelScheduleAuto
    cutlass::gemm::KernelTmaWarpSpecializedPingpong,
    PresumTileShapeA,
    PresumTileShapeB,
    PresumOpt
  >;

  template<typename StrassenMiGroup, typename DefaultTileShape>
  using GemmKernel = cutlass::gemm::kernel::StrassenGemmUniversal<
    StrassenMiGroup,
    Shape<int,int,int>, // Indicates ProblemShape
    typename CollectiveMainloop<StrassenMiGroup, DefaultTileShape>::CollectiveOp,
    typename CollectiveEpilogue<StrassenMiGroup, DefaultTileShape>::CollectiveOp
  >;

  using GemmKernelM0 = GemmKernel<typename StrassenGroups::Group0, TileShape>;
  using GemmKernelM1 = GemmKernel<typename StrassenGroups::Group1, TileShape>;
  using GemmKernelM2 = GemmKernel<typename StrassenGroups::Group2, TileShape>;
  using GemmKernelM3 = GemmKernel<typename StrassenGroups::Group3, TileShape>;
  using GemmKernelM4 = GemmKernel<typename StrassenGroups::Group4, TileShape>;
  using GemmKernelM5 = GemmKernel<typename StrassenGroups::Group5, TileShape>;
  using GemmKernelM6 = GemmKernel<typename StrassenGroups::Group6, TileShape>;
};

template <typename ScheduleStrassenGroups, typename StrassenGemmKernels>
class StrassenGemmUniversalAdapter<
  ScheduleStrassenGroups,
  StrassenGemmKernels,
  cute::enable_if_t<true>>
  // cute::enable_if_t<gemm::detail::IsCutlass3GemmKernel<GetUnderlyingKernel_t<typename StrassenGemmKernels::GemmKernelM0>>::value>>
{
public:
  using GemmKernelM0 = GetUnderlyingKernel_t<typename StrassenGemmKernels::GemmKernelM0>;
  using GemmKernelM1 = GetUnderlyingKernel_t<typename StrassenGemmKernels::GemmKernelM1>;
  using GemmKernelM2 = GetUnderlyingKernel_t<typename StrassenGemmKernels::GemmKernelM2>;
  using GemmKernelM3 = GetUnderlyingKernel_t<typename StrassenGemmKernels::GemmKernelM3>;
  using GemmKernelM4 = GetUnderlyingKernel_t<typename StrassenGemmKernels::GemmKernelM4>;
  using GemmKernelM5 = GetUnderlyingKernel_t<typename StrassenGemmKernels::GemmKernelM5>;
  using GemmKernelM6 = GetUnderlyingKernel_t<typename StrassenGemmKernels::GemmKernelM6>;
  using StrassenGroups = typename StrassenGemmKernels::StrassenGroups;

  using GemmKernel = GemmKernelM0;
  using TileShape = typename GemmKernel::TileShape;
  using ElementA = typename GemmKernel::ElementA;
  using ElementB = typename GemmKernel::ElementB;
  using ElementC = typename GemmKernel::ElementC;
  using ElementD = typename GemmKernel::ElementD;
  using ElementAccumulator = typename GemmKernel::ElementAccumulator;
  using DispatchPolicy = typename GemmKernel::DispatchPolicy;
  using CollectiveMainloop = typename GemmKernel::CollectiveMainloop;
  using CollectiveEpilogue = typename GemmKernel::CollectiveEpilogue;

  // Map back to 2.x type as best as possible
  using LayoutA = gemm::detail::StrideToLayoutTagA_t<typename GemmKernel::StrideA>;
  using LayoutB = gemm::detail::StrideToLayoutTagB_t<typename GemmKernel::StrideB>;
  using LayoutC = gemm::detail::StrideToLayoutTagC_t<typename GemmKernel::StrideC>;
  using LayoutD = gemm::detail::StrideToLayoutTagC_t<typename GemmKernel::StrideD>;

  static bool const kEnableCudaHostAdapter = CUTLASS_ENABLE_CUDA_HOST_ADAPTER;

  static ComplexTransform const kTransformA = cute::is_same_v<typename GemmKernel::CollectiveMainloop::TransformA, cute::conjugate> ?
                                              ComplexTransform::kConjugate : ComplexTransform::kNone;
  static ComplexTransform const kTransformB = cute::is_same_v<typename GemmKernel::CollectiveMainloop::TransformB, cute::conjugate> ?
                                              ComplexTransform::kConjugate : ComplexTransform::kNone;

  // Legacy: Assume MultiplyAdd only since we do not use this tag type in 3.0
  using MathOperator = cutlass::arch::OpMultiplyAdd;

  using OperatorClass = cutlass::detail::get_operator_class_t<typename CollectiveMainloop::TiledMma>;

  using ArchTag = typename GemmKernel::ArchTag;

  // NOTE: Assume identity swizzle for now
  using ThreadblockSwizzle = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;

  // Assume TiledMma's ShapeMNK is the same as 2.x's ThreadblockShape
  using ThreadblockShape = cutlass::gemm::GemmShape<
      cute::size<0>(TileShape{}),
      cute::size<1>(TileShape{}),
      cute::size<2>(TileShape{})>;

  using ClusterShape = cutlass::gemm::GemmShape<
      cute::size<0>(typename GemmKernel::DispatchPolicy::ClusterShape{}),
      cute::size<1>(typename GemmKernel::DispatchPolicy::ClusterShape{}),
      cute::size<2>(typename GemmKernel::DispatchPolicy::ClusterShape{})>;

  // Instruction shape is easy too, since we get that directly from our TiledMma's atom shape
  using InstructionShape = cutlass::gemm::GemmShape<
      cute::size<0>(typename CollectiveMainloop::TiledMma::AtomShape_MNK{}),
      cute::size<1>(typename CollectiveMainloop::TiledMma::AtomShape_MNK{}),
      cute::size<2>(typename CollectiveMainloop::TiledMma::AtomShape_MNK{})>;

  // Legacy: provide a correct warp count, but no reliable warp shape
  static int const kThreadCount = GemmKernel::MaxThreadsPerBlock;

  // Warp shape is not a primary API type in 3.x
  // But we can best approximate it by inspecting the TiledMma
  // For this, we make the assumption that we always have 4 warps along M, and rest along N, none along K
  // We also always round up the warp count to 4 if the tiled mma is smaller than 128 threads
  static constexpr int WarpsInMma = cute::max(4, CUTE_STATIC_V(cute::size(typename GemmKernel::TiledMma{})) / 32);
  static constexpr int WarpsInMmaM = 4;
  static constexpr int WarpsInMmaN = cute::ceil_div(WarpsInMma, WarpsInMmaM);
  using WarpCount = cutlass::gemm::GemmShape<WarpsInMmaM, WarpsInMmaN, 1>;
  using WarpShape = cutlass::gemm::GemmShape<
      CUTE_STATIC_V(cute::tile_size<0>(typename CollectiveMainloop::TiledMma{})) / WarpsInMmaM,
      CUTE_STATIC_V(cute::tile_size<1>(typename CollectiveMainloop::TiledMma{})) / WarpsInMmaN,
      CUTE_STATIC_V(cute::tile_size<2>(typename CollectiveMainloop::TiledMma{}))>;

  static int constexpr kStages = detail::stages_member(typename CollectiveMainloop::DispatchPolicy{});

  // Inspect TiledCopy for A and B to compute the alignment size
  static int constexpr kAlignmentA = cutlass::detail::get_alignment_count_from_gmem_tiled_copy<
      typename CollectiveMainloop::GmemTiledCopyA, ElementA, typename CollectiveMainloop::TiledMma::ValTypeA>();
  static int constexpr kAlignmentB = cutlass::detail::get_alignment_count_from_gmem_tiled_copy<
      typename CollectiveMainloop::GmemTiledCopyB, ElementB, typename CollectiveMainloop::TiledMma::ValTypeB>();
  static int constexpr kAlignmentC = cutlass::detail::get_alignment_count_from_gmem_tiled_copy<
      typename CollectiveEpilogue::GmemTiledCopyC, ElementC>();
  static int constexpr kAlignmentD = cutlass::detail::get_alignment_count_from_gmem_tiled_copy<
      typename CollectiveEpilogue::GmemTiledCopyD, ElementD>();

  using EpilogueOutputOp = typename CollectiveEpilogue::ThreadEpilogueOp;

  // Split-K preserves splits that are 128b aligned
  static int constexpr kSplitKAlignment = cute::max(
      128 / sizeof_bits<ElementA>::value, 128 / sizeof_bits<ElementB>::value);

  /// Argument structure: User API
  using Arguments = typename GemmKernel::Arguments;
  /// Argument structure: Kernel API
  using Params = typename GemmKernel::Params;

private:

  /// Kernel API parameters object
  typename GemmKernelM0::Params paramsM0_;
  typename GemmKernelM1::Params paramsM1_;
  typename GemmKernelM2::Params paramsM2_;
  typename GemmKernelM3::Params paramsM3_;
  typename GemmKernelM4::Params paramsM4_;
  typename GemmKernelM5::Params paramsM5_;
  typename GemmKernelM6::Params paramsM6_;

public:

  // /// Access the Params structure
  // Params const& params() const {
  //   return params_;
  // }

  /// Determines whether the GEMM can execute the given problem.
  static Status
  can_implement(Arguments const& args) {
    if (GemmKernel::can_implement(args)) {
      if (!GemmKernelM0::StrassenMiGroup::hasAllM()) {
        //Multiple kernels
        if (!GemmKernelM0::StrassenMiGroup::hasM0() ||
            !(GemmKernelM0::StrassenMiGroup::hasM0() && GemmKernelM0::StrassenMiGroup::hasM1()))
          return Status::kErrorInvalidProblem;
        
        bool has_A_presums = GemmKernelM0::StrassenMiGroup::AllPresums::computeAnyAPresum(MmaStrassen::PresumCompute);
        bool has_B_presums = GemmKernelM0::StrassenMiGroup::AllPresums::computeAnyBPresum(MmaStrassen::PresumCompute);
        using PresumOpt = typename GemmKernelM0::Mma::PresumOpt;
        const int presum_a_log_tile_multiplier = PresumOpt::FixedPresumTileMultilplierLogA != UINT32_MAX ?
                                                    PresumOpt::FixedPresumTileMultilplierLogA :
                                                    GemmKernelM0::Mma::get_presum_log_multiplier(args.get_problem_shape_k(), args.get_problem_shape_n());
        const int presum_b_log_tile_multiplier = PresumOpt::FixedPresumTileMultilplierLogB != UINT32_MAX ?
                                                    PresumOpt::FixedPresumTileMultilplierLogB :
                                                    GemmKernelM0::Mma::get_presum_log_multiplier(args.get_problem_shape_k(), args.get_problem_shape_m());
        const int total_presum_iterations = std::max(GemmKernelM0::Mma::kPresumComputeIterationsA*has_A_presums*(1<<presum_a_log_tile_multiplier),
                                                     GemmKernelM0::Mma::kPresumComputeIterationsB*has_B_presums*(1<<presum_b_log_tile_multiplier));
        int required_k = total_presum_iterations * size<2>(typename GemmKernelM0::Mma::TileShape{});
        if (args.get_problem_shape_k()/2 < required_k)
          return Status::kErrorInvalidProblem;
      }
      return Status::kSuccess;
    }
    else {
      return Status::kInvalid;
    }
  }

  static size_t get_presum_a_workspace_size(Arguments const &args) {
    return (StrassenGroups::Group0::AllPresums::numAPresumYes()) *
            sizeof(ElementA) * args.get_problem_shape_m()/2 * args.get_problem_shape_k()/2;
  }

  static size_t get_presum_b_workspace_size(Arguments const &args) {
    return (StrassenGroups::Group0::AllPresums::numBPresumYes()) *
            sizeof(ElementB) * args.get_problem_shape_n()/2 * args.get_problem_shape_k()/2;
  }

  static size_t get_postsum_m_workspace_size(Arguments const &args) {
    return 7*(args.get_problem_shape_m()/2 * args.get_problem_shape_n()/2) * sizeof(ElementB);
  }

  /// Gets the workspace size
  static size_t
  get_workspace_size(Arguments const& args) {
    size_t workspace_bytes = 0;
    if (args.mode == GemmUniversalMode::kGemmSplitKParallel) {
      workspace_bytes += sizeof(int) * size_t(cute::size<0>(TileShape{})) * size_t(cute::size<1>(TileShape{}));
    }

    workspace_bytes += get_presum_a_workspace_size(args) + get_presum_b_workspace_size(args) +
                       get_postsum_m_workspace_size(args) + GemmKernel::get_workspace_size(args);

    CUTLASS_TRACE_HOST("  workspace_bytes: " << workspace_bytes);

    return workspace_bytes;
  }

  /// Computes the grid shape
  static dim3
  get_grid_shape(Arguments const& args, void* workspace = nullptr) {
    auto tmp_params = GemmKernel::to_underlying_arguments(args, workspace);
    return GemmKernel::get_grid_shape(tmp_params);
  }

  // Computes the grid shape
  template<typename GemmKernel>
  static dim3
  get_grid_shape(typename GemmKernel::Params const& params) {
    return GemmKernel::get_grid_shape(params);
  }

  /// Computes the maximum number of active blocks per multiprocessor
  static int maximum_active_blocks(int /* smem_capacity */ = -1) {
    CUTLASS_TRACE_HOST("GemmUniversal::maximum_active_blocks()");
    int max_active_blocks = -1;
    int smem_size = GemmKernel::SharedStorageSize;

    // first, account for dynamic smem capacity if needed
    cudaError_t result;
    if (smem_size >= (48 << 10)) {
      CUTLASS_TRACE_HOST("  Setting smem size to " << smem_size);
      result = cudaFuncSetAttribute(
          device_kernel<GemmKernel>,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          smem_size);
      if (cudaSuccess != result) {
        result = cudaGetLastError(); // to clear the error bit
        CUTLASS_TRACE_HOST(
          "  cudaFuncSetAttribute() returned error: "
          << cudaGetErrorString(result));
        return -1;
      }
    }

    // query occupancy after setting smem size
    result = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_active_blocks,
        device_kernel<GemmKernel>,
        GemmKernel::MaxThreadsPerBlock,
        smem_size);

    if (cudaSuccess != result) {
      result = cudaGetLastError(); // to clear the error bit
      CUTLASS_TRACE_HOST(
        "  cudaOccupancyMaxActiveBlocksPerMultiprocessor() returned error: "
        << cudaGetErrorString(result));
      return -1;
    }

    CUTLASS_TRACE_HOST("  max_active_blocks: " << max_active_blocks);
    return max_active_blocks;
  }

  template<typename GemmKernel>
  Status
  initialize(
    typename GemmKernel::Arguments const args,
    typename GemmKernel::Params& params,
    ElementA* presum_m_a, ElementB* presum_m_b, ElementC* postsum_m,
    void* sem_workspace,
    cudaStream_t stream = nullptr,
    CudaHostAdapter* cuda_adapter = nullptr) {

    CUTLASS_TRACE_HOST("GemmUniversal::initialize() - workspace "
      << sem_workspace << ", stream: " << (stream ? "non-null" : "null"));

    // Initialize the workspace
    Status status = GemmKernel::initialize_workspace(args, sem_workspace, stream, cuda_adapter);
    if (status != Status::kSuccess) {
      return status;
    }
    // Initialize the Params structure
    params = GemmKernel::to_underlying_arguments(args, presum_m_a, presum_m_b, postsum_m, sem_workspace);
    // Don't set the function attributes - require the CudaHostAdapter to set it.
    if constexpr (kEnableCudaHostAdapter) {
      CUTLASS_ASSERT(cuda_adapter);
      return Status::kSuccess;
    }
    else {
      //
      // Account for dynamic smem capacity if needed
      //
      int smem_size = GemmKernel::SharedStorageSize;
      printf("493 %d = + %ld \n", smem_size, sizeof(typename GemmKernel::SharedStorage::TensorStorage::ExtraStorage2));
      CUTLASS_ASSERT(cuda_adapter == nullptr);

      if (smem_size >= (48 << 10)) {
        CUTLASS_TRACE_HOST("  Setting smem size to " << smem_size);
        cudaError_t result = cudaFuncSetAttribute(
            device_kernel<GemmKernel>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size);
        if (cudaSuccess != result) {
          result = cudaGetLastError(); // to clear the error bit
          CUTLASS_TRACE_HOST("  cudaFuncSetAttribute() returned error: " << cudaGetErrorString(result));
          return Status::kErrorInternal;
        }
      }
    }
    return Status::kSuccess;
  }

  /// Initializes GEMM state from arguments.
  Status
  initialize(
    Arguments const& args,
    int swizzles[7],
    void* workspace = nullptr,
    cudaStream_t stream = nullptr,
    CudaHostAdapter* cuda_adapter = nullptr) {

    CUTLASS_TRACE_HOST("GemmUniversal::initialize() - workspace "
      << workspace << ", stream: " << (stream ? "non-null" : "null"));

    using ParallelGroup0 = ParallelMiKernels<typename ScheduleStrassenGroups::ParallelGroups0,
                                              GemmKernelM0, GemmKernelM1, GemmKernelM2, GemmKernelM3,
                                              GemmKernelM4, GemmKernelM5, GemmKernelM6>;
    using ParallelGroup1 = ParallelMiKernels<typename ScheduleStrassenGroups::ParallelGroups1,
                                              GemmKernelM0, GemmKernelM1, GemmKernelM2, GemmKernelM3,
                                              GemmKernelM4, GemmKernelM5, GemmKernelM6>;
    using ParallelGroup2 = ParallelMiKernels<typename ScheduleStrassenGroups::ParallelGroups2,
                                              GemmKernelM0, GemmKernelM1, GemmKernelM2, GemmKernelM3,
                                              GemmKernelM4, GemmKernelM5, GemmKernelM6>;
    using ParallelGroup3 = ParallelMiKernels<typename ScheduleStrassenGroups::ParallelGroups3,
                                              GemmKernelM0, GemmKernelM1, GemmKernelM2, GemmKernelM3,
                                              GemmKernelM4, GemmKernelM5, GemmKernelM6>;
    using ParallelGroup4 = ParallelMiKernels<typename ScheduleStrassenGroups::ParallelGroups4,
                                              GemmKernelM0, GemmKernelM1, GemmKernelM2, GemmKernelM3,
                                              GemmKernelM4, GemmKernelM5, GemmKernelM6>;
    using ParallelGroup5 = ParallelMiKernels<typename ScheduleStrassenGroups::ParallelGroups5,
                                              GemmKernelM0, GemmKernelM1, GemmKernelM2, GemmKernelM3,
                                              GemmKernelM4, GemmKernelM5, GemmKernelM6>;
    using ParallelGroup6 = ParallelMiKernels<typename ScheduleStrassenGroups::ParallelGroups6,
                                              GemmKernelM0, GemmKernelM1, GemmKernelM2, GemmKernelM3,
                                              GemmKernelM4, GemmKernelM5, GemmKernelM6>;

    ElementA* presum_a_workspace = (ElementA*)workspace;
    ElementA* presum_b_workspace = (ElementA*)presum_a_workspace + get_presum_a_workspace_size(args)/sizeof(ElementA);
    ElementC* postsum_m_workspace = (ElementC*)presum_b_workspace + get_presum_b_workspace_size(args)/sizeof(ElementB);
    int* sem_workspace = (int*)(postsum_m_workspace + get_postsum_m_workspace_size(args)/sizeof(ElementC));
    int swizzle_idx = 0;
    auto args0 = args, args1 = args, args2 = args, args3 = args, args4 = args, args5 = args, args6 = args;
    args0.scheduler.max_swizzle_size = swizzles[swizzle_idx++];
    args1.scheduler.max_swizzle_size = swizzles[swizzle_idx++];
    args2.scheduler.max_swizzle_size = swizzles[swizzle_idx++];
    args3.scheduler.max_swizzle_size = swizzles[swizzle_idx++];
    args4.scheduler.max_swizzle_size = swizzles[swizzle_idx++];
    args5.scheduler.max_swizzle_size = swizzles[swizzle_idx++];
    args6.scheduler.max_swizzle_size = swizzles[swizzle_idx++];

    Status err;

    // err = initialize<GemmKernelM0>(typename GemmKernelM0::Arguments(args0), paramsM0_, presum_a_workspace, presum_b_workspace,
    //                                 postsum_m_workspace, sem_workspace, stream, cuda_adapter);
    // if (err == Status::kErrorInternal) return err;

    if (ParallelGroup0::HasAKernel()) {
      err = initialize_parallel_kernels<ParallelGroup0>(
                                        typename GemmKernelM0::Arguments(args0), paramsM0_,
                                        typename GemmKernelM0::Arguments(args1), paramsM1_,
                                        typename GemmKernelM0::Arguments(args2), paramsM2_,
                                        typename GemmKernelM0::Arguments(args3), paramsM3_,
                                        typename GemmKernelM0::Arguments(args4), paramsM4_,
                                        typename GemmKernelM0::Arguments(args5), paramsM5_,
                                        typename GemmKernelM0::Arguments(args6), paramsM6_,
                                        presum_a_workspace, presum_b_workspace,
                                        postsum_m_workspace, sem_workspace, stream, cuda_adapter);
      if (err == Status::kErrorInternal) return err;
    }

    if (ParallelGroup1::HasAKernel()) {
      err = initialize_parallel_kernels<ParallelGroup1>(
                                        typename GemmKernelM0::Arguments(args0), paramsM0_,
                                        typename GemmKernelM0::Arguments(args1), paramsM1_,
                                        typename GemmKernelM0::Arguments(args2), paramsM2_,
                                        typename GemmKernelM0::Arguments(args3), paramsM3_,
                                        typename GemmKernelM0::Arguments(args4), paramsM4_,
                                        typename GemmKernelM0::Arguments(args5), paramsM5_,
                                        typename GemmKernelM0::Arguments(args6), paramsM6_,
                                        presum_a_workspace, presum_b_workspace,
                                        postsum_m_workspace, sem_workspace, stream, cuda_adapter);
      if (err == Status::kErrorInternal) return err;
    }

    if (ParallelGroup2::HasAKernel()) {
      err = initialize_parallel_kernels<ParallelGroup2>(
                                        typename GemmKernelM0::Arguments(args0), paramsM0_,
                                        typename GemmKernelM0::Arguments(args1), paramsM1_,
                                        typename GemmKernelM0::Arguments(args2), paramsM2_,
                                        typename GemmKernelM0::Arguments(args3), paramsM3_,
                                        typename GemmKernelM0::Arguments(args4), paramsM4_,
                                        typename GemmKernelM0::Arguments(args5), paramsM5_,
                                        typename GemmKernelM0::Arguments(args6), paramsM6_,
                                        presum_a_workspace, presum_b_workspace,
                                        postsum_m_workspace, sem_workspace, stream, cuda_adapter);
      if (err == Status::kErrorInternal) return err;
    }

    if (ParallelGroup3::HasAKernel()) {
      err = initialize_parallel_kernels<ParallelGroup3>(
                                        typename GemmKernelM0::Arguments(args0), paramsM0_,
                                        typename GemmKernelM0::Arguments(args1), paramsM1_,
                                        typename GemmKernelM0::Arguments(args2), paramsM2_,
                                        typename GemmKernelM0::Arguments(args3), paramsM3_,
                                        typename GemmKernelM0::Arguments(args4), paramsM4_,
                                        typename GemmKernelM0::Arguments(args5), paramsM5_,
                                        typename GemmKernelM0::Arguments(args6), paramsM6_,
                                        presum_a_workspace, presum_b_workspace,
                                        postsum_m_workspace, sem_workspace, stream, cuda_adapter);
      if (err == Status::kErrorInternal) return err;
    }

    if (ParallelGroup4::HasAKernel()) {
      err = initialize_parallel_kernels<ParallelGroup4>(
                                        typename GemmKernelM0::Arguments(args0), paramsM0_,
                                        typename GemmKernelM0::Arguments(args1), paramsM1_,
                                        typename GemmKernelM0::Arguments(args2), paramsM2_,
                                        typename GemmKernelM0::Arguments(args3), paramsM3_,
                                        typename GemmKernelM0::Arguments(args4), paramsM4_,
                                        typename GemmKernelM0::Arguments(args5), paramsM5_,
                                        typename GemmKernelM0::Arguments(args6), paramsM6_,
                                        presum_a_workspace, presum_b_workspace,
                                        postsum_m_workspace, sem_workspace, stream, cuda_adapter);
      if (err == Status::kErrorInternal) return err;
    }

    if (ParallelGroup5::HasAKernel()) {
      err = initialize_parallel_kernels<ParallelGroup5>(
                                        typename GemmKernelM0::Arguments(args0), paramsM0_,
                                        typename GemmKernelM0::Arguments(args1), paramsM1_,
                                        typename GemmKernelM0::Arguments(args2), paramsM2_,
                                        typename GemmKernelM0::Arguments(args3), paramsM3_,
                                        typename GemmKernelM0::Arguments(args4), paramsM4_,
                                        typename GemmKernelM0::Arguments(args5), paramsM5_,
                                        typename GemmKernelM0::Arguments(args6), paramsM6_,
                                        presum_a_workspace, presum_b_workspace,
                                        postsum_m_workspace, sem_workspace, stream, cuda_adapter);
      if (err == Status::kErrorInternal) return err;
    }

    if (ParallelGroup6::HasAKernel()) {
      err = initialize_parallel_kernels<ParallelGroup6>(
                                        typename GemmKernelM0::Arguments(args0), paramsM0_,
                                        typename GemmKernelM0::Arguments(args1), paramsM1_,
                                        typename GemmKernelM0::Arguments(args2), paramsM2_,
                                        typename GemmKernelM0::Arguments(args3), paramsM3_,
                                        typename GemmKernelM0::Arguments(args4), paramsM4_,
                                        typename GemmKernelM0::Arguments(args5), paramsM5_,
                                        typename GemmKernelM0::Arguments(args6), paramsM6_,
                                        presum_a_workspace, presum_b_workspace,
                                        postsum_m_workspace, sem_workspace, stream, cuda_adapter);
      if (err == Status::kErrorInternal) return err;
    }

    return Status::kSuccess;
  }

   template<typename ParallelGroup>
   Status
   initialize_parallel_kernels(
    typename GemmKernelM0::Arguments const args0,
    typename GemmKernelM0::Params& params0,
    typename GemmKernelM1::Arguments const args1,
    typename GemmKernelM1::Params& params1,
    typename GemmKernelM2::Arguments const args2,
    typename GemmKernelM2::Params& params2,
    typename GemmKernelM3::Arguments const args3,
    typename GemmKernelM3::Params& params3,
    typename GemmKernelM4::Arguments const args4,
    typename GemmKernelM4::Params& params4,
    typename GemmKernelM5::Arguments const args5,
    typename GemmKernelM5::Params& params5,
    typename GemmKernelM6::Arguments const args6,
    typename GemmKernelM6::Params& params6,
    ElementA* presum_m_a, ElementB* presum_m_b, ElementC* postsum_m,
    void* sem_workspace,
    cudaStream_t stream = nullptr,
    CudaHostAdapter* cuda_adapter = nullptr) {

    CUTLASS_TRACE_HOST("GemmUniversal::initialize() - workspace "
      << sem_workspace << ", stream: " << (stream ? "non-null" : "null"));

    // Initialize the workspace
    Status status = GemmKernelM0::initialize_workspace(args0, sem_workspace, stream, cuda_adapter);
    if (status != Status::kSuccess) {
      return status;
    }
    status = GemmKernelM1::initialize_workspace(args1, sem_workspace, stream, cuda_adapter);
    if (status != Status::kSuccess) {
      return status;
    }
    status = GemmKernelM2::initialize_workspace(args2, sem_workspace, stream, cuda_adapter);
    if (status != Status::kSuccess) {
      return status;
    }
    status = GemmKernelM3::initialize_workspace(args3, sem_workspace, stream, cuda_adapter);
    if (status != Status::kSuccess) {
      return status;
    }
    status = GemmKernelM4::initialize_workspace(args4, sem_workspace, stream, cuda_adapter);
    if (status != Status::kSuccess) {
      return status;
    }
    status = GemmKernelM5::initialize_workspace(args5, sem_workspace, stream, cuda_adapter);
    if (status != Status::kSuccess) {
      return status;
    }
    status = GemmKernelM6::initialize_workspace(args6, sem_workspace, stream, cuda_adapter);
    if (status != Status::kSuccess) {
      return status;
    }

    // Initialize the Params structure
    params0 = GemmKernelM0::to_underlying_arguments(args0, presum_m_a, presum_m_b, postsum_m, sem_workspace);
    params1 = GemmKernelM1::to_underlying_arguments(args1, presum_m_a, presum_m_b, postsum_m, sem_workspace);
    params2 = GemmKernelM2::to_underlying_arguments(args2, presum_m_a, presum_m_b, postsum_m, sem_workspace);
    params3 = GemmKernelM3::to_underlying_arguments(args3, presum_m_a, presum_m_b, postsum_m, sem_workspace);
    params4 = GemmKernelM4::to_underlying_arguments(args4, presum_m_a, presum_m_b, postsum_m, sem_workspace);
    params5 = GemmKernelM5::to_underlying_arguments(args5, presum_m_a, presum_m_b, postsum_m, sem_workspace);
    params6 = GemmKernelM6::to_underlying_arguments(args6, presum_m_a, presum_m_b, postsum_m, sem_workspace);

    // Don't set the function attributes - require the CudaHostAdapter to set it.
    if constexpr (kEnableCudaHostAdapter) {
      CUTLASS_ASSERT(cuda_adapter);
      return Status::kSuccess;
    }
    else {
      //

      // Account for dynamic smem capacity if needed
      //
      int smem_size = ParallelGroup::SharedStorageSize();
      printf("693 %d\n", smem_size);
      CUTLASS_ASSERT(cuda_adapter == nullptr);

      if (smem_size >= (48 << 10)) {
        CUTLASS_TRACE_HOST("  Setting smem size to " << smem_size);
        cudaError_t result = cudaFuncSetAttribute(
            KernelParallelMiGroup<ParallelGroup>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            smem_size);
        if (cudaSuccess != result) {
          result = cudaGetLastError(); // to clear the error bit
          CUTLASS_TRACE_HOST("  cudaFuncSetAttribute() returned error: " << cudaGetErrorString(result));
          return Status::kErrorInternal;
        }
      }
    }
    return Status::kSuccess;
  }

  /// Update API is preserved in 3.0, but does not guarantee a lightweight update of params.
  Status
  update(Arguments const& args, void* workspace = nullptr) {
    printf("474 should not be called\n"); abort();
    // CUTLASS_TRACE_HOST("GemmUniversal()::update() - workspace: " << workspace);

    // size_t workspace_bytes = get_workspace_size(args);
    // if (workspace_bytes > 0 && nullptr == workspace) {
    //   return Status::kErrorWorkspaceNull;
    // }

    // params_ = GemmKernel::to_underlying_arguments(args, workspace);
    // return Status::kSuccess;
  }

  /// Primary run() entry point API that is static allowing users to create and manage their own params.
  /// Supplied params struct must be construct by calling GemmKernel::to_underlying_arguments()
  template<typename GemmKernel>
  static Status
  run(typename GemmKernel::Params& params,
      cudaStream_t stream = nullptr,
      CudaHostAdapter *cuda_adapter = nullptr,
      bool launch_with_pdl = false) {
    CUTLASS_TRACE_HOST("GemmUniversal::run()");
    dim3 const block = GemmKernel::get_block_shape();
    dim3 const grid = get_grid_shape<GemmKernel>(params);
    // configure smem size and carveout
    int smem_size = GemmKernel::SharedStorageSize;

    Status launch_result{ Status::kSuccess };
    // Use extended launch API only for mainloops that use it
    if constexpr (GemmKernel::ArchTag::kMinComputeCapability >= 90) {
#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
      CUTLASS_TRACE_HOST("GemmUniversal::run: Use extended launch API");
#endif
      [[maybe_unused]] constexpr bool is_static_1x1x1 =
        cute::is_static_v<typename GemmKernel::DispatchPolicy::ClusterShape> and
        cute::size(typename GemmKernel::DispatchPolicy::ClusterShape{}) == 1;
      [[maybe_unused]] dim3 cluster(cute::size<0>(typename GemmKernel::DispatchPolicy::ClusterShape{}),
        cute::size<1>(typename GemmKernel::DispatchPolicy::ClusterShape{}),
        cute::size<2>(typename GemmKernel::DispatchPolicy::ClusterShape{}));
      
      // Dynamic cluster support
      [[maybe_unused]] dim3 fallback_cluster = dim3{0,0,0};
      if constexpr (GemmKernel::ArchTag::kMinComputeCapability == 100 
                    || GemmKernel::ArchTag::kMinComputeCapability == 101
                    || GemmKernel::ArchTag::kMinComputeCapability == 103
                    ) {
        if constexpr (!cute::is_static_v<typename GemmKernel::DispatchPolicy::ClusterShape>) {
          fallback_cluster = params.hw_info.cluster_shape_fallback;
          cluster = params.hw_info.cluster_shape;
        }
      }
      
      [[maybe_unused]] void* kernel_params[] = {&params};

      if constexpr (kEnableCudaHostAdapter) {
        //
        // Use the cuda host adapter
        //
        CUTLASS_ASSERT(cuda_adapter);
        if (cuda_adapter) {
          if (launch_with_pdl) {
            CUTLASS_TRACE_HOST(
              "GemmUniversal::run() does not support launching with PDL and a custom cuda adapter.");
            return Status::kErrorInternal;
          }
#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
          CUTLASS_TRACE_HOST("GemmUniversal::run: Launching kernel with CUDA host adapter");
#endif
          if constexpr (is_static_1x1x1) {
            launch_result = cuda_adapter->launch(grid,
                                                block,
                                                smem_size,
                                                stream,
                                                kernel_params,
                                                0);
          }
          else {
            launch_result = cuda_adapter->launch(grid,
                                                cluster,
                                                fallback_cluster, 
                                                block,
                                                smem_size,
                                                stream,
                                                kernel_params,
                                                0);
          }
        }
        else {
          CUTLASS_TRACE_HOST("GemmUniversal::run: kEnableCudaHostAdapter is true, but CUDA host adapter is null");
          return Status::kErrorInternal;
        }
      }
      else {
        CUTLASS_ASSERT(cuda_adapter == nullptr);
        [[maybe_unused]] void const* kernel = (void const*) device_kernel<GemmKernel>;
        static constexpr bool kClusterLaunch = GemmKernel::ArchTag::kMinComputeCapability == 90;
        if constexpr (kClusterLaunch) {
          if constexpr (is_static_1x1x1) {
#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
            CUTLASS_TRACE_HOST("GemmUniversal::run: Launching static 1x1x1 kernel");
#endif
            launch_result = cutlass::kernel_launch<GemmKernel>(
              grid, block, smem_size, stream, params, launch_with_pdl);
            if (launch_result != Status::kSuccess) {
              CUTLASS_TRACE_HOST("GemmUniversal::run: cutlass::kernel_launch reports failure");
            }
#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
            else {
              CUTLASS_TRACE_HOST("GemmUniversal::run: cutlass::kernel_launch reports success");
            }
#endif
          }
          else {
#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
            CUTLASS_TRACE_HOST("GemmUniversal::run: Launching dynamic cluster kernel");
#endif
            launch_result = ClusterLauncher::launch(
              grid, cluster, block, smem_size, stream, kernel, kernel_params, launch_with_pdl);
          }
        }
        
        else {
          if constexpr (GemmKernel::ArchTag::kMinComputeCapability == 100
                        || GemmKernel::ArchTag::kMinComputeCapability == 101
                        || GemmKernel::ArchTag::kMinComputeCapability == 120
                        || GemmKernel::ArchTag::kMinComputeCapability == 103
                       ) {
            if constexpr (is_static_1x1x1) {
#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
              CUTLASS_TRACE_HOST("GemmUniversal::run: Launching static 1x1x1 kernel");
#endif
              launch_result = cutlass::kernel_launch<GemmKernel>(grid, block, smem_size, stream, params, launch_with_pdl);
              if (launch_result != Status::kSuccess) {
                CUTLASS_TRACE_HOST("GemmUniversal::run: cutlass::kernel_launch reports failure");
              }
#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
              else {
                CUTLASS_TRACE_HOST("GemmUniversal::run: cutlass::kernel_launch reports success");
              }
#endif
            }
            else {
#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
              CUTLASS_TRACE_HOST("GemmUniversal::run: Launching kernel with fall-back cluster");
#endif
              launch_result = ClusterLauncher::launch_with_fallback_cluster(
                grid, 
                cluster,
                fallback_cluster,
                block,
                smem_size,
                stream,
                kernel,
                kernel_params,
                launch_with_pdl);
            }
          }
        }
        
      }
    }
    else {
      launch_result = Status::kSuccess;
      cutlass::arch::synclog_setup();

      if constexpr (kEnableCudaHostAdapter) {
        CUTLASS_ASSERT(cuda_adapter);
        if (cuda_adapter) {
          void* kernel_params[] = {&params};
#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
          CUTLASS_TRACE_HOST("GemmUniversal::run: Launching kernel with CUDA host adapter");
#endif
          launch_result = cuda_adapter->launch(
            grid, block, smem_size, stream, kernel_params, 0
          );

        }
        else {
          CUTLASS_TRACE_HOST("GemmUniversal::run: CUDA host adapter is null");
          return Status::kErrorInternal;
        }
      }
      else {
        CUTLASS_ASSERT(cuda_adapter == nullptr);
#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
        CUTLASS_TRACE_HOST("GemmUniversal::run: Launching kernel with cutlass::kernel_launch");
#endif
        launch_result = cutlass::kernel_launch<GemmKernel>(
          grid, block, smem_size, stream, params, launch_with_pdl);
        if (launch_result != Status::kSuccess) {
          CUTLASS_TRACE_HOST("GemmUniversal::run: cutlass::kernel_launch reports failure");
        }
#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
        else {
          CUTLASS_TRACE_HOST("GemmUniversal::run: cutlass::kernel_launch reports success");
        }
#endif
      }
    }

    cudaError_t result = cudaGetLastError();
    if (cudaSuccess == result && Status::kSuccess == launch_result) {
#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
      CUTLASS_TRACE_HOST("GemmUniversal::run: cudaGetLastError reports success");
#endif
      return Status::kSuccess;
    }
    else {
      CUTLASS_TRACE_HOST("  Kernel launch failed. Reason: " << result);
      return Status::kErrorInternal;
    }
  }

  template<typename ParallelMiKernels>
  static Status
  run_parallel(typename GemmKernelM0::Params& params0, typename GemmKernelM1::Params& params1,
               typename GemmKernelM2::Params& params2, typename GemmKernelM3::Params& params3,
               typename GemmKernelM4::Params& params4, typename GemmKernelM5::Params& params5,
               typename GemmKernelM6::Params& params6,
      cudaStream_t stream = nullptr,
      CudaHostAdapter *cuda_adapter = nullptr,
      bool launch_with_pdl = false) {
    CUTLASS_TRACE_HOST("GemmUniversal::run()");
    dim3 const block = GemmKernelM0::get_block_shape();
    dim3 grid = get_grid_shape<GemmKernelM0>(params0);
    dim3 origGrid = grid;
    grid.z = ParallelMiKernels::NumKernels()*grid.z;
    ParallelMiKernels parallel_kernels(params0, params1, params2, params3, params4, params5, params6);

    // configure smem size and carveout
    int smem_size = ParallelMiKernels::SharedStorageSize();

    Status launch_result{ Status::kSuccess };
    // Use extended launch API only for mainloops that use it
    if constexpr (GemmKernelM0::ArchTag::kMinComputeCapability >= 90) {
#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
      CUTLASS_TRACE_HOST("GemmUniversal::run: Use extended launch API");
#endif
      [[maybe_unused]] constexpr bool is_static_1x1x1 =
        cute::is_static_v<typename GemmKernelM0::DispatchPolicy::ClusterShape> and
        cute::size(typename GemmKernelM0::DispatchPolicy::ClusterShape{}) == 1;
      [[maybe_unused]] dim3 cluster(cute::size<0>(typename GemmKernelM0::DispatchPolicy::ClusterShape{}),
        cute::size<1>(typename GemmKernelM0::DispatchPolicy::ClusterShape{}),
        cute::size<2>(typename GemmKernelM0::DispatchPolicy::ClusterShape{}));
      
      // Dynamic cluster support
      [[maybe_unused]] dim3 fallback_cluster = dim3{0,0,0};
      if constexpr (GemmKernelM0::ArchTag::kMinComputeCapability == 100 
                    || GemmKernelM0::ArchTag::kMinComputeCapability == 101
                    || GemmKernelM0::ArchTag::kMinComputeCapability == 103
                    ) {
        if constexpr (!cute::is_static_v<typename GemmKernelM0::DispatchPolicy::ClusterShape>) {
          fallback_cluster = params0.hw_info.cluster_shape_fallback;
          cluster = params0.hw_info.cluster_shape;
        }
      }
      
      [[maybe_unused]] void* kernel_params[] = {&parallel_kernels, &origGrid};
      CUTLASS_ASSERT(cuda_adapter == nullptr);
      [[maybe_unused]] void const* kernel = (void const*) KernelParallelMiGroup<ParallelMiKernels>;
      static constexpr bool kClusterLaunch = GemmKernelM0::ArchTag::kMinComputeCapability == 90;
      if constexpr (kClusterLaunch) {
#if (CUTLASS_DEBUG_TRACE_LEVEL > 1)
        CUTLASS_TRACE_HOST("GemmUniversal::run: Launching dynamic cluster kernel");
#endif
        launch_result = ClusterLauncher::launch(
          grid, cluster, block, smem_size, stream, kernel, kernel_params, launch_with_pdl);
        return launch_result;
      }
    }

    return Status::kErrorInternal;
  }

  //
  // Non-static launch overloads that first create and set the internal params struct of this kernel handle.
  //

  /// Launches the kernel after first constructing Params internal state from supplied arguments.
  // Status
  // run(
  //   Arguments const& args,
  //   void* workspace = nullptr,
  //   cudaStream_t stream = nullptr,
  //   CudaHostAdapter *cuda_adapter = nullptr,
  //   bool launch_with_pdl = false
  // ) {
  //   Status status = initialize(args, workspace, stream, cuda_adapter);

  //   if (Status::kSuccess == status) {
  //     status = run(paramsM0_, stream, cuda_adapter, launch_with_pdl);
  //   }
  //   return status;
  // }

  /// Launches the kernel after first constructing Params internal state from supplied arguments.
  // Status
  // operator()(
  //   Arguments const& args,
  //   void* workspace = nullptr,
  //   cudaStream_t stream = nullptr,
  //   CudaHostAdapter *cuda_adapter = nullptr,
  //   bool launch_with_pdl = false) {
  //   return run(args, workspace, stream, cuda_adapter, launch_with_pdl);
  // }

  /// Overload that allows a user to re-launch the   same kernel without updating internal params struct.
  Status
  run(
    cudaStream_t* streams = nullptr,
    int num_streams = 0,
    CudaHostAdapter *cuda_adapter = nullptr,
    bool launch_with_pdl = false) {
    if (num_streams == 1) {
      for (int i = 0; i < 7; i++)
        streams[i] = streams[0];
    }

    using ParallelGroup0 = ParallelMiKernels<typename ScheduleStrassenGroups::ParallelGroups0,
                                              GemmKernelM0, GemmKernelM1, GemmKernelM2, GemmKernelM3,
                                              GemmKernelM4, GemmKernelM5, GemmKernelM6>;
    using ParallelGroup1 = ParallelMiKernels<typename ScheduleStrassenGroups::ParallelGroups1,
                                              GemmKernelM0, GemmKernelM1, GemmKernelM2, GemmKernelM3,
                                              GemmKernelM4, GemmKernelM5, GemmKernelM6>;
    using ParallelGroup2 = ParallelMiKernels<typename ScheduleStrassenGroups::ParallelGroups2,
                                              GemmKernelM0, GemmKernelM1, GemmKernelM2, GemmKernelM3,
                                              GemmKernelM4, GemmKernelM5, GemmKernelM6>;
    using ParallelGroup3 = ParallelMiKernels<typename ScheduleStrassenGroups::ParallelGroups3,
                                              GemmKernelM0, GemmKernelM1, GemmKernelM2, GemmKernelM3,
                                              GemmKernelM4, GemmKernelM5, GemmKernelM6>;
    using ParallelGroup4 = ParallelMiKernels<typename ScheduleStrassenGroups::ParallelGroups4,
                                              GemmKernelM0, GemmKernelM1, GemmKernelM2, GemmKernelM3,
                                              GemmKernelM4, GemmKernelM5, GemmKernelM6>;
    using ParallelGroup5 = ParallelMiKernels<typename ScheduleStrassenGroups::ParallelGroups5,
                                              GemmKernelM0, GemmKernelM1, GemmKernelM2, GemmKernelM3,
                                              GemmKernelM4, GemmKernelM5, GemmKernelM6>;
    using ParallelGroup6 = ParallelMiKernels<typename ScheduleStrassenGroups::ParallelGroups6,
                                              GemmKernelM0, GemmKernelM1, GemmKernelM2, GemmKernelM3,
                                              GemmKernelM4, GemmKernelM5, GemmKernelM6>;

    Status result = Status::kSuccess;

    if (StrassenGroups::PresumGroup::AllPresums::APresumComputeLoads(PresumGlobalKernel).numAccess() > 0 ||
        StrassenGroups::PresumGroup::AllPresums::BPresumComputeLoads(PresumGlobalKernel).numAccess() > 0) {
      //TODO: Add a swizzle?
      dim3 grid = {uint((paramsM0_.get_problem_shape_n()/2)/GemmKernelM0::Mma::PresumShape::kN),
                   uint((paramsM0_.get_problem_shape_m()/2)/GemmKernelM0::Mma::PresumShape::kM),
                   1};
      KernelPresumGlobalCompute<typename StrassenGroups::PresumGroup, GemmKernelM0, 128><<<grid, 128, 0, streams[0]>>>(paramsM0_);
      auto result = cudaDeviceSynchronize();
      if (result != cudaSuccess)
      {printf("Error at %d: %s\n", __LINE__, cudaGetErrorString(result)); return Status::kErrorInternal;}
      paramsM0_.run += 1;
    }

    char* only_m = getenv("ONLY_M");
    int valid_ms[] = {1,1,1,1,1,1,1};

    if (only_m) {
      std::string str = std::string(only_m);
      std::stringstream ss(str);
      std::string t;
      char del = ',';
      for (int i = 0; i < 7; i++) valid_ms[i] = 0;
      while (getline(ss, t, del)) {
        valid_ms[std::stoi(t)] = 1;
      }
    }
    uint stream_idx = 0;
    if ((!only_m or valid_ms[0] == 1) && ParallelGroup0::HasAKernel()) {
      result = run_parallel<ParallelGroup0>(paramsM0_, paramsM1_, paramsM2_, paramsM3_, paramsM4_, paramsM5_, paramsM6_,
                                            streams[(stream_idx++)%num_streams], cuda_adapter, launch_with_pdl);
      if (result != Status::kSuccess) {
        printf("Error at %d: %s\n", __LINE__, cudaGetErrorString(cudaGetLastError())); return Status::kErrorInternal;
      }
      // cudaStreamSynchronize(streams[(stream_idx-1)%num_streams]);
    }

    if (false) {
      cudaDeviceSynchronize();
      #if 0
      uint R = 8*1024/2, C = 9*1024/2;
      ElementB* h_presum_b = new ElementB[R*C];
      ElementB* b = new ElementB[2*R*2*C];
      cudaMemcpy(h_presum_b, &paramsM0_.presum_m_b_workspace[2*R*C], R*C*sizeof(ElementB), cudaMemcpyDeviceToHost);
      cudaMemcpy(b, paramsM0_.get_ptr_B(), 2*R*2*C*sizeof(ElementB), cudaMemcpyDeviceToHost);

      for (int c = 0; c < C; c++) {
        bool to_break = false;
        float accum = 0;
        for (int r = 0; r < R; r++) {
          auto b0 = b[r*2*C+c];
          auto b1 = b[r*2*C+C+c];
          auto b2 = b[(R+r)*2*C+c];
          auto b3 = b[(R+r)*2*C+C+c];

          auto b31 = b3-b1;
          auto b10 = b1-b0;
          auto s3 = b31+b0;
          accum += float(h_presum_b[r*C+c]);
          if (s3 != h_presum_b[r*C+c]) {
            printf("910 %d, %d : %f %f : = %f %f %f %f\n", r,c, float(s3), float(h_presum_b[r*C+c]), float(b0), float(b1), float(b2), float(b3));
            to_break = true;
            break;
          }
        }
        printf("916 %f\n", accum);
        break;
        if (to_break) break;
      }
      #endif
      presumcheck<ElementA><<<paramsM0_.get_problem_shape_k()/2,1024>>>(paramsM0_.get_problem_shape_k(), paramsM0_.get_problem_shape_n(), paramsM0_.ptr_B, paramsM0_.presum_m_b_workspace);
      cudaDeviceSynchronize();
      exit(EXIT_SUCCESS);
    }
// postsumcheck<<<4096,1024,0,streams[4]>>>(paramsM0_.postsum_m_workspace);
    if ((!only_m or valid_ms[1] == 1) && ParallelGroup1::HasAKernel()) {
      result = run_parallel<ParallelGroup1>(paramsM0_, paramsM1_, paramsM2_, paramsM3_, paramsM4_, paramsM5_, paramsM6_,
                                            streams[(stream_idx++)%num_streams], cuda_adapter, launch_with_pdl);
      if (result != Status::kSuccess) {
        printf("Error at %d: %s\n", __LINE__, cudaGetErrorString(cudaGetLastError())); return Status::kErrorInternal;
      }
      // cudaStreamSynchronize(streams[(stream_idx-1)%num_streams]);
    }

    if ((!only_m or valid_ms[2] == 1) && ParallelGroup2::HasAKernel()) {
      result = run_parallel<ParallelGroup2>(paramsM0_, paramsM1_, paramsM2_, paramsM3_, paramsM4_, paramsM5_, paramsM6_,
                                            streams[(stream_idx++)%num_streams], cuda_adapter, launch_with_pdl);
      if (result != Status::kSuccess) {
        printf("Error at %d: %s\n", __LINE__, cudaGetErrorString(cudaGetLastError())); return Status::kErrorInternal;
      }
      // cudaStreamSynchronize(streams[(stream_idx-1)%num_streams]);
    }

    if ((!only_m or valid_ms[3] == 1) && ParallelGroup3::HasAKernel()) {
      result = run_parallel<ParallelGroup3>(paramsM0_, paramsM1_, paramsM2_, paramsM3_, paramsM4_, paramsM5_, paramsM6_,
                                            streams[(stream_idx++)%num_streams], cuda_adapter, launch_with_pdl);
      if (result != Status::kSuccess) {
        printf("Error at %d: %s\n", __LINE__, cudaGetErrorString(cudaGetLastError())); return Status::kErrorInternal;
      }
      // cudaStreamSynchronize(streams[(stream_idx-1)%num_streams]);
    }
    if ((!only_m or valid_ms[4] == 1) && ParallelGroup4::HasAKernel()) {
      result = run_parallel<ParallelGroup4>(paramsM0_, paramsM1_, paramsM2_, paramsM3_, paramsM4_, paramsM5_, paramsM6_,
                                            streams[(stream_idx++)%num_streams], cuda_adapter, launch_with_pdl);
      if (result != Status::kSuccess) {
        printf("Error at %d: %s\n", __LINE__, cudaGetErrorString(cudaGetLastError())); return Status::kErrorInternal;
      }
      // cudaStreamSynchronize(streams[(stream_idx-1)%num_streams]);
    }
    if ((!only_m or valid_ms[5] == 1) && ParallelGroup5::HasAKernel()) {
      result = run_parallel<ParallelGroup5>(paramsM0_, paramsM1_, paramsM2_, paramsM3_, paramsM4_, paramsM5_, paramsM6_,
                                            streams[(stream_idx++)%num_streams], cuda_adapter, launch_with_pdl);
      if (result != Status::kSuccess) {
        printf("Error at %d: %s\n", __LINE__, cudaGetErrorString(cudaGetLastError())); return Status::kErrorInternal;
      }
      // cudaStreamSynchronize(streams[(stream_idx-1)%num_streams]);
    }
    if ((!only_m or valid_ms[6] == 1) && ParallelGroup6::HasAKernel()) {
      result = run_parallel<ParallelGroup6>(paramsM0_, paramsM1_, paramsM2_, paramsM3_, paramsM4_, paramsM5_, paramsM6_,
                                            streams[(stream_idx++)%num_streams], cuda_adapter, launch_with_pdl);
      if (result != Status::kSuccess) {
        printf("Error at %d: %s\n", __LINE__, cudaGetErrorString(cudaGetLastError())); return Status::kErrorInternal;
      }
      // cudaStreamSynchronize(streams[(stream_idx-1)%num_streams]);
    }
    /*
    //M2 = M0 + M2
    if ((!only_m or valid_ms[2] == 1) && ParallelGroup2::HasAKernel()) {
      if (launch_parallel_kernels) {
        result = run_parallel<ParallelGroup1>(paramsM0_, paramsM1_, paramsM2_, paramsM3_, paramsM4_, paramsM5_, paramsM6_,
                                              streams[(stream_idx++)%num_streams], cuda_adapter, launch_with_pdl);
      } else {
        // result = run<GemmKernelM2>(paramsM2_, streams[(stream_idx++)%num_streams], cuda_adapter, launch_with_pdl);
        if (result != Status::kSuccess) {
          printf("Error at %d: %s\n", __LINE__, cudaGetErrorString(cudaGetLastError())); return Status::kErrorInternal;
        }
      }
    }

    if (!launch_parallel_kernels) {
    //M3 = M2 + M3 = M0 + M2 + M3
    if ((!only_m or valid_ms[3] == 1) and StrassenGroups::Group3::hasM3() and !StrassenGroups::Group2::hasM3()) {
      // result = run<GemmKernelM3>(paramsM3_, streams[(stream_idx++)%num_streams], cuda_adapter, launch_with_pdl);
      if (result != Status::kSuccess) {
        printf("Error at %d: %s\n", __LINE__, cudaGetErrorString(cudaGetLastError())); return Status::kErrorInternal;
      }
    }

    if (!only_m or valid_ms[4] == 1) {
      //M4 = M4; C3 = M3 + M4 //Update M2 with M4 using TMA's reduce?
      // result = run<GemmKernelM4>(paramsM4_, streams[(stream_idx++)%num_streams], cuda_adapter, launch_with_pdl);
      if (result != Status::kSuccess) {
        printf("Error at %d: %s\n", __LINE__, cudaGetErrorString(cudaGetLastError())); return Status::kErrorInternal;
      }
    }

    // postsumcheck<<<4096,1024,0,streams[4]>>>(paramsM0_.postsum_m_workspace);

    if ((!only_m or valid_ms[5] == 1) and StrassenGroups::Group5::hasM5() and !StrassenGroups::Group4::hasM5()) {
      //C1 = M2 + M4 + M5 = M0 + M2 + M4 + M5
      // result = run<GemmKernelM5>(paramsM5_, streams[(stream_idx++)%num_streams], cuda_adapter, launch_with_pdl);
      if (result != Status::kSuccess) {
        printf("Error at %d: %s\n", __LINE__, cudaGetErrorString(cudaGetLastError())); return Status::kErrorInternal;
      }
    }
    // cudaDeviceSynchronize();
    // postsumcheck<<<9*1024/2,1024,0,streams[0]>>>(paramsM0_.postsum_m_workspace);
    // cudaDeviceSynchronize();
    // exit(EXIT_SUCCESS);
    if (!only_m or valid_ms[6] == 1) {
      //C2 = M3 - M6 = M0 + M2 + M3 - M6
      // result = run<GemmKernelM6>(paramsM6_, streams[(stream_idx++)%num_streams], cuda_adapter, launch_with_pdl);
      if (result != Status::kSuccess) {
        printf("Error at %d: %s\n", __LINE__, cudaGetErrorString(cudaGetLastError())); return Status::kErrorInternal;
      }
    }
    */

    return result;
  }

  /// Overload that allows a user to re-launch the same kernel without updating internal params struct.
  // Status
  // operator()(cudaStream_t stream = nullptr, CudaHostAdapter *cuda_adapter = nullptr, bool launch_with_pdl = false) {
  //   return run(paramsM0_, stream, cuda_adapter, launch_with_pdl);
  // }
};

////////////////////////////////////////////////////////////////////////////////
////////////////////////////// CUTLASS 2.x API /////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

template <typename ScheduleStrassenGroups, class StrassenGemmKernels>
class StrassenGemmUniversalAdapter<
  ScheduleStrassenGroups,
  StrassenGemmKernels,
  cute::enable_if_t<not gemm::detail::IsCutlass3GemmKernel<GetUnderlyingKernel_t<typename StrassenGemmKernels::GemmKernel0>>::value>>
{
public:

  using GemmKernel = GetUnderlyingKernel_t<typename StrassenGemmKernels::GemmKernel0>;

  static bool const kInternalTranspose =
    !cutlass::epilogue::threadblock::detail::is_2x_evt_v<typename GemmKernel::Epilogue> &&  // 2.x EVT does not require internal transpose
    cute::is_same<typename GemmKernel::LayoutC, cutlass::layout::RowMajor>::value;

  using ThreadblockShape = typename GemmKernel::Mma::Shape;
  using WarpShape = typename GemmKernel::WarpShape;
  using InstructionShape = typename GemmKernel::InstructionShape;

  // warp-level, arch-level (instruction), math operator
  using WarpMmaOperator = typename GemmKernel::Mma::Policy::Operator;
  using ArchMmaOperator = typename WarpMmaOperator::ArchMmaOperator;
  using MathOperator = typename WarpMmaOperator::MathOperator;

  // Operator class and arch tag extract bottom-up
  // set it for top-level gemm device-level template
  using OperatorClass = typename WarpMmaOperator::OperatorClass;
  using ArchTag = typename WarpMmaOperator::ArchTag;

  // Type, layout, and complex transform deliberately exchanged with B
  using MapArguments = kernel::detail::MapArguments<
    typename GemmKernel::ElementA,
    typename GemmKernel::LayoutA,
    GemmKernel::kTransformA,
    GemmKernel::kAlignmentA,
    typename GemmKernel::ElementB,
    typename GemmKernel::LayoutB,
    GemmKernel::kTransformB,
    GemmKernel::kAlignmentB,
    typename GemmKernel::LayoutC,
    kInternalTranspose
  >;

  using ElementA = typename MapArguments::ElementA;
  using LayoutA = typename MapArguments::LayoutA;
  static ComplexTransform const kTransformA = MapArguments::kTransformA;
  static int const kAlignmentA = MapArguments::kAlignmentA;

  using ElementB = typename MapArguments::ElementB;
  using LayoutB = typename MapArguments::LayoutB;
  static ComplexTransform const kTransformB = MapArguments::kTransformB;
  static int const kAlignmentB = MapArguments::kAlignmentB;

  using ElementC = typename GemmKernel::ElementC;
  using LayoutC = typename MapArguments::LayoutC;
  static int const kAlignmentC = GemmKernel::kAlignmentC;

  // C and D same type for 2.x kernel
  using ElementD = ElementC;
  using LayoutD = LayoutC;

  using TensorRefA = TensorRef<ElementA const, LayoutA>;
  using TensorRefB = TensorRef<ElementB const, LayoutB>;
  using TensorRefC = TensorRef<ElementC const, LayoutC>;
  using TensorRefD = TensorRef<ElementD, LayoutD>;

  static int const kStages = GemmKernel::Mma::kStages;

  using EpilogueOutputOp = typename GemmKernel::EpilogueOutputOp;
  using ElementAccumulator = typename EpilogueOutputOp::ElementAccumulator;
  using ThreadblockSwizzle = typename GemmKernel::ThreadblockSwizzle;
  using UnderlyingOperator = GemmUniversalBase<GemmKernel>;
  using Arguments = typename UnderlyingOperator::Arguments;

private:

  UnderlyingOperator underlying_operator_;

public:

  /// Constructs the GEMM.
  StrassenGemmUniversalAdapter() { }

  /// Helper to construct a transposed equivalent for the underlying GEMM operator
  static Arguments to_underlying_arguments(Arguments const &args) {
    if (kInternalTranspose) {
      return args.transposed_problem();
    }
    else {
      return args;
    }
  }

  /// Determines whether the GEMM can execute the given problem.
  static Status can_implement(Arguments const &args, CudaHostAdapter *cuda_adapter = nullptr) {

    return UnderlyingOperator::can_implement(to_underlying_arguments(args), cuda_adapter);
  }

  /// Gets the workspace size
  static size_t get_workspace_size(Arguments const &args, CudaHostAdapter *cuda_adapter = nullptr) {

    return UnderlyingOperator::get_workspace_size(to_underlying_arguments(args), cuda_adapter);
  }

  /// Computes the grid shape
  static dim3 get_grid_shape(Arguments const &args) {
    return UnderlyingOperator::get_grid_shape(to_underlying_arguments(args));
  }

  /// Computes the maximum number of active blocks per multiprocessor
  static int maximum_active_blocks(int smem_capacity = -1) {
    return UnderlyingOperator::maximum_active_blocks(smem_capacity);
  }

  /// Initializes GEMM state from arguments.
  Status initialize(
    Arguments const &args,
    void *workspace = nullptr,
    cudaStream_t stream = nullptr,
    CudaHostAdapter *cuda_adapter = nullptr
  ) {

    return underlying_operator_.initialize(to_underlying_arguments(args), workspace, stream, cuda_adapter);
  }

  /// Lightweight update given a subset of arguments.
  Status update(Arguments const &args) {

    return underlying_operator_.update(to_underlying_arguments(args));
  }

  /// Runs the kernel using initialized state.
  Status run(
    cudaStream_t stream = nullptr,
    CudaHostAdapter *cuda_adapter = nullptr) {

    return underlying_operator_.run(stream, cuda_adapter);
  }

  /// Runs the kernel using initialized state.
  Status operator()(
    cudaStream_t stream = nullptr,
    CudaHostAdapter *cuda_adapter = nullptr) {

    return run(stream);
  }

  /// Runs the kernel using initialized state.
  Status operator()(
    Arguments const &args,
    void *workspace = nullptr,
    cudaStream_t stream = nullptr,
    CudaHostAdapter *cuda_adapter = nullptr) {

    Status status = initialize(args, workspace, stream, cuda_adapter);

    if (status == Status::kSuccess) {
      status = run(stream, cuda_adapter);
    }

    return status;
  }
};

////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::device

////////////////////////////////////////////////////////////////////////////////
