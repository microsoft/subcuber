/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/workspace.h"
#include "cutlass/kernel_hardware_info.hpp"
#include "cutlass/fast_math.h"
#include "cute/arch/cluster_sm90.hpp"
#include "cutlass/arch/reg_reconfig.h"
#include "cutlass/arch/mma_sm90.h"
#include "cutlass/epilogue/collective/detail.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/kernel/sm90_tile_scheduler.hpp"
#include "cutlass/gemm/kernel/sm90_strassen_tile_scheduler.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/gemm/kernel/strassen_tile_scheduler.hpp"
#include "cutlass/gemm/kernel/gemm_universal_decl.h"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/trace.h"

#include "cute/tensor.hpp"
#include "cutlass/arch/grid_dependency_control.h"

///////////////////////////////////////////////////////////////////////////////

template<bool EpilogueAndPresumDifferent, bool EpilogueStructOrUnion, typename CollectiveMainloop, typename CollectiveEpilogue>
struct ExtraStorage {};

template<typename CollectiveMainloop, typename CollectiveEpilogue>
struct ExtraStorage<false, false, CollectiveMainloop, CollectiveEpilogue> {
  union {
    union {
      typename CollectiveEpilogue::TensorStorage epilogue;
      typename CollectiveEpilogue::TensorStorage epilogue2;
    };

    struct {
      typename CollectiveMainloop::PresumTensorStorage presum_tensors;
      typename CollectiveMainloop::PresumTensorStorage2 presum_tensors2;
    };

    struct {
      typename CollectiveEpilogue::PresumTensorStorage epi_presum_tensors;
      typename CollectiveEpilogue::PresumTensorStorage epi_presum_tensors2;
    };
  };
};

template<typename CollectiveMainloop, typename CollectiveEpilogue>
struct ExtraStorage<false, true, CollectiveMainloop, CollectiveEpilogue> {
  union {
    union {
      typename CollectiveEpilogue::TensorStorage epilogue;
      typename CollectiveEpilogue::TensorStorage epilogue2;
    };

    struct {
      typename CollectiveMainloop::PresumTensorStorage presum_tensors;
      typename CollectiveMainloop::PresumTensorStorage2 presum_tensors2;
    };

    struct {
      typename CollectiveEpilogue::PresumTensorStorage epi_presum_tensors;
      typename CollectiveEpilogue::PresumTensorStorage epi_presum_tensors2;
    };
  };
};

template<typename CollectiveMainloop, typename CollectiveEpilogue>
struct ExtraStorage<true, false, CollectiveMainloop, CollectiveEpilogue> {
  typename CollectiveEpilogue::TensorStorage epilogue;
  // using FusionStorage = cute::conditional_t<(CollectiveMainloop::StrassenMiGroup::numMs() > 1),
  //                                           char[7680], char[8]>;
  // FusionStorage fusion;

  union {
    struct {
      typename CollectiveEpilogue::TensorStorage epilogue2;
    };

    struct {
      typename CollectiveMainloop::PresumTensorStorage presum_tensors;
      typename CollectiveMainloop::PresumTensorStorage2 presum_tensors2;
    };

    struct {
      typename CollectiveEpilogue::PresumTensorStorage epi_presum_tensors;
      typename CollectiveEpilogue::PresumTensorStorage epi_presum_tensors2;
    };
  };
};

template<typename CollectiveMainloop, typename CollectiveEpilogue>
struct ExtraStorage<true, true, CollectiveMainloop, CollectiveEpilogue> {
  typename CollectiveEpilogue::TensorStorage epilogue;
  // char fused_storage[7680];

  union {
    struct {
      typename CollectiveEpilogue::TensorStorage epilogue2;
    };

    struct {
      typename CollectiveMainloop::PresumTensorStorage presum_tensors;
      typename CollectiveMainloop::PresumTensorStorage2 presum_tensors2;
    };

    struct {
      typename CollectiveEpilogue::PresumTensorStorage epi_presum_tensors;
      typename CollectiveEpilogue::PresumTensorStorage epi_presum_tensors2;
    };
  };
};

namespace cutlass::gemm::kernel {

///////////////////////////////////////////////////////////////////////////////

template <
  typename StrassenMiGroup_,
  class ProblemShape_,
  class CollectiveMainloop_,
  class CollectiveEpilogue_,
  class TileScheduler_
>
class StrassenGemmUniversal<
  StrassenMiGroup_,
  ProblemShape_,
  CollectiveMainloop_,
  CollectiveEpilogue_,
  TileScheduler_,
  cute::enable_if_t<cute::is_base_of_v<KernelTmaWarpSpecializedPingpong, typename CollectiveMainloop_::DispatchPolicy::Schedule>>>
{
public:
  //
  // Type Aliases
  //
  using ProblemShape = ProblemShape_;
  static_assert(cute::rank(ProblemShape{}) == 3 or cute::rank(ProblemShape{}) == 4,
    "ProblemShape{} should be <M,N,K> or <M,N,K,L>");
  static constexpr bool IsGdcEnabled = cutlass::arch::IsGdcGloballyEnabled;

  using StrassenMiGroup = StrassenMiGroup_;
  using Mma = CollectiveMainloop_;

  // Mainloop derived types
  using CollectiveMainloop = CollectiveMainloop_;
  using TileShape = typename CollectiveMainloop::TileShape;
  using TiledMma  = typename CollectiveMainloop::TiledMma;
  using ArchTag   = typename CollectiveMainloop::ArchTag;
  using ElementA  = typename CollectiveMainloop::ElementA;
  using StrideA   = typename CollectiveMainloop::StrideA;
  using ElementB  = typename CollectiveMainloop::ElementB;
  using StrideB   = typename CollectiveMainloop::StrideB;
  using DispatchPolicy = typename CollectiveMainloop::DispatchPolicy;
  using ElementAccumulator = typename CollectiveMainloop::ElementAccumulator;
  using ClusterShape = typename DispatchPolicy::ClusterShape;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using MainloopParams = typename CollectiveMainloop::Params;
  static_assert(ArchTag::kMinComputeCapability >= 90);

  // Epilogue derived types
  using CollectiveEpilogue = CollectiveEpilogue_;
  using ElementC = typename CollectiveEpilogue::ElementC;
  using StrideC  = typename CollectiveEpilogue::StrideC;
  using ElementD = typename CollectiveEpilogue::ElementD;
  using StrideD  = typename CollectiveEpilogue::StrideD;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;

  static_assert(!cute::is_same_v<TileScheduler_, StreamKScheduler>, "Ping-pong kernel does not currently support stream-K scheduler.");
  static constexpr uint32_t TileSchedulerPipelineStageCount = DispatchPolicy::Schedule::SchedulerPipelineStageCount;
  using TileSchedulerTag = TileScheduler_;
  using TileScheduler = typename detail::StrassenTileSchedulerSelector<
                                          TileSchedulerTag, 
                                          ArchTag, 
                                          TileShape,
                                          ClusterShape,
                                          TileSchedulerPipelineStageCount
                                          >::Scheduler;
  using TileSchedulerArguments = typename TileScheduler::Arguments;
  using TileSchedulerParams = typename TileScheduler::Params;
  using TileSchedulerPipeline = typename TileScheduler::Pipeline;
  using TileSchedulerPipelineState = typename TileSchedulerPipeline::PipelineState;
  using TileSchedulerStorage = typename TileScheduler::SharedStorage;

  using TileSchedulerThrottlePipeline = typename TileScheduler::ThrottlePipeline;
  using TileSchedulerThrottlePipelineState = typename TileSchedulerThrottlePipeline::PipelineState;

  static constexpr bool IsSchedDynamicPersistent = TileScheduler::IsDynamicPersistent;
  static constexpr bool UseLinearStoreLoads = CollectiveEpilogue::UseLinearStoreLoads;

  // Warp specialization thread count per threadblock
  static constexpr uint32_t NumSchedThreads        = NumThreadsPerWarp;      // 1 warp
  static constexpr uint32_t NumMainloopLoadThreads = NumThreadsPerWarp;      // 1 warp
  static constexpr uint32_t NumEpilogueLoadThreads = NumThreadsPerWarp;      // 1 warp for C
  static constexpr uint32_t NumLoadWarpGroups = 1;
  static constexpr uint32_t NumMmaWarpGroups = 2;
  static constexpr uint32_t NumProducerThreads = CollectiveMainloop::NumProducerThreadEvents;
  static constexpr uint32_t NumMMAThreads = size(TiledMma{});                 // 4 warp
  static constexpr uint32_t MaxThreadsPerBlock = NumMMAThreads * NumMmaWarpGroups + (NumLoadWarpGroups * NumThreadsPerWarpGroup);
  static constexpr uint32_t kThreadCount = MaxThreadsPerBlock; 
  static constexpr uint32_t MinBlocksPerMultiprocessor = 1;
  static constexpr bool     IsMainloopAuxiliaryLoadNeeded = detail::HasAuxiliaryLoad_v<typename CollectiveMainloop::DispatchPolicy>;
  
  static constexpr uint32_t kPresumThreads = CollectiveMainloop::kPresumThreads;

  static_assert(NumMMAThreads == 128, "Pingpong kernel must have TiledMMA operating using 128 threads.");
  static_assert(MaxThreadsPerBlock == 384, "Pingpong kernel must have 384 threads in total.");
  static const bool IsFusedM2M3 = StrassenMiGroup::hasM2() && StrassenMiGroup::hasM3();
  static const bool IsFusedM4M5 = StrassenMiGroup::hasM4() && StrassenMiGroup::hasM5();

  /// Register requirement for Load and Math WGs
  static constexpr int RegsPerThread =
    (size<0>(TileShape{}) * size<1>(TileShape{}) * sizeof(ElementAccumulator))
    / (NumMMAThreads * sizeof(uint32_t));
  static constexpr bool HeavyRegisterPressure = RegsPerThread >= 208;
  static constexpr uint32_t LoadRegisterRequirement = !HeavyRegisterPressure ? 40 : 24;
  static constexpr uint32_t MmaRegisterRequirement = !HeavyRegisterPressure ? 232 : 240;

  // 1 stage ordered sequence between mainloop and epilogue producer load threads
  using LoadWarpOrderBarrier = cutlass::OrderedSequenceBarrier<1,2>;

  // Order Sequence barrier with two stages: one for Mainloop and one for Epilogue
  static constexpr uint32_t StagesPerMathWarpGroup = 2;
  using MathWarpGroupOrderBarrier = cutlass::OrderedSequenceBarrier<
    StagesPerMathWarpGroup, NumMmaWarpGroups>;
  using MathWarpGroupOrderBarrierSharedStorage =
    cutlass::PipelineDetail::OrderedSequenceBarrierSharedStorage<
      MathWarpGroupOrderBarrier::SequenceDepth,
      MathWarpGroupOrderBarrier::SequenceLength>;

  // Kernel level shared memory storage
  struct SharedStorage {
    struct PipelineStorage : cute::aligned_struct<16, _1> {
      using MainloopPipelineStorage = typename CollectiveMainloop::PipelineStorage;
      using PresumLoadPipelineStorage = typename CollectiveMainloop::PresumLoadPipeline::SharedStorage;
      using EpiLoadPipelineStorage = typename CollectiveEpilogue::PipelineStorage;
      using MathWarpGroupOrderBarrierStorage = MathWarpGroupOrderBarrierSharedStorage;

      alignas(16) MainloopPipelineStorage mainloop;
      // alignas(16) PresumLoadPipelineStorage presum_load;
      alignas(16) EpiLoadPipelineStorage epi_load;
      alignas(16) MathWarpGroupOrderBarrierStorage math_wg_order;
      alignas(16) typename LoadWarpOrderBarrier::SharedStorage load_order;
    } pipelines;
    
    alignas(16) TileSchedulerStorage scheduler;

    struct TensorStorage : cute::aligned_struct<128, _1> {
      using MainloopTensorStorage = typename CollectiveMainloop::TensorStorage;
      using EpilogueTensorStorage = typename CollectiveEpilogue::TensorStorage;

      MainloopTensorStorage mainloop;
      using ExtraStorage2 = ExtraStorage<(StrassenMiGroup::hasM0() && StrassenMiGroup::AllPresums::computeAnyAPresum()) ||
                                         (StrassenMiGroup::hasM1() && StrassenMiGroup::AllPresums::computeAnyBPresum()),
                                          StrassenMiGroup::numMs() == 2, CollectiveMainloop, CollectiveEpilogue>;
      ExtraStorage2 extra_storage;
    } tensors;
  };

  static constexpr int SharedStorageSize = sizeof(SharedStorage);

  // Device side arguments
  struct Arguments {
    GemmUniversalMode mode{};
    ProblemShape problem_shape{};
    MainloopArguments mainloop{};
    EpilogueArguments epilogue{};
    KernelHardwareInfo hw_info{};
    TileSchedulerArguments scheduler{};

    Arguments(GemmUniversalMode mode, ProblemShape problem_shape,
              MainloopArguments mainloop, EpilogueArguments epilogue,
              KernelHardwareInfo hw_info, TileSchedulerArguments scheduler = TileSchedulerArguments()) :
      mode(mode), problem_shape(problem_shape), mainloop(mainloop), epilogue(epilogue),
      hw_info(hw_info), scheduler(scheduler) 
    {}

    CUTLASS_HOST_DEVICE
    int get_problem_shape_m() const {
      return get<0>(problem_shape);
    }

    CUTLASS_HOST_DEVICE
    int get_problem_shape_n() const {
      return get<1>(problem_shape);
    }

    CUTLASS_HOST_DEVICE
    int get_problem_shape_k() const {
      return get<2>(problem_shape);
    }

    CUTLASS_HOST_DEVICE
    ProblemShape get_half_problem_shape() const {
      return ProblemShape{get_problem_shape_m()/2, get_problem_shape_n()/2,
                          get_problem_shape_k()/2};
    }

    template<typename Other>
    Arguments(const Other& other) :
      mode(other.mode), problem_shape(other.problem_shape), mainloop(other.mainloop),
      epilogue(other.epilogue), hw_info(other.hw_info), scheduler(other.scheduler)
      {}
  };

  // Kernel entry point API
  struct Params {
    GemmUniversalMode mode{};
    ProblemShape problem_shape{};
    MainloopParams mainloop{};
    EpilogueParams epilogue{};
    KernelHardwareInfo hw_info{};
    TileSchedulerParams scheduler{};
    ElementA* ptr_A;
    ElementB* ptr_B;
    ElementD* ptr_D;
    ElementA* presum_m_a_workspace;
    ElementB* presum_m_b_workspace;
    ElementD* postsum_m_workspace;

    int run = 0;

    CUTLASS_HOST_DEVICE
    int get_problem_shape_m() const {
      return get<0>(problem_shape);
    }

    CUTLASS_HOST_DEVICE
    int get_problem_shape_n() const {
      return get<1>(problem_shape);
    }

    CUTLASS_HOST_DEVICE
    int get_problem_shape_k() const {
      return get<2>(problem_shape);
    }

    CUTLASS_HOST_DEVICE
    int get_stride_A() const {
      return get_problem_shape_k();
    }

    CUTLASS_HOST_DEVICE
    int get_stride_B() const {
      return get_problem_shape_n();
    }

    CUTLASS_HOST_DEVICE
    int get_stride_MA() const {
      return get_problem_shape_k()/2;
    }

    CUTLASS_HOST_DEVICE
    int get_stride_MB() const {
      return get_problem_shape_n()/2;
    }

    CUTLASS_HOST_DEVICE
    int get_presum_log_tile_multiplier_a() const {
      return mainloop.get_presum_tile_log_multiplier_a();
    }
  
    CUTLASS_HOST_DEVICE
    int get_presum_log_tile_multiplier_b() const {
      return mainloop.get_presum_tile_log_multiplier_b();
    }

    CUTLASS_HOST_DEVICE
    ProblemShape get_half_problem_shape() const {
      return ProblemShape{get_problem_shape_m()/2, get_problem_shape_n()/2,
                          get_problem_shape_k()/2};
    }

    CUTLASS_HOST_DEVICE
    ElementA* get_ptr_A() const {
      return ptr_A;
    }

    CUTLASS_HOST_DEVICE
    ElementB* get_ptr_B() const {
      return ptr_B;
    }

    CUTLASS_HOST_DEVICE
    ElementD* get_ptr_D() const {
      return ptr_D;
    }
  };

  //
  // Methods
  //

  // Convert to underlying arguments. In this case, a simple copy for the aliased type.
  static
  Params
  to_underlying_arguments(Arguments const& args, ElementA* presum_m_a, ElementB* presum_m_b, ElementD* postsum_m, void* workspace) {
    CUTLASS_TRACE_HOST("to_underlying_arguments():");

    (void) workspace;
    auto problem_shape = ProblemShape{get<0>(args.problem_shape),
                                      get<1>(args.problem_shape),
                                      get<2>(args.problem_shape)};

    if constexpr (detail::Has_SwapAB_v<CollectiveMainloop>) {
      // swap M/N
      get<0>(problem_shape) = get<1>(args.problem_shape);
      get<1>(problem_shape) = get<0>(args.problem_shape);
    }
    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto half_problem_shape_MNKL = append<4>(args.get_half_problem_shape(), 1);


    // Get SM count if needed, otherwise use user supplied SM count
    int sm_count = args.hw_info.sm_count;
    if (sm_count <= 0) {
      CUTLASS_TRACE_HOST("  WARNING: Arguments do not include a valid SM count.\n"
          "  For optimal performance, populate the arguments KernelHardwareInfo struct with the SM count.");
      sm_count = KernelHardwareInfo::query_device_multiprocessor_count(args.hw_info.device_id);
    }
    CUTLASS_TRACE_HOST("to_underlying_arguments(): Setting persistent grid SM count to " << sm_count);

    // Get maximum number of clusters that could co-exist on the target device
    int max_active_clusters = args.hw_info.max_active_clusters;
    if (max_active_clusters <= 0) {
      max_active_clusters = 0;
      CUTLASS_TRACE_HOST("  WARNING: Arguments do not include a valid max cluster count.\n"
          "  For optimal performance, populate the arguments KernelHardwareInfo struct with the max_active_clusters.");
    }
    else {
      CUTLASS_TRACE_HOST("to_underlying_arguments(): Setting persistent grid cluster count to " << max_active_clusters);
    }

    KernelHardwareInfo hw_info{args.hw_info.device_id, sm_count, max_active_clusters};

    // Calculate workspace pointers
    uint8_t* workspace_ptr = reinterpret_cast<uint8_t*>(workspace);
    size_t workspace_offset = 0;

    void* epilogue_workspace = workspace_ptr + workspace_offset;
    workspace_offset += CollectiveEpilogue::get_workspace_size(args.problem_shape, args.epilogue);
    workspace_offset = round_nearest(workspace_offset,  MinWorkspaceAlignment);

    void* scheduler_workspace = workspace_ptr + workspace_offset;
    workspace_offset += TileScheduler::template get_workspace_size<ProblemShape, ElementAccumulator>(
      args.scheduler, args.problem_shape, args.hw_info, NumMmaWarpGroups);
    workspace_offset = round_nearest(workspace_offset,  MinWorkspaceAlignment);

    void* mainloop_workspace = nullptr;
    constexpr uint32_t NumEpilogueSubTiles = CollectiveEpilogue::get_store_pipe_increment(TileShape{});

    return {
      args.mode,
      problem_shape,
      CollectiveMainloop::to_underlying_arguments(args.problem_shape, presum_m_a, presum_m_b, args.mainloop, mainloop_workspace),
      CollectiveEpilogue::to_underlying_arguments(args.problem_shape, args.epilogue, const_cast<ElementA*>(args.mainloop.ptr_A), postsum_m, epilogue_workspace),
      hw_info,
      TileScheduler::to_underlying_arguments(
        half_problem_shape_MNKL, TileShape{}, ClusterShape{}, hw_info, args.scheduler, scheduler_workspace, NumEpilogueSubTiles
      ),
      const_cast<ElementA*>(args.mainloop.ptr_A),
      const_cast<ElementB*>(args.mainloop.ptr_B),
      const_cast<ElementD*>(args.epilogue.ptr_D),
      presum_m_a, presum_m_b, postsum_m
    };
  }

  static bool
  can_implement(Arguments const& args) {
    bool implementable = (args.mode == GemmUniversalMode::kGemm) or
        (args.mode == GemmUniversalMode::kBatched && cute::rank(ProblemShape{}) == 4);
    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Arguments or Problem Shape don't meet the requirements.\n");
      return implementable;
    }
    implementable &= CollectiveMainloop::can_implement(args.problem_shape, args.mainloop);
    implementable &= CollectiveEpilogue::can_implement(args.problem_shape, args.epilogue);
    implementable &= TileScheduler::can_implement(args.scheduler);

    return implementable;
  }

  static size_t
  get_workspace_size(Arguments const& args) {
    size_t workspace_size = 0;

    workspace_size += CollectiveEpilogue::get_workspace_size(args.problem_shape, args.epilogue);
    workspace_size = round_nearest(workspace_size,  MinWorkspaceAlignment);

    workspace_size += TileScheduler::template get_workspace_size<ProblemShape, ElementAccumulator>(
      args.scheduler, args.problem_shape, args.hw_info, NumMmaWarpGroups);
    workspace_size = round_nearest(workspace_size,  MinWorkspaceAlignment);

    return workspace_size;
  }

  static cutlass::Status
  initialize_workspace(Arguments const& args, void* workspace = nullptr, cudaStream_t stream = nullptr,
    CudaHostAdapter* cuda_adapter = nullptr) {
    Status status = Status::kSuccess;
    uint8_t* workspace_ptr = reinterpret_cast<uint8_t*>(workspace);
    size_t workspace_offset = 0;
    static constexpr uint32_t NumEpilogueSubTiles = 1;
    static constexpr uint32_t NumAccumulatorMtxs = 1;

    status = CollectiveEpilogue::initialize_workspace(args.problem_shape, args.epilogue, workspace_ptr + workspace_offset, stream, cuda_adapter);
    workspace_offset += CollectiveEpilogue::get_workspace_size(args.problem_shape, args.epilogue);
    workspace_offset = round_nearest(workspace_offset,  MinWorkspaceAlignment);
    if (status != Status::kSuccess) {
      return status;
    }

    status = TileScheduler::template initialize_workspace<ProblemShape, ElementAccumulator>(
      args.scheduler, workspace_ptr + workspace_offset, stream, args.problem_shape, args.hw_info, NumMmaWarpGroups, NumEpilogueSubTiles, NumAccumulatorMtxs, cuda_adapter);
    workspace_offset += TileScheduler::template get_workspace_size<ProblemShape, ElementAccumulator>(
      args.scheduler, args.problem_shape, args.hw_info, NumMmaWarpGroups);
    workspace_offset = round_nearest(workspace_offset,  MinWorkspaceAlignment);
    if (status != Status::kSuccess) {
      return status;
    }

    return status;
  }

  // Computes the kernel launch grid shape based on runtime parameters
  static dim3
  get_grid_shape(Params const& params) {
    // Given device SM count, set grid size s.t. we do not launch more thread blocks than we can run concurrently
    TileSchedulerArguments args{};
    if constexpr (!std::is_const_v<decltype(args.max_swizzle_size)>) {
      args.max_swizzle_size = 1 << params.scheduler.log_swizzle_size_;
    }
    args.raster_order = params.scheduler.raster_order_ == TileScheduler::RasterOrder::AlongN ? TileScheduler::RasterOrderOptions::AlongN : TileScheduler::RasterOrderOptions::AlongM;
    return TileScheduler::get_grid_shape(params.scheduler, params.get_half_problem_shape(), TileShape{}, ClusterShape{}, params.hw_info, args);
  }

  static dim3
  get_block_shape() {
    return dim3(MaxThreadsPerBlock, 1, 1);
  }

  CUTLASS_DEVICE
  void
  operator()(Params const& params, char* smem_buf, char* in_accums = nullptr, dim3 base_block = {0,0,0}) {
    operator()(params, *reinterpret_cast<SharedStorage*>(smem_buf), in_accums, base_block); 
  }

  CUTLASS_DEVICE
  void
  operator()(Params const& params, SharedStorage& shared_storage, char* in_accums = nullptr, dim3 base_block = {0,0,0}) {
    using namespace cute;
    using X = Underscore;

#  if (defined(__CUDA_ARCH_FEAT_SM90_ALL) || defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL) ||\
      CUDA_ARCH_CONDITIONAL_OR_FAMILY(1200) || CUDA_ARCH_CONDITIONAL_OR_FAMILY(1210))
#    define ENABLE_SM90_KERNEL_LEVEL 1
#  endif

// Any Tensor Op MMA Atom in the ISA is arch conditional.
// #if ! defined(ENABLE_SM90_KERNEL_LEVEL)
//     MY_PRINTF("ERROR : Arch conditional MMA instruction used without targeting appropriate compute capability. Aborting.\n");
#if 1

    // Preconditions
    static_assert(cute::rank(StrideA{}) == 3, "StrideA must be rank-3: [M, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(cute::rank(StrideB{}) == 3, "StrideB must be rank-3: [N, K, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(cute::rank(StrideC{}) == 3, "StrideC must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");
    static_assert(cute::rank(StrideD{}) == 3, "StrideD must be rank-3: [M, N, L]. If batch mode is not needed, set L stride to Int<0>.");

    enum class WarpGroupRole {
      Producer = 0,
      Consumer0 = 1,
      Consumer1 = 2
    };
    enum class ProducerWarpRole {
      Mainloop = 0,
      Warp1 = 1,
      Epilogue = 2,
      MainloopAux = 3
    };

    // Kernel level shared memory storage
    // SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);
    
    int thread_idx = int(threadIdx.x);
    int lane_idx = canonical_lane_idx();
    int warp_idx = canonical_warp_idx_sync();
    int warp_idx_in_warp_group = warp_idx % NumWarpsPerWarpGroup;
    int warp_group_thread_idx = thread_idx % NumThreadsPerWarpGroup;
    auto warp_group_role = WarpGroupRole(canonical_warp_group_idx());
    auto producer_warp_role = ProducerWarpRole(warp_idx_in_warp_group);
    int lane_predicate = cute::elect_one_sync();
    uint32_t block_rank_in_cluster = cute::block_rank_in_cluster();

    // Issue Tma Descriptor Prefetch from a single thread
    if ((warp_idx == 0) && lane_predicate) {
      CollectiveMainloop::prefetch_tma_descriptors(params.mainloop);
      CollectiveEpilogue::prefetch_tma_descriptors(params.epilogue);
    }

    // TileScheduler pipeline
    typename TileSchedulerPipeline::Params scheduler_pipeline_params;
    typename TileSchedulerThrottlePipeline::Params scheduler_throttle_pipeline_params;
    if constexpr (IsSchedDynamicPersistent) { 
      if (warp_group_role == WarpGroupRole::Producer && producer_warp_role == ProducerWarpRole::Warp1) {
        scheduler_pipeline_params.role = TileSchedulerPipeline::ThreadCategory::ProducerConsumer;
      }
      else {
        scheduler_pipeline_params.role = TileSchedulerPipeline::ThreadCategory::Consumer;
      }
      scheduler_pipeline_params.producer_blockid = 0;
      scheduler_pipeline_params.producer_arv_count = 1;
      scheduler_pipeline_params.consumer_arv_count = NumSchedThreads + NumMainloopLoadThreads + NumMMAThreads;

      CollectiveEpilogue collective_epilogue(params.epilogue, shared_storage.tensors.extra_storage.epilogue);
      bool is_epi_load_needed = collective_epilogue.is_producer_load_needed();

      if (is_epi_load_needed) {
        scheduler_pipeline_params.consumer_arv_count += NumEpilogueLoadThreads;
      } 
      scheduler_pipeline_params.transaction_bytes = sizeof(typename TileScheduler::CLCResponse);

      scheduler_throttle_pipeline_params.producer_arv_count = NumMainloopLoadThreads;
      scheduler_throttle_pipeline_params.consumer_arv_count = NumSchedThreads;
      scheduler_throttle_pipeline_params.dst_blockid = 0;
      if (warp_group_role == WarpGroupRole::Producer &&
          producer_warp_role == ProducerWarpRole::Warp1) {
        scheduler_throttle_pipeline_params.role =
            TileSchedulerThrottlePipeline::ThreadCategory::Consumer;
      }
      // set role when it is for DMA warp in Mainloop
      else if (warp_group_role == WarpGroupRole::Producer &&
               producer_warp_role == ProducerWarpRole::Mainloop) {
        scheduler_throttle_pipeline_params.role =
            TileSchedulerThrottlePipeline::ThreadCategory::Producer;
      }
    }
    TileSchedulerPipeline scheduler_pipeline(shared_storage.scheduler.pipeline(), scheduler_pipeline_params);
    TileSchedulerPipelineState scheduler_pipe_consumer_state;

    TileSchedulerThrottlePipeline scheduler_throttle_pipeline(shared_storage.scheduler.throttle_pipeline(), scheduler_throttle_pipeline_params);
    TileSchedulerThrottlePipelineState scheduler_pipe_throttle_consumer_state;
    TileSchedulerThrottlePipelineState scheduler_pipe_throttle_producer_state = cutlass::make_producer_start_state<TileSchedulerThrottlePipeline>();

    // Mainloop Load pipeline
    using MainloopPipeline = typename CollectiveMainloop::MainloopPipeline;
    typename MainloopPipeline::Params mainloop_pipeline_params;
    if (warp_group_role == WarpGroupRole::Producer && (producer_warp_role == ProducerWarpRole::Mainloop 
        || producer_warp_role == ProducerWarpRole::MainloopAux)) {
      mainloop_pipeline_params.role = MainloopPipeline::ThreadCategory::Producer;
    }
    if (warp_group_role == WarpGroupRole::Consumer0 || warp_group_role == WarpGroupRole::Consumer1) {
      mainloop_pipeline_params.role = MainloopPipeline::ThreadCategory::Consumer;
    }
    mainloop_pipeline_params.is_leader = warp_group_thread_idx == 0;
    mainloop_pipeline_params.num_consumers = NumThreadsPerWarpGroup;
    mainloop_pipeline_params.num_producers = NumProducerThreads;
    mainloop_pipeline_params.transaction_bytes = params.mainloop.tma_transaction_bytes;
    MainloopPipeline mainloop_pipeline(shared_storage.pipelines.mainloop, mainloop_pipeline_params, ClusterShape{});

    // using PresumLoadPipeline = typename CollectiveMainloop::PresumLoadPipeline;
    // typename PresumLoadPipeline::Params presum_load_pipeline_params;
    // if (warp_group_role == WarpGroupRole::Producer && (producer_warp_role == ProducerWarpRole::Mainloop 
    //     || producer_warp_role == ProducerWarpRole::MainloopAux)) {
    //   presum_load_pipeline_params.role = PresumLoadPipeline::ThreadCategory::Producer;
    // }
    // if (warp_group_role == WarpGroupRole::Consumer0 || warp_group_role == WarpGroupRole::Consumer1) {
    //   presum_load_pipeline_params.role = PresumLoadPipeline::ThreadCategory::Consumer;
    // }
    // presum_load_pipeline_params.is_leader = warp_group_thread_idx == 0;
    // presum_load_pipeline_params.num_consumers = NumThreadsPerWarpGroup;
    // presum_load_pipeline_params.num_producers = NumProducerThreads;
    // presum_load_pipeline_params.transaction_bytes = params.mainloop.tma_transaction_bytes_presum_mk;
    // PresumLoadPipeline presum_load_pipeline(shared_storage.pipelines.presum_load, presum_load_pipeline_params, Shape<_1,_1,_1>{});

    // Epilogue Load pipeline
    using EpiLoadPipeline = typename CollectiveEpilogue::LoadPipeline;
    typename EpiLoadPipeline::Params epi_load_pipeline_params;
    if (warp_group_role == WarpGroupRole::Producer && producer_warp_role == ProducerWarpRole::Epilogue) {
      epi_load_pipeline_params.role = EpiLoadPipeline::ThreadCategory::Producer;
    }
    if (warp_group_role == WarpGroupRole::Consumer0 || warp_group_role == WarpGroupRole::Consumer1) {
      epi_load_pipeline_params.role = EpiLoadPipeline::ThreadCategory::Consumer;
    }
    epi_load_pipeline_params.dst_blockid = cute::block_rank_in_cluster();
    epi_load_pipeline_params.producer_arv_count = NumThreadsPerWarp;
    epi_load_pipeline_params.consumer_arv_count = NumThreadsPerWarpGroup;
    if constexpr (CollectiveEpilogue::RequiresTransactionBytes) {
      epi_load_pipeline_params.transaction_bytes = params.epilogue.tma_transaction_bytes;
    }
    EpiLoadPipeline epi_load_pipeline(shared_storage.pipelines.epi_load, epi_load_pipeline_params);

    // Epilogue Store pipeline
    using EpiStorePipeline = typename CollectiveEpilogue::StorePipeline;
    typename EpiStorePipeline::Params epi_store_pipeline_params;
    epi_store_pipeline_params.always_wait = true;
    EpiStorePipeline epi_store_pipeline(epi_store_pipeline_params);

    typename LoadWarpOrderBarrier::Params params_load_order_barrier;
    params_load_order_barrier.group_id = producer_warp_role == ProducerWarpRole::Mainloop ? 0 : 1;
    params_load_order_barrier.group_size = NumThreadsPerWarp;
    LoadWarpOrderBarrier load_order_barrier(shared_storage.pipelines.load_order, params_load_order_barrier);

    typename MathWarpGroupOrderBarrier::Params params_math_wg_order_barrier;
    // DMA Load WG will not participate in these Ordered Barrier syncs
    params_math_wg_order_barrier.group_id = canonical_warp_group_idx() - static_cast<int>(WarpGroupRole::Consumer0);
    params_math_wg_order_barrier.group_size = NumThreadsPerWarpGroup; // Number of threads / participants in a group
    MathWarpGroupOrderBarrier math_wg_order_barrier(shared_storage.pipelines.math_wg_order, params_math_wg_order_barrier);

    // Initialize starting pipeline states for the collectives
    // Epilogue store pipe is producer-only (consumer is TMA unit, waits via scoreboarding)
    typename CollectiveMainloop::PipelineState mainloop_pipe_consumer_state;
    typename CollectiveEpilogue::LoadPipelineState epi_load_pipe_consumer_state;

    // For the DMA Load (producer) we start with an opposite phase
    // i.e., we skip all waits since we know that the buffer is indeed empty
    PipelineState mainloop_pipe_producer_state = cutlass::make_producer_start_state<MainloopPipeline>();
    PipelineState epi_load_pipe_producer_state = cutlass::make_producer_start_state<EpiLoadPipeline>();
    PipelineState epi_store_pipe_producer_state = cutlass::make_producer_start_state<EpiStorePipeline>();

    using RWCTypes = typename StrassenMiGroup::RWCTypes;

    auto cluster_wait_fn = [&] () {
      // We need this to guarantee that the Pipeline init is visible
      // To all producers and consumer thread blocks in the Cluster
      if constexpr (size(ClusterShape{}) > 1) {
        cute::cluster_arrive_relaxed();
        return [] () { cute::cluster_wait(); };
      }
      else {
        __syncthreads();
        return [] () {}; // do nothing
      }
    } ();

    // Separate out problem shape for convenience
    // Optionally append 1s until problem shape is rank-4 in case it is only rank-3 (MNK)
    auto problem_shape_MNKL = append<4>(params.problem_shape, Int<1>{});
    auto half_problem_shape_MNKL = append<4>(params.get_half_problem_shape(), Int<1>{});

    // Get the appropriate blocks for this thread block -- potential for thread block locality
    TiledMma tiled_mma;
    auto blk_shape = TileShape{};                                                                // (BLK_M,BLK_N,BLK_K)

    // In a warp specialized kernel, collectives expose data movement and compute operations separately
    CollectiveMainloop collective_mainloop;
    CollectiveEpilogue collective_epilogue(params.epilogue, shared_storage.tensors.extra_storage.epilogue);

    // Prepare and partition the input tensors. Expects a tuple of tensors where:
    // get<0>(load_inputs) is the tma tensor A after local tiling so that it has shape (BLK_M,BLK_K,m,k,l)
    // get<1>(load_inputs) is the tma tensor B after local tiling so that it has shape (BLK_N,BLK_K,n,k,l)
    auto all_inputs = collective_mainloop.load_init(problem_shape_MNKL, half_problem_shape_MNKL, params.mainloop);
    auto all_presumld_inputs = collective_mainloop.presumld_inputs(problem_shape_MNKL, half_problem_shape_MNKL, params.mainloop);
    auto load_inputs = collective_mainloop.get_inputs(all_inputs);
    auto load_inputs2 = collective_mainloop.get_inputs(all_inputs, 1);
    static_assert(cute::tuple_size_v<decltype(load_inputs)> >= 2, "Output of load_init must have at least two elements (A, B)");

    // Extract out partitioned A and B.
    Tensor gA_mkl = get<0>(load_inputs);
    Tensor gB_nkl = get<1>(load_inputs);

    Tensor gA2_mkl = get<0>(load_inputs2);
    Tensor gB2_nkl = get<1>(load_inputs2);

    bool is_fused = StrassenMiGroup::hasM0() && StrassenMiGroup::hasM1();

    // Get pipeline stage increments from tensor shapes
    auto k_tile_count = size<3>(gA_mkl);
    if (StrassenMiGroup::hasM0() || StrassenMiGroup::hasM1() ||
        StrassenMiGroup::hasM6())
      k_tile_count = k_tile_count/2;

    auto c_tile_count = CollectiveEpilogue::get_load_pipe_increment(blk_shape);
    auto d_tile_count = CollectiveEpilogue::get_store_pipe_increment(blk_shape);

    TileScheduler scheduler{params.scheduler};
    if constexpr (IsSchedDynamicPersistent) {
      scheduler.set_data_ptr(shared_storage.scheduler.data());
    }


    if (warp_group_role == WarpGroupRole::Consumer1) {
      if (StrassenMiGroup::numMs() == 1) {
        if constexpr (not IsSchedDynamicPersistent) {
          // Advance 2nd Math WG to the next work tile for the startup
          scheduler.template advance_to_next_work<StrassenMiGroup> ();
        }
      }

      // Advance 2nd Math WG pipeline states to the end of 1st Math WG
      mainloop_pipe_consumer_state.advance(k_tile_count);
      epi_load_pipe_consumer_state.advance(c_tile_count);
      epi_store_pipe_producer_state.advance(d_tile_count);
    }

    auto work_tile_info = scheduler.initial_work_tile_info(ClusterShape{});

    // Wait for all thread blocks in the Cluster
    cluster_wait_fn();

    if (warp_group_role == WarpGroupRole::Producer) {
      cutlass::arch::warpgroup_reg_dealloc<LoadRegisterRequirement>();
    if (threadIdx.x % 32 == 0 && blockIdx.x == 0 && blockIdx.y == 0)
      MY_PRINTF("806 %d: %d %d\n", threadIdx.x, blockIdx.x, blockIdx.y);
      // Scheduler Producer Warp
      if (producer_warp_role == ProducerWarpRole::Warp1) {
        if constexpr (IsSchedDynamicPersistent) {
          //IsSchedDynamicPersistent is false, so this is dead code
          bool requires_clc_query = true;
          TileSchedulerPipelineState scheduler_pipe_producer_state = cutlass::make_producer_start_state<TileSchedulerPipeline>();

          while (work_tile_info.is_valid()) {
            
            if (requires_clc_query) {

              // Throttle CLC query to mitigate workload imbalance caused by skews among persistent workers.
              scheduler_throttle_pipeline.consumer_wait(scheduler_pipe_throttle_consumer_state);
              scheduler_throttle_pipeline.consumer_release(scheduler_pipe_throttle_consumer_state);
              ++scheduler_pipe_throttle_consumer_state;

              // Query next work tile
              scheduler_pipe_producer_state = scheduler.template advance_to_next_work<StrassenMiGroup>(scheduler_pipeline, scheduler_pipe_producer_state);
            }

            // Fetch next work tile
            auto [next_work_tile_info, increment_pipe] = 
              scheduler.fetch_next_work(
                  work_tile_info, scheduler_pipeline, scheduler_pipe_consumer_state);
            
            work_tile_info = next_work_tile_info;
            requires_clc_query = increment_pipe;
            if (increment_pipe) {
              ++scheduler_pipe_consumer_state;
            }
          }

          // Terminal condition - if work_tile_info is end-of-grid, produce an extra invalid tile
          scheduler_pipeline.producer_acquire(scheduler_pipe_producer_state);
          scheduler.store_invalid_response(scheduler_pipe_producer_state); // Push invalid tile to smem
          scheduler_pipeline.producer_commit(scheduler_pipe_producer_state); // Manual completion of transaction
          ++scheduler_pipe_producer_state;

          auto [next_work_tile_info, increment_pipe] = 
            scheduler.fetch_next_work(
                work_tile_info, scheduler_pipeline, scheduler_pipe_consumer_state);

          scheduler_pipeline.producer_tail(scheduler_pipe_producer_state);
        } 
      } // Scheduler Producer Warp End  
      else
      
      // Mainloop Producer Warp
      if (producer_warp_role == ProducerWarpRole::Mainloop) {
        // Ensure that the prefetched kernel does not touch
        // unflushed global memory prior to this instruction
        cutlass::arch::wait_on_dependent_grids();
        bool do_load_order_arrive = true;
        bool requires_clc_query = true;
        int sub_m_idx = 0;
        if (threadIdx.x % 32 == 0 && blockIdx.x == 0 && blockIdx.y == 0)
          MY_PRINTF("863 %d: %d %d\n", threadIdx.x, blockIdx.x, blockIdx.y);
        while (work_tile_info.is_valid()) {
          // Compute m_coord, n_coord, l_coord with the post-tiled m-shape and n-shape
          auto m_coord = idx2crd(work_tile_info.M_idx, shape<2>(gA_mkl));
          auto n_coord = idx2crd(work_tile_info.N_idx, shape<2>(gB_nkl));
          auto l_coord = idx2crd(work_tile_info.L_idx, shape<4>(gB_nkl));

          auto blk_coord = make_coord(m_coord, n_coord, _, l_coord);

          auto k_tile_iter  = cute::make_coord_iterator(shape<3>(gA_mkl));
          sub_m_idx = 0;

          if (requires_clc_query) {
            scheduler_throttle_pipeline.producer_acquire(scheduler_pipe_throttle_producer_state);
            scheduler_throttle_pipeline.producer_commit(scheduler_pipe_throttle_producer_state);
            ++scheduler_pipe_throttle_producer_state;
          }

          collective_mainloop.load(
            params.mainloop, half_problem_shape_MNKL,
            mainloop_pipeline,
            mainloop_pipe_producer_state,
            load_inputs,
            blk_coord, sub_m_idx,
            k_tile_iter,
            k_tile_count,
            lane_idx,
            block_rank_in_cluster,
            shared_storage.tensors.mainloop,
            all_presumld_inputs,
            shared_storage.tensors.extra_storage.presum_tensors
          );
          // Update starting pipeline state for the next tile
          mainloop_pipe_producer_state.advance(k_tile_count);

          // Signal for the epilogue load warp to begin
          if (do_load_order_arrive) {
            load_order_barrier.arrive();
            do_load_order_arrive = false;
          }
          if (StrassenMiGroup::numMs() > 1) {
            sub_m_idx = 1;
            k_tile_iter.coord += (is_fused) ? k_tile_count : 0;
            
            collective_mainloop.load(
              params.mainloop, half_problem_shape_MNKL,
              mainloop_pipeline,
              mainloop_pipe_producer_state,
              (is_fused) ? load_inputs : load_inputs2,
              blk_coord, sub_m_idx,
              k_tile_iter,
              k_tile_count,
              lane_idx,
              block_rank_in_cluster,
              shared_storage.tensors.mainloop,
              all_presumld_inputs,
              shared_storage.tensors.extra_storage.presum_tensors
            );
            // Update starting pipeline state for the next tile
            mainloop_pipe_producer_state.advance(k_tile_count);
          }

          if constexpr (IsSchedDynamicPersistent) {  
            // Get next work tile
            auto [next_work_tile_info, increment_pipe] =
              scheduler.fetch_next_work(
                  work_tile_info, scheduler_pipeline, scheduler_pipe_consumer_state);

            work_tile_info = next_work_tile_info;
            requires_clc_query = increment_pipe;
            if (increment_pipe) {
              ++scheduler_pipe_consumer_state;
            }
          }
          else {
          // Get next work tile
          scheduler.template advance_to_next_work<StrassenMiGroup>();
          work_tile_info = scheduler.get_current_work();
          }
        } // Scheduler work fetch loop

        // Make sure all Consumer Warp Groups have been waited upon
        collective_mainloop.load_tail(mainloop_pipeline, mainloop_pipe_producer_state);

        if constexpr (IsSchedDynamicPersistent) {  
          auto [next_work_tile_info, increment_pipe] = 
            scheduler.fetch_next_work(
                work_tile_info, scheduler_pipeline, scheduler_pipe_consumer_state);
        }
        
      } // Mainloop Producer Warp End

      else if (producer_warp_role == ProducerWarpRole::MainloopAux) {
        if (threadIdx.x % 32 == 0 && blockIdx.x == 0 && blockIdx.y == 0)
          MY_PRINTF("957 %d: %d %d\n", threadIdx.x, blockIdx.x, blockIdx.y);
        if constexpr (IsMainloopAuxiliaryLoadNeeded) {
          // Ensure that the prefetched kernel does not touch
          // unflushed global memory prior to this instruction
          cutlass::arch::wait_on_dependent_grids();
          while (work_tile_info.is_valid()) {
            // Compute m_coord, n_coord, l_coord with the post-tiled m-shape and n-shape
            auto m_coord = idx2crd(work_tile_info.M_idx, shape<2>(gA_mkl));
            auto n_coord = idx2crd(work_tile_info.N_idx, shape<2>(gB_nkl));
            auto l_coord = idx2crd(work_tile_info.L_idx, shape<4>(gB_nkl));
            auto blk_coord = make_coord(m_coord, n_coord, _, l_coord);

            auto k_tile_iter = cute::make_coord_iterator(shape<3>(gA_mkl));
            collective_mainloop.load_auxiliary(
              params.mainloop,
              mainloop_pipeline,
              mainloop_pipe_producer_state,
              load_inputs,
              blk_coord,
              k_tile_iter, k_tile_count,
              lane_idx,
              block_rank_in_cluster,
              shared_storage.tensors.mainloop
            );
            // Update starting pipeline state for the next tile
            mainloop_pipe_producer_state.advance(k_tile_count);

            scheduler.template advance_to_next_work<StrassenMiGroup>();
            work_tile_info = scheduler.get_current_work();
          } // Scheduler work fetch loop

          // Make sure all Consumer Warp Groups have been waited upon
          collective_mainloop.load_tail(mainloop_pipeline, mainloop_pipe_producer_state);

          if constexpr (IsSchedDynamicPersistent) {  
            auto [next_work_tile_info, increment_pipe] = 
              scheduler.fetch_next_work(
                work_tile_info,
                scheduler_pipeline,
                scheduler_pipe_consumer_state
              );
          }
          
        }
      }

      // Epilogue Producer Warp
      else if (producer_warp_role == ProducerWarpRole::Epilogue &&
               collective_epilogue.is_producer_load_needed()) {
        if (threadIdx.x % 32 == 0 && blockIdx.x == 0 && blockIdx.y == 0)
          MY_PRINTF("EpiWarp 1007 %d: %d %d\n", threadIdx.x, blockIdx.x, blockIdx.y);
        // Ensure that the prefetched kernel does not touch
        // unflushed global memory prior to this instruction
        cutlass::arch::wait_on_dependent_grids();
        bool do_load_order_wait = true;
        while (work_tile_info.is_valid()) {
          if (do_load_order_wait) {
            load_order_barrier.wait();
            do_load_order_wait = false;
          }

          // Compute m_coord, n_coord, l_coord with the post-tiled m-shape and n-shape
          auto m_coord = idx2crd(work_tile_info.M_idx, shape<2>(gA_mkl));
          auto n_coord = idx2crd(work_tile_info.N_idx, shape<2>(gB_nkl));
          auto l_coord = idx2crd(work_tile_info.L_idx, shape<4>(gB_nkl));
          auto blk_coord = make_coord(m_coord, n_coord, _, l_coord);

          #pragma unroll (StrassenMiGroup::numMs())
          for (int fused_mi = 0; fused_mi < StrassenMiGroup::numMs(); fused_mi++) {
            #pragma unroll 4
            for (int c = 0; c < 4; c++) {
              const MmaStrassen::PostsumOp postsum_global_dest = RWCTypes::PostsumGlobalDestByOutputIndex(c);
              const MmaStrassen::PostsumOp postsum_shared_dest = RWCTypes::PostsumSharedDestByOutputIndex(c);

              uint mi = StrassenMiGroup::getMi(fused_mi);
              int misign = RWCTypes::MiSignByOutputIndex(c, mi);

              if (misign == 0 || (!postsum_shared_dest.valid() && !postsum_global_dest.valid())) continue;

              int read_c = 0;
              MmaStrassen::PostsumOp postsum_srcs[4] = {MmaStrassen::PostsumOp(), MmaStrassen::PostsumOp(), MmaStrassen::PostsumOp(), MmaStrassen::PostsumOp()};
              int postsum_src_len = 0;
              #pragma unroll 4
              for (read_c = 0; read_c < 4; read_c++) {
                auto postsum_src = RWCTypes::PostsumSrcByOutputIndex(c, read_c);
                if (postsum_src.valid() && postsum_src.is_mem_global() && postsum_src.is_layout_interim()) {
                  postsum_srcs[postsum_src_len++] = postsum_src;
                }
              }

              if (postsum_src_len > 0) {
                if (false) {
                  epi_load_pipe_producer_state =
                  collective_epilogue.load(//TODO: Give postsum as argument
                    epi_load_pipeline,
                    epi_load_pipe_producer_state,
                    problem_shape_MNKL,
                    blk_shape,
                    blk_coord,
                    tiled_mma,
                    lane_idx,
                    shared_storage.tensors.extra_storage.epilogue,
                    shared_storage.tensors.extra_storage.epilogue2
                  );
                } else {
                  epi_load_pipe_producer_state =
                  collective_epilogue.load_m0(//TODO: Give postsum as argument
                    epi_load_pipeline,
                    epi_load_pipe_producer_state,
                    problem_shape_MNKL,
                    blk_shape,
                    blk_coord, fused_mi,
                    tiled_mma,
                    lane_idx,
                    shared_storage.tensors.extra_storage.epilogue,
                    shared_storage.tensors.extra_storage.epilogue2,
                    postsum_srcs
                  );
                }
                if (fused_mi < StrassenMiGroup::numMs() - 1)
                  asm volatile("bar.cta.sync %0, %1;" : : "r"(9), "r"(NumThreadsPerWarpGroup+32));
              }
            }
          }

          if constexpr (IsSchedDynamicPersistent) {  
            // Get next work tile
            auto [next_work_tile_info, increment_pipe] = 
              scheduler.fetch_next_work(
                  work_tile_info, scheduler_pipeline, scheduler_pipe_consumer_state);

            work_tile_info = next_work_tile_info;
            if (increment_pipe) {
              ++scheduler_pipe_consumer_state;
            }
          }
          else {
          // Get next work tile
          scheduler.template advance_to_next_work<StrassenMiGroup>();
          work_tile_info = scheduler.get_current_work();
          }
        } // Scheduler work fetch loop

        // Make sure all Consumer Warp Groups have been waited upon
        collective_epilogue.load_tail(epi_load_pipeline, epi_load_pipe_producer_state);
        if constexpr (IsSchedDynamicPersistent) {  
          auto [next_work_tile_info, increment_pipe] = 
            scheduler.fetch_next_work(
                work_tile_info, scheduler_pipeline, scheduler_pipe_consumer_state);
        }
      } // Epilogue Producer Warp End
    } // Producer Warp Group End

    else if (warp_group_role == WarpGroupRole::Consumer0 || warp_group_role == WarpGroupRole::Consumer1) {
      cutlass::arch::warpgroup_reg_alloc<MmaRegisterRequirement>();

      #ifdef CUTLASS_ENABLE_GDC_FOR_SM90
      // It is possible to have work tiles start off invalid,
      // so we have to check that first.
      if (not work_tile_info.is_valid()) {
        // Hint on an early release of global memory resources.
        // The timing of calling this function only influences performance,
        // not functional correctness.
        cutlass::arch::launch_dependent_grids();

        return;
      }
      #endif
      
      if constexpr (IsSchedDynamicPersistent) {
        // Consumer0's initial tile is static. It starts consuming the 2nd tile.
        if (warp_group_role == WarpGroupRole::Consumer0) {
            ++scheduler_pipe_consumer_state;
        } 

        if (warp_group_role == WarpGroupRole::Consumer1) {
          // Get next work tile
          auto [next_work_tile_info, increment_pipe] = 
            scheduler.fetch_next_work(
                work_tile_info, scheduler_pipeline, scheduler_pipe_consumer_state);

          work_tile_info = next_work_tile_info;
          if (increment_pipe) {
            ++scheduler_pipe_consumer_state;
            ++scheduler_pipe_consumer_state;
          }
        }
      }
        if (warp_group_thread_idx == 0 && blockIdx.x == 0 && blockIdx.y == 0)
          MY_PRINTF("1127\n");
      while (work_tile_info.is_valid()) {
        // Compute m_coord, n_coord, l_coord with the post-tiled m-shape and n-shape
        auto m_coord = idx2crd(work_tile_info.M_idx, shape<2>(gA_mkl));
        auto n_coord = idx2crd(work_tile_info.N_idx, shape<2>(gB_nkl));
        auto l_coord = idx2crd(work_tile_info.L_idx, shape<4>(gB_nkl));
        auto sub_m_idx = (StrassenMiGroup::numMs() == 2) ? canonical_warp_group_idx() - 1 : 0;
        if (sub_m_idx == 0 || sub_m_idx == 1) {} else CUTE_GCC_UNREACHABLE;
        auto blk_coord = make_coord(m_coord, n_coord, _, l_coord);

        // Allocate the accumulators for the (M,N) blk_shape
        Tensor accumulators = partition_fragment_C(tiled_mma, take<0,2>(blk_shape));               // (MMA,MMA_M,MMA_N)
        // Order two Math WG's MMA one after the other, helps hide Epilogue
        math_wg_order_barrier.wait();
        if (warp_group_thread_idx == 0 && blockIdx.x == 0 && blockIdx.y == 0)
          MY_PRINTF("1141 %d: %d %d\n", threadIdx.x, m_coord, n_coord);
        if ((is_fused /*|| IsFusedM2M3*/) && sub_m_idx == 1) {
          NumericArrayConverter<float, ElementD, 8> converter;
          //Read from this shared memory
          cutlass::Array<float, 8>* arrs = (cutlass::Array<float, 8>*)&accumulators;
          auto sh_ptr = ((cutlass::Array<ElementD, 8>*)shared_storage.tensors.extra_storage.epilogue.collective.smem_C.begin());
          const uint SHMEM_SIZE = 4096/(NumThreadsPerWarpGroup*sizeof(ElementD));

          for (int i = 0; i < accumulators.size(); i += 2*SHMEM_SIZE) {
            asm volatile("bar.cta.sync %0, %1;" : : "r"(7), "r"(256));
            for (int e = 0; e < SHMEM_SIZE; e += 8) {
              arrs[(i + e)/8] = converter(sh_ptr[warp_group_thread_idx + 128*(e/8)]);
            }
            asm volatile("bar.cta.sync %0, %1;" : : "r"(8), "r"(256));
            if (i + SHMEM_SIZE < accumulators.size()) {
              for (int e = 0; e < SHMEM_SIZE; e += 8) {
                arrs[(i + SHMEM_SIZE + e)/8] = converter(sh_ptr[warp_group_thread_idx + 128*(SHMEM_SIZE+e)/8]);
              }
            }
          }
          asm volatile("bar.cta.sync %0, %1;" : : "r"(9), "r"(256));
        }

        using EpilogueTile = typename CollectiveEpilogue::EpilogueTile;

        if (is_fused) {
          if (sub_m_idx == 0)
            collective_mainloop.mma(
              blk_coord, 0, problem_shape_MNKL, half_problem_shape_MNKL,
              mainloop_pipeline,
              mainloop_pipe_consumer_state,
              accumulators,
              k_tile_count,
              warp_group_thread_idx,
              shared_storage.tensors.mainloop,
              shared_storage.tensors.extra_storage.presum_tensors,
              shared_storage.tensors.extra_storage.presum_tensors2,
              params.mainloop
            );
          else
            collective_mainloop.mma(
              blk_coord, 1, problem_shape_MNKL, half_problem_shape_MNKL,
              mainloop_pipeline,
              mainloop_pipe_consumer_state,
              accumulators,
              k_tile_count,
              warp_group_thread_idx,
              shared_storage.tensors.mainloop,
              shared_storage.tensors.extra_storage.presum_tensors,
              shared_storage.tensors.extra_storage.presum_tensors2,
              params.mainloop
            );
        } else {
          collective_mainloop.mma(
            blk_coord, sub_m_idx, problem_shape_MNKL, half_problem_shape_MNKL,
            mainloop_pipeline,
            mainloop_pipe_consumer_state,
            accumulators,
            k_tile_count,
            warp_group_thread_idx,
            shared_storage.tensors.mainloop,
            shared_storage.tensors.extra_storage.presum_tensors,
            shared_storage.tensors.extra_storage.presum_tensors2,
            params.mainloop
          );
        }

        // Cue for next Math WG's MMA to start
        // Make sure the math instructions are done and free buffers before entering the epilogue

        // Next Mi=1 can read from this shared memory 
        math_wg_order_barrier.arrive();
        collective_mainloop.mma_tail(
          mainloop_pipeline,
          mainloop_pipe_consumer_state,
          k_tile_count
        );

        if (StrassenMiGroup::hasM2() && warp_group_thread_idx == 0 && blockIdx.x == 0 && blockIdx.y == 0)
          MY_PRINTF("1163 %d: %d %d ; %f\n", threadIdx.x, m_coord, n_coord, float(accumulators[0]));
        if ((is_fused /*|| IsFusedM2M3*/) && sub_m_idx == 0) {
          NumericArrayConverter<ElementD, float, 8> converter;
          cutlass::Array<float, 8>* arrs = (cutlass::Array<float, 8>*)&accumulators;
          auto sh_ptr = ((cutlass::Array<ElementD, 8>*)shared_storage.tensors.extra_storage.epilogue.collective.smem_C.begin());

          const uint SHMEM_SIZE = 4096/(NumThreadsPerWarpGroup*sizeof(ElementD)); 
          for (int i = 0; i < accumulators.size(); i += 2*SHMEM_SIZE) {
            for (int e = 0; e < SHMEM_SIZE; e += 8) {
              cutlass::Array<ElementD, 8> arr = converter(arrs[(i+e)/8]);
              sh_ptr[warp_group_thread_idx + NumThreadsPerWarpGroup*(e/8)] = arr;
            }
            asm volatile("bar.cta.sync %0, %1;" : : "r"(7), "r"(256));
            if (i + SHMEM_SIZE < accumulators.size()) {
              for (int e = 0; e < SHMEM_SIZE; e += 8) {
                cutlass::Array<ElementD, 8> arr = converter(arrs[(i+SHMEM_SIZE+e)/8]);
                sh_ptr[warp_group_thread_idx + 128*(SHMEM_SIZE+e)/8] = arr;
              }
            }
            asm volatile("bar.cta.sync %0, %1;" : : "r"(8), "r"(256));
          }
          asm volatile("bar.cta.sync %0, %1;" : : "r"(9), "r"(256));
        }

        // if (sub_m_idx == 1)
          // Update starting mainloop pipeline state for the next tile
          mainloop_pipe_consumer_state.advance(k_tile_count * NumMmaWarpGroups);

        #ifdef CUTLASS_ENABLE_GDC_FOR_SM90
        if (scheduler.is_last_tile(work_tile_info, 1)) {//NumMmaWarpGroups
          // Hint on an early release of global memory resources.
          // The timing of calling this function only influences performance,
          // not functional correctness.
          cutlass::arch::launch_dependent_grids();

        }
        #endif
        if (warp_group_thread_idx == 0 && blockIdx.x == 0 && blockIdx.y == 0)
          MY_PRINTF("1211 %d: %d %d\n", threadIdx.x, m_coord, n_coord);

        // Order two Math WG's Epilogue one after the other
        math_wg_order_barrier.wait();

        if (StrassenMiGroup::numMs() == 2 && sub_m_idx == 1) {
          uint fused_mi = sub_m_idx;
          #pragma unroll 4
          for (int c = 0; c < 4; c++) {
            const MmaStrassen::PostsumOp postsum_global_dest = RWCTypes::PostsumGlobalDestByOutputIndex(c);
            const MmaStrassen::PostsumOp postsum_shared_dest = RWCTypes::PostsumSharedDestByOutputIndex(c);
            uint mi = StrassenMiGroup::getMi(fused_mi);
            int misign = RWCTypes::MiSignByOutputIndex(c, mi);

            if (misign == 0 || (!postsum_shared_dest.valid() && !postsum_global_dest.valid())) continue;

            int read_c = 0;
            MmaStrassen::PostsumOp postsum_src;
            #pragma unroll 4
            for (read_c = 0; read_c < 4; read_c++) {
              postsum_src = RWCTypes::PostsumSrcByOutputIndex(c, read_c);
              if (postsum_src.valid() && postsum_src.is_mem_shared() && postsum_src.is_layout_interim()) {
                break;
              }
            }

            if (postsum_src.valid() && postsum_src.is_layout_interim()) {
              NumericArrayConverter<float, ElementD, 8> converter;
              //Read from this shared memory
              cutlass::Array<float, 8>* arrs = (cutlass::Array<float, 8>*)&accumulators;
              auto sh_ptr = ((cutlass::Array<ElementD, 8>*)shared_storage.tensors.extra_storage.epilogue.collective.smem_C.begin());
              const uint SHMEM_SIZE = (size<0>(TileShape{}) * size<1>(TileShape{}))/NumThreadsPerWarpGroup;

              for (int i = 0; i < accumulators.size(); i += 8) {
                arrs[i/8] = arrs[i/8] + converter(sh_ptr[warp_group_thread_idx + NumThreadsPerWarpGroup*(i/8)]);
              }
              asm volatile("bar.cta.sync %0, %1;" : : "r"(9), "r"(128+32));
            }
          }
        }

        decltype(epi_load_pipe_consumer_state) epi_load_pipe_consumer_state_next;
        decltype(epi_store_pipe_producer_state) epi_store_pipe_producer_state_next;
        decltype(epi_load_pipe_consumer_state) epi_load_pipe_consumer_state_next_;
        decltype(epi_store_pipe_producer_state) epi_store_pipe_producer_state_next_;
        
        bool has_global_src = false;
        uint fused_mi = sub_m_idx;

        bool any_global_dst_final = false;
        bool any_global_dst_valid = false;

        #pragma unroll 4
        for (int c = 0; c < 4; c++) {
          const MmaStrassen::PostsumOp postsum_global_dest = RWCTypes::PostsumGlobalDestByOutputIndex(c);
          const MmaStrassen::PostsumOp postsum_shared_dest = RWCTypes::PostsumSharedDestByOutputIndex(c);

          //TODO: This adds a stack size
          uint mi = StrassenMiGroup::getMi(fused_mi);
          int misign = RWCTypes::MiSignByOutputIndex(c, mi);

          if (misign == 0 || (!postsum_global_dest.valid())) continue;
          
          any_global_dst_final = any_global_dst_final || postsum_global_dest.is_layout_final();
          any_global_dst_valid = any_global_dst_valid || postsum_global_dest.valid();

          #pragma unroll 4
          for (int read_c = 0; read_c < 4; read_c++) {
            auto postsum_src = RWCTypes::PostsumSrcByOutputIndex(c, read_c);
            if (postsum_src.valid() && postsum_src.is_mem_global()) {
              has_global_src = true;
            }
          }
        }

        if (any_global_dst_valid) {
        if (!any_global_dst_final) {
          auto ret = collective_epilogue.store_m2(
            epi_load_pipeline,
            epi_load_pipe_consumer_state,
            epi_store_pipeline,
            epi_store_pipe_producer_state,
            problem_shape_MNKL, sub_m_idx,
            blk_shape,
            blk_coord,
            accumulators,
            tiled_mma,
            warp_group_thread_idx,
            shared_storage.tensors.extra_storage.epilogue,
            shared_storage.tensors.extra_storage.epilogue2
            // postsum_global_srcs[0][0],
            // postsum_global_dsts[0]
          );
          epi_load_pipe_consumer_state_next = get<0>(ret);
          epi_store_pipe_producer_state_next = get<1>(ret);
        } else {
          auto ret =
          collective_epilogue.store(
            epi_load_pipeline,
            epi_load_pipe_consumer_state,
            epi_store_pipeline,
            epi_store_pipe_producer_state,
            problem_shape_MNKL,
            blk_shape,
            blk_coord, sub_m_idx,
            accumulators,
            tiled_mma,
            warp_group_thread_idx,
            shared_storage.tensors.extra_storage.epilogue,
            shared_storage.tensors.extra_storage.epilogue2
            // postsum_global_srcs,
            // postsum_global_dsts
          );
          epi_load_pipe_consumer_state_next = get<0>(ret);
          epi_store_pipe_producer_state_next = get<1>(ret);
        }
        }

        // TMA store pipeline wait is only visible to TMA-issuing warp, so for multiple-consumer kernels
        // we need to wait for all TMA stores to complete before issuing consumer order barrier arrives
        // to ensure next math consumer doesn't overwrite smem of in-flight TMA stores of current consumer.
        auto ret =
        collective_epilogue.store_tail(
          epi_load_pipeline,
          epi_load_pipe_consumer_state_next,
          epi_store_pipeline,
          epi_store_pipe_producer_state_next,
          has_global_src,
          sub_m_idx
        );

        if (StrassenMiGroup::numMs() == 2 && sub_m_idx == 0) {
          //Wait for all warpgroup threads to finish
          uint fused_mi = sub_m_idx;
          #pragma unroll 4
          for (int c = 0; c < 4; c++) {
            const MmaStrassen::PostsumOp postsum_global_dest = RWCTypes::PostsumGlobalDestByOutputIndex(c);
            const MmaStrassen::PostsumOp postsum_shared_dest = RWCTypes::PostsumSharedDestByOutputIndex(c);
            uint mi = StrassenMiGroup::getMi(fused_mi);
            int misign = RWCTypes::MiSignByOutputIndex(c, mi);

            if (misign != 0 && postsum_shared_dest.valid() && //&& !postsum_global_dest.valid()
                RWCTypes::HasPostsumSrc(postsum_global_dest.get_op())) {
              asm volatile("bar.cta.sync %0, %1;" : : "r"(10), "r"(NumThreadsPerWarpGroup));
              //Transfer only current Mi to shared memory
              //If this shared memory destination is different from global memory destination
              NumericArrayConverter<ElementD, float, 8> converter;
              //Read from this shared memory
              cutlass::Array<float, 8>* arrs = (cutlass::Array<float, 8>*)&accumulators;
              auto sh_ptr = ((cutlass::Array<ElementD, 8>*)shared_storage.tensors.extra_storage.epilogue.collective.smem_C.begin());
              const uint SHMEM_SIZE = (size<0>(TileShape{}) * size<1>(TileShape{}))/NumThreadsPerWarpGroup;

              for (int i = 0; i < accumulators.size(); i += 8) {
                sh_ptr[warp_group_thread_idx + NumThreadsPerWarpGroup*(i/8)] = converter(arrs[i/8]);
              }

              cutlass::arch::fence_view_async_shared();

              break; //There can only be as many writes as shared memory buffers available 
            }
          }
        }

        if (warp_group_thread_idx == 0 && blockIdx.x == 0 && blockIdx.y == 0)
          MY_PRINTF("1428 %d: %d %d\n", threadIdx.x, m_coord, n_coord);

        epi_load_pipe_consumer_state_next_ = get<0>(ret);
        epi_store_pipe_producer_state_next_ = get<1>(ret);

        // Update starting load/store pipeline states for the next tile
        // state has already been incremented by 1 tile in collective calls, advance once again for ping pong
        bool is_epi_load_needed = collective_epilogue.is_producer_load_needed();
        if (is_epi_load_needed)
          epi_load_pipe_consumer_state = epi_load_pipe_consumer_state_next_;
        epi_store_pipe_producer_state = epi_store_pipe_producer_state_next_;

        if (is_epi_load_needed)
          epi_load_pipe_consumer_state.advance(c_tile_count);
        epi_store_pipe_producer_state.advance(d_tile_count);

        // Cue for next Math WG's Epilogue to start
        math_wg_order_barrier.arrive();
        if constexpr (IsSchedDynamicPersistent) {  
          // Get next work tile
          auto [next_work_tile_info, increment_pipe] = 
            scheduler.fetch_next_work(
                work_tile_info, scheduler_pipeline, scheduler_pipe_consumer_state);

          work_tile_info = next_work_tile_info;
          if (increment_pipe) {
            ++scheduler_pipe_consumer_state;
            ++scheduler_pipe_consumer_state;
          }
        }
        else {
        // Get next work tile
        scheduler.template advance_to_next_work<StrassenMiGroup>(NumMmaWarpGroups/((StrassenMiGroup::numMs() == 2) ? 2 : 1));
        work_tile_info = scheduler.get_current_work();
        }
      } // Scheduler work fetch loop
    } // Consumer Warp Groups End
#endif
  if (threadIdx.x % 32 == 0 && blockIdx.x == 0 && blockIdx.y == 0)
    MY_PRINTF("1467 %d: %d %d\n", threadIdx.x, blockIdx.x, blockIdx.y);
  __syncthreads();
  if (threadIdx.x == 0)
    MY_PRINTF("1470 %d: %d %d\n", threadIdx.x, blockIdx.x, blockIdx.y);
  }
};

///////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::kernel
