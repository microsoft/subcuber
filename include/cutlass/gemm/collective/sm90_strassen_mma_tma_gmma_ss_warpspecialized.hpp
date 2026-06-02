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
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/trace.h"
#include "cutlass/barrier.h"

#include "cute/arch/cluster_sm90.hpp"
#include "cute/arch/copy_sm90.hpp"
#include "cute/algorithm/functional.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/numeric/arithmetic_tuple.hpp"

#include "cutlass/gemm/threadblock/presum_detail.h"
#include "cutlass/gemm/device/strassen_decls.h"

#define MAX(x,y) (((x)<(y)) ? (y) : (x))

using namespace cutlass::gemm::threadblock;

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective {
using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////

// WarpSpecialized Mainloop
template <
  class StrassenMiGroup_,
  int Stages,
  class ClusterShape,
  class KernelSchedule,
  class TileShape_,
  class ElementA_,
  class StrideA_,
  class ElementB_,
  class StrideB_,
  class TiledMma_,
  class GmemTiledCopyA_,
  class SmemLayoutAtomA_,
  class SmemCopyAtomA_,
  class TransformA_,
  class GmemTiledCopyB_,
  class SmemLayoutAtomB_,
  class SmemCopyAtomB_,
  class TransformB_,
  class PresumClusterShape,
  class PresumTileShapeA,
  class PresumGmemTiledCopyA,
  class PresumSmemLayoutAtomA,
  class PresumSmemCopyAtomA,
  class PresumTileShapeB,
  class PresumGmemTiledCopyB,
  class PresumSmemLayoutAtomB,
  class PresumSmemCopyAtomB,
  class PresumOpt_>
struct CollectiveStrassenMma<
    StrassenMiGroup_,
    MainloopSm90TmaGmmaWarpSpecialized<Stages, ClusterShape, KernelSchedule>,
    TileShape_,
    ElementA_,
    StrideA_,
    ElementB_,
    StrideB_,
    TiledMma_,
    GmemTiledCopyA_,
    SmemLayoutAtomA_,
    SmemCopyAtomA_,
    TransformA_,
    GmemTiledCopyB_,
    SmemLayoutAtomB_,
    SmemCopyAtomB_,
    TransformB_,
    PresumClusterShape,
    PresumTileShapeA,
    PresumGmemTiledCopyA,
    PresumSmemLayoutAtomA,
    PresumSmemCopyAtomA,
    PresumTileShapeB,
    PresumGmemTiledCopyB,
    PresumSmemLayoutAtomB,
    PresumSmemCopyAtomB,
    PresumOpt_>
{
  //
  // Type Aliases
  //
  using StrassenMiGroup = StrassenMiGroup_;
  using PresumOpt = PresumOpt_;
  using DispatchPolicy = MainloopSm90TmaGmmaWarpSpecialized<Stages, ClusterShape, KernelSchedule>;
  using TileShape = TileShape_;
  using ElementA = ElementA_;
  using StrideA = StrideA_;
  using ElementB = ElementB_;
  using StrideB = StrideB_;
  using TiledMma = TiledMma_;
  using ElementAccumulator = typename TiledMma::ValTypeC;
  using FragmentC = decltype(partition_fragment_C(TiledMma(), take<0,2>(TileShape{})));
  using GmemTiledCopyA = GmemTiledCopyA_;
  using GmemTiledCopyB = GmemTiledCopyB_;
  using SmemLayoutAtomA = SmemLayoutAtomA_;
  using SmemLayoutAtomB = SmemLayoutAtomB_;
  using SmemCopyAtomA = SmemCopyAtomA_;
  using SmemCopyAtomB = SmemCopyAtomB_;
  using TransformA = TransformA_;
  using TransformB = TransformB_;
  using ArchTag = typename DispatchPolicy::ArchTag;
  using PresumShape = GemmShape<size<0>(TileShape{}), size<1>(TileShape{}), 1>;
  using PresumShapeA = PresumShape;
  using PresumShapeB = PresumShape; 
  using PresumVecTypeA = Array<ElementA, (size<0>(PresumTileShapeA{})*sizeof(ElementA))/sizeof(ElementA)>;
  using PresumVecTypeB = Array<ElementA, (size<0>(PresumTileShapeB{})*sizeof(ElementA))/sizeof(ElementA)>;
  using PresumStoreVecType = Array<ElementA, 16/sizeof(ElementA)>;
  using PresumComputeType = ElementAccumulator;
  using PresumComputeToIOTypeA = NumericArrayConverter<ElementA, PresumComputeType, PresumVecTypeA::kElements>;
  using PresumIOToComputeTypeA = NumericArrayConverter<PresumComputeType, ElementA, PresumVecTypeA::kElements>;
  using PresumComputeToIOTypeB = NumericArrayConverter<ElementA, PresumComputeType, PresumVecTypeB::kElements>;
  using PresumIOToComputeTypeB = NumericArrayConverter<PresumComputeType, ElementA, PresumVecTypeB::kElements>;
  static const int kPresumThreads = 32 * (size<0>(PresumTileShapeA{})/2);//TODO: Pass this from StrassenGemmKernel
  using PresumGlobalIteratorA = PresumDetail::GlobalIterator<ElementA, kPresumThreads,  
                                                             PresumShape, PresumStoreVecType, false, false>;
  using PresumGlobalIteratorB = PresumDetail::GlobalIterator<ElementB, kPresumThreads, 
                                                             PresumShape, PresumStoreVecType, true, false>;
  // using PresumSharedIterator = PresumDetail::SharedIterator<ElementA, PresumVecType, NumThreads,
  //                                                           Base::SharedStorage::PresumBuffer::NumLoadsForPresum,
  //                                                           Base::SharedStorage::PresumBuffer::SingleStageSize,
  //                                                           Stages, PresumAccessPerIter>;
  static const uint kPresumComputeIterationsA = size<0>(TileShape{})/size<0>(PresumTileShapeA{});
  static const uint kPresumComputeIterationsB = size<0>(TileShape{})/size<0>(PresumTileShapeB{});

  static const bool is_fused = StrassenMiGroup::hasM0() && StrassenMiGroup::hasM1();
  using CtaShape_MNK = decltype(shape_div(TileShape{}, ClusterShape{}));
  using MainloopPipeline = cutlass::PipelineTmaAsync<DispatchPolicy::Stages>;
  using PipelineState = cutlass::PipelineState<DispatchPolicy::Stages>;

  static const int PresumStages = DispatchPolicy::Stages;
  using PresumLoadPipeline = cutlass::PipelineTmaAsync<PresumStages>;
  using PresumLoadPipelineState = cutlass::PipelineState<PresumStages>;
  using PresumLoadPipelineParams = typename PresumLoadPipeline::Params;

  using PipelineParams = typename MainloopPipeline::Params;

  // One threads per CTA are producers (1 for operand tile)
  static constexpr int NumProducerThreadEvents = 1;

  static_assert(cute::rank(SmemLayoutAtomA{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<0>(TileShape{}) % size<0>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomA{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  static_assert(cute::rank(SmemLayoutAtomB{}) == 2, "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<1>(TileShape{}) % size<0>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomB{})) == 0, "SmemLayoutAtom must evenly divide tile shape.");

  // Tile along modes in a way that maximizes the TMA box size.
  using SmemLayoutA = decltype(tile_to_shape(
      SmemLayoutAtomA{},
      make_shape(shape<0>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{}),
      cute::conditional_t< ::cutlass::gemm::detail::is_major<0,StrideA>(), Step<_2,_1,_3>, Step<_1,_2,_3>>{}));
  using SmemLayoutB = decltype(tile_to_shape(
      SmemLayoutAtomB{},
      make_shape(shape<1>(TileShape{}), shape<2>(TileShape{}), Int<DispatchPolicy::Stages>{}),
      cute::conditional_t< ::cutlass::gemm::detail::is_major<0,StrideB>(), Step<_2,_1,_3>, Step<_1,_2,_3>>{}));

  // using PresumSmemLayoutA = decltype(tile_to_shape(
  //     PresumSmemLayoutAtomA{},
  //     make_shape(shape<0>(PresumTileShapeA{}), shape<1>(PresumTileShapeA{}), Int<1>{}),
  //     Step<_1,_2,_3>{}));
  // using PresumSmemLayoutB = decltype(tile_to_shape(
  //     PresumSmemLayoutAtomB{},
  //     make_shape(shape<0>(PresumTileShapeB{}), shape<1>(PresumTileShapeB{}), Int<1>{}),
  //     Step<_1,_2,_3>{}));
  using PresumSmemLayoutA = decltype(make_layout(make_shape(shape<0>(PresumTileShapeA{}), shape<1>(PresumTileShapeA{}), Int<PresumStages>{}), Step<_1,_2,_3>{}));
  using PresumSmemLayoutB = decltype(make_layout(make_shape(shape<0>(PresumTileShapeB{}), shape<1>(PresumTileShapeB{}), Int<PresumStages>{}), Step<_1,_2,_3>{}));

  using PresumSmemShapeA__ = decltype(make_shape(shape<0>(PresumTileShapeB{}), shape<1>(PresumTileShapeB{}), Int<PresumStages>{}));
  using PresumSmemLayoutA__ = decltype(make_layout(make_shape(shape<0>(PresumTileShapeA{}), shape<1>(PresumTileShapeA{})), LayoutRight{}));
  using PresumSmemLayoutB__ = decltype(make_layout(make_shape(shape<0>(PresumTileShapeB{}), shape<1>(PresumTileShapeB{})), LayoutRight{}));

  static_assert(DispatchPolicy::Stages >= 2, "Specialization requires Stages set to value 2 or more.");
  static_assert(cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeA>::value &&
                cute::is_base_of<cute::GMMA::DescriptorIterator, typename TiledMma::FrgTypeB>::value,
                "MMA atom must source both A and B operand from smem_desc for this mainloop.");
  static_assert(cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD> || cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD_MULTICAST>,
      "GmemTiledCopy - invalid SM90 TMA copy atom specified.");
  static_assert(cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD> || cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD_MULTICAST>,
      "GmemTiledCopy - invalid SM90 TMA copy atom specified.");

  // TMA converts f32 input to tf32 when copying from GMEM to SMEM
  // For all other types, cast to size equivalent uint type to avoid any rounding by TMA.
  static constexpr bool ConvertF32toTF32A = cute::is_same_v<float, ElementA>;
  static constexpr bool ConvertF32toTF32B = cute::is_same_v<float, ElementB>;
  using InternalElementA = cute::conditional_t<ConvertF32toTF32A, tfloat32_t, uint_bit_t<sizeof_bits_v<ElementA>>>;
  using InternalElementB = cute::conditional_t<ConvertF32toTF32B, tfloat32_t, uint_bit_t<sizeof_bits_v<ElementB>>>;

  struct SharedStorage
  {
    struct TensorStorage : cute::aligned_struct<128, _0> {
      cute::array_aligned<typename TiledMma::ValTypeA, cute::cosize_v<SmemLayoutA>> smem_A;
      cute::array_aligned<typename TiledMma::ValTypeB, cute::cosize_v<SmemLayoutB>> smem_B;
    } tensors;

    using PipelineStorage = typename MainloopPipeline::SharedStorage;
    PipelineStorage pipeline;
    using PresumLoadPipelineStorage = typename PresumLoadPipeline::SharedStorage;
    PresumLoadPipelineStorage presum_load_pipeline;
  };

  static const size_t PresumSingleStageSizeA = size<0>(PresumTileShapeA{})*size<1>(PresumTileShapeA{});
  static const size_t PresumSingleStageSizeB = size<0>(PresumTileShapeB{})*size<1>(PresumTileShapeB{});

  static const size_t PresumSingleStageSize = is_fused ? MAX(PresumSingleStageSizeA, PresumSingleStageSizeB) : // && StrassenMiGroup::AllPresums::computeAnyAPresum() && StrassenMiGroup::AllPresums::computeAnyBPresum()
                                                  (StrassenMiGroup::hasM0() ? PresumSingleStageSizeA : PresumSingleStageSizeB);
  static const size_t PresumSmemSize = PresumSingleStageSize*PresumStages;
  static const uint PresumWriteStages = 1;
  static const size_t PresumOutSmemSize = 1;//size<0>(PresumTileShapeA{})*size<1>(PresumTileShapeA{}) * PresumWriteStages;

  struct PresumTensorStorageEmpty : cute::aligned_struct<128, _0> {
    cute::array_aligned<typename TiledMma::ValTypeA, 1> smem_A0;
    cute::array_aligned<typename TiledMma::ValTypeA, 1> smem_A1;
    cute::array_aligned<typename TiledMma::ValTypeA, 1> smem_A2;
    cute::array_aligned<typename TiledMma::ValTypeA, 1> smem_A3;
    cute::array_aligned<typename TiledMma::ValTypeA, 1> smem_out_A0;
    cute::array_aligned<typename TiledMma::ValTypeA, 1> smem_out_A1;
    cute::array_aligned<typename TiledMma::ValTypeA, 1> smem_out_A2;
    cute::array_aligned<typename TiledMma::ValTypeA, 1> smem_out_A3;
  };

  struct PresumTensorStorageNonEmpty : cute::aligned_struct<128, _0> {
    cute::array_aligned<typename TiledMma::ValTypeA, PresumSmemSize> smem_A0;
    cute::array_aligned<typename TiledMma::ValTypeA, PresumSmemSize> smem_A1;
    cute::array_aligned<typename TiledMma::ValTypeA, PresumSmemSize> smem_A2;
    cute::array_aligned<typename TiledMma::ValTypeA, PresumSmemSize> smem_A3;
    // cute::array_aligned<typename TiledMma::ValTypeA, PresumOutSmemSize> smem_out_A0;
    // cute::array_aligned<typename TiledMma::ValTypeA, PresumOutSmemSize> smem_out_A1;
    // cute::array_aligned<typename TiledMma::ValTypeA, PresumOutSmemSize> smem_out_A2;
    // cute::array_aligned<typename TiledMma::ValTypeA, PresumOutSmemSize> smem_out_A3;
    // union {
    //   cute::array_aligned<typename TiledMma::ValTypeA, cute::cosize_v<PresumSmemLayoutA>> smem_A0;
    //   // cute::array_aligned<typename TiledMma::ValTypeB, cute::cosize_v<PresumSmemLayoutB>> smem_B0;
    // };

    // union {
    //   cute::array_aligned<typename TiledMma::ValTypeA, cute::cosize_v<PresumSmemLayoutA>> smem_A1;
    //   // cute::array_aligned<typename TiledMma::ValTypeB, cute::cosize_v<PresumSmemLayoutB>> smem_B1;
    // };

    // union {
    //   cute::array_aligned<typename TiledMma::ValTypeA, cute::cosize_v<PresumSmemLayoutA>> smem_A2;
    //   // cute::array_aligned<typename TiledMma::ValTypeB, cute::cosize_v<PresumSmemLayoutB>> smem_B2;
    // };
    // union {
    //   cute::array_aligned<typename TiledMma::ValTypeA, cute::cosize_v<PresumSmemLayoutA>> smem_A3;
    //   // cute::array_aligned<typename TiledMma::ValTypeB, cute::cosize_v<PresumSmemLayoutB>> smem_B3;
    // };
  };

  using PresumTensorStorage = std::conditional_t<(StrassenMiGroup::hasM0() && StrassenMiGroup::AllPresums::computeAnyAPresum()) ||
                                                 (StrassenMiGroup::hasM1() && StrassenMiGroup::AllPresums::computeAnyBPresum()),
                                                 PresumTensorStorageNonEmpty,
                                                 PresumTensorStorageEmpty>;

  struct PresumTensorStorage2 : cute::aligned_struct<128, _0> {
    // cute::array_aligned<typename TiledMma::ValTypeA, PresumSmemSize> smem_A0;
    // cute::array_aligned<typename TiledMma::ValTypeA, PresumSmemSize> smem_A1;
    // cute::array_aligned<typename TiledMma::ValTypeA, PresumSmemSize> smem_A2;
    // cute::array_aligned<typename TiledMma::ValTypeA, PresumSmemSize> smem_A3;
    // cute::array_aligned<typename TiledMma::ValTypeA, PresumOutSmemSize> smem_out_A0;
    // cute::array_aligned<typename TiledMma::ValTypeA, PresumOutSmemSize> smem_out_A1;
    // cute::array_aligned<typename TiledMma::ValTypeA, PresumOutSmemSize> smem_out_A2;
    // cute::array_aligned<typename TiledMma::ValTypeA, PresumOutSmemSize> smem_out_A3;
    // union {
    //   cute::array_aligned<typename TiledMma::ValTypeA, cute::cosize_v<PresumSmemLayoutA>> smem_A0;
    //   // cute::array_aligned<typename TiledMma::ValTypeB, cute::cosize_v<PresumSmemLayoutB>> smem_B0;
    // };

    // union {
    //   cute::array_aligned<typename TiledMma::ValTypeA, cute::cosize_v<PresumSmemLayoutA>> smem_A1;
    //   // cute::array_aligned<typename TiledMma::ValTypeB, cute::cosize_v<PresumSmemLayoutB>> smem_B1;
    // };

    // union {
    //   cute::array_aligned<typename TiledMma::ValTypeA, cute::cosize_v<PresumSmemLayoutA>> smem_A2;
    //   // cute::array_aligned<typename TiledMma::ValTypeB, cute::cosize_v<PresumSmemLayoutB>> smem_B2;
    // };
    // union {
    //   cute::array_aligned<typename TiledMma::ValTypeA, cute::cosize_v<PresumSmemLayoutA>> smem_A3;
    //   // cute::array_aligned<typename TiledMma::ValTypeB, cute::cosize_v<PresumSmemLayoutB>> smem_B3;
    // };
  };
  static const bool IsFusedM4M5 = StrassenMiGroup::hasM4() && StrassenMiGroup::hasM5();

  using AllPresums = typename StrassenMiGroup::AllPresums;
  using TensorStorage = typename SharedStorage::TensorStorage;
  using PipelineStorage = typename SharedStorage::PipelineStorage;

  // Host side kernel arguments
  struct Arguments {
    ElementA const* ptr_A;
    StrideA dA;
    ElementB const* ptr_B;
    StrideB dB;
    uint32_t mma_promotion_interval = 4;

    Arguments(ElementA const* ptr_A, StrideA dA, ElementB const* ptr_B, StrideB dB,
              uint32_t mma_promotion_interval = 4) :
              ptr_A(ptr_A), dA(dA), ptr_B(ptr_B), dB(dB), mma_promotion_interval(mma_promotion_interval)
              {}

    template<typename Other>
    Arguments(const Other& other) : ptr_A(other.ptr_A), dA(other.dA), ptr_B(other.ptr_B), dB(other.dB),
                                   mma_promotion_interval(other.mma_promotion_interval)
    {}
  };

  // Device side kernel params
  struct Params {
    // Assumption: StrideA is congruent with Problem_MK
    using TMA_A = decltype(make_tma_copy_A_sm90(
        GmemTiledCopyA{},
        make_tensor(static_cast<InternalElementA const*>(nullptr), repeat_like(StrideA{}, int32_t(0)), StrideA{}),
        SmemLayoutA{}(_,_,cute::Int<0>{}),
        TileShape{},
        ClusterShape{}));
    // Assumption: StrideB is congruent with Problem_NK
    using TMA_B = decltype(make_tma_copy_B_sm90(
        GmemTiledCopyB{},
        make_tensor(static_cast<InternalElementB const*>(nullptr), repeat_like(StrideB{}, int32_t(0)), StrideB{}),
        SmemLayoutB{}(_,_,cute::Int<0>{}),
        TileShape{},
        ClusterShape{}));
    TMA_A tma_load_a;
    TMA_B tma_load_b;
    TMA_A tma_load_presum_a;
    TMA_B tma_load_presum_b;

    // Assumption: StrideA is congruent with Problem_MK
    using TMA_PresumLoad_A = decltype(make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(static_cast<InternalElementA const*>(nullptr), repeat_like(StrideA{}, int32_t(0)), StrideA{}),
        PresumSmemLayoutA__{}));
    using TMA_PresumLoad_B = decltype(make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(static_cast<InternalElementB const*>(nullptr), repeat_like(StrideA{}, int32_t(0)), StrideA{}),
        PresumSmemLayoutB__{}));
    // Assumption: StrideB is congruent with Problem_NK
    using TMA_PresumStore_A = decltype(make_tma_copy(
        SM90_TMA_STORE{},
        make_tensor(static_cast<InternalElementA const*>(nullptr), repeat_like(StrideA{}, int32_t(0)), StrideA{}),
        PresumSmemLayoutA__{}));
    using TMA_PresumStore_B = decltype(make_tma_copy(
        SM90_TMA_STORE{},
        make_tensor(static_cast<InternalElementB const*>(nullptr), repeat_like(StrideA{}, int32_t(0)), StrideA{}),
        PresumSmemLayoutB__{}));
    TMA_PresumLoad_A tma_load_presumld_a;
    TMA_PresumLoad_B tma_load_presumld_b;
    TMA_PresumStore_A tma_store_presumld_a;
    TMA_PresumStore_B tma_store_presumld_b;

    __restrict__ ElementA* ptr_presum_A;
    __restrict__ ElementB* ptr_presum_B;

    uint32_t tma_transaction_bytes = TmaTransactionBytes;
    uint32_t tma_transaction_bytes_mk = TmaTransactionBytesMK;
    uint32_t tma_transaction_bytes_nk = TmaTransactionBytesNK;
    uint32_t tma_transaction_bytes_presum_mk = TmaTransactionBytesPresumA;
    uint32_t presum_tile_log_multiplier_a_;
    uint32_t presum_tile_log_multiplier_b_;
    uint32_t presum_tile_log_divider_a_;
    uint32_t presum_tile_log_divider_b_;

    CUTLASS_HOST_DEVICE
    uint32_t get_presum_tile_log_multiplier_a() const {
      return (PresumOpt::FixedPresumTileMultilplierLogA != UINT32_MAX) ?
              PresumOpt::FixedPresumTileMultilplierLogA :
              presum_tile_log_multiplier_a_;
    }

    CUTLASS_HOST_DEVICE
    uint32_t get_presum_tile_log_multiplier_b() const {
      return (PresumOpt::FixedPresumTileMultilplierLogB != UINT32_MAX) ?
              PresumOpt::FixedPresumTileMultilplierLogB :
              presum_tile_log_multiplier_b_;
    }

    CUTLASS_HOST_DEVICE
    uint32_t get_presum_tile_log_divider_a() const {
      return (PresumOpt::FixedPresumTileDividerLogA != UINT32_MAX) ? 
              PresumOpt::FixedPresumTileDividerLogA :
              presum_tile_log_divider_a_;
    }

    CUTLASS_HOST_DEVICE
    uint32_t get_presum_tile_log_divider_b() const {
      return (PresumOpt::FixedPresumTileDividerLogB != UINT32_MAX) ? 
              PresumOpt::FixedPresumTileDividerLogB :
              presum_tile_log_divider_b_;
    }
  };

  //
  // Methods
  //

  //TODO: collect these functions in one place.
  CUTLASS_HOST_DEVICE
  static uint32_t get_presum_log_multiplier(int k, int m_or_n) {
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

  CUTLASS_HOST_DEVICE
  static uint32_t get_presum_log_divider(int k, int m_or_n) {
    //TODO: We want k/2 is a divisor of Number of threads (which is usually 256 for us)
    if (k >= m_or_n) return 0;
    int divider = m_or_n/k;

    if (divider >= 8) {
      return 3; //divide with 8
    } else if (divider >= 4) {
      return 2; //divide with 4
    } else if (divider >= 2) {
      return 1; //divide with 2
    } else {
      return 0; //divide with 1
    }
  }

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, ElementA* ptr_presum_a, ElementB* ptr_presum_b, Arguments const& args, void* workspace) {
    (void) workspace;

    // Optionally append 1s until problem shape is rank-4 (MNKL), in case it is only rank-3 (MNK)
    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto [M,N,K,L] = problem_shape_MNKL;

    auto presum_tile_log_multiplier_a = get_presum_log_multiplier(K, N);
    auto presum_tile_log_multiplier_b = get_presum_log_multiplier(K, M);
    auto presum_tile_log_divider_a = get_presum_log_divider(K, N);
    auto presum_tile_log_divider_b = get_presum_log_divider(K, M);

    auto ptr_A = reinterpret_cast<InternalElementA const*>(args.ptr_A);
    auto ptr_B = reinterpret_cast<InternalElementB const*>(args.ptr_B);
    auto ptr_presum_A = reinterpret_cast<InternalElementA const*>(ptr_presum_a);
    auto ptr_presum_B = reinterpret_cast<InternalElementB const*>(ptr_presum_b);

    Tensor tensor_a = make_tensor(ptr_A, make_layout(make_shape(M,K,L), args.dA));
    Tensor tensor_b = make_tensor(ptr_B, make_layout(make_shape(N,K,L), args.dB));

    Tensor tensor_presum_a = make_tensor(ptr_presum_A, make_layout(make_shape(4*M/2,K/2,L), make_stride(get<0>(args.dA)/2, get<1>(args.dA), get<2>(args.dA))));
    Tensor tensor_presum_b = make_tensor(ptr_presum_B, make_layout(make_shape(N/2,4*K/2,L), make_stride(get<0>(args.dB), get<1>(args.dB)/2, get<2>(args.dB))));

    typename Params::TMA_A tma_load_a = make_tma_copy_A_sm90(
        GmemTiledCopyA{},
        tensor_a,
        SmemLayoutA{}(_,_,cute::Int<0>{}),
        TileShape{},
        ClusterShape{});
    typename Params::TMA_B tma_load_b = make_tma_copy_B_sm90(
        GmemTiledCopyB{},
        tensor_b,
        SmemLayoutB{}(_,_,cute::Int<0>{}),
        TileShape{},
        ClusterShape{});
    typename Params::TMA_A tma_load_presum_a = make_tma_copy_A_sm90(
        GmemTiledCopyA{},
        tensor_presum_a,
        SmemLayoutA{}(_,_,cute::Int<0>{}),
        TileShape{},
        ClusterShape{});
    typename Params::TMA_B tma_load_presum_b = make_tma_copy_B_sm90(
        GmemTiledCopyB{},
        tensor_presum_b,
        SmemLayoutB{}(_,_,cute::Int<0>{}),
        TileShape{},
        ClusterShape{});

    typename Params::TMA_PresumLoad_A tma_load_presumld_a = make_tma_copy(
        SM90_TMA_LOAD{},
        tensor_a,
        PresumSmemLayoutA__{});
    typename Params::TMA_PresumLoad_B tma_load_presumld_b = make_tma_copy(
        SM90_TMA_LOAD{},
        make_tensor(ptr_B, make_layout(make_shape(K,N,L), make_stride(get<1>(args.dB), get<1>(args.dA), get<2>(args.dA)))),
        PresumSmemLayoutB__{});
    
    typename Params::TMA_PresumStore_A tma_store_presumld_a = make_tma_copy(
        SM90_TMA_STORE{},
        tensor_presum_a,
        PresumSmemLayoutA__{});
        
    typename Params::TMA_PresumStore_B tma_store_presumld_b = make_tma_copy(
        SM90_TMA_STORE{},
        make_tensor(ptr_presum_B, make_layout(make_shape(4*K/2,N/2,L), make_stride(get<1>(args.dB)/2, get<1>(args.dA), get<2>(args.dA)))),
        PresumSmemLayoutB__{});

    uint32_t transaction_bytes_mk = TmaTransactionBytesMK;
    uint32_t transaction_bytes_nk = TmaTransactionBytesNK;
    uint32_t transaction_bytes = transaction_bytes_mk + transaction_bytes_nk;

    return {
      tma_load_a,
      tma_load_b,
      tma_load_presum_a,
      tma_load_presum_b,
      tma_load_presumld_a,
      tma_load_presumld_b,
      tma_store_presumld_a,
      tma_store_presumld_b,
      (ElementA*)ptr_presum_A, (ElementA*)ptr_presum_B,
      transaction_bytes,// + (((StrassenMiGroup::hasM0() && StrassenMiGroup::AllPresums::computeAnyAPresum()) ||
                          //  (StrassenMiGroup::hasM1() && StrassenMiGroup::AllPresums::computeAnyBPresum())) ? 1 : 0)*4*TmaTransactionBytesPresum,
      transaction_bytes_mk,
      transaction_bytes_nk,0,
      presum_tile_log_multiplier_a,
      presum_tile_log_multiplier_b,
      presum_tile_log_divider_a, //TODO: Specialize to the value when needed
      presum_tile_log_divider_b
    };
  }

  template<class ProblemShape>
  static bool
  can_implement(
      ProblemShape const& problem_shape,
      [[maybe_unused]] Arguments const& args) {
    constexpr int tma_alignment_bits = 128;
    auto problem_shape_MNKL = append<4>(problem_shape, 1);
    auto [M,N,K,L] = problem_shape_MNKL;

    bool implementable = true;
    constexpr int min_tma_aligned_elements_A = tma_alignment_bits / cutlass::sizeof_bits<ElementA>::value;
    implementable = implementable && cutlass::detail::check_alignment<min_tma_aligned_elements_A>(cute::make_shape(M,K,L), StrideA{});
    constexpr int min_tma_aligned_elements_B = tma_alignment_bits / cutlass::sizeof_bits<ElementB>::value;
    implementable = implementable && cutlass::detail::check_alignment<min_tma_aligned_elements_B>(cute::make_shape(N,K,L), StrideB{});

    if (!implementable) {
      CUTLASS_TRACE_HOST("  CAN IMPLEMENT: Problem Size doesn't meet the minimum alignment requirements for TMA.\n");
    }
    return implementable;
  }

  static constexpr int K_PIPE_MAX = DispatchPolicy::Stages;
  static constexpr int K_PIPE_MMAS = 1;
  static constexpr uint32_t TmaTransactionBytesMK =
        cutlass::bits_to_bytes(size<0>(SmemLayoutA{}) * size<1>(SmemLayoutA{}) * static_cast<uint32_t>(sizeof_bits<ElementA>::value));
  static constexpr uint32_t TmaTransactionBytesNK =
        cutlass::bits_to_bytes(size<0>(SmemLayoutB{}) * size<1>(SmemLayoutB{}) * static_cast<uint32_t>(sizeof_bits<ElementB>::value));
  static constexpr uint32_t TmaTransactionBytes = TmaTransactionBytesMK + TmaTransactionBytesNK;

  static constexpr uint32_t TmaTransactionBytesPresumA =
        cutlass::bits_to_bytes(PresumSingleStageSizeA * static_cast<uint32_t>(sizeof_bits<ElementA>::value));
  static constexpr uint32_t TmaTransactionBytesPresumB =
      cutlass::bits_to_bytes(PresumSingleStageSizeB * static_cast<uint32_t>(sizeof_bits<ElementA>::value));

  /// Issue Tma Descriptor Prefetch -- ideally from a single thread for best performance
  CUTLASS_DEVICE
  static void prefetch_tma_descriptors(Params const& mainloop_params) {
    cute::prefetch_tma_descriptor(mainloop_params.tma_load_a.get_tma_descriptor());
    cute::prefetch_tma_descriptor(mainloop_params.tma_load_b.get_tma_descriptor());
    //TODO: Do this only when presum_a or presum_b is needed
    cute::prefetch_tma_descriptor(mainloop_params.tma_load_presum_a.get_tma_descriptor());
    cute::prefetch_tma_descriptor(mainloop_params.tma_load_presum_b.get_tma_descriptor());
    if (StrassenMiGroup::hasM0() && StrassenMiGroup::AllPresums::computeAnyAPresum()) {
      cute::prefetch_tma_descriptor(mainloop_params.tma_load_presumld_a.get_tma_descriptor());
      cute::prefetch_tma_descriptor(mainloop_params.tma_store_presumld_a.get_tma_descriptor());
    }
    if (StrassenMiGroup::hasM1() && StrassenMiGroup::AllPresums::computeAnyBPresum()) {
      cute::prefetch_tma_descriptor(mainloop_params.tma_load_presumld_b.get_tma_descriptor());
      cute::prefetch_tma_descriptor(mainloop_params.tma_store_presumld_b.get_tma_descriptor());
    }
  }

  /// Set up the data needed by this collective for load and mma.
  /// Returns a tuple of tensors. The collective and the kernel layer have the contract
  /// Returned tuple must contain at least two elements, with the first two elements being:
  /// gA_mkl - The tma tensor, A after a local tile so it has shape  (BLK_M,BLK_K,m,k,l)
  /// gB_nkl - The tma tensor, B after a local tile so it has shape  (BLK_N,BLK_K,n,k,l)
  /// The rest of the tensors can be specified as needed by this collective.
  template <class ProblemShape_MNKL>
  CUTLASS_DEVICE auto
  load_init(ProblemShape_MNKL const& problem_shape_MNKL, ProblemShape_MNKL const& half_problem_shape_MNKL, Params const& mainloop_params) const {
    using X = Underscore;
    // Separate out problem shape for convenience
    auto [M,N,K,L] = problem_shape_MNKL;
    auto [halfM, halfN, halfK, halfL] = half_problem_shape_MNKL;

    // TMA requires special handling of strides to deal with coord codomain mapping
    // Represent the full tensors -- get these from TMA
    Tensor mA_mkl = mainloop_params.tma_load_a.get_tma_tensor(make_shape(M,K,L));                            // (m,k,l)
    Tensor mB_nkl = mainloop_params.tma_load_b.get_tma_tensor(make_shape(N,K,L));                            // (n,k,l)

    Tensor presum_mA_mkl = mainloop_params.tma_load_presum_a.get_tma_tensor(make_shape(halfM,halfK,L));                            // (m,k,l)
    Tensor presum_mB_nkl = mainloop_params.tma_load_presum_b.get_tma_tensor(make_shape(halfN,halfK,L));                            // (n,k,l)
    auto presum_mA_ptr = presum_mA_mkl.data();

    auto presum_A02_ptr = presum_mA_ptr + make_coord(0,StrassenMiGroup::AllPresums::indexAPresum(StrassenMiGroup::APresums::A02)*halfM,_);
    auto presum_S1_ptr = presum_mA_ptr + make_coord(0,StrassenMiGroup::AllPresums::indexAPresum(StrassenMiGroup::APresums::S1)*halfM,_);
    auto presum_S2_ptr = presum_mA_ptr + make_coord(0,StrassenMiGroup::AllPresums::indexAPresum(StrassenMiGroup::APresums::S2)*halfM,_);
    auto presum_A1S2_ptr = presum_mA_ptr + make_coord(0,StrassenMiGroup::AllPresums::indexAPresum(StrassenMiGroup::APresums::A1S2)*halfM,_);

    auto presum_A02 = make_tensor(presum_A02_ptr, presum_mA_mkl.layout());
    auto presum_S1 = make_tensor(presum_S1_ptr, presum_mA_mkl.layout());
    auto presum_S2 = make_tensor(presum_S2_ptr, presum_mA_mkl.layout());
    auto presum_A1S2 = make_tensor(presum_A1S2_ptr, presum_mA_mkl.layout());

    auto presum_mB_ptr = presum_mB_nkl.data();

    auto presum_B31_ptr = presum_mB_ptr + make_coord(0,StrassenMiGroup::AllPresums::indexBPresum(StrassenMiGroup::BPresums::B31)*halfK,_);
    auto presum_B10_ptr = presum_mB_ptr + make_coord(0,StrassenMiGroup::AllPresums::indexBPresum(StrassenMiGroup::BPresums::B10)*halfK,_);
    auto presum_S3_ptr = presum_mB_ptr + make_coord(0,StrassenMiGroup::AllPresums::indexBPresum(StrassenMiGroup::BPresums::S3)*halfK,_);
    auto presum_S3B2_ptr = presum_mB_ptr + make_coord(0,StrassenMiGroup::AllPresums::indexBPresum(StrassenMiGroup::BPresums::S3B2)*halfK,_);

    auto presum_B31 = make_tensor(presum_B31_ptr, presum_mB_nkl.layout());
    auto presum_B10 = make_tensor(presum_B10_ptr, presum_mB_nkl.layout());
    auto presum_S3 = make_tensor(presum_S3_ptr, presum_mB_nkl.layout());
    auto presum_S3B2 = make_tensor(presum_S3B2_ptr, presum_mB_nkl.layout());

    auto A00_data = mA_mkl.data() + make_coord(0*halfK,0*halfM,_);
    auto A01_data = mA_mkl.data() + make_coord(1*halfK,0*halfM,_);
    auto A10_data = mA_mkl.data() + make_coord(0*halfK,1*halfM,_);
    auto A11_data = mA_mkl.data() + make_coord(1*halfK,1*halfM,_);

    auto B00_data = mB_nkl.data() + make_coord(0*halfN,0*halfK,_);
    auto B01_data = mB_nkl.data() + make_coord(1*halfN,0*halfK,_);
    auto B10_data = mB_nkl.data() + make_coord(0*halfN,1*halfK,_);
    auto B11_data = mB_nkl.data() + make_coord(1*halfN,1*halfK,_);

    // Make tiled views, defer the slice
    Tensor gA00_mkl = local_tile(make_tensor(A00_data, mA_mkl.layout()), TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});        // (BLK_M,BLK_K,m,k,l)
    Tensor gA01_mkl = local_tile(make_tensor(A01_data, mA_mkl.layout()), TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});
    Tensor gA10_mkl = local_tile(make_tensor(A10_data, mA_mkl.layout()), TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});
    Tensor gA11_mkl = local_tile(make_tensor(A11_data, mA_mkl.layout()), TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});

    Tensor gB00_nkl = local_tile(make_tensor(B00_data, mB_nkl.layout()), TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});        // (BLK_N,BLK_K,n,k,l)
    Tensor gB01_nkl = local_tile(make_tensor(B01_data, mB_nkl.layout()), TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});
    Tensor gB10_nkl = local_tile(make_tensor(B10_data, mB_nkl.layout()), TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});
    Tensor gB11_nkl = local_tile(make_tensor(B11_data, mB_nkl.layout()), TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});

    Tensor gPresumA02 = local_tile(presum_A02, TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});        // (BLK_M,BLK_K,m,k,l)
    Tensor gPresumS1  = local_tile(presum_S1,  TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});
    Tensor gPresumS2  = local_tile(presum_S2,  TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});
    Tensor gPresumA1S2 = local_tile(presum_A1S2, TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});

    Tensor gPresumB31 = local_tile(presum_B31, TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});
    Tensor gPresumB10 = local_tile(presum_B10, TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});
    Tensor gPresumS3 = local_tile(presum_S3, TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});
    Tensor gPresumS3B2 = local_tile(presum_S3B2, TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});
    
    return cute::make_tuple(gA00_mkl, gA01_mkl, gA10_mkl, gA11_mkl,
                            gPresumA02, gPresumS1, gPresumS2, gPresumA1S2,
                            gB00_nkl, gB01_nkl, gB10_nkl, gB11_nkl,
                            gPresumB31, gPresumB10, gPresumS3, gPresumS3B2);
  }

  template<typename Tuple>
  CUTLASS_DEVICE auto
  get_inputs(Tuple const& load_inputs, int sub_m_idx = 0) {
    auto load_inputs_a = take<0, 8  >(load_inputs);
    auto load_inputs_b = take<8, 8+8>(load_inputs);

    if ((StrassenMiGroup::hasM0() && !is_fused) || (is_fused && sub_m_idx == 0)) {
      return make_tuple(get<MmaStrassen::APresums::A0>(load_inputs_a),
                        get<MmaStrassen::BPresums::B0>(load_inputs_b));
    }
    if ((StrassenMiGroup::hasM1() && !is_fused) || (is_fused && sub_m_idx == 1)) {
      return make_tuple(get<MmaStrassen::APresums::A1>(load_inputs_a),
                        get<MmaStrassen::BPresums::B2>(load_inputs_b));
    }

    bool IsFusedM2M3 = StrassenMiGroup::hasM2() && StrassenMiGroup::hasM3();

    if ((StrassenMiGroup::hasM2() && !IsFusedM2M3) || (IsFusedM2M3 && sub_m_idx == 0)) {
       return make_tuple(get<MmaStrassen::APresums::S2>(load_inputs_a),
                         get<MmaStrassen::BPresums::S3>(load_inputs_b));
    }

    if ((StrassenMiGroup::hasM3() && !IsFusedM2M3) || (IsFusedM2M3 && sub_m_idx == 1)) {
      return make_tuple(get<StrassenMiGroup::APresums::A02>(load_inputs_a),
                        get<StrassenMiGroup::BPresums::B31>(load_inputs_b));
    }
    if ((StrassenMiGroup::hasM4() && !IsFusedM4M5) || (IsFusedM4M5 && sub_m_idx == 0)) {
      return make_tuple(get<StrassenMiGroup::APresums::S1>(load_inputs_a),
                        get<StrassenMiGroup::BPresums::B10>(load_inputs_b));
    }
    if ((StrassenMiGroup::hasM5() && !IsFusedM4M5) || (IsFusedM4M5 && sub_m_idx == 1)) {
      return make_tuple(get<StrassenMiGroup::APresums::A1S2>(load_inputs_a),
                        get<StrassenMiGroup::BPresums::B3>(load_inputs_b));
    }
    if (StrassenMiGroup::hasM6()) {
      return make_tuple(get<StrassenMiGroup::APresums::A3>(load_inputs_a),
                        get<StrassenMiGroup::BPresums::S3B2>(load_inputs_b));
    }

    return make_tuple(get<0>(load_inputs_a), get<0>(load_inputs_b));
    // CUTE_GCC_UNREACHABLE;
  }

  template <class ProblemShape_MNKL>
  CUTLASS_DEVICE auto
  presumld_inputs(ProblemShape_MNKL const& problem_shape_MNKL, ProblemShape_MNKL const& half_problem_shape_MNKL, Params const& mainloop_params) const {
    using X = Underscore;
    // Separate out problem shape for convenience
    auto [M,N,K,L] = problem_shape_MNKL;
    auto [halfM, halfN, halfK, halfL] = half_problem_shape_MNKL;

    // TMA requires special handling of strides to deal with coord codomain mapping
    // Represent the full tensors -- get these from TMA
    Tensor mA_mkl = mainloop_params.tma_load_presumld_a.get_tma_tensor(make_shape(M,K,L));                            // (m,k,l)
    Tensor mB_nkl = mainloop_params.tma_load_presumld_b.get_tma_tensor(make_shape(K,N,L));                            // (n,k,l)

    auto A00_data = mA_mkl.data() + make_coord(0*halfK,0*halfM,_);
    auto A01_data = mA_mkl.data() + make_coord(1*halfK,0*halfM,_);
    auto A10_data = mA_mkl.data() + make_coord(0*halfK,1*halfM,_);
    auto A11_data = mA_mkl.data() + make_coord(1*halfK,1*halfM,_);

    auto B00_data = mB_nkl.data() + make_coord(0*halfN,0*halfK,_);
    auto B01_data = mB_nkl.data() + make_coord(1*halfN,0*halfK,_);
    auto B10_data = mB_nkl.data() + make_coord(0*halfN,1*halfK,_);
    auto B11_data = mB_nkl.data() + make_coord(1*halfN,1*halfK,_);

    // // Make tiled views, defer the slice
    // Tensor gA00_mkl = local_tile(make_tensor(A00_data, mA_mkl.layout()), PresumTileShapeA{}, make_coord(_,_,_));        // (BLK_M,BLK_K,m,k,l)
    // Tensor gA01_mkl = local_tile(make_tensor(A01_data, mA_mkl.layout()), PresumTileShapeA{}, make_coord(_,_,_));
    // Tensor gA10_mkl = local_tile(make_tensor(A10_data, mA_mkl.layout()), PresumTileShapeA{}, make_coord(_,_,_));
    // Tensor gA11_mkl = local_tile(make_tensor(A11_data, mA_mkl.layout()), PresumTileShapeA{}, make_coord(_,_,_));

    Tensor gA00_mkl = make_tensor(A00_data, mA_mkl.layout());       // (BLK_M,BLK_K,m,k,l)
    Tensor gA01_mkl = make_tensor(A01_data, mA_mkl.layout());
    Tensor gA10_mkl = make_tensor(A10_data, mA_mkl.layout());
    Tensor gA11_mkl = make_tensor(A11_data, mA_mkl.layout());

    Tensor gB00_nkl = make_tensor(B00_data, mB_nkl.layout());        // (BLK_N,BLK_K,n,k,l)
    Tensor gB01_nkl = make_tensor(B01_data, mB_nkl.layout());
    Tensor gB10_nkl = make_tensor(B10_data, mB_nkl.layout());
    Tensor gB11_nkl = make_tensor(B11_data, mB_nkl.layout());

    // Tensor gPresumA02 = local_tile(presum_A02, TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});        // (BLK_M,BLK_K,m,k,l)
    // Tensor gPresumS1  = local_tile(presum_S1,  TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});
    // Tensor gPresumS2  = local_tile(presum_S2,  TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});
    // Tensor gPresumA1S2 = local_tile(presum_A1S2, TileShape{}, make_coord(_,_,_), Step<_1, X,_1>{});

    // Tensor gPresumB31 = local_tile(presum_B31, TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});
    // Tensor gPresumB10 = local_tile(presum_B10, TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});
    // Tensor gPresumS3 = local_tile(presum_S3, TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});
    // Tensor gPresumS3B2 = local_tile(presum_S3B2, TileShape{}, make_coord(_,_,_), Step< X,_1,_1>{});
    
    return cute::make_tuple(gA00_mkl, gA01_mkl, gA10_mkl, gA11_mkl,
                            gB00_nkl, gB01_nkl, gB10_nkl, gB11_nkl);
                            // gPresumA02, gPresumS1, gPresumS2, gPresumA1S2,
                            // gPresumB31, gPresumB10, gPresumS3, gPresumS3B2
                          // );
  }

  template <class ProblemShape_MNKL>
  CUTLASS_DEVICE auto
  presumld_outputs(ProblemShape_MNKL const& problem_shape_MNKL, ProblemShape_MNKL const& half_problem_shape_MNKL, Params const& mainloop_params) const {
    using X = Underscore;
    // Separate out problem shape for convenience
    auto [M,N,K,L] = problem_shape_MNKL;
    auto [halfM, halfN, halfK, halfL] = half_problem_shape_MNKL;

    // TMA requires special handling of strides to deal with coord codomain mapping
    // Represent the full tensors -- get these from TMA
    Tensor presum_mA_mkl = mainloop_params.tma_store_presumld_a.get_tma_tensor(make_shape(halfM,halfK,L));                            // (m,k,l)
    auto presum_mA_ptr = presum_mA_mkl.data();

    auto presum_A02_ptr = presum_mA_ptr + make_coord(0,StrassenMiGroup::AllPresums::indexAPresum(StrassenMiGroup::APresums::A02)*halfM,_);
    auto presum_S1_ptr = presum_mA_ptr + make_coord(0,StrassenMiGroup::AllPresums::indexAPresum(StrassenMiGroup::APresums::S1)*halfM,_);
    auto presum_S2_ptr = presum_mA_ptr + make_coord(0,StrassenMiGroup::AllPresums::indexAPresum(StrassenMiGroup::APresums::S2)*halfM,_);
    auto presum_A1S2_ptr = presum_mA_ptr + make_coord(0,StrassenMiGroup::AllPresums::indexAPresum(StrassenMiGroup::APresums::A1S2)*halfM,_);

    auto gPresumA02 = make_tensor(presum_A02_ptr, presum_mA_mkl.layout());
    auto gPresumS1 = make_tensor(presum_S1_ptr, presum_mA_mkl.layout());
    auto gPresumS2 = make_tensor(presum_S2_ptr, presum_mA_mkl.layout());
    auto gPresumA1S2 = make_tensor(presum_A1S2_ptr, presum_mA_mkl.layout());

    // auto presum_mB_ptr = presum_mB_nkl.data();

    // auto presum_B31_ptr = presum_mB_ptr + make_coord(0,StrassenMiGroup::AllPresums::indexBPresum(StrassenMiGroup::BPresums::B31)*halfK,_);
    // auto presum_B10_ptr = presum_mB_ptr + make_coord(0,StrassenMiGroup::AllPresums::indexBPresum(StrassenMiGroup::BPresums::B10)*halfK,_);
    // auto presum_S3_ptr = presum_mB_ptr + make_coord(0,StrassenMiGroup::AllPresums::indexBPresum(StrassenMiGroup::BPresums::S3)*halfK,_);
    // auto presum_S3B2_ptr = presum_mB_ptr + make_coord(0,StrassenMiGroup::AllPresums::indexBPresum(StrassenMiGroup::BPresums::S3B2)*halfK,_);

    // auto presum_B31 = make_tensor(presum_B31_ptr, presum_mB_nkl.layout());
    // auto presum_B10 = make_tensor(presum_B10_ptr, presum_mB_nkl.layout());
    // auto presum_S3 = make_tensor(presum_S3_ptr, presum_mB_nkl.layout());
    // auto presum_S3B2 = make_tensor(presum_S3B2_ptr, presum_mB_nkl.layout());

    return cute::make_tuple(gPresumA02, gPresumS1, gPresumS2, gPresumA1S2);
  }

  /// Perform a collective-scoped matrix multiply-accumulate
  /// Producer Perspective
  template <
    class TensorA, class TensorB,
    class PresumLDInputs,
    class KTileIterator, class BlockCoord,
    class ProblemShape_MNKL
  >
  CUTLASS_DEVICE void
  load(
      Params const& mainloop_params,
      ProblemShape_MNKL const& half_problem_shape,
      MainloopPipeline pipeline,
      PipelineState smem_pipe_write,
      cute::tuple<TensorA, TensorB> const& load_inputs,
      BlockCoord const& blk_coord, int sub_m_idx,
      KTileIterator k_tile_iter, int k_tile_count,
      int thread_idx,
      uint32_t block_rank_in_cluster,
      TensorStorage& shared_tensors,
      PresumLDInputs const& all_presumld_inputs,
      PresumTensorStorage& shared_presum_tensors) {
    int lane_predicate = cute::elect_one_sync();
    //TODO: Optimize for when the presum tile log parameters are 0
    if (lane_predicate) {
      auto [m_coord, n_coord, k_coord, l_coord] = blk_coord;
      uint block_idx = 0;
      auto [halfM, halfN, halfK, halfL] = half_problem_shape;
      const int n_coord_div = (n_coord * (1 << mainloop_params.get_presum_tile_log_multiplier_a())) % (1 << mainloop_params.get_presum_tile_log_divider_a());
      const int new_n_coord = (n_coord * (1 << mainloop_params.get_presum_tile_log_multiplier_a())) >> mainloop_params.get_presum_tile_log_divider_a();

      PresumGlobalIteratorA iter_PresumA_M(
        mainloop_params.ptr_presum_A, halfK,
        {halfM, halfK},
        {m_coord * size<0>(TileShape{}) + ((n_coord_div * size<0>(TileShape{})) >> mainloop_params.get_presum_tile_log_divider_a()),
          new_n_coord * size<1>(TileShape{})},
        block_idx, {0, 0}, 0, {1*halfM, 0}, {2*halfM, 0}, {3*halfM, 0}
      );
      PresumGlobalIteratorB iter_PresumB_M(
        mainloop_params.ptr_presum_B, halfN,
        {halfK, halfN},
        {(m_coord * (1 << mainloop_params.get_presum_tile_log_multiplier_b()) * (size<0>(TileShape{}))) >> mainloop_params.get_presum_tile_log_divider_b(),
          n_coord * size<1>(TileShape{})},
        block_idx, {0, 0}, 0, {1*halfK, 0}, {2*halfK, 0}, {3*halfK, 0}
      );

      Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.data()), SmemLayoutA{});        // (BLK_M,BLK_K,PIPE)
      Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.data()), SmemLayoutB{});        // (BLK_N,BLK_K,PIPE)
      
      Tensor gA0 = get<0>(all_presumld_inputs);
      Tensor gA1 = get<1>(all_presumld_inputs);
      Tensor gA2 = get<2>(all_presumld_inputs);
      Tensor gA3 = get<3>(all_presumld_inputs);

      Tensor gB0 = get<4>(all_presumld_inputs);
      Tensor gB1 = get<5>(all_presumld_inputs);
      Tensor gB2 = get<6>(all_presumld_inputs);
      Tensor gB3 = get<7>(all_presumld_inputs);
      //
      // Prepare the TMA loads for A and B
      //

      constexpr uint32_t cluster_shape_x = get<0>(typename DispatchPolicy::ClusterShape());
      uint2 cluster_local_block_id = {block_rank_in_cluster % cluster_shape_x, block_rank_in_cluster / cluster_shape_x};

      Tensor gA_mkl = get<0>(load_inputs);
      Tensor gB_nkl = get<1>(load_inputs);

      auto& tma_load_a = (IsFusedM4M5) ? (sub_m_idx == 0 ? mainloop_params.tma_load_presum_a : mainloop_params.tma_load_presum_a) :
                          ((StrassenMiGroup::APresumLoads().get_first_access_idx() < MmaStrassen::APresums::APresumStart) ?
                            mainloop_params.tma_load_a :
                            mainloop_params.tma_load_presum_a);
      auto& tma_load_b = (IsFusedM4M5) ? (sub_m_idx == 0 ? mainloop_params.tma_load_presum_b : mainloop_params.tma_load_b) :
                          ((StrassenMiGroup::BPresumLoads().get_first_access_idx() < MmaStrassen::BPresums::BPresumStart) ?
                            mainloop_params.tma_load_b : 
                            mainloop_params.tma_load_presum_b);

      auto block_tma_a = tma_load_a.get_slice(cluster_local_block_id.y);
      auto block_tma_b = tma_load_b.get_slice(cluster_local_block_id.x);

      // Partition the inputs based on the current block coordinates.
      auto block_tma_presum_ld_a = mainloop_params.tma_load_presumld_a.get_slice(0);
      auto block_tma_presum_ld_b = mainloop_params.tma_load_presumld_b.get_slice(0);

      Tensor gA = gA_mkl(_,_,m_coord,_,l_coord);                                                     // (BLK_M,BLK_K,k)
      Tensor gB = gB_nkl(_,_,n_coord,_,l_coord);                                                     // (BLK_N,BLK_K,k)

      // Applies the mapping from block_tma_a
      Tensor tAgA = block_tma_a.partition_S(gA);                                                 // (TMA,TMA_M,TMA_K,k)
      Tensor tAsA = block_tma_a.partition_D(sA);                                              // (TMA,TMA_M,TMA_K,PIPE)

      Tensor tBgB = block_tma_b.partition_S(gB);                                                 // (TMA,TMA_N,TMA_K,k)
      Tensor tBsB = block_tma_b.partition_D(sB);                                              // (TMA,TMA_N,TMA_K,PIPE)

      uint16_t mcast_mask_a = 0;
      uint16_t mcast_mask_b = 0;

      // Issue TmaLoads
      // Maps the tile -> block, value
      if constexpr (cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD_MULTICAST>) {
        auto block_layout = Layout<typename DispatchPolicy::ClusterShape>{}; // (m,n) -> block_id
        for (int n = 0; n < size<1>(block_layout); ++n) {
          mcast_mask_a |= (uint16_t(1) << block_layout(cluster_local_block_id.x,n,Int<0>{}));
        }
      }

      if constexpr (cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD_MULTICAST>) {
        auto block_layout = Layout<typename DispatchPolicy::ClusterShape>{}; // (m,n) -> block_id
        for (int m = 0; m < size<0>(block_layout); ++m) {
          mcast_mask_b |= (uint16_t(1) << block_layout(m,cluster_local_block_id.y,Int<0>{}));
        }
      }
      int presum_k_iter = 0;
      bool ComputesPresum = false;
      bool validTB_A = StrassenMiGroup::hasM0() && sub_m_idx == 0 && iter_PresumA_M.validTB();
      bool validTB_B = StrassenMiGroup::hasM1() && sub_m_idx == 1 && iter_PresumB_M.validTB();

      if ((StrassenMiGroup::hasM0() && StrassenMiGroup::AllPresums::computeAnyAPresum()) ||
          (StrassenMiGroup::hasM1() && StrassenMiGroup::AllPresums::computeAnyBPresum())) {
        if (is_fused) {
          if ((sub_m_idx == 0 && StrassenMiGroup::AllPresums::computeAnyAPresum() && validTB_A) || 
              (sub_m_idx == 1 && StrassenMiGroup::AllPresums::computeAnyBPresum() && validTB_B)) {
            ComputesPresum = true;
            pipeline.params_.transaction_bytes = TmaTransactionBytesMK+TmaTransactionBytesNK +
                                                 ((sub_m_idx == 0) ? 4*TmaTransactionBytesPresumA : 4*TmaTransactionBytesPresumB);
          } else {
            pipeline.params_.transaction_bytes = TmaTransactionBytesMK+TmaTransactionBytesNK;
          }
        } else if (StrassenMiGroup::hasM0() && StrassenMiGroup::AllPresums::computeAnyAPresum() && validTB_A) {
          ComputesPresum = true;
          pipeline.params_.transaction_bytes = TmaTransactionBytesMK+TmaTransactionBytesNK + 4*TmaTransactionBytesPresumA;
        } else if (StrassenMiGroup::hasM1() && StrassenMiGroup::AllPresums::computeAnyBPresum() && validTB_B) {
          ComputesPresum = true;
          pipeline.params_.transaction_bytes = TmaTransactionBytesMK+TmaTransactionBytesNK + 4*TmaTransactionBytesPresumB;
        } else
            pipeline.params_.transaction_bytes = TmaTransactionBytesMK+TmaTransactionBytesNK;
        // if (k_tile_count >= MAX(kPresumComputeIterationsA, kPresumComputeIterationsB)) {
        // } else __builtin_unreachable();
      }

      //Either presum_tile_log_multiplier_a is >= 1 or presum_tile_log_divider_a >= 1 both cannot happen the same type
      const uint presumComputeIterationsA = (kPresumComputeIterationsA * (1 << mainloop_params.get_presum_tile_log_multiplier_a())) >> mainloop_params.get_presum_tile_log_divider_a();
      const uint presumComputeIterationsB = (kPresumComputeIterationsB * (1 << mainloop_params.get_presum_tile_log_multiplier_b())) >> mainloop_params.get_presum_tile_log_divider_b();
      const uint presumComputeIterationsAB = (StrassenMiGroup::hasM0() && sub_m_idx == 0) ? presumComputeIterationsA : presumComputeIterationsB;

      // Mainloop
      CUTLASS_PRAGMA_NO_UNROLL
      for ( ; k_tile_count > 0; --k_tile_count) {
        if (ComputesPresum && presum_k_iter >= presumComputeIterationsAB)
          pipeline.params_.transaction_bytes = TmaTransactionBytesMK+TmaTransactionBytesNK;

        // LOCK smem_pipe_write for _writing_
        pipeline.producer_acquire(smem_pipe_write);

        //
        // Copy gmem to smem for *k_tile_iter
        //

        using BarrierType = typename MainloopPipeline::ProducerBarrierType;
        BarrierType* tma_barrier = pipeline.producer_get_barrier(smem_pipe_write);
        
        int write_stage = smem_pipe_write.index();

        if (ComputesPresum && presum_k_iter >= PresumStages && presum_k_iter < presumComputeIterationsAB + PresumStages) {
          if (sub_m_idx == 0 && validTB_A) {
            int presum_write_iter = presum_k_iter - PresumStages;
            int presum_tile = presum_write_iter/kPresumComputeIterationsA;
            //Using % makes code slower NVCC 12.9
            presum_write_iter = presum_write_iter - presum_tile*kPresumComputeIterationsA;
            iter_PresumA_M.reset(presum_tile);
            iter_PresumA_M.set_iteration(presum_write_iter);

            for (int wid = 0; wid < 4; wid++) {
              auto smem_dst_ptr = shared_presum_tensors.smem_A0.data() +
                                  write_stage*PresumSingleStageSize +
                                  wid * PresumStages * PresumSingleStageSize +
                                  (0) * sizeof(PresumStoreVecType)/sizeof(ElementA);
              ElementA* st_ptr = ((ElementA*)iter_PresumA_M.get(0)) + wid*iter_PresumA_M.extent.row()*iter_PresumA_M.extent.column();

              SM90_TMA_STORE().copy(mainloop_params.tma_store_presumld_a.get_tma_descriptor(), smem_dst_ptr,
                                    presum_tile*size<1>(TileShape{}) + iter_PresumA_M.tb_offset.column(),
                                    wid*iter_PresumA_M.extent.row() +
                                    iter_PresumA_M.tb_offset.row() + presum_write_iter*size<0>(PresumTileShapeA{}), 0);
              asm volatile("cp.async.bulk.commit_group;");
            }
          } else if (sub_m_idx == 1 && validTB_B) {
            int presum_write_iter = presum_k_iter - PresumStages;
            int presum_tile = presum_write_iter/kPresumComputeIterationsB;
            presum_write_iter = presum_write_iter - presum_tile*kPresumComputeIterationsB;
            iter_PresumB_M.reset(presum_tile);
            iter_PresumB_M.set_iteration(presum_write_iter);

            for (int wid = 0; wid < 4; wid++) {
              auto smem_dst_ptr = shared_presum_tensors.smem_A0.data() +
                                  write_stage*PresumSingleStageSize +
                                  wid * PresumStages * PresumSingleStageSize +
                                  (0) * sizeof(PresumStoreVecType)/sizeof(ElementA);
              ElementA* st_ptr = ((ElementA*)iter_PresumB_M.get(0)) +
                                  wid*iter_PresumB_M.extent.row()*iter_PresumB_M.extent.column();

              SM90_TMA_STORE().copy(mainloop_params.tma_store_presumld_b.get_tma_descriptor(), smem_dst_ptr,
                                    iter_PresumB_M.tb_offset.column(),
                                    wid*iter_PresumB_M.extent.row() +
                                    iter_PresumB_M.tb_offset.row() + presum_tile*size<0>(TileShape{}) + presum_write_iter*size<0>(PresumTileShapeB{}), 0);
              asm volatile("cp.async.bulk.commit_group;");
            }
          } else CUTE_GCC_UNREACHABLE;
        }

        //Using this is the problem
        copy(tma_load_a.with(*tma_barrier, mcast_mask_a), tAgA(_,_,_,*k_tile_iter), tAsA(_,_,_,write_stage));
        copy(tma_load_b.with(*tma_barrier, mcast_mask_b), tBgB(_,_,_,*k_tile_iter), tBsB(_,_,_,write_stage));
        // copy(tma_load_b.with(*tma_barrier, mcast_mask_b), tBgB(_,_,_,*k_tile_iter), tBsB(_,_,_,write_stage));
        if (sub_m_idx == 0 && validTB_A && StrassenMiGroup::hasM0() &&
            StrassenMiGroup::AllPresums::computeAnyAPresum() && presum_k_iter < presumComputeIterationsA) {
          uint RR = kPresumComputeIterationsA;
          uint RR2 = presumComputeIterationsA;
          uint presum_tile = presum_k_iter/kPresumComputeIterationsA;
          uint presum_tile_k_iter = presum_k_iter - presum_tile*kPresumComputeIterationsA;
          uint presum_row = m_coord*RR + n_coord_div*RR2 + presum_tile_k_iter;
          uint presum_col = presum_tile + new_n_coord;
          Tensor gA0_tile = local_tile(gA0, PresumTileShapeA{}, make_coord(presum_row, presum_col, 0));
          Tensor gA1_tile = local_tile(gA1, PresumTileShapeA{}, make_coord(presum_row, presum_col, 0));
          Tensor gA2_tile = local_tile(gA2, PresumTileShapeA{}, make_coord(presum_row, presum_col, 0));
          Tensor gA3_tile = local_tile(gA3, PresumTileShapeA{}, make_coord(presum_row, presum_col, 0));
          
          Tensor tAgA0 = block_tma_presum_ld_a.partition_S(gA0_tile);
          Tensor tAgA1 = block_tma_presum_ld_a.partition_S(gA1_tile);
          Tensor tAgA2 = block_tma_presum_ld_a.partition_S(gA2_tile);
          Tensor tAgA3 = block_tma_presum_ld_a.partition_S(gA3_tile);

          // Tensor gA0_tile2 = local_tile(gA0_tile, PresumTileShapeA{}, make_coord(presum_k_iter,_,_));
          Tensor sA0_tile = make_tensor(make_smem_ptr(shared_presum_tensors.smem_A0.data() + write_stage*PresumSingleStageSize), PresumSmemLayoutA__{});//local_tile(sPresumA0, PresumSmemShapeA__{}, make_coord(_,_,write_stage));
          Tensor sA1_tile = make_tensor(make_smem_ptr(shared_presum_tensors.smem_A1.data() + write_stage*PresumSingleStageSize), PresumSmemLayoutA__{});
          Tensor sA2_tile = make_tensor(make_smem_ptr(shared_presum_tensors.smem_A2.data() + write_stage*PresumSingleStageSize), PresumSmemLayoutA__{});
          Tensor sA3_tile = make_tensor(make_smem_ptr(shared_presum_tensors.smem_A3.data() + write_stage*PresumSingleStageSize), PresumSmemLayoutA__{});
          // typename decltype(sA0_tile)::x y;
          // typename decltype(gA0_tile2)::x z;
          
          Tensor tAsA0 = block_tma_presum_ld_a.partition_D(sA0_tile);
          Tensor tAsA1 = block_tma_presum_ld_a.partition_D(sA1_tile);
          Tensor tAsA2 = block_tma_presum_ld_a.partition_D(sA2_tile);
          Tensor tAsA3 = block_tma_presum_ld_a.partition_D(sA3_tile);

          cute::tma_store_wait<3>();
          copy(mainloop_params.tma_load_presumld_a.with(*tma_barrier),
                tAgA0,
                tAsA0(_,_,_));
          cute::tma_store_wait<2>();
          copy(mainloop_params.tma_load_presumld_a.with(*tma_barrier),
                tAgA1,
                tAsA1(_,_,_));
          cute::tma_store_wait<1>();
          copy(mainloop_params.tma_load_presumld_a.with(*tma_barrier),
                tAgA2,
                tAsA2(_,_,_));
          cute::tma_store_wait<0>();
          copy(mainloop_params.tma_load_presumld_a.with(*tma_barrier),
                tAgA3,
                tAsA3(_,_,_));
          // copy(mainloop_params.tma_load_presumld_a.with(*tma_barrier), block_tma_presum_ld_a.partition_S(gA1_), block_tma_presum_ld_a.partition_D(sPresumA1));
          // copy(mainloop_params.tma_load_presumld_a.with(*tma_barrier), block_tma_presum_ld_a.partition_S(gA2_), block_tma_presum_ld_a.partition_D(sPresumA2));
          // copy(mainloop_params.tma_load_presumld_a.with(*tma_barrier), block_tma_presum_ld_a.partition_S(gA3_), block_tma_presum_ld_a.partition_D(sPresumA3));
        }
        if (((StrassenMiGroup::hasM1() && !is_fused) || (is_fused && sub_m_idx == 1)) && validTB_B &&
            StrassenMiGroup::AllPresums::computeAnyBPresum() && presum_k_iter < presumComputeIterationsB) {
          uint RR = presumComputeIterationsB;
          uint presum_tile = presum_k_iter/kPresumComputeIterationsB;
          uint presum_tile_k_iter = presum_k_iter - presum_tile*kPresumComputeIterationsB;
          uint presum_row = m_coord*RR + presum_tile * kPresumComputeIterationsB + presum_tile_k_iter;
          uint presum_col = n_coord;
          Tensor gB0_tile = local_tile(gB0, PresumTileShapeB{}, make_coord(presum_row, presum_col, 0));
          Tensor gB1_tile = local_tile(gB1, PresumTileShapeB{}, make_coord(presum_row, presum_col, 0));
          Tensor gB2_tile = local_tile(gB2, PresumTileShapeB{}, make_coord(presum_row, presum_col, 0));
          Tensor gB3_tile = local_tile(gB3, PresumTileShapeB{}, make_coord(presum_row, presum_col, 0));
          
          Tensor tBgB0 = block_tma_presum_ld_b.partition_S(gB0_tile);
          Tensor tBgB1 = block_tma_presum_ld_b.partition_S(gB1_tile);
          Tensor tBgB2 = block_tma_presum_ld_b.partition_S(gB2_tile);
          Tensor tBgB3 = block_tma_presum_ld_b.partition_S(gB3_tile);

          // Tensor gA0_tile2 = local_tile(gA0_tile, PresumTileShapeA{}, make_coord(presum_k_iter,_,_));
          Tensor sB0_tile = make_tensor(make_smem_ptr(shared_presum_tensors.smem_A0.data() + write_stage*PresumSingleStageSize), PresumSmemLayoutB__{});//local_tile(sPresumA0, PresumSmemShapeA__{}, make_coord(_,_,write_stage));
          Tensor sB1_tile = make_tensor(make_smem_ptr(shared_presum_tensors.smem_A1.data() + write_stage*PresumSingleStageSize), PresumSmemLayoutB__{});
          Tensor sB2_tile = make_tensor(make_smem_ptr(shared_presum_tensors.smem_A2.data() + write_stage*PresumSingleStageSize), PresumSmemLayoutB__{});
          Tensor sB3_tile = make_tensor(make_smem_ptr(shared_presum_tensors.smem_A3.data() + write_stage*PresumSingleStageSize), PresumSmemLayoutB__{});
          // typename decltype(sA0_tile)::x y;
          // typename decltype(gA0_tile2)::x z;
          
          Tensor tBsB0 = block_tma_presum_ld_b.partition_D(sB0_tile);
          Tensor tBsB1 = block_tma_presum_ld_b.partition_D(sB1_tile);
          Tensor tBsB2 = block_tma_presum_ld_b.partition_D(sB2_tile);
          Tensor tBsB3 = block_tma_presum_ld_b.partition_D(sB3_tile);

          cute::tma_store_wait<3>();
          copy(mainloop_params.tma_load_presumld_b.with(*tma_barrier),
                tBgB0,
                tBsB0(_,_,_));
          cute::tma_store_wait<2>();
          copy(mainloop_params.tma_load_presumld_b.with(*tma_barrier),
                tBgB1,
                tBsB1(_,_,_));
          cute::tma_store_wait<1>();
          copy(mainloop_params.tma_load_presumld_b.with(*tma_barrier),
                tBgB2,
                tBsB2(_,_,_));
          cute::tma_store_wait<0>();
          copy(mainloop_params.tma_load_presumld_b.with(*tma_barrier),
                tBgB3,
                tBsB3(_,_,_));
        }
        ++k_tile_iter;
        ++presum_k_iter;
        // Advance smem_pipe_write
        ++smem_pipe_write;
      }

      if (ComputesPresum) cute::tma_store_wait<0>();
    }
  }

  /// Perform a Producer Epilogue to prevent early exit of blocks in a Cluster
  CUTLASS_DEVICE void
  load_tail(MainloopPipeline pipeline, PipelineState smem_pipe_write) {
    int lane_predicate = cute::elect_one_sync();

    // Issue the epilogue waits
    if (lane_predicate) {
      /* This helps avoid early exit of blocks in Cluster
       * Waits for all stages to either be released (all
       * Consumer UNLOCKs), or if the stage was never used
       * then would just be acquired since the phase was
       * still inverted from make_producer_start_state
       */
      pipeline.producer_tail(smem_pipe_write);
    }
  }

  template<class PresumOutputs>
  CUTLASS_DEVICE void
  presum_compute_store(int presumComputeIterationsA, int thread_idx, uint presum_read_stage, uint presum_write_stage, int presumIter, uint m_coord, uint n_coord, int sub_m_idx,
                       PresumGlobalIteratorA& iter_PresumA_M,
                       PresumTensorStorage& shared_presum_tensors, 
                       PresumTensorStorage2& shared_presum_tensors2,
                       PresumOutputs& all_presum_outputs, Params const& mainloop_params) {
    // uint thread_idx = threadIdx.x - 128;
    if (StrassenMiGroup::hasM0() && StrassenMiGroup::AllPresums::computeAnyAPresum() && 
        presumIter < presumComputeIterationsA*1 and thread_idx < 128) {
      //This code above mac_loop_iter gives some improvement.
      //Changes done after commit: 853df006e0f2bfad3313460b2fcfdabb15d31067

      for (int v = 0; v < 1; v += 1) {
        // iter_PresumA_M.reset();
        // iter_PresumA_M.row += presumIter * iter_PresumA_M.row_increment();

        PresumVecTypeA a0; a0.clear();
        PresumVecTypeA a1; a1.clear();
        PresumVecTypeA a2; a2.clear();
        PresumVecTypeA a3; a3.clear();

        auto smem_a0_ptr = shared_presum_tensors.smem_A0.data() + presum_read_stage*PresumSingleStageSize + thread_idx * sizeof(PresumVecTypeA)/sizeof(ElementA);
        auto smem_a1_ptr = shared_presum_tensors.smem_A1.data() + presum_read_stage*PresumSingleStageSize + thread_idx * sizeof(PresumVecTypeA)/sizeof(ElementA);
        auto smem_a2_ptr = shared_presum_tensors.smem_A2.data() + presum_read_stage*PresumSingleStageSize + thread_idx * sizeof(PresumVecTypeA)/sizeof(ElementA);
        auto smem_a3_ptr = shared_presum_tensors.smem_A3.data() + presum_read_stage*PresumSingleStageSize + thread_idx * sizeof(PresumVecTypeA)/sizeof(ElementA);
        // PresumDetail::shared_load_32b(&a0, smem_a0_ptr);
        // PresumDetail::shared_load_32b(&a1, smem_a1_ptr);
        // PresumDetail::shared_load_32b(&a2, smem_a2_ptr);
        // PresumDetail::shared_load_32b(&a3, smem_a3_ptr);

        a0 = *(PresumVecTypeA*)smem_a0_ptr;
        a1 = *(PresumVecTypeA*)smem_a1_ptr;
        a2 = *(PresumVecTypeA*)smem_a2_ptr;
        a3 = *(PresumVecTypeA*)smem_a3_ptr;


        // if (presumAComputeLoads.hasAccess(MmaStrassen::APresums::A0))
      //   PresumDetail::shared_load_128b(&a0, sharedPreSums.get(presumAComputeLoads.indzex(MmaStrassen::APresums::A0), presum_read_stage, 0));
        // if (presumAComputeLoads.hasAccess(MmaStrassen::APresums::A1))
        //   PresumDetail::shared_load_128b(&a1, sharedPreSums.get(presumAComputeLoads.index(MmaStrassen::APresums::A1), presum_read_stage, 0));
        // if (presumAComputeLoads.hasAccess(MmaStrassen::APresums::A2))
        //   PresumDetail::shared_load_128b(&a2, sharedPreSums.get(presumAComputeLoads.index(MmaStrassen::APresums::A2), presum_read_stage, 0));
        // if (presumAComputeLoads.hasAccess(MmaStrassen::APresums::A3))
        //   PresumDetail::shared_load_128b(&a3, sharedPreSums.get(presumAComputeLoads.index(MmaStrassen::APresums::A3), presum_read_stage, 0));

        PresumIOToComputeTypeA presum_io_to_compute_type;
        PresumComputeToIOTypeA presum_compute_to_io_type;

        auto s1   = presum_io_to_compute_type(a2) + presum_io_to_compute_type(a3);
        auto s2   = s1 - presum_io_to_compute_type(a0);
        auto a02  = presum_io_to_compute_type(a0) - presum_io_to_compute_type(a2);
        auto a1s2 = presum_io_to_compute_type(a1) - s2;


        // Tensor sA02_tile = make_tensor(make_smem_ptr(smem_a02_ptr), PresumSmemLayoutA__{});//local_tile(sPresumA0, PresumSmemShapeA__{}, make_coord(_,_,write_stage));
        // Tensor sS1_tile = make_tensor(make_smem_ptr(smem_s1_ptr), PresumSmemLayoutA__{});
        // Tensor sS2_tile = make_tensor(make_smem_ptr(smem_s2_ptr), PresumSmemLayoutA__{});
        // Tensor sA1S2_tile = make_tensor(make_smem_ptr(smem_a1s2_ptr), PresumSmemLayoutA__{});

        // auto s1   = a2 + a3;
        // auto s2   = s1 - a0;
        // auto a02  = a0 - a2;
        // auto a1s2 = a1 - s2;

        // using AllPresums = typename StrassenMiGroup::AllPresums;
        
        // ThrCopy thrblk_s2g = mainloop_params.tma_store_presumld_a.get_slice(Int<0>{});
        
        // Tensor tAsA02 = thrblk_s2g.partition_S(sA02_tile);                                    // (S2G,S2G_M,S2G_N,EPI_M,EPI_N)
        // Tensor tAsS1 = thrblk_s2g.partition_S(sS1_tile);                                    // (S2G,S2G_M,S2G_N,EPI_M,EPI_N)
        // Tensor tAsS2 = thrblk_s2g.partition_S(sS2_tile);                                    // (S2G,S2G_M,S2G_N,EPI_M,EPI_N)
        // Tensor tAsA1S2 = thrblk_s2g.partition_S(sA1S2_tile);                                    // (S2G,S2G_M,S2G_N,EPI_M,EPI_N)

        // Tensor gA02 = get<0>(all_presum_outputs);
        // Tensor gS1 = get<1>(all_presum_outputs);
        // Tensor gS2 = get<2>(all_presum_outputs);
        // Tensor gA1S2 = get<3>(all_presum_outputs);

        // Tensor gA02_tile = local_tile(gA02, PresumTileShapeA{}, make_coord(m_coord*16 + presumIter, n_coord, 0));
        // Tensor gS1_tile = local_tile(gS1, PresumTileShapeA{}, make_coord(m_coord*16 + presumIter, n_coord, 0));
        // Tensor gS2_tile = local_tile(gS2, PresumTileShapeA{}, make_coord(m_coord*16 + presumIter, n_coord, 0));
        // Tensor gA1S2_tile = local_tile(gA1S2, PresumTileShapeA{}, make_coord(m_coord*16 + presumIter, n_coord, 0));
        
        // Tensor tgA02 = thrblk_s2g.partition_D(gA02_tile);
        // Tensor tgS1 = thrblk_s2g.partition_D(gS1_tile);
        // Tensor tgS2 = thrblk_s2g.partition_D(gS2_tile);
        // Tensor tgA1S2 = thrblk_s2g.partition_D(gA1S2_tile);

        // if (cute::elect_one_sync()) {
        //   copy(mainloop_params.tma_store_presumld_a,
        //        tAsA02(_,_,_), tgA02);
        //   copy(mainloop_params.tma_store_presumld_a,
        //        tAsS1(_,_,_), tgS1);
        //   copy(mainloop_params.tma_store_presumld_a,
        //        tAsS2(_,_,_), tgS2);
        //   copy(mainloop_params.tma_store_presumld_a,
        //        tAsA1S2(_,_,_), tgA1S2);
        // }

        if (true) {
          auto smem_a02_ptr = shared_presum_tensors.smem_A0.data() + presum_read_stage * PresumSingleStageSize + thread_idx * sizeof(PresumVecTypeA)/sizeof(ElementA);
          // presum_read_stage*size<0>(PresumTileShapeA{})*size<1>(PresumTileShapeA{}) ;
          auto smem_s1_ptr = shared_presum_tensors.smem_A1.data()  + presum_read_stage * PresumSingleStageSize + thread_idx * sizeof(PresumVecTypeA)/sizeof(ElementA);
          auto smem_s2_ptr = shared_presum_tensors.smem_A2.data() + presum_read_stage * PresumSingleStageSize + thread_idx * sizeof(PresumVecTypeA)/sizeof(ElementA);
          auto smem_a1s2_ptr = shared_presum_tensors.smem_A3.data() + presum_read_stage * PresumSingleStageSize + thread_idx * sizeof(PresumVecTypeA)/sizeof(ElementA);

          // if (thread_idx % 32 == 0) cute::tma_store_wait<0>();
          // asm volatile("bar.cta.sync %0, %1;" : : "r"(5), "r"(128));

          // if (threadIdx.x==128 && m_coord == 0) {
          //   printf("1229 %d,%d : %f\n", n_coord*128, presumIter, s2[0]);
          // }
          *(PresumVecTypeA*)smem_a02_ptr = presum_compute_to_io_type(a02);
          *(PresumVecTypeA*)smem_s1_ptr = presum_compute_to_io_type(s1);
          *(PresumVecTypeA*)smem_s2_ptr = presum_compute_to_io_type(s2);
          *(PresumVecTypeA*)smem_a1s2_ptr = presum_compute_to_io_type(a1s2);

          //Do not need this before pipeline.cosumer_release would synchronize
          // asm volatile("bar.cta.sync %0, %1;" : : "r"(4), "r"(128));

          return;
          int wid = thread_idx / 32;
          auto smem_dst_ptr = shared_presum_tensors.smem_A0.data() + presum_read_stage*size<0>(PresumTileShapeA{})*size<1>(PresumTileShapeA{}) + wid * PresumStages*size<0>(PresumTileShapeA{})*size<1>(PresumTileShapeA{}) + (thread_idx%32) * sizeof(PresumStoreVecType)/sizeof(ElementA);

          PresumStoreVecType e = *(PresumStoreVecType*)smem_dst_ptr;

          arch::global_store<PresumStoreVecType, sizeof(PresumStoreVecType)>(e, ((ElementA*)iter_PresumA_M.get(0)) + wid*iter_PresumA_M.extent.row()*iter_PresumA_M.extent.column(),
                                                                            iter_PresumA_M.validTB());
          return;
        }

        if (AllPresums::doesComputeA(MmaStrassen::APresums::A02)) {
          arch::global_store<PresumVecTypeA, sizeof(PresumVecTypeA)>(presum_compute_to_io_type(a02), iter_PresumA_M.get(AllPresums::indexAPresum(MmaStrassen::APresums::A02)),
                                                                    iter_PresumA_M.validTB());
        }

        if (AllPresums::doesComputeA(MmaStrassen::APresums::S1)) {
          // auto f16s1 = presum_compute_to_io_type(s1);
          // if (presumIter == 0 && threadIdx.x < 256 && iter_PresumA_M.tb_offset.row() == 0 && iter_PresumA_M.tb_offset.column() == 0)
          //   printf("848 %d %p : %f %f\n", threadIdx.x, iter_PresumA_M.get(AllPresums::indexAPresum(MmaStrassen::APresums::S1)), float(f16s1[0]), float(f16s1[1]));
          arch::global_store<PresumVecTypeA, sizeof(PresumVecTypeA)>(presum_compute_to_io_type(s1), iter_PresumA_M.get(AllPresums::indexAPresum(MmaStrassen::APresums::S1)),
                                                                    iter_PresumA_M.validTB());
        }
        if (AllPresums::doesComputeA(MmaStrassen::APresums::S2)) {
          arch::global_store<PresumVecTypeA, sizeof(PresumVecTypeA)>(presum_compute_to_io_type(s2), iter_PresumA_M.get(AllPresums::indexAPresum(MmaStrassen::APresums::S2)),
                                                                    iter_PresumA_M.validTB());
        }
        if (AllPresums::doesComputeA(MmaStrassen::APresums::A1S2)) {
          arch::global_store<PresumVecTypeA, sizeof(PresumVecTypeA)>(presum_compute_to_io_type(a1s2), iter_PresumA_M.get(AllPresums::indexAPresum(MmaStrassen::APresums::A1S2)),
                                                                    iter_PresumA_M.validTB());
        }

        // iter_PresumA_M.inc();
      }
    }
  }

  template<class PresumOutputs>
  CUTLASS_DEVICE void
  presum_compute_store_b(int presumComputeIterationsB, int thread_idx, uint presum_read_stage, uint presum_write_stage, int presumIter, uint m_coord, uint n_coord, int sub_m_idx,
                       PresumGlobalIteratorB iter_PresumB_M,
                       PresumTensorStorage& shared_presum_tensors, 
                       PresumTensorStorage2& shared_presum_tensors2,
                       PresumOutputs& all_presum_outputs, Params const& mainloop_params) {
    if (((StrassenMiGroup::hasM1() && !is_fused) || is_fused) && 
        StrassenMiGroup::AllPresums::computeAnyBPresum() &&
        presumIter < presumComputeIterationsB*1) {
      //This code above mac_loop_iter gives some improvement.
      //Changes done after commit: 853df006e0f2bfad3313460b2fcfdabb15d31067

      for (int v = 0; v < 1; v += 1) {
        // iter_PresumA_M.reset();
        // iter_PresumA_M.row += presumIter * iter_PresumA_M.row_increment();

        PresumVecTypeB b0; b0.clear();
        PresumVecTypeB b1; b1.clear();
        PresumVecTypeB b2; b2.clear();
        PresumVecTypeB b3; b3.clear();

        auto smem_b0_ptr = shared_presum_tensors.smem_A0.data() + presum_read_stage*PresumSingleStageSize + thread_idx * sizeof(PresumVecTypeB)/sizeof(ElementA);
        auto smem_b1_ptr = shared_presum_tensors.smem_A1.data() + presum_read_stage*PresumSingleStageSize + thread_idx * sizeof(PresumVecTypeB)/sizeof(ElementA);
        auto smem_b2_ptr = shared_presum_tensors.smem_A2.data() + presum_read_stage*PresumSingleStageSize + thread_idx * sizeof(PresumVecTypeB)/sizeof(ElementA);
        auto smem_b3_ptr = shared_presum_tensors.smem_A3.data() + presum_read_stage*PresumSingleStageSize + thread_idx * sizeof(PresumVecTypeB)/sizeof(ElementA);

        b0 = *(PresumVecTypeB*)smem_b0_ptr;
        b1 = *(PresumVecTypeB*)smem_b1_ptr;
        b2 = *(PresumVecTypeB*)smem_b2_ptr;
        b3 = *(PresumVecTypeB*)smem_b3_ptr;

        PresumIOToComputeTypeB presum_io_to_compute_type;
        PresumComputeToIOTypeB presum_compute_to_io_type;

        auto b31   = presum_io_to_compute_type(b3) - presum_io_to_compute_type(b1);
        auto b10   = presum_io_to_compute_type(b1) - presum_io_to_compute_type(b0);
        auto s3    = b31 + presum_io_to_compute_type(b0);
        auto s3b2  = s3 - presum_io_to_compute_type(b2);

        if (true) {
          auto smem_b31_ptr = shared_presum_tensors.smem_A0.data() + presum_read_stage * PresumSingleStageSize + thread_idx * sizeof(PresumVecTypeB)/sizeof(ElementA);
          // presum_read_stage*size<0>(PresumTileShapeA{})*size<1>(PresumTileShapeA{}) ;
          auto smem_b10_ptr = shared_presum_tensors.smem_A1.data()  + presum_read_stage * PresumSingleStageSize + thread_idx * sizeof(PresumVecTypeB)/sizeof(ElementA);
          auto smem_s3_ptr = shared_presum_tensors.smem_A2.data() + presum_read_stage * PresumSingleStageSize + thread_idx * sizeof(PresumVecTypeB)/sizeof(ElementA);
          auto smem_s3b2_ptr = shared_presum_tensors.smem_A3.data() + presum_read_stage * PresumSingleStageSize + thread_idx * sizeof(PresumVecTypeB)/sizeof(ElementA);

          // if (thread_idx % 32 == 0) cute::tma_store_wait<0>();
          // asm volatile("bar.cta.sync %0, %1;" : : "r"(5), "r"(128));

          *(PresumVecTypeB*)smem_b31_ptr = presum_compute_to_io_type(b31);
          *(PresumVecTypeB*)smem_b10_ptr = presum_compute_to_io_type(b10);
          *(PresumVecTypeB*)smem_s3_ptr = presum_compute_to_io_type(s3);
          *(PresumVecTypeB*)smem_s3b2_ptr = presum_compute_to_io_type(s3b2);

          // asm volatile("bar.cta.sync %0, %1;" : : "r"(4), "r"(128));
        }

        // iter_PresumA_M.inc();
      }
    }
  }

  template<class PresumOutputs>
  CUTLASS_DEVICE void
  presum_compute_store2(int presumComputeIterationsA, int thread_idx, uint presum_write_stage, bool last_presum_iters, int presumIter, uint m_coord, uint n_coord,
                       PresumGlobalIteratorA iter_PresumA_M,
                       PresumTensorStorage& shared_presum_tensors, 
                       PresumTensorStorage2& shared_presum_tensors2,
                       PresumOutputs& all_presum_outputs, Params const& mainloop_params,
                       PresumStoreVecType& e2) {
    // uint thread_idx = threadIdx.x - 128;
    if (StrassenMiGroup::hasM0() && StrassenMiGroup::AllPresums::computeAnyAPresum() &&
        presumIter < presumComputeIterationsA*1 and thread_idx < 128) {
        if (true) {
          int presum_tile = presumIter/kPresumComputeIterationsA;
          presumIter = presumIter - presum_tile*kPresumComputeIterationsA;
          iter_PresumA_M.reset(presum_tile);
          iter_PresumA_M.set_iteration(presumIter);
          // asm volatile("wgmma.fence.sync.aligned;");
          // __syncwarp();
          // asm volatile("bar.sync %0, %1;" : : "r"(3), "r"(128));

          PresumStoreVecType data[kPresumThreads/32];

          //TODO: Following code is probably wrong but it works when PresumShapeTile's row is 2 and 4
          if (kPresumThreads < 128 /*NumThreadsPerWarpGroup*/) {
            if (last_presum_iters) {
              #pragma unroll (kPresumThreads/32)
              for (int wid_ = 0; wid_ < kPresumThreads/32; wid_++) {
                int wid = wid_*(kPresumThreads/32) + thread_idx / kPresumThreads;
                auto smem_dst_ptr = shared_presum_tensors.smem_A0.data() +
                                    presum_write_stage * PresumSingleStageSize +
                                    wid * PresumSmemSize +
                                    (thread_idx%kPresumThreads) * sizeof(PresumStoreVecType)/sizeof(ElementA);

                data[wid_] = *(PresumStoreVecType*)smem_dst_ptr;
              }
            }

            // asm volatile("bar.cta.sync %0, %1;" : : "r"(3), "r"(128));

            if (last_presum_iters) {
              #pragma unroll (kPresumThreads/32)
              for (int wid_ = 0; wid_ < kPresumThreads/32; wid_++) {
                int wid = wid_*(kPresumThreads/32) + thread_idx / kPresumThreads;
                arch::global_store<PresumStoreVecType, sizeof(PresumStoreVecType)>(data[wid_], ((ElementA*)iter_PresumA_M.get(0)) + wid*iter_PresumA_M.extent.row()*iter_PresumA_M.extent.column(),
                                                                                  iter_PresumA_M.validTB() && iter_PresumA_M.valid());
              }
            }
          } else {
            if (last_presum_iters) {
              #pragma unroll (kPresumThreads/32)
              for (int wid_ = 0; wid_ < kPresumThreads/32; wid_++) {
                int wid = wid_;
                auto smem_dst_ptr = shared_presum_tensors.smem_A0.data() +
                                    presum_write_stage * PresumSingleStageSize +
                                    wid * PresumSmemSize +
                                    (thread_idx%kPresumThreads) * sizeof(PresumStoreVecType)/sizeof(ElementA);

                data[wid_] = *(PresumStoreVecType*)smem_dst_ptr;
              }
            }

            // asm volatile("bar.cta.sync %0, %1;" : : "r"(3), "r"(128));

            if (last_presum_iters) {
              #pragma unroll (kPresumThreads/32)
              for (int wid_ = 0; wid_ < kPresumThreads/32; wid_++) {
                int wid = wid_;
                arch::global_store<PresumStoreVecType, sizeof(PresumStoreVecType)>(data[wid_], ((ElementA*)iter_PresumA_M.get(0)) + wid*iter_PresumA_M.extent.row()*iter_PresumA_M.extent.column(),
                                                                                  iter_PresumA_M.validTB() && iter_PresumA_M.valid());
              }
            }
          }

          // if (false && thread_idx % kPresumThreads == 0) {
          // //Example of using TMA store
          //   ElementA* st_ptr = ((ElementA*)iter_PresumA_M.get(0)) + wid*iter_PresumA_M.extent.row()*iter_PresumA_M.extent.column();
          //   ElementA* st_ptr_2 = ((ElementA*)iter_PresumA_M.get(0)) + (wid+2)*iter_PresumA_M.extent.row()*iter_PresumA_M.extent.column();
            
          //   SM90_TMA_STORE().copy(mainloop_params.tma_store_presumld_a.get_tma_descriptor(), smem_dst_ptr,  
          //   iter_PresumA_M.tb_offset.column(),
          //   wid*iter_PresumA_M.extent.row() +
          //   iter_PresumA_M.tb_offset.row() + presumIter*size<0>(PresumTileShapeA{}), 0); //iter_PresumA_M.row, iter_PresumA_M.col);
            // asm volatile("cp.async.bulk.commit_group;");
          // }
          // return;
        }

        // iter_PresumA_M.inc();
      }
    }

  template<class PresumOutputs>
  CUTLASS_DEVICE void
  presum_compute_store2_b(int presumComputeIterationsB, int thread_idx, uint presum_write_stage, bool last_presum_iters, int presumIter, uint m_coord, uint n_coord,
                       PresumGlobalIteratorB iter_PresumB_M,
                       PresumTensorStorage& shared_presum_tensors, 
                       PresumTensorStorage2& shared_presum_tensors2,
                       PresumOutputs& all_presum_outputs, Params const& mainloop_params,
                       PresumStoreVecType& e2) {
    // uint thread_idx = threadIdx.x - 128;
    if (((StrassenMiGroup::hasM1() && !is_fused) || is_fused) &&
        StrassenMiGroup::AllPresums::computeAnyBPresum() &&
        presumIter < presumComputeIterationsB*1) {
        if (true) {
          int presum_tile = presumIter/kPresumComputeIterationsB;
          presumIter = presumIter - presum_tile*kPresumComputeIterationsB;
          iter_PresumB_M.reset(presum_tile);
          iter_PresumB_M.set_iteration(presumIter);

          // asm volatile("wgmma.fence.sync.aligned;");
          // __syncwarp();
          // asm volatile("bar.sync %0, %1;" : : "r"(3), "r"(128));

          PresumStoreVecType data[kPresumThreads/32];
          if (kPresumThreads < 128) {
            // data[0].fill(ElementA(0));
            //TODO: Following code is probably wrong but it works when PresumShapeTile's row is 2 and 4
            if (last_presum_iters) {
              #pragma unroll (kPresumThreads/32)
              for (int wid_ = 0; wid_ < kPresumThreads/32; wid_++) {
                int wid = wid_*(kPresumThreads/32) + thread_idx / kPresumThreads;
                auto smem_dst_ptr = shared_presum_tensors.smem_A0.data() +
                                    presum_write_stage * PresumSingleStageSize +
                                    wid * PresumSmemSize +
                                    (thread_idx%kPresumThreads) * sizeof(PresumStoreVecType)/sizeof(ElementA);

                data[wid_] = *(PresumStoreVecType*)smem_dst_ptr;
              }
            }

            // asm volatile("bar.cta.sync %0, %1;" : : "r"(3), "r"(128));

            if (last_presum_iters) {
              #pragma unroll (kPresumThreads/32)
              for (int wid_ = 0; wid_ < kPresumThreads/32; wid_++) {
                int wid = wid_*(kPresumThreads/32) + thread_idx / kPresumThreads;
                // if (wid == 2&& threadIdx.x%32 == 0 && m_coord < 32 && n_coord == 0)
                  // printf("1478 %d %d %d : %p %f ; %d %d\n", m_coord, presumIter, presum_tile, iter_PresumB_M.get(0), float(data[wid_][0]), iter_PresumB_M.row, iter_PresumB_M.col);
                arch::global_store<PresumStoreVecType, sizeof(PresumStoreVecType)>(data[wid_], ((ElementB*)iter_PresumB_M.get(0)) + wid*iter_PresumB_M.extent.row()*iter_PresumB_M.extent.column(),
                                                                                  iter_PresumB_M.validTB() && iter_PresumB_M.valid() && last_presum_iters);
              }
            }
          } else {
            if (last_presum_iters) {
              #pragma unroll (kPresumThreads/32)
              for (int wid_ = 0; wid_ < kPresumThreads/32; wid_++) {
                int wid = wid_;
                auto smem_dst_ptr = shared_presum_tensors.smem_A0.data() +
                                    presum_write_stage * PresumSingleStageSize +
                                    wid * PresumSmemSize +
                                    (thread_idx%kPresumThreads) * sizeof(PresumStoreVecType)/sizeof(ElementA);

                data[wid_] = *(PresumStoreVecType*)smem_dst_ptr;
              }
            }

            // asm volatile("bar.cta.sync %0, %1;" : : "r"(3), "r"(128));

            if (last_presum_iters) {
              #pragma unroll (kPresumThreads/32)
              for (int wid_ = 0; wid_ < kPresumThreads/32; wid_++) {
                int wid = wid_;
                // if (wid == 2&& threadIdx.x%32 == 0 && m_coord < 32 && n_coord == 0)
                  // printf("1478 %d %d %d : %p %f ; %d %d\n", m_coord, presumIter, presum_tile, iter_PresumB_M.get(0), float(data[wid_][0]), iter_PresumB_M.row, iter_PresumB_M.col);
                arch::global_store<PresumStoreVecType, sizeof(PresumStoreVecType)>(data[wid_], ((ElementB*)iter_PresumB_M.get(0)) + wid*iter_PresumB_M.extent.row()*iter_PresumB_M.extent.column(),
                                                                                  iter_PresumB_M.validTB() && iter_PresumB_M.valid() && last_presum_iters);
              }
            }
          }

          // if (false && thread_idx % kPresumThreads == 0) {
          // //Example of using TMA store
          //   ElementA* st_ptr = ((ElementA*)iter_PresumA_M.get(0)) + wid*iter_PresumA_M.extent.row()*iter_PresumA_M.extent.column();
          //   ElementA* st_ptr_2 = ((ElementA*)iter_PresumA_M.get(0)) + (wid+2)*iter_PresumA_M.extent.row()*iter_PresumA_M.extent.column();
            
          //   SM90_TMA_STORE().copy(mainloop_params.tma_store_presumld_a.get_tma_descriptor(), smem_dst_ptr,  
          //   iter_PresumA_M.tb_offset.column(),
          //   wid*iter_PresumA_M.extent.row() +
          //   iter_PresumA_M.tb_offset.row() + presumIter*size<0>(PresumTileShapeA{}), 0); //iter_PresumA_M.row, iter_PresumA_M.col);
            // asm volatile("cp.async.bulk.commit_group;");
          // }
          // return;
        }

        // iter_PresumA_M.inc();
      }
    }

  /// Perform a collective-scoped matrix multiply-accumulate
  /// Consumer Perspective
  template <
    class BlockCoord,
    class FrgTensorC,
    class ProblemShape_MNKL
  >
  CUTLASS_DEVICE void
  mma(BlockCoord const& blk_coord, int sub_m_idx, 
      ProblemShape_MNKL const& problem_shape,
      ProblemShape_MNKL const& half_problem_shape,
      MainloopPipeline pipeline,
      PipelineState smem_pipe_read,
      FrgTensorC& accum,
      int k_tile_count,
      int thread_idx,
      TensorStorage& shared_tensors,
      PresumTensorStorage& shared_presum_tensors,
      PresumTensorStorage2& shared_presum_tensors2,
      Params const& mainloop_params) {
    static_assert(is_rmem<FrgTensorC>::value, "C tensor must be rmem resident.");
    static_assert(cute::rank(SmemLayoutA{}) == 3, "Smem layout must be rank 3.");
    static_assert(cute::rank(SmemLayoutB{}) == 3, "Smem layout must be rank 3.");
    static_assert(cute::is_void_v<SmemCopyAtomA>,
      "SM90 GMMA mainloops cannot have a non-void copy atom for smem sourced instructions.");
    static_assert(cute::is_void_v<SmemCopyAtomB>,
      "SM90 GMMA mainloops cannot have a non-void copy atom for smem sourced instructions.");

    auto [m_coord, n_coord, k_coord, l_coord] = blk_coord;
    auto [M,N,K,L] = problem_shape;
    auto [halfM, halfN, halfK, halfL] = half_problem_shape;

    Tensor sA = make_tensor(make_smem_ptr(shared_tensors.smem_A.data()), SmemLayoutA{});          // (BLK_M,BLK_K,PIPE)
    Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.data()), SmemLayoutB{});          // (BLK_N,BLK_K,PIPE)

    const uint presumComputeIterationsA = (kPresumComputeIterationsA * (1 << mainloop_params.get_presum_tile_log_multiplier_a())) >> mainloop_params.get_presum_tile_log_divider_a();
    const uint presumComputeIterationsB = (kPresumComputeIterationsB * (1 << mainloop_params.get_presum_tile_log_multiplier_b())) >> mainloop_params.get_presum_tile_log_divider_b();

    Tensor sPresumA0 = make_tensor(make_smem_ptr(shared_presum_tensors.smem_A0.data()), PresumSmemLayoutA{});
    //
    // Define C accumulators and A/B partitioning
    //

    // Layout of warp group to thread mapping

    static_assert(stride<0>(typename TiledMma::ALayout{}) == 0 and
                  stride<0>(typename TiledMma::BLayout{}) == 0 and
                  size<0>(typename TiledMma::ALayout{}) == NumThreadsPerWarpGroup and
                  size<0>(typename TiledMma::BLayout{}) == NumThreadsPerWarpGroup,
                  "Stride of the first mode must be 0 and the size of the mode must be NumThreadsPerWarpGroup");

    constexpr int MmaWarpGroups = size(TiledMma{}) / NumThreadsPerWarpGroup;
    Layout warp_group_thread_layout = make_layout(Int<MmaWarpGroups>{},
                                                  Int<NumThreadsPerWarpGroup>{});

    int warp_group_idx = __shfl_sync(0xFFFFFFFF, thread_idx / NumThreadsPerWarpGroup, 0);

    TiledMma tiled_mma;
    auto thread_mma = tiled_mma.get_slice(warp_group_thread_layout(warp_group_idx));

    Tensor tCsA = thread_mma.partition_A(sA);                                                 // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCsB = thread_mma.partition_B(sB);                                                 // (MMA,MMA_N,MMA_K,PIPE)

    // Allocate "fragments/descriptors"
    Tensor tCrA = thread_mma.make_fragment_A(tCsA);                                           // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCrB = thread_mma.make_fragment_B(tCsB);                                           // (MMA,MMA_N,MMA_K,PIPE)

    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(accum));                                                         // M
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<2>(accum));                                                         // N
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCsB));                                                          // K
    CUTE_STATIC_ASSERT_V(size<3>(tCsA) == size<3>(tCsB));                                                       // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sA));                                         // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sB));                                         // PIPE

    //
    // PIPELINED MAIN LOOP
    //
    static_assert((0 <= K_PIPE_MMAS) && (K_PIPE_MMAS <  K_PIPE_MAX),
        "ERROR : Incorrect number of MMAs in flight");

    int block_idx = m_coord + n_coord * 1;///params.grid_tiled_shape.m();
    int n_coord_div = (n_coord * (1 << mainloop_params.get_presum_tile_log_multiplier_a())) % (1<<mainloop_params.get_presum_tile_log_divider_a());
    int new_n_coord = (n_coord * (1 << mainloop_params.get_presum_tile_log_multiplier_a())) >> mainloop_params.get_presum_tile_log_divider_a();

    auto presum_outputs = presumld_outputs(problem_shape, half_problem_shape, mainloop_params);
    PresumGlobalIteratorA iter_PresumA_M(
      mainloop_params.ptr_presum_A, halfK,
      {halfM, halfK},
      {m_coord * size<0>(TileShape{}) + ((n_coord_div * size<0>(TileShape{})) >> mainloop_params.get_presum_tile_log_divider_a()),
        new_n_coord * size<1>(TileShape{})},
      block_idx, {0, 0}, thread_idx%kPresumThreads, {1*halfM, 0}, {2*halfM, 0}, {3*halfM, 0}
    );
    PresumGlobalIteratorB iter_PresumB_M(
      mainloop_params.ptr_presum_B, halfN,
      {halfK, halfN},
      {(m_coord * (1 << mainloop_params.get_presum_tile_log_multiplier_b()) * (size<0>(TileShape{}))) >> mainloop_params.get_presum_tile_log_divider_b(),
       n_coord * size<1>(TileShape{})},
      block_idx, {0, 0}, thread_idx%kPresumThreads, {1*halfK, 0}, {2*halfK, 0}, {3*halfK, 0}
    );
    // if (threadIdx.x == 128 or threadIdx.x == 256)
    //   printf("963 %d ; %d %d ; %d %d\n", threadIdx.x, m_coord * size<0>(TileShape{}), n_coord * size<1>(TileShape{}), iter_PresumA_M.tb_offset.row(), iter_PresumA_M.tb_offset.column());
    // We release buffers to producer warps(dma load) with some mmas in flight
    PipelineState smem_pipe_release = smem_pipe_read;

    // Prologue GMMAs
    int prologue_mma_count = min(K_PIPE_MMAS, k_tile_count);
    // assert(k_tile_count >= 1);

    // if ((StrassenMiGroup::hasM0() || StrassenMiGroup::hasM1()) && StrassenMiGroup::AllPresums::computeAnyAPresum())
    //   if (k_tile_count >= kPresumComputeIterations) {} else __builtin_unreachable();
    const bool require_presum_stores_in_last_iters =  (sub_m_idx == 0) ? (k_tile_count < presumComputeIterationsA + PresumStages) :
                                                                         (k_tile_count < presumComputeIterationsB + PresumStages);

    int prev_read_stage = 0;
    int presum_iter = 0;
    warpgroup_fence_operand(accum);
    {
      // WAIT on smem_pipe_read until its data are available (phase bit flips from rdPhaseBit value)
      auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
      pipeline.consumer_wait(smem_pipe_read, barrier_token);

      int read_stage = smem_pipe_read.index();
      if (sub_m_idx == 0) {
        presum_compute_store(presumComputeIterationsA, thread_idx, read_stage, 0, presum_iter, m_coord, n_coord, sub_m_idx, iter_PresumA_M, shared_presum_tensors, shared_presum_tensors2, presum_outputs, mainloop_params);
      } else if (sub_m_idx == 1) {
        presum_compute_store_b(presumComputeIterationsB, thread_idx, read_stage, 0, presum_iter, m_coord, n_coord, sub_m_idx, iter_PresumB_M, shared_presum_tensors, shared_presum_tensors2, presum_outputs, mainloop_params);
      }
      prev_read_stage = read_stage;
      warpgroup_arrive();
      if ((is_fused || IsFusedM4M5) && sub_m_idx == 1) tiled_mma.accumulate_ = GMMA::ScaleOut::One;
      else tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;
      // Unroll the K mode manually to set scale D to 1
      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
        // (V,M,K) x (V,N,K) => (V,M,N)
        cute::gemm(tiled_mma, tCrA(_,_,k_block,read_stage), tCrB(_,_,k_block,read_stage), accum);
        tiled_mma.accumulate_ = GMMA::ScaleOut::One;
      }

      warpgroup_commit_batch();
      // presum_compute_store2(thread_idx, read_stage, presum_iter, m_coord, n_coord, iter_PresumA_M, shared_presum_tensors, shared_presum_tensors2,presum_outputs, mainloop_params);
      // if (presum_iter < kPresumComputeIterations) {
      //   asm volatile("cp.async.bulk.commit_group;");
      //   cute::tma_store_wait<0>();
      // }
      ++smem_pipe_read;
    }

    tiled_mma.accumulate_ = GMMA::ScaleOut::One;

    warpgroup_fence_operand(accum);
    //This code do not run as K_PIPE_MMAS = 1
    CUTLASS_PRAGMA_UNROLL
    for (int k_tile_prologue = prologue_mma_count - 1; k_tile_prologue > 0; --k_tile_prologue)
    {
      // WAIT on smem_pipe_read until its data are available (phase bit flips from rdPhaseBit value)
      auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
      pipeline.consumer_wait(smem_pipe_read, barrier_token);
      int read_stage = smem_pipe_read.index();
      // presum_compute_store(thread_idx, read_stage, presum_iter, m_coord, n_coord, iter_PresumA_M, shared_presum_tensors, shared_presum_tensors2, presum_outputs, mainloop_params);
      // presum_compute_store2(thread_idx, read_stage, presum_iter, m_coord, n_coord, iter_PresumA_M, shared_presum_tensors, shared_presum_tensors2,presum_outputs, mainloop_params);
      // warpgroup_arrive();
      // (V,M,K) x (V,N,K) => (V,M,N)
      cute::gemm(tiled_mma, tCrA(_,_,_,read_stage), tCrB(_,_,_,read_stage), accum);
      warpgroup_commit_batch();
      // if (presum_iter < kPresumComputeIterations) {
      //   asm volatile("cp.async.bulk.commit_group;");
      //   cute::tma_store_wait<0>();
      // }
      // presum_iter++;
      ++smem_pipe_read;
    }

    warpgroup_fence_operand(accum);
    // Mainloop GMMAs
    k_tile_count -= prologue_mma_count;

    // int presum_write_stage = 0;

    // presum_write_stage = presum_write_stage ^ 1;

    CUTLASS_PRAGMA_NO_UNROLL
    for ( ; k_tile_count > 0; --k_tile_count)
    {
      PresumStoreVecType e2;

      // presum_compute_store3(thread_idx, 0, presum_iter, m_coord, n_coord, iter_PresumA_M, shared_presum_tensors, shared_presum_tensors2,presum_outputs, mainloop_params, e2);
      // WAIT on smem_pipe_read until its data are available (phase bit flips from rdPhaseBit value)
      auto barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
      pipeline.consumer_wait(smem_pipe_read, barrier_token);
      int read_stage = smem_pipe_read.index();
      //TODO: This needs fix when k_tile_count < kPresum
      bool last_presum_iters = require_presum_stores_in_last_iters && k_tile_count < PresumStages;

      //
      // Compute on k_tile
      //
      if ((StrassenMiGroup::hasM0() && !is_fused) || (is_fused && sub_m_idx == 0)) {
        last_presum_iters = last_presum_iters && presum_iter < presumComputeIterationsA + PresumStages;
        presum_compute_store2(presumComputeIterationsA, thread_idx, prev_read_stage, last_presum_iters, presum_iter,
                              m_coord, n_coord * (1 << mainloop_params.get_presum_tile_log_multiplier_a()),
                              iter_PresumA_M, shared_presum_tensors, shared_presum_tensors2,presum_outputs, mainloop_params, e2);
        presum_compute_store(presumComputeIterationsA, thread_idx, read_stage, 0, presum_iter,
                             m_coord, n_coord * (1 << mainloop_params.get_presum_tile_log_multiplier_a()),
                             sub_m_idx, iter_PresumA_M, shared_presum_tensors, shared_presum_tensors2, presum_outputs, mainloop_params);
      } else if ((StrassenMiGroup::hasM1() && !is_fused) || (is_fused && sub_m_idx == 1)) {
        last_presum_iters = last_presum_iters && presum_iter < presumComputeIterationsB + PresumStages;
        presum_compute_store2_b(presumComputeIterationsB, thread_idx, prev_read_stage, last_presum_iters, presum_iter,
                                m_coord * (1 << mainloop_params.get_presum_tile_log_multiplier_b()), n_coord, iter_PresumB_M, shared_presum_tensors, shared_presum_tensors2,presum_outputs, mainloop_params, e2);
        presum_compute_store_b(presumComputeIterationsB, thread_idx, read_stage, 0, presum_iter,
                               m_coord * (1 << mainloop_params.get_presum_tile_log_multiplier_b()),
                               n_coord, sub_m_idx, iter_PresumB_M, shared_presum_tensors, shared_presum_tensors2, presum_outputs, mainloop_params);
      }
      prev_read_stage = read_stage;
      // presum_write_stage = presum_write_stage ^ 1;
      warpgroup_fence_operand(accum);
      warpgroup_arrive();
      // (V,M,K) x (V,N,K) => (V,M,N)
      cute::gemm(tiled_mma, tCrA(_,_,_,read_stage), tCrB(_,_,_,read_stage), accum);
      warpgroup_commit_batch();

      /// Wait on the GMMA barrier for K_PIPE_MMAS (or fewer) outstanding to ensure smem_pipe_write is consumed
      warpgroup_wait<K_PIPE_MMAS>();
      warpgroup_fence_operand(accum);

      // UNLOCK smem_pipe_release, done _computing_ on it
      pipeline.consumer_release(smem_pipe_release);
      presum_iter++;
      // presum_write_stage = presum_write_stage ^ 1;
      // if (presum_iter < kPresumComputeIterations) {
      //   asm volatile("cp.async.bulk.commit_group;");
      //   cute::tma_store_wait<0>();
      // }
      // Advance smem_pipe_read and smem_pipe_release
      ++smem_pipe_read;
      ++smem_pipe_release;
    }

    PresumStoreVecType e2;
    if ((StrassenMiGroup::hasM0() && !is_fused) || (is_fused && sub_m_idx == 0)) {
      bool last_presum_iters = require_presum_stores_in_last_iters && presum_iter < presumComputeIterationsA + PresumStages;
      presum_compute_store2(presumComputeIterationsA, thread_idx, prev_read_stage, last_presum_iters, presum_iter,
                            m_coord, n_coord * (1 << mainloop_params.get_presum_tile_log_multiplier_a()),
                            iter_PresumA_M, shared_presum_tensors, shared_presum_tensors2,presum_outputs, mainloop_params, e2);
    } else if ((StrassenMiGroup::hasM1() && !is_fused) || (is_fused && sub_m_idx == 1)) {
      bool last_presum_iters = require_presum_stores_in_last_iters && presum_iter < presumComputeIterationsB + PresumStages;
      presum_compute_store2_b(presumComputeIterationsB, thread_idx, prev_read_stage, last_presum_iters, presum_iter,
                              m_coord * (1 << mainloop_params.get_presum_tile_log_multiplier_b()), n_coord, iter_PresumB_M, shared_presum_tensors, shared_presum_tensors2,presum_outputs, mainloop_params, e2);
    }

    warpgroup_fence_operand(accum);
  }

  /// Perform a Consumer Epilogue to release all buffers
  CUTLASS_DEVICE void
  mma_tail(MainloopPipeline pipeline, PipelineState smem_pipe_release, int k_tile_count) {
    // Prologue GMMAs
    int prologue_mma_count = min(K_PIPE_MMAS, k_tile_count);
    k_tile_count -= prologue_mma_count;

    smem_pipe_release.advance(k_tile_count);

    // Wait on all GMMAs to complete
    warpgroup_wait<0>();

    for (int count = 0; count < prologue_mma_count; ++count) {
      pipeline.consumer_release(smem_pipe_release);                 // UNLOCK smem_pipe_release, done _computing_ on it
      ++smem_pipe_release;
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
