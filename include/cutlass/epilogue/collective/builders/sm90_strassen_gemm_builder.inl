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

#include "cute/atom/mma_traits_sm90.hpp"
#include "cute/atom/mma_traits_sm90_gmma.hpp"
#include "cute/atom/copy_traits_sm90.hpp"

#include "cutlass/detail/dependent_false.hpp"
#include "cutlass/detail/layout.hpp"
#include "cutlass/gemm/collective/builders/sm90_common.inl"
#include "cutlass/epilogue/dispatch_policy.hpp"
#include "cutlass/epilogue/collective/collective_strassen_epilogue.hpp"
#include "cutlass/epilogue/collective/builders/sm90_common.inl"
#include "cutlass/epilogue/collective/builders/sm90_builder.inl"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/epilogue/thread/linear_combination_generic.h"
#include "cutlass/epilogue/thread/linear_combination_bias_elementwise.h"
#include "cutlass/epilogue/fusion/callbacks.hpp"
#include "cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp"
#include "cutlass/cutlass.h"
#if defined(__CUDACC_RTC__)
#include CUDA_STD_HEADER(type_traits)
#else
#include <type_traits>
#endif

///////////////////////////////////////////////////////////////////////////////

namespace cutlass::epilogue::collective {

///////////////////////////////////////////////////////////////////////////////
namespace detail {
// Helper for building TMA warp-specialized collective epilogues, specialized by
// the fusion operation performed and the dispatch policy to use.
template <
  class StrassenMiGroup_,
  class TileShape_MNK,
  class EpilogueTile_MN,
  class ElementAccumulator,
  class ElementCompute,
  class ElementC_,
  class GmemLayoutTagC_,
  int AlignmentC,
  class ElementD_,
  class GmemLayoutTagD,
  int AlignmentD,
  class FusionOpOrCallbacks,
  class DispatchPolicy
>
struct Sm90TmaStrassenBuilderImpl {
  // C/D should meet TMA alignment requirement if not void
  static_assert(detail::is_aligned<ElementC_, AlignmentC, ElementD_, AlignmentD>(),
                "C/D Should meet TMA alignment requirement\n");
  // Passing void D disables destination store + smem allocation
  using ElementD = cute::conditional_t<cute::is_void_v<ElementD_>,
                     fusion::get_element_aux_t<FusionOpOrCallbacks>, ElementD_>;

  // Passing void C disables source load + smem allocation
  using ElementC = cute::conditional_t<cute::is_void_v<ElementC_>,ElementD,ElementC_>; // prevents void ref breakages
  using GmemLayoutTagC = cute::conditional_t<cute::is_void_v<ElementC_>,GmemLayoutTagD,GmemLayoutTagC_>;

  using GmemStrideTypeC = cutlass::detail::TagToStrideC_t<GmemLayoutTagC>;
  using GmemStrideTypeD = cutlass::detail::TagToStrideC_t<GmemLayoutTagD>;
  
  using UnderlyingGmemStrideTypeC = cute::remove_pointer_t<GmemStrideTypeC>;
  using UnderlyingGmemStrideTypeD = cute::remove_pointer_t<GmemStrideTypeD>;

  using CopyOpS2G = cute::conditional_t<detail::is_im2col_mode<GmemLayoutTagD>,
      SM90_TMA_STORE_IM2COL,
      SM90_TMA_STORE
    >;
  using CopyOpG2S = cute::conditional_t<detail::is_im2col_mode<GmemLayoutTagC>,
      SM90_TMA_LOAD_IM2COL,
      SM90_TMA_LOAD
    >;

  // Get the smallest tiled copy we can use to retile the accumulators
  // using CopyAtomC = Copy_Atom<SM90_U32x4_STSM_N, cutlass::half_t>;
  using CopyAtomC = cute::conditional_t<
    size<1>(EpilogueTile_MN{}) % 16 == 0,
    Copy_Atom<SM90_U32x4_STSM_N, cutlass::half_t>,
    cute::conditional_t<
      size<1>(EpilogueTile_MN{}) % 8 == 0,
      Copy_Atom<SM90_U32x2_STSM_N, cutlass::half_t>,
      void
    >
  >;
  static_assert(!cute::is_same_v<CopyAtomC, void>, "CopyAtomC can't be void, divisiblity check for EpilogueTile_MN failed");
  // Get register to register tiled copy that happen before shared memory store.
  // Apply void as no register transform op needed currently.
  using CopyOpR2R = void;

  // TMA builder allows for passing callbacks directly, which is either a fusion::FusionCallbacks
  // instance or a direct visitor implementation, e.g. fusion::Sm90LinearCombination
  using FusionCallbacks = 
    typename CallbacksBuilder<
      DispatchPolicy,
      FusionOpOrCallbacks,
      TileShape_MNK,
      EpilogueTile_MN,
      ElementAccumulator
    >::Callbacks;

  using CollectiveOp = cutlass::epilogue::collective::CollectiveStrassenEpilogue<
      StrassenMiGroup_,
      DispatchPolicy,
      TileShape_MNK,
      EpilogueTile_MN,
      ElementC_, // Need to pass void through to expose via GemmUniversal
      GmemStrideTypeC,
      ElementD_,
      GmemStrideTypeD,
      FusionCallbacks,
      CopyOpG2S,
      decltype(detail::sm90_get_epilogue_smem_swizzle_layout_atom<UnderlyingGmemStrideTypeC, ElementC, EpilogueTile_MN>()),
      decltype(detail::sm90_get_smem_load_op_for_source<UnderlyingGmemStrideTypeC, ElementC, EpilogueTile_MN>()),
      CopyOpS2G,
      decltype(detail::sm90_get_epilogue_smem_swizzle_layout_atom<UnderlyingGmemStrideTypeD, ElementD, EpilogueTile_MN>()),
      decltype(detail::sm90_get_smem_store_op_for_accumulator<UnderlyingGmemStrideTypeD, ElementD, EpilogueTile_MN>()),
      CopyAtomC,
      CopyOpR2R
    >;
};
}

// No-smem builder
template <
  typename StrassenMiGroup,
  class OpClass,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class EpilogueTileType,
  class ElementAccumulator,
  class ElementCompute,
  class ElementC_,
  class GmemLayoutTagC_,
  int AlignmentC,
  class ElementD,
  class GmemLayoutTagD,
  int AlignmentD,
  class Schedule,
  FloatRoundStyle RoundStyle
>
struct CollectiveStrassenBuilder<
    StrassenMiGroup,
    arch::Sm90,
    OpClass,
    TileShape_MNK,
    ClusterShape_MNK,
    EpilogueTileType,
    ElementAccumulator,
    ElementCompute,
    ElementC_,
    GmemLayoutTagC_,
    AlignmentC,
    ElementD,
    GmemLayoutTagD,
    AlignmentD,
    Schedule,
    fusion::LinearCombination<ElementD,ElementCompute,ElementC_,ElementCompute,RoundStyle>,
    cute::enable_if_t<cute::is_same_v<Schedule, NoSmemWarpSpecialized> ||
                      cute::is_same_v<Schedule, PtrArrayNoSmemWarpSpecialized> ||
                      cute::is_same_v<Schedule, PtrArrayNoSmemWarpSpecializedTransposed> >> {

  // Passing void C disables source load
  using ElementC = cute::conditional_t<cute::is_void_v<ElementC_>,
      ElementD, ElementC_>; // prevents cute breakages
  using GmemLayoutTagC = cute::conditional_t<cute::is_void_v<ElementC_>,
      GmemLayoutTagD, GmemLayoutTagC_>;
  static constexpr thread::ScaleType::Kind ScaleType = cute::is_void_v<ElementC_> ?
      thread::ScaleType::OnlyAlphaScaling : thread::ScaleType::Default;

  static constexpr int FragmentSize = 1;
  using ThreadOp = thread::LinearCombination<
    ElementD, FragmentSize, ElementAccumulator, ElementCompute,
    ScaleType, RoundStyle, ElementC>;

  using CollectiveOp = cute::conditional_t<
    cute::is_same_v<Schedule, NoSmemWarpSpecialized>,
    //TODO: Change
    cutlass::epilogue::collective::detail::Sm90TmaWarpSpecializedAdapter<
      cutlass::epilogue::collective::DefaultEpilogue<
        ElementC_,
        cutlass::detail::TagToStrideC_t<GmemLayoutTagC>,
        cutlass::detail::TagToStrideC_t<GmemLayoutTagD>,
        ThreadOp,
        cutlass::gemm::EpilogueDefault>>,
    // Epilogue for Ptr-Array and Grouped Gemm
    cutlass::epilogue::collective::detail::Sm90TmaWarpSpecializedAdapter<
      cutlass::epilogue::collective::DefaultEpilogueArray<
        ElementC_,
        cutlass::detail::TagToStrideC_t<GmemLayoutTagC>,
        cutlass::detail::TagToStrideC_t<GmemLayoutTagD>,
        ThreadOp,
        Schedule>>
    >;
};

// Tma warp-specialized builder
template <
  typename StrassenMiGroup,
  class OpClass,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class EpilogueTileType,
  class ElementAccumulator,
  class ElementCompute,
  class ElementC,
  class GmemLayoutTagC,
  int AlignmentC,
  class ElementD_,
  class GmemLayoutTagD,
  int AlignmentD,
  class Schedule,
  class FusionOperation
>
struct CollectiveStrassenBuilder<
    StrassenMiGroup,
    arch::Sm90,
    OpClass,
    TileShape_MNK,
    ClusterShape_MNK,
    EpilogueTileType,
    ElementAccumulator,
    ElementCompute,
    ElementC,
    GmemLayoutTagC,
    AlignmentC,
    ElementD_,
    GmemLayoutTagD,
    AlignmentD,
    Schedule,
    FusionOperation,
    cute::enable_if_t<cute::is_same_v<Schedule, TmaWarpSpecialized> ||
                      cute::is_same_v<Schedule, TmaWarpSpecializedCooperative> ||
                      detail::sm90_is_ptr_array_tma_v<Schedule>>> {
private:
  using ElementD = cute::conditional_t<cute::is_void_v<ElementD_>,
                     fusion::get_element_aux_t<FusionOperation>, ElementD_>;
  static const bool is_fused_m2_m3 = StrassenMiGroup::hasM2() && StrassenMiGroup::hasM3();
  static const bool is_fused_m4_m5 = StrassenMiGroup::hasM4() && StrassenMiGroup::hasM5();
  static const uint StagesC = StrassenMiGroup::hasM0() ? 2 : ((is_fused_m2_m3) ? 4 : (is_fused_m4_m5 ? 4 : 4));
  static const uint StagesD = (is_fused_m2_m3) ? 4 : (is_fused_m4_m5 ? 4 : 2);
  using EpilogueTile_MN = cute::conditional_t<StrassenMiGroup::hasM0() or StrassenMiGroup::hasM1() or StrassenMiGroup::hasM2() or StrassenMiGroup::hasM3() or is_fused_m4_m5,
                                              cute::tuple<_64, _32>,
                          decltype(detail::sm90_compute_tile_shape_or_override<ElementD, EpilogueTileType, Schedule, TileShape_MNK>())>;
  using DispatchPolicy = cute::conditional_t<StrassenMiGroup::hasM0() or StrassenMiGroup::hasM1() or StrassenMiGroup::hasM2() or StrassenMiGroup::hasM3() or is_fused_m4_m5,
                                             cutlass::epilogue::Sm90TmaWarpSpecialized<StagesC, StagesD, size<1>(EpilogueTile_MN{})/2, true, false>,
                                             decltype(detail::sm90_get_tma_dispatch_policy<TileShape_MNK,EpilogueTile_MN,ElementC,ElementD,Schedule>())>;
  // typename DispatchPolicy::x y;
public:
  using CollectiveOp =
    typename detail::Sm90TmaStrassenBuilderImpl<
      StrassenMiGroup,
      TileShape_MNK,
      EpilogueTile_MN,
      ElementAccumulator,
      ElementCompute,
      ElementC,
      GmemLayoutTagC,
      AlignmentC,
      ElementD_,
      GmemLayoutTagD,
      AlignmentD,
      FusionOperation,
      DispatchPolicy
    >::CollectiveOp;
};

// Auto builder
template <
  typename StrassenMiGroup,
  class OpClass,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class EpilogueTileType,
  class ElementAccumulator,
  class ElementCompute,
  class ElementC,
  class GmemLayoutTagC,
  int AlignmentC,
  class ElementD,
  class GmemLayoutTagD,
  int AlignmentD,
  class FusionOperation
>
struct CollectiveStrassenBuilder<
    StrassenMiGroup,
    arch::Sm90,
    OpClass,
    TileShape_MNK,
    ClusterShape_MNK,
    EpilogueTileType,
    ElementAccumulator,
    ElementCompute,
    ElementC,
    GmemLayoutTagC,
    AlignmentC,
    ElementD,
    GmemLayoutTagD,
    AlignmentD,
    EpilogueScheduleAuto,
    FusionOperation,
    void> {
private:
  static_assert(cute::is_same_v<FusionOperation, fusion::LinearCombination<ElementD,ElementCompute,ElementC,ElementCompute>>,
                "Auto schedule doesn't support fusion. Use one of the TmaWarpSpecialized schedules instead.");

  // Pick No-Smem epilogue as the Auto Epilogue Schedule (Auto schedules do not guarantee best performance) 
  // since TMA epilogues are not compatible with non-TMA non-WS mainloops
  using EpilogueSchedule = NoSmemWarpSpecialized;
  using _CollectiveStrassenBuilder = CollectiveStrassenBuilder<
    StrassenMiGroup,
    arch::Sm90,
    OpClass,
    TileShape_MNK,
    ClusterShape_MNK,
    EpilogueTileType,
    ElementAccumulator,
    ElementCompute,
    ElementC,
    GmemLayoutTagC,
    AlignmentC,
    ElementD,
    GmemLayoutTagD,
    AlignmentD,
    EpilogueSchedule,
    FusionOperation
  >;

public:
  using CollectiveOp = typename _CollectiveStrassenBuilder::CollectiveOp;
};

// DEPRECATED Tma warp-specialized builder for elementwise fusion
template <
  typename StrassenMiGroup,
  class OpClass,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class EpilogueTileType,
  class ElementAccumulator,
  class ElementCompute,
  class ElementC,
  class GmemLayoutTagC,
  int AlignmentC,
  class ElementD,
  class GmemLayoutTagD,
  int AlignmentD,
  class Schedule,
  class UnusedFusionOp
>
struct [[deprecated("Use TmaWarpSpecialized with fusion::LinCombEltAct instead")]]
CollectiveStrassenBuilder<
    StrassenMiGroup,
    arch::Sm90,
    OpClass,
    TileShape_MNK,
    ClusterShape_MNK,
    EpilogueTileType,
    ElementAccumulator,
    ElementCompute,
    ElementC,
    GmemLayoutTagC,
    AlignmentC,
    ElementD,
    GmemLayoutTagD,
    AlignmentD,
    Schedule,
    UnusedFusionOp,
    cute::enable_if_t<cute::is_base_of_v<TmaWarpSpecializedElementwiseBase, Schedule> ||
                      cute::is_base_of_v<TmaWarpSpecializedCooperativeElementwiseBase, Schedule> >> {
private:
  using FusionOp =
    fusion::LinCombEltAct<Schedule::template ActivationFunctor, ElementD, ElementCompute, ElementC, ElementCompute, Schedule::Round>;
  using ImplSchedule =
    cute::conditional_t<cute::is_base_of_v<TmaWarpSpecializedElementwiseBase, Schedule>,
      TmaWarpSpecialized, TmaWarpSpecializedCooperative>;

public:
  using CollectiveOp =
    typename CollectiveStrassenBuilder<
      StrassenMiGroup,
      arch::Sm90,
      OpClass,
      TileShape_MNK,
      ClusterShape_MNK,
      EpilogueTileType,
      ElementAccumulator,
      ElementCompute,
      ElementC,
      GmemLayoutTagC,
      AlignmentC,
      ElementD,
      GmemLayoutTagD,
      AlignmentD,
      ImplSchedule,
      FusionOp
    >::CollectiveOp;
};

// DEPRECATED Tma warp-specialized builder for bias + elementwise fusion
template <
  typename StrassenMiGroup,
  class OpClass,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class EpilogueTileType,
  class ElementAccumulator,
  class ElementCompute,
  class ElementC_,
  class GmemLayoutTagC_,
  int AlignmentC,
  class ElementD,
  class GmemLayoutTagD,
  int AlignmentD,
  class Schedule,
  class UnusedFusionOp
>
struct [[deprecated("Use TmaWarpSpecialized with fusion::LinCombPerRowBiasEltAct or fusion::LinCombPerRowBiasEltActAux instead")]]
CollectiveStrassenBuilder<
    StrassenMiGroup,
    arch::Sm90,
    OpClass,
    TileShape_MNK,
    ClusterShape_MNK,
    EpilogueTileType,
    ElementAccumulator,
    ElementCompute,
    ElementC_,
    GmemLayoutTagC_,
    AlignmentC,
    ElementD,
    GmemLayoutTagD,
    AlignmentD,
    Schedule,
    UnusedFusionOp,
    cute::enable_if_t<cute::is_base_of_v<TmaWarpSpecializedBiasElementwiseBase, Schedule> ||
                      cute::is_base_of_v<TmaWarpSpecializedCooperativeBiasElementwiseBase, Schedule> >> {
private:
  using EpilogueTile_MN = decltype(detail::sm90_compute_tile_shape_or_override<
    ElementD, EpilogueTileType, Schedule, TileShape_MNK>());
  // MSVC doesn't seem to be able to deduce DispatchPolicy correctly if it's
  // defined as decltype of a detail::sm90_get_tma_dispatch_policy call.
  // Instead, we paste in the contents of that function.  A natural refactoring
  // would be to create a type alias in the detail namespace.
  using DispatchPolicy = Sm90TmaWarpSpecialized<
    /* StagesC = */ size(shape_div(take<0, 2>(TileShape_MNK{}), EpilogueTile_MN{})),
    /* StagesD = */ 2,
    /* FragmentSize = */ size(EpilogueTile_MN{}) / (detail::sm90_is_cooperative_v<Schedule> ? 256 : 128),
    /* ReuseSmemC = */ sizeof_bits_v<ElementC_> == sizeof_bits_v<ElementD>,
    false
  >;

  using GmemStrideTypeAux = gemm::TagToStrideC_t<GmemLayoutTagD>;
  using SmemLayoutAtomAux = decltype(detail::sm90_get_epilogue_smem_swizzle_layout_atom<
    GmemStrideTypeAux, typename Schedule::ElementT, EpilogueTile_MN>());
  using SmemCopyOpAux = decltype(detail::sm90_get_smem_store_op_for_accumulator<
    GmemStrideTypeAux, typename Schedule::ElementT, EpilogueTile_MN>());
  using FusionOperationAux = fusion::LinCombPerRowBiasEltActAux<
    GmemLayoutTagD, Schedule::template ActivationFunctor, ElementD, ElementCompute,
    typename Schedule::ElementT, typename Schedule::ElementBias, ElementC_, ElementCompute
  >;
  using FusionCallbacksAux = fusion::FusionCallbacks<
    DispatchPolicy, FusionOperationAux, TileShape_MNK, EpilogueTile_MN, SmemLayoutAtomAux, SmemCopyOpAux
  >;

  using FusionOperationNoAux = fusion::LinCombPerRowBiasEltAct<
    Schedule::template ActivationFunctor, ElementD, ElementCompute,
    typename Schedule::ElementBias, ElementC_, ElementCompute
  >;
  using FusionCallbacksNoAux = fusion::FusionCallbacks<
    DispatchPolicy, FusionOperationNoAux, TileShape_MNK, EpilogueTile_MN
  >;

  using ElementC = cute::conditional_t<cute::is_void_v<ElementC_>,ElementD,ElementC_>; // prevents void ref breakages
  using GmemLayoutTagC = cute::conditional_t<cute::is_void_v<ElementC_>,GmemLayoutTagD,GmemLayoutTagC_>;

  using GmemStrideTypeC = gemm::TagToStrideC_t<GmemLayoutTagC>;
  using GmemStrideTypeD = gemm::TagToStrideC_t<GmemLayoutTagD>;

  // Get the smallest tiled copy we can use to retile the accumulators
  using CopyAtomC = cute::conditional_t<
    size<1>(EpilogueTile_MN{}) % 16 == 0,
    Copy_Atom<SM90_U32x4_STSM_N, cutlass::half_t>,
    cute::conditional_t<
      size<1>(EpilogueTile_MN{}) % 8 == 0,
      Copy_Atom<SM90_U32x2_STSM_N, cutlass::half_t>,
      void
    >
  >;
  static_assert(!cute::is_same_v<CopyAtomC, void>, "CopyAtomC can't be void, divisiblity check for EpilogueTile_MN failed");

  // Get register to register tiled copy that happen before shared memory store.
  // Apply void as no register transform op needed.
  using CopyOpR2R = void;

public:
  using CollectiveOp = cutlass::epilogue::collective::Sm90EpilogueTmaWarpSpecializedBiasElementwise<
      DispatchPolicy::StagesC,
      DispatchPolicy::StagesD,
      DispatchPolicy::FragmentSize,
      TileShape_MNK,
      EpilogueTile_MN,
      ElementC_, // Need to pass void through to expose via GemmUniversal
      GmemStrideTypeC,
      ElementD,
      GmemStrideTypeD,
      cute::conditional_t<Schedule::StoreT, FusionCallbacksAux, FusionCallbacksNoAux>,
      SM90_TMA_LOAD,
      decltype(detail::sm90_get_epilogue_smem_swizzle_layout_atom<GmemStrideTypeC, ElementC, EpilogueTile_MN>()),
      decltype(detail::sm90_get_smem_load_op_for_source<GmemStrideTypeC, ElementC, EpilogueTile_MN>()),
      SM90_TMA_STORE,
      decltype(detail::sm90_get_epilogue_smem_swizzle_layout_atom<GmemStrideTypeD, ElementD, EpilogueTile_MN>()),
      decltype(detail::sm90_get_smem_store_op_for_accumulator<GmemStrideTypeD, ElementD, EpilogueTile_MN>()),
      CopyAtomC,
      CopyOpR2R
    >;
};

// CollectiveStrassenBuilder that transposed epilogue below is used for sm90 gmma RS TT kernels
// since swapping NNN kernels input matrix and transposing its output at the same time then
// we can get TTN kernel.
template <
  typename StrassenMiGroup,
  class OpClass,
  class TileShape_MNK,
  class ClusterShape_MNK,
  class EpilogueTileType,
  class ElementAccumulator,
  class ElementCompute,
  class ElementC_,
  class GmemLayoutTagC_,
  int AlignmentC,
  class ElementD,
  class GmemLayoutTagD,
  int AlignmentD,
  class FusionOperation
>
struct CollectiveStrassenBuilder<
    StrassenMiGroup,
    arch::Sm90,
    OpClass,
    TileShape_MNK,
    ClusterShape_MNK,
    EpilogueTileType,
    ElementAccumulator,
    ElementCompute,
    ElementC_,
    GmemLayoutTagC_,
    AlignmentC,
    ElementD,
    GmemLayoutTagD,
    AlignmentD,
    cutlass::gemm::EpilogueTransposed,
    FusionOperation,
    void> {
private:
  static_assert(cute::is_same_v<FusionOperation, fusion::LinearCombination<ElementD,ElementCompute,ElementC_,ElementCompute>>,
                "EpilogueTransposed schedule doesn't support fusion.");
  // Passing void C disables source load
  using ElementC = cute::conditional_t<cute::is_void_v<ElementC_>,
      ElementD, ElementC_>; // prevents cute breakages
  using GmemLayoutTagC = cute::conditional_t<cute::is_void_v<ElementC_>,
      GmemLayoutTagD, GmemLayoutTagC_>;
  static constexpr thread::ScaleType::Kind ScaleType = cute::is_void_v<ElementC_> ?
      thread::ScaleType::OnlyAlphaScaling : thread::ScaleType::Default;

  static constexpr int FragmentSize = 1;
  using ThreadOp = thread::LinearCombination<
    ElementD, FragmentSize, ElementAccumulator, ElementCompute,
    ScaleType, cutlass::FloatRoundStyle::round_to_nearest, ElementC>;

public:
  using CollectiveOp = cutlass::epilogue::collective::detail::Sm90TmaWarpSpecializedAdapter<
    cutlass::epilogue::collective::DefaultEpilogue<
      ElementC_,
      cutlass::detail::TagToStrideC_t<GmemLayoutTagC>,
      cutlass::detail::TagToStrideC_t<GmemLayoutTagD>,
      ThreadOp,
      cutlass::gemm::EpilogueTransposed>
    >;
};

///////////////////////////////////////////////////////////////////////////////

} // namespace cutlass::epilogue::collective
