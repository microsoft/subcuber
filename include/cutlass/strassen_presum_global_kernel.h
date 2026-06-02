#pragma once

#include "cutlass/arch/memory.h"
#include "cutlass/gemm/threadblock/presum_detail.h"

namespace cutlass {
namespace gemm {
namespace device {

template <typename GemmKernel0>
CUTLASS_GLOBAL
__launch_bounds__(GemmKernel0::kThreadCount, 1)
void print_kernel(typename GemmKernel0::Params params) {
  using AllPresums = typename GemmKernel0::StrassenMiGroup::AllPresums;

}

template <typename PresumGroup, typename GemmKernel0, uint kThreadCount>
CUTLASS_GLOBAL
__launch_bounds__(kThreadCount, 1)
void KernelPresumGlobalCompute(typename GemmKernel0::Params params) {
  using ElementA = typename GemmKernel0::ElementA;
  using ElementB = typename GemmKernel0::ElementB;

  using AllPresums = typename PresumGroup::AllPresums;
  using PresumVecType = typename GemmKernel0::Mma::PresumVecTypeA;
  using PresumComputeType = typename GemmKernel0::Mma::PresumComputeType;
  using PresumComputeToIOType = NumericArrayConverter<ElementA, PresumComputeType, PresumVecType::kElements>;
  using PresumIOToComputeType = NumericArrayConverter<PresumComputeType, ElementA, PresumVecType::kElements>;

  using PresumShapeA = typename GemmKernel0::Mma::PresumShapeA;
  using PresumShapeB = typename GemmKernel0::Mma::PresumShapeB;

  using PresumGlobalIteratorA = threadblock::PresumDetail::GlobalIterator<ElementA, kThreadCount, PresumShapeA, PresumVecType, false, false>;
  using PresumGlobalIteratorB = threadblock::PresumDetail::GlobalIterator<ElementB, kThreadCount, PresumShapeB, PresumVecType, true, false>;

  cutlass::gemm::GemmCoord threadblock_tile_offset = {(int)blockIdx.y, (int)blockIdx.x, 0};

  const int halfM = params.get_problem_shape_m()/2;
  const int halfN = params.get_problem_shape_n()/2;
  const int halfK = params.get_problem_shape_k()/2;
  
  int block_idx = threadblock_tile_offset.m() + threadblock_tile_offset.n() * gridDim.x;
  uint thread_idx = threadIdx.x;
  const int presum_multiplier_a = 1 << params.get_presum_log_tile_multiplier_a();
  const int presum_multiplier_b = 1 << params.get_presum_log_tile_multiplier_b();

  PresumGlobalIteratorA iter_PresumA(
    params.get_ptr_A(), params.get_stride_A(),
    {params.get_problem_shape_m(), params.get_problem_shape_k()},
    {threadblock_tile_offset.m() * PresumShapeA::kM, threadblock_tile_offset.n() * PresumShapeA::kN * presum_multiplier_a},
    block_idx, {0, 0}, thread_idx, {0, halfK}, {halfM, 0}, {halfM, halfK}
  );

  PresumGlobalIteratorB iter_PresumB(
    params.get_ptr_B(), params.get_stride_B(),
    {params.get_problem_shape_k(), params.get_problem_shape_n()},
    {threadblock_tile_offset.m() * PresumShapeB::kM * presum_multiplier_b, threadblock_tile_offset.n() * PresumShapeB::kN},
    block_idx, {0, 0}, thread_idx, {0, halfN}, {halfK, 0}, {halfK, halfN}
  );

  PresumGlobalIteratorA iter_PresumA_M(
    params.presum_m_a_workspace, params.get_stride_MA(),
    {halfM, halfK},
    {threadblock_tile_offset.m() * PresumShapeA::kM, threadblock_tile_offset.n() * PresumShapeA::kN * presum_multiplier_a},
    block_idx, {0, 0}, thread_idx, {1*halfM, 0}, {2*halfM, 0}, {3*halfM, 0}
  );

  PresumGlobalIteratorB iter_PresumB_M(
    params.presum_m_b_workspace, params.get_stride_MB(),
    {halfK, halfN},
    {threadblock_tile_offset.m() * PresumShapeB::kM * presum_multiplier_b, threadblock_tile_offset.n() * PresumShapeB::kN},
    block_idx, {0, 0}, thread_idx, {1*halfK, 0}, {2*halfK, 0}, {3*halfK, 0}
  );

  auto presumAComputeLoads = AllPresums::APresumComputeLoads(PresumGlobalKernel);
  auto presumBComputeLoads = AllPresums::BPresumComputeLoads(PresumGlobalKernel);
  PresumVecType a0; a0.clear();
  PresumVecType a1; a1.clear();
  PresumVecType a2; a2.clear();
  PresumVecType a3; a3.clear();

  PresumVecType b0; b0.clear();
  PresumVecType b1; b1.clear();
  PresumVecType b2; b2.clear();
  PresumVecType b3; b3.clear();

  PresumIOToComputeType presum_io_to_compute_type;
  PresumComputeToIOType presum_compute_to_io_type;

  const uint IterationsA = (PresumShapeA::kM*PresumShapeA::kN*sizeof(ElementA))/(sizeof(PresumVecType)*kThreadCount);
  //TODO: Separate IterationsA and IterationsB
  for (int presum_iter = 0; presum_iter < IterationsA * presum_multiplier_a; presum_iter++) {
    const uint presum_tile = presum_iter / IterationsA;
    iter_PresumA.reset(presum_tile);
    iter_PresumA_M.reset(presum_tile);
    iter_PresumA.set_iteration(presum_iter - presum_tile * IterationsA);
    iter_PresumA_M.set_iteration(presum_iter - presum_tile * IterationsA);

    if (!(iter_PresumA_M.validTB() && iter_PresumA_M.valid())) continue;
    if (presumAComputeLoads.numAccess() > 0) {
      //Do loads needed for Computing Global Presums
      if (presumAComputeLoads.hasAccess(MmaStrassen::APresums::A0)) {
        arch::global_load<PresumVecType, sizeof(PresumVecType)>(a0, iter_PresumA.get(0),
                                                                iter_PresumA_M.validTB());
      }
      if (presumAComputeLoads.hasAccess(MmaStrassen::APresums::A1)) {
        arch::global_load<PresumVecType, sizeof(PresumVecType)>(a1, iter_PresumA.get(1),
                                                                iter_PresumA_M.validTB());
      }
      if (presumAComputeLoads.hasAccess(MmaStrassen::APresums::A2)) {
        arch::global_load<PresumVecType, sizeof(PresumVecType)>(a2, iter_PresumA.get(2),
                                                                iter_PresumA_M.validTB());
      }
      if (presumAComputeLoads.hasAccess(MmaStrassen::APresums::A3)) {
        arch::global_load<PresumVecType, sizeof(PresumVecType)>(a3, iter_PresumA.get(3),
                                                                iter_PresumA_M.validTB());
      }
      // iter_PresumA.inc();
    }

    auto s1   = presum_io_to_compute_type(a2) + presum_io_to_compute_type(a3);
    auto s2   = s1 - presum_io_to_compute_type(a0);
    auto a02  = presum_io_to_compute_type(a0) - presum_io_to_compute_type(a2);
    auto a1s2 = presum_io_to_compute_type(a1) - s2;

    if (AllPresums::doesComputeA(MmaStrassen::APresums::A02, PresumGlobalKernel)) {
      arch::global_store<PresumVecType, sizeof(PresumVecType)>(presum_compute_to_io_type(a02), iter_PresumA_M.get(AllPresums::indexAPresum(MmaStrassen::APresums::A02)),
                                                               iter_PresumA_M.validTB());
    }
    if (AllPresums::doesComputeA(MmaStrassen::APresums::S1, PresumGlobalKernel)) {
      arch::global_store<PresumVecType, sizeof(PresumVecType)>(presum_compute_to_io_type(s1), iter_PresumA_M.get(AllPresums::indexAPresum(MmaStrassen::APresums::S1)),
                                                               iter_PresumA_M.validTB());
    }
    if (AllPresums::doesComputeA(MmaStrassen::APresums::S2, PresumGlobalKernel)) {
      arch::global_store<PresumVecType, sizeof(PresumVecType)>(presum_compute_to_io_type(s2), iter_PresumA_M.get(AllPresums::indexAPresum(MmaStrassen::APresums::S2)),
                                                               iter_PresumA_M.validTB());
    }
    if (AllPresums::doesComputeA(MmaStrassen::APresums::A1S2, PresumGlobalKernel)) {
      arch::global_store<PresumVecType, sizeof(PresumVecType)>(presum_compute_to_io_type(a1s2), iter_PresumA_M.get(AllPresums::indexAPresum(MmaStrassen::APresums::A1S2)),
                                                               iter_PresumA_M.validTB());
    }

    // iter_PresumA_M.inc();
  }

  const uint IterationsB = (PresumShapeB::kM*PresumShapeB::kN*sizeof(ElementB))/(sizeof(PresumVecType)*kThreadCount);
  for (int presum_iter = 0; presum_iter < IterationsB * presum_multiplier_b; presum_iter++) {
    const uint presum_tile = presum_iter / IterationsB;
    iter_PresumB.reset(presum_tile);
    iter_PresumB_M.reset(presum_tile);
    iter_PresumB.set_iteration(presum_iter - presum_tile * IterationsB);
    iter_PresumB_M.set_iteration(presum_iter - presum_tile * IterationsB);

    if (!(iter_PresumB_M.validTB() && iter_PresumB_M.valid())) continue;

    if (presumBComputeLoads.numAccess() > 0) {
      if (presumBComputeLoads.hasAccess(MmaStrassen::BPresums::B0)) {
        arch::global_load<PresumVecType, sizeof(PresumVecType)>(b0, iter_PresumB.get(0),
                                                                iter_PresumB_M.validTB());
      }
      if (presumBComputeLoads.hasAccess(MmaStrassen::BPresums::B1)) {
        arch::global_load<PresumVecType, sizeof(PresumVecType)>(b1, iter_PresumB.get(1),
                                                                iter_PresumB_M.validTB());
      }
      if (presumBComputeLoads.hasAccess(MmaStrassen::BPresums::B2)) {
        arch::global_load<PresumVecType, sizeof(PresumVecType)>(b2, iter_PresumB.get(2),
                                                                iter_PresumB_M.validTB());
      }
      if (presumBComputeLoads.hasAccess(MmaStrassen::BPresums::B3)) {
        arch::global_load<PresumVecType, sizeof(PresumVecType)>(b3, iter_PresumB.get(3),
                                                                iter_PresumB_M.validTB());
      }
      // iter_PresumB.inc();
    }

    auto b10  = presum_io_to_compute_type(b1) - presum_io_to_compute_type(b0);
    auto b31  = presum_io_to_compute_type(b3) - presum_io_to_compute_type(b1);
    auto s3   = b31 + presum_io_to_compute_type(b0);
    auto s3b2 = s3 - presum_io_to_compute_type(b2);


    if (AllPresums::doesComputeB(MmaStrassen::BPresums::B10, PresumGlobalKernel)) {
      arch::global_store<PresumVecType, sizeof(PresumVecType)>(presum_compute_to_io_type(b10), iter_PresumB_M.get(AllPresums::indexBPresum(MmaStrassen::BPresums::B10)),
                                                               iter_PresumB_M.validTB());
    }
    if (AllPresums::doesComputeB(MmaStrassen::BPresums::B31, PresumGlobalKernel)) {
      arch::global_store<PresumVecType, sizeof(PresumVecType)>(presum_compute_to_io_type(b31), iter_PresumB_M.get(AllPresums::indexBPresum(MmaStrassen::BPresums::B31)),
                                                               iter_PresumB_M.validTB());
    }
    if (AllPresums::doesComputeB(MmaStrassen::BPresums::S3, PresumGlobalKernel)) {
      arch::global_store<PresumVecType, sizeof(PresumVecType)>(presum_compute_to_io_type(s3), iter_PresumB_M.get(AllPresums::indexBPresum(MmaStrassen::BPresums::S3)),
                                                               iter_PresumB_M.validTB());
    }
    if (AllPresums::doesComputeB(MmaStrassen::BPresums::S3B2, PresumGlobalKernel)) {
      arch::global_store<PresumVecType, sizeof(PresumVecType)>(presum_compute_to_io_type(s3b2), iter_PresumB_M.get(AllPresums::indexBPresum(MmaStrassen::BPresums::S3B2)),
                                                               iter_PresumB_M.validTB());
    }

    // iter_PresumB_M.inc();
  }
}
}
}
}