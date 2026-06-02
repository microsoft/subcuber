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

#include "cutlass/tensor_ref.h"
#include "cutlass/aligned_buffer.h"
#include "cutlass/arch/memory.h"
#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_types.h"
#include <bit>

#include "cutlass/gemm/device/strassen_decls.h"

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {
namespace gemm {
namespace threadblock {

////////////////////////////////////////////////////////////////////////////////

// /// Policy object describing MmaTensorOp
// template <
//     /// Warp-level GEMM operator (concept: gemm::warp::Mma)
//     typename Operator_,
//     /// Padding used for A operand in shared memory (concept: MatrixShape)
//     typename SmemPaddingA_,
//     /// Padding used for B operand in shared memory (concept: MatrixShape)
//     typename SmemPaddingB_,
//     /// Number of partitions of K dimension of GEMM
//     int PartitionsK = 1>
// struct MmaPolicy {
//   /// Warp-level GEMM operator (concept: gemm::warp::MmaTensorOp or gemm::warp::MmaSimt)
//   using Operator = Operator_;

//   /// Padding used for A operand in shared memory
//   using SmemPaddingA = SmemPaddingA_;

//   /// Padding used for B operand in shared memory
//   using SmemPaddingB = SmemPaddingB_;

//   /// Number of partitions of K dimension
//   static int const kPartitionsK = PartitionsK;
// };

////////////////////////////////////////////////////////////////////////////////

/// Structure to compute the matrix product targeting CUDA cores and SIMT math
/// instructions.
template <
    /// Size of the Gemm problem - concept: gemm::GemmShape<>
    typename Shape_,
    typename StrassenShape_,
    /// Policy describing tuning details (concept: MmaPolicy)
    typename Policy_,
    /// Number of stages,
    int Stages_,
    MmaStrassen::Type MmaStrassenKind_,
    typename StrassenMiGroup_,
    /// Used for partial specialization
    typename Enable = bool>
class StrassenMmaBase {
 public:
  static const int Stages = Stages_;
  static const auto MmaStrassenKind = MmaStrassenKind_;
  ///< Size of the Gemm problem - concept: gemm::GemmShape<>
  using Shape = Shape_;
  using StrassenShape = StrassenShape_;
  using StrassenMiGroup = StrassenMiGroup_;

  ///< Policy describing tuning details
  using Policy = Policy_;

  //
  // Dependent types
  //

  /// Warp-level Mma
  using Operator = typename Policy::Operator;

  /// Shape describing the overall GEMM computed from shared memory
  /// by each warp.
  using WarpGemm = typename Policy::Operator::Shape;

  /// Shape describing the number of warps filling the CTA
  using WarpCount = GemmShape<Shape::kM / WarpGemm::kM,
                              Shape::kN / WarpGemm::kN,
                              Shape::kK / WarpGemm::kK>;
  static const int NumThreads = WarpCount::kCount * 32;
 
  /// Number of warp-level GEMM oeprations
  static int const kWarpGemmIterations =
      (WarpGemm::kK / Operator::Policy::MmaShape::kK);

  /// Number of stages
  static int const kStages = Stages;

  /// Tensor reference to the A operand
  using TensorRefA = TensorRef<typename Operator::ElementA, typename Operator::LayoutA>;

  /// Tensor reference to the B operand
  using TensorRefB = TensorRef<typename Operator::ElementB, typename Operator::LayoutB>;
  using TensorRefD = TensorRef<typename Operator::ElementC, typename Operator::LayoutC>;

  static_assert(kWarpGemmIterations > 1,
                "The pipelined structure requires at least two warp-level "
                "GEMM operations.");

  static_assert((kWarpGemmIterations % 2) == 0,
                "Inner loop iteration must be an even number.");

  //
  // Nested structs
  //

  template<int kNumLoadsForPresum, int Stages, int NumThreads, typename ElementA>
  class PresumBuffer {
    public:
      static const int SingleStageSize = NumThreads * 16/sizeof(ElementA);
      static const int NumLoadsForPresum = kNumLoadsForPresum;
    private:
      AlignedBuffer<ElementA, SingleStageSize * NumLoadsForPresum * Stages> buff;
    public:
      CUTLASS_DEVICE ElementA* data() {return buff.data();}
      CUTLASS_DEVICE ElementA* data(int load) {return buff.data() + load * SingleStageSize * Stages;}
  };

  template<int NumThreads, int Stages, typename ElementA>
  class PresumBuffer<0, Stages, NumThreads, ElementA> {
    public:
      static const int SingleStageSize = 0;
      static const int NumLoadsForPresum = 0;

      CUTLASS_DEVICE ElementA* data() {return nullptr;}
      CUTLASS_DEVICE ElementA* data(int load) {return nullptr;}
  };

  CUTLASS_HOST_DEVICE
  static constexpr bool RequireNoLocalPresum() {
    if constexpr (StrassenMiGroup_::numMs() == 1 and
                  StrassenMiGroup_::APresumLoads().numAccess() == 1 and StrassenMiGroup_::APresumSharedBuffs() == 1 and
                  StrassenMiGroup_::BPresumLoads().numAccess() == 1 and StrassenMiGroup_::BPresumSharedBuffs() == 1)
      return true;
    return false;
  }

  template<MmaStrassen::Type MmaKind, int Stages, typename MmaShape,
           typename ElementA, typename StrassenShapeA, typename SingleStrassenShapeA,
           typename ElementB, typename StrassenShapeB, typename SingleStrassenShapeB>
  class SharedBuffers {
    CUTLASS_HOST_DEVICE
    static constexpr bool RequireNoLocalPresum() {
      if constexpr (StrassenMiGroup_::numMs() == 1 and 
                    StrassenMiGroup_::APresumLoads().numAccess() == 1 and StrassenMiGroup_::APresumSharedBuffs() == 1 and
                    StrassenMiGroup_::BPresumLoads().numAccess() == 1 and StrassenMiGroup_::BPresumSharedBuffs() == 1)
        return true;
      return false;
    }

    AlignedBuffer<ElementA, StrassenShapeA::kCount> operand_As[std::max(StrassenMiGroup_::APresumLoads().numAccess(),
                                                                        StrassenMiGroup_::APresumSharedBuffs())];
    AlignedBuffer<ElementB, StrassenShapeB::kCount> operand_Bs[std::max(StrassenMiGroup_::BPresumLoads().numAccess(),
                                                                        StrassenMiGroup_::BPresumSharedBuffs())];

    public:
    CUTLASS_DEVICE ElementA* dataA0()   {return operand_As[StrassenMiGroup_::APresumLoads().index(0)].data();}
    CUTLASS_DEVICE ElementA* dataA1()   {return operand_As[StrassenMiGroup_::APresumLoads().index(1)].data();}
    CUTLASS_DEVICE ElementA* dataA2()   {return operand_As[StrassenMiGroup_::APresumLoads().index(2)].data();}
    CUTLASS_DEVICE ElementA* dataA3()   {return operand_As[StrassenMiGroup_::APresumLoads().index(3)].data();}
    CUTLASS_DEVICE ElementA* dataA02()  {return operand_As[StrassenMiGroup_::APresumLoads().index(StrassenMiGroup_::APresums::A02)].data();}
    CUTLASS_DEVICE ElementA* dataS1()   {return operand_As[StrassenMiGroup_::APresumLoads().index(StrassenMiGroup_::APresums::S1)].data();}
    CUTLASS_DEVICE ElementA* dataS2()   {return operand_As[StrassenMiGroup_::APresumLoads().index(StrassenMiGroup_::APresums::S2)].data();}
    CUTLASS_DEVICE ElementA* dataA1S2() {return operand_As[StrassenMiGroup_::APresumLoads().index(StrassenMiGroup_::APresums::A1S2)].data();}
    
    CUTLASS_DEVICE ElementA* dataA_M0() {return operand_As[StrassenMiGroup_::APresumStores(true).index(StrassenMiGroup_::APresums::A0)].data();}
    CUTLASS_DEVICE ElementA* dataA_M1() {return operand_As[StrassenMiGroup_::APresumStores(true).index(StrassenMiGroup_::APresums::A1)].data();}
    CUTLASS_DEVICE ElementA* dataA_M2() {return operand_As[StrassenMiGroup_::APresumStores(true).index(StrassenMiGroup_::APresums::S2)].data();}
    CUTLASS_DEVICE ElementA* dataA_M3() {return operand_As[StrassenMiGroup_::APresumStores(true).index(StrassenMiGroup_::APresums::A02)].data();}
    CUTLASS_DEVICE ElementA* dataA_M4() {return operand_As[StrassenMiGroup_::APresumStores(true).index(StrassenMiGroup_::APresums::S1)].data();}
    CUTLASS_DEVICE ElementA* dataA_M5() {return operand_As[StrassenMiGroup_::APresumStores(true).index(StrassenMiGroup_::APresums::A1S2)].data();}
    CUTLASS_DEVICE ElementA* dataA_M6() {return operand_As[StrassenMiGroup_::APresumStores(true).index(StrassenMiGroup_::APresums::A3)].data();}

    CUTLASS_DEVICE ElementB* dataB0()   {return operand_Bs[StrassenMiGroup_::BPresumLoads().index(0)].data();}
    CUTLASS_DEVICE ElementB* dataB1()   {return operand_Bs[StrassenMiGroup_::BPresumLoads().index(1)].data();}
    CUTLASS_DEVICE ElementB* dataB2()   {return operand_Bs[StrassenMiGroup_::BPresumLoads().index(2)].data();}
    CUTLASS_DEVICE ElementB* dataB3()   {return operand_Bs[StrassenMiGroup_::BPresumLoads().index(3)].data();}
    CUTLASS_DEVICE ElementB* dataB10()  {return operand_Bs[StrassenMiGroup_::BPresumLoads().index(StrassenMiGroup_::BPresums::B10)].data();}
    CUTLASS_DEVICE ElementB* dataB31()  {return operand_Bs[StrassenMiGroup_::BPresumLoads().index(StrassenMiGroup_::BPresums::B31)].data();}
    CUTLASS_DEVICE ElementB* dataS3()   {return operand_Bs[StrassenMiGroup_::BPresumLoads().index(StrassenMiGroup_::BPresums::S3)].data();}
    CUTLASS_DEVICE ElementB* dataS3B2() {return operand_Bs[StrassenMiGroup_::BPresumLoads().index(StrassenMiGroup_::BPresums::S3B2)].data();}
    
    CUTLASS_DEVICE ElementB* dataB_M0() {return operand_Bs[StrassenMiGroup_::BPresumStores(true).index(StrassenMiGroup_::BPresums::B0)].data();}
    CUTLASS_DEVICE ElementB* dataB_M1() {return operand_Bs[StrassenMiGroup_::BPresumStores(true).index(StrassenMiGroup_::BPresums::B2)].data();}
    CUTLASS_DEVICE ElementB* dataB_M2() {return operand_Bs[StrassenMiGroup_::BPresumStores(true).index(StrassenMiGroup_::BPresums::S3)].data();}
    CUTLASS_DEVICE ElementB* dataB_M3() {return operand_Bs[StrassenMiGroup_::BPresumStores(true).index(StrassenMiGroup_::BPresums::B31)].data();}
    CUTLASS_DEVICE ElementB* dataB_M4() {return operand_Bs[StrassenMiGroup_::BPresumStores(true).index(StrassenMiGroup_::BPresums::B10)].data();}
    CUTLASS_DEVICE ElementB* dataB_M5() {return operand_Bs[StrassenMiGroup_::BPresumStores(true).index(StrassenMiGroup_::BPresums::B3)].data();}
    CUTLASS_DEVICE ElementB* dataB_M6() {return operand_Bs[StrassenMiGroup_::BPresumStores(true).index(StrassenMiGroup_::BPresums::S3B2)].data();}

    CUTLASS_DEVICE ElementA* data_input_A()  {
      if (RequireNoLocalPresum()) return operand_As[0].data();
      return nullptr;
    }

    CUTLASS_DEVICE ElementB* data_input_B()  {
      if (RequireNoLocalPresum()) return operand_Bs[0].data();
      return nullptr;
    }
  };

  template<MmaStrassen::Type MmaKind,
           typename MmaShape,
           typename ElementA, typename StrassenShapeA, typename SingleStrassenShapeA,
           typename ElementB, typename StrassenShapeB, typename SingleStrassenShapeB>
  class SharedBuffers<MmaKind, 2, MmaShape, 
                      ElementA, StrassenShapeA, SingleStrassenShapeA,
                      ElementB, StrassenShapeB, SingleStrassenShapeB> {
    AlignedBuffer<ElementA, StrassenShapeA::kCount> operand_As[StrassenMiGroup_::APresumSharedBuffs()];
    AlignedBuffer<ElementB, StrassenShapeB::kCount> operand_Bs[StrassenMiGroup_::BPresumSharedBuffs()];

    public:
    CUTLASS_DEVICE ElementA* dataA0() {return nullptr;}
    CUTLASS_DEVICE ElementA* dataA1() {return nullptr;}
    CUTLASS_DEVICE ElementA* dataA2() {return nullptr;}
    CUTLASS_DEVICE ElementA* dataA3() {return nullptr;}

    CUTLASS_DEVICE ElementA* dataA02()  {return nullptr;}
    CUTLASS_DEVICE ElementA* dataS1()   {return nullptr;}
    CUTLASS_DEVICE ElementA* dataS2()   {return nullptr;}
    CUTLASS_DEVICE ElementA* dataA1S2() {return nullptr;}

    CUTLASS_DEVICE ElementA* dataB0() {return nullptr;}
    CUTLASS_DEVICE ElementA* dataB1() {return nullptr;}
    CUTLASS_DEVICE ElementA* dataB2() {return nullptr;}
    CUTLASS_DEVICE ElementA* dataB3() {return nullptr;}

    CUTLASS_DEVICE ElementB* dataB10()  {return nullptr;}
    CUTLASS_DEVICE ElementB* dataB31()  {return nullptr;}
    CUTLASS_DEVICE ElementB* dataS3()   {return nullptr;}
    CUTLASS_DEVICE ElementB* dataS3B2() {return nullptr;}

    CUTLASS_DEVICE ElementA* dataA_M0()   {return operand_As[StrassenMiGroup_::APresumStores().index(StrassenMiGroup_::APresums::A0)].data();}
    CUTLASS_DEVICE ElementA* dataA_M1()   {return operand_As[StrassenMiGroup_::APresumStores().index(StrassenMiGroup_::APresums::A1)].data();}
    CUTLASS_DEVICE ElementA* dataA_M2()   {return operand_As[StrassenMiGroup_::APresumStores().index(StrassenMiGroup_::APresums::S2)].data();}
    CUTLASS_DEVICE ElementA* dataA_M3()   {return operand_As[StrassenMiGroup_::APresumStores().index(StrassenMiGroup_::APresums::A02)].data();}
    CUTLASS_DEVICE ElementA* dataA_M4()  {return operand_As[StrassenMiGroup_::APresumStores().index(StrassenMiGroup_::APresums::S1)].data();}
    CUTLASS_DEVICE ElementA* dataA_M5()   {return operand_As[StrassenMiGroup_::APresumStores().index(StrassenMiGroup_::APresums::A1S2)].data();}
    CUTLASS_DEVICE ElementA* dataA_M6() {return operand_As[StrassenMiGroup_::APresumStores().index(StrassenMiGroup_::APresums::A3)].data();}

    CUTLASS_DEVICE ElementB* dataB_M0()   {return operand_Bs[StrassenMiGroup_::BPresumStores().index(StrassenMiGroup_::BPresums::B0)].data();}
    CUTLASS_DEVICE ElementB* dataB_M1()   {return operand_Bs[StrassenMiGroup_::BPresumStores().index(StrassenMiGroup_::BPresums::B2)].data();}
    CUTLASS_DEVICE ElementB* dataB_M2()   {return operand_Bs[StrassenMiGroup_::BPresumStores().index(StrassenMiGroup_::BPresums::S3)].data();}
    CUTLASS_DEVICE ElementB* dataB_M3()   {return operand_Bs[StrassenMiGroup_::BPresumStores().index(StrassenMiGroup_::BPresums::B31)].data();}
    CUTLASS_DEVICE ElementB* dataB_M4()  {return operand_Bs[StrassenMiGroup_::BPresumStores().index(StrassenMiGroup_::BPresums::B10)].data();}
    CUTLASS_DEVICE ElementB* dataB_M5()  {return operand_Bs[StrassenMiGroup_::BPresumStores().index(StrassenMiGroup_::BPresums::B3)].data();}
    CUTLASS_DEVICE ElementB* dataB_M6() {return operand_Bs[StrassenMiGroup_::BPresumStores().index(StrassenMiGroup_::BPresums::S3B2)].data();}

    CUTLASS_DEVICE ElementA* data_input_A()  {
      if (RequireNoLocalPresum()) return operand_As[0].data();
      return nullptr;
    }

    CUTLASS_DEVICE ElementB* data_input_B()  {
      if (RequireNoLocalPresum()) return operand_Bs[0].data();
      return nullptr;
    }
  };

  /// Shared storage object needed by threadblock-scoped GEMM
  class SharedStorage {
   public:
    //
    // Type definitions
    //

    /// Shape of the A matrix operand in shared memory
    using ShapeA = MatrixShape<Shape::kM + Policy::SmemPaddingA::kRow,
                               Shape::kK * kStages +
                                   Policy::SmemPaddingA::kColumn>;
    using StrassenShapeA = MatrixShape<StrassenShape::kM + Policy::SmemPaddingA::kRow,
                                       StrassenShape::kK * kStages + 
                                        Policy::SmemPaddingA::kColumn>;
    using SingleStageStrassenShapeA = MatrixShape<StrassenShape::kM + Policy::SmemPaddingA::kRow,
                                       StrassenShape::kK + 
                                        Policy::SmemPaddingA::kColumn>;
    /// Shape of the B matrix operand in shared memory
    using ShapeB =
        MatrixShape<Shape::kK * kStages + Policy::SmemPaddingB::kRow,
                    Shape::kN + Policy::SmemPaddingB::kColumn>;
    using StrassenShapeB =
        MatrixShape<StrassenShape::kK * kStages + Policy::SmemPaddingB::kRow,
                    StrassenShape::kN + Policy::SmemPaddingB::kColumn>;
    using SingleStageStrassenShapeB =
        MatrixShape<StrassenShape::kK + Policy::SmemPaddingB::kRow,
                    StrassenShape::kN + Policy::SmemPaddingB::kColumn>; 

    // using ShapeM2M4M0 = MatrixShape<(StoreMinSharedMemory) ? StrassenShape::kM : 1, (StoreMinSharedMemory) ? StrassenShape::kN : 1>;

   public:
    //
    // Data members
    //

    /// Buffer for A operand
    SharedBuffers<MmaStrassenKind, Stages, Shape,
                  typename Operator::ElementA, StrassenShapeA, SingleStageStrassenShapeA, 
                  typename Operator::ElementB, StrassenShapeB, SingleStageStrassenShapeB> buffers;
    using PresumBuffer = PresumBuffer<StrassenMiGroup::FusedOrContinueMMA() == 1 ? 4 :
                                      std::max(StrassenMiGroup::AllPresums::APresumComputeLoads().numAccess(),
                                               StrassenMiGroup::AllPresums::BPresumComputeLoads().numAccess()),
                                      std::max(Stages, 4), NumThreads, typename Operator::ElementA>;
    PresumBuffer presum_buff;

   public:

    //
    // Methods
    //

    /// Returns a layout object for the A matrix
    CUTLASS_DEVICE
    static typename Operator::LayoutA LayoutA() {
      return Operator::LayoutA::packed({StrassenShapeA::kRow, StrassenShapeA::kColumn});
    }

    CUTLASS_HOST_DEVICE
    static typename Operator::LayoutA LayoutSingleStageA() {
      return Operator::LayoutA::packed({SingleStageStrassenShapeA::kRow, SingleStageStrassenShapeA::kColumn});
    }

    /// Returns a layout object for the B matrix
    CUTLASS_HOST_DEVICE
    static typename Operator::LayoutB LayoutB() {
      return Operator::LayoutB::packed({StrassenShapeB::kRow, StrassenShapeB::kColumn});
    }
    CUTLASS_HOST_DEVICE
    static typename Operator::LayoutB LayoutSingleStageB() {
      return Operator::LayoutB::packed({SingleStageStrassenShapeB::kRow, SingleStageStrassenShapeB::kColumn});
    }

    // CUTLASS_HOST_DEVICE
    // static typename Operator::LayoutC LayoutC() {
    //   return Operator::LayoutC::packed({ShapeM2M4M0::kRow, ShapeM2M4M0::kColumn});
    // }

    /// Returns a TensorRef to the A operand
    CUTLASS_HOST_DEVICE
    TensorRefA operand_A0_ref() {
      return TensorRefA{buffers.dataA0(), LayoutA()};
    }
    CUTLASS_HOST_DEVICE
    TensorRefA operand_A1_ref() {
      return TensorRefA{buffers.dataA1(), LayoutA()};
    }
    CUTLASS_HOST_DEVICE
    TensorRefA operand_A2_ref() {
      return TensorRefA{buffers.dataA2(), LayoutA()};
    }
    CUTLASS_HOST_DEVICE
    TensorRefA operand_A3_ref() {
      return TensorRefA{buffers.dataA3(), LayoutA()};
    }
    CUTLASS_HOST_DEVICE
    TensorRefA operand_A02_ref() {
      return TensorRefA{buffers.dataA02(), LayoutA()};
    }
    CUTLASS_HOST_DEVICE
    TensorRefA operand_S1_ref() {
      return TensorRefA{buffers.dataS1(), LayoutA()};
    }
    CUTLASS_HOST_DEVICE
    TensorRefA operand_S2_ref() {
      return TensorRefA{buffers.dataS2(), LayoutA()};
    }
    CUTLASS_HOST_DEVICE
    TensorRefA operand_A1S2_ref() {
      return TensorRefA{buffers.dataA1S2(), LayoutA()};
    }

    CUTLASS_HOST_DEVICE
    TensorRefA operand_A_M0_ref() {
      return TensorRefA{buffers.dataA_M0(), LayoutA()};
    }
    CUTLASS_HOST_DEVICE
    TensorRefA operand_A_M1_ref() {
      return TensorRefA{buffers.dataA_M1(), LayoutA()};
    }
    CUTLASS_HOST_DEVICE
    TensorRefA operand_A_M2_ref() {
      return TensorRefA{buffers.dataA_M2(), LayoutA()};
    }
    CUTLASS_HOST_DEVICE
    TensorRefA operand_A_M3_ref() {
      return TensorRefA{buffers.dataA_M3(), LayoutA()};
    }
    CUTLASS_HOST_DEVICE
    TensorRefA operand_A_M4_ref() {
      return TensorRefA{buffers.dataA_M4(), LayoutA()};
    }
    CUTLASS_HOST_DEVICE
    TensorRefA operand_A_M5_ref() {
      return TensorRefA{buffers.dataA_M5(), LayoutA()};
    }
    CUTLASS_HOST_DEVICE
    TensorRefA operand_A_M6_ref() {
      return TensorRefA{buffers.dataA_M6(), LayoutA()};
    }

    CUTLASS_HOST_DEVICE
    TensorRefA operand_input_A_ref() {
      return TensorRefA{buffers.data_input_A(), LayoutA()};
    }

    /// Returns a TensorRef to the B operand
    CUTLASS_HOST_DEVICE
    TensorRefB operand_B0_ref() {
      return TensorRefB{buffers.dataB0(), LayoutB()};
    }
    CUTLASS_HOST_DEVICE
    TensorRefB operand_B1_ref() {
      return TensorRefB{buffers.dataB1(), LayoutB()};
    }
    CUTLASS_HOST_DEVICE
    TensorRefB operand_B2_ref() {
      return TensorRefB{buffers.dataB2(), LayoutB()};
    }
    CUTLASS_HOST_DEVICE
    TensorRefB operand_B3_ref() {
      return TensorRefB{buffers.dataB3(), LayoutB()};
    }
    CUTLASS_HOST_DEVICE
    TensorRefB operand_B31_ref() {
      return TensorRefB{buffers.dataB31(), LayoutB()};
    }
    CUTLASS_HOST_DEVICE
    TensorRefB operand_B10_ref() {
      return TensorRefB{buffers.dataB10(), LayoutB()};
    }
    CUTLASS_HOST_DEVICE
    TensorRefB operand_S3_ref() {
      return TensorRefB{buffers.dataS3(), LayoutB()};
    }
    CUTLASS_HOST_DEVICE
    TensorRefB operand_S3B2_ref() {
      return TensorRefB{buffers.dataS3B2(), LayoutB()};
    }

    CUTLASS_HOST_DEVICE
    TensorRefB operand_B_M0_ref() {
      return TensorRefB{buffers.dataB_M0(), LayoutB()};
    }
    CUTLASS_HOST_DEVICE
    TensorRefB operand_B_M1_ref() {
      return TensorRefB{buffers.dataB_M1(), LayoutB()};
    }
    CUTLASS_HOST_DEVICE
    TensorRefB operand_B_M2_ref() {
      return TensorRefB{buffers.dataB_M2(), LayoutB()};
    }
    CUTLASS_HOST_DEVICE
    TensorRefB operand_B_M3_ref() {
      return TensorRefB{buffers.dataB_M3(), LayoutB()};
    }
    CUTLASS_HOST_DEVICE
    TensorRefB operand_B_M4_ref() {
      return TensorRefB{buffers.dataB_M4(), LayoutB()};
    }
    CUTLASS_HOST_DEVICE
    TensorRefB operand_B_M5_ref() {
      return TensorRefB{buffers.dataB_M5(), LayoutB()};
    }
    CUTLASS_HOST_DEVICE
    TensorRefB operand_B_M6_ref() {
      return TensorRefB{buffers.dataB_M6(), LayoutB()};
    }

    CUTLASS_HOST_DEVICE
    TensorRefB operand_input_B_ref() {
      return TensorRefB{buffers.data_input_B(), LayoutB()};
    }

    CUTLASS_HOST_DEVICE
    typename Operator::ElementA* operand_presum(int i) {return presum_buff.data(i);}
  };

 protected:

  //
  // Data members
  //

  /// Iterator to load a warp-scoped tile of A operand from shared memory
  typename Operator::IteratorA warp_tile_iterator_A_M_[7];
  typename Operator::IteratorA warp_tile_iterator_input_A_;
  using WarpTileIteratorA = typename Operator::IteratorA;

  /// Iterator to load a warp-scoped tile of B operand from shared memory
  typename Operator::IteratorB warp_tile_iterator_B_M_[7];
  typename Operator::IteratorB warp_tile_iterator_input_B_;
  using WarpTileIteratorB = typename Operator::IteratorB;

public:

  /// Construct from tensor references
  CUTLASS_DEVICE
  StrassenMmaBase(
      ///< Shared storage needed for internal use by threadblock-scoped GEMM
      SharedStorage &shared_storage,
      ///< ID within the threadblock
      int thread_idx,
      ///< ID of warp
      int warp_idx,
      ///< ID of each thread within a warp
      int lane_idx
    ):
      warp_tile_iterator_B_M_{
        WarpTileIteratorB(shared_storage.operand_B_M0_ref(), lane_idx),
        WarpTileIteratorB(shared_storage.operand_B_M1_ref(), lane_idx),
        WarpTileIteratorB(shared_storage.operand_B_M2_ref(), lane_idx),
        WarpTileIteratorB(shared_storage.operand_B_M3_ref(), lane_idx),
        WarpTileIteratorB(shared_storage.operand_B_M4_ref(), lane_idx),
        WarpTileIteratorB(shared_storage.operand_B_M5_ref(), lane_idx),
        WarpTileIteratorB(shared_storage.operand_B_M6_ref(), lane_idx)
      },

      warp_tile_iterator_A_M_{
        WarpTileIteratorA(shared_storage.operand_A_M0_ref(), lane_idx),
        WarpTileIteratorA(shared_storage.operand_A_M1_ref(), lane_idx),
        WarpTileIteratorA(shared_storage.operand_A_M2_ref(), lane_idx),
        WarpTileIteratorA(shared_storage.operand_A_M3_ref(), lane_idx),
        WarpTileIteratorA(shared_storage.operand_A_M4_ref(), lane_idx),
        WarpTileIteratorA(shared_storage.operand_A_M5_ref(), lane_idx),
        WarpTileIteratorA(shared_storage.operand_A_M6_ref(), lane_idx)
      },

      warp_tile_iterator_input_A_(shared_storage.operand_input_A_ref(), lane_idx),
      warp_tile_iterator_input_B_(shared_storage.operand_input_B_ref(), lane_idx)
    {}
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace gemm
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
