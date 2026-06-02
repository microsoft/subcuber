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

#include "cutlass/gemm/threadblock/strassen_mma_base.h"

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
    int Stages,
    MmaStrassen::Type MmaStrassenKind_,
    typename StrassenMiGroup_,
    /// Used for partial specialization
    typename Enable = bool>
class GlobalStrassenMmaBase {
 public:
  static const auto MmaStrassenKind = MmaStrassenKind_;
  ///< Size of the Gemm problem - concept: gemm::GemmShape<>
  using Shape = Shape_;
  using StrassenShape = StrassenShape_;

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

  template<MmaStrassen::Type MmaKind,
           typename ElementA, typename StrassenShapeA,
           typename ElementB, typename StrassenShapeB>
  class SharedBuffers {
    AlignedBuffer<ElementA, StrassenShapeA::kCount> operand_A;
    AlignedBuffer<ElementB, StrassenShapeB::kCount> operand_B;
    public:
    CUTLASS_DEVICE ElementA* dataA() {return operand_A.data();}
    CUTLASS_DEVICE ElementB* dataB() {return operand_B.data();}
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
    /// Shape of the B matrix operand in shared memory
    using ShapeB =
        MatrixShape<Shape::kK * kStages + Policy::SmemPaddingB::kRow,
                    Shape::kN + Policy::SmemPaddingB::kColumn>;
     using StrassenShapeB =
        MatrixShape<StrassenShape::kK * kStages + Policy::SmemPaddingB::kRow,
                    StrassenShape::kN + Policy::SmemPaddingB::kColumn>;

    // using ShapeM2M4M0 = MatrixShape<(StoreMinSharedMemory) ? StrassenShape::kM : 1, (StoreMinSharedMemory) ? StrassenShape::kN : 1>;
    // typename Policy::SmemPaddingA::x y;
   public:
    //
    // Data members
    //

    /// Buffer for A operand
    SharedBuffers<MmaStrassenKind, typename Operator::ElementA, StrassenShapeA,
                  typename Operator::ElementB, StrassenShapeB> buffers;

    //TODO: not needed in Pipelined and not always needed in MmaMultistage
    SharedBuffers<MmaStrassenKind, typename Operator::ElementA, StrassenShapeA,
                  typename Operator::ElementB, StrassenShapeB> buffers_1;

   public:

    //
    // Methods
    //

    /// Returns a layout object for the A matrix
    CUTLASS_DEVICE
    static typename Operator::LayoutA LayoutA() {
      return Operator::LayoutA::packed({StrassenShapeA::kRow, StrassenShapeA::kColumn});
    }

    /// Returns a layout object for the B matrix
    CUTLASS_HOST_DEVICE
    static typename Operator::LayoutB LayoutB() {
      return Operator::LayoutB::packed({StrassenShapeB::kRow, StrassenShapeB::kColumn});
    }

    // CUTLASS_HOST_DEVICE
    // static typename Operator::LayoutC LayoutC() {
    //   return Operator::LayoutC::packed({ShapeM2M4M0::kRow, ShapeM2M4M0::kColumn});
    // }

    /// Returns a TensorRef to the A operand
    CUTLASS_HOST_DEVICE
    TensorRefA operand_A_ref() {
      return TensorRefA{buffers.dataA(), LayoutA()};
    }

    /// Returns a TensorRef to the B operand
    CUTLASS_HOST_DEVICE
    TensorRefB operand_B_ref() {
      return TensorRefB{buffers.dataB(), LayoutB()};
    }

    /// Returns a TensorRef to the A operand
    CUTLASS_HOST_DEVICE
    TensorRefA operand_A1_ref() {
      return TensorRefA{buffers_1.dataA(), LayoutA()}; //operand_A_ref() + TensorRefA::TensorCoord(StrassenShape::kM, 0); //
    }

    /// Returns a TensorRef to the B operand
    CUTLASS_HOST_DEVICE
    TensorRefB operand_B1_ref() {
      return TensorRefB{buffers_1.dataB(), LayoutB()};
    }
  };

 protected:

  //
  // Data members
  //

  /// Iterator to load a warp-scoped tile of A operand from shared memory
  typename Operator::IteratorA warp_tile_iterator_A_;

  /// Iterator to load a warp-scoped tile of B operand from shared memory
  typename Operator::IteratorB warp_tile_iterator_B_;

public:

  /// Construct from tensor references
  CUTLASS_DEVICE
  GlobalStrassenMmaBase(
      ///< Shared storage needed for internal use by threadblock-scoped GEMM
      SharedStorage &shared_storage,
      ///< ID within the threadblock
      int thread_idx,
      ///< ID of warp
      int warp_idx,
      ///< ID of each thread within a warp
      int lane_idx
    ):
      warp_tile_iterator_B_(shared_storage.operand_B_ref(), lane_idx),
      warp_tile_iterator_A_(shared_storage.operand_A_ref(), lane_idx) {}
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace threadblock
}  // namespace gemm
}  // namespace cutlass

/////////////////////////////////////////////////////////////////////////////////////////////////
