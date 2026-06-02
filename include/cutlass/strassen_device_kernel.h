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
    \brief Template for generic CUTLASS kernel.
*/

#pragma once

// __grid_constant__ was introduced in CUDA 11.7.
#if ((__CUDACC_VER_MAJOR__ >= 12) || ((__CUDACC_VER_MAJOR__ == 11) && (__CUDACC_VER_MINOR__ >= 7)))
#  define CUTLASS_GRID_CONSTANT_SUPPORTED
#endif

// __grid_constant__ can be enabled only on SM70+
#if defined(CUTLASS_GRID_CONSTANT_SUPPORTED) && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
#  define CUTLASS_GRID_CONSTANT_ENABLED
#endif

#if ! defined(CUTLASS_GRID_CONSTANT)
#  if defined(CUTLASS_GRID_CONSTANT_ENABLED)
#    define CUTLASS_GRID_CONSTANT __grid_constant__
#  else
#    define CUTLASS_GRID_CONSTANT
#  endif
#endif

////////////////////////////////////////////////////////////////////////////////

namespace cutlass {

////////////////////////////////////////////////////////////////////////////////

/// Generic CUTLASS kernel template.
// template <typename Operator>
// CUTLASS_GLOBAL
// void Kernel(typename Operator::Params params) {
//   // Dynamic shared memory base pointer
//   extern __shared__ int SharedStorageBase[];
//   // Declare pointer to dynamic shared memory.
//   typename Operator::SharedStorage *shared_storage =
//       reinterpret_cast<typename Operator::SharedStorage *>(SharedStorageBase);

//   Operator op;
//   typename Operator::Mma::FragmentC srcFragment; srcFragment.clear();
//   op(params, *shared_storage);
// }

template <typename Operator, uint Presum, uint ReadWrite>
CUTLASS_GLOBAL
void Kernel(typename Operator::Params params) {
  // Dynamic shared memory base pointer
  extern __shared__ int SharedStorageBase[];
  // Declare pointer to dynamic shared memory.
  typename Operator::SharedStorage *shared_storage =
      reinterpret_cast<typename Operator::SharedStorage *>(SharedStorageBase);
  dim3 baseGrid = {0,0};
  dim3 origGrid = gridDim;

  Operator op;
  typename Operator::Mma::FragmentC srcFragment; srcFragment.clear();
  op(params, *shared_storage, baseGrid, origGrid, Presum, ReadWrite);
}

template<typename kOperator1,
         uint64_t Schedule = 0, 
         uint64_t PresumM1 = 0, uint64_t ReadWriteM1 = 0>
class FusedOperator1 {
  public:
  using Operator1 = kOperator1;

  CUTLASS_DEVICE
  void operator()(typename Operator1::Params& params1,
                  int* SharedStorageBase,
                  int startBlock, int endBlock,
                  dim3 origGrid) {
    if (blockIdx.z >= startBlock * origGrid.z && blockIdx.z < endBlock * origGrid.z) {
      dim3 baseGrid = {0, 0, startBlock * origGrid.z};

      typename Operator1::SharedStorage *shared_storage =
          reinterpret_cast<typename Operator1::SharedStorage *>(SharedStorageBase);

      Operator1 op1;
      op1(params1, *shared_storage, baseGrid, origGrid, Schedule, PresumM1, ReadWriteM1);
    }
  }
};

template<typename kOperator1, typename kOperator2,
         uint64_t Schedule = 0,
         uint64_t PresumM1 = 0, uint64_t PresumM2 = 0, uint64_t ReadWriteM1 = 0, uint64_t ReadWriteM2 = 0>
class FusedOperator2 {
  public:
  using Operator1 = kOperator1;
  using Operator2 = kOperator2;

  CUTLASS_DEVICE
  void operator()(typename Operator1::Params params1,
                  int* SharedStorageBase,
                  uint startBlock, uint endBlock,
                  dim3 origGrid) {
    if (blockIdx.z >= startBlock * origGrid.z && blockIdx.z < endBlock * origGrid.z) {
      dim3 baseGrid = {0, 0, startBlock * origGrid.z};

      typename Operator1::SharedStorage *shared_storage_1 =
          reinterpret_cast<typename Operator1::SharedStorage *>(SharedStorageBase);

      Operator1 op1;
      op1(params1, *shared_storage_1, baseGrid, origGrid, Schedule, PresumM1, ReadWriteM1);

      typename Operator2::SharedStorage *shared_storage_2 =
          reinterpret_cast<typename Operator2::SharedStorage *>(SharedStorageBase);

      Operator2 op2;
      op2(typename Operator2::Params(params1), *shared_storage_2, baseGrid, origGrid, Schedule, PresumM2, ReadWriteM2);
    }
  }
};

template<typename kOperator1, typename kOperator2, typename kOperator3,
         uint64_t Schedule = 0,
         uint64_t PresumM1 = 0, uint64_t PresumM2 = 0, uint PresumM3 = 0,
         uint64_t ReadWriteM1 = 0, uint64_t ReadWriteM2 = 0, uint64_t ReadWriteM3 = 0>
class FusedOperator3 {
  public:
  using Operator1 = kOperator1;
  using Operator2 = kOperator2;
  using Operator3 = kOperator3;

  CUTLASS_DEVICE
  void operator()(typename Operator1::Params params1,
                  int* SharedStorageBase,
                  uint startBlock, uint endBlock,
                  dim3 origGrid) {
    if (blockIdx.z >= startBlock * origGrid.z && blockIdx.z < endBlock * origGrid.z) {
      dim3 baseGrid = {0, 0, startBlock * origGrid.z};

      typename Operator1::SharedStorage *shared_storage_1 =
          reinterpret_cast<typename Operator1::SharedStorage *>(SharedStorageBase);

      Operator1 op1;
      op1(params1, *shared_storage_1, baseGrid, origGrid, Schedule, PresumM1, ReadWriteM1);

      typename Operator2::SharedStorage *shared_storage_2 =
          reinterpret_cast<typename Operator2::SharedStorage *>(SharedStorageBase);

      Operator2 op2;
      auto params2 = typename Operator2::Params(params1);
      op2(params2, *shared_storage_2, baseGrid, origGrid, Schedule, PresumM2, ReadWriteM2);

      typename Operator3::SharedStorage *shared_storage_3 =
          reinterpret_cast<typename Operator3::SharedStorage *>(SharedStorageBase);

      Operator3 op3;
      auto params3 = typename Operator3::Params(params1);
      op3(params3, *shared_storage_3, baseGrid, origGrid, Schedule, PresumM3, ReadWriteM3);
    }
  }
};


template<typename kOperator1, typename kOperator2, typename kOperator3,
         uint64_t Schedule = 0,
         uint64_t PresumM1 = 0, uint64_t PresumM2 = 0, uint64_t PresumM3 = 0,
         uint64_t ReadWriteM1 = 0, uint64_t ReadWriteM2 = 0, uint64_t ReadWriteM3 = 0>
class ParallelOperator3 {
  public:
  using Operator1 = kOperator1;
  using Operator2 = kOperator2;
  using Operator3 = kOperator3;

  CUTLASS_DEVICE
  void operator()(typename Operator1::Params params1,
                  int* SharedStorageBase,
                  uint startBlock, uint endBlock,
                  dim3 origGrid) {
    dim3 baseGrid = {0,0,0};
    if (threadIdx.x == 0 && blockIdx.x == 0) printf("205: Do not work\n");
    if (blockIdx.y >= startBlock * origGrid.y &&
      blockIdx.y < endBlock * origGrid.y) {
      baseGrid = {0, startBlock * origGrid.y};
    } else if (blockIdx.y >= (startBlock + 1) * origGrid.y &&
               blockIdx.y < (endBlock + 1) * origGrid.y) {
      baseGrid = {0, (startBlock + 1) * origGrid.y};
    } else if (blockIdx.y >= (startBlock + 2) * origGrid.y &&
               blockIdx.y < (endBlock + 3) * origGrid.y) {
      baseGrid = {0, (startBlock + 2) * origGrid.y};
    }
      typename Operator1::SharedStorage *shared_storage_1 =
          reinterpret_cast<typename Operator1::SharedStorage *>(SharedStorageBase);

      Operator1 op1;
      op1(params1, *shared_storage_1, baseGrid, origGrid, Schedule, PresumM1, ReadWriteM1);

      // typename Operator2::SharedStorage *shared_storage_2 =
          // reinterpret_cast<typename Operator2::SharedStorage *>(SharedStorageBase);

      // Operator2 op2;
      // auto params2 = typename Operator2::Params(params1);
      // op2(params2, *shared_storage_2, baseGrid, origGrid, Schedule, PresumM2, ReadWriteM2);

      // typename Operator3::SharedStorage *shared_storage_3 =
      //     reinterpret_cast<typename Operator3::SharedStorage *>(SharedStorageBase);

      // Operator3 op3;
      // auto params3 = typename Operator3::Params(params1);
      // op3(params3, *shared_storage_3, baseGrid, origGrid, Schedule, PresumM3, ReadWriteM3);
  }
};

template<typename kOperator1, typename kOperator2, typename kOperator3, typename kOperator4,
         uint64_t Schedule = 0,
         uint64_t PresumM1 = 0, uint64_t PresumM2 = 0, uint PresumM3 = 0, uint PresumM4 = 0,
         uint64_t ReadWriteM1 = 0, uint64_t ReadWriteM2 = 0, uint64_t ReadWriteM3 = 0, uint64_t ReadWriteM4 = 0>
class FusedOperator4 {
  public:
  using Operator1 = kOperator1;
  using Operator2 = kOperator2;
  using Operator3 = kOperator3;
  using Operator4 = kOperator4;

  CUTLASS_DEVICE
  void operator()(typename Operator1::Params params1,
                  int* SharedStorageBase,
                  int startBlock, int endBlock,
                  dim3 origGrid) {
    if (blockIdx.z >= startBlock * origGrid.z && blockIdx.z < endBlock * origGrid.z) {
      dim3 baseGrid = {0, 0, startBlock * origGrid.z};
      typename Operator1::SharedStorage *shared_storage_1 =
          reinterpret_cast<typename Operator1::SharedStorage *>(SharedStorageBase);

      Operator1 op1;
      op1(params1, *shared_storage_1, baseGrid, origGrid, Schedule, PresumM1, ReadWriteM1);

      typename Operator2::SharedStorage *shared_storage_2 =
          reinterpret_cast<typename Operator2::SharedStorage *>(SharedStorageBase);
      
      auto params2 = Operator2::Params(params1);
      Operator2 op2;
      op2(params2, *shared_storage_2, baseGrid, origGrid, Schedule, PresumM2, ReadWriteM2);

      typename Operator3::SharedStorage *shared_storage_3 =
          reinterpret_cast<typename Operator3::SharedStorage *>(SharedStorageBase);
      auto params3 = Operator3::Params(params1);
      Operator3 op3;
      op3(params3, *shared_storage_3, baseGrid, origGrid, Schedule, PresumM3, ReadWriteM3);

      typename Operator4::SharedStorage *shared_storage_4 =
          reinterpret_cast<typename Operator4::SharedStorage *>(SharedStorageBase);
      auto params4 = Operator4::Params(params1);
      Operator4 op4;
      op4(params4, *shared_storage_4, baseGrid, origGrid, Schedule, PresumM4, ReadWriteM4);
    }
  }
};

template<typename kOperator1, typename kOperator2, typename kOperator3, typename kOperator4, typename kOperator5,
         uint64_t Schedule,
         uint64_t PresumM1, uint64_t PresumM2, uint PresumM3, uint PresumM4, uint PresumM5,
         uint64_t ReadWriteM1, uint64_t ReadWriteM2, uint64_t ReadWriteM3, uint64_t ReadWriteM4, uint64_t ReadWriteM5>
class FusedOperator5 {
  public:
  using Operator1 = kOperator1;
  using Operator2 = kOperator2;
  using Operator3 = kOperator3;
  using Operator4 = kOperator4;
  using Operator5 = kOperator5;

  CUTLASS_DEVICE
  void operator()(typename Operator1::Params& params1,
                  int* SharedStorageBase,
                  int startBlock, int endBlock,
                  dim3 origGrid) {
    if (blockIdx.z >= startBlock * origGrid.z && blockIdx.z < endBlock * origGrid.z) {
      dim3 baseGrid = {0, 0, startBlock * origGrid.z};
      // auto params3 = *reinterpret_cast<typename Operator3::Params*> (&params1);
      // auto params4 = *reinterpret_cast<typename Operator4::Params*> (&params1);
      // auto params5 = *reinterpret_cast<typename Operator5::Params*> (&params1);

      typename Operator1::SharedStorage *shared_storage_1 =
          reinterpret_cast<typename Operator1::SharedStorage *>(SharedStorageBase);

      Operator1 op1;
      op1(params1, *shared_storage_1, baseGrid, origGrid, Schedule, PresumM1, ReadWriteM1);

      typename Operator2::SharedStorage *shared_storage_2 =
          reinterpret_cast<typename Operator2::SharedStorage *>(SharedStorageBase);

      auto params2 = Operator2::Params(params1);
      Operator2 op2;
      op2(params2, *shared_storage_2, baseGrid, origGrid, Schedule, PresumM2, ReadWriteM2);

      typename Operator3::SharedStorage *shared_storage_3 =
          reinterpret_cast<typename Operator3::SharedStorage *>(SharedStorageBase);
      auto params3 = Operator3::Params(params1);
      Operator3 op3;
      op3(params3, *shared_storage_3, baseGrid, origGrid, Schedule, PresumM3, ReadWriteM3);

      typename Operator4::SharedStorage *shared_storage_4 =
          reinterpret_cast<typename Operator4::SharedStorage *>(SharedStorageBase);
      auto params4 = Operator4::Params(params1);
      Operator4 op4;
      op4(params4, *shared_storage_4, baseGrid, origGrid, Schedule, PresumM4, ReadWriteM4);

      typename Operator5::SharedStorage *shared_storage_5 =
          reinterpret_cast<typename Operator5::SharedStorage *>(SharedStorageBase);
      auto params5 = Operator5::Params(params1);
      Operator5 op5;
      op5(params5, *shared_storage_5, baseGrid, origGrid, Schedule, PresumM5, ReadWriteM5);
    }
  }
};

template <typename ParallelMiKernels>
CUTLASS_GLOBAL
__launch_bounds__(ParallelMiKernels::ThreadCount(), 1)
void KernelParallelMiGroup(CUTLASS_GRID_CONSTANT ParallelMiKernels const parallel_kernels, dim3 origGrid) {
  // Dynamic shared memory base pointer
  extern __shared__ int SharedStorageBase[];
  
  parallel_kernels(&SharedStorageBase[0], origGrid);
}

template <typename FusedOperator>
CUTLASS_GLOBAL
__launch_bounds__(FusedOperator::Operator1::kThreadCount, 1)
void Kernel1(typename FusedOperator::Operator1::Params params1, dim3 origGrid) {
  // Dynamic shared memory base pointer
  extern __shared__ int SharedStorageBase[];
  
  FusedOperator()(params1, &SharedStorageBase[0], 0, 1, origGrid);
}

template <typename FusedOperator1, typename FusedOperator2>
CUTLASS_GLOBAL
__launch_bounds__(FusedOperator1::Operator1::kThreadCount, 1)
void Kernel2(typename FusedOperator1::Operator1::Params params1,
             typename FusedOperator2::Operator1::Params params2,
             dim3 origGrid) {
  // Dynamic shared memory base pointer
  extern __shared__ int SharedStorageBase[];
  
  FusedOperator1()(params1, &SharedStorageBase[0], 0, 1, origGrid);
  FusedOperator2()(params2, &SharedStorageBase[0], 1, 2, origGrid);
}

template <typename FusedOperator1, typename FusedOperator2, typename FusedOperator3>
CUTLASS_GLOBAL
__launch_bounds__(FusedOperator1::Operator1::kThreadCount, 1)
void Kernel3(typename FusedOperator1::Operator1::Params params1,
             typename FusedOperator2::Operator1::Params params2,
             typename FusedOperator3::Operator1::Params params3,
             dim3 origGrid) {
  // Dynamic shared memory base pointer
  extern __shared__ int SharedStorageBase[];
  
  FusedOperator1()(params1, &SharedStorageBase[0], 0, 1, origGrid);
  FusedOperator2()(params2, &SharedStorageBase[0], 1, 2, origGrid);
  FusedOperator3()(params3, &SharedStorageBase[0], 2, 3, origGrid);
}

template <typename FusedOperator1, typename FusedOperator2, typename FusedOperator3, typename FusedOperator4>
CUTLASS_GLOBAL
__launch_bounds__(FusedOperator1::Operator1::kThreadCount, 1)
void Kernel4(typename FusedOperator1::Operator1::Params params1,
             typename FusedOperator2::Operator1::Params params2,
             typename FusedOperator3::Operator1::Params params3,
             typename FusedOperator4::Operator1::Params params4,
             dim3 origGrid) {
  // Dynamic shared memory base pointer
  extern __shared__ int SharedStorageBase[];
  
  FusedOperator1()(params1, &SharedStorageBase[0], 0, 1, origGrid);
  FusedOperator2()(params2, &SharedStorageBase[0], 1, 2, origGrid);
  FusedOperator3()(params3, &SharedStorageBase[0], 2, 3, origGrid);
  FusedOperator4()(params4, &SharedStorageBase[0], 3, 4, origGrid);
}

template <typename FusedOperator1, typename FusedOperator2, typename FusedOperator3, typename FusedOperator4, typename FusedOperator5>
CUTLASS_GLOBAL
__launch_bounds__(FusedOperator1::Operator1::kThreadCount, 1)
void Kernel5(typename FusedOperator1::Operator1::Params params1,
             typename FusedOperator2::Operator1::Params params2,
             typename FusedOperator3::Operator1::Params params3,
             typename FusedOperator4::Operator1::Params params4,
             typename FusedOperator5::Operator1::Params params5,
             dim3 origGrid) {
  // Dynamic shared memory base pointer
  extern __shared__ int SharedStorageBase[];
  
  FusedOperator1()(params1, &SharedStorageBase[0], 0, 1, origGrid);
  FusedOperator2()(params2, &SharedStorageBase[0], 1, 2, origGrid);
  FusedOperator3()(params3, &SharedStorageBase[0], 2, 3, origGrid);
  FusedOperator4()(params4, &SharedStorageBase[0], 3, 4, origGrid);
  FusedOperator5()(params5, &SharedStorageBase[0], 4, 5, origGrid);
}

template <typename FusedOperator1, typename FusedOperator2, typename FusedOperator3, typename FusedOperator4, typename FusedOperator5, typename FusedOperator6, typename FusedOperator7>
CUTLASS_GLOBAL
__launch_bounds__(FusedOperator1::Operator1::kThreadCount, 1)
void Kernel7(typename FusedOperator1::Operator1::Params params1,
             typename FusedOperator2::Operator1::Params params2,
             typename FusedOperator3::Operator1::Params params3,
             typename FusedOperator4::Operator1::Params params4,
             typename FusedOperator5::Operator1::Params params5,
             typename FusedOperator6::Operator1::Params params6,
             typename FusedOperator7::Operator1::Params params7,
             dim3 origGrid) {
  // Dynamic shared memory base pointer
  extern __shared__ int SharedStorageBase[];
  
  FusedOperator1()(params1, &SharedStorageBase[0], 0, 1, origGrid);
  FusedOperator2()(params2, &SharedStorageBase[0], 1, 2, origGrid);
  FusedOperator3()(params3, &SharedStorageBase[0], 2, 3, origGrid);
  FusedOperator4()(params4, &SharedStorageBase[0], 3, 4, origGrid);
  FusedOperator5()(params5, &SharedStorageBase[0], 4, 5, origGrid);
  FusedOperator6()(params6, &SharedStorageBase[0], 5, 6, origGrid);
  FusedOperator7()(params7, &SharedStorageBase[0], 6, 7, origGrid);
}

template <typename Operator1, typename Operator2, uint64_t Schedule,
          uint64_t PresumM1, uint64_t PresumM2, uint64_t ReadWriteM1, uint64_t ReadWriteM2>
CUTLASS_GLOBAL
__launch_bounds__(Operator1::kThreadCount, 1)
void Kernel2x1(typename Operator1::Params params1, typename Operator2::Params params2, dim3 origGrid) {
  // Dynamic shared memory base pointer
  extern __shared__ int SharedStorageBase[];
  dim3 baseGrid = 0;

  if (blockIdx.y < 1 * origGrid.y) {
    baseGrid = {0, 0 * origGrid.y};
    typename Operator1::SharedStorage *shared_storage =
        reinterpret_cast<typename Operator1::SharedStorage *>(SharedStorageBase);

    Operator1 op1;
    op1(params1, *shared_storage,  baseGrid, origGrid, Schedule, PresumM1, ReadWriteM1);
  } else if (blockIdx.y < 2 * origGrid.y) {
    baseGrid = {0, 1 * origGrid.y};
    typename Operator2::SharedStorage *shared_storage_2 =
        reinterpret_cast<typename Operator2::SharedStorage *>(SharedStorageBase);
    Operator2 op2;

    op2(params2, *shared_storage_2, baseGrid, origGrid, Schedule, PresumM2, ReadWriteM2);
  }
}

template <uint SizeRange, typename Operator1, typename Operator2, typename Operator3>
CUTLASS_GLOBAL
__launch_bounds__(Operator1::kThreadCount, 1)
void Kernel3x1(typename Operator1::Params params1, typename Operator2::Params params2, typename Operator3::Params params3, uint perYBlocks) {
  // Dynamic shared memory base pointer
  extern __shared__ int SharedStorageBase[];
  // Declare pointer to dynamic shared memory.
  typename Operator1::SharedStorage *shared_storage =
      reinterpret_cast<typename Operator1::SharedStorage *>(SharedStorageBase);

    
  if (blockIdx.y < perYBlocks) {
    Operator1 op1;
    typename Operator1::Mma::FragmentC srcFragment; srcFragment.clear();

    op1(params1, *shared_storage, 0, SizeRange);
  } else if (blockIdx.y < 2* perYBlocks) {
    typename Operator2::SharedStorage *shared_storage_2 =
        reinterpret_cast<typename Operator2::SharedStorage *>(SharedStorageBase);
    Operator2 op2;

    op2(params2, *shared_storage_2, perYBlocks, SizeRange);
  } else if (blockIdx.y < 3 * perYBlocks) {
    typename Operator3::SharedStorage *shared_storage =
    reinterpret_cast<typename Operator3::SharedStorage *>(SharedStorageBase);
    Operator3 op3;

    op3(params3, *shared_storage, 2*perYBlocks, SizeRange);
  }
}

template <uint SizeRange, typename Operator1, typename Operator2, typename Operator3>
CUTLASS_GLOBAL
__launch_bounds__(Operator1::kThreadCount, 1)
void Kernel3x3(typename Operator1::Params params1, typename Operator2::Params params2, typename Operator3::Params params3, uint perYBlocks) {
  // Dynamic shared memory base pointer
  extern __shared__ int SharedStorageBase[];
  // Declare pointer to dynamic shared memory.
  typename Operator1::SharedStorage *shared_storage =
      reinterpret_cast<typename Operator1::SharedStorage *>(SharedStorageBase);
  
  Operator1 op1;
  typename Operator1::Mma::FragmentC srcFragment; srcFragment.clear();

  op1(params1, *shared_storage, 0, SizeRange);

  typename Operator2::SharedStorage *shared_storage_2 =
      reinterpret_cast<typename Operator2::SharedStorage *>(SharedStorageBase);
  Operator2 op2;

  op2(params2, *shared_storage_2, 0, SizeRange);

  typename Operator3::SharedStorage *shared_storage_3 =
  reinterpret_cast<typename Operator3::SharedStorage *>(SharedStorageBase);
  Operator3 op3;

  op3(params3, *shared_storage_3, 0, SizeRange);
}

template <typename Operator1, typename Operator2, typename Operator3, typename Operator4>
CUTLASS_GLOBAL
__launch_bounds__(Operator1::kThreadCount, 1)
void Kernel4x1(typename Operator1::Params params1, typename Operator2::Params params2, typename Operator3::Params params3, typename Operator4::Params params4,uint perYBlocks) {
  // Dynamic shared memory base pointer
  extern __shared__ int SharedStorageBase[];
  // Declare pointer to dynamic shared memory.
  typename Operator1::SharedStorage *shared_storage =
      reinterpret_cast<typename Operator1::SharedStorage *>(SharedStorageBase);

    
  if (blockIdx.y < perYBlocks) {
    Operator1 op1;
    typename Operator1::Mma::FragmentC srcFragment; srcFragment.clear();

    op1(params1, *shared_storage, 0);
  } else if (blockIdx.y < 2* perYBlocks) {
    typename Operator2::SharedStorage *shared_storage_2 =
        reinterpret_cast<typename Operator2::SharedStorage *>(SharedStorageBase);
    Operator2 op2;

    op2(params2, *shared_storage_2, perYBlocks);
  } else if (blockIdx.y < 3 * perYBlocks) {
    typename Operator3::SharedStorage *shared_storage =
    reinterpret_cast<typename Operator3::SharedStorage *>(SharedStorageBase);
    Operator3 op3;

    op3(params3, *shared_storage, 2*perYBlocks);
  } else if (blockIdx.y < 4 * perYBlocks) {
    typename Operator4::SharedStorage *shared_storage =
    reinterpret_cast<typename Operator4::SharedStorage *>(SharedStorageBase);
    Operator4 op4;

    op4(params4, *shared_storage, 3*perYBlocks);
  }
}

template <uint SizeRange, typename Operator1, typename Operator2, typename Operator3, typename Operator4>
CUTLASS_GLOBAL
__launch_bounds__(Operator1::kThreadCount, 1)
void Kernel4x2(typename Operator1::Params params1, typename Operator2::Params params2, typename Operator3::Params params3, typename Operator4::Params params4,uint perYBlocks) {
  // Dynamic shared memory base pointer
  extern __shared__ int SharedStorageBase[];
  // Declare pointer to dynamic shared memory.
  typename Operator1::SharedStorage *shared_storage =
      reinterpret_cast<typename Operator1::SharedStorage *>(SharedStorageBase);

    
  if (blockIdx.y < perYBlocks) {
    Operator1 op1;
    typename Operator1::Mma::FragmentC srcFragment; srcFragment.clear();

    op1(params1, *shared_storage, 0, SizeRange);
  
    typename Operator2::SharedStorage *shared_storage_2 =
        reinterpret_cast<typename Operator2::SharedStorage *>(SharedStorageBase);
    Operator2 op2;

    op2(params2, *shared_storage_2, 0, SizeRange);
  } else if (blockIdx.y < 2 * perYBlocks) {
    typename Operator3::SharedStorage *shared_storage =
    reinterpret_cast<typename Operator3::SharedStorage *>(SharedStorageBase);
    Operator3 op3;

    op3(params3, *shared_storage, perYBlocks, SizeRange);
  
    typename Operator4::SharedStorage *shared_storage_2 =
    reinterpret_cast<typename Operator4::SharedStorage *>(SharedStorageBase);
    Operator4 op4;

    op4(params4, *shared_storage_2, perYBlocks, SizeRange);
  }
}

template <uint SizeRange, typename Operator1, typename Operator2, typename Operator3, typename Operator4>
CUTLASS_GLOBAL
__launch_bounds__(Operator1::kThreadCount, 1)
void Kernel4x4(typename Operator1::Params params1, typename Operator2::Params params2, typename Operator3::Params params3, typename Operator4::Params params4,uint perYBlocks) {
  // Dynamic shared memory base pointer
  extern __shared__ int SharedStorageBase[];
  // Declare pointer to dynamic shared memory.
  typename Operator1::SharedStorage *shared_storage =
      reinterpret_cast<typename Operator1::SharedStorage *>(SharedStorageBase);
  
  Operator1 op1;
  typename Operator1::Mma::FragmentC srcFragment; srcFragment.clear();

  op1(params1, *shared_storage, 0, SizeRange);

  typename Operator3::SharedStorage *shared_storage_3 =
  reinterpret_cast<typename Operator3::SharedStorage *>(SharedStorageBase);
  Operator3 op3;

  op3(params3, *shared_storage_3, 0, SizeRange);

  typename Operator2::SharedStorage *shared_storage_2 =
      reinterpret_cast<typename Operator2::SharedStorage *>(SharedStorageBase);
  Operator2 op2;

  op2(params2, *shared_storage_2, 0, SizeRange);

  typename Operator4::SharedStorage *shared_storage_4 =
  reinterpret_cast<typename Operator4::SharedStorage *>(SharedStorageBase);
  Operator4 op4;

  op4(params4, *shared_storage_4, 0, SizeRange);
}

template <typename Operator1, typename Operator2, typename Operator3,
          typename Operator4, typename Operator5, uint64_t Schedule,
          uint64_t PresumM1, uint64_t PresumM2,
          uint64_t ReadWriteM1, uint64_t ReadWriteM2, uint64_t ReadWriteM3, uint64_t ReadWriteM4, uint64_t ReadWriteM5>
CUTLASS_GLOBAL
__launch_bounds__(Operator1::kThreadCount, 1)
void Kernel5x1(typename Operator1::Params params1, typename Operator2::Params params2, typename Operator3::Params params3, typename Operator4::Params params4, typename Operator5::Params params5, dim3 origGrid) {
// Dynamic shared memory base pointer
  extern __shared__ int SharedStorageBase[];

  dim3 baseGrid = 0;

  if (blockIdx.y < 1 * origGrid.y) {
    baseGrid = {0, 0 * origGrid.y};
    // Declare pointer to dynamic shared memory.
    typename Operator1::SharedStorage *shared_storage =
        reinterpret_cast<typename Operator1::SharedStorage *>(SharedStorageBase);
    Operator1 op1;
    typename Operator1::Mma::FragmentC srcFragment; srcFragment.clear();

    op1(params1, *shared_storage, baseGrid, origGrid, Schedule, PresumM1, ReadWriteM1);
  } else if (blockIdx.y < 2 * origGrid.y) {
    baseGrid = {0, 1 * origGrid.y};
    typename Operator2::SharedStorage *shared_storage_2 =
        reinterpret_cast<typename Operator2::SharedStorage *>(SharedStorageBase);
    Operator2 op2;

    op2(params2, *shared_storage_2, baseGrid, origGrid, Schedule, PresumM2, ReadWriteM2);
  } else if (blockIdx.y < 3 * origGrid.y) {
    baseGrid = {0, 2 * origGrid.y};
    typename Operator3::SharedStorage *shared_storage =
    reinterpret_cast<typename Operator3::SharedStorage *>(SharedStorageBase);
    Operator3 op3;

    op3(params3, *shared_storage, baseGrid, origGrid, Schedule, 0, ReadWriteM3);
  } else if (blockIdx.y < 4 * origGrid.y) {
    baseGrid = {0, 3 * origGrid.y};
    typename Operator4::SharedStorage *shared_storage =
    reinterpret_cast<typename Operator4::SharedStorage *>(SharedStorageBase);
    Operator4 op4;

    op4(params4, *shared_storage, baseGrid, origGrid, Schedule, 0, ReadWriteM4);
  } else if (blockIdx.y < 5 * origGrid.y) {
    baseGrid = {0, 4 * origGrid.y};
    typename Operator5::SharedStorage *shared_storage_2 =
    reinterpret_cast<typename Operator5::SharedStorage *>(SharedStorageBase);
    Operator5 op5;

    op5(params5, *shared_storage_2, baseGrid, origGrid, Schedule, 0, ReadWriteM5);
  }
}

template <typename Operator1, typename Operator2, typename Operator3,
          typename Operator4, typename Operator5, uint64_t Schedule,
          uint64_t PresumM1, uint64_t PresumM2,
          uint64_t ReadWriteM1, uint64_t ReadWriteM2, uint64_t ReadWriteM3, uint64_t ReadWriteM4, uint64_t ReadWriteM5>
CUTLASS_GLOBAL
__launch_bounds__(Operator1::kThreadCount, 1)
void Kernel5x2(typename Operator1::Params params1, typename Operator2::Params params2, typename Operator3::Params params3, typename Operator4::Params params4, typename Operator5::Params params5, dim3 origGrid) {
  // Dynamic shared memory base pointer
  extern __shared__ int SharedStorageBase[];

  dim3 baseGrid = 0;

  if (blockIdx.y < 1 * origGrid.y) {
    baseGrid = {0, 0 * origGrid.y};
    // Declare pointer to dynamic shared memory.
    typename Operator1::SharedStorage *shared_storage =
        reinterpret_cast<typename Operator1::SharedStorage *>(SharedStorageBase);
    Operator1 op1;
    typename Operator1::Mma::FragmentC srcFragment; srcFragment.clear();

    op1(params1, *shared_storage, baseGrid, origGrid, Schedule, PresumM1, ReadWriteM1);
  } 
  else if (blockIdx.y < 2 * origGrid.y) {
    baseGrid = {0, 1 * origGrid.y};
    typename Operator2::SharedStorage *shared_storage_2 =
        reinterpret_cast<typename Operator2::SharedStorage *>(SharedStorageBase);
    Operator2 op2;

    op2(params2, *shared_storage_2, baseGrid, origGrid, Schedule, PresumM2, ReadWriteM2);
  } else if (blockIdx.y < 3 * origGrid.y) {
    baseGrid = {0, 2 * origGrid.y};
    typename Operator3::SharedStorage *shared_storage =
    reinterpret_cast<typename Operator3::SharedStorage *>(SharedStorageBase);
    Operator3 op3;

    op3(params3, *shared_storage, baseGrid, origGrid, Schedule, 0, ReadWriteM3);
  } else if (blockIdx.y < 4 * origGrid.y) {
    baseGrid = {0, 3 * origGrid.y};
    typename Operator4::SharedStorage *shared_storage =
    reinterpret_cast<typename Operator4::SharedStorage *>(SharedStorageBase);
    Operator4 op4;

    op4(params4, *shared_storage, baseGrid, origGrid, Schedule, 0, ReadWriteM4);
    __syncthreads();
    typename Operator5::SharedStorage *shared_storage_2 =
    reinterpret_cast<typename Operator5::SharedStorage *>(SharedStorageBase);
    Operator5 op5;

    op5(params5, *shared_storage_2, baseGrid, origGrid, Schedule, 0, ReadWriteM5);
  }
}

template <typename Operator0, typename Operator1, typename Operator2,
          typename Operator3, typename Operator4, typename Operator5,
          typename Operator6, uint64_t Schedule, uint64_t PresumM2, uint64_t PresumM3,
          uint64_t ReadWriteM0, uint64_t ReadWriteM1, uint64_t ReadWriteM2, uint64_t ReadWriteM3, uint64_t ReadWriteM4, uint64_t ReadWriteM5, uint64_t ReadWriteM6>
CUTLASS_GLOBAL
__launch_bounds__(Operator0::kThreadCount, 1)
void Kernel7x2(typename Operator0::Params params0, typename Operator1::Params params1, typename Operator2::Params params2, typename Operator3::Params params3, typename Operator4::Params params4, typename Operator5::Params params5, typename Operator6::Params params6, dim3 origGrid) {
  // Dynamic shared memory base pointer
  extern __shared__ int SharedStorageBase[];
  dim3 baseGrid = 0;

  if (blockIdx.y < 1 * origGrid.y) {
    baseGrid = {0, 0 * origGrid.y};
    // Declare pointer to dynamic shared memory.
    typename Operator2::SharedStorage *shared_storage_2 =
      reinterpret_cast<typename Operator2::SharedStorage *>(SharedStorageBase);
    Operator2 op2;
    typename Operator2::Mma::FragmentC srcFragment; srcFragment.clear();

    op2(params2, *shared_storage_2, baseGrid, origGrid, Schedule, PresumM2, ReadWriteM2);
  } else if (blockIdx.y < 2 * origGrid.y) {
    baseGrid = {0, 1 * origGrid.y};
    typename Operator3::SharedStorage *shared_storage_3 =
        reinterpret_cast<typename Operator3::SharedStorage *>(SharedStorageBase);
    Operator3 op3;

    op3(params3, *shared_storage_3, baseGrid, origGrid, Schedule, PresumM3, ReadWriteM3);
  } else if (blockIdx.y < 3 * origGrid.y) {
    baseGrid = {0, 2 * origGrid.y};
    typename Operator0::SharedStorage *shared_storage_0 =
    reinterpret_cast<typename Operator0::SharedStorage *>(SharedStorageBase);
    Operator0 op0;

    op0(params0, *shared_storage_0, baseGrid, origGrid, Schedule, 0, ReadWriteM0);
  } 
  else if (blockIdx.y < 4 * origGrid.y) {
    baseGrid = {0, 3 * origGrid.y};
    typename Operator1::SharedStorage *shared_storage_1 =
    reinterpret_cast<typename Operator1::SharedStorage *>(SharedStorageBase);
    Operator1 op1;

    op1(params1, *shared_storage_1, baseGrid, origGrid, Schedule, 0, ReadWriteM1);

    __syncthreads();
    typename Operator5::SharedStorage *shared_storage_5 =
    reinterpret_cast<typename Operator5::SharedStorage *>(SharedStorageBase);
    Operator5 op5;

    op5(params5, *shared_storage_5, baseGrid, origGrid, Schedule, 0, ReadWriteM5);
  } else if (blockIdx.y < 5 * origGrid.y) {
    baseGrid = {0, 4 * origGrid.y};
    typename Operator4::SharedStorage *shared_storage_4 =
    reinterpret_cast<typename Operator4::SharedStorage *>(SharedStorageBase);
    Operator4 op4;

    op4(params4, *shared_storage_4, baseGrid, origGrid, Schedule, 0, ReadWriteM4);
    __syncthreads();
    typename Operator6::SharedStorage *shared_storage_6 =
    reinterpret_cast<typename Operator6::SharedStorage *>(SharedStorageBase);
    Operator6 op6;

    op6(params6, *shared_storage_6, baseGrid, origGrid, Schedule, 0, ReadWriteM6);
  }
}

template <typename Operator0, typename Operator1, typename Operator2,
          typename Operator3, typename Operator4, typename Operator5,
          typename Operator6, uint PresumM2, uint PresumM3,
          uint ReadWriteM0, uint ReadWriteM1, uint ReadWriteM2, uint ReadWriteM3, uint ReadWriteM4, uint ReadWriteM5, uint ReadWriteM6>
CUTLASS_GLOBAL
__launch_bounds__(Operator0::kThreadCount, 1)
void KernelCompressed(typename Operator0::Params params0, typename Operator1::Params params1, typename Operator2::Params params2, typename Operator3::Params params3, typename Operator4::Params params4, typename Operator5::Params params5, typename Operator6::Params params6, dim3 origGrid) {
  // Dynamic shared memory base pointer
  extern __shared__ int SharedStorageBase[];
  dim3 baseGrid = 0;

  if (blockIdx.y < 1 * origGrid.y) {
    baseGrid = {0, 0 * origGrid.y};
    // Declare pointer to dynamic shared memory.
    typename Operator2::SharedStorage *shared_storage_2 =
      reinterpret_cast<typename Operator2::SharedStorage *>(SharedStorageBase);
    Operator2 op2;
    typename Operator2::Mma::FragmentC srcFragment; srcFragment.clear();

    op2(params2, *shared_storage_2, baseGrid, origGrid, PresumM2, ReadWriteM2);
  } else if (blockIdx.y < 2 * origGrid.y) {
    baseGrid = {0, 1 * origGrid.y};
    typename Operator3::SharedStorage *shared_storage_3 =
        reinterpret_cast<typename Operator3::SharedStorage *>(SharedStorageBase);
    Operator3 op3;

    op3(params3, *shared_storage_3, baseGrid, origGrid, PresumM3, ReadWriteM3);
  } else if (blockIdx.y < 3 * origGrid.y) {
    baseGrid = {0, 2 * origGrid.y};
    typename Operator0::SharedStorage *shared_storage_0 =
    reinterpret_cast<typename Operator0::SharedStorage *>(SharedStorageBase);
    Operator0 op0;

    op0(params0, *shared_storage_0, baseGrid, origGrid, 0, ReadWriteM0);
  } 
  else if (blockIdx.y < 4 * origGrid.y) {
    baseGrid = {0, 3 * origGrid.y};
    typename Operator1::SharedStorage *shared_storage_1 =
    reinterpret_cast<typename Operator1::SharedStorage *>(SharedStorageBase);
    Operator1 op1;

    op1(params1, *shared_storage_1, baseGrid, origGrid, 0, ReadWriteM1);

    __syncthreads();
    typename Operator5::SharedStorage *shared_storage_5 =
    reinterpret_cast<typename Operator5::SharedStorage *>(SharedStorageBase);
    Operator5 op5;

    op5(params5, *shared_storage_5, baseGrid, origGrid, 0, ReadWriteM5);
  } else if (blockIdx.y < 5 * origGrid.y) {
    baseGrid = {0, 4 * origGrid.y};
    typename Operator4::SharedStorage *shared_storage_4 =
    reinterpret_cast<typename Operator4::SharedStorage *>(SharedStorageBase);
    Operator4 op4;

    op4(params4, *shared_storage_4, baseGrid, origGrid, 0, ReadWriteM4);
    __syncthreads();
    typename Operator6::SharedStorage *shared_storage_6 =
    reinterpret_cast<typename Operator6::SharedStorage *>(SharedStorageBase);
    Operator6 op6;

    op6(params6, *shared_storage_6, baseGrid, origGrid, 0, ReadWriteM6);
  }
}

template <typename Operator0, typename Operator1,
          typename Operator3, typename Operator4, typename Operator5,
          typename Operator6, uint PresumM3,
          uint ReadWriteM0, uint ReadWriteM1, uint ReadWriteM3, uint ReadWriteM4, uint ReadWriteM5, uint ReadWriteM6>
CUTLASS_GLOBAL
__launch_bounds__(Operator0::kThreadCount, 1)
void Kernel6x2(typename Operator0::Params params0, typename Operator1::Params params1, typename Operator3::Params params3, typename Operator4::Params params4, typename Operator5::Params params5, typename Operator6::Params params6, dim3 origGrid) {
  // Dynamic shared memory base pointer
  extern __shared__ int SharedStorageBase[];
  dim3 baseGrid = 0;

  if (blockIdx.y <  origGrid.y) {
    baseGrid = {0, 0 * origGrid.y};
    typename Operator3::SharedStorage *shared_storage_3 =
        reinterpret_cast<typename Operator3::SharedStorage *>(SharedStorageBase);
    Operator3 op3;

    op3(params3, *shared_storage_3, baseGrid, origGrid, PresumM3, ReadWriteM3);
  } else if (blockIdx.y < 2 * origGrid.y) {
    baseGrid = {0, 1 * origGrid.y};
    typename Operator0::SharedStorage *shared_storage_0 =
    reinterpret_cast<typename Operator0::SharedStorage *>(SharedStorageBase);
    Operator0 op0;

    op0(params0, *shared_storage_0, baseGrid, origGrid, 0, ReadWriteM0);
  } 
  else if (blockIdx.y < 3 * origGrid.y) {
    baseGrid = {0, 2 * origGrid.y};
    typename Operator1::SharedStorage *shared_storage_1 =
    reinterpret_cast<typename Operator1::SharedStorage *>(SharedStorageBase);
    Operator1 op1;

    op1(params1, *shared_storage_1, baseGrid, origGrid, 0, ReadWriteM1);

    __syncthreads();
    typename Operator5::SharedStorage *shared_storage_5 =
    reinterpret_cast<typename Operator5::SharedStorage *>(SharedStorageBase);
    Operator5 op5;

    op5(params5, *shared_storage_5, baseGrid, origGrid, 0, ReadWriteM5);
  } else if (blockIdx.y < 4 * origGrid.y) {
    baseGrid = {0, 3 * origGrid.y};
    typename Operator4::SharedStorage *shared_storage_4 =
    reinterpret_cast<typename Operator4::SharedStorage *>(SharedStorageBase);
    Operator4 op4;

    op4(params4, *shared_storage_4, baseGrid, origGrid, 0, ReadWriteM4);
    __syncthreads();
    typename Operator6::SharedStorage *shared_storage_6 =
    reinterpret_cast<typename Operator6::SharedStorage *>(SharedStorageBase);
    Operator6 op6;

    op6(params6, *shared_storage_6, baseGrid, origGrid, 0, ReadWriteM6);
  }
}

template <typename Operator0, typename Operator1, typename Operator2,
          typename Operator3, typename Operator4, typename Operator5,
          typename Operator6, uint64_t Schedule,
          uint PresumM0, uint PresumM1,
          uint ReadWriteM0, uint ReadWriteM1, uint ReadWriteM2, uint ReadWriteM3, uint ReadWriteM4, uint ReadWriteM5, uint ReadWriteM6>
CUTLASS_GLOBAL
__launch_bounds__(Operator0::kThreadCount, 1)
void Kernel7(typename Operator0::Params params0, typename Operator1::Params params1, typename Operator2::Params params2, typename Operator3::Params params3, typename Operator4::Params params4, typename Operator5::Params params5, typename Operator6::Params params6, dim3 origGrid) {
  // Dynamic shared memory base pointer
  extern __shared__ int SharedStorageBase[];
  dim3 baseGrid = 0;

  if (blockIdx.y < 1 * origGrid.y) {
    baseGrid = {0, 0 * origGrid.y};
    // Declare pointer to dynamic shared memory.
    typename Operator0::SharedStorage *shared_storage_0 =
      reinterpret_cast<typename Operator0::SharedStorage *>(SharedStorageBase);
    Operator0 op0;
    typename Operator0::Mma::FragmentC srcFragment; srcFragment.clear();

    op0(params0, *shared_storage_0, baseGrid, origGrid, Schedule, PresumM0, ReadWriteM0);
  } else if (blockIdx.y < 2 * origGrid.y) {
    baseGrid = {0, 1 * origGrid.y};
    typename Operator1::SharedStorage *shared_storage_1 =
        reinterpret_cast<typename Operator1::SharedStorage *>(SharedStorageBase);
    Operator1 op1;

    op1(params1, *shared_storage_1, baseGrid, origGrid, Schedule, PresumM1, ReadWriteM1);
  } else if (blockIdx.y < 3 * origGrid.y) {
    baseGrid = {0, 2 * origGrid.y};
    typename Operator2::SharedStorage *shared_storage_2 =
    reinterpret_cast<typename Operator2::SharedStorage *>(SharedStorageBase);
    Operator2 op2;

    op2(params2, *shared_storage_2, baseGrid, origGrid, Schedule, 0, ReadWriteM2);
  } else if (blockIdx.y < 4 * origGrid.y) {
    baseGrid = {0, 3 * origGrid.y};
    typename Operator3::SharedStorage *shared_storage_3 =
    reinterpret_cast<typename Operator3::SharedStorage *>(SharedStorageBase);
    Operator3 op3;

    op3(params3, *shared_storage_3, baseGrid, origGrid, Schedule, 0, ReadWriteM3);
  } else if (blockIdx.y < 5 * origGrid.y) {
    baseGrid = {0, 4 * origGrid.y};
    typename Operator4::SharedStorage *shared_storage_4 =
    reinterpret_cast<typename Operator4::SharedStorage *>(SharedStorageBase);
    Operator4 op4;

    op4(params4, *shared_storage_4, baseGrid, origGrid, Schedule, 0, ReadWriteM4);
  } else if (blockIdx.y < 6 * origGrid.y) {
    baseGrid = {0, 5 * origGrid.y};
    typename Operator5::SharedStorage *shared_storage_5 =
    reinterpret_cast<typename Operator5::SharedStorage *>(SharedStorageBase);
    Operator5 op5;

    op5(params5, *shared_storage_5, baseGrid, origGrid, Schedule, 0, ReadWriteM5);
  } else if (blockIdx.y < 7 * origGrid.y) {
    baseGrid = {0, 6 * origGrid.y};
    typename Operator6::SharedStorage *shared_storage_6 =
    reinterpret_cast<typename Operator6::SharedStorage *>(SharedStorageBase);
    Operator6 op6;

    op6(params6, *shared_storage_6, baseGrid, origGrid, Schedule, 0, ReadWriteM6);
  }
}

template <typename Operator0, typename Operator1, typename Operator2,
          typename Operator3, typename Operator4, typename Operator5,
          typename Operator6, uint64_t Schedule,
          uint64_t PresumM0, uint64_t PresumM1,
          uint64_t ReadWriteM0, uint64_t ReadWriteM1, uint64_t ReadWriteM2, uint64_t ReadWriteM3, uint64_t ReadWriteM4, uint64_t ReadWriteM5, uint64_t ReadWriteM6>
CUTLASS_GLOBAL
__launch_bounds__(Operator0::kThreadCount, 1)
void KernelM2ToM6(typename Operator0::Params params0, typename Operator1::Params params1, typename Operator2::Params params2, typename Operator3::Params params3, typename Operator4::Params params4, typename Operator5::Params params5, typename Operator6::Params params6, dim3 origGrid) {
  // Dynamic shared memory base pointer
  extern __shared__ int SharedStorageBase[];
  dim3 baseGrid = 0;

  if (blockIdx.y < 1 * origGrid.y) {
    baseGrid = {0, 0 * origGrid.y};
    typename Operator2::SharedStorage *shared_storage_2 =
    reinterpret_cast<typename Operator2::SharedStorage *>(SharedStorageBase);
    Operator2 op2;

    op2(params2, *shared_storage_2, baseGrid, origGrid, Schedule, 0, ReadWriteM2);
  // } else if (blockIdx.y < 3 * origGrid.y) {
    baseGrid = {0, 0 * origGrid.y};
    typename Operator3::SharedStorage *shared_storage_3 =
    reinterpret_cast<typename Operator3::SharedStorage *>(SharedStorageBase);
    Operator3 op3;

    op3(params3, *shared_storage_3, baseGrid, origGrid, Schedule, 0, ReadWriteM3);

    baseGrid = {0, 0 * origGrid.y};
    typename Operator6::SharedStorage *shared_storage_6 =
    reinterpret_cast<typename Operator6::SharedStorage *>(SharedStorageBase);
    Operator6 op6;

    op6(params6, *shared_storage_6, baseGrid, origGrid, Schedule, 0, ReadWriteM6);
  // } else if (blockIdx.y < 2 * origGrid.y) {
    baseGrid = {0, 0 * origGrid.y};
    typename Operator4::SharedStorage *shared_storage_4 =
    reinterpret_cast<typename Operator4::SharedStorage *>(SharedStorageBase);
    Operator4 op4;

    op4(params4, *shared_storage_4, baseGrid, origGrid, Schedule, 0, ReadWriteM4);
  // } else if (blockIdx.y < 2 * origGrid.y) {
    baseGrid = {0, 0 * origGrid.y};
    typename Operator5::SharedStorage *shared_storage_5 =
    reinterpret_cast<typename Operator5::SharedStorage *>(SharedStorageBase);
    Operator5 op5;

    op5(params5, *shared_storage_5, baseGrid, origGrid, Schedule, 0, ReadWriteM5);
  }
}

template <typename Operator0, typename Operator1, typename Operator2,
          typename Operator3, typename Operator4, typename Operator5,
          typename Operator6, uint64_t Schedule,
          uint64_t PresumM0, uint64_t PresumM1,
          uint64_t ReadWriteM0, uint64_t ReadWriteM1, uint64_t ReadWriteM2, uint64_t ReadWriteM3, uint64_t ReadWriteM4, uint64_t ReadWriteM5, uint64_t ReadWriteM6>
CUTLASS_GLOBAL
__launch_bounds__(Operator0::kThreadCount, 1)
void KernelFusedM2M3M4M6andM5(typename Operator0::Params params0, typename Operator1::Params params1, typename Operator2::Params params2, typename Operator3::Params params3, typename Operator4::Params params4, typename Operator5::Params params5, typename Operator6::Params params6, dim3 origGrid) {
  // Dynamic shared memory base pointer
  extern __shared__ int SharedStorageBase[];
  dim3 baseGrid = 0;

  if (blockIdx.y < 1 * origGrid.y) {
    baseGrid = {0, 0 * origGrid.y};
    typename Operator2::SharedStorage *shared_storage_2 =
    reinterpret_cast<typename Operator2::SharedStorage *>(SharedStorageBase);
    Operator2 op2;

    op2(params2, *shared_storage_2, baseGrid, origGrid, Schedule, 0, ReadWriteM2);
  // } else if (blockIdx.y < 3 * origGrid.y) {
    baseGrid = {0, 0 * origGrid.y};
    typename Operator3::SharedStorage *shared_storage_3 =
    reinterpret_cast<typename Operator3::SharedStorage *>(SharedStorageBase);
    Operator3 op3;

    op3(params3, *shared_storage_3, baseGrid, origGrid, Schedule, 0, ReadWriteM3);

    baseGrid = {0, 0 * origGrid.y};
    typename Operator6::SharedStorage *shared_storage_6 =
    reinterpret_cast<typename Operator6::SharedStorage *>(SharedStorageBase);
    Operator6 op6;

    op6(params6, *shared_storage_6, baseGrid, origGrid, Schedule, 0, ReadWriteM6);
  // } else if (blockIdx.y < 2 * origGrid.y) {
    baseGrid = {0, 0 * origGrid.y};
    typename Operator4::SharedStorage *shared_storage_4 =
    reinterpret_cast<typename Operator4::SharedStorage *>(SharedStorageBase);
    Operator4 op4;

    op4(params4, *shared_storage_4, baseGrid, origGrid, Schedule, 0, ReadWriteM4);
  } else if (blockIdx.y < 2 * origGrid.y) {
    baseGrid = {0, 1 * origGrid.y};
    typename Operator5::SharedStorage *shared_storage_5 =
    reinterpret_cast<typename Operator5::SharedStorage *>(SharedStorageBase);
    Operator5 op5;

    op5(params5, *shared_storage_5, baseGrid, origGrid, Schedule, 0, ReadWriteM5);
  }
}

template <uint SizeRange, typename Operator1, typename Operator2, typename Operator3, typename Operator4>
CUTLASS_GLOBAL
__launch_bounds__(Operator1::kThreadCount, 1)
void Kernel2x2(typename Operator1::Params params1, typename Operator2::Params params2,
               typename Operator3::Params params3, typename Operator4::Params params4,
               uint perYBlocks) {
  // Dynamic shared memory base pointer
  extern __shared__ int SharedStorageBase[];

  if (blockIdx.y < perYBlocks) {
    // Declare pointer to dynamic shared memory.
    typename Operator1::SharedStorage *shared_storage =
        reinterpret_cast<typename Operator1::SharedStorage *>(SharedStorageBase);

    Operator1 op1;
    op1(params1, *shared_storage, 0, SizeRange);

    typename Operator2::SharedStorage *shared_storage_2 =
        reinterpret_cast<typename Operator2::SharedStorage *>(SharedStorageBase);
    Operator2 op2;

    op2(params2, *shared_storage_2, 0, SizeRange);
  } else if (blockIdx.y < 2*perYBlocks) {
    // Declare pointer to dynamic shared memory.
    typename Operator3::SharedStorage *shared_storage =
        reinterpret_cast<typename Operator3::SharedStorage *>(SharedStorageBase);

    Operator3 op1;
    op1(params3, *shared_storage, perYBlocks, SizeRange);

    typename Operator4::SharedStorage *shared_storage_2 =
        reinterpret_cast<typename Operator4::SharedStorage *>(SharedStorageBase);
    Operator4 op2;

    op2(params4, *shared_storage_2, perYBlocks, SizeRange);
  }
}

// /// Generic CUTLASS kernel template.
// template <typename Operator>
// CUTLASS_GLOBAL
// void Kernel2(typename Operator::Params params) {
//   // Dynamic shared memory base pointer
//   extern __shared__ int SharedStorageBase[];
//   // Declare pointer to dynamic shared memory.
//   typename Operator::SharedStorage *shared_storage =
//       reinterpret_cast<typename Operator::SharedStorage *>(SharedStorageBase);

//   Operator::invoke(params, *shared_storage);
// }


////////////////////////////////////////////////////////////////////////////////
//
// 3.0 specific launch
//
////////////////////////////////////////////////////////////////////////////////

/// Generic CUTLASS kernel template.
template <typename Operator1, typename Operator2>
CUTLASS_GLOBAL
#ifdef __CUDACC__
// Enclosing this in __CUDACC__ suppresses MSVC warnings.
__launch_bounds__(Operator1::MaxThreadsPerBlock, Operator1::MinBlocksPerMultiprocessor)
#endif // __CUDACC__
void device_kernel(CUTLASS_GRID_CONSTANT typename Operator1::Params const params, CUTLASS_GRID_CONSTANT typename Operator2::Params const params2)
{
  // Dynamic shared memory base pointer
  extern __shared__ char smem[];
  Operator1 op1;
  op1(params, smem);
  __syncthreads();
  Operator2 op2;
  op2(params, smem);
}

template <typename Operator0, typename Operator1, typename Operator2>
CUTLASS_GLOBAL
#ifdef __CUDACC__
// Enclosing this in __CUDACC__ suppresses MSVC warnings.
__launch_bounds__(Operator0::MaxThreadsPerBlock, Operator0::MinBlocksPerMultiprocessor)
#endif // __CUDACC__
void Kernel3(CUTLASS_GRID_CONSTANT typename Operator0::Params const params0,
       CUTLASS_GRID_CONSTANT typename Operator1::Params const params1,
       CUTLASS_GRID_CONSTANT typename Operator2::Params const params2)
{
  // Dynamic shared memory base pointer
  extern __shared__ char smem[];
  if (blockIdx.z == 0) {
  Operator0 op0;
  op0(params0, smem);
  } else if (blockIdx.z == 1) {
  Operator1 op1;
  op1(params1, smem);
  } else if (blockIdx.z == 2) {
  Operator2 op2;
  op2(params2, smem);
  }
  // cutlass::arch::synclog_print();
}

////////////////////////////////////////////////////////////////////////////////
} /// namespace cutlass
