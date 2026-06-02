#include "cutlass/gemm/device/strassen_gemm.h"
#include "cutlass/cutlass.h"

using EpilogueOp = cutlass::epilogue::thread::StrassenLinearCombination<
  float,                                     // <- data type of output matrix
  1,                                         // <- This is the number of elements per
                                             // vectorized memory access. For half
                                             // precision, it's 8 elements. This becomes
                                             // the vector width of math instructions in
                                             // epilogue too
  float,                                // <- data type of accumulator
  float>;  // <- data type for alpha/beta in linear combination function

using InterimEpilogueOp = cutlass::epilogue::thread::StrassenLinearCombination<
  float,                                     // <- data type of output matrix
  4,                                         // <- This is the number of elements per
                                             // vectorized memory access. For half
                                             // precision, it's 8 elements. This becomes
                                             // the vector width of math instructions in
                                             // epilogue too
  float,                                // <- data type of accumulator
  float>;  // <- data type for alpha/beta in linear combination function

using ColumnMajor = cutlass::layout::ColumnMajor;
using RowMajor = cutlass::layout::RowMajor;

using ThreadBlockShape = cutlass::gemm::GemmShape<256, 128, 8>;
using WarpShape = cutlass::gemm::GemmShape<64, 64, 8>;
using InstructionShape = cutlass::gemm::GemmShape<1,1,1>;
#ifndef SPLIT_K
  #define SPLIT_K 0
#endif

const uint NumStages = 3;

const bool splitK = SPLIT_K;
const auto StrassenKind = StrassenType::StrassenWinograd;

using AllPresumsKernel = AllPresums<>;
using AllPresumsM0    = AllPresums<PresumCompute, PresumCompute, PresumCompute, PresumCompute,
                                    PresumCompute, PresumCompute, PresumCompute, PresumCompute>;
// using AllPresumsM0    = AllPresums<PresumGlobalKernel,   PresumGlobalKernel,   PresumGlobalKernel,   PresumGlobalKernel,   PresumGlobalKernel,    PresumGlobalKernel,  PresumGlobalKernel,    PresumGlobalKernel>;
using AllPresumsM1To6 = AllPresums<PresumAvailable, PresumAvailable, PresumAvailable, PresumAvailable, PresumAvailable, PresumAvailable, PresumAvailable, PresumAvailable>;

using StrassenGroupsM0 = StrassenLevel1Groups<StrassenPresum<2, 0, ThreadBlockShape,
                                                            AllPresumsKernel>,
                                              StrassenLevel1M0Group<2, 0, ThreadBlockShape, WarpShape, 3,
                                                                  RWMTypes<KeepAccums>,
                                                                  RWCTypes<CUW<0, LayoutInterim1D, LayoutNone, Expr<Plus<0>>>>,//C1 = M0
                                                                  AllPresumsM0>
                                                                  >;

template<int level_1_idx>
using StrassenGroups = StrassenLevel1Groups<StrassenPresum<1, level_1_idx, ThreadBlockShape,
                                                              AllPresumsKernel>,
                                              StrassenLevel1M0Group<1, level_1_idx, ThreadBlockShape, WarpShape, 3,
                                                                    RWMTypes<KeepAccums>,
                                                                    RWCTypes<CUW<0, LayoutInterim1D, LayoutNone, Expr<Plus<0>>>>,//M0 = M0
                                                                    AllPresumsM0>,
                                              StrassenLevel1M1Group<1, level_1_idx, ThreadBlockShape, WarpShape, 3,
                                                                    RWMTypes<ContinueAccums>,
                                                                    RWCTypes<CUW<0, LayoutFinal,   LayoutNone, Expr<Plus<1>>>>,//C0 = M0(reg)+M1
                                                                    AllPresumsM1To6>,
                                              StrassenLevel1M2Group<1, level_1_idx, ThreadBlockShape, WarpShape, 3,
                                                                    RWMTypes<>,
                                                                    // RWCTypes<CUW<2, LayoutFinal, LayoutNone, Expr<Plus<2>>>>,
                                                                    RWCTypes<CUW<1, LayoutInterim1D, LayoutNone, Expr<Plus<2>>, Expr<Plus<0, MemGlobal, LayoutInterim1D>>>>,//M2(1) = M0+M2 ; Reg = M2(1)
                                                                    AllPresumsM1To6>,
                                              StrassenLevel1M3Group<1, level_1_idx, ThreadBlockShape, WarpShape, 3,
                                                                    RWMTypes<>,
                                                                    // RWCTypes<CUW<3, LayoutFinal, LayoutNone, Expr<Plus<3>>>>,
                                                                    RWCTypes<CUW<2, LayoutInterim1D, LayoutNone, Expr<Plus<3>>, Expr<Plus<1, MemGlobal, LayoutInterim1D>>>>,//M3(2) = M2(1)+M3
                                                                    AllPresumsM1To6>,
                                              StrassenLevel1M4Group<1, level_1_idx, ThreadBlockShape, WarpShape, 3,
                                                                    RWMTypes<>,
                                                                    // RWCTypes<CUW<0, LayoutFinal, LayoutNone, Expr<Plus<4>>>>,
                                                                    RWCTypes<CUW<3, LayoutInterim1D, LayoutNone, Expr<Plus<4>>>, //M4(3) = M4
                                                                             CUW<3, LayoutFinal,   LayoutNone, Expr<Plus<4>>, Expr<Plus<2, MemGlobal, LayoutInterim1D>>>>,//C3 = M3(2)+M4
                                                                    AllPresumsM1To6>,
                                              StrassenLevel1M5Group<1, level_1_idx, ThreadBlockShape, WarpShape, 3,
                                                                    RWMTypes<>,
                                                                    // RWCTypes<CUW<1, LayoutFinal, LayoutNone, Expr<Plus<5>>>>,
                                                                    RWCTypes<CUW<1, LayoutFinal, LayoutNone, Expr<Plus<5>>, Expr<Plus<1, MemGlobal, LayoutInterim1D>,   //C1 = M2(1)+M4(3)+M5
                                                                                                                                 Plus<3, MemGlobal, LayoutInterim1D>>>>,
                                                                    AllPresumsM1To6>,
                                              StrassenLevel1M6Group<1, level_1_idx, ThreadBlockShape, WarpShape, 3,
                                                                    RWMTypes<>,
                                                                    // RWCTypes<CUW<2, LayoutFinal, LayoutNone, Expr<Plus<6>>>>,
                                                                    RWCTypes<CUW<2, LayoutFinal, LayoutNone, Expr<Neg<6>>, Expr<Plus<2, MemGlobal, LayoutInterim1D>>>>, //C2 = M3(2)-M6
                                                                    AllPresumsM1To6>
                                              >;
using ScheduleStrassenGroups1 = ScheduleStrassenGroups<ParallelMiGroups<true, FusedMiGroup<7, 0, 1>>,
                                                        ParallelMiGroups<true, FusedMiGroup<7, 2>>,
                                                        ParallelMiGroups<true, FusedMiGroup<7, 3>>,
                                                        ParallelMiGroups<true, FusedMiGroup<7, 4>>,
                                                        ParallelMiGroups<true, FusedMiGroup<7, 5>>,
                                                        ParallelMiGroups<true, FusedMiGroup<7, 6>>
                                                        >;

using AmpereF32SWInterleavedPresumLevel2Kernel = cutlass::gemm::device::StrassenGemmLevel2<StrassenKind,
                                                        StrassenGroupsM0, StrassenGroups<1>, StrassenGroups<2>,
                                                        StrassenGroups<3>, StrassenGroups<4>, StrassenGroups<5>,
                                                        StrassenGroups<6>,
                                                        ScheduleStrassenGroups1,
                                                        float,        // Data-type of A matrix
                                                        RowMajor,  // Layout of A matrix
                                                        float,        // Data-type of B matrix
                                                        RowMajor,  // Layout of B matrix
                                                        float,        // Data-type of C matrix
                                                        RowMajor,
                                                        float,
                                                        cutlass::arch::OpClassSimt,
                                                        cutlass::arch::Sm80,
                                                        ThreadBlockShape,
                                                        WarpShape,
                                                        InstructionShape,
                                                        EpilogueOp,
                                                        InterimEpilogueOp,
                                                        cutlass::gemm::threadblock::StrassenGemmIdentityThreadblockSwizzle<8>,
                                                        NumStages,
                                                        1, 1, splitK,
                                                        true>;

class AmpereF32SWInterleavedPresumLevel2 {
public:
  using StrassenGemmKernel = AmpereF32SWInterleavedPresumLevel2Kernel;
  using Arguments = typename StrassenGemmKernel::Arguments;

  static cutlass::Status can_implement(Arguments const &args);
  static size_t get_workspace_size(Arguments const &args);

  cutlass::Status init(Arguments const &args, void *workspace = nullptr, cudaStream_t stream = nullptr);
  cutlass::Status launch(cudaStream_t *streams = nullptr, int num_streams = 0);

  cutlass::Status initialize(Arguments const &args, void *workspace = nullptr, cudaStream_t stream = nullptr);
  cutlass::Status run(cudaStream_t *streams = nullptr, int num_streams = 0);
  cutlass::Status operator()(Arguments const &args, void *workspace = nullptr,
                             cudaStream_t *streams = nullptr, int num_streams = 0);

private:
  StrassenGemmKernel gemm_;
};
