#include "cutlass/gemm/device/strassen_gemm.h"
#include "cutlass/cutlass.h"

using EpilogueOp = cutlass::epilogue::thread::StrassenLinearCombination<
    double,
    1,
    double,
    double>;

using InterimEpilogueOp = cutlass::epilogue::thread::StrassenLinearCombination<
    double,
    4,
    double,
    double>;

using RowMajor = cutlass::layout::RowMajor;

using ThreadBlockShape = cutlass::gemm::GemmShape<64, 64, 16>;
using WarpShape = cutlass::gemm::GemmShape<32, 32, 16>;
using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
#ifndef SPLIT_K
    #define SPLIT_K 0
#endif

const uint NumStages = 4;
const bool splitK = SPLIT_K;
const auto StrassenKind = StrassenType::StrassenWinograd;

using namespace MmaStrassen;

using AllPresumsKernel = AllPresums<>;
using AllPresumsM0    = AllPresums<PresumCompute, PresumCompute, PresumCompute, PresumCompute,
                                                                        PresumCompute, PresumCompute, PresumCompute, PresumCompute>;
using AllPresumsM1To6 = AllPresums<PresumAvailable, PresumAvailable, PresumAvailable, PresumAvailable,
                                                                        PresumAvailable, PresumAvailable, PresumAvailable, PresumAvailable>;

using StrassenGroupsM0 = StrassenLevel1Groups<StrassenPresum<2, 0, ThreadBlockShape,
                                                                                                                        AllPresumsKernel>,
                                                                                            StrassenLevel1M0Group<2, 0, ThreadBlockShape, WarpShape, 3,
                                                                                                                                    RWMTypes<KeepAccums>,
                                                                                                                                    RWCTypes<CUW<0, LayoutInterim1D, LayoutNone, Expr<Plus<0>>>>,
                                                                                                                                    AllPresumsM0>
                                                                                                                                    >;

template<int level_1_idx>
using StrassenGroups = StrassenLevel1Groups<StrassenPresum<1, level_1_idx, ThreadBlockShape,
                                                                                                                            AllPresumsKernel>,
                                                                                            StrassenLevel1M0Group<1, level_1_idx, ThreadBlockShape, WarpShape, 3,
                                                                                                                                        RWMTypes<KeepAccums>,
                                                                                                                                        RWCTypes<CUW<0, LayoutInterim1D, LayoutNone, Expr<Plus<0>>>>,
                                                                                                                                        AllPresumsM0>,
                                                                                            StrassenLevel1M1Group<1, level_1_idx, ThreadBlockShape, WarpShape, 3,
                                                                                                                                        RWMTypes<ContinueAccums>,
                                                                                                                                        RWCTypes<CUW<0, LayoutFinal, LayoutNone, Expr<Plus<1>>>>,
                                                                                                                                        AllPresumsM1To6>,
                                                                                            StrassenLevel1M2Group<1, level_1_idx, ThreadBlockShape, WarpShape, 3,
                                                                                                                                        RWMTypes<>,
                                                                                                                                        RWCTypes<CUW<1, LayoutInterim1D, LayoutNone, Expr<Plus<2>>, Expr<Plus<0, MemGlobal, LayoutInterim1D>>>>,
                                                                                                                                        AllPresumsM1To6>,
                                                                                            StrassenLevel1M3Group<1, level_1_idx, ThreadBlockShape, WarpShape, 3,
                                                                                                                                        RWMTypes<>,
                                                                                                                                        RWCTypes<CUW<2, LayoutInterim1D, LayoutNone, Expr<Plus<3>>, Expr<Plus<1, MemGlobal, LayoutInterim1D>>>>,
                                                                                                                                        AllPresumsM1To6>,
                                                                                            StrassenLevel1M4Group<1, level_1_idx, ThreadBlockShape, WarpShape, 3,
                                                                                                                                        RWMTypes<>,
                                                                                                                                        RWCTypes<CUW<3, LayoutInterim1D, LayoutNone, Expr<Plus<4>>>,
                                                                                                                                                         CUW<3, LayoutFinal, LayoutNone, Expr<Plus<4>>, Expr<Plus<2, MemGlobal, LayoutInterim1D>>>>,
                                                                                                                                        AllPresumsM1To6>,
                                                                                            StrassenLevel1M5Group<1, level_1_idx, ThreadBlockShape, WarpShape, 3,
                                                                                                                                        RWMTypes<>,
                                                                                                                                        RWCTypes<CUW<1, LayoutFinal, LayoutNone, Expr<Plus<5>>, Expr<Plus<1, MemGlobal, LayoutInterim1D>, Plus<3, MemGlobal, LayoutInterim1D>>>>,
                                                                                                                                        AllPresumsM1To6>,
                                                                                            StrassenLevel1M6Group<1, level_1_idx, ThreadBlockShape, WarpShape, 3,
                                                                                                                                        RWMTypes<>,
                                                                                                                                        RWCTypes<CUW<2, LayoutFinal, LayoutNone, Expr<Neg<6>>, Expr<Plus<2, MemGlobal, LayoutInterim1D>>>>,
                                                                                                                                        AllPresumsM1To6>
                                                                                            >;
using ScheduleStrassenGroups1 = ScheduleStrassenGroups<ParallelMiGroups<true, FusedMiGroup<7, 0, 1>>,
                                                                                                                ParallelMiGroups<true, FusedMiGroup<7, 2>>,
                                                                                                                ParallelMiGroups<true, FusedMiGroup<7, 3>>,
                                                                                                                ParallelMiGroups<true, FusedMiGroup<7, 4>>,
                                                                                                                ParallelMiGroups<true, FusedMiGroup<7, 5>>,
                                                                                                                ParallelMiGroups<true, FusedMiGroup<7, 6>>
                                                                                                                >;

using AmpereF64SWInterleavedPresumLevel2_64x64Kernel = cutlass::gemm::device::StrassenGemmLevel2<StrassenKind,
                                                                                                                StrassenGroupsM0, StrassenGroups<1>, StrassenGroups<2>,
                                                                                                                StrassenGroups<3>, StrassenGroups<4>, StrassenGroups<5>,
                                                                                                                StrassenGroups<6>,
                                                                                                                ScheduleStrassenGroups1,
                                                                                                                double,
                                                                                                                RowMajor,
                                                                                                                double,
                                                                                                                RowMajor,
                                                                                                                double,
                                                                                                                RowMajor,
                                                                                                                double,
                                                                                                                cutlass::arch::OpClassTensorOp,
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

class AmpereF64SWInterleavedPresumLevel2_64x64 {
public:
    using StrassenGemmKernel = AmpereF64SWInterleavedPresumLevel2_64x64Kernel;
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