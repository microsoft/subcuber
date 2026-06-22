#include "cutlass/gemm/device/strassen_gemm.h"
#include "cutlass/cutlass.h"

using EpilogueOp = cutlass::epilogue::thread::StrassenLinearCombination<
    double,
    1,
    double,
    double>;

using InterimEpilogueOp = cutlass::epilogue::thread::StrassenLinearCombination<
    double,
    2,
    double,
    double>;

using RowMajor = cutlass::layout::RowMajor;

using ThreadBlockShape = cutlass::gemm::GemmShape<128, 128, 16>;
using WarpShape = cutlass::gemm::GemmShape<32, 64, 16>;
using InstructionShape = cutlass::gemm::GemmShape<8, 8, 4>;
#ifndef SPLIT_K
    #define SPLIT_K 0
#endif

const uint NumStages = 3;
const bool splitK = SPLIT_K;
const uint kStrassenLevel = 1;
const auto StrassenKind = StrassenType::StrassenWinograd;

using namespace MmaStrassen;

using AllPresumsKernel = AllPresums<>;
using AllPresumsM0    = AllPresums<PresumCompute, PresumCompute, PresumCompute, PresumCompute,
                                                                        PresumCompute, PresumCompute, PresumCompute, PresumCompute>;
using AllPresumsM1To6 = AllPresums<PresumAvailable, PresumAvailable, PresumAvailable, PresumAvailable,
                                                                        PresumAvailable, PresumAvailable, PresumAvailable, PresumAvailable>;

using StrassenGroups = StrassenLevel1Groups<StrassenPresum<kStrassenLevel, 0, ThreadBlockShape,
                                                                                                                            AllPresumsKernel>,
                                                                                            StrassenLevel1M0Group<kStrassenLevel, 0, ThreadBlockShape, WarpShape, 3,
                                                                                                                                        RWMTypes<KeepAccums>,
                                                                                                                                        RWCTypes<CUW<0, LayoutInterim1D, LayoutNone, Expr<Plus<0>>>>,
                                                                                                                                        AllPresumsM0>,
                                                                                            StrassenLevel1M1Group<kStrassenLevel, 0, ThreadBlockShape, WarpShape, 3,
                                                                                                                                        RWMTypes<ContinueAccums>,
                                                                                                                                        RWCTypes<CUW<0, LayoutFinal, LayoutNone, Expr<Plus<1>>>>,
                                                                                                                                        AllPresumsM1To6>,
                                                                                            StrassenLevel1M2Group<kStrassenLevel, 0, ThreadBlockShape, WarpShape, 3,
                                                                                                                                        RWMTypes<>,
                                                                                                                                        RWCTypes<CUW<1, LayoutInterim1D, LayoutNone, Expr<Plus<2>>, Expr<Plus<0, MemGlobal, LayoutInterim1D>>>>,
                                                                                                                                        AllPresumsM1To6>,
                                                                                            StrassenLevel1M3Group<kStrassenLevel, 0, ThreadBlockShape, WarpShape, 3,
                                                                                                                                        RWMTypes<>,
                                                                                                                                        RWCTypes<CUW<2, LayoutInterim1D, LayoutNone, Expr<Plus<3>>, Expr<Plus<1, MemGlobal, LayoutInterim1D>>>>,
                                                                                                                                        AllPresumsM1To6>,
                                                                                            StrassenLevel1M4Group<kStrassenLevel, 0, ThreadBlockShape, WarpShape, 3,
                                                                                                                                        RWMTypes<>,
                                                                                                                                        RWCTypes<CUW<3, LayoutInterim1D, LayoutNone, Expr<Plus<4>>>,
                                                                                                                                                         CUW<3, LayoutFinal, LayoutNone, Expr<Plus<4>>, Expr<Plus<2, MemGlobal, LayoutInterim1D>>>>,
                                                                                                                                        AllPresumsM1To6>,
                                                                                            StrassenLevel1M5Group<kStrassenLevel, 0, ThreadBlockShape, WarpShape, 3,
                                                                                                                                        RWMTypes<>,
                                                                                                                                        RWCTypes<CUW<1, LayoutFinal, LayoutNone, Expr<Plus<5>>, Expr<Plus<1, MemGlobal, LayoutInterim1D>, Plus<3, MemGlobal, LayoutInterim1D>>>>,
                                                                                                                                        AllPresumsM1To6>,
                                                                                            StrassenLevel1M6Group<kStrassenLevel, 0, ThreadBlockShape, WarpShape, 3,
                                                                                                                                        RWMTypes<>,
                                                                                                                                        RWCTypes<CUW<2, LayoutFinal, LayoutNone, Expr<Neg<6>>, Expr<Plus<2, MemGlobal, LayoutInterim1D>>>>,
                                                                                                                                        AllPresumsM1To6>
                                                                                            >;
  using ScheduleStrassenGroups1 = ScheduleStrassenGroups<ParallelMiGroups<true, FusedMiGroup<7, 0, 1>>,
                                                         ParallelMiGroups<true, FusedMiGroup<7, 2>,
                                                                                FusedMiGroup<7, 3>,
                                                                                FusedMiGroup<7, 4>,
                                                                                FusedMiGroup<7, 5>,
                                                                                FusedMiGroup<7, 6>>
                                                         >;

using AmpereF64SWInterleavedPresumLevel1_128x128Kernel = cutlass::gemm::device::StrassenGemm<StrassenKind, StrassenGroups, ScheduleStrassenGroups1,
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

class AmpereF64SWInterleavedPresumLevel1_128x128 {
public:
    using StrassenGemmKernel = AmpereF64SWInterleavedPresumLevel1_128x128Kernel;
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