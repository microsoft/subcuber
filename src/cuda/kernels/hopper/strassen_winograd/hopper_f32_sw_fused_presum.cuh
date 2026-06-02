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
const bool splitK = false;
const uint kStrassenLevel = 1;
const auto StrassenKind = StrassenType::StrassenWinograd;
const uint NumStages = 3;

//[m0], [m1], [m5, m4, m3, m2], [m6]
// using StrassenGroups = MmaStrassen::StrassenLevel1Groups<MmaStrassen::StrassenLevel1SingleMiGroup::Group0,
//                                                          MmaStrassen::StrassenLevel1SingleMiGroup::Group1,
//                                                          MmaStrassen::StrassenLevel1MiGroup<false, false, true, true, true, true, false>,
//                                                          MmaStrassen::StrassenLevel1SingleMiGroup::Group6>;

//[m0], [m1], [m3, m2], [m5, m4], [m6]
                                    
using StrassenGroups = StrassenLevel1Groups<StrassenPresum<kStrassenLevel, 0, ThreadBlockShape,
                                                            AllPresums<>>,
                                            StrassenLevel1M0Group<kStrassenLevel, 0, ThreadBlockShape, WarpShape, 3,
                                                                  RWMTypes<KeepAccums>,
                                                                  RWCTypes<CUW<1, LayoutInterim1D, LayoutNone, Expr<Plus<0>>>>,//C1 = M0
                                                                  AllPresums<>>,
                                            StrassenLevel1M1Group<kStrassenLevel, 0, ThreadBlockShape, WarpShape, 3,
                                                                  RWMTypes<ContinueAccums>,
                                                                  RWCTypes<CUW<0, LayoutFinal,   LayoutNone, Expr<Plus<1>>>>,//C0 = C0+M1
                                                                  AllPresums<>>,
                                            StrassenLevel1M2Group<kStrassenLevel, 0, ThreadBlockShape, WarpShape, 3,
                                                                  RWMTypes<>,
                                                                  // RWCTypes<CUW<2, LayoutFinal, LayoutNone, Expr<Plus<2>>>>,
                                                                  RWCTypes<CUW<1, LayoutInterim1D, LayoutNone, Expr<Plus<2>>, Expr<Plus<1, MemGlobal, LayoutInterim1D>>>>,//C1 = C1+M2 ; Reg = C1
                                                                  AllPresums<>>,
                                            StrassenLevel1M3Group<kStrassenLevel, 0, ThreadBlockShape, WarpShape, 3,
                                                                  RWMTypes<>,
                                                                  // RWCTypes<CUW<3, LayoutFinal, LayoutNone, Expr<Plus<3>>>>,
                                                                  RWCTypes<CUW<2, LayoutInterim1D, LayoutNone, Expr<Plus<3>>, Expr<Plus<1, MemGlobal, LayoutInterim1D>>>>,//C2 = C1(Reg)+M3 
                                                                  AllPresums<>>,
                                            StrassenLevel1M4Group<kStrassenLevel, 0, ThreadBlockShape, WarpShape, 3,
                                                                  RWMTypes<>,
                                                                  // RWCTypes<CUW<0, LayoutFinal, LayoutNone, Expr<Plus<4>>>>,
                                                                  RWCTypes<CUW<0, LayoutInterim1D, LayoutNone, Expr<Plus<4>>>, //C1 = C1+M4
                                                                            CUW<3, LayoutFinal,   LayoutNone, Expr<Plus<4>>, Expr<Plus<2, MemGlobal, LayoutInterim1D>>>>,//C3 = C2+M4
                                                                  AllPresums<>>,
                                            StrassenLevel1M5Group<kStrassenLevel, 0, ThreadBlockShape, WarpShape, 3,
                                                                  RWMTypes<>,
                                                                  // RWCTypes<CUW<1, LayoutFinal, LayoutNone, Expr<Plus<5>>>>,
                                                                  RWCTypes<CUW<1, LayoutFinal, LayoutNone, Expr<Plus<5>>, Expr<Plus<1, MemGlobal, LayoutInterim1D>,
                                                                                                                                Plus<0, MemGlobal, LayoutInterim1D>>>>, //C1 = C1+M5
                                                                  AllPresums<>>,
                                            StrassenLevel1M6Group<kStrassenLevel, 0, ThreadBlockShape, WarpShape, 3,
                                                                  RWMTypes<>,
                                                                  // RWCTypes<CUW<2, LayoutFinal, LayoutNone, Expr<Plus<6>>>>,
                                                                  RWCTypes<CUW<2, LayoutFinal, LayoutNone, Expr<Neg<6>>, Expr<Plus<2, MemGlobal, LayoutInterim1D>>>>, //C2 = C2-M6
                                                                  AllPresums<>>
                                            >;
using ScheduleStrassenGroups1 = ScheduleStrassenGroups<ParallelMiGroups<true, FusedMiGroup<7, 0, 1>>,
                                                        ParallelMiGroups<true, FusedMiGroup<7, 2>>,
                                                        ParallelMiGroups<true, FusedMiGroup<7, 3>>,
                                                        ParallelMiGroups<true, FusedMiGroup<7, 4>>,
                                                        ParallelMiGroups<true, FusedMiGroup<7, 5>>,
                                                        ParallelMiGroups<true, FusedMiGroup<7, 6>>>;

using HopperF32SWFusedPresumLevel1Kernel = cutlass::gemm::device::StrassenGemm<StrassenKind, StrassenGroups, ScheduleStrassenGroups1,
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

class HopperF32SWFusedPresumLevel1 {
public:
  using StrassenGemmKernel = HopperF32SWFusedPresumLevel1Kernel;
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
