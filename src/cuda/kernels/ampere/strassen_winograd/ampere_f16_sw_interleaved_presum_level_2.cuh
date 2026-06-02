#include "cutlass/gemm/device/strassen_gemm.h"
#include "cutlass/cutlass.h"

// The code section below describes datatype for input, output matrices and computation between
// elements in input matrices.
using ElementAccumulator = float;                   // <- data type of accumulator
using ElementComputeEpilogue = float;  // <- data type of epilogue operations
using ElementInputA = cutlass::half_t;                        // <- data type of elements in input matrix A
using ElementInputB = cutlass::half_t;                        // <- data type of elements in input matrix B
using ElementOutput = cutlass::half_t;                        // <- data type of elements in output matrix D

// The code section below describes matrix layout of input and output matrices. Column Major for
// Matrix A, Row Major for Matrix B and Row Major for Matrix C
using RowMajor = cutlass::layout::RowMajor;
using LayoutInputA = RowMajor;
using LayoutInputB = RowMajor;
using LayoutOutput = RowMajor;

// This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
using MMAOp = cutlass::arch::OpClassTensorOp;

// This code section describes CUDA SM architecture number
using SmArch = cutlass::arch::Sm80;

// This code section describes the tile size a thread block will compute
using ShapeMMAThreadBlock =
    cutlass::gemm::GemmShape<128, 256, 32>;  // <- threadblock tile M = 128, N = 128, K = 16
// This code section describes tile size a warp will compute
using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 32>;  // <- warp tile M = 64, N = 64, K = 16
// This code section describes the size of MMA op
using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 16>;  // <- MMA Op tile M = 16, N = 8, K = 8

constexpr int kStrassenLevel = 1;

// This code section describes how threadblocks are scheduled on GPU
//1 for 5k x 5k x 5k and 8 for others
using SwizzleThreadBlock = cutlass::gemm::threadblock::StrassenGemmIdentityThreadblockSwizzle<8>;  // <- ??

// This code section describes the epilogue part of the kernel
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,                                     // <- data type of output matrix
    128 / cutlass::sizeof_bits<ElementOutput>::value,  // <- the number of elements per vectorized
                                                       // memory access. For a byte, it's 16
                                                       // elements. This becomes the vector width of
                                                       // math instructions in the epilogue too
    ElementAccumulator,                                // <- data type of accumulator
    ElementComputeEpilogue>;  // <- data type for alpha/beta in linear combination function

using InterimEpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,                                     // <- data type of output matrix
    128 / cutlass::sizeof_bits<ElementOutput>::value,  // <- the number of elements per vectorized
                                                       // memory access. For a byte, it's 16
                                                       // elements. This becomes the vector width of
                                                       // math instructions in the epilogue too
    ElementAccumulator,                                // <- data type of accumulator
    ElementComputeEpilogue>;  // <- data type for alpha/beta in linear combination function

// Number of pipelines you want to use
constexpr int NumStages = 4; //TODO: Try with NumStages=3

using AllPresumsKernel = AllPresums<>;
                          //  AllPresums<PresumGlobalKernel,   PresumGlobalKernel,  PresumGlobalKernel,   PresumGlobalKernel,  //A Presums
                                        // PresumGlobalKernel,   PresumGlobalKernel,  PresumGlobalKernel,   PresumGlobalKernel>; //B Presums
using AllPresumsM0    = AllPresums<PresumCompute, PresumCompute, PresumCompute, PresumCompute,
                                    PresumCompute, PresumCompute, PresumCompute, PresumCompute>;
using AllPresumsM1To6 = AllPresums<PresumAvailable, PresumAvailable, PresumAvailable, PresumAvailable,
                                    PresumAvailable, PresumAvailable, PresumAvailable, PresumAvailable>;

using StrassenGroupsM0 = StrassenLevel1Groups<StrassenPresum<kStrassenLevel, 0, ShapeMMAThreadBlock,
                                                               AllPresumsKernel>,
                                                StrassenLevel1MiGroup<2, 0, ShapeMMAThreadBlock, ShapeMMAWarp, 3,
                                                                    RWMTypes<>,
                                                                    RWCTypes<CUW<1, LayoutInterim, LayoutNone, Expr<Plus<0>>>//C1 = M0
                                                                            //  CUW<0, LayoutFinal, LayoutNone, Expr<Plus<1>>>
                                                                             >,//C0 = M1
                                                                    AllPresumsM0, 0, 0>>;
  template<int level_1_idx>
  using StrassenGroups = StrassenLevel1Groups<StrassenPresum<kStrassenLevel, 0, ShapeMMAThreadBlock,
                                                             AllPresumsKernel>,
                                              StrassenLevel1MiGroup<kStrassenLevel, level_1_idx, ShapeMMAThreadBlock, ShapeMMAWarp, 3,
                                                                    RWMTypes<>,
                                                                    RWCTypes<//CUW<1, LayoutInterim, LayoutNone, Expr<Plus<0>>>,//C1 = M0
                                                                             CUW<0, LayoutFinal, LayoutNone, Expr<Plus<1>>>>,//C0 = M1
                                                                    AllPresumsM0, 1, 0, 1>,
                                              StrassenLevel1M2Group<kStrassenLevel, level_1_idx, ShapeMMAThreadBlock, ShapeMMAWarp, 3,
                                                                    RWMTypes<>,
                                                                    RWCTypes<CUW<1, LayoutInterim1D, LayoutInterim1D, Expr<Plus<2>>, Expr<Plus<1, MemGlobal, LayoutInterim1D>>>>,//C1 = C1+M2 ; Reg = C1 //TODO: pass C1 through registers
                                                                    AllPresumsM1To6>,
                                              StrassenLevel1M3Group<kStrassenLevel, level_1_idx, ShapeMMAThreadBlock, ShapeMMAWarp, 3,
                                                                    RWMTypes<>,
                                                                    RWCTypes<CUW<2, LayoutNone, LayoutInterim1D, Expr<Plus<3>>, Expr<Plus<1, MemShared, LayoutInterim1D>>>>,//C2 = C1(Reg)+M3 
                                                                    AllPresumsM1To6>,
                                              StrassenLevel1M4Group<kStrassenLevel, level_1_idx, ShapeMMAThreadBlock, ShapeMMAWarp, 3,
                                                                    RWMTypes<>,
                                                                    RWCTypes<CUW<0, LayoutInterim1D, LayoutNone, Expr<Plus<4>>>, //C1 (stored at M0) = C1+M4
                                                                             CUW<3, LayoutFinal,   LayoutNone, Expr<Plus<4>>, Expr<Plus<2, MemShared, LayoutInterim1D>>>>,//C3 = C2+M4
                                                                    AllPresumsM1To6>,
                                              StrassenLevel1M5Group<kStrassenLevel, level_1_idx, ShapeMMAThreadBlock, ShapeMMAWarp, 3,
                                                                    RWMTypes<>,
                                                                    RWCTypes<CUW<1, LayoutFinal, LayoutNone, Expr<Plus<5>>, Expr<Plus<1, MemGlobal, LayoutInterim1D>,
                                                                                                                                 Plus<0, MemGlobal, LayoutInterim1D>>>>, //C1 = C1+M5 //TODO: in code M5 reads M1 and M0 (written by M4)
                                                                    AllPresumsM1To6>,
                                              StrassenLevel1M6Group<kStrassenLevel, level_1_idx, ShapeMMAThreadBlock, ShapeMMAWarp, 3,
                                                                    RWMTypes<>,
                                                                    RWCTypes<CUW<2, LayoutFinal, LayoutNone, Expr<Neg<6>>, Expr<Plus<2, MemShared, LayoutInterim1D>>>>, //C2 = C2-M6
                                                                    AllPresumsM1To6>
                                              >;
  using ScheduleStrassenGroups1 = ScheduleStrassenGroups<ParallelMiGroups<true, FusedMiGroup<7, 0>>,
                                                         ParallelMiGroups<true, FusedMiGroup<7, 1, 2, 5, 3>,
                                                        //  ParallelMiGroups<FusedMiGroup<7, 2>>,
                                                                            FusedMiGroup<7, 4>>
                                                         //ParallelMiGroups<true, FusedMiGroup<7, 4>>
                                                        //  ParallelMiGroups<FusedMiGroup<7, 5>>
                                                         >;


const auto StrassenKind = StrassenType::StrassenWinograd;
using AmpereF16SWInterleavedPresumLevel2Kernel = cutlass::gemm::device::StrassenGemmLevel2<StrassenKind, 
                                                            StrassenGroupsM0, StrassenGroups<1>, StrassenGroups<2>,
                                                            StrassenGroups<3>, StrassenGroups<4>, StrassenGroups<5>,
                                                            StrassenGroups<6>,
                                                            ScheduleStrassenGroups1,
                                                            ElementInputA,
                                                            LayoutInputA,
                                                            ElementInputB,
                                                            LayoutInputB,
                                                            ElementOutput,
                                                            LayoutOutput,
                                                            ElementAccumulator,
                                                            MMAOp,
                                                            SmArch,
                                                            ShapeMMAThreadBlock,
                                                            ShapeMMAWarp,
                                                            ShapeMMAOp,
                                                            EpilogueOp,
                                                            InterimEpilogueOp,
                                                            SwizzleThreadBlock,
                                                            NumStages, 8, 8,
                                                            false, false>;

class AmpereF16SWInterleavedPresumLevel2 {
public:
  using StrassenGemmKernel = AmpereF16SWInterleavedPresumLevel2Kernel;
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
