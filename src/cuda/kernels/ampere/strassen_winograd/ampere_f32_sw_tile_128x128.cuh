#include "cutlass/gemm/device/strassen_gemm.h"
#include "cutlass/cutlass.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"

using ThreadBlockShape = cutlass::gemm::GemmShape<128, 128, 8>;
using WarpShape = cutlass::gemm::GemmShape<64, 32, 8>;
using InstructionShape = cutlass::gemm::GemmShape<1,1,1>;
const bool splitK = SPLIT_K;
constexpr int kStrassenLevel = 1;
const auto StrassenKind = StrassenType::StrassenWinograd;
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

using StrassenGroups = MmaStrassen::StrassenLevel1Groups<StrassenPresum<kStrassenLevel, 0, ThreadBlockShape,
                                                                        AllPresums<>>,
                                                          MmaStrassen::StrassenLevel1MiGroup<kStrassenLevel, 0, ThreadBlockShape, WarpShape, 2,
                                                                                            MmaStrassen::RWMTypes<>, MmaStrassen::RWCTypes<>,
                                                                                            MmaStrassen::AllPresums<>, 0, 0,1,2,3,4,5,6>>;
using ScheduleStrassenGroups1 = ScheduleStrassenGroups<ParallelMiGroups<true, FusedMiGroup<7, 0>>>;

class AmpereF32SWTileGemm_128x128 {
public:
  using StrassenGemmKernel = cutlass::gemm::device::StrassenGemm<StrassenKind, StrassenGroups, ScheduleStrassenGroups1,
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
                                                        2, 1, 1, splitK>; // Layout of C matrix

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
