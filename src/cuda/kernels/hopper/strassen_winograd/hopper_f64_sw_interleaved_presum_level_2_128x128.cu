#include "cuda/kernels/hopper/strassen_winograd/hopper_f64_sw_interleaved_presum_level_2_128x128.cuh"
#include "cuda/kernel_runner_support.cuh"

cutlass::Status HopperF64SWInterleavedPresumLevel2_128x128::can_implement(Arguments const &args) {
  if (!splitK && args.split_k_slices > 1) {
    return cutlass::Status::kErrorInvalidProblem;
  }

  if (args.problem_size.m() < 4 * ThreadBlockShape::kM ||
      args.problem_size.n() < 4 * ThreadBlockShape::kN ||
      args.problem_size.k() < 4 * ThreadBlockShape::kK ||
      args.problem_size.m() % (4 * ThreadBlockShape::kM) != 0 ||
      args.problem_size.n() % (4 * ThreadBlockShape::kN) != 0 ||
      args.problem_size.k() % (4 * ThreadBlockShape::kK) != 0) {
    return cutlass::Status::kErrorInvalidProblem;
  }

  return cutlass::Status::kSuccess;
}

size_t HopperF64SWInterleavedPresumLevel2_128x128::get_workspace_size(Arguments const &args) {
  return StrassenGemmKernel::get_workspace_size(args);
}

cutlass::Status HopperF64SWInterleavedPresumLevel2_128x128::init(Arguments const &args, void *workspace, cudaStream_t stream) {
  return gemm_.initialize(args, workspace, stream);
}

cutlass::Status HopperF64SWInterleavedPresumLevel2_128x128::launch(cudaStream_t *streams, int num_streams) {
  return gemm_.run(streams, num_streams);
}

cutlass::Status HopperF64SWInterleavedPresumLevel2_128x128::initialize(Arguments const &args, void *workspace, cudaStream_t stream) {
  return init(args, workspace, stream);
}

cutlass::Status HopperF64SWInterleavedPresumLevel2_128x128::run(cudaStream_t *streams, int num_streams) {
  return launch(streams, num_streams);
}

cutlass::Status HopperF64SWInterleavedPresumLevel2_128x128::operator()(Arguments const &args, void *workspace,
                                                                       cudaStream_t *streams, int num_streams) {
  cutlass::Status status = init(args, workspace, streams == nullptr || num_streams == 0 ? nullptr : streams[0]);
  if (status == cutlass::Status::kSuccess) {
    status = launch(streams, num_streams);
  }
  return status;
}

STRASSEN_RUNNER_EXPORT_CUTLASS2(run_hopper_f64_sw_interleaved_presum_level_2_128x128, HopperF64SWInterleavedPresumLevel2_128x128)