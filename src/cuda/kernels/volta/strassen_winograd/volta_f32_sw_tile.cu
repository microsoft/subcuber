#include "cuda/kernels/volta/strassen_winograd/volta_f32_sw_tile.cuh"
#include "cuda/kernel_runner_support.cuh"

cutlass::Status VoltaF32SWTileGemm::can_implement(Arguments const &args) {
  return StrassenGemmKernel::can_implement(args);
}

size_t VoltaF32SWTileGemm::get_workspace_size(Arguments const &args) {
  return StrassenGemmKernel::get_workspace_size(args);
}

cutlass::Status VoltaF32SWTileGemm::init(Arguments const &args, void *workspace, cudaStream_t stream) {
  return gemm_.initialize(args, workspace, stream);
}

cutlass::Status VoltaF32SWTileGemm::launch(cudaStream_t *streams, int num_streams) {
  return gemm_.run(streams, num_streams);
}

cutlass::Status VoltaF32SWTileGemm::initialize(Arguments const &args, void *workspace, cudaStream_t stream) {
  return init(args, workspace, stream);
}

cutlass::Status VoltaF32SWTileGemm::run(cudaStream_t *streams, int num_streams) {
  return launch(streams, num_streams);
}

cutlass::Status VoltaF32SWTileGemm::operator()(Arguments const &args, void *workspace,
                                               cudaStream_t *streams, int num_streams) {
  cutlass::Status status = init(args, workspace, streams == nullptr || num_streams == 0 ? nullptr : streams[0]);
  if (status == cutlass::Status::kSuccess) {
    status = launch(streams, num_streams);
  }
  return status;
}

STRASSEN_RUNNER_EXPORT_CUTLASS2(run_volta_f32_sw_tile, VoltaF32SWTileGemm)