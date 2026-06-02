#include "cuda/kernels/ampere/strassen_winograd/ampere_f32_sw_tile_64x128.cuh"
#include "cuda/kernel_runner_support.cuh"

cutlass::Status AmpereF32SWTileGemm::can_implement(Arguments const &args) {
  return StrassenGemmKernel::can_implement(args);
}

size_t AmpereF32SWTileGemm::get_workspace_size(Arguments const &args) {
  return StrassenGemmKernel::get_workspace_size(args);
}

cutlass::Status AmpereF32SWTileGemm::init(Arguments const &args, void *workspace, cudaStream_t stream) {
  return gemm_.initialize(args, workspace, stream);
}

cutlass::Status AmpereF32SWTileGemm::launch(cudaStream_t *streams, int num_streams) {
  return gemm_.run(streams, num_streams);
}

cutlass::Status AmpereF32SWTileGemm::initialize(Arguments const &args, void *workspace, cudaStream_t stream) {
  return init(args, workspace, stream);
}

cutlass::Status AmpereF32SWTileGemm::run(cudaStream_t *streams, int num_streams) {
  return launch(streams, num_streams);
}

cutlass::Status AmpereF32SWTileGemm::operator()(Arguments const &args, void *workspace,
                                                cudaStream_t *streams, int num_streams) {
  cutlass::Status status = init(args, workspace, streams == nullptr || num_streams == 0 ? nullptr : streams[0]);
  if (status == cutlass::Status::kSuccess) {
    status = launch(streams, num_streams);
  }
  return status;
}

STRASSEN_RUNNER_EXPORT_CUTLASS2(run_ampere_f32_sw_tile, AmpereF32SWTileGemm)