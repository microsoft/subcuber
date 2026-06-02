#include "cuda/kernels/ampere/cubic/ampere_f16_cutlass_128x256.cuh"
#include "cuda/kernel_runner_support.cuh"

cutlass::Status AmpereF16Cutlass128x256::can_implement(Arguments const &args) {
  return CutlassGemm::can_implement(args);
}

size_t AmpereF16Cutlass128x256::get_workspace_size(Arguments const &args) {
  return CutlassGemm::get_workspace_size(args);
}

cutlass::Status AmpereF16Cutlass128x256::init(Arguments const &args, void *workspace, cudaStream_t stream) {
  return gemm_.initialize(args, workspace, stream);
}

cutlass::Status AmpereF16Cutlass128x256::launch(cudaStream_t *streams, int num_streams) {
  cudaStream_t stream = streams == nullptr || num_streams == 0 ? nullptr : streams[0];
  return gemm_.run(stream);
}

cutlass::Status AmpereF16Cutlass128x256::initialize(Arguments const &args, void *workspace, cudaStream_t stream) {
  return init(args, workspace, stream);
}

cutlass::Status AmpereF16Cutlass128x256::run(cudaStream_t *streams, int num_streams) {
  return launch(streams, num_streams);
}

cutlass::Status AmpereF16Cutlass128x256::operator()(Arguments const &args, void *workspace,
                                                    cudaStream_t *streams, int num_streams) {
  cutlass::Status status = init(args, workspace, streams == nullptr || num_streams == 0 ? nullptr : streams[0]);
  if (status == cutlass::Status::kSuccess) {
    status = launch(streams, num_streams);
  }
  return status;
}

STRASSEN_RUNNER_EXPORT_GEMM_CUTLASS2(run_ampere_f16_cutlass_128x256, AmpereF16Cutlass128x256)
