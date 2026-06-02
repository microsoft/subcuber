#include "cuda/kernels/ampere/strassen_winograd/ampere_f16_sw_interleaved_presum_max_fusion.cuh"
#include "cuda/kernel_runner_support.cuh"

cutlass::Status AmpereF16SWInterleavedPresumMaxFusion::can_implement(Arguments const &args) {
  return StrassenGemmKernel::can_implement(args);
}

size_t AmpereF16SWInterleavedPresumMaxFusion::get_workspace_size(Arguments const &args) {
  return StrassenGemmKernel::get_workspace_size(args);
}

cutlass::Status AmpereF16SWInterleavedPresumMaxFusion::init(Arguments const &args, void *workspace, cudaStream_t stream) {
  return gemm_.initialize(args, workspace, stream);
}

cutlass::Status AmpereF16SWInterleavedPresumMaxFusion::launch(cudaStream_t *streams, int num_streams) {
  return gemm_.run(streams, num_streams);
}

cutlass::Status AmpereF16SWInterleavedPresumMaxFusion::initialize(Arguments const &args, void *workspace, cudaStream_t stream) {
  return init(args, workspace, stream);
}

cutlass::Status AmpereF16SWInterleavedPresumMaxFusion::run(cudaStream_t *streams, int num_streams) {
  return launch(streams, num_streams);
}

cutlass::Status AmpereF16SWInterleavedPresumMaxFusion::operator()(Arguments const &args, void *workspace,
                                    cudaStream_t *streams, int num_streams) {
  cutlass::Status status = init(args, workspace, streams == nullptr || num_streams == 0 ? nullptr : streams[0]);
  if (status == cutlass::Status::kSuccess) {
    status = launch(streams, num_streams);
  }
  return status;
}

STRASSEN_RUNNER_EXPORT_CUTLASS2(run_ampere_f16_sw_interleaved_presum_max_fusion, AmpereF16SWInterleavedPresumMaxFusion)
