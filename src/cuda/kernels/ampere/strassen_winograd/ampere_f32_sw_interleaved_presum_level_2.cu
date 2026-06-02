#include "cuda/kernels/ampere/strassen_winograd/ampere_f32_sw_interleaved_presum_level_2.cuh"
#include "cuda/kernel_runner_support.cuh"

cutlass::Status AmpereF32SWInterleavedPresumLevel2::can_implement(Arguments const &args) {
  return StrassenGemmKernel::can_implement(args);
}

size_t AmpereF32SWInterleavedPresumLevel2::get_workspace_size(Arguments const &args) {
  return StrassenGemmKernel::get_workspace_size(args);
}

cutlass::Status AmpereF32SWInterleavedPresumLevel2::init(Arguments const &args, void *workspace, cudaStream_t stream) {
  return gemm_.initialize(args, workspace, stream);
}

cutlass::Status AmpereF32SWInterleavedPresumLevel2::launch(cudaStream_t *streams, int num_streams) {
  return gemm_.run(streams, num_streams);
}

cutlass::Status AmpereF32SWInterleavedPresumLevel2::initialize(Arguments const &args, void *workspace, cudaStream_t stream) {
  return init(args, workspace, stream);
}

cutlass::Status AmpereF32SWInterleavedPresumLevel2::run(cudaStream_t *streams, int num_streams) {
  return launch(streams, num_streams);
}

cutlass::Status AmpereF32SWInterleavedPresumLevel2::operator()(Arguments const &args, void *workspace,
                                    cudaStream_t *streams, int num_streams) {
  cutlass::Status status = init(args, workspace, streams == nullptr || num_streams == 0 ? nullptr : streams[0]);
  if (status == cutlass::Status::kSuccess) {
    status = launch(streams, num_streams);
  }
  return status;
}

STRASSEN_RUNNER_EXPORT_CUTLASS2(run_ampere_f32_sw_interleaved_presum_level_2, AmpereF32SWInterleavedPresumLevel2)
