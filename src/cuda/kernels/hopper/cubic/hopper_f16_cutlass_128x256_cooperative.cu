#include "cuda/kernels/hopper/cubic/hopper_f16_cutlass_128x256_cooperative.cuh"
#include "cuda/kernel_runner_support.cuh"

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)

cutlass::Status HopperF16Cutlass128x256Cooperative::can_implement(Arguments const &args) {
  return CutlassGemm::can_implement(args);
}

size_t HopperF16Cutlass128x256Cooperative::get_workspace_size(Arguments const &args) {
  return CutlassGemm::get_workspace_size(args);
}

cutlass::Status HopperF16Cutlass128x256Cooperative::init(Arguments const &args, void *workspace, cudaStream_t stream) {
  return gemm_.initialize(args, workspace, stream);
}

cutlass::Status HopperF16Cutlass128x256Cooperative::launch(cudaStream_t *streams, int num_streams) {
  cudaStream_t stream = streams == nullptr || num_streams == 0 ? nullptr : streams[0];
  return gemm_.run(stream);
}

cutlass::Status HopperF16Cutlass128x256Cooperative::initialize(Arguments const &args, void *workspace, cudaStream_t stream) {
  return init(args, workspace, stream);
}

cutlass::Status HopperF16Cutlass128x256Cooperative::run(cudaStream_t *streams, int num_streams) {
  return launch(streams, num_streams);
}

cutlass::Status HopperF16Cutlass128x256Cooperative::operator()(Arguments const &args, void *workspace,
                                                               cudaStream_t *streams, int num_streams) {
  cutlass::Status status = init(args, workspace, streams == nullptr || num_streams == 0 ? nullptr : streams[0]);
  if (status == cutlass::Status::kSuccess) {
    status = launch(streams, num_streams);
  }
  return status;
}

STRASSEN_RUNNER_EXPORT_GEMM_CUTLASS3(run_hopper_f16_cutlass_128x256_cooperative, HopperF16Cutlass128x256Cooperative)

#endif
