#include "cuda/kernels/hopper/strassen_winograd/hopper_f64_sw_fused_presum_128x128.cuh"
#include "cuda/kernel_runner_support.cuh"

cutlass::Status HopperF64SWFusedPresumLevel1_128x128::can_implement(Arguments const &args) {
  return StrassenGemmKernel::can_implement(args);
}

size_t HopperF64SWFusedPresumLevel1_128x128::get_workspace_size(Arguments const &args) {
  return StrassenGemmKernel::get_workspace_size(args);
}

cutlass::Status HopperF64SWFusedPresumLevel1_128x128::init(Arguments const &args, void *workspace, cudaStream_t stream) {
  return gemm_.initialize(args, workspace, stream);
}

cutlass::Status HopperF64SWFusedPresumLevel1_128x128::launch(cudaStream_t *streams, int num_streams) {
  return gemm_.run(streams, num_streams);
}

cutlass::Status HopperF64SWFusedPresumLevel1_128x128::initialize(Arguments const &args, void *workspace, cudaStream_t stream) {
  return init(args, workspace, stream);
}

cutlass::Status HopperF64SWFusedPresumLevel1_128x128::run(cudaStream_t *streams, int num_streams) {
  return launch(streams, num_streams);
}

cutlass::Status HopperF64SWFusedPresumLevel1_128x128::operator()(Arguments const &args, void *workspace,
                                                                 cudaStream_t *streams, int num_streams) {
  cutlass::Status status = init(args, workspace, streams == nullptr || num_streams == 0 ? nullptr : streams[0]);
  if (status == cutlass::Status::kSuccess) {
    status = launch(streams, num_streams);
  }
  return status;
}

STRASSEN_RUNNER_EXPORT_CUTLASS2(run_hopper_f64_sw_fused_presum_128x128, HopperF64SWFusedPresumLevel1_128x128)