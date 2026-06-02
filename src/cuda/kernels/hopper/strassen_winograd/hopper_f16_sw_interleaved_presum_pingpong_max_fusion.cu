#include "cuda/kernels/hopper/strassen_winograd/hopper_f16_sw_interleaved_presum_pingpong_max_fusion.cuh"
#include "cuda/kernel_runner_support.cuh"

cutlass::Status HopperF16InterleavedPresumPingpongMaxFusion_2x128_2x128_OptNo::can_implement(Arguments const &args, cutlass::CudaHostAdapter *cuda_adapter) {
  return StrassenGemmKernel::can_implement(args);
}

size_t HopperF16InterleavedPresumPingpongMaxFusion_2x128_2x128_OptNo::get_workspace_size(Arguments const &args, cutlass::CudaHostAdapter *cuda_adapter) {
  return StrassenGemmKernel::get_workspace_size(args);
}

cutlass::Status HopperF16InterleavedPresumPingpongMaxFusion_2x128_2x128_OptNo::init(Arguments const &args, int swizzles[7], void *workspace,
                              cudaStream_t stream, cutlass::CudaHostAdapter *cuda_adapter) {
  return gemm_.initialize(args, swizzles, workspace, stream, cuda_adapter);
}

cutlass::Status HopperF16InterleavedPresumPingpongMaxFusion_2x128_2x128_OptNo::launch(cudaStream_t *streams, int num_streams,
                                cutlass::CudaHostAdapter *cuda_adapter, bool launch_with_pdl) {
  return gemm_.run(streams, num_streams, cuda_adapter, launch_with_pdl);
}

cutlass::Status HopperF16InterleavedPresumPingpongMaxFusion_2x128_2x128_OptNo::initialize(Arguments const &args, int swizzles[7], void *workspace,
                                    cudaStream_t stream, cutlass::CudaHostAdapter *cuda_adapter) {
  return init(args, swizzles, workspace, stream, cuda_adapter);
}

cutlass::Status HopperF16InterleavedPresumPingpongMaxFusion_2x128_2x128_OptNo::run(cudaStream_t *streams, int num_streams,
                             cutlass::CudaHostAdapter *cuda_adapter, bool launch_with_pdl) {
  return launch(streams, num_streams, cuda_adapter, launch_with_pdl);
}

cutlass::Status HopperF16InterleavedPresumPingpongMaxFusion_2x128_2x128_OptNo::operator()(Arguments const &args, int swizzles[7], void *workspace,
                                    cudaStream_t *streams, int num_streams,
                                    cutlass::CudaHostAdapter *cuda_adapter, bool launch_with_pdl) {
  cutlass::Status status = init(args, swizzles, workspace,
                                streams == nullptr || num_streams == 0 ? nullptr : streams[0], cuda_adapter);
  if (status == cutlass::Status::kSuccess) {
    status = launch(streams, num_streams, cuda_adapter, launch_with_pdl);
  }
  return status;
}

cutlass::Status HopperF16InterleavedPresumPingpongMaxFusion_2x128_2x128_Opt_0000::can_implement(Arguments const &args, cutlass::CudaHostAdapter *cuda_adapter) {
  return StrassenGemmKernel::can_implement(args);
}

size_t HopperF16InterleavedPresumPingpongMaxFusion_2x128_2x128_Opt_0000::get_workspace_size(Arguments const &args, cutlass::CudaHostAdapter *cuda_adapter) {
  return StrassenGemmKernel::get_workspace_size(args);
}

cutlass::Status HopperF16InterleavedPresumPingpongMaxFusion_2x128_2x128_Opt_0000::init(Arguments const &args, int swizzles[7], void *workspace,
                              cudaStream_t stream, cutlass::CudaHostAdapter *cuda_adapter) {
  return gemm_.initialize(args, swizzles, workspace, stream, cuda_adapter);
}

cutlass::Status HopperF16InterleavedPresumPingpongMaxFusion_2x128_2x128_Opt_0000::launch(cudaStream_t *streams, int num_streams,
                                cutlass::CudaHostAdapter *cuda_adapter, bool launch_with_pdl) {
  return gemm_.run(streams, num_streams, cuda_adapter, launch_with_pdl);
}

cutlass::Status HopperF16InterleavedPresumPingpongMaxFusion_2x128_2x128_Opt_0000::initialize(Arguments const &args, int swizzles[7], void *workspace,
                                    cudaStream_t stream, cutlass::CudaHostAdapter *cuda_adapter) {
  return init(args, swizzles, workspace, stream, cuda_adapter);
}

cutlass::Status HopperF16InterleavedPresumPingpongMaxFusion_2x128_2x128_Opt_0000::run(cudaStream_t *streams, int num_streams,
                             cutlass::CudaHostAdapter *cuda_adapter, bool launch_with_pdl) {
  return launch(streams, num_streams, cuda_adapter, launch_with_pdl);
}

cutlass::Status HopperF16InterleavedPresumPingpongMaxFusion_2x128_2x128_Opt_0000::operator()(Arguments const &args, int swizzles[7], void *workspace,
                                    cudaStream_t *streams, int num_streams,
                                    cutlass::CudaHostAdapter *cuda_adapter, bool launch_with_pdl) {
  cutlass::Status status = init(args, swizzles, workspace,
                                streams == nullptr || num_streams == 0 ? nullptr : streams[0], cuda_adapter);
  if (status == cutlass::Status::kSuccess) {
    status = launch(streams, num_streams, cuda_adapter, launch_with_pdl);
  }
  return status;
}

cutlass::Status HopperF16InterleavedPresumPingpongMaxFusion_4x128_4x128_OptNo::can_implement(Arguments const &args, cutlass::CudaHostAdapter *cuda_adapter) {
  return StrassenGemmKernel::can_implement(args);
}

size_t HopperF16InterleavedPresumPingpongMaxFusion_4x128_4x128_OptNo::get_workspace_size(Arguments const &args, cutlass::CudaHostAdapter *cuda_adapter) {
  return StrassenGemmKernel::get_workspace_size(args);
}

cutlass::Status HopperF16InterleavedPresumPingpongMaxFusion_4x128_4x128_OptNo::init(Arguments const &args, int swizzles[7], void *workspace,
                              cudaStream_t stream, cutlass::CudaHostAdapter *cuda_adapter) {
  return gemm_.initialize(args, swizzles, workspace, stream, cuda_adapter);
}

cutlass::Status HopperF16InterleavedPresumPingpongMaxFusion_4x128_4x128_OptNo::launch(cudaStream_t *streams, int num_streams,
                                cutlass::CudaHostAdapter *cuda_adapter, bool launch_with_pdl) {
  return gemm_.run(streams, num_streams, cuda_adapter, launch_with_pdl);
}

cutlass::Status HopperF16InterleavedPresumPingpongMaxFusion_4x128_4x128_OptNo::initialize(Arguments const &args, int swizzles[7], void *workspace,
                                    cudaStream_t stream, cutlass::CudaHostAdapter *cuda_adapter) {
  return init(args, swizzles, workspace, stream, cuda_adapter);
}

cutlass::Status HopperF16InterleavedPresumPingpongMaxFusion_4x128_4x128_OptNo::run(cudaStream_t *streams, int num_streams,
                             cutlass::CudaHostAdapter *cuda_adapter, bool launch_with_pdl) {
  return launch(streams, num_streams, cuda_adapter, launch_with_pdl);
}

cutlass::Status HopperF16InterleavedPresumPingpongMaxFusion_4x128_4x128_OptNo::operator()(Arguments const &args, int swizzles[7], void *workspace,
                                    cudaStream_t *streams, int num_streams,
                                    cutlass::CudaHostAdapter *cuda_adapter, bool launch_with_pdl) {
  cutlass::Status status = init(args, swizzles, workspace,
                                streams == nullptr || num_streams == 0 ? nullptr : streams[0], cuda_adapter);
  if (status == cutlass::Status::kSuccess) {
    status = launch(streams, num_streams, cuda_adapter, launch_with_pdl);
  }
  return status;
}

cutlass::Status HopperF16InterleavedPresumPingpongMaxFusion_8x128_8x128_OptNo::can_implement(Arguments const &args, cutlass::CudaHostAdapter *cuda_adapter) {
  return StrassenGemmKernel::can_implement(args);
}

size_t HopperF16InterleavedPresumPingpongMaxFusion_8x128_8x128_OptNo::get_workspace_size(Arguments const &args, cutlass::CudaHostAdapter *cuda_adapter) {
  return StrassenGemmKernel::get_workspace_size(args);
}

cutlass::Status HopperF16InterleavedPresumPingpongMaxFusion_8x128_8x128_OptNo::init(Arguments const &args, int swizzles[7], void *workspace,
                              cudaStream_t stream, cutlass::CudaHostAdapter *cuda_adapter) {
  return gemm_.initialize(args, swizzles, workspace, stream, cuda_adapter);
}

cutlass::Status HopperF16InterleavedPresumPingpongMaxFusion_8x128_8x128_OptNo::launch(cudaStream_t *streams, int num_streams,
                                cutlass::CudaHostAdapter *cuda_adapter, bool launch_with_pdl) {
  return gemm_.run(streams, num_streams, cuda_adapter, launch_with_pdl);
}

cutlass::Status HopperF16InterleavedPresumPingpongMaxFusion_8x128_8x128_OptNo::initialize(Arguments const &args, int swizzles[7], void *workspace,
                                    cudaStream_t stream, cutlass::CudaHostAdapter *cuda_adapter) {
  return init(args, swizzles, workspace, stream, cuda_adapter);
}

cutlass::Status HopperF16InterleavedPresumPingpongMaxFusion_8x128_8x128_OptNo::run(cudaStream_t *streams, int num_streams,
                             cutlass::CudaHostAdapter *cuda_adapter, bool launch_with_pdl) {
  return launch(streams, num_streams, cuda_adapter, launch_with_pdl);
}

cutlass::Status HopperF16InterleavedPresumPingpongMaxFusion_8x128_8x128_OptNo::operator()(Arguments const &args, int swizzles[7], void *workspace,
                                    cudaStream_t *streams, int num_streams,
                                    cutlass::CudaHostAdapter *cuda_adapter, bool launch_with_pdl) {
  cutlass::Status status = init(args, swizzles, workspace,
                                streams == nullptr || num_streams == 0 ? nullptr : streams[0], cuda_adapter);
  if (status == cutlass::Status::kSuccess) {
    status = launch(streams, num_streams, cuda_adapter, launch_with_pdl);
  }
  return status;
}

STRASSEN_RUNNER_EXPORT_CUTLASS3(run_hopper_f16_sw_interleaved_presum_pingpong_max_fusion_2x128_2x128_opt_no, HopperF16InterleavedPresumPingpongMaxFusion_2x128_2x128_OptNo)
STRASSEN_RUNNER_EXPORT_CUTLASS3(run_hopper_f16_sw_interleaved_presum_pingpong_max_fusion_2x128_2x128_opt_0000, HopperF16InterleavedPresumPingpongMaxFusion_2x128_2x128_Opt_0000)
STRASSEN_RUNNER_EXPORT_CUTLASS3(run_hopper_f16_sw_interleaved_presum_pingpong_max_fusion_4x128_4x128_opt_no, HopperF16InterleavedPresumPingpongMaxFusion_4x128_4x128_OptNo)
STRASSEN_RUNNER_EXPORT_CUTLASS3(run_hopper_f16_sw_interleaved_presum_pingpong_max_fusion_8x128_8x128_opt_no, HopperF16InterleavedPresumPingpongMaxFusion_8x128_8x128_OptNo)
