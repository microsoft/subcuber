#include "cuda/kernels/ampere/strassen_winograd/ampere_f64_sw_interleaved_presum_64x64.cuh"
#include "cuda/kernel_runner_support.cuh"

cutlass::Status AmpereF64SWInterleavedPresumLevel1_64x64::can_implement(Arguments const &args) {
	return StrassenGemmKernel::can_implement(args);
}

size_t AmpereF64SWInterleavedPresumLevel1_64x64::get_workspace_size(Arguments const &args) {
	return StrassenGemmKernel::get_workspace_size(args);
}

cutlass::Status AmpereF64SWInterleavedPresumLevel1_64x64::init(Arguments const &args, void *workspace, cudaStream_t stream) {
	return gemm_.initialize(args, workspace, stream);
}

cutlass::Status AmpereF64SWInterleavedPresumLevel1_64x64::launch(cudaStream_t *streams, int num_streams) {
	return gemm_.run(streams, num_streams);
}

cutlass::Status AmpereF64SWInterleavedPresumLevel1_64x64::initialize(Arguments const &args, void *workspace, cudaStream_t stream) {
	return init(args, workspace, stream);
}

cutlass::Status AmpereF64SWInterleavedPresumLevel1_64x64::run(cudaStream_t *streams, int num_streams) {
	return launch(streams, num_streams);
}

cutlass::Status AmpereF64SWInterleavedPresumLevel1_64x64::operator()(Arguments const &args, void *workspace,
																																		 cudaStream_t *streams, int num_streams) {
	cutlass::Status status = init(args, workspace, streams == nullptr || num_streams == 0 ? nullptr : streams[0]);
	if (status == cutlass::Status::kSuccess) {
		status = launch(streams, num_streams);
	}
	return status;
}

STRASSEN_RUNNER_EXPORT_CUTLASS2(run_ampere_f64_sw_interleaved_presum_64x64, AmpereF64SWInterleavedPresumLevel1_64x64)