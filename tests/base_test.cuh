#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "gtest/gtest.h"

#include "cutlass/util/command_line.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"

#include "src/cuda/allocate_float_matrices.cuh"
#include "src/cuda/cubic_reference.cuh"
#include "src/cuda/strassen_reference_l1.cuh"

namespace strassen_tests {

struct TestCase {
	cutlass::gemm::GemmCoord problem_size;
	int split_k_slices;
	int level;
	const char *name;
};

inline void PrintTo(TestCase const &test_case, std::ostream *os) {
	*os << test_case.name;
}

enum KernelType {
	Threadblock,
	Presum
};

enum GPUArch {
	Ampere,
	Hopper
};

enum DataType {
	F32,
	F16
};

inline std::string gtest_case_name(testing::TestParamInfo<TestCase> const &info) {
	std::string name = info.param.name;
	for (char &ch : name) {
		bool valid = (ch >= 'a' && ch <= 'z') ||
								 (ch >= 'A' && ch <= 'Z') ||
								 (ch >= '0' && ch <= '9');
		if (!valid) {
			ch = '_';
		}
	}

	return name;
}

inline bool device_supports_tests() {
	cudaDeviceProp props;
	cudaError_t error = cudaGetDeviceProperties(&props, 0);
	if (error != cudaSuccess) {
		std::cerr << "cudaGetDeviceProperties failed: " << cudaGetErrorString(error) << std::endl;
		return false;
	}

	if ((props.major * 10 + props.minor) < 80) {
		std::cout << "Skipping Strassen-Winograd tests: compute capability "
							<< props.major << "." << props.minor << " is below sm80." << std::endl;
		return false;
	}

	return true;
}

template<typename ElementOutput>
inline bool compare_results(
	std::vector<ElementOutput> const &actual,
	std::vector<ElementOutput> const &expected,
	int n, int level) {
	static_assert(std::is_same<ElementOutput, float>::value || std::is_same<ElementOutput, cutlass::half_t>::value);

	float max_rel_tol = std::is_same<ElementOutput, float>::value ? 1e-4f : 1e-2f;
	float max_abs_tol = std::is_same<ElementOutput, float>::value ? 1e-4f : (level == 1 ? 4 : 7);
	for (std::size_t idx = 0; idx < actual.size(); ++idx) {
		float cutlass_value = static_cast<float>(actual[idx]);
		float reference_value = static_cast<float>(expected[idx]);
		float rel_diff;
		float abs_diff;
		bool equal = false;
		if ((reference_value == 0.0f && cutlass_value == 0.0f) ||
				(std::isnan(reference_value) && std::isnan(cutlass_value))) {
			equal = true;
		} else if (!std::isnan(reference_value) && !std::isnan(cutlass_value)) {
			abs_diff = std::fabs(reference_value - cutlass_value);
			rel_diff = (std::fabs(reference_value - cutlass_value) / std::fabs(reference_value + 1.0e-6f));
			equal = (abs_diff <= max_abs_tol || rel_diff <= max_rel_tol);
		}

		if (!equal) {
			std::cerr << "Mismatch at (" << (idx / n) << ", " << (idx % n) << "): "
								<< cutlass_value << " != " << reference_value << " rel_diff = " << rel_diff << " abs_diff = " << abs_diff << std::endl;
			return false;
		}
	}

	return true;
}

template<typename CutlassGemm,
				 typename TensorRefA, typename TensorRefB,
		 		 typename TensorRefC>
inline cudaError_t run_cutlass_gemm(
	TestCase const &test_case,
	TensorRefA ref_a,
	TensorRefB ref_b,
	TensorRefC ref_c,
	TensorRefC ref_d,
	std::array<cudaStream_t, 49> &streams) {

	auto problem_size = test_case.problem_size;
#ifdef CUTLASS_API_v2
	int ldc = problem_size.n();

	CutlassGemm gemm_operator;
	typename CutlassGemm::Arguments args(
		problem_size,
		ref_a,
		ref_b,
		ref_c,
		ref_d,
		{1.0f, 0.0f},
		test_case.split_k_slices);

	size_t workspace_size = CutlassGemm::get_workspace_size(args);
	cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);

	cutlass::Status status = gemm_operator.initialize(args, workspace.get());
	if (status != cutlass::Status::kSuccess) {
		std::cerr << "CUTLASS initialize failed with status " << int(status) << std::endl;
		return cudaErrorUnknown;
	}

	auto run_streams = streams;
	status = gemm_operator.run(run_streams.data(), 1);
	if (status != cutlass::Status::kSuccess) {
		std::cerr << "CUTLASS run failed with status " << int(status) << std::endl;
		return cudaErrorUnknown;
	}

	cudaError_t result = cudaDeviceSynchronize();
	if (result != cudaSuccess) {
		std::cerr << "CUTLASS synchronization failed: " << cudaGetErrorString(result) << std::endl;
	}

	return result;

#elif CUTLASS_API_v3
	using StrideA = typename CutlassGemm::GemmKernel::StrideA;
	using StrideB = typename CutlassGemm::GemmKernel::StrideB;
	using StrideC = typename CutlassGemm::GemmKernel::StrideC;
	using StrideD = typename CutlassGemm::GemmKernel::StrideD;

	StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, {problem_size.m(), problem_size.k(), 1});
	StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, {problem_size.n(), problem_size.k(), 1});
	StrideC stride_c = cutlass::make_cute_packed_stride(StrideC{}, {problem_size.m(), problem_size.n(), 1});
	StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, {problem_size.m(), problem_size.n(), 1});

	int device_id = 0;
	auto kernel_hw_info = cutlass::KernelHardwareInfo::make_kernel_hardware_info<typename CutlassGemm::GemmKernel>(device_id);

	typename CutlassGemm::Arguments args(
		cutlass::gemm::GemmUniversalMode::kGemm,
		{problem_size.m(), problem_size.n(), problem_size.k()},
		{ref_a.data(), stride_a, ref_b.data(), stride_b},
		{{1.0f, 0.0f}, ref_c.data(), stride_c, ref_d.data(), stride_d},
		kernel_hw_info);

	CutlassGemm gemm_operator;
	cutlass::Status status = CutlassGemm::can_implement(args);
	if (status != cutlass::Status::kSuccess) {
		std::cerr << "CUTLASS can_implement failed with status " << int(status) << std::endl;
		return cudaErrorNotSupported;
	}

	size_t workspace_size = CutlassGemm::get_workspace_size(args);
	cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);
	int swizzles[7] = {1, 1, 1, 1, 1, 1, 1};

	status = gemm_operator.initialize(args, swizzles, workspace.get());
	if (status != cutlass::Status::kSuccess) {
		std::cerr << "CUTLASS initialize failed with status " << int(status) << std::endl;
		return cudaErrorUnknown;
	}

	auto run_streams = streams;
	status = gemm_operator.run(run_streams.data(), 1);
	if (status != cutlass::Status::kSuccess) {
		std::cerr << "CUTLASS run failed with status " << int(status) << std::endl;
		return cudaErrorUnknown;
	}

	cudaError_t result = cudaDeviceSynchronize();
	if (result != cudaSuccess) {
		std::cerr << "CUTLASS synchronization failed: " << cudaGetErrorString(result) << std::endl;
	}

	return result;

#else
	return cudaErrorNotSupported;
#endif
}

template<typename CutlassGemm,
				 typename ElementInputA,
		 		 typename ElementInputB,
		 		 typename ElementOutput>
inline cudaError_t run_case(TestCase const &test_case) {
	auto problem_size = test_case.problem_size;
	int lda = problem_size.k();
	int ldb = problem_size.n();
	int ldc = problem_size.n();
	size_t sizeof_c = sizeof(ElementOutput) * size_t(problem_size.m()) * size_t(problem_size.n());

	cutlass::HostTensor<ElementInputA, cutlass::layout::RowMajor> tensor_a(
		problem_size.mk());
	cutlass::HostTensor<ElementInputB, cutlass::layout::RowMajor> tensor_b(
		problem_size.kn());  // <- Create matrix B with dimensions K x N
	cutlass::HostTensor<ElementOutput, cutlass::layout::RowMajor> tensor_c(
		problem_size.mn());  // <- Create matrix C with dimensions M x N
	cutlass::HostTensor<ElementOutput, cutlass::layout::RowMajor> tensor_d(
		problem_size.mn());  // <- Create matrix D with dimensions M x N used to store output from
							// CUTLASS kernel
	cutlass::HostTensor<ElementOutput, cutlass::layout::RowMajor> tensor_ref_d(
		problem_size.mn());  // <- Create matrix D with dimensions M x N used to store output from
							// reference kernel
	
	int range_end = std::is_same<ElementInputA, float>::value ? 16 : 4;
	int range_start = std::is_same<ElementInputA, float>::value ? -16 : -4;
	cutlass::reference::host::TensorFillRandomUniform(
         tensor_a.host_view(),
         1,
         ElementInputA(range_end),
         ElementInputA(range_start),
         2);

	cutlass::reference::host::TensorFillRandomUniform(
			tensor_b.host_view(),
			1,
			ElementInputB(range_end),
			ElementInputB(range_start),
			2);

	cutlass::reference::host::TensorFill(tensor_c.host_view());
	cutlass::reference::host::TensorFill(tensor_d.host_view());
	cutlass::reference::host::TensorFill(tensor_ref_d.host_view());

	tensor_a.sync_device(); tensor_b.sync_device(); tensor_c.sync_device();
	tensor_d.sync_device(); tensor_ref_d.sync_device();

	cudaError_t result;
	std::array<cudaStream_t, 49> streams{};
	for (cudaStream_t &stream : streams) {
		result = cudaStreamCreate(&stream);
		if (result != cudaSuccess) {
			std::cerr << "cudaStreamCreate failed: " << cudaGetErrorString(result) << std::endl;
			for (cudaStream_t created_stream : streams) {
				if (created_stream) {
					cudaStreamDestroy(created_stream);
				}
			}
			return result;
		}
	}

	result = run_cutlass_gemm<CutlassGemm>(test_case, tensor_a.device_ref(), tensor_b.device_ref(),
							  												 tensor_c.device_ref(), tensor_d.device_ref(), streams);

	if (result == cudaSuccess) {
		result = ReferenceStrassenWinogradGemm<ElementInputA, ElementInputB, ElementOutput, float, float>(
			problem_size.m(),
			problem_size.n(),
			problem_size.k(),
			1,
			1.0f,
			tensor_a.device_ref().data(),
			lda,
			tensor_b.device_ref().data(),
			ldb,
			0.0f,
			tensor_ref_d.device_ref().data(),
			ldc);
		if (result != cudaSuccess) {
			std::cerr << "Reference GEMM failed: " << cudaGetErrorString(result) << std::endl;
		}
	}

	if (result == cudaSuccess) {
		std::vector<ElementOutput> host_cutlass(size_t(problem_size.m()) * size_t(problem_size.n()));
		std::vector<ElementOutput> host_reference(size_t(problem_size.m()) * size_t(problem_size.n()));

		result = cudaMemcpy(host_cutlass.data(), tensor_d.device_ref().data(), sizeof_c, cudaMemcpyDeviceToHost);
		if (result != cudaSuccess) {
			std::cerr << "Copying CUTLASS result failed: " << cudaGetErrorString(result) << std::endl;
		}

		if (result == cudaSuccess) {
			result = cudaMemcpy(host_reference.data(), tensor_ref_d.device_ref().data(), sizeof_c, cudaMemcpyDeviceToHost);
			if (result != cudaSuccess) {
				std::cerr << "Copying reference result failed: " << cudaGetErrorString(result) << std::endl;
			}
		}

		if (result == cudaSuccess &&
			!compare_results(host_cutlass, host_reference, problem_size.n(), test_case.level)) {
			result = cudaErrorUnknown;
		}
	}

	for (cudaStream_t stream : streams) {
		cudaStreamDestroy(stream);
	}

	return result;
}

template<typename CutlassGemm,
				 typename ElementInputA,
		 		 typename ElementInputB,
		 		 typename ElementOutput>
inline void run_gtest_case(TestCase const &test_case) {
	if (!device_supports_tests()) {
		GTEST_SKIP() << "Device does not support these sm80 Strassen-Winograd tests.";
	}

	cudaError_t result = run_case<CutlassGemm, ElementInputA, ElementInputB, ElementOutput>(test_case);
	if (result == cudaErrorNotSupported) {
		GTEST_SKIP() << "CUTLASS kernel does not support this problem shape.";
	}
	ASSERT_EQ(result, cudaSuccess) << cudaGetErrorString(result);
}

}  // namespace strassen_tests
