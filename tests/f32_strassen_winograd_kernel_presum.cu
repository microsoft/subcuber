#if GENCODE_ARCH == 900
#include "cuda/kernels/hopper/strassen_winograd/hopper_f32_sw_kernel_presum.cuh"
#define F32_STRASSEN_WINOGRAD_GLOBAL_PRESUM_TEST F32HopperSWGlobalPresumTest
using CutlassGemm = HopperF32SWKernelPresumLevel1;
#elif GENCODE_ARCH == 800
#include "cuda/kernels/ampere/strassen_winograd/ampere_f32_sw_kernel_presum.cuh"
#define F32_STRASSEN_WINOGRAD_GLOBAL_PRESUM_TEST F32AmpereSWGlobalPresumTest
using CutlassGemm = AmpereF32SWKernelPresumLevel1;
#elif GENCODE_ARCH == 700
#include "cuda/kernels/volta/strassen_winograd/volta_f32_sw_kernel_presum.cuh"
#define F32_STRASSEN_WINOGRAD_GLOBAL_PRESUM_TEST F32VoltaSWGlobalPresumTest
using CutlassGemm = VoltaF32SWKernelPresumLevel1;
#else
#error invalid GENCODE_ARCH
#endif

#include "base_test.cuh"

static std::vector<strassen_tests::TestCase> test_cases() {
	std::vector<strassen_tests::TestCase> cases = {
		{{8192, 8192, 8192}, 1, 1, "8192x8192x8192 split_k=1"},
		{{2048, 2048, 2048}, 1, 1, "2048x2048x2048 split_k=1"},
		{{6144, 2560, 2048}, 1, 1, "6144x2560x2048 split_k=1"},
		{{2048, 4096, 6144}, 1, 1, "2048x4096x6144 split_k=1"},
    {{512, 512, 512}, 1, 1, "512x512x512 split_k=1"},
	};

	return cases;
}

class F32_STRASSEN_WINOGRAD_GLOBAL_PRESUM_TEST : public testing::TestWithParam<strassen_tests::TestCase> {};

TEST_P(F32_STRASSEN_WINOGRAD_GLOBAL_PRESUM_TEST, MatchesReference) {
	strassen_tests::run_gtest_case<CutlassGemm, float, float, float>(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
  ,
  F32_STRASSEN_WINOGRAD_GLOBAL_PRESUM_TEST,
	testing::ValuesIn(test_cases()),
	strassen_tests::gtest_case_name);

#undef F32_STRASSEN_WINOGRAD_GLOBAL_PRESUM_TEST

int main(int argc, char **argv) {
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
