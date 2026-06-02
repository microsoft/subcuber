#if GENCODE_ARCH == 900
#include "cuda/kernels/hopper/strassen_winograd/hopper_f32_sw_interleaved_presum.cuh"
#define F32_STRASSEN_WINOGRAD_PRESUM_TEST F32HopperSWPresumTest
using CutlassGemm = HopperF32SWInterleavedPresumLevel1;
#elif GENCODE_ARCH == 800
#ifndef F32_STRASSEN_WINOGRAD_PRESUM_TILE_SIZE
#define F32_STRASSEN_WINOGRAD_PRESUM_TILE_SIZE 256
#endif
#if F32_STRASSEN_WINOGRAD_PRESUM_TILE_SIZE == 128
#include "cuda/kernels/ampere/strassen_winograd/ampere_f32_sw_interleaved_presum_128x128.cuh"
#define F32_STRASSEN_WINOGRAD_PRESUM_TEST F32AmpereSWPresum128x128Test
using CutlassGemm = AmpereF32SWInterleavedPresumLevel1_128x128;
#elif F32_STRASSEN_WINOGRAD_PRESUM_TILE_SIZE == 256
#include "cuda/kernels/ampere/strassen_winograd/ampere_f32_sw_interleaved_presum.cuh"
#define F32_STRASSEN_WINOGRAD_PRESUM_TEST F32AmpereSWPresum256x128Test
using CutlassGemm = AmpereF32SWInterleavedPresumLevel1;
#else
#error invalid F32_STRASSEN_WINOGRAD_PRESUM_TILE_SIZE
#endif
#elif GENCODE_ARCH == 700
#include "cuda/kernels/volta/strassen_winograd/volta_f32_sw_interleaved_presum.cuh"
#define F32_STRASSEN_WINOGRAD_PRESUM_TEST F32VoltaSWPresumTest
using CutlassGemm = VoltaF32SWInterleavedPresumLevel1;
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
	};

	return cases;
}

class F32_STRASSEN_WINOGRAD_PRESUM_TEST : public testing::TestWithParam<strassen_tests::TestCase> {};

TEST_P(F32_STRASSEN_WINOGRAD_PRESUM_TEST, MatchesReference) {
	strassen_tests::run_gtest_case<CutlassGemm, float, float, float>(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
  ,
  F32_STRASSEN_WINOGRAD_PRESUM_TEST,
	testing::ValuesIn(test_cases()),
	strassen_tests::gtest_case_name);

#undef F32_STRASSEN_WINOGRAD_PRESUM_TEST
#undef F32_STRASSEN_WINOGRAD_PRESUM_TILE_SIZE

int main(int argc, char **argv) {
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
