#if GENCODE_ARCH == 900
#include "cuda/kernels/hopper/strassen_winograd/hopper_f32_sw_tile.cuh"
#define F32_STRASSEN_WINOGRAD_THREADBLOCK_TEST F32HopperSWThreadblockTest
using CutlassGemm = HopperF32SWTileGemm;
#elif GENCODE_ARCH == 800
#include "cuda/kernels/ampere/strassen_winograd/ampere_f32_sw_tile_64x128.cuh"
#define F32_STRASSEN_WINOGRAD_THREADBLOCK_TEST F32AmpereSWThreadblockTest
using CutlassGemm = AmpereF32SWTileGemm;
#elif GENCODE_ARCH == 700
#include "cuda/kernels/volta/strassen_winograd/volta_f32_sw_tile.cuh"
#define F32_STRASSEN_WINOGRAD_THREADBLOCK_TEST F32VoltaSWThreadblockTest
using CutlassGemm = VoltaF32SWTileGemm;
#else
#error invalid GENCODE_ARCH
#endif

#include "base_test.cuh"

class F32_STRASSEN_WINOGRAD_THREADBLOCK_TEST : public testing::TestWithParam<strassen_tests::TestCase> {};

static std::vector<strassen_tests::TestCase> test_cases() {
	std::vector<strassen_tests::TestCase> cases = {
		{{8192, 8192, 8192}, 1, 1, "8192x8192x8192 split_k=1"},
		{{2048, 2048, 2048}, 1, 1, "2048x2048x2048 split_k=1"},
		{{512, 512, 512}, 1, 1, "512x512x512 split_k=1"},
		{{6144, 2560, 2048}, 1, 1, "6144x2560x2048 split_k=1"},
		{{2048, 4096, 6144}, 1, 1, "2048x4096x6144 split_k=1"},
    {{512, 512, 512}, 2, 1, "512x512x512 split_k=2"}
	};

	return cases;
}

TEST_P(F32_STRASSEN_WINOGRAD_THREADBLOCK_TEST, MatchesReference) {
	strassen_tests::run_gtest_case<CutlassGemm, float, float, float>(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
  ,
  F32_STRASSEN_WINOGRAD_THREADBLOCK_TEST,
	testing::ValuesIn(test_cases()),
	strassen_tests::gtest_case_name);

#undef F32_STRASSEN_WINOGRAD_THREADBLOCK_TEST

int main(int argc, char **argv) {
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
