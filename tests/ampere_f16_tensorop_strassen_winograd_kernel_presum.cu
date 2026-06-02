#include "cuda/kernels/ampere/strassen_winograd/ampere_f16_sw_kernel_presum.cuh"
#include "base_test.cuh"

class F16AmpereSWGlobalPresumTest : public testing::TestWithParam<strassen_tests::TestCase> {};

static std::vector<strassen_tests::TestCase> test_cases() {
	std::vector<strassen_tests::TestCase> cases = {
		{{8192, 8192, 8192}, 1, 1, "8192x8192x8192 split_k=1"},
		{{2048, 2048, 2048}, 1, 1, "2048x2048x2048 split_k=1"},
		{{6144, 2560, 2048}, 1, 1, "6144x2560x2048 split_k=1"},
		{{2048, 4096, 6144}, 1, 1, "2048x4096x6144 split_k=1"},
	};

	return cases;
}

TEST_P(F16AmpereSWGlobalPresumTest, MatchesReference) {
	strassen_tests::run_gtest_case<AmpereF16SWKernelPresum, ElementInputA, ElementInputB, ElementOutput>(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    ,
	F16AmpereSWGlobalPresumTest,
	testing::ValuesIn(test_cases()),
	strassen_tests::gtest_case_name);

int main(int argc, char **argv) {
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
