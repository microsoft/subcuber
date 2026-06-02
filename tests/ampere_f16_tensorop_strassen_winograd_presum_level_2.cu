#include "cuda/kernels/ampere/strassen_winograd/ampere_f16_sw_interleaved_presum_level_2.cuh"
#include "base_test.cuh"

class F16AmpereSWPresumLevel2Test : public testing::TestWithParam<strassen_tests::TestCase> {};

static std::vector<strassen_tests::TestCase> test_cases() {
	std::vector<strassen_tests::TestCase> cases = {
		{{8192, 8192, 4096}, 1, 2, "8192x8192x4096 split_k=1"},
		{{4096, 4096, 4096}, 1, 2, "4096x4096x4096 split_k=1"},
		{{6144, 5120, 4096}, 1, 2, "6144x5120x4096 split_k=1"},
		{{7168, 8192, 6144}, 1, 2, "7168x8192x6144 split_k=1"},
	};

	return cases;
}

TEST_P(F16AmpereSWPresumLevel2Test, MatchesReference) {
	strassen_tests::run_gtest_case<AmpereF16SWInterleavedPresumLevel2, ElementInputA, ElementInputB, ElementOutput>(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    ,
	F16AmpereSWPresumLevel2Test,
	testing::ValuesIn(test_cases()),
	strassen_tests::gtest_case_name);

int main(int argc, char **argv) {
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
