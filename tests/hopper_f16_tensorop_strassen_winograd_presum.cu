#include "cuda/kernels/hopper/strassen_winograd/hopper_f16_sw_interleaved_presum.cuh"

#include "base_test.cuh"

class F16HopperStrassenWinogradPresumTest_2x128_2x128_OptNo : public testing::TestWithParam<strassen_tests::TestCase> {};

static std::vector<strassen_tests::TestCase> test_cases_2x128_2x128_OptNo() {
	std::vector<strassen_tests::TestCase> cases = {
		{{8192, 8192, 8192}, 1, 1, "8192x8192x8192 split_k=1"},
		{{8192, 12288, 16384}, 1, 1, "8192x12288x16384 split_k=1"},
        {{8192, 8192, 16384}, 1, 1, "8192x8192x16384 split_k=1"},
        {{12288, 10240, 16384}, 1, 1, "12288x10240x16384 split_k=1"},
	};

	return cases;
}

TEST_P(F16HopperStrassenWinogradPresumTest_2x128_2x128_OptNo, MatchesReference) {
	strassen_tests::run_gtest_case<Gemm_PresumShape_2x128_2x128_OptNo, ElementA, ElementB, ElementC>(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    ,
	F16HopperStrassenWinogradPresumTest_2x128_2x128_OptNo,
	testing::ValuesIn(test_cases_2x128_2x128_OptNo()),
	strassen_tests::gtest_case_name);

class F16HopperStrassenWinogradPresumTest_2x128_2x128_Opt_0000 : public testing::TestWithParam<strassen_tests::TestCase> {};

static std::vector<strassen_tests::TestCase> test_cases_2x128_2x128_Opt_0000() {
	std::vector<strassen_tests::TestCase> cases = {
		{{9216, 9216, 9216}, 1, 1, "9216x9216x9216 split_k=1"},
		// {{2048, 2048, 2048}, 1, "2048x2048x2048 split_k=1"},
		{{12288, 12288, 8192}, 1, 1, "12288x12288x8192 split_k=1"},
		{{16384, 16384, 8192}, 1, 1, "16384x16384x8192 split_k=1"},
	};

	return cases;
}

TEST_P(F16HopperStrassenWinogradPresumTest_2x128_2x128_Opt_0000, MatchesReference) {
	strassen_tests::run_gtest_case<Gemm_PresumShape_2x128_2x128_Opt_0000, ElementA, ElementB, ElementC>(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    ,
	F16HopperStrassenWinogradPresumTest_2x128_2x128_Opt_0000,
	testing::ValuesIn(test_cases_2x128_2x128_Opt_0000()),
	strassen_tests::gtest_case_name);


class F16HopperStrassenWinogradPresumTest_4x128_4x128_OptNo : public testing::TestWithParam<strassen_tests::TestCase> {};

static std::vector<strassen_tests::TestCase> test_cases_4x128_4x128_OptNo() {
	std::vector<strassen_tests::TestCase> cases = {
		{{7168, 7168, 7168}, 1, 1, "7168x7168x7168 split_k=1"},
		// {{2048, 2048, 2048}, 1, "2048x2048x2048 split_k=1"},
		{{6144, 6144, 4096}, 1, 1, "6144x6144x4096 split_k=1"},
		{{8192, 4096, 8192}, 1, 1, "8192x4096x8192 split_k=1"},
	};

	return cases;
}

TEST_P(F16HopperStrassenWinogradPresumTest_4x128_4x128_OptNo, MatchesReference) {
	strassen_tests::run_gtest_case<Gemm_PresumShape_4x128_4x128_OptNo, ElementA, ElementB, ElementC>(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    ,
	F16HopperStrassenWinogradPresumTest_4x128_4x128_OptNo,
	testing::ValuesIn(test_cases_4x128_4x128_OptNo()),
	strassen_tests::gtest_case_name);

class F16HopperStrassenWinogradPresumTest_8x128_8x128_OptNo : public testing::TestWithParam<strassen_tests::TestCase> {};

static std::vector<strassen_tests::TestCase> test_cases_8x128_8x128_OptNo() {
	std::vector<strassen_tests::TestCase> cases = {
		{{3072, 3072, 3072}, 1, 1, "3072x3072x3072 split_k=1"},
		{{2048, 2048, 2048}, 1, 1, "2048x2048x2048 split_k=1"},
		{{6144, 6144, 2048}, 1, 1, "6144x6144x2048 split_k=1"},
		{{8192, 4096, 2048}, 1, 1, "8192x4096x2048 split_k=1"},
	};

	return cases;
}

TEST_P(F16HopperStrassenWinogradPresumTest_8x128_8x128_OptNo, MatchesReference) {
	strassen_tests::run_gtest_case<Gemm_PresumShape_8x128_8x128_OptNo, ElementA, ElementB, ElementC>(GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    ,
	F16HopperStrassenWinogradPresumTest_8x128_8x128_OptNo,
	testing::ValuesIn(test_cases_8x128_8x128_OptNo()),
	strassen_tests::gtest_case_name);


int main(int argc, char **argv) {
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
