CUTLASS := ../cutlass/include
CUTLASS_COMMON := ../cutlass/examples/common
UTIL_INCLUDE := ../cutlass/tools/util/include
STRASSEN_CUTLASS := ../include
STRASSEN_ROOT := ..
GTEST_ROOT := ../googletest/googletest
GTEST_INCLUDE := $(GTEST_ROOT)/include
GTEST_SRC_INCLUDE := $(GTEST_ROOT)
GTEST_SRC := $(GTEST_ROOT)/src/gtest-all.cc
GTEST_OBJECT := gtest-all.o

SPLIT_COMPILE ?= 10
NVCC ?= /usr/local/cuda/bin/nvcc
CXX ?= g++
NVCC_FLAGS ?= -std=c++20 -O3 -Xptxas -v --split-compile=$(SPLIT_COMPILE) --expt-relaxed-constexpr -DNDEBUG
VOLTA_GENCODE := --generate-code=arch=compute_80,code=[compute_80,sm_80]    -DGENCODE_ARCH=700
AMPERE_GENCODE := --generate-code=arch=compute_80,code=[compute_80,sm_80]    -DGENCODE_ARCH=800
HOPPER_GENCODE := --generate-code=arch=compute_90a,code=[compute_90a,sm_90a] -DGENCODE_ARCH=900
INCLUDES := -I $(STRASSEN_CUTLASS) -I $(CUTLASS) -I $(CUTLASS_COMMON) -I $(UTIL_INCLUDE) -I $(STRASSEN_ROOT) -I $(GTEST_INCLUDE) -I ../src/
GTEST_CXXFLAGS := -std=c++20 -O3 -pthread -I $(GTEST_INCLUDE) -I $(GTEST_SRC_INCLUDE)
AMPERE_F32_SW_TILE_SRC := ../src/cuda/kernels/ampere/strassen_winograd/ampere_f32_sw_tile_64x128.cu
HOPPER_F32_SW_TILE_SRC := ../src/cuda/kernels/hopper/strassen_winograd/hopper_f32_sw_tile.cu
VOLTA_F32_SW_TILE_SRC := ../src/cuda/kernels/volta/strassen_winograd/volta_f32_sw_tile.cu
AMPERE_F32_SW_INTERLEAVED_PRESUM_SRC := ../src/cuda/kernels/ampere/strassen_winograd/ampere_f32_sw_interleaved_presum.cu
AMPERE_F32_SW_INTERLEAVED_PRESUM_128X128_SRC := ../src/cuda/kernels/ampere/strassen_winograd/ampere_f32_sw_interleaved_presum_128x128.cu
AMPERE_F32_SW_INTERLEAVED_PRESUM_LEVEL_2_SRC := ../src/cuda/kernels/ampere/strassen_winograd/ampere_f32_sw_interleaved_presum_level_2.cu
AMPERE_F32_SW_INTERLEAVED_PRESUM_LEVEL_2_128X128_SRC := ../src/cuda/kernels/ampere/strassen_winograd/ampere_f32_sw_interleaved_presum_level_2_128x128.cu
AMPERE_F32_SW_KERNEL_PRESUM_SRC := ../src/cuda/kernels/ampere/strassen_winograd/ampere_f32_sw_kernel_presum.cu
AMPERE_F32_SW_FUSED_PRESUM_SRC := ../src/cuda/kernels/ampere/strassen_winograd/ampere_f32_sw_fused_presum.cu
AMPERE_F16_SW_INTERLEAVED_PRESUM_SRC := ../src/cuda/kernels/ampere/strassen_winograd/ampere_f16_sw_interleaved_presum.cu
AMPERE_F16_SW_INTERLEAVED_PRESUM_LEVEL_2_SRC := ../src/cuda/kernels/ampere/strassen_winograd/ampere_f16_sw_interleaved_presum_level_2.cu
AMPERE_F16_SW_KERNEL_PRESUM_SRC := ../src/cuda/kernels/ampere/strassen_winograd/ampere_f16_sw_kernel_presum.cu
HOPPER_F32_SW_INTERLEAVED_PRESUM_SRC := ../src/cuda/kernels/hopper/strassen_winograd/hopper_f32_sw_interleaved_presum.cu
HOPPER_F32_SW_INTERLEAVED_PRESUM_LEVEL_2_SRC := ../src/cuda/kernels/hopper/strassen_winograd/hopper_f32_sw_interleaved_presum_level_2.cu
HOPPER_F32_SW_KERNEL_PRESUM_SRC := ../src/cuda/kernels/hopper/strassen_winograd/hopper_f32_sw_kernel_presum.cu
HOPPER_F32_SW_FUSED_PRESUM_SRC := ../src/cuda/kernels/hopper/strassen_winograd/hopper_f32_sw_fused_presum.cu
HOPPER_F16_SW_INTERLEAVED_PRESUM_SRC := ../src/cuda/kernels/hopper/strassen_winograd/hopper_f16_sw_interleaved_presum.cu
VOLTA_F32_SW_INTERLEAVED_PRESUM_SRC := ../src/cuda/kernels/volta/strassen_winograd/volta_f32_sw_interleaved_presum.cu
VOLTA_F32_SW_INTERLEAVED_PRESUM_LEVEL_2_SRC := ../src/cuda/kernels/volta/strassen_winograd/volta_f32_sw_interleaved_presum_level_2.cu
VOLTA_F32_SW_KERNEL_PRESUM_SRC := ../src/cuda/kernels/volta/strassen_winograd/volta_f32_sw_kernel_presum.cu
VOLTA_F32_SW_FUSED_PRESUM_SRC := ../src/cuda/kernels/volta/strassen_winograd/volta_f32_sw_fused_presum.cu

VOLTA_TESTS := \
  test_volta_f32_strassen_winograd_tile \
  test_volta_f32_strassen_winograd_fused_presum \
  test_volta_f32_strassen_winograd_kernel_presum
#   test_volta_f32_strassen_winograd_presum \
# 	test_volta_f32_strassen_winograd_interleaved_presum_level_2

AMPERE_TESTS := \
  test_ampere_f32_strassen_winograd_tile \
  test_ampere_f32_strassen_winograd_fused_presum \
	test_ampere_f32_strassen_winograd_presum_256x128 \
	test_ampere_f32_strassen_winograd_presum_128x128 \
	test_ampere_f32_strassen_winograd_presum_level_2_256x128 \
	test_ampere_f32_strassen_winograd_presum_level_2_128x128 \
  test_ampere_f32_strassen_winograd_kernel_presum \
  test_ampere_f16_tensorop_strassen_winograd_kernel_presum \
  test_ampere_f16_tensorop_strassen_winograd_presum \
  test_ampere_f16_tensorop_strassen_winograd_presum_level_2

HOPPER_TESTS := test_hopper_f32_strassen_winograd_fused_presum \
	test_hopper_f32_strassen_winograd_presum \
	test_hopper_f32_strassen_winograd_presum_level_2 \
	test_hopper_f32_strassen_winograd_kernel_presum \
	test_hopper_f16_tensorop_strassen_winograd_presum

TARGETS := $(VOLTA_TESTS) $(AMPERE_TESTS) $(HOPPER_TESTS)

.PHONY: all clean test run-tests run-volta-tests run-ampere-tests run-hopper-tests $(addprefix run-,$(TARGETS))

all: $(TARGETS)

$(GTEST_OBJECT): $(GTEST_SRC)
	$(CXX) $(GTEST_CXXFLAGS) -c $< -o $@

test_volta_f32_strassen_winograd_tile: f32_strassen_winograd_tile.cu base_test.cuh $(VOLTA_F32_SW_TILE_SRC) $(GTEST_OBJECT)
	$(NVCC) $(NVCC_FLAGS) $< $(VOLTA_F32_SW_TILE_SRC) $(GTEST_OBJECT) $(INCLUDES) -DCUTLASS_API_v2 -DTHREADBLOCK -DTILE_SIZE_128 -DSPLIT_K=1 -o $@ $(VOLTA_GENCODE) -Xcompiler -pthread

test_volta_f32_strassen_winograd_fused_presum: f32_strassen_winograd_fused_presum.cu base_test.cuh $(VOLTA_F32_SW_FUSED_PRESUM_SRC) $(GTEST_OBJECT)
	$(NVCC) $(NVCC_FLAGS) $< $(VOLTA_F32_SW_FUSED_PRESUM_SRC) $(GTEST_OBJECT) $(INCLUDES) -DCUTLASS_API_v2 -DFUSED_IN_SUM -DTILE_SIZE_256 -DSPLIT_K=0 -o $@ $(VOLTA_GENCODE) -Xcompiler -pthread

# test_volta_f32_strassen_winograd_presum: f32_strassen_winograd_presum.cu base_test.cuh $(GTEST_OBJECT)
# 	$(NVCC) $(NVCC_FLAGS) $< $(GTEST_OBJECT) $(INCLUDES) -DCUTLASS_API_v2 -DPRESUM -DTILE_SIZE_256 -DSPLIT_K=0 -o $@ $(VOLTA_GENCODE) -Xcompiler -pthread

test_volta_f32_strassen_winograd_kernel_presum: f32_strassen_winograd_kernel_presum.cu base_test.cuh $(VOLTA_F32_SW_KERNEL_PRESUM_SRC) $(GTEST_OBJECT)
	$(NVCC) $(NVCC_FLAGS) $< $(VOLTA_F32_SW_KERNEL_PRESUM_SRC) $(GTEST_OBJECT) $(INCLUDES) -DCUTLASS_API_v2 -Dkernel_PRESUM -DTILE_SIZE_256 -DSPLIT_K=0 -o $@ $(VOLTA_GENCODE) -Xcompiler -pthread

# test_volta_f32_strassen_winograd_interleaved_presum_level_2: f32_strassen_winograd_presum_level_2.cu base_test.cuh $(GTEST_OBJECT)
# 	$(NVCC) $(NVCC_FLAGS) $< $(GTEST_OBJECT) $(INCLUDES) -DCUTLASS_API_v2 -Dkernel_PRESUM -DTILE_SIZE_256 -DSPLIT_K=0 -o $@ $(VOLTA_GENCODE) -Xcompiler -pthread

test_ampere_f32_strassen_winograd_tile: f32_strassen_winograd_tile.cu base_test.cuh $(AMPERE_F32_SW_TILE_SRC) $(GTEST_OBJECT)
	$(NVCC) $(NVCC_FLAGS) $< $(AMPERE_F32_SW_TILE_SRC) $(GTEST_OBJECT) $(INCLUDES) -DCUTLASS_API_v2 -DTHREADBLOCK -DTILE_SIZE_128 -DSPLIT_K=1 -o $@ $(AMPERE_GENCODE) -Xcompiler -pthread

test_ampere_f32_strassen_winograd_fused_presum: f32_strassen_winograd_fused_presum.cu base_test.cuh $(AMPERE_F32_SW_FUSED_PRESUM_SRC) $(GTEST_OBJECT)
	$(NVCC) $(NVCC_FLAGS) $< $(AMPERE_F32_SW_FUSED_PRESUM_SRC) $(GTEST_OBJECT) $(INCLUDES) -DCUTLASS_API_v2 -DFUSED_IN_SUM -DTILE_SIZE_256 -DSPLIT_K=0 -o $@ $(AMPERE_GENCODE) -Xcompiler -pthread

test_ampere_f32_strassen_winograd_presum_256x128: f32_strassen_winograd_presum.cu base_test.cuh $(AMPERE_F32_SW_INTERLEAVED_PRESUM_SRC) $(GTEST_OBJECT)
	$(NVCC) $(NVCC_FLAGS) $< $(AMPERE_F32_SW_INTERLEAVED_PRESUM_SRC) $(GTEST_OBJECT) $(INCLUDES) -DCUTLASS_API_v2 -DPRESUM -DF32_STRASSEN_WINOGRAD_PRESUM_TILE_SIZE=256 -DTILE_SIZE_256 -DSPLIT_K=0 -o $@ $(AMPERE_GENCODE) -Xcompiler -pthread

test_ampere_f32_strassen_winograd_presum_128x128: f32_strassen_winograd_presum.cu base_test.cuh $(AMPERE_F32_SW_INTERLEAVED_PRESUM_128X128_SRC) $(GTEST_OBJECT)
	$(NVCC) $(NVCC_FLAGS) $< $(AMPERE_F32_SW_INTERLEAVED_PRESUM_128X128_SRC) $(GTEST_OBJECT) $(INCLUDES) -DCUTLASS_API_v2 -DPRESUM -DF32_STRASSEN_WINOGRAD_PRESUM_TILE_SIZE=128 -DTILE_SIZE_128 -DSPLIT_K=0 -o $@ $(AMPERE_GENCODE) -Xcompiler -pthread

test_ampere_f32_strassen_winograd_presum_level_2_256x128: f32_strassen_winograd_presum_level_2.cu base_test.cuh $(AMPERE_F32_SW_INTERLEAVED_PRESUM_LEVEL_2_SRC) $(GTEST_OBJECT)
	$(NVCC) $(NVCC_FLAGS) $< $(AMPERE_F32_SW_INTERLEAVED_PRESUM_LEVEL_2_SRC) $(GTEST_OBJECT) $(INCLUDES) -DCUTLASS_API_v2 -DPRESUM -DF32_STRASSEN_WINOGRAD_PRESUM_LEVEL_2_TILE_SIZE=256 -DTILE_SIZE_256 -DSPLIT_K=0 -o $@ $(AMPERE_GENCODE) -Xcompiler -pthread

test_ampere_f32_strassen_winograd_presum_level_2_128x128: f32_strassen_winograd_presum_level_2.cu base_test.cuh $(AMPERE_F32_SW_INTERLEAVED_PRESUM_LEVEL_2_128X128_SRC) $(GTEST_OBJECT)
	$(NVCC) $(NVCC_FLAGS) $< $(AMPERE_F32_SW_INTERLEAVED_PRESUM_LEVEL_2_128X128_SRC) $(GTEST_OBJECT) $(INCLUDES) -DCUTLASS_API_v2 -DPRESUM -DF32_STRASSEN_WINOGRAD_PRESUM_LEVEL_2_TILE_SIZE=128 -DTILE_SIZE_128 -DSPLIT_K=0 -o $@ $(AMPERE_GENCODE) -Xcompiler -pthread

test_ampere_f32_strassen_winograd_kernel_presum: f32_strassen_winograd_kernel_presum.cu base_test.cuh $(AMPERE_F32_SW_KERNEL_PRESUM_SRC) $(GTEST_OBJECT)
	$(NVCC) $(NVCC_FLAGS) $< $(AMPERE_F32_SW_KERNEL_PRESUM_SRC) $(GTEST_OBJECT) $(INCLUDES) -DCUTLASS_API_v2 -Dkernel_PRESUM -DTILE_SIZE_256 -DSPLIT_K=0 -o $@ $(AMPERE_GENCODE) -Xcompiler -pthread

test_ampere_f16_tensorop_strassen_winograd_presum: ampere_f16_tensorop_strassen_winograd_presum.cu base_test.cuh $(AMPERE_F16_SW_INTERLEAVED_PRESUM_SRC) $(GTEST_OBJECT)
	$(NVCC) $(NVCC_FLAGS) $< $(AMPERE_F16_SW_INTERLEAVED_PRESUM_SRC) $(GTEST_OBJECT) $(INCLUDES) -DCUTLASS_API_v2 -DPRESUM -DTILE_SIZE_256 -DSPLIT_K=0 -o $@ $(AMPERE_GENCODE) -Xcompiler -pthread

test_ampere_f16_tensorop_strassen_winograd_kernel_presum: ampere_f16_tensorop_strassen_winograd_kernel_presum.cu base_test.cuh $(AMPERE_F16_SW_KERNEL_PRESUM_SRC) $(GTEST_OBJECT)
	$(NVCC) $(NVCC_FLAGS) $< $(AMPERE_F16_SW_KERNEL_PRESUM_SRC) $(GTEST_OBJECT) $(INCLUDES) -DCUTLASS_API_v2 -Dkernel_PRESUM -DTILE_SIZE_256 -DSPLIT_K=0 -o $@ $(AMPERE_GENCODE) -Xcompiler -pthread

test_ampere_f16_tensorop_strassen_winograd_presum_level_2: ampere_f16_tensorop_strassen_winograd_presum_level_2.cu base_test.cuh $(AMPERE_F16_SW_INTERLEAVED_PRESUM_LEVEL_2_SRC) $(GTEST_OBJECT)
	$(NVCC) $(NVCC_FLAGS) $< $(AMPERE_F16_SW_INTERLEAVED_PRESUM_LEVEL_2_SRC) $(GTEST_OBJECT) $(INCLUDES) -DCUTLASS_API_v2 -DPRESUM -DTILE_SIZE_256 -DSPLIT_K=0 -o $@ $(AMPERE_GENCODE) -Xcompiler -pthread

test_hopper_f32_strassen_winograd_tile: f32_strassen_winograd_tile.cu base_test.cuh $(HOPPER_F32_SW_TILE_SRC) $(GTEST_OBJECT)
	$(NVCC) $(NVCC_FLAGS) $< $(HOPPER_F32_SW_TILE_SRC) $(GTEST_OBJECT) $(INCLUDES) -DCUTLASS_API_v2 -DTHREADBLOCK -DTILE_SIZE_128 -DSPLIT_K=1 -o $@ $(HOPPER_GENCODE) -Xcompiler -pthread

test_hopper_f32_strassen_winograd_fused_presum: f32_strassen_winograd_fused_presum.cu base_test.cuh $(HOPPER_F32_SW_FUSED_PRESUM_SRC) $(GTEST_OBJECT)
	$(NVCC) $(NVCC_FLAGS) $< $(HOPPER_F32_SW_FUSED_PRESUM_SRC) $(GTEST_OBJECT) $(INCLUDES) -DCUTLASS_API_v2 -DFUSED_IN_SUM -DTILE_SIZE_256 -DSPLIT_K=0 -o $@ $(HOPPER_GENCODE) -Xcompiler -pthread

test_hopper_f32_strassen_winograd_presum: f32_strassen_winograd_presum.cu base_test.cuh $(HOPPER_F32_SW_INTERLEAVED_PRESUM_SRC) $(GTEST_OBJECT)
	$(NVCC) $(NVCC_FLAGS) $< $(HOPPER_F32_SW_INTERLEAVED_PRESUM_SRC) $(GTEST_OBJECT) $(INCLUDES) -DCUTLASS_API_v2 -DPRESUM -DTILE_SIZE_256 -DSPLIT_K=0 -o $@ $(HOPPER_GENCODE) -Xcompiler -pthread

test_hopper_f32_strassen_winograd_presum_level_2: f32_strassen_winograd_presum_level_2.cu base_test.cuh $(HOPPER_F32_SW_INTERLEAVED_PRESUM_LEVEL_2_SRC) $(GTEST_OBJECT)
	$(NVCC) $(NVCC_FLAGS) $< $(HOPPER_F32_SW_INTERLEAVED_PRESUM_LEVEL_2_SRC) $(GTEST_OBJECT) $(INCLUDES) -DCUTLASS_API_v2 -DPRESUM -DTILE_SIZE_256 -DSPLIT_K=0 -o $@ $(HOPPER_GENCODE) -Xcompiler -pthread

test_hopper_f32_strassen_winograd_kernel_presum: f32_strassen_winograd_kernel_presum.cu base_test.cuh $(HOPPER_F32_SW_KERNEL_PRESUM_SRC) $(GTEST_OBJECT)
	$(NVCC) $(NVCC_FLAGS) $< $(HOPPER_F32_SW_KERNEL_PRESUM_SRC) $(GTEST_OBJECT) $(INCLUDES) -DCUTLASS_API_v2 -Dkernel_PRESUM -DTILE_SIZE_256 -DSPLIT_K=0 -o $@ $(HOPPER_GENCODE) -Xcompiler -pthread

test_hopper_f16_tensorop_strassen_winograd_presum: hopper_f16_tensorop_strassen_winograd_presum.cu base_test.cuh $(HOPPER_F16_SW_INTERLEAVED_PRESUM_SRC) $(GTEST_OBJECT)
	$(NVCC) $(NVCC_FLAGS) $< $(HOPPER_F16_SW_INTERLEAVED_PRESUM_SRC) $(GTEST_OBJECT) $(INCLUDES) -DCUTLASS_API_v3 -o $@ $(HOPPER_GENCODE) -Xcompiler -pthread

run-volta-tests: $(VOLTA_TESTS)
	@set -e; \
	for target in $(VOLTA_TESTS); do \
		echo "== $$target =="; \
		./$$target; \
	done

run-ampere-tests: $(AMPERE_TESTS)
	@set -e; \
	for target in $(AMPERE_TESTS); do \
		echo "== $$target =="; \
		./$$target; \
	done

run-hopper-tests: $(HOPPER_TESTS)
	@set -e; \
	for target in $(HOPPER_TESTS); do \
		echo "== $$target =="; \
		./$$target; \
	done

run-tests: $(TARGETS)
	@status=0; \
	for target in $(TARGETS); do \
		echo "== $$target =="; \
		./$$target || status=$$?; \
	done; \
	exit $$status

$(addprefix run-,$(TARGETS)): run-%: %
	./$<

clean:
	rm -f $(TARGETS) $(GTEST_OBJECT)
