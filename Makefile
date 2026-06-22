CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
CXX ?= g++
SPLIT_COMPILE ?= 1
PRESUM_LEVEL_2_SPLIT_COMPILE ?= 10

MAKEFILE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
CUDA_SRC_DIR := $(MAKEFILE_DIR)src/cuda
BUILD_ROOT ?= $(MAKEFILE_DIR)build
BUILD_DIR ?= $(BUILD_ROOT)/obj/kernel_runner
TARGET ?= $(BUILD_ROOT)/kernel_runner
NO_CUDA_DECL_TARGET ?= $(BUILD_ROOT)/kernel_runner_no_cuda_declarations
NO_CUDA_DECL_BUILD_DIR ?= $(BUILD_ROOT)/obj/no_cuda_declarations

CUTLASS := $(MAKEFILE_DIR)cutlass/include
CUTLASS_COMMON := $(MAKEFILE_DIR)cutlass/examples/common
UTIL_INCLUDE := $(MAKEFILE_DIR)cutlass/tools/util/include
STRASSEN_CUTLASS := $(MAKEFILE_DIR)include
SRC_ROOT := $(MAKEFILE_DIR)src
ROOT := $(MAKEFILE_DIR)
CUDA_CCCL_INCLUDE := $(CUDA_HOME)/include/cccl
CUDA_TARGET_INCLUDE := $(CUDA_HOME)/targets/x86_64-linux/include

NVCC_FLAGS := -std=c++20 -O3 --expt-relaxed-constexpr -DNDEBUG -lineinfo
NVCC_FLAGS += -Xptxas -v,-O3 -ftemplate-backtrace-limit=0
CXX_FLAGS := -std=c++20 -O3 -DNDEBUG -ftemplate-backtrace-limit=0
INCLUDES := -I $(STRASSEN_CUTLASS) -I $(CUTLASS) -I $(CUTLASS_COMMON) -I $(UTIL_INCLUDE) -I $(SRC_ROOT) -I $(ROOT)
CXX_INCLUDES := $(INCLUDES) -I $(CUDA_HOME)/include -I $(CUDA_CCCL_INCLUDE) -I $(CUDA_TARGET_INCLUDE)
CXX_LDFLAGS := -L$(CUDA_HOME)/lib64
LDLIBS := -lcublasLt -lcublas
CXX_LDLIBS := $(LDLIBS) -lcudart
SPLIT_COMPILE_FLAGS = --split-compile=$(if $(findstring presum_level_2,$<),$(PRESUM_LEVEL_2_SPLIT_COMPILE),$(SPLIT_COMPILE))

VOLTA_GENCODE := --generate-code=arch=compute_80,code=[compute_80,sm_80] -DGENCODE_ARCH=700
AMPERE_GENCODE := --generate-code=arch=compute_80,code=[compute_80,sm_80] -DGENCODE_ARCH=800
HOPPER_GENCODE := --generate-code=arch=compute_90a,code=[compute_90a,sm_90a] -DGENCODE_ARCH=900

RUNNER_SRC := kernel_runner.cpp cublas_runner.cpp cublaslt_runner.cpp

AMPERE_CUBIC_SRCS := \
	kernels/ampere/cubic/ampere_f16_cutlass_128x256.cu \
	kernels/ampere/cubic/ampere_f64_cutlass_64x64.cu \
	kernels/ampere/cubic/ampere_f64_cutlass_128x128.cu \
	kernels/ampere/cubic/ampere_f32_cutlass_128x128.cu \
	kernels/ampere/cubic/ampere_f32_cutlass_256x128.cu

HOPPER_CUBIC_SRCS := \
	kernels/hopper/cubic/hopper_f16_cutlass_128x128_pingpong.cu \
	kernels/hopper/cubic/hopper_f16_cutlass_128x256_cooperative.cu \
	kernels/hopper/cubic/hopper_f64_cutlass_128x64.cu \
	kernels/hopper/cubic/hopper_f64_cutlass_128x128.cu \
	kernels/hopper/cubic/hopper_f32_cutlass_128x128.cu \
	kernels/hopper/cubic/hopper_f32_cutlass_256x128.cu

VOLTA_CUBIC_SRCS := \
	kernels/volta/cubic/volta_f32_cutlass_128x128.cu \
	kernels/volta/cubic/volta_f32_cutlass_256x128.cu

AMPERE_V2_SRCS := kernels/ampere/strassen_winograd/ampere_f32_sw_interleaved_presum.cu \
	kernels/ampere/strassen_winograd/ampere_f32_sw_interleaved_presum_128x128.cu \
	kernels/ampere/strassen_winograd/ampere_f32_sw_interleaved_presum_level_2.cu \
	kernels/ampere/strassen_winograd/ampere_f32_sw_interleaved_presum_level_2_128x128.cu \
	kernels/ampere/strassen_winograd/ampere_f64_sw_interleaved_presum_64x64.cu \
	kernels/ampere/strassen_winograd/ampere_f64_sw_interleaved_presum_128x128.cu \
	kernels/ampere/strassen_winograd/ampere_f64_sw_interleaved_presum_level_2_64x64.cu \
	kernels/ampere/strassen_winograd/ampere_f64_sw_interleaved_presum_level_2_128x128.cu \
	kernels/ampere/strassen_winograd/ampere_f64_sw_fused_presum_128x128.cu \
	kernels/ampere/strassen_winograd/ampere_f32_sw_tile_64x128.cu \
	kernels/ampere/strassen_winograd/ampere_f32_sw_tile_128x128.cu \
	kernels/ampere/strassen_winograd/ampere_f32_sw_kernel_presum.cu \
	kernels/ampere/strassen_winograd/ampere_f32_sw_fused_presum.cu \
	kernels/ampere/strassen_winograd/ampere_f16_sw_interleaved_presum_max_fusion.cu \
	kernels/ampere/strassen_winograd/ampere_f16_sw_interleaved_presum_low_fusion.cu \
	kernels/ampere/strassen_winograd/ampere_f16_sw_interleaved_presum_level_2.cu \
	kernels/ampere/strassen_winograd/ampere_f16_sw_kernel_presum.cu

HOPPER_V2_SRCS := \
	kernels/hopper/strassen_winograd/hopper_f32_sw_tile.cu \
	kernels/hopper/strassen_winograd/hopper_f32_sw_interleaved_presum.cu \
	kernels/hopper/strassen_winograd/hopper_f32_sw_interleaved_presum_level_2.cu \
	kernels/hopper/strassen_winograd/hopper_f64_sw_interleaved_presum_128x64.cu \
	kernels/hopper/strassen_winograd/hopper_f64_sw_interleaved_presum_128x128.cu \
	kernels/hopper/strassen_winograd/hopper_f64_sw_interleaved_presum_level_2_128x64.cu \
	kernels/hopper/strassen_winograd/hopper_f64_sw_interleaved_presum_level_2_128x128.cu \
	kernels/hopper/strassen_winograd/hopper_f64_sw_fused_presum_128x128.cu \
	kernels/hopper/strassen_winograd/hopper_f32_sw_kernel_presum.cu \
	kernels/hopper/strassen_winograd/hopper_f32_sw_fused_presum.cu

HOPPER_V3_SRCS := \
	kernels/hopper/strassen_winograd/hopper_f16_sw_interleaved_presum_pingpong_max_fusion.cu \
	kernels/hopper/strassen_winograd/hopper_f16_sw_interleaved_presum_pingpong_low_fusion.cu

VOLTA_V2_SRCS := \
	kernels/volta/strassen_winograd/volta_f32_sw_tile.cu \
	kernels/volta/strassen_winograd/volta_f32_sw_interleaved_presum.cu \
	kernels/volta/strassen_winograd/volta_f32_sw_interleaved_presum_level_2.cu \
	kernels/volta/strassen_winograd/volta_f32_sw_kernel_presum.cu \
	kernels/volta/strassen_winograd/volta_f32_sw_fused_presum.cu

KERNEL_SRCS := $(AMPERE_V2_SRCS) $(AMPERE_CUBIC_SRCS) $(HOPPER_CUBIC_SRCS) $(VOLTA_CUBIC_SRCS) $(HOPPER_V2_SRCS) $(HOPPER_V3_SRCS) $(VOLTA_V2_SRCS)
RUNNER_OBJS := $(addprefix $(BUILD_DIR)/,$(RUNNER_SRC:.cpp=.o))
KERNEL_OBJS := $(addprefix $(BUILD_DIR)/,$(KERNEL_SRCS:.cu=.o))
OBJS := $(RUNNER_OBJS) $(KERNEL_OBJS)
NO_CUDA_DECL_OBJS := $(addprefix $(NO_CUDA_DECL_BUILD_DIR)/,$(RUNNER_SRC:.cpp=.o))
DEPS := $(OBJS:.o=.d)
NO_CUDA_DECL_DEPS := $(NO_CUDA_DECL_OBJS:.o=.d)

V2_FLAGS := -DCUTLASS_API_v2 -DTILE_SIZE_256 -DSPLIT_K=0
TILE_FLAGS := -DCUTLASS_API_v2 -DTHREADBLOCK -DTILE_SIZE_128 -DSPLIT_K=1
FUSED_FLAGS := -DCUTLASS_API_v2 -DFUSED_IN_SUM -DTILE_SIZE_256 -DSPLIT_K=0
PRESUM_FLAGS := -DCUTLASS_API_v2 -DPRESUM -DTILE_SIZE_256 -DSPLIT_K=0
KERNEL_PRESUM_FLAGS := -DCUTLASS_API_v2 -Dkernel_PRESUM -DTILE_SIZE_256 -DSPLIT_K=0
CUBIC_FLAGS := -DCUTLASS_API_v2 -DSPLIT_K=1
V3_FLAGS := -DCUTLASS_API_v3 -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1 -DCUTE_SM90_EXTENDED_MMA_SHAPES_ENABLED

.PHONY: all clean run disable_cuda_declarations no_cuda_declarations no-cuda-declarations

all: $(TARGET)

$(TARGET): $(OBJS)
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCC_FLAGS) $(OBJS) $(LDLIBS) -o $@

disable_cuda_declarations: $(NO_CUDA_DECL_TARGET)

no_cuda_declarations: disable_cuda_declarations

no-cuda-declarations: disable_cuda_declarations

$(NO_CUDA_DECL_TARGET): $(NO_CUDA_DECL_OBJS)
	@mkdir -p $(dir $@)
	$(CXX) $(CXX_FLAGS) $(NO_CUDA_DECL_OBJS) $(CXX_LDFLAGS) $(CXX_LDLIBS) -o $@

$(BUILD_DIR)/%.o: $(CUDA_SRC_DIR)/%.cpp $(CUDA_SRC_DIR)/kernel_runner_support.cuh
	@mkdir -p $(dir $@)
	$(CXX) $(CXX_FLAGS) $(CXX_INCLUDES) -MMD -MP -c $< -o $@

$(NO_CUDA_DECL_BUILD_DIR)/%.o: $(CUDA_SRC_DIR)/%.cpp $(CUDA_SRC_DIR)/kernel_runner_support.cuh
	@mkdir -p $(dir $@)
	$(CXX) $(CXX_FLAGS) -DSTRASSEN_DISABLE_CUDA_DECLARATIONS $(CXX_INCLUDES) -MMD -MP -c $< -o $@

$(BUILD_DIR)/kernels/%.o: $(CUDA_SRC_DIR)/kernels/%.cu $(CUDA_SRC_DIR)/kernel_runner_support.cuh
	@mkdir -p $(dir $@)
	$(NVCC) $(NVCC_FLAGS) $(SPLIT_COMPILE_FLAGS) $(INCLUDES) \
	  $(if $(findstring /hopper/,$<),$(HOPPER_GENCODE),$(if $(findstring /volta/,$<),$(VOLTA_GENCODE),$(AMPERE_GENCODE))) \
	    $(if $(findstring /hopper/cubic/hopper_f16_cutlass_,$<),$(V3_FLAGS), \
	    $(if $(findstring /cubic/,$<),$(CUBIC_FLAGS), \
	  $(if $(findstring hopper_f16_sw_interleaved_presum,$<),$(V3_FLAGS), \
	    $(if $(findstring _sw_tile,$<),$(TILE_FLAGS), \
	    $(if $(findstring _fused_presum.cu,$<),$(FUSED_FLAGS), \
	    $(if $(findstring _kernel_presum.cu,$<),$(KERNEL_PRESUM_FLAGS),$(PRESUM_FLAGS))))))) \
	  -MMD -MP -c $< -o $@

clean:
	rm -rf $(BUILD_ROOT) $(TARGET) $(NO_CUDA_DECL_TARGET)

-include $(DEPS)
-include $(NO_CUDA_DECL_DEPS)