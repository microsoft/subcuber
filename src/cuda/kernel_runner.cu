#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <regex>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

#include <cuda_runtime.h>

#include "cutlass/gemm/gemm.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/numeric_types.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cuda/kernel_runner_support.cuh"

using KernelRunFn = int (*)(KernelRunnerBuffers, int, int, int, int, int,
                            cudaStream_t *, int, int, float *);

#define DECLARE_KERNEL_RUN_FN(function_name) \
  extern "C" int function_name(KernelRunnerBuffers, int, int, int, int, int, \
                               cudaStream_t *, int, int, float *)

DECLARE_KERNEL_RUN_FN(run_ampere_f32_sw_tile);
DECLARE_KERNEL_RUN_FN(run_ampere_f32_sw_tile_128x128);
DECLARE_KERNEL_RUN_FN(run_ampere_f32_sw_interleaved_presum);
DECLARE_KERNEL_RUN_FN(run_ampere_f32_sw_interleaved_presum_128x128);
DECLARE_KERNEL_RUN_FN(run_ampere_f32_sw_interleaved_presum_level_2);
DECLARE_KERNEL_RUN_FN(run_ampere_f32_sw_interleaved_presum_level_2_128x128);
DECLARE_KERNEL_RUN_FN(run_ampere_f32_sw_kernel_presum);
DECLARE_KERNEL_RUN_FN(run_ampere_f32_sw_fused_presum);
DECLARE_KERNEL_RUN_FN(run_ampere_f16_cutlass_128x256);
DECLARE_KERNEL_RUN_FN(run_ampere_f32_cutlass_128x128);
DECLARE_KERNEL_RUN_FN(run_ampere_f32_cutlass_256x128);
DECLARE_KERNEL_RUN_FN(run_ampere_f16_sw_interleaved_presum_max_fusion);
DECLARE_KERNEL_RUN_FN(run_ampere_f16_sw_interleaved_presum_low_fusion);
DECLARE_KERNEL_RUN_FN(run_ampere_f16_sw_interleaved_presum_level_2);
DECLARE_KERNEL_RUN_FN(run_ampere_f16_sw_kernel_presum);
DECLARE_KERNEL_RUN_FN(run_hopper_f32_sw_tile);
DECLARE_KERNEL_RUN_FN(run_hopper_f32_sw_interleaved_presum);
DECLARE_KERNEL_RUN_FN(run_hopper_f32_sw_interleaved_presum_level_2);
DECLARE_KERNEL_RUN_FN(run_hopper_f32_sw_kernel_presum);
DECLARE_KERNEL_RUN_FN(run_hopper_f32_sw_fused_presum);
DECLARE_KERNEL_RUN_FN(run_hopper_f16_cutlass_128x128_pingpong);
DECLARE_KERNEL_RUN_FN(run_hopper_f16_cutlass_128x256_cooperative);
DECLARE_KERNEL_RUN_FN(run_hopper_f32_cutlass_128x128);
DECLARE_KERNEL_RUN_FN(run_hopper_f32_cutlass_256x128);
DECLARE_KERNEL_RUN_FN(run_hopper_f16_sw_interleaved_presum_pingpong_max_fusion_2x128_2x128_opt_no);
DECLARE_KERNEL_RUN_FN(run_hopper_f16_sw_interleaved_presum_pingpong_max_fusion_2x128_2x128_opt_0000);
DECLARE_KERNEL_RUN_FN(run_hopper_f16_sw_interleaved_presum_pingpong_max_fusion_4x128_4x128_opt_no);
DECLARE_KERNEL_RUN_FN(run_hopper_f16_sw_interleaved_presum_pingpong_max_fusion_8x128_8x128_opt_no);
DECLARE_KERNEL_RUN_FN(run_hopper_f16_sw_interleaved_presum_pingpong_low_fusion_2x128_2x128_opt_no);
DECLARE_KERNEL_RUN_FN(run_hopper_f16_sw_interleaved_presum_pingpong_low_fusion_2x128_2x128_opt_0000);
DECLARE_KERNEL_RUN_FN(run_hopper_f16_sw_interleaved_presum_pingpong_low_fusion_4x128_4x128_opt_no);
DECLARE_KERNEL_RUN_FN(run_hopper_f16_sw_interleaved_presum_pingpong_low_fusion_8x128_8x128_opt_no);
DECLARE_KERNEL_RUN_FN(run_volta_f32_cutlass_128x128);
DECLARE_KERNEL_RUN_FN(run_volta_f32_cutlass_256x128);
DECLARE_KERNEL_RUN_FN(run_volta_f32_sw_tile);
DECLARE_KERNEL_RUN_FN(run_volta_f32_sw_interleaved_presum);
DECLARE_KERNEL_RUN_FN(run_volta_f32_sw_interleaved_presum_level_2);
DECLARE_KERNEL_RUN_FN(run_volta_f32_sw_kernel_presum);
DECLARE_KERNEL_RUN_FN(run_volta_f32_sw_fused_presum);
DECLARE_KERNEL_RUN_FN(run_cublas_f32);
DECLARE_KERNEL_RUN_FN(run_cublas_f16);

#undef DECLARE_KERNEL_RUN_FN

struct KernelEntry {
  const char *name;
  const char *arch;
  const char *dtype;
  int strassen_level;
  KernelRunFn run;
};

static const KernelEntry kKernels[] = {
    {"cublas_f32", "volta", "f32", 0, run_cublas_f32},
    {"cublas_f16", "volta", "f16", 0, run_cublas_f16},
    {"cublas_f32", "ampere", "f32", 0, run_cublas_f32},
    {"cublas_f16", "ampere", "f16", 0, run_cublas_f16},
    {"ampere_f16_cutlass_128x256", "ampere", "f16", 0, run_ampere_f16_cutlass_128x256},
    {"ampere_f32_cutlass_128x128", "ampere", "f32", 0, run_ampere_f32_cutlass_128x128},
    {"ampere_f32_cutlass_256x128", "ampere", "f32", 0, run_ampere_f32_cutlass_256x128},
    {"cublas_f32", "hopper", "f32", 0, run_cublas_f32},
    {"cublas_f16", "hopper", "f16", 0, run_cublas_f16},
    {"hopper_f16_cutlass_128x128_pingpong", "hopper", "f16", 0, run_hopper_f16_cutlass_128x128_pingpong},
    {"hopper_f16_cutlass_128x256_cooperative", "hopper", "f16", 0, run_hopper_f16_cutlass_128x256_cooperative},
    {"hopper_f32_cutlass_128x128", "hopper", "f32", 0, run_hopper_f32_cutlass_128x128},
    {"hopper_f32_cutlass_256x128", "hopper", "f32", 0, run_hopper_f32_cutlass_256x128},
    {"ampere_f32_tile_64x128", "ampere", "f32", 1, run_ampere_f32_sw_tile},
    {"ampere_f32_tile_128x128", "ampere", "f32", 1, run_ampere_f32_sw_tile_128x128},
    {"ampere_f32_sw_interleaved_presum_256x128", "ampere", "f32", 1, run_ampere_f32_sw_interleaved_presum},
    {"ampere_f32_sw_interleaved_presum_128x128", "ampere", "f32", 1, run_ampere_f32_sw_interleaved_presum_128x128},
    {"ampere_f32_sw_interleaved_presum_level_2_128x128", "ampere", "f32", 2, run_ampere_f32_sw_interleaved_presum_level_2_128x128},
    {"ampere_f32_sw_interleaved_presum_level_2_256x128", "ampere", "f32", 2, run_ampere_f32_sw_interleaved_presum_level_2},
    {"ampere_f32_sw_kernel_presum_256x128", "ampere", "f32", 1, run_ampere_f32_sw_kernel_presum},
    {"ampere_f32_sw_fused_presum_256x128", "ampere", "f32", 1, run_ampere_f32_sw_fused_presum},
    {"ampere_f16_sw_interleaved_presum_max_fusion", "ampere", "f16", 1, run_ampere_f16_sw_interleaved_presum_max_fusion},
    {"ampere_f16_sw_interleaved_presum_low_fusion", "ampere", "f16", 1, run_ampere_f16_sw_interleaved_presum_low_fusion},
    {"ampere_f16_sw_interleaved_presum_level_2", "ampere", "f16", 2, run_ampere_f16_sw_interleaved_presum_level_2},
    {"ampere_f16_sw_kernel_presum", "ampere", "f16", 1, run_ampere_f16_sw_kernel_presum},
    {"hopper_f32_sw_interleaved_presum", "hopper", "f32", 1, run_hopper_f32_sw_interleaved_presum},
    {"hopper_f32_sw_interleaved_presum_level_2", "hopper", "f32", 2, run_hopper_f32_sw_interleaved_presum_level_2},
    {"hopper_f32_sw_tile", "hopper", "f32", 1, run_hopper_f32_sw_tile},
    {"hopper_f32_sw_kernel_presum", "hopper", "f32", 1, run_hopper_f32_sw_kernel_presum},
    {"hopper_f32_sw_fused_presum", "hopper", "f32", 1, run_hopper_f32_sw_fused_presum},
    {"hopper_f16_sw_interleaved_presum_pingpong_max_fusion_2x128_2x128_opt_no", "hopper", "f16", 1, run_hopper_f16_sw_interleaved_presum_pingpong_max_fusion_2x128_2x128_opt_no},
    {"hopper_f16_sw_interleaved_presum_pingpong_max_fusion_2x128_2x128_opt_0000", "hopper", "f16", 1, run_hopper_f16_sw_interleaved_presum_pingpong_max_fusion_2x128_2x128_opt_0000},
    {"hopper_f16_sw_interleaved_presum_pingpong_max_fusion_4x128_4x128_opt_no", "hopper", "f16", 1, run_hopper_f16_sw_interleaved_presum_pingpong_max_fusion_4x128_4x128_opt_no},
    {"hopper_f16_sw_interleaved_presum_pingpong_max_fusion_8x128_8x128_opt_no", "hopper", "f16", 1, run_hopper_f16_sw_interleaved_presum_pingpong_max_fusion_8x128_8x128_opt_no},
    {"hopper_f16_sw_interleaved_presum_pingpong_low_fusion_2x128_2x128_opt_no", "hopper", "f16", 1, run_hopper_f16_sw_interleaved_presum_pingpong_low_fusion_2x128_2x128_opt_no},
    {"hopper_f16_sw_interleaved_presum_pingpong_low_fusion_2x128_2x128_opt_0000", "hopper", "f16", 1, run_hopper_f16_sw_interleaved_presum_pingpong_low_fusion_2x128_2x128_opt_0000},
    {"hopper_f16_sw_interleaved_presum_pingpong_low_fusion_4x128_4x128_opt_no", "hopper", "f16", 1, run_hopper_f16_sw_interleaved_presum_pingpong_low_fusion_4x128_4x128_opt_no},
    {"hopper_f16_sw_interleaved_presum_pingpong_low_fusion_8x128_8x128_opt_no", "hopper", "f16", 1, run_hopper_f16_sw_interleaved_presum_pingpong_low_fusion_8x128_8x128_opt_no},
    {"volta_f32_cutlass_128x128", "volta", "f32", 0, run_volta_f32_cutlass_128x128},
    {"volta_f32_cutlass_256x128", "volta", "f32", 0, run_volta_f32_cutlass_256x128},
    {"volta_f32_sw_tile", "volta", "f32", 1, run_volta_f32_sw_tile},
    // {"volta_f32_sw_interleaved_presum", "volta", "f32", 1, run_volta_f32_sw_interleaved_presum},
    // {"volta_f32_sw_interleaved_presum_level_2", "volta", "f32", 2, run_volta_f32_sw_interleaved_presum_level_2},
    {"volta_f32_sw_kernel_presum", "volta", "f32", 1, run_volta_f32_sw_kernel_presum},
    {"volta_f32_sw_fused_presum", "volta", "f32", 1, run_volta_f32_sw_fused_presum},
};

static std::string lower(std::string value) {
  for (char &ch : value) {
    ch = char(std::tolower(static_cast<unsigned char>(ch)));
  }
  return value;
}

static std::string normalize_arch(std::string arch) {
  arch = lower(arch);
  if (arch == "sm70" || arch == "70" || arch == "compute_70") return "volta";
  if (arch == "sm80" || arch == "80" || arch == "compute_80") return "ampere";
  if (arch == "sm90" || arch == "sm90a" || arch == "90" || arch == "90a" || arch == "compute_90a") return "hopper";
  return arch;
}

static std::string normalize_dtype(std::string dtype) {
  dtype = lower(dtype);
  if (dtype == "float" || dtype == "fp32" || dtype == "s") return "f32";
  if (dtype == "half" || dtype == "fp16" || dtype == "h") return "f16";
  return dtype;
}

static bool parse_arg(int argc, char **argv, const char *name, std::string &value) {
  std::string prefix = std::string("--") + name + "=";
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg.rfind(prefix, 0) == 0) {
      value = arg.substr(prefix.size());
      return true;
    }
    if (arg == std::string("--") + name && i + 1 < argc) {
      value = argv[i + 1];
      return true;
    }
  }
  return false;
}

static bool get_required_int_arg(int argc, char **argv, const char *name, int &out) {
  std::string value;
  if (!parse_arg(argc, argv, name, value)) {
    return false;
  }
  try {
    size_t parsed_chars = 0;
    out = std::stoi(value, &parsed_chars);
    return parsed_chars == value.size();
  } catch (std::exception const &) {
    return false;
  }
}

static bool get_required_string_arg(int argc, char **argv, const char *name, std::string &out) {
  return parse_arg(argc, argv, name, out) && !out.empty();
}

static void usage(char const *program) {
  std::cerr << "Usage: " << program << " --m=<M> --n=<N> --k=<K> --dtype=f32|f16 --gpu_arch=volta|ampere|hopper --strassen_level=0|1|2|all --iterations=N --warmup=N --streams=N [--kernel_regex=REGEX]\n";
}

static bool tunes_split_k(KernelEntry const &kernel) {
  return (std::strcmp(kernel.dtype, "f32") == 0 &&
          (std::strstr(kernel.name, "_f32_sw_tile") != nullptr ||
           std::strstr(kernel.name, "_f32_tile_") != nullptr)) ||
         std::strstr(kernel.name, "_f32_cutlass_") != nullptr ||
         std::strstr(kernel.name, "ampere_f16_cutlass_") != nullptr;
}

static bool allows_split_k_greater_than_one(int m, int n) {
  constexpr int64_t kSplitKOutputElementLimit = int64_t(4) * 1024 * 4 * 1024;
  return int64_t(m) * int64_t(n) <= kSplitKOutputElementLimit;
}

template <typename Element>
int run_benchmark(std::vector<KernelEntry> const &candidates, int m, int n, int k,
                  std::string const &dtype, std::string const &arch,
                  std::string const &strassen_level_label,
                  int iterations, int warmup, int num_streams) {
  cutlass::gemm::GemmCoord problem_size(m, n, k);
  cutlass::HostTensor<Element, cutlass::layout::RowMajor> tensor_a(problem_size.mk());
  cutlass::HostTensor<Element, cutlass::layout::RowMajor> tensor_b(problem_size.kn());
  cutlass::HostTensor<Element, cutlass::layout::RowMajor> tensor_c(problem_size.mn());
  cutlass::HostTensor<Element, cutlass::layout::RowMajor> tensor_d(problem_size.mn());

  int range_end = std::is_same<Element, float>::value ? 16 : 4;
  int range_start = std::is_same<Element, float>::value ? -16 : -4;
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_a.host_view(), 1, Element(range_end), Element(range_start), 2);
  cutlass::reference::host::TensorFillRandomUniform(
      tensor_b.host_view(), 1, Element(range_end), Element(range_start), 2);
  cutlass::reference::host::TensorFill(tensor_c.host_view());
  cutlass::reference::host::TensorFill(tensor_d.host_view());

  tensor_a.sync_device();
  tensor_b.sync_device();
  tensor_c.sync_device();
  tensor_d.sync_device();

  KernelRunnerBuffers buffers{tensor_a.device_data(), tensor_b.device_data(),
                              nullptr, tensor_d.device_data()};
  num_streams = std::max(1, std::min(num_streams, kKernelRunnerMaxStreams));
  cudaStream_t runner_streams[kKernelRunnerMaxStreams];
  cudaError_t stream_err = kernel_runner_create_streams(runner_streams, num_streams);
  if (stream_err != cudaSuccess) {
    return static_cast<int>(stream_err);
  }

  std::string best_name;
  int best_split_k = 1;
  float best_ms = std::numeric_limits<float>::infinity();
  double best_gflops = 0.0;
  int runnable = 0;

  std::cout << "Running " << candidates.size() << " candidate kernels for m=" << m
            << " n=" << n << " k=" << k << " dtype=" << dtype
            << " gpu_arch=" << arch << " strassen_level=" << strassen_level_label
            << " streams=" << num_streams << "\n";
  std::cout << std::left << std::setw(52) << "kernel" << std::right
            << std::setw(10) << "split_k" << std::setw(14) << "avg_ms"
            << std::setw(16) << "gflops" << "\n";

  for (KernelEntry const &kernel : candidates) {
    bool tune_split_k = tunes_split_k(kernel);
    int first_split_k = 1;
    int last_split_k = tune_split_k && allows_split_k_greater_than_one(m, n) ? 4 : 1;
    float kernel_best_ms = std::numeric_limits<float>::infinity();
    double kernel_best_gflops = 0.0;
    int kernel_best_split_k = 0;
    int last_rc = 0;

    for (int split_k = first_split_k; split_k <= last_split_k; ++split_k) {
      cutlass::reference::host::TensorFill(tensor_d.host_view());
      tensor_d.sync_device();

      float avg_ms = 0.0f;
      int rc = kernel.run(buffers, m, n, k, warmup, iterations, runner_streams,
              num_streams, split_k, &avg_ms);
      last_rc = rc;
      std::this_thread::sleep_for(std::chrono::seconds((dtype != "f32") ? 10 : 5));
      if (rc != 0 || !std::isfinite(avg_ms)) {
        if (!tune_split_k) {
          std::cout << std::left << std::setw(52) << kernel.name << std::right
                    << std::setw(10) << split_k << std::setw(14) << "skipped"
                    << std::setw(16) << rc << "\n";
        }
        continue;
      }

      double gflops = (2.0 * double(m) * double(n) * double(k)) / (double(avg_ms) * 1.0e6);
      if (!tune_split_k) {
        ++runnable;
        std::cout << std::left << std::setw(52) << kernel.name << std::right
                  << std::setw(10) << split_k
                  << std::setw(14) << std::fixed << std::setprecision(4) << avg_ms
                  << std::setw(16) << std::fixed << std::setprecision(2) << gflops << "\n";
      }

      if (avg_ms < kernel_best_ms) {
        kernel_best_ms = avg_ms;
        kernel_best_gflops = gflops;
        kernel_best_split_k = split_k;
      }
    }

    if (tune_split_k) {
      if (kernel_best_split_k == 0) {
        std::cout << std::left << std::setw(52) << kernel.name << std::right
                  << std::setw(10) << "-" << std::setw(14) << "skipped"
                  << std::setw(16) << last_rc << "\n";
        continue;
      }

      ++runnable;
      std::cout << std::left << std::setw(52) << kernel.name << std::right
                << std::setw(10) << kernel_best_split_k
                << std::setw(14) << std::fixed << std::setprecision(4) << kernel_best_ms
                << std::setw(16) << std::fixed << std::setprecision(2) << kernel_best_gflops << "\n";
    }

    if (kernel_best_split_k != 0 && kernel_best_ms < best_ms) {
      best_ms = kernel_best_ms;
      best_split_k = kernel_best_split_k;
      best_gflops = kernel_best_gflops;
      best_name = kernel.name;
    }
  }

  if (runnable == 0) {
    std::cerr << "No registered kernel could run this problem.\n";
    kernel_runner_destroy_streams(runner_streams, num_streams);
    return 2;
  }

  std::cout << "Best kernel: " << best_name << " split_k=" << best_split_k
            << " avg_ms=" << std::fixed << std::setprecision(4)
            << best_ms << " gflops=" << std::fixed << std::setprecision(2) << best_gflops << "\n";
  kernel_runner_destroy_streams(runner_streams, num_streams);
  return 0;
}

int main(int argc, char **argv) {
  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
      usage(argv[0]);
      return 0;
    }
  }

  int m = 0;
  int n = 0;
  int k = 0;
  int iterations = 0;
  int warmup = 0;
  int streams = 0;
  std::string dtype_arg;
  std::string arch_arg;
  std::string strassen_level_arg;
  std::string kernel_regex_arg;

  bool valid_args = true;
  valid_args = get_required_int_arg(argc, argv, "m", m) && valid_args;
  valid_args = get_required_int_arg(argc, argv, "n", n) && valid_args;
  valid_args = get_required_int_arg(argc, argv, "k", k) && valid_args;
  valid_args = get_required_int_arg(argc, argv, "iterations", iterations) && valid_args;
  valid_args = get_required_int_arg(argc, argv, "warmup", warmup) && valid_args;
  valid_args = get_required_int_arg(argc, argv, "streams", streams) && valid_args;
  bool has_strassen_level = parse_arg(argc, argv, "strassen_level", strassen_level_arg) ||
                            parse_arg(argc, argv, "level", strassen_level_arg);
  valid_args = has_strassen_level && valid_args;
  valid_args = get_required_string_arg(argc, argv, "dtype", dtype_arg) && valid_args;
  valid_args = get_required_string_arg(argc, argv, "gpu_arch", arch_arg) && valid_args;
  bool has_kernel_regex = parse_arg(argc, argv, "kernel_regex", kernel_regex_arg);

  std::string dtype = normalize_dtype(dtype_arg);
  std::string arch = normalize_arch(arch_arg);
  std::string strassen_level_label = lower(strassen_level_arg);
  bool all_strassen_levels = strassen_level_label == "all";
  int strassen_level = -1;
  if (!all_strassen_levels && has_strassen_level) {
    try {
      size_t parsed_chars = 0;
      strassen_level = std::stoi(strassen_level_label, &parsed_chars);
      valid_args = parsed_chars == strassen_level_label.size() && valid_args;
    } catch (std::exception const &) {
      valid_args = false;
    }
  }
  int max_streams = all_strassen_levels ? 49 : (strassen_level == 0 ? 1 : (strassen_level == 2 ? 49 : 7));

  std::regex kernel_regex;
  if (has_kernel_regex) {
    try {
      kernel_regex = std::regex(kernel_regex_arg);
    } catch (std::regex_error const &err) {
      std::cerr << "Invalid --kernel_regex: " << err.what() << "\n";
      return 1;
    }
  }

  if (!valid_args || m <= 0 || n <= 0 || k <= 0 || iterations <= 0 || warmup < 0 ||
      streams <= 0 || streams > max_streams || (dtype != "f32" && dtype != "f16") ||
      (arch != "volta" && arch != "ampere" && arch != "hopper") ||
      (!all_strassen_levels && strassen_level != 0 && strassen_level != 1 && strassen_level != 2)) {
    usage(argv[0]);
    return 1;
  }

  std::vector<KernelEntry> candidates;
  for (KernelEntry const &kernel : kKernels) {
    if (kernel.dtype == dtype && kernel.arch == arch &&
        (all_strassen_levels || kernel.strassen_level == strassen_level) &&
        (!has_kernel_regex || std::regex_search(kernel.name, kernel_regex))) {
      candidates.push_back(kernel);
    }
  }

  if (candidates.empty()) {
    std::cerr << "No kernels registered for dtype=" << dtype << " gpu_arch=" << arch
              << " strassen_level=" << strassen_level_label;
    if (has_kernel_regex) {
      std::cerr << " kernel_regex=" << kernel_regex_arg;
    }
    std::cerr << "\n";
    return 1;
  }

  if (dtype == "f32") {
    return run_benchmark<float>(candidates, m, n, k, dtype, arch, strassen_level_label,
                                iterations, warmup, streams);
  }
  return run_benchmark<cutlass::half_t>(candidates, m, n, k, dtype, arch, strassen_level_label,
                                        iterations, warmup, streams);
}
