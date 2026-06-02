#include <cuda_fp16.h>
#include <cuda_fp8.h>
#define HAS_CUDA_FP8 1
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CHECK_CUDA(call)                                                         \
  do {                                                                           \
    cudaError_t status_ = (call);                                                \
    if (status_ != cudaSuccess) {                                                \
      std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                   cudaGetErrorString(status_));                                 \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                            \
  } while (0)

#define CHECK_CUBLAS(call)                                                       \
  do {                                                                           \
    cublasStatus_t status_ = (call);                                             \
    if (status_ != CUBLAS_STATUS_SUCCESS) {                                      \
      std::fprintf(stderr, "cuBLAS error at %s:%d: code=%d\n", __FILE__,      \
                   __LINE__, static_cast<int>(status_));                         \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                            \
  } while (0)

#define CHECK_CUBLASLT(call)                                                     \
  do {                                                                           \
    cublasStatus_t status_ = (call);                                             \
    if (status_ != CUBLAS_STATUS_SUCCESS) {                                      \
      std::fprintf(stderr, "cuBLASLt error at %s:%d: code=%d\n", __FILE__,    \
                   __LINE__, static_cast<int>(status_));                         \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                            \
  } while (0)

enum class InputType {
  kFp16,
  kFp32,
  kFp8E4M3,
};

using Fp8E4M3 = __nv_fp8_e4m3;
constexpr cudaDataType_t kFp8CudaDataType = CUDA_R_8F_E4M3;

int main(int argc, char** argv) {
  int m = 8192;
  int n = 8192;
  int k = 8192;
  int iters = 100;
  int warmup = 10;
  bool a_row_major = false;
  bool b_row_major = false;
  bool use_cublaslt = false;
  InputType input_type = InputType::kFp16;

  for (int i = 1; i < argc; ++i) {
    if (std::strncmp(argv[i], "--m=", 4) == 0) {
      m = std::atoi(argv[i] + 4);
    } else if (std::strncmp(argv[i], "--n=", 4) == 0) {
      n = std::atoi(argv[i] + 4);
    } else if (std::strncmp(argv[i], "--k=", 4) == 0) {
      k = std::atoi(argv[i] + 4);
    } else if (std::strncmp(argv[i], "--iters=", 8) == 0) {
      iters = std::atoi(argv[i] + 8);
    } else if (std::strncmp(argv[i], "--warmup=", 9) == 0) {
      warmup = std::atoi(argv[i] + 9);
    } else if (std::strncmp(argv[i], "--a_layout=", 11) == 0) {
      const char* value = argv[i] + 11;
      if (std::strcmp(value, "row") == 0) {
        a_row_major = true;
      } else if (std::strcmp(value, "col") == 0) {
        a_row_major = false;
      } else {
        std::fprintf(stderr, "Invalid --a_layout value: %s (expected row|col)\n", value);
        return EXIT_FAILURE;
      }
    } else if (std::strncmp(argv[i], "--b_layout=", 11) == 0) {
      const char* value = argv[i] + 11;
      if (std::strcmp(value, "row") == 0) {
        b_row_major = true;
      } else if (std::strcmp(value, "col") == 0) {
        b_row_major = false;
      } else {
        std::fprintf(stderr, "Invalid --b_layout value: %s (expected row|col)\n", value);
        return EXIT_FAILURE;
      }
    } else if (std::strncmp(argv[i], "--backend=", 10) == 0) {
      const char* value = argv[i] + 10;
      if (std::strcmp(value, "cublas") == 0) {
        use_cublaslt = false;
      } else if (std::strcmp(value, "cublaslt") == 0) {
        use_cublaslt = true;
      } else {
        std::fprintf(stderr, "Invalid --backend value: %s (expected cublas|cublaslt)\n", value);
        return EXIT_FAILURE;
      }
    } else if (std::strncmp(argv[i], "--dtype=", 8) == 0) {
      const char* value = argv[i] + 8;
      if (std::strcmp(value, "fp16") == 0) {
        input_type = InputType::kFp16;
      } else if (std::strcmp(value, "fp32") == 0) {
        input_type = InputType::kFp32;
      } else if (std::strcmp(value, "fp8") == 0) {
        input_type = InputType::kFp8E4M3;
      } else {
        std::fprintf(stderr, "Invalid --dtype value: %s (expected fp16|fp32|fp8)\n", value);
        return EXIT_FAILURE;
      }
    } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
      std::printf("Usage: %s [--m=<M>] [--n=<N>] [--k=<K>] [--iters=<I>] [--warmup=<W>] [--a_layout=row|col] [--b_layout=row|col] [--backend=cublas|cublaslt] [--dtype=fp16|fp32|fp8]\\n", argv[0]);
      return 0;
    }
  }

  bool use_fp32 = input_type == InputType::kFp32;
  bool use_fp8 = input_type == InputType::kFp8E4M3;
  bool output_fp32 = use_fp32 || use_fp8;
  bool effective_a_row_major = use_fp8 ? true : a_row_major;
  bool effective_b_row_major = use_fp8 ? false : b_row_major;

  if (use_fp8) {
    use_cublaslt = true;
  }

  std::printf("cuBLAS GEMM benchmark\n");
  std::printf("M=%d, N=%d, K=%d, warmup=%d, iters=%d\n", m, n, k, warmup, iters);
  std::printf("A layout=%s, B layout=%s\n", a_row_major ? "row" : "col", b_row_major ? "row" : "col");
  std::printf("Backend=%s\n", use_cublaslt ? "cublaslt" : "cublas");
  std::printf("Datatype=%s\n", use_fp8 ? "fp8-e4m3" : (use_fp32 ? "fp32" : "fp16"));
  if (use_fp8) {
    std::printf("FP8 internal layout=%s/%s (TN requirement on Hopper)\n",
                effective_a_row_major ? "row" : "col",
                effective_b_row_major ? "row" : "col");
  }

  cublasOperation_t op_a = effective_a_row_major ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t op_b = effective_b_row_major ? CUBLAS_OP_T : CUBLAS_OP_N;
  int lda = effective_a_row_major ? k : m;
  int ldb = effective_b_row_major ? n : k;

  size_t size_a = static_cast<size_t>(m) * k;
  size_t size_b = static_cast<size_t>(k) * n;
  size_t size_c = static_cast<size_t>(m) * n;

  std::vector<half> h_a_f16;
  std::vector<half> h_b_f16;
  std::vector<half> h_c_f16;
  std::vector<float> h_a_f32;
  std::vector<float> h_b_f32;
  std::vector<float> h_c_f32;
  std::vector<Fp8E4M3> h_a_fp8;
  std::vector<Fp8E4M3> h_b_fp8;

  std::mt19937 rng(std::random_device{}());
  std::uniform_int_distribution<int> dist(-4, 4);

  if (use_fp32) {
    h_a_f32.resize(size_a);
    h_b_f32.resize(size_b);
    h_c_f32.assign(size_c, 0.0f);
    for (size_t i = 0; i < size_a; ++i) {
      h_a_f32[i] = static_cast<float>(dist(rng));
    }
    for (size_t i = 0; i < size_b; ++i) {
      h_b_f32[i] = static_cast<float>(dist(rng));
    }
  } else if (use_fp8) {
    h_c_f32.assign(size_c, 0.0f);
    h_a_fp8.resize(size_a);
    h_b_fp8.resize(size_b);
    for (int row = 0; row < m; ++row) {
      for (int col = 0; col < k; ++col) {
        h_a_fp8[static_cast<size_t>(row) * k + col] = __nv_fp8_e4m3(static_cast<float>(dist(rng)));
      }
    }
    for (int row = 0; row < k; ++row) {
      for (int col = 0; col < n; ++col) {
        h_b_fp8[static_cast<size_t>(row) + static_cast<size_t>(col) * k] = __nv_fp8_e4m3(static_cast<float>(dist(rng)));
      }
    }
  } else {
    h_a_f16.resize(size_a);
    h_b_f16.resize(size_b);
    h_c_f16.assign(size_c, __float2half(0.0f));
    for (size_t i = 0; i < size_a; ++i) {
      h_a_f16[i] = __float2half(static_cast<float>(dist(rng)));
    }
    for (size_t i = 0; i < size_b; ++i) {
      h_b_f16[i] = __float2half(static_cast<float>(dist(rng)));
    }
  }

  cudaDataType_t data_type_a = use_fp32 ? CUDA_R_32F : (use_fp8 ? kFp8CudaDataType : CUDA_R_16F);
  cudaDataType_t data_type_b = data_type_a;
  cudaDataType_t data_type_c = output_fp32 ? CUDA_R_32F : CUDA_R_16F;
  cublasComputeType_t compute_type = (use_fp32 || use_fp8) ? CUBLAS_COMPUTE_32F : CUBLAS_COMPUTE_32F_FAST_16F;
  cublasGemmAlgo_t gemm_algo = use_fp32 ? CUBLAS_GEMM_DEFAULT : CUBLAS_GEMM_DEFAULT_TENSOR_OP;
  size_t element_size_a = use_fp32 ? sizeof(float) : (use_fp8 ? sizeof(Fp8E4M3) : sizeof(half));
  size_t element_size_b = element_size_a;
  size_t element_size_c = output_fp32 ? sizeof(float) : sizeof(half);

  void* d_a = nullptr;
  void* d_b = nullptr;
  void* d_c = nullptr;

  CHECK_CUDA(cudaMalloc(&d_a, size_a * element_size_a));
  CHECK_CUDA(cudaMalloc(&d_b, size_b * element_size_b));
  CHECK_CUDA(cudaMalloc(&d_c, size_c * element_size_c));

  if (use_fp32) {
    CHECK_CUDA(cudaMemcpy(d_a, h_a_f32.data(), size_a * element_size_a, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b_f32.data(), size_b * element_size_b, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_c, h_c_f32.data(), size_c * element_size_c, cudaMemcpyHostToDevice));
  } else if (use_fp8) {
    CHECK_CUDA(cudaMemcpy(d_a, h_a_fp8.data(), size_a * element_size_a, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b_fp8.data(), size_b * element_size_b, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_c, h_c_f32.data(), size_c * element_size_c, cudaMemcpyHostToDevice));
  } else {
    CHECK_CUDA(cudaMemcpy(d_a, h_a_f16.data(), size_a * element_size_a, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b_f16.data(), size_b * element_size_b, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_c, h_c_f16.data(), size_c * element_size_c, cudaMemcpyHostToDevice));
  }

  float alpha = 1.0f;
  float beta = 0.0f;

  cublasHandle_t cublas_handle = nullptr;
  cublasLtHandle_t cublaslt_handle = nullptr;
  cublasLtMatmulDesc_t matmul_desc = nullptr;
  cublasLtMatrixLayout_t layout_a = nullptr;
  cublasLtMatrixLayout_t layout_b = nullptr;
  cublasLtMatrixLayout_t layout_c = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;
  cublasLtMatmulHeuristicResult_t heuristic_result{};
  void* workspace = nullptr;
  float* scale_a = nullptr;
  float* scale_b = nullptr;
  size_t workspace_size = 32 * 1024 * 1024;

  if (use_cublaslt) {
    int layout_a_rows = use_fp8 ? k : m;
    int layout_a_cols = use_fp8 ? m : k;
    int layout_b_rows = k;
    int layout_b_cols = n;
    CHECK_CUBLASLT(cublasLtCreate(&cublaslt_handle));
    CHECK_CUBLASLT(cublasLtMatmulDescCreate(&matmul_desc, compute_type, CUDA_R_32F));
    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_a, sizeof(op_a)));
    CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_b, sizeof(op_b)));

    if (use_fp8) {
      float host_scale = 1.0f;
      cublasLtMatmulMatrixScale_t scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F;
      CHECK_CUDA(cudaMalloc(&scale_a, sizeof(float)));
      CHECK_CUDA(cudaMalloc(&scale_b, sizeof(float)));
      CHECK_CUDA(cudaMemcpy(scale_a, &host_scale, sizeof(float), cudaMemcpyHostToDevice));
      CHECK_CUDA(cudaMemcpy(scale_b, &host_scale, sizeof(float), cudaMemcpyHostToDevice));
      CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(matmul_desc,
                                                    CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
                                                    &scale_mode,
                                                    sizeof(scale_mode)));
      CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(matmul_desc,
                                                    CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
                                                    &scale_mode,
                                                    sizeof(scale_mode)));
      CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(matmul_desc,
                                                    CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                                    &scale_a,
                                                    sizeof(scale_a)));
      CHECK_CUBLASLT(cublasLtMatmulDescSetAttribute(matmul_desc,
                                                    CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                                    &scale_b,
                                                    sizeof(scale_b)));
    }

    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&layout_a, data_type_a, layout_a_rows, layout_a_cols, lda));
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&layout_b, data_type_b, layout_b_rows, layout_b_cols, ldb));
    CHECK_CUBLASLT(cublasLtMatrixLayoutCreate(&layout_c, data_type_c, m, n, m));

    cublasLtOrder_t order_a = use_fp8 ? CUBLASLT_ORDER_COL : (effective_a_row_major ? CUBLASLT_ORDER_ROW : CUBLASLT_ORDER_COL);
    cublasLtOrder_t order_b = use_fp8 ? CUBLASLT_ORDER_COL : (effective_b_row_major ? CUBLASLT_ORDER_ROW : CUBLASLT_ORDER_COL);
    cublasLtOrder_t order_c = CUBLASLT_ORDER_COL;
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(layout_a, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_a, sizeof(order_a)));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(layout_b, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_b, sizeof(order_b)));
    CHECK_CUBLASLT(cublasLtMatrixLayoutSetAttribute(layout_c, CUBLASLT_MATRIX_LAYOUT_ORDER, &order_c, sizeof(order_c)));

    CHECK_CUBLASLT(cublasLtMatmulPreferenceCreate(&preference));
    CHECK_CUDA(cudaMalloc(&workspace, workspace_size));
    CHECK_CUBLASLT(cublasLtMatmulPreferenceSetAttribute(preference,
                                                        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                                        &workspace_size,
                                                        sizeof(workspace_size)));
    int returned_results = 0;
    CHECK_CUBLASLT(cublasLtMatmulAlgoGetHeuristic(cublaslt_handle,
                                                  matmul_desc,
                                                  layout_a,
                                                  layout_b,
                                                  layout_c,
                                                  layout_c,
                                                  preference,
                                                  1,
                                                  &heuristic_result,
                                                  &returned_results));
    if (returned_results == 0) {
      std::fprintf(stderr, "No cuBLASLt heuristic result found for this configuration.\n");
      return EXIT_FAILURE;
    }
  } else {
    CHECK_CUBLAS(cublasCreate(&cublas_handle));
    CHECK_CUBLAS(cublasSetMathMode(cublas_handle, CUBLAS_TENSOR_OP_MATH));
  }

  auto run_gemm = [&]() {
    if (use_cublaslt) {
      CHECK_CUBLASLT(cublasLtMatmul(cublaslt_handle,
                                    matmul_desc,
                                    &alpha,
                                    d_a,
                                    layout_a,
                                    d_b,
                                    layout_b,
                                    &beta,
                                    d_c,
                                    layout_c,
                                    d_c,
                                    layout_c,
                                    &heuristic_result.algo,
                                    workspace,
                                    workspace_size,
                                    0));
    } else {
      CHECK_CUBLAS(cublasGemmEx(cublas_handle,
                                op_a,
                                op_b,
                                m,
                                n,
                                k,
                                &alpha,
                                d_a,
                                data_type_a,
                                lda,
                                d_b,
                                data_type_b,
                                ldb,
                                &beta,
                                d_c,
                                data_type_c,
                                m,
                                compute_type,
                                gemm_algo));
    }
  };

  for (int i = 0; i < warmup; ++i) {
    run_gemm();
  }
  CHECK_CUDA(cudaDeviceSynchronize());

  cudaEvent_t start, stop;
  CHECK_CUDA(cudaEventCreate(&start));
  CHECK_CUDA(cudaEventCreate(&stop));

  CHECK_CUDA(cudaEventRecord(start));
  for (int i = 0; i < iters; ++i) {
    run_gemm();
  }
  CHECK_CUDA(cudaEventRecord(stop));
  CHECK_CUDA(cudaEventSynchronize(stop));

  float elapsed_ms = 0.0f;
  CHECK_CUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
  float avg_ms = elapsed_ms / static_cast<float>(iters);

  double flops = 2.0 * static_cast<double>(m) * static_cast<double>(n) * static_cast<double>(k);
  double tflops = flops / (avg_ms * 1.0e-3) / 1.0e12;

  float c0 = 0.0f;
  if (output_fp32) {
    CHECK_CUDA(cudaMemcpy(h_c_f32.data(), d_c, size_c * element_size_c, cudaMemcpyDeviceToHost));
    c0 = h_c_f32[0];
  } else {
    CHECK_CUDA(cudaMemcpy(h_c_f16.data(), d_c, size_c * element_size_c, cudaMemcpyDeviceToHost));
    c0 = __half2float(h_c_f16[0]);
  }

  std::printf("Average runtime: %.3f ms\n", avg_ms);
  std::printf("Throughput: %.2f TFLOP/s\n", tflops);
  std::printf("C[0]=%.3f (sanity)\n", c0);

  CHECK_CUDA(cudaEventDestroy(start));
  CHECK_CUDA(cudaEventDestroy(stop));

  if (use_cublaslt) {
    CHECK_CUBLASLT(cublasLtMatmulPreferenceDestroy(preference));
    CHECK_CUBLASLT(cublasLtMatrixLayoutDestroy(layout_a));
    CHECK_CUBLASLT(cublasLtMatrixLayoutDestroy(layout_b));
    CHECK_CUBLASLT(cublasLtMatrixLayoutDestroy(layout_c));
    CHECK_CUBLASLT(cublasLtMatmulDescDestroy(matmul_desc));
    CHECK_CUBLASLT(cublasLtDestroy(cublaslt_handle));
    if (scale_a != nullptr) {
      CHECK_CUDA(cudaFree(scale_a));
    }
    if (scale_b != nullptr) {
      CHECK_CUDA(cudaFree(scale_b));
    }
    CHECK_CUDA(cudaFree(workspace));
  } else {
    CHECK_CUBLAS(cublasDestroy(cublas_handle));
  }

  CHECK_CUDA(cudaFree(d_a));
  CHECK_CUDA(cudaFree(d_b));
  CHECK_CUDA(cudaFree(d_c));

  return 0;
}
