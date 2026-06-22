#include "cutlass/cutlass.h"
#include "cutlass/numeric_conversion.h"

#pragma once

struct one {
  __device__
  one(int seed = 0) {}

  __device__
  float operator()(int i, int j) {return 1;}
};

struct zero {
  __device__
  zero(int seed = 0) {}

  __device__
  float operator()(int i, int j) {return 0;}
};

struct setI {
  __device__
  setI(int seed = 0) {}

  __device__
  float operator()(int i, int j) {return i;}
};

struct setJ {
  __device__
  setJ(int seed = 0) {}

  __device__
  float operator()(int i, int j) {return j;}
};

struct divI {
  __device__
  divI(int seed = 0) {}

  __device__
  float operator()(int i, int j) {return i/64;}
};

template<uint DIV>
struct sum {
  __device__
  sum(int seed = 0) {}

  __device__
  float operator()(int i, int j) {return ((i + j)%DIV + (i + j)/DIV);}
};

template<int SCALE=10000>
struct random_float {
  int seed;

  __device__
  random_float(int seed_ = 0): seed(seed_) {}

  __device__
  float operator()(int i, int j) {
    unsigned int hash = static_cast<unsigned int>(i) * 0x1f123bb5u +
                        static_cast<unsigned int>(j) * 0x159a55e5u +
                        static_cast<unsigned int>(seed) * 0x2c1b3c6du +
                        0x9e3779b9u;

    hash ^= hash >> 16;
    hash *= 0x7feb352du;
    hash ^= hash >> 15;
    hash *= 0x846ca68bu;
    hash ^= hash >> 16;

    return float(hash%SCALE);
  }
};

/// Kernel to initialize a matrix with small integers.
template<typename F, typename T>
__global__ void InitializeMatrix_kernel(
  T *matrix,
  int rows,
  int columns,
  int seed = 0) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < rows && j < columns) {
    int offset = i * columns + j;

    // Generate arbitrary elements.
    int const k = 16807;
    int const m = 16;
    F f(seed);
    T value = T(f(i,j));//float(((offset + seed) * k % m) - m / 2);

    matrix[offset] = value;
  }
}

/// Simple function to initialize a matrix to arbitrary small integers.
template<typename F, typename T>
cudaError_t InitializeMatrix(T *matrix, int rows, int columns, int seed = 0) {

  dim3 block(16, 16);
  dim3 grid(
    (rows + block.x - 1) / block.x,
    (columns + block.y - 1) / block.y
  );

  InitializeMatrix_kernel<F><<< grid, block >>>(matrix, rows, columns, seed);

  return cudaGetLastError();
}

template<typename T>
__global__ void InitializeMatrix_kernelB(
  T *matrix,
  int rows,
  int columns,
  int seed = 0) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  if (i < rows && j < columns) {
    int offset = i * columns + j;

    // Generate arbitrary elements.
    int const k = 16807;
    int const m = 31;
    T value = ::sum<511>()(i,j);//sum()(i,j); //(i == j)? 1 : 0; //float(((offset + seed) * k % m) - m / 2);

    matrix[offset] = value;
  }
}

template<typename T>
cudaError_t InitializeMatrixB(T *matrix, int rows, int columns, int seed = 0) {

  dim3 block(16, 16);
  dim3 grid(
    (rows + block.x - 1) / block.x,
    (columns + block.y - 1) / block.y
  );

  InitializeMatrix_kernelB<T><<< grid, block >>>(matrix, rows, columns, seed);

  return cudaDeviceSynchronize();
}

/// Allocates device memory for a matrix then fills with arbitrary small integers.
template<typename F, typename T>
cudaError_t AllocateMatrix(T **matrix, int rows, int columns, int seed = 0) {
  cudaError_t result;

  size_t sizeof_matrix = sizeof(T) * rows * columns;
  printf("sizeof_matrix %ld\n", sizeof_matrix);
  // Allocate device memory.
  result = cudaMalloc(reinterpret_cast<void **>(matrix), sizeof_matrix);

  if (result != cudaSuccess) {
    std::cerr << "Failed to allocate matrix: "
      << cudaGetErrorString(result) << std::endl;
    return result;
  }

  // Clear the allocation.
  result = cudaMemset(*matrix, 0, sizeof_matrix);

  if (result != cudaSuccess) {
    std::cerr << "Failed to clear matrix device memory: "
      << cudaGetErrorString(result) << std::endl;
    return result;
  }

  // Initialize matrix elements to arbitrary small integers.
  result = InitializeMatrix<F>(*matrix, rows, columns, seed);
  result = cudaDeviceSynchronize();
  if (result != cudaSuccess) {
    std::cerr << "Failed to initialize matrix: "
      << cudaGetErrorString(result) << std::endl;
    return result;
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////////////////////////