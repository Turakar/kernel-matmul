#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <iostream>
#include <stdio.h>

#include "kernel_matmul.h"

#define CONSTANT_PI_F 3.141592741f
#define NUM_PARAMS 5

// #define PRINT_SIZE
// #define GPU_ASSERT

#define gpuErrchk(ans)                                                                             \
    { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

namespace {

template <int b_block_size, int m_block_size, int num_threads_x, int n_block_size>
__global__ void mosm_vecmul_cuda_kernel(
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits>
        x1, // (samples of task1)
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits>
        params1, // (NUM_PARAMS, batch)
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits>
        x2, // (samples of all tasks)
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits>
        params2, // (tasks, NUM_PARAMS, batch)
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> rhs, // (samples)
    const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits>
        start, // (row_blocks, tasks)
    const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits>
        end, // (row_blocks, tasks)
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits>
        out // (batch / b_block_size, samples)
) {
    const int b_offset = blockIdx.x * b_block_size;
    const int m_offset = blockIdx.y * m_block_size;
    const int m_size = x1.size(0);
    const int b_size = params1.size(1);
    const int num_tasks = params2.size(0);

    // Indexing 1: Threads spread across m
    static_assert(m_block_size % num_threads_x == 0,
                  "m_block_size must be divisible by num_threads");
    const int index1_base = threadIdx.x;
    const int index1_num_threads = num_threads_x;
    const int index1_per_thread = m_block_size / index1_num_threads;

    // Indexing 2: Threads spread across m and b
    static_assert(num_threads_x % b_block_size == 0,
                  "num_threads must be divisible by b_block_size");
    const int index2_m_base = threadIdx.x / b_block_size;
    const int index2_m_num_threads = num_threads_x / b_block_size;
    static_assert(m_block_size % index2_m_num_threads == 0,
                  "m_block_size must be divisible by index2_m_num_threads");
    const int index2_m_per_thread = m_block_size / index2_m_num_threads;
    const int index2_b_base = threadIdx.x % b_block_size;
    const int index2_b = b_offset + index2_b_base;

    // Indexing 3: Threads spread across n
    static_assert(n_block_size % num_threads_x == 0,
                  "n_block_size must be divisible by num_threads");
    const int index3_base = threadIdx.x;
    const int index3_num_threads = num_threads_x;
    const int index3_per_thread = n_block_size / index3_num_threads;

    // Shared memory buffer for x2 and rhs loading and output accumulation
    __shared__ std::array<float, n_block_size> x2_shm;
    __shared__ std::array<float, n_block_size> rhs_shm;
    __shared__ std::array<float, m_block_size> out_shm;
    for (int i = 0; i < index1_per_thread; i++) {
        out_shm[i * index1_num_threads + index1_base] = 0.0f;
    }

    // Load parameters from task1.
    const float sigma1 = index2_b < b_size ? params1[0][index2_b] : 1.0f;
    const float mu1 = index2_b < b_size ? params1[1][index2_b] : 1.0f;
    const float w1 = index2_b < b_size ? params1[2][index2_b] : 1.0f;
    const float theta1 = index2_b < b_size ? params1[3][index2_b] : 0.0f;
    const float phi1 = index2_b < b_size ? params1[4][index2_b] : 0.0f;

    // Load x1 to registers.
    std::array<float, index2_m_per_thread> x1_reg;
    for (int i = 0; i < index2_m_per_thread; i++) {
        const int m = m_offset + i * index2_m_num_threads + index2_m_base;
        x1_reg[i] = m < m_size ? x1[m] : 0.0f;
    }

    // Iterate over all tasks
    for (int task2 = 0; task2 < num_tasks; task2++) {

        // Load parameters from task2.
        const float sigma2 = index2_b < b_size ? params2[task2][0][index2_b] : 1.0f;
        const float mu2 = index2_b < b_size ? params2[task2][1][index2_b] : 1.0f;
        const float w2 = index2_b < b_size ? params2[task2][2][index2_b] : 1.0f;
        const float theta2 = index2_b < b_size ? params2[task2][3][index2_b] : 0.0f;
        const float phi2 = index2_b < b_size ? params2[task2][4][index2_b] : 0.0f;

        // Pre-compute some values we will need later.
        const float sigma_inv = 1 / (sigma1 + sigma2);
        const float sigma = 2.0f * sigma1 * sigma2 * sigma_inv;
        const float mu = (sigma1 * mu2 + sigma2 * mu1) * sigma_inv;
        const float w = w1 * w2 * exp(-0.25f * (mu1 - mu2) * (mu1 - mu2) * sigma_inv);
        const float theta = theta1 - theta2;
        const float phi = phi1 - phi2;
        const float alpha = w * sqrt(2.0f * CONSTANT_PI_F * sigma);

        // Iterate over n
        const int start_m = start[blockIdx.y][task2];
        const int end_m = end[blockIdx.y][task2];
        for (int n_base = start_m; n_base < end_m; n_base += n_block_size) {

            // Load x2 and rhs into shared memory
            for (int i = 0; i < index3_per_thread; i++) {
                const int n = i * index3_num_threads + index3_base;
                const int n_global = n + n_base;
                if (n_global < end_m) {
                    x2_shm[n] = x2[n_global];
                    rhs_shm[n] = rhs[n_global];
                } else {
                    x2_shm[n] = 0.0f;
                    rhs_shm[n] = 0.0f;
                }
            }
            __syncthreads();

            // Compute kernel matrix and perform reduction across n and b
            for (int n = 0; n < n_block_size; n++) {
                for (int i = 0; i < index2_m_per_thread; i++) {
                    const int m_local = i * index2_m_num_threads + index2_m_base;
                    const int m_global = m_offset + m_local;
                    // Start loading old value from shared memory for later summation
                    const float old_value = index2_b_base == 0 ? out_shm[m_local] : 0.0f;
                    // Compute kernel value * rhs
                    float value;
                    if (m_global < m_size) {
                        const float tau_theta = x1_reg[i] - x2_shm[n] + theta;
                        const float exp_term = exp(-0.5f * tau_theta * tau_theta * sigma);
                        const float cos_term = cos(tau_theta * mu + phi);
                        const float kernel_value = alpha * exp_term * cos_term;
                        value = kernel_value * rhs_shm[n];
                    } else {
                        value = 0.0f;
                    }
                    // Reduce kernel values over b_block_size
                    for (int offset = b_block_size / 2; offset > 0; offset /= 2) {
                        value += __shfl_down_sync(0xffffffff, value, offset, b_block_size);
                    }
                    // Write to shared memory
                    if (index2_b_base == 0) {
                        out_shm[m_local] = old_value + value;
                    }
                }
                __syncthreads();
            }
        }
    }

    // Write output to global memory
    for (int i = 0; i < index1_per_thread; i++) {
        const int m_local = i * index1_num_threads + index1_base;
        const int m_global = m_offset + m_local;
        if (m_global < m_size) {
            out[blockIdx.x][m_global] = out_shm[m_local];
        }
    }
}

template <int b_block_size, int m_block_size, int num_threads_x, int n_block_size, int k_size>
__global__ void mosm_matmul_cuda_kernel(
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits>
        x1, // (samples of task1)
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits>
        params1, // (NUM_PARAMS, batch)
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits>
        x2, // (samples of all tasks)
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits>
        params2, // (tasks, NUM_PARAMS, batch)
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> rhs, // (samples, k)
    const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits>
        start, // (row_blocks, tasks)
    const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits>
        end, // (row_blocks, tasks)
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits>
        out // (batch / b_block_size, samples, k)
) {
    const int b_offset = blockIdx.y * b_block_size;
    const int m_offset = blockIdx.x * m_block_size;
    const int m_size = x1.size(0);
    const int b_size = params1.size(1);
    const int num_tasks = params2.size(0);

    // We will buffer the rhs and the kernel matrix in shared memory.
    // The kernel matrix is stored in column-major order for coalescing.
    __shared__ std::array<std::array<float, m_block_size>, n_block_size> kernel_shm;
    __shared__ std::array<std::array<float, k_size>, n_block_size> rhs_shm;

    // Calculate indices for loading kernel matrix.
    static_assert(num_threads_x % b_block_size == 0,
                  "num_threads must be divisible by b_block_size");
    static_assert(32 % b_block_size == 0, "32 must be divisible by b_block_size");
    static_assert((b_block_size & (b_block_size - 1)) == 0, "b_block_size must be a power of 2");
    const int load_kernel_b_base = threadIdx.x % b_block_size;
    const int load_kernel_b = b_offset + load_kernel_b_base;
    const int load_kernel_m_base = threadIdx.x / b_block_size;
    const int load_kernel_m_threads = num_threads_x / b_block_size;
    const int load_kernel_m_per_thread = m_block_size / load_kernel_m_threads;

    // Calculate indices for loading rhs.
    const int load_rhs_size = k_size * n_block_size;
    const int load_rhs_steps = (load_rhs_size + num_threads_x - 1) / num_threads_x;

    // Calculate indices for matmul.
    static_assert(k_size % b_block_size == 0, "k_size must be divisible by b_block_size");
    const int matmul_k_base = threadIdx.x % b_block_size;
    const int matmul_k_threads = b_block_size;
    const int matmul_k_per_thread = k_size / matmul_k_threads;
    const int matmul_m_base = threadIdx.x / b_block_size;
    const int matmul_m_threads = num_threads_x / b_block_size;
    const int matmul_m_per_thread = m_block_size / matmul_m_threads;

    // During matmul, each thread accumulates values for the following entries:
    // m: m_offset + matmul_m_base + i * matmul_m_threads
    // k: matmul_k_base + j * matmul_k_threads
    std::array<std::array<float, matmul_k_per_thread>, matmul_m_per_thread> accumulator;
    for (int i = 0; i < matmul_m_per_thread; i++) {
        for (int j = 0; j < matmul_k_per_thread; j++) {
            accumulator[i][j] = 0;
        }
    }

    // Load parameters from task1.
    const float sigma1 = load_kernel_b < b_size ? params1[0][load_kernel_b] : 1.0f;
    const float mu1 = load_kernel_b < b_size ? params1[1][load_kernel_b] : 1.0f;
    const float w1 = load_kernel_b < b_size ? params1[2][load_kernel_b] : 1.0f;
    const float theta1 = load_kernel_b < b_size ? params1[3][load_kernel_b] : 0.0f;
    const float phi1 = load_kernel_b < b_size ? params1[4][load_kernel_b] : 0.0f;

    // Load x1 to registers.
    std::array<float, load_kernel_m_per_thread> x1_reg;
    for (int i = 0; i < load_kernel_m_per_thread; i++) {
        const int m = m_offset + load_kernel_m_base + i * load_kernel_m_threads;
        x1_reg[i] = m < m_size ? x1[m] : 0.0f;
    }

    // Outer-most loop: Iterate over tasks.
    for (int task2 = 0; task2 < num_tasks; task2++) {

        // Load parameters from task2.
        const float sigma2 = load_kernel_b < b_size ? params2[task2][0][load_kernel_b] : 1.0f;
        const float mu2 = load_kernel_b < b_size ? params2[task2][1][load_kernel_b] : 1.0f;
        const float w2 = load_kernel_b < b_size ? params2[task2][2][load_kernel_b] : 1.0f;
        const float theta2 = load_kernel_b < b_size ? params2[task2][3][load_kernel_b] : 0.0f;
        const float phi2 = load_kernel_b < b_size ? params2[task2][4][load_kernel_b] : 0.0f;

        // Pre-compute some values we will need later.
        const float sigma_inv = 1 / (sigma1 + sigma2);
        const float sigma = 2.0f * sigma1 * sigma2 * sigma_inv;
        const float mu = (sigma1 * mu2 + sigma2 * mu1) * sigma_inv;
        const float w = w1 * w2 * exp(-0.25f * (mu1 - mu2) * (mu1 - mu2) * sigma_inv);
        const float theta = theta1 - theta2;
        const float phi = phi1 - phi2;
        const float alpha = w * sqrt(2.0f * CONSTANT_PI_F * sigma);

        // Iterate over n in blocks of size n_block_size.
        const int start_m = start[blockIdx.x][task2];
        const int end_m = end[blockIdx.x][task2];
        for (int n_base = start_m; n_base < end_m; n_base += n_block_size) {

            // Interleaved load of rhs and kernel matrix:

            // Start loading rhs to registers.
            std::array<float, load_rhs_steps> rhs_buffer;
            for (int i = 0; i < load_rhs_steps; i++) {
                const int flat_i = i * num_threads_x + threadIdx.x;
                const int k = flat_i % k_size;
                const int n = flat_i / k_size;
                const int n_global = n_base + n;
                if (n < n_block_size && n_global < end_m) {
                    rhs_buffer[i] = rhs[n_global][k];
                } else {
                    rhs_buffer[i] = 0;
                }
            }

            // Load kernel matrix to shm
            // For kernel matrix we additionally need a sum reduction along b_dim.
            // We use warp-based shuffle functions for this.
            for (int i = 0; i < n_block_size; i++) {
                const int n = n_base + i;
                const float x2_n = n < end_m ? x2[n] : 0.0f;
                for (int j = 0; j < load_kernel_m_per_thread; j++) {
                    const int m = m_offset + load_kernel_m_base + j * load_kernel_m_threads;
                    // Calculate kernel value for load_kernel_b
                    float kernel_value;
                    if (n < end_m && m < m_size && load_kernel_b < b_size) {
                        const float tau = x1_reg[j] - x2_n + theta;
                        const float exp_term = exp(-0.5f * tau * tau * sigma);
                        const float cos_term = cos(tau * mu + phi);
                        kernel_value = alpha * exp_term * cos_term;
                    } else {
                        kernel_value = 0;
                    }
                    // Reduce kernel values over b_block_size
                    for (int offset = b_block_size / 2; offset > 0; offset /= 2) {
                        kernel_value +=
                            __shfl_down_sync(0xffffffff, kernel_value, offset, b_block_size);
                    }
                    // Store kernel value to shm
                    if (load_kernel_b_base == 0) {
                        kernel_shm[i][m - m_offset] = kernel_value;
                    }
                }
            }

            // Store rhs to shm
            for (int i = 0; i < load_rhs_steps; i++) {
                const int flat_i = i * num_threads_x + threadIdx.x;
                const int k = flat_i % k_size;
                const int n = flat_i / k_size;
                if (n < n_block_size) {
                    rhs_shm[n][k] = rhs_buffer[i];
                }
            }

            __syncthreads();

            // Perform matmul:
            // Here, all inner loops are unrolled for ILP.
            for (int i = 0; i < n_block_size; i++) {
                // Load rhs and kernel matrix from shared memory into registers.
                std::array<float, matmul_k_per_thread> rhs_reg;
                for (int k = 0; k < matmul_k_per_thread; k++) {
                    rhs_reg[k] = rhs_shm[i][k * matmul_k_threads + matmul_k_base];
                }
                std::array<float, matmul_m_per_thread> kernel_reg;
                for (int j = 0; j < matmul_m_per_thread; j++) {
                    kernel_reg[j] = kernel_shm[i][j * matmul_m_threads + matmul_m_base];
                }

                // Caculate outer product of rhs and kernel matrix.
                for (int k = 0; k < matmul_k_per_thread; k++) {
                    for (int j = 0; j < matmul_m_per_thread; j++) {
                        accumulator[j][k] += rhs_reg[k] * kernel_reg[j];
                    }
                }
            }

            __syncthreads();
        }
    }

    // Write output to global memory.
    for (int i = 0; i < matmul_m_per_thread; i++) {
        const int m = m_offset + matmul_m_base + i * matmul_m_threads;
        for (int j = 0; j < matmul_k_per_thread; j++) {
            const int k = matmul_k_base + j * matmul_k_threads;
            if (m < m_size) {
                out[blockIdx.y][m][k] = accumulator[i][j];
            }
        }
    }
}

template <int matmul_k_threads, int m_block_size, int num_threads_x, int n_block_size, int k_size>
__global__ void mosm_matmul_bwd_cuda_kernel(
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits>
        x1, // (samples of task1)
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits>
        params1, // (NUM_PARAMS, batch)
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits>
        x2, // (samples of all tasks)
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits>
        params2, // (tasks, NUM_PARAMS, batch)
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> rhs, // (samples, k)
    const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits>
        start, // (row_blocks, tasks)
    const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits>
        end, // (row_blocks, tasks)
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits>
        out_grad, // (samples of task 1, k)
    torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits>
        params1_grad, // (NUM_PARAMS, batch, gridDim.x, blockDim.x)
    torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits>
        params2_grad // (tasks, NUM_PARAMS, batch, ...)
) {
    // This kernel cannot sum over the batch dimension in advance, because we require the gradient
    // per batch element. As such, we do not have threads over multiple batch entries inside one
    // block.

    const int b = blockIdx.y;
    const int m_offset = blockIdx.x * m_block_size;
    const int m_size = x1.size(0);
    const int num_tasks = params2.size(0);

    // We will buffer the rhs and the kernel matrix gradients in shared memory.
    // The kernel matrix gradients are stored in column-major order for coalescing.
    __shared__ std::array<std::array<std::array<float, m_block_size>, n_block_size>, NUM_PARAMS>
        grads1_shm;
    __shared__ std::array<std::array<std::array<float, m_block_size>, n_block_size>, NUM_PARAMS>
        grads2_shm;
    __shared__ std::array<std::array<float, k_size>, n_block_size> rhs_shm;

    // Calculate indices for loading kernel matrix.
    static_assert(m_block_size % num_threads_x == 0,
                  "m_block_size must be divisible by num_threads");
    const int load_kernel_m_base = threadIdx.x;
    const int load_kernel_m_threads = num_threads_x;
    const int load_kernel_m_per_thread = m_block_size / num_threads_x;

    // Calculate indices for loading rhs.
    const int load_rhs_size = k_size * n_block_size;
    const int load_rhs_steps = (load_rhs_size + num_threads_x - 1) / num_threads_x;

    // Calculate indices for matmul.
    static_assert(num_threads_x % matmul_k_threads == 0,
                  "num_threads must be divisible by matmul_k_threads");
    const int matmul_m_threads = num_threads_x / matmul_k_threads;
    static_assert(m_block_size % matmul_m_threads == 0,
                  "m_block_size must be divisible by matmul_m_threads");
    const int matmul_m_per_thread = m_block_size / matmul_m_threads;
    const int matmul_m_base = threadIdx.x / matmul_k_threads;
    static_assert(k_size % matmul_k_threads == 0, "k_size must be divisible by matmul_k_threads");
    const int matmul_k_per_thread = k_size / matmul_k_threads;
    const int matmul_k_base = threadIdx.x % matmul_k_threads;

    // We can cache the entries of the output gradient in registers.
    // This allows us to reduce across m, k inside one thread immediately.
    std::array<std::array<float, matmul_k_per_thread>, matmul_m_per_thread> out_grad_reg;
    for (int i = 0; i < matmul_m_per_thread; i++) {
        for (int j = 0; j < matmul_k_per_thread; j++) {
            const int m = m_offset + matmul_m_base + i * matmul_m_threads;
            const int k = matmul_k_base + j * matmul_k_threads;
            if (m < m_size) {
                out_grad_reg[i][j] = out_grad[m][k];
            } else {
                out_grad_reg[i][j] = 0;
            }
        }
    }

    // Load parameters of task1.
    const float sigma1 = params1[0][b];
    const float mu1 = params1[1][b];
    const float w1 = params1[2][b];
    const float theta1 = params1[3][b];
    const float phi1 = params1[4][b];

    // And initialize accumulator for gradients of params1.
    std::array<float, NUM_PARAMS> accumulator1 = {};

    // Load x1 into registers.
    std::array<float, load_kernel_m_per_thread> x1_reg;
    for (int i = 0; i < load_kernel_m_per_thread; i++) {
        const int m = load_kernel_m_base + i * load_kernel_m_threads;
        x1_reg[i] = m < m_size ? x1[m] : 0.0f;
    }

    // Outer-most loop: Iterate over tasks.
    for (int task2 = 0; task2 < num_tasks; task2++) {

        // Load parameters from task2.
        const float sigma2 = params2[task2][0][b];
        const float mu2 = params2[task2][1][b];
        const float w2 = params2[task2][2][b];
        const float theta2 = params2[task2][3][b];
        const float phi2 = params2[task2][4][b];

        // And initialize accumulator for gradients of params2.
        std::array<float, NUM_PARAMS> accumulator2 = {};

        // Pre-compute some values we will need later.
        const float sigma_inv = 1 / (sigma1 + sigma2);
        const float mu_diff = mu1 - mu2;
        const float mu_diff_sq = mu_diff * mu_diff;
        const float sigma = 2.0f * sigma1 * sigma2 * sigma_inv;
        const float mu = (sigma1 * mu2 + sigma2 * mu1) * sigma_inv;
        const float w = w1 * w2 * exp(-0.25f * (mu1 - mu2) * (mu1 - mu2) * sigma_inv);
        const float theta = theta1 - theta2;
        const float phi = phi1 - phi2;
        const float sqrt_term = sqrt(2.0f * CONSTANT_PI_F * sigma);
        const float alpha = w * sqrt_term;

        // Iterate over n in blocks of size n_block_size.
        const int start_m = start[blockIdx.x][task2];
        const int end_m = end[blockIdx.x][task2];
        for (int n_base = start_m; n_base < end_m; n_base += n_block_size) {

            // Interleaved load of rhs and kernel matrix:

            // Start loading rhs to registers.
            std::array<float, load_rhs_steps> rhs_buffer;
            for (int i = 0; i < load_rhs_steps; i++) {
                const int flat_i = i * num_threads_x + threadIdx.x;
                const int k = flat_i % k_size;
                const int n = flat_i / k_size;
                const int n_global = n_base + n;
                if (n < n_block_size && n_global < end_m) {
                    rhs_buffer[i] = rhs[n_global][k];
                } else {
                    rhs_buffer[i] = 0;
                }
            }

            // Load kernel matrix gradients to shm.
            for (int i = 0; i < n_block_size; i++) {
                const int n = n_base + i;
                const float x2_n = n < end_m ? x2[n] : 0.0f;
                for (int j = 0; j < load_kernel_m_per_thread; j++) {
                    const int m = m_offset + load_kernel_m_base + j * load_kernel_m_threads;
                    // Calculate kernel value for load_kernel_b
                    std::array<float, NUM_PARAMS> grads1_reg;
                    std::array<float, NUM_PARAMS> grads2_reg;
                    if (n < end_m && m < m_size) {
                        // Forward pass
                        const float tau_theta = x1_reg[j] - x2_n + theta;
                        const float exp_term = exp(-0.5f * tau_theta * tau_theta * sigma);
                        const float cos_term = cos(tau_theta * mu + phi);
                        const float kernel_value = alpha * exp_term * cos_term;

                        // Backward pass
                        const float sigma1_sq = sigma1 * sigma1;
                        const float sigma2_sq = sigma2 * sigma2;
                        const float sin_term = sin(tau_theta * mu + phi);
                        const float sigma_grad_factor_a = exp_term * sigma_inv * sigma_inv;
                        const float sigma_grad_factor_b = cos_term * w;
                        const float sigma_grad_sum_a = 2.0f * CONSTANT_PI_F / sqrt_term;
                        const float sigma_grad_sum_b = sqrt_term * mu_diff_sq / 4.0f;
                        const float sigma_grad_sum_c = alpha * cos_term * tau_theta * tau_theta;
                        const float sigma_grad_sum_d = alpha * sin_term * tau_theta * mu_diff;
                        const float mu_grad_factor = alpha * exp_term * sigma_inv;
                        const float mu_grad_sum_a = 0.5f * cos_term * mu_diff;
                        const float mu_grad_sum_b = sin_term * tau_theta;
                        grads1_reg[0] = sigma_grad_factor_a *
                                        (sigma_grad_factor_b *
                                             (sigma_grad_sum_a * sigma2_sq + sigma_grad_sum_b) -
                                         sigma_grad_sum_c * sigma2_sq + sigma_grad_sum_d * sigma2);
                        grads2_reg[0] = sigma_grad_factor_a *
                                        (sigma_grad_factor_b *
                                             (sigma_grad_sum_a * sigma1_sq + sigma_grad_sum_b) -
                                         sigma_grad_sum_c * sigma1_sq - sigma_grad_sum_d * sigma1);
                        grads1_reg[1] = -mu_grad_factor * (mu_grad_sum_a + mu_grad_sum_b * sigma2);
                        grads2_reg[1] = -mu_grad_factor * (-mu_grad_sum_a + mu_grad_sum_b * sigma1);
                        grads1_reg[2] = kernel_value / w1;
                        grads2_reg[2] = kernel_value / w2;
                        grads2_reg[3] =
                            alpha * exp_term * (cos_term * tau_theta * sigma + sin_term * mu);
                        grads1_reg[3] = -grads2_reg[3];
                        grads2_reg[4] = alpha * exp_term * sin_term;
                        grads1_reg[4] = -grads2_reg[4];
                    } else {
                        for (int p = 0; p < NUM_PARAMS; p++) {
                            grads1_reg[p] = 0;
                            grads2_reg[p] = 0;
                        }
                    }
                    // Store kernel value to shm.
                    for (int p = 0; p < NUM_PARAMS; p++) {
                        grads1_shm[p][i][m - m_offset] = grads1_reg[p];
                        grads2_shm[p][i][m - m_offset] = grads2_reg[p];
                    }
                }
            }

            // Store rhs to shm.
            for (int i = 0; i < load_rhs_steps; i++) {
                const int flat_i = i * num_threads_x + threadIdx.x;
                const int k = flat_i % k_size;
                const int n = flat_i / k_size;
                if (n < n_block_size) {
                    rhs_shm[n][k] = rhs_buffer[i];
                }
            }

            __syncthreads();

            // Perform matmul:
            // Here, all inner loops may be unrolled for ILP.
            for (int i = 0; i < n_block_size; i++) {
                std::array<float, matmul_k_per_thread> rhs_reg;
                for (int k = 0; k < matmul_k_per_thread; k++) {
                    rhs_reg[k] = rhs_shm[i][k * matmul_k_threads + matmul_k_base];
                }
                for (int p = 0; p < NUM_PARAMS; p++) {
                    std::array<float, matmul_m_per_thread> grads1_reg;
                    for (int j = 0; j < matmul_m_per_thread; j++) {
                        grads1_reg[j] = grads1_shm[p][i][j * matmul_m_threads + matmul_m_base];
                    }
                    for (int k = 0; k < matmul_k_per_thread; k++) {
                        for (int j = 0; j < matmul_m_per_thread; j++) {
                            accumulator1[p] += rhs_reg[k] * grads1_reg[j] * out_grad_reg[j][k];
                        }
                    }
                    std::array<float, matmul_m_per_thread> grads2_reg;
                    for (int j = 0; j < matmul_m_per_thread; j++) {
                        grads2_reg[j] = grads2_shm[p][i][j * matmul_m_threads + matmul_m_base];
                    }
                    for (int k = 0; k < matmul_k_per_thread; k++) {
                        for (int j = 0; j < matmul_m_per_thread; j++) {
                            accumulator2[p] += rhs_reg[k] * grads2_reg[j] * out_grad_reg[j][k];
                        }
                    }
                }
            }

            __syncthreads();
        }

        // Write gradients for params2 to global memory.
        for (int p = 0; p < NUM_PARAMS; p++) {
            params2_grad[task2][p][b][blockIdx.x][threadIdx.x] = accumulator2[p];
        }
    }

    // Write gradients for params1 to global memory.
    for (int p = 0; p < NUM_PARAMS; p++) {
        params1_grad[p][b][blockIdx.x][threadIdx.x] = accumulator1[p];
    }
}

template <int block_size, int num_threads_split>
__global__ void mosm_bilinear_derivative_cuda_kernel(
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits>
        x1, // (samples of task1)
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits>
        params1, // (NUM_PARAMS, batch)
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits>
        x2, // (samples of all tasks)
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits>
        params2, // (tasks, NUM_PARAMS, batch)
    const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits>
        start, // (row_blocks, tasks)
    const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits>
        end, // (row_blocks, tasks)
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits>
        left_vectors, // (k, samples of task 1)
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits>
        right_vectors, // (k, samples of task 2)
    torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits>
        out1, // (tasks2, NUM_PARAMS, batch, grid.x, threads)
    torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits>
        out2 // (tasks2, NUM_PARAMS, batch, grid.x, threads)
) {
    // Index calculations
    const int num_threads = num_threads_split * num_threads_split;
    static_assert(block_size % num_threads_split == 0,
                  "block_size must be divisible by num_threads_split");
    static_assert(num_threads % block_size == 0, "num_threads must be divisible by block_size");
    const int load_size = num_threads / block_size;
    const int load_offset = threadIdx.x / block_size;
    const int load_index = threadIdx.x % block_size;
    const int per_thread_split = block_size / num_threads_split;
    const int split_index1 = threadIdx.x / num_threads_split;
    const int split_index2 = threadIdx.x % num_threads_split;
    const int task2 = blockIdx.z;
    const int b = blockIdx.y;
    const int m_offset = blockIdx.x * block_size;
    const int num_tasks = params2.size(0);
    const int k_size = left_vectors.size(0);
    const int m_size = x1.size(0);

    // Shared memory buffers:
    __shared__ std::array<std::array<float, block_size>, load_size> left_vectors_shm;
    __shared__ std::array<std::array<float, block_size>, load_size> right_vectors_shm;

    // Accumulator for gradients
    std::array<float, NUM_PARAMS> accumulator1 = {};
    std::array<float, NUM_PARAMS> accumulator2 = {};

    // Load x1 to registers.
    std::array<float, per_thread_split> x1_reg;
    for (int i = 0; i < per_thread_split; i++) {
        const int m = m_offset + i * num_threads_split + split_index1;
        x1_reg[i] = m < m_size ? x1[m] : 0.0f;
    }

    // Load parameters.
    const float sigma1 = params1[0][b];
    const float mu1 = params1[1][b];
    const float w1 = params1[2][b];
    const float theta1 = params1[3][b];
    const float phi1 = params1[4][b];
    const float sigma2 = params2[task2][0][b];
    const float mu2 = params2[task2][1][b];
    const float w2 = params2[task2][2][b];
    const float theta2 = params2[task2][3][b];
    const float phi2 = params2[task2][4][b];

    // Pre-compute some values we will need later.
    const float sigma_inv = 1 / (sigma1 + sigma2);
    const float mu_diff = mu1 - mu2;
    const float mu_diff_sq = mu_diff * mu_diff;
    const float sigma = 2.0f * sigma1 * sigma2 * sigma_inv;
    const float mu = (sigma1 * mu2 + sigma2 * mu1) * sigma_inv;
    const float w = w1 * w2 * exp(-0.25f * (mu1 - mu2) * (mu1 - mu2) * sigma_inv);
    const float theta = theta1 - theta2;
    const float phi = phi1 - phi2;
    const float sqrt_term = sqrt(2.0f * CONSTANT_PI_F * sigma);
    const float alpha = w * sqrt_term;
    const float sigma1_sq = sigma1 * sigma1;
    const float sigma2_sq = sigma2 * sigma2;

    // Iterate over n in blocks of size block_size
    const int start_m = start[blockIdx.x][task2];
    const int end_m = end[blockIdx.x][task2];
    for (int n_base = start_m; n_base < end_m; n_base += block_size) {

        // Compute \sum_d U_{d,i} V_{d,j} for all i, j in this block.
        std::array<std::array<float, per_thread_split>, per_thread_split> coefficients = {};
        for (int k_base = 0; k_base < k_size; k_base += load_size) {
            // Load left and right vectors to shared memory.
            const int m_global = m_offset + load_index;
            const int k = k_base + load_offset;
            if (k < k_size && m_global < m_size) {
                left_vectors_shm[load_offset][load_index] = left_vectors[k][m_global];
            } else {
                left_vectors_shm[load_offset][load_index] = 0;
            }
            const int n_global = n_base + load_index;
            if (k < k_size && n_global < end_m) {
                right_vectors_shm[load_offset][load_index] = right_vectors[k][n_global];
            } else {
                right_vectors_shm[load_offset][load_index] = 0;
            }

            __syncthreads();

            for (int step = 0; step < load_size; step++) {
                // Load left and right vectors from shared memory to registers.
                std::array<float, per_thread_split> left_vectors_reg;
                for (int i = 0; i < per_thread_split; i++) {
                    left_vectors_reg[i] =
                        left_vectors_shm[step][i * num_threads_split + split_index1];
                }
                std::array<float, per_thread_split> right_vectors_reg;
                for (int i = 0; i < per_thread_split; i++) {
                    right_vectors_reg[i] =
                        right_vectors_shm[step][i * num_threads_split + split_index2];
                }

                // Outer product and accumulate.
                for (int i = 0; i < per_thread_split; i++) {
                    for (int j = 0; j < per_thread_split; j++) {
                        coefficients[i][j] += left_vectors_reg[i] * right_vectors_reg[j];
                    }
                }
            }

            __syncthreads();
        }

        // Compute gradients for params1 and params2.
        for (int j = 0; j < per_thread_split; j++) {
            const int n = n_base + j * num_threads_split + split_index2;
            const float x2_n = n < end_m ? x2[n] : 0.0f;
            for (int i = 0; i < per_thread_split; i++) {
                // Forward pass
                const float tau_theta = x1_reg[i] - x2_n + theta;
                const float exp_term = exp(-0.5f * tau_theta * tau_theta * sigma);
                const float cos_term = cos(tau_theta * mu + phi);
                const float kernel_value = alpha * exp_term * cos_term;

                // Backward pass
                const float sin_term = sin(tau_theta * mu + phi);
                const float sigma_grad_factor_a = exp_term * sigma_inv * sigma_inv;
                const float sigma_grad_factor_b = cos_term * w;
                const float sigma_grad_sum_a = 2.0f * CONSTANT_PI_F / sqrt_term;
                const float sigma_grad_sum_b = sqrt_term * mu_diff_sq / 4.0f;
                const float sigma_grad_sum_c = alpha * cos_term * tau_theta * tau_theta;
                const float sigma_grad_sum_d = alpha * sin_term * tau_theta * mu_diff;
                const float mu_grad_factor = alpha * exp_term * sigma_inv;
                const float mu_grad_sum_a = 0.5f * cos_term * mu_diff;
                const float mu_grad_sum_b = sin_term * tau_theta;
                const float sigma1_grad =
                    sigma_grad_factor_a *
                    (sigma_grad_factor_b * (sigma_grad_sum_a * sigma2_sq + sigma_grad_sum_b) -
                     sigma_grad_sum_c * sigma2_sq + sigma_grad_sum_d * sigma2);
                const float sigma2_grad =
                    sigma_grad_factor_a *
                    (sigma_grad_factor_b * (sigma_grad_sum_a * sigma1_sq + sigma_grad_sum_b) -
                     sigma_grad_sum_c * sigma1_sq - sigma_grad_sum_d * sigma1);
                const float mu1_grad = -mu_grad_factor * (mu_grad_sum_a + mu_grad_sum_b * sigma2);
                const float mu2_grad = -mu_grad_factor * (-mu_grad_sum_a + mu_grad_sum_b * sigma1);
                const float w1_grad = kernel_value / w1;
                const float w2_grad = kernel_value / w2;
                const float theta2_grad =
                    alpha * exp_term * (cos_term * tau_theta * sigma + sin_term * mu);
                const float theta1_grad = -theta2_grad;
                const float phi2_grad = alpha * exp_term * sin_term;
                const float phi1_grad = -phi2_grad;

                // Accumulate gradients.
                const float coef = coefficients[i][j];
                accumulator1[0] += coef * sigma1_grad;
                accumulator1[1] += coef * mu1_grad;
                accumulator1[2] += coef * w1_grad;
                accumulator1[3] += coef * theta1_grad;
                accumulator1[4] += coef * phi1_grad;
                accumulator2[0] += coef * sigma2_grad;
                accumulator2[1] += coef * mu2_grad;
                accumulator2[2] += coef * w2_grad;
                accumulator2[3] += coef * theta2_grad;
                accumulator2[4] += coef * phi2_grad;
            }
        }
    }

    // Write result to global memory.
    for (int p = 0; p < NUM_PARAMS; p++) {
        out1[task2][p][b][blockIdx.x][threadIdx.x] = accumulator1[p];
        out2[task2][p][b][blockIdx.x][threadIdx.x] = accumulator2[p];
    }
}

template <int block_size, int num_threads_split, int per_thread, int num_tasks>
__global__ void mosm_dense_cuda_kernel(
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> x1,
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> params1,
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> x2,
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> params2,
    const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> start,
    const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> end,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> tasks,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> tasks_block,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> index1,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> index2,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> out) {
    // Shared memory buffers:
    const int num_threads = num_threads_split * num_threads_split;
    const int tile_size = num_threads * per_thread;
    __shared__ std::array<float, tile_size> x1_shm;
    __shared__ std::array<float, tile_size> x2_shm;
    __shared__ std::array<float, tile_size * NUM_PARAMS> params1_shm;
    __shared__ std::array<float, tile_size * NUM_PARAMS> params2_shm;
    __shared__ std::array<int, tile_size * num_tasks> start1_shm;
    __shared__ std::array<int, tile_size * num_tasks> end1_shm;
    __shared__ std::array<int, tile_size> index2_shm;

    // Alternative block indexing:
    const int per_thread_split = tile_size / num_threads_split;
    const int thread_idx1 = threadIdx.x / num_threads_split;
    const int thread_idx2 = threadIdx.x % num_threads_split;

    // Prepare output accumulators per thread:
    std::array<std::array<float, per_thread_split>, per_thread_split> accumulator;
    for (int i = 0; i < accumulator.size(); i++) {
        for (int j = 0; j < accumulator[i].size(); j++) {
            accumulator[i][j] = 0.0f;
        }
    }

    // Step 1: Each thread i loads corresponding values for row i and column i.
    // Except for parameters, which are loaded separately.
    std::array<int, per_thread> task1;
    std::array<int, per_thread> task2;
    for (int i = 0; i < per_thread; i++) {
        const int local_i = i * num_threads + threadIdx.x;
        const int global_i1 = blockIdx.x * tile_size + local_i;
        if (global_i1 < index1.size(0)) {
            const int index1_reg = index1[global_i1];
            x1_shm[local_i] = x1[index1_reg];
            // Find task index.
            for (task1[i] = 0; tasks[task1[i]] <= index1_reg; task1[i]++) {
            }
            task1[i]--;
            // Compute block index from task index.
            const int index1_block =
                (index1_reg - tasks[task1[i]]) / block_size + tasks_block[task1[i]];
            // Load start and end indices for this block.
            for (int j = 0; j < num_tasks; j++) {
                start1_shm[j * tile_size + local_i] = start[index1_block][j];
                end1_shm[j * tile_size + local_i] = end[index1_block][j];
            }
        } else {
            x1_shm[local_i] = 0.0f;
            task1[i] = 0;
            for (int j = 0; j < num_tasks; j++) {
                start1_shm[j * tile_size + local_i] = 0;
                end1_shm[j * tile_size + local_i] = 0;
            }
        }
        const int global_i2 = blockIdx.y * tile_size + local_i;
        if (global_i2 < index2.size(0)) {
            const int index2_reg = index2[global_i2];
            x2_shm[local_i] = x2[index2_reg];
            // Find task index. (This is the only case where we use the symmetry in our kernel!)
            for (task2[i] = 0; tasks[task2[i]] <= index2_reg; task2[i]++) {
            }
            task2[i]--;
            index2_shm[local_i] = index2_reg;
        } else {
            x2_shm[local_i] = 0.0f;
            task2[i] = 0;
            index2_shm[local_i] = x2.size(0);
        }
    }

    // Step 2: Loop over batch dimension.
    const int b_size = params1.size(2);
    for (int b = 0; b < b_size; b++) {
        // Step 2.1: Load parameters.
        for (int i = 0; i < per_thread; i++) {
            const int local_i = i * num_threads + threadIdx.x;
            for (int p = 0; p < NUM_PARAMS; p++) {
                params1_shm[p * tile_size + local_i] = params1[task1[i]][p][b];
            }
            for (int p = 0; p < NUM_PARAMS; p++) {
                params2_shm[p * tile_size + local_i] = params2[task2[i]][p][b];
            }
        }

        __syncthreads();

        // Step 2.2: Load parameters from shared memory.
        std::array<float, per_thread_split> x1_reg;
        std::array<float, per_thread_split> sigma1;
        std::array<float, per_thread_split> mu1;
        std::array<float, per_thread_split> w1;
        std::array<float, per_thread_split> theta1;
        std::array<float, per_thread_split> phi1;
        std::array<float, per_thread_split> x2_reg;
        std::array<float, per_thread_split> sigma2;
        std::array<float, per_thread_split> mu2;
        std::array<float, per_thread_split> w2;
        std::array<float, per_thread_split> theta2;
        std::array<float, per_thread_split> phi2;
        std::array<int, per_thread_split> index2_reg;
        for (int i = 0; i < per_thread_split; i++) {
            const int shm_idx1 = i * num_threads_split + thread_idx1;
            x1_reg[i] = x1_shm[shm_idx1];
            sigma1[i] = params1_shm[0 * num_threads + shm_idx1];
            mu1[i] = params1_shm[1 * num_threads + shm_idx1];
            w1[i] = params1_shm[2 * num_threads + shm_idx1];
            theta1[i] = params1_shm[3 * num_threads + shm_idx1];
            phi1[i] = params1_shm[4 * num_threads + shm_idx1];
            const int shm_idx2 = i * num_threads_split + thread_idx2;
            x2_reg[i] = x2_shm[shm_idx2];
            sigma2[i] = params2_shm[0 * num_threads + shm_idx2];
            mu2[i] = params2_shm[1 * num_threads + shm_idx2];
            w2[i] = params2_shm[2 * num_threads + shm_idx2];
            theta2[i] = params2_shm[3 * num_threads + shm_idx2];
            phi2[i] = params2_shm[4 * num_threads + shm_idx2];
            index2_reg[i] = index2_shm[shm_idx2];
        }

        // Step 2.3: Compute kernel values.
        for (int i = 0; i < per_thread_split; i++) {
            for (int j = 0; j < per_thread_split; j++) {
                // Find out whether we are in range.
                bool in_range = false;
                for (int t = 0; t < num_tasks; t++) {
                    const int idx = t * tile_size + i * num_threads_split + thread_idx1;
                    const int s = start1_shm[idx];
                    const int e = end1_shm[idx];
                    if (s <= index2_reg[j] && index2_reg[j] < e) {
                        in_range = true;
                    }
                }
                // Compute kernel value.
                if (in_range) {
                    const float sigma_inv = 1 / (sigma1[i] + sigma2[j]);
                    const float mu_diff = mu1[i] - mu2[j];
                    const float sigma = 2.0f * sigma1[i] * sigma2[j] * sigma_inv;
                    const float mu = (sigma1[i] * mu2[j] + sigma2[j] * mu1[i]) * sigma_inv;
                    const float w = w1[i] * w2[j] * exp(-0.25f * mu_diff * mu_diff * sigma_inv);
                    const float theta = theta1[i] - theta2[j];
                    const float phi = phi1[i] - phi2[j];
                    const float alpha = w * sqrt(2.0f * CONSTANT_PI_F * sigma);
                    const float tau_theta = x1_reg[i] - x2_reg[j] + theta;
                    const float exp_term = exp(-0.5f * tau_theta * tau_theta * sigma);
                    const float cos_term = cos(tau_theta * mu + phi);
                    const float kernel_value = alpha * exp_term * cos_term;
                    accumulator[i][j] += kernel_value;
                }
            }
        }

        __syncthreads();
    }

// Step 3: Write output to global memory.
#pragma unroll 1
    for (int i = 0; i < per_thread_split; i++) {
        for (int j = 0; j < per_thread_split; j++) {
            const int idx1 = blockIdx.x * tile_size + i * num_threads_split + thread_idx1;
            const int idx2 = blockIdx.y * tile_size + j * num_threads_split + thread_idx2;
            if (idx1 < index1.size(0) && idx2 < index2.size(0)) {
                out[idx1][idx2] = accumulator[i][j];
            }
        }
    }
}

template <int block_size, int num_threads_split, int per_thread, int num_tasks>
__global__ void mosm_dense_bwd_cuda_kernel(
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> x1,
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> params1,
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> x2,
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> params2,
    const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> start,
    const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> end,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> tasks,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> tasks_block,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> index1,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> index2,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> out_grad,
    torch::PackedTensorAccessor32<float, 6, torch::RestrictPtrTraits> params1_grad,
    torch::PackedTensorAccessor32<float, 6, torch::RestrictPtrTraits> params2_grad) {
    // Shared memory buffers:
    const int num_threads = num_threads_split * num_threads_split;
    const int tile_size = num_threads * per_thread;
    __shared__ std::array<float, tile_size> x1_shm;
    __shared__ std::array<float, tile_size> x2_shm;
    __shared__ std::array<float, tile_size * NUM_PARAMS> params1_shm;
    __shared__ std::array<float, tile_size * NUM_PARAMS> params2_shm;
    __shared__ std::array<int, tile_size * num_tasks> start1_shm;
    __shared__ std::array<int, tile_size * num_tasks> end1_shm;
    __shared__ std::array<int, tile_size> index2_shm;
    __shared__ std::array<int, tile_size> task1_shm;
    __shared__ std::array<int, tile_size> task2_shm;

    // Alternative block indexing:
    const int per_thread_split = tile_size / num_threads_split;
    const int thread_idx1 = threadIdx.x / num_threads_split;
    const int thread_idx2 = threadIdx.x % num_threads_split;

    // Load output gradients for reduction
    std::array<std::array<float, per_thread_split>, per_thread_split> out_grad_reg;
    for (int i = 0; i < per_thread_split; i++) {
        for (int j = 0; j < per_thread_split; j++) {
            const int idx1 = blockIdx.x * tile_size + i * num_threads_split + thread_idx1;
            const int idx2 = blockIdx.y * tile_size + j * num_threads_split + thread_idx2;
            if (idx1 < index1.size(0) && idx2 < index2.size(0)) {
                out_grad_reg[i][j] = out_grad[idx1][idx2];
            } else {
                out_grad_reg[i][j] = 0;
            }
        }
    }

    // Step 1: Each thread i loads corresponding values for row i and column i.
    // Except for parameters, which are loaded separately.
    std::array<int, per_thread> task1;
    std::array<int, per_thread> task2;
    for (int i = 0; i < per_thread; i++) {
        const int local_i = i * num_threads + threadIdx.x;
        const int global_i1 = blockIdx.x * tile_size + local_i;
        if (global_i1 < index1.size(0)) {
            const int index1_reg = index1[global_i1];
            x1_shm[local_i] = x1[index1_reg];
            // Find task index.
            for (task1[i] = 0; tasks[task1[i]] <= index1_reg; task1[i]++) {
            }
            task1[i]--;
            task1_shm[local_i] = task1[i];
            // Compute block index from task index.
            const int index1_block =
                (index1_reg - tasks[task1[i]]) / block_size + tasks_block[task1[i]];
            // Load start and end indices for this block.
            for (int j = 0; j < num_tasks; j++) {
                start1_shm[j * tile_size + local_i] = start[index1_block][j];
                end1_shm[j * tile_size + local_i] = end[index1_block][j];
            }
        } else {
            x1_shm[local_i] = 0.0f;
            task1[i] = 0;
            task1_shm[local_i] = 0;
            for (int j = 0; j < num_tasks; j++) {
                start1_shm[j * tile_size + local_i] = 0;
                end1_shm[j * tile_size + local_i] = 0;
            }
        }
        const int global_i2 = blockIdx.y * tile_size + local_i;
        if (global_i2 < index2.size(0)) {
            const int index2_reg = index2[global_i2];
            x2_shm[local_i] = x2[index2_reg];
            // Find task index. (This is the only case where we use the symmetry in our kernel!)
            for (task2[i] = 0; tasks[task2[i]] <= index2_reg; task2[i]++) {
            }
            task2[i]--;
            task2_shm[local_i] = task2[i];
            index2_shm[local_i] = index2_reg;
        } else {
            x2_shm[local_i] = 0.0f;
            task2[i] = 0;
            task2_shm[local_i] = 0;
            index2_shm[local_i] = x2.size(0);
        }
    }

    // Step 2: Loop over batch dimension.
    const int b_size = params1.size(2);
    for (int b = 0; b < b_size; b++) {
        // Step 2.1: Load parameters.
        for (int i = 0; i < per_thread; i++) {
            const int local_i = i * num_threads + threadIdx.x;
            for (int p = 0; p < NUM_PARAMS; p++) {
                params1_shm[p * tile_size + local_i] = params1[task1[i]][p][b];
            }
            for (int p = 0; p < NUM_PARAMS; p++) {
                params2_shm[p * tile_size + local_i] = params2[task2[i]][p][b];
            }
        }

        __syncthreads();

        // Step 2.2: Load parameters from shared memory.
        std::array<float, per_thread_split> x1_reg;
        std::array<float, per_thread_split> sigma1;
        std::array<float, per_thread_split> mu1;
        std::array<float, per_thread_split> w1;
        std::array<float, per_thread_split> theta1;
        std::array<float, per_thread_split> phi1;
        std::array<int, per_thread_split> task1_reg;
        std::array<float, per_thread_split> x2_reg;
        std::array<float, per_thread_split> sigma2;
        std::array<float, per_thread_split> mu2;
        std::array<float, per_thread_split> w2;
        std::array<float, per_thread_split> theta2;
        std::array<float, per_thread_split> phi2;
        std::array<int, per_thread_split> index2_reg;
        std::array<int, per_thread_split> task2_reg;
        for (int i = 0; i < per_thread_split; i++) {
            const int shm_idx1 = i * num_threads_split + thread_idx1;
            x1_reg[i] = x1_shm[shm_idx1];
            sigma1[i] = params1_shm[0 * num_threads + shm_idx1];
            mu1[i] = params1_shm[1 * num_threads + shm_idx1];
            w1[i] = params1_shm[2 * num_threads + shm_idx1];
            theta1[i] = params1_shm[3 * num_threads + shm_idx1];
            phi1[i] = params1_shm[4 * num_threads + shm_idx1];
            task1_reg[i] = task1_shm[shm_idx1];
            const int shm_idx2 = i * num_threads_split + thread_idx2;
            x2_reg[i] = x2_shm[shm_idx2];
            sigma2[i] = params2_shm[0 * num_threads + shm_idx2];
            mu2[i] = params2_shm[1 * num_threads + shm_idx2];
            w2[i] = params2_shm[2 * num_threads + shm_idx2];
            theta2[i] = params2_shm[3 * num_threads + shm_idx2];
            phi2[i] = params2_shm[4 * num_threads + shm_idx2];
            index2_reg[i] = index2_shm[shm_idx2];
            task2_reg[i] = task2_shm[shm_idx2];
        }

        // Step 2.3: Prepare gradient accumulators.
        std::array<std::array<float, NUM_PARAMS>, num_tasks> accumulator1 = {};
        std::array<std::array<float, NUM_PARAMS>, num_tasks> accumulator2 = {};

        // Step 2.4: Compute kernel gradients.
        for (int i = 0; i < per_thread_split; i++) {
            for (int j = 0; j < per_thread_split; j++) {
                // Find out whether we are in range.
                bool in_range = false;
                for (int t = 0; t < num_tasks; t++) {
                    const int idx = t * tile_size + i * num_threads_split + thread_idx1;
                    const int s = start1_shm[idx];
                    const int e = end1_shm[idx];
                    if (s <= index2_reg[j] && index2_reg[j] < e) {
                        in_range = true;
                    }
                }
                // Compute kernel gradients.
                if (in_range) {
                    // Forward pass
                    const float sigma_inv = 1 / (sigma1[i] + sigma2[j]);
                    const float mu_diff = mu1[i] - mu2[j];
                    const float mu_diff_sq = mu_diff * mu_diff;
                    const float sigma = 2.0f * sigma1[i] * sigma2[j] * sigma_inv;
                    const float mu = (sigma1[i] * mu2[j] + sigma2[j] * mu1[i]) * sigma_inv;
                    const float w = w1[i] * w2[j] *
                                    exp(-0.25f * (mu1[i] - mu2[j]) * (mu1[i] - mu2[j]) * sigma_inv);
                    const float theta = theta1[i] - theta2[j];
                    const float phi = phi1[i] - phi2[j];
                    const float sqrt_term = sqrt(2.0f * CONSTANT_PI_F * sigma);
                    const float alpha = w * sqrt_term;
                    const float tau_theta = x1_reg[i] - x2_reg[j] + theta;
                    const float exp_term = exp(-0.5f * tau_theta * tau_theta * sigma);
                    const float cos_term = cos(tau_theta * mu + phi);
                    const float kernel_value = alpha * exp_term * cos_term;

                    // Backward pass
                    const float sigma1_sq = sigma1[i] * sigma1[i];
                    const float sigma2_sq = sigma2[j] * sigma2[j];
                    const float sin_term = sin(tau_theta * mu + phi);
                    const float sigma_grad_factor_a = exp_term * sigma_inv * sigma_inv;
                    const float sigma_grad_factor_b = cos_term * w;
                    const float sigma_grad_sum_a = 2.0f * CONSTANT_PI_F / sqrt_term;
                    const float sigma_grad_sum_b = sqrt_term * mu_diff_sq / 4.0f;
                    const float sigma_grad_sum_c = alpha * cos_term * tau_theta * tau_theta;
                    const float sigma_grad_sum_d = alpha * sin_term * tau_theta * mu_diff;
                    const float mu_grad_factor = alpha * exp_term * sigma_inv;
                    const float mu_grad_sum_a = 0.5f * cos_term * mu_diff;
                    const float mu_grad_sum_b = sin_term * tau_theta;
                    const float sigma1_grad =
                        sigma_grad_factor_a *
                        (sigma_grad_factor_b * (sigma_grad_sum_a * sigma2_sq + sigma_grad_sum_b) -
                         sigma_grad_sum_c * sigma2_sq + sigma_grad_sum_d * sigma2[j]);
                    const float sigma2_grad =
                        sigma_grad_factor_a *
                        (sigma_grad_factor_b * (sigma_grad_sum_a * sigma1_sq + sigma_grad_sum_b) -
                         sigma_grad_sum_c * sigma1_sq - sigma_grad_sum_d * sigma1[i]);
                    const float mu1_grad =
                        -mu_grad_factor * (mu_grad_sum_a + mu_grad_sum_b * sigma2[j]);
                    const float mu2_grad =
                        -mu_grad_factor * (-mu_grad_sum_a + mu_grad_sum_b * sigma1[i]);
                    const float w1_grad = kernel_value / w1[i];
                    const float w2_grad = kernel_value / w2[j];
                    const float theta2_grad =
                        alpha * exp_term * (cos_term * tau_theta * sigma + sin_term * mu);
                    const float theta1_grad = -theta2_grad;
                    const float phi2_grad = alpha * exp_term * sin_term;
                    const float phi1_grad = -phi2_grad;

                    // Accumulate gradients
                    accumulator1[task1_reg[i]][0] += sigma1_grad * out_grad_reg[i][j];
                    accumulator2[task2_reg[j]][0] += sigma2_grad * out_grad_reg[i][j];
                    accumulator1[task1_reg[i]][1] += mu1_grad * out_grad_reg[i][j];
                    accumulator2[task2_reg[j]][1] += mu2_grad * out_grad_reg[i][j];
                    accumulator1[task1_reg[i]][2] += w1_grad * out_grad_reg[i][j];
                    accumulator2[task2_reg[j]][2] += w2_grad * out_grad_reg[i][j];
                    accumulator1[task1_reg[i]][3] += theta1_grad * out_grad_reg[i][j];
                    accumulator2[task2_reg[j]][3] += theta2_grad * out_grad_reg[i][j];
                    accumulator1[task1_reg[i]][4] += phi1_grad * out_grad_reg[i][j];
                    accumulator2[task2_reg[j]][4] += phi2_grad * out_grad_reg[i][j];
                }
            }
        }

        // Step 2.5: Write parameter gradients to global memory.
        for (int t = 0; t < num_tasks; t++) {
            for (int p = 0; p < NUM_PARAMS; p++) {
                params1_grad[t][p][b][blockIdx.y][blockIdx.x][threadIdx.x] = accumulator1[t][p];
                params2_grad[t][p][b][blockIdx.y][blockIdx.x][threadIdx.x] = accumulator2[t][p];
            }
        }

        __syncthreads();
    }
}

} // namespace

torch::Tensor mosm_vecmul_cuda(torch::Tensor x1, torch::Tensor params1, torch::Tensor x2,
                               torch::Tensor params2, torch::Tensor rhs, torch::Tensor start,
                               torch::Tensor end, torch::Tensor tasks, torch::Tensor tasks_block) {
    using namespace torch::indexing;

    const auto m = (unsigned int)x1.size(0);
    const auto n = (unsigned int)x2.size(0);
    const auto b = (unsigned int)params1.size(2);
    const auto t = (unsigned int)tasks.size(0) - 1;

#ifdef PRINT_SIZE
    printf("m, n, b, t: (%d, %d, %d, %d, %d)\n", m, n, b, t);
#endif

    const int b_block_size = 4;
    const int m_block_size = MOSM_MATMUL_BLOCK_SIZE;
    const int num_threads = 64;
    const int n_block_size = 64;
    const int b_blocks = (b + b_block_size - 1) / b_block_size;

    const auto out_opts =
        torch::TensorOptions().dtype(x1.dtype()).layout(x1.layout()).device(x1.device());
    auto out = torch::zeros({m}, out_opts);

    for (int task = 0; task < t; task++) {
        const int task_start = tasks[task].item<int>();
        const int task_end = tasks[task + 1].item<int>();
        if (task_start == task_end) {
            continue;
        }
        const int task_block_start = tasks_block[task].item<int>();
        const int task_block_end = tasks_block[task + 1].item<int>();

        const dim3 threads{num_threads, 1, 1};
        const dim3 blocks{b_blocks, task_block_end - task_block_start, 1};

#ifdef PRINT_SIZE
        printf("task: %d (%d - %d / %d - %d)\n", task, task_start, task_end, task_block_start,
               task_block_end);
        printf("threads: (%d, %d, %d)\n", threads.x, threads.y, threads.z);
        printf("blocks: (%d, %d, %d)\n", blocks.x, blocks.y, blocks.z);
#endif

        const auto x1_chunk = x1.slice(0, task_start, task_end);
        const auto params1_chunk = params1[task];
        const auto start_chunk = start.slice(0, task_block_start, task_block_end);
        const auto end_chunk = end.slice(0, task_block_start, task_block_end);
        auto out_chunk = torch::zeros({b_blocks, task_end - task_start}, out_opts);

        mosm_vecmul_cuda_kernel<b_block_size, m_block_size, num_threads, n_block_size>
            <<<blocks, threads>>>(
                x1_chunk.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
                params1_chunk.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                x2.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
                params2.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
                rhs.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
                start_chunk.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
                end_chunk.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
                out_chunk.packed_accessor32<float, 2, torch::RestrictPtrTraits>());

#ifdef GPU_ASSERT
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
#endif

        auto out_chunk_target = out.slice(0, task_start, task_end);
        at::sum_out(out_chunk_target, out_chunk, 0);
    }

    return out;
}

#ifdef MOSM_MATMUL_K_SIZE
torch::Tensor mosm_matmul_cuda(torch::Tensor x1, torch::Tensor params1, torch::Tensor x2,
                               torch::Tensor params2, torch::Tensor rhs, torch::Tensor start,
                               torch::Tensor end, torch::Tensor tasks, torch::Tensor tasks_block) {
    using namespace torch::indexing;

    const int k_size = MOSM_MATMUL_K_SIZE;
    const auto m = (unsigned int)x1.size(0);
    const auto n = (unsigned int)x2.size(0);
    const auto b = (unsigned int)params1.size(2);
    const auto t = (unsigned int)tasks.size(0) - 1;

#ifdef PRINT_SIZE
    printf("m, n, b, t: (%d, %d, %d, %d, %d)\n", m, n, b, t);
#endif

    const int b_block_size = 4;
    const int m_block_size = MOSM_MATMUL_BLOCK_SIZE;
    const int num_threads = 64;
    const int n_block_size = 12;
    const int b_blocks = (b + b_block_size - 1) / b_block_size;

    const auto out_opts =
        torch::TensorOptions().dtype(x1.dtype()).layout(x1.layout()).device(x1.device());
    auto out = torch::zeros({m, k_size}, out_opts);

    for (int task = 0; task < t; task++) {
        const int task_start = tasks[task].item<int>();
        const int task_end = tasks[task + 1].item<int>();
        if (task_start == task_end) {
            continue;
        }
        const int task_block_start = tasks_block[task].item<int>();
        const int task_block_end = tasks_block[task + 1].item<int>();

        const dim3 threads{num_threads, 1, 1};
        const dim3 blocks{task_block_end - task_block_start, b_blocks, 1};

#ifdef PRINT_SIZE
        printf("task: %d (%d - %d / %d - %d)\n", task, task_start, task_end, task_block_start,
               task_block_end);
        printf("threads: (%d, %d, %d)\n", threads.x, threads.y, threads.z);
        printf("blocks: (%d, %d, %d)\n", blocks.x, blocks.y, blocks.z);
#endif

        const auto x1_chunk = x1.slice(0, task_start, task_end);
        const auto params1_chunk = params1[task];
        const auto start_chunk = start.slice(0, task_block_start, task_block_end);
        const auto end_chunk = end.slice(0, task_block_start, task_block_end);
        auto out_chunk = torch::zeros({b_blocks, task_end - task_start, k_size}, out_opts);

        mosm_matmul_cuda_kernel<b_block_size, m_block_size, num_threads, n_block_size, k_size>
            <<<blocks, threads>>>(
                x1_chunk.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
                params1_chunk.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                x2.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
                params2.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
                rhs.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                start_chunk.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
                end_chunk.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
                out_chunk.packed_accessor32<float, 3, torch::RestrictPtrTraits>());

#ifdef GPU_ASSERT
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
#endif

        auto out_chunk_target = out.slice(0, task_start, task_end);
        at::sum_out(out_chunk_target, out_chunk, 0);
    }

    return out;
}

std::array<torch::Tensor, 2> mosm_matmul_bwd_cuda(torch::Tensor x1, torch::Tensor params1,
                                                  torch::Tensor x2, torch::Tensor params2,
                                                  torch::Tensor rhs, torch::Tensor start,
                                                  torch::Tensor end, torch::Tensor tasks,
                                                  torch::Tensor tasks_block,
                                                  torch::Tensor out_grad) {
    using namespace torch::indexing;

    const int matmul_k_threads = 4;
    const int m_block_size = MOSM_MATMUL_BLOCK_SIZE;
    const int num_threads = 64;
    const int n_block_size = 4;
    const int k_size = MOSM_MATMUL_K_SIZE;

    const auto m = (unsigned int)x1.size(0);
    const auto n = (unsigned int)x2.size(0);
    const auto k = (unsigned int)rhs.size(1);
    const auto b = (unsigned int)params1.size(2);
    const auto t = (unsigned int)tasks.size(0) - 1;

    const auto out_opts =
        torch::TensorOptions().dtype(x1.dtype()).layout(x1.layout()).device(x1.device());
    auto grad1 = torch::zeros(params1.sizes(), out_opts);
    auto grad2 = torch::zeros(params2.sizes(), out_opts);

#ifdef PRINT_SIZE
    printf("m, n, k, b, t: (%d, %d, %d, %d, %d)\n", m, n, k, b, t);
#endif

    for (int task = 0; task < t; task++) {
        const int task_start = tasks[task].item<int>();
        const int task_end = tasks[task + 1].item<int>();
        if (task_start == task_end) {
            continue;
        }
        const int task_block_start = tasks_block[task].item<int>();
        const int task_block_end = tasks_block[task + 1].item<int>();

        const dim3 threads{num_threads, 1, 1};
        const dim3 blocks{task_block_end - task_block_start, b, 1};

#ifdef PRINT_SIZE
        printf("task: %d (%d - %d / %d - %d)\n", task, task_start, task_end, task_block_start,
               task_block_end);
        printf("threads: (%d, %d, %d)\n", threads.x, threads.y, threads.z);
        printf("blocks: (%d, %d, %d)\n", blocks.x, blocks.y, blocks.z);
#endif

        const auto x1_chunk = x1.slice(0, task_start, task_end);
        const auto params1_chunk = params1[task];
        const auto start_chunk = start.slice(0, task_block_start, task_block_end);
        const auto end_chunk = end.slice(0, task_block_start, task_block_end);
        const auto out_grad_chunk = out_grad.slice(0, task_start, task_end);
        auto grad1_chunk =
            torch::zeros({NUM_PARAMS, params1.size(2), blocks.x, num_threads}, out_opts);
        auto grad2_chunk = torch::zeros(
            {params2.size(0), NUM_PARAMS, params2.size(2), blocks.x, num_threads}, out_opts);

        mosm_matmul_bwd_cuda_kernel<matmul_k_threads, m_block_size, num_threads, n_block_size,
                                    k_size><<<blocks, threads>>>(
            x1_chunk.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
            params1_chunk.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            x2.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
            params2.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            rhs.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            start_chunk.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
            end_chunk.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
            out_grad_chunk.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            grad1_chunk.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            grad2_chunk.packed_accessor32<float, 5, torch::RestrictPtrTraits>());

#ifdef GPU_ASSERT
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
#endif

        grad1[task] += grad1_chunk.sum({2, 3});
        grad2 += grad2_chunk.sum({3, 4});
    }

    return {grad1, grad2};
}
#endif

std::array<torch::Tensor, 2>
mosm_bilinear_derivative_cuda(torch::Tensor x1, torch::Tensor params1, torch::Tensor x2,
                              torch::Tensor params2, torch::Tensor start, torch::Tensor end,
                              torch::Tensor tasks, torch::Tensor tasks_block,
                              torch::Tensor left_vectors, torch::Tensor right_vectors) {
    using namespace torch::indexing;

    const int block_size = MOSM_MATMUL_BLOCK_SIZE;
    const int num_threads_split = 16;
    const int num_threads = num_threads_split * num_threads_split;

    const auto m = (unsigned int)x1.size(0);
    const auto n = (unsigned int)x2.size(0);
    const auto k = (unsigned int)left_vectors.size(1);
    const auto b = (unsigned int)params1.size(2);
    const auto t = (unsigned int)tasks.size(0) - 1;

    const auto out_opts =
        torch::TensorOptions().dtype(x1.dtype()).layout(x1.layout()).device(x1.device());
    auto grad1 = torch::zeros(params1.sizes(), out_opts);
    auto grad2 = torch::zeros(params2.sizes(), out_opts);

    const auto right_vectors_t = right_vectors.transpose(0, 1).contiguous();

#ifdef PRINT_SIZE
    printf("m, n, k, b, t: (%d, %d, %d, %d, %d)\n", m, n, k, b, t);
#endif

    for (int task = 0; task < t; task++) {
        const int task_start = tasks[task].item<int>();
        const int task_end = tasks[task + 1].item<int>();
        if (task_start == task_end) {
            continue;
        }
        const int task_block_start = tasks_block[task].item<int>();
        const int task_block_end = tasks_block[task + 1].item<int>();
        const int m_blocks = task_block_end - task_block_start;

        const dim3 threads{num_threads, 1, 1};
        const dim3 blocks{m_blocks, b, t};

#ifdef PRINT_SIZE
        printf("task: %d (%d - %d / %d - %d)\n", task, task_start, task_end, task_block_start,
               task_block_end);
        printf("threads: (%d, %d, %d)\n", threads.x, threads.y, threads.z);
        printf("blocks: (%d, %d, %d)\n", blocks.x, blocks.y, blocks.z);
#endif

        const auto x1_chunk = x1.slice(0, task_start, task_end);
        const auto params1_chunk = params1[task];
        const auto start_chunk = start.slice(0, task_block_start, task_block_end);
        const auto end_chunk = end.slice(0, task_block_start, task_block_end);
        const auto left_vectors_t_chunk =
            left_vectors.slice(0, task_start, task_end).transpose(0, 1).contiguous();
        auto grad1_chunk = torch::zeros({t, NUM_PARAMS, b, m_blocks, num_threads}, out_opts);
        auto grad2_chunk = torch::zeros({t, NUM_PARAMS, b, m_blocks, num_threads}, out_opts);

        mosm_bilinear_derivative_cuda_kernel<block_size, num_threads_split><<<blocks, threads>>>(
            x1_chunk.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
            params1_chunk.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            x2.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
            params2.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            start_chunk.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
            end_chunk.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
            left_vectors_t_chunk.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            right_vectors_t.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            grad1_chunk.packed_accessor32<float, 5, torch::RestrictPtrTraits>(),
            grad2_chunk.packed_accessor32<float, 5, torch::RestrictPtrTraits>());

#ifdef GPU_ASSERT
        gpuErrchk(cudaPeekAtLastError());
        gpuErrchk(cudaDeviceSynchronize());
#endif

        grad1[task] += grad1_chunk.sum({0, 3, 4});
        grad2 += grad2_chunk.sum({3, 4});
    }

    return {grad1, grad2};
}

torch::Tensor mosm_dense_cuda(torch::Tensor x1, torch::Tensor params1, torch::Tensor x2,
                              torch::Tensor params2, torch::Tensor start, torch::Tensor end,
                              torch::Tensor tasks, torch::Tensor tasks_block, torch::Tensor index1,
                              torch::Tensor index2) {
    const int num_tasks = MOSM_MATMUL_NUM_TASKS;
    const int block_size = MOSM_MATMUL_BLOCK_SIZE;
    const int num_threads_split = 8;
    const int num_threads = num_threads_split * num_threads_split;
    const int per_thread = 1;
    const int tile_size = num_threads * per_thread;

    const auto m = (unsigned int)index1.size(0);
    const auto n = (unsigned int)index2.size(0);

    const auto out_opts =
        torch::TensorOptions().dtype(x1.dtype()).layout(x1.layout()).device(x1.device());
    auto out = torch::zeros({m, n}, out_opts);

    const dim3 threads{num_threads, 1, 1};
    const dim3 blocks{(m + tile_size - 1) / tile_size, (n + tile_size - 1) / tile_size, 1};

#ifdef PRINT_SIZE
    printf("m, n: (%d, %d, %d)\n", m, n);
    printf("threads: (%d, %d, %d)\n", threads.x, threads.y, threads.z);
    printf("blocks: (%d, %d, %d)\n", blocks.x, blocks.y, blocks.z);
#endif

    mosm_dense_cuda_kernel<block_size, num_threads_split, per_thread, num_tasks>
        <<<blocks, threads>>>(x1.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
                              params1.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
                              x2.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
                              params2.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
                              start.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
                              end.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
                              tasks.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                              tasks_block.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                              index1.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                              index2.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                              out.packed_accessor32<float, 2, torch::RestrictPtrTraits>());

#ifdef GPU_ASSERT
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#endif

    return out;
}

std::array<torch::Tensor, 2> mosm_dense_bwd_cuda(torch::Tensor x1, torch::Tensor params1,
                                                 torch::Tensor x2, torch::Tensor params2,
                                                 torch::Tensor start, torch::Tensor end,
                                                 torch::Tensor tasks, torch::Tensor tasks_block,
                                                 torch::Tensor index1, torch::Tensor index2,
                                                 torch::Tensor out_grad) {
    const int num_tasks = MOSM_MATMUL_NUM_TASKS;
    const int block_size = MOSM_MATMUL_BLOCK_SIZE;
    const int num_threads_split = 8;
    const int num_threads = num_threads_split * num_threads_split;
    const int per_thread = 1;
    const int tile_size = num_threads * per_thread;

    const auto m = (unsigned int)index1.size(0);
    const auto n = (unsigned int)index2.size(0);

    const dim3 threads{num_threads, 1, 1};
    const dim3 blocks{(m + tile_size - 1) / tile_size, (n + tile_size - 1) / tile_size, 1};

#ifdef PRINT_SIZE
    printf("m, n: (%d, %d)\n", m, n);
    printf("threads: (%d, %d, %d)\n", threads.x, threads.y, threads.z);
    printf("blocks: (%d, %d, %d)\n", blocks.x, blocks.y, blocks.z);
#endif

    const auto out_opts =
        torch::TensorOptions().dtype(x1.dtype()).layout(x1.layout()).device(x1.device());
    auto out1 = torch::zeros(
        {params1.size(0), params1.size(1), params1.size(2), blocks.y, blocks.x, threads.x},
        out_opts);
    auto out2 = torch::zeros(
        {params2.size(0), params2.size(1), params2.size(2), blocks.y, blocks.x, threads.x},
        out_opts);

    mosm_dense_bwd_cuda_kernel<block_size, num_threads_split, per_thread, num_tasks>
        <<<blocks, threads>>>(x1.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
                              params1.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
                              x2.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
                              params2.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
                              start.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
                              end.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
                              tasks.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                              tasks_block.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                              index1.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                              index2.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                              out_grad.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                              out1.packed_accessor32<float, 6, torch::RestrictPtrTraits>(),
                              out2.packed_accessor32<float, 6, torch::RestrictPtrTraits>());

#ifdef GPU_ASSERT
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#endif

    auto out1_sum = out1.sum({3, 4, 5});
    auto out2_sum = out2.sum({3, 4, 5});
    return {out1_sum, out2_sum};
}
