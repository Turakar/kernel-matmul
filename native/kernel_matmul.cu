#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <stdio.h>

#include "kernel_matmul.h"

#define CONSTANT_PI_F 3.141592654f

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

/*
 * Kernel function
 */

template <int kernel_function_type, int num_params>
__device__ float kernel_function(const float x1, const float x2,
                                 const std::array<float, num_params> params);

template <int kernel_function_type, int num_params>
__device__ std::array<float, num_params>
kernel_function_bwd(const float x1, const float x2, const std::array<float, num_params> params);

template <>
__device__ __forceinline__ float kernel_function<LOCALLY_PERIODIC, LOCALLY_PERIODIC_NUM_PARAMS>(
    const float x1, const float x2, const std::array<float, LOCALLY_PERIODIC_NUM_PARAMS> params) {
    const float lengthscale_rbf = params[0];
    const float lengthscale_periodic = params[1];
    const float period_length = params[2];
    const float outputscale = params[3];
    const float diff = x1 - x2;
    const float rbf = diff * diff / lengthscale_rbf / lengthscale_rbf / ((float)2);
    const float periodic_inner = sin(diff / period_length);
    const float periodic = ((float)2) * periodic_inner * periodic_inner / lengthscale_periodic;
    return outputscale * exp(-(rbf + periodic));
}

template <>
__device__ __forceinline__ std::array<float, LOCALLY_PERIODIC_NUM_PARAMS>
kernel_function_bwd<LOCALLY_PERIODIC, LOCALLY_PERIODIC_NUM_PARAMS>(
    const float x1, const float x2, const std::array<float, LOCALLY_PERIODIC_NUM_PARAMS> params) {
    const float lengthscale_rbf = params[0];
    const float lengthscale_periodic = params[1];
    const float period_length = params[2];
    const float outputscale = params[3];

    // forward pass
    const float diff = x1 - x2;
    const float diff_sq = diff * diff;
    const float rbf = diff_sq / lengthscale_rbf / lengthscale_rbf / 2;
    const float periodic_inner = sin(diff / period_length);
    const float periodic = 2 * periodic_inner * periodic_inner / lengthscale_periodic;
    const float exp_term = exp(-(rbf + periodic));
    const float value = outputscale * exp_term;

    // backward pass
    // wolfram alpha:
    // - D[α*Exp[-(0.5 *(x-y)^2/λ + 2 * (Sin[(x-y)/p])^2/μ)],λ]
    // - D[α*Exp[-(0.5 *(x-y)^2/λ + 2 * (Sin[(x-y)/p])^2/μ)],μ]
    // - D[α*Exp[-(0.5 *(x-y)^2/λ + 2 * (Sin[(x-y)/p])^2/μ)],p]
    // - D[α*Exp[-(0.5 *(x-y)^2/λ + 2 * (Sin[(x-y)/p])^2/μ)],α]
    const float lengthscale_rbf_diff =
        value * diff_sq / (lengthscale_rbf * lengthscale_rbf * lengthscale_rbf);
    const float lengthscale_periodic_diff = value * periodic / lengthscale_periodic;
    const float period_length_diff = value * 2 * sin(2 * diff / period_length) * diff /
                                     (lengthscale_periodic * period_length * period_length);
    const float outputscale_diff = exp_term;

    return {lengthscale_rbf_diff, lengthscale_periodic_diff, period_length_diff, outputscale_diff};
}

template <>
__device__ __forceinline__ float
kernel_function<RBF, RBF_NUM_PARAMS>(const float x1, const float x2,
                                     const std::array<float, RBF_NUM_PARAMS> params) {
    const float lengthscale_rbf = params[0];
    const float outputscale = params[1];
    const float diff = x1 - x2;
    const float rbf = diff * diff / lengthscale_rbf / lengthscale_rbf / ((float)2);
    return outputscale * exp(-rbf);
}

template <>
__device__ __forceinline__ std::array<float, RBF_NUM_PARAMS>
kernel_function_bwd<RBF, RBF_NUM_PARAMS>(const float x1, const float x2,
                                         const std::array<float, RBF_NUM_PARAMS> params) {
    const float lengthscale_rbf = params[0];
    const float outputscale = params[1];

    // forward pass
    const float diff = x1 - x2;
    const float diff_sq = diff * diff;
    const float rbf = diff_sq / lengthscale_rbf / lengthscale_rbf / 2;
    const float exp_term = exp(-rbf);
    const float value = outputscale * exp_term;

    // backward pass
    const float lengthscale_rbf_diff =
        value * diff_sq / (lengthscale_rbf * lengthscale_rbf * lengthscale_rbf);
    const float outputscale_diff = exp_term;

    return {lengthscale_rbf_diff, outputscale_diff};
}

template <>
__device__ __forceinline__ float kernel_function<SPECTRAL, SPECTRAL_NUM_PARAMS>(
    const float x1, const float x2, const std::array<float, SPECTRAL_NUM_PARAMS> params) {
    const float lengthscale = params[0];
    const float frequency = params[1];
    const float outputscale = params[2];

    const float diff = x1 - x2;
    const float rbf = diff * diff / lengthscale / lengthscale / ((float)2);
    const float cos_term = cos(2.0f * CONSTANT_PI_F * frequency * diff);
    const float value = outputscale * exp(-rbf) * cos_term;
    return value;
}

template <>
__device__ __forceinline__ std::array<float, SPECTRAL_NUM_PARAMS>
kernel_function_bwd<SPECTRAL, SPECTRAL_NUM_PARAMS>(
    const float x1, const float x2, const std::array<float, SPECTRAL_NUM_PARAMS> params) {
    const float lengthscale = params[0];
    const float frequency = params[1];
    const float outputscale = params[2];

    const float diff = x1 - x2;
    const float rbf = diff * diff / lengthscale / lengthscale / 2.0f;
    const float cos_factor = 2.0f * CONSTANT_PI_F * diff;
    const float cos_inner = cos_factor * frequency;
    const float cos_term = cos(cos_inner);
    const float exp_term = exp(-rbf);
    const float value = outputscale * exp_term * cos_term;

    const float outputscale_grad = exp_term * cos_term;
    const float lengthscale_grad = value * diff * diff / (lengthscale * lengthscale * lengthscale);
    const float period_grad = -outputscale * exp_term * sin(cos_inner) * cos_factor;

    return {lengthscale_grad, period_grad, outputscale_grad};
}

/*
 * Implicit kernel matmul
 */

template <int block_size, int thread_dim, int per_thread, int kernel_function_type, int num_params>
__global__ void kernel_matmul_cuda_kernel(
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> x1,
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> x2,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> rhs,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> params,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> start,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> end,
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> out) {
    // Index calculations
    static_assert(thread_dim * per_thread == block_size,
                  "block_size must be the product of thread_dim and per_thread");
    static_assert((thread_dim * thread_dim) % 32 == 0,
                  "Thread block must be evenly divisible in warps.");
    const int k_base = blockIdx.x * block_size;
    const int m_base = blockIdx.y * block_size;
    const int b = blockIdx.z;
    const int k_size = rhs.size(1);
    const int m_size = x1.size(0);

    // This is an alternative indexing that is used for loading from global to shared memory to
    // avoid bank conflicts.
    const auto thread_rank = threadIdx.y * thread_dim + threadIdx.x;
    const auto warp_based_x = thread_rank % 32;
    const auto warp_based_y = thread_rank / 32;
    const auto warp_num = thread_dim * thread_dim / 32;
    static_assert(thread_dim % warp_num == 0,
                  "thread_dim must be evenly divisible by the number of warps.");

    // Shared memory buffer
    // kernel_values: block_size, thread_dim
    // rhs: block_size, thread_dim
    extern __shared__ int sdata[];
    auto shm_rhs = (float *)sdata;
    auto shm_kernel = shm_rhs + block_size * thread_dim;

    // Register buffer
    float reg_rhs[per_thread];
    float reg_kernel[per_thread];

    // Load parameters to registers
    std::array<float, num_params> reg_params;
#pragma unroll
    for (int i = 0; i < num_params; i++) {
        reg_params[i] = params[i][b];
    }
    const int start_m = start[blockIdx.y];
    const int end_m = end[blockIdx.y];

    // Initialize accumulator for output
    // Each thread accumulates outputs for the entries
    // (m_base + m * thread_dim + threadIdx.y, k_base + k * thread_dim + threadIdx.x).
    float accumulator[per_thread][per_thread];
    for (int m = 0; m < per_thread; m++) {
        for (int k = 0; k < per_thread; k++) {
            accumulator[m][k] = 0;
        }
    }

    // Outer loop advances blocks along columns of the kernel matrix and rows of the rhs
    for (int n_base = start_m; n_base < end_m; n_base += thread_dim) {
        // Load rhs and kernel matrix blocks into shared memory
        // n is always associated with threadIdx.y here to allow for coalesced access.
        // Trick: We can transpose the kernel matrix at virtually no cost here.
        // We use the warp-based indexing calculated above to avoid shm bank conflicts.
        for (int j = warp_based_y; j < thread_dim; j += warp_num) {
            for (int i = warp_based_x; i < block_size; i += 32) {
                const auto shm_index = j * block_size + i;
                const auto n = n_base + j;
                const auto k = k_base + i;
                const auto m = m_base + i;
                if (k < k_size && n < end_m) {
                    shm_rhs[shm_index] = rhs[n][k];
                } else {
                    shm_rhs[shm_index] = 0;
                }
                if (m < m_size && n < end_m) {
                    shm_kernel[shm_index] =
                        kernel_function<kernel_function_type, num_params>(x1[m], x2[n], reg_params);
                } else {
                    shm_kernel[shm_index] = 0;
                }
            }
        }
        __syncthreads();

        // Outer loop iterates over n.
        // We unroll all inner loops for ILP.
        for (int i = 0; i < thread_dim; i++) {
// Load from shm into registers
#pragma unroll
            for (int j = 0; j < per_thread; j++) {
                reg_rhs[j] = shm_rhs[i * block_size + j * thread_dim + threadIdx.x];
                reg_kernel[j] = shm_kernel[i * block_size + j * thread_dim + threadIdx.y];
            }

// Inner loops iterate over m and k.
#pragma unroll
            for (int m = 0; m < per_thread; m++) {
#pragma unroll
                for (int k = 0; k < per_thread; k++) {
                    accumulator[m][k] += reg_kernel[m] * reg_rhs[k];
                }
            }
        }
        __syncthreads();
    }

    // Write output to global memory
    for (int m_offset = 0; m_offset < per_thread; m_offset++) {
        const auto m = m_base + m_offset * thread_dim + threadIdx.y;
        if (m < m_size) {
            for (int k_offset = 0; k_offset < per_thread; k_offset++) {
                const auto k = k_base + k_offset * thread_dim + threadIdx.x;
                if (k < k_size) {
                    out[b][m][k] = accumulator[m_offset][k_offset];
                }
            }
        }
    }
}

template <int block_size, int thread_dim, int per_thread, int kernel_function_type, int num_params>
__global__ void kernel_bilinear_derivative_cuda_kernel(
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> x1,
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> x2,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> left_vecs,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> right_vecs,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> params,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> start,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> end,
    torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> params_grad) {
    // Index calculations
    static_assert(thread_dim * per_thread == block_size,
                  "block_size must be the product of thread_dim and per_thread");

    // Prepare buffers
    std::array<float, num_params> accumulator = {};
    __shared__ std::array<float, block_size> shm_left_vecs;
    __shared__ std::array<float, block_size> shm_right_vecs;
    __shared__ std::array<float, block_size> shm_x2;

    // Load x1 to registers
    std::array<float, per_thread> x1_reg = {};
    for (int i = 0; i < per_thread; i++) {
        const auto x1_index = blockIdx.x * block_size + i * thread_dim + threadIdx.x;
        if (x1_index < x1.size(0)) {
            x1_reg[i] = x1[x1_index];
        } else {
            x1_reg[i] = 0;
        }
    }

    // Load params to registers
    std::array<float, num_params> params_reg = {};
    for (int i = 0; i < num_params; i++) {
        params_reg[i] = params[i][blockIdx.y];
    }

    // Loop over x2
    const auto start_m = start[blockIdx.x];
    const auto end_m = end[blockIdx.x];
    for (int n = start_m; n < end_m; n += block_size) {

        // Load x2 to shared memory
        for (int j = 0; j < per_thread; j++) {
            const auto x2_index = n + j * thread_dim + threadIdx.y;
            if (x2_index < end_m) {
                shm_x2[j * thread_dim + threadIdx.y] = x2[x2_index];
            } else {
                shm_x2[j * thread_dim + threadIdx.y] = 0;
            }
        }

        // Compute coefficients from outer product of left and right vectors
        std::array<std::array<float, per_thread>, per_thread> coefficients = {};
        for (int k = 0; k < left_vecs.size(0); k++) {
            for (int i = 0; i < per_thread; i++) {
                const auto x1_index = blockIdx.x * block_size + i * thread_dim + threadIdx.x;
                if (x1_index < x1.size(0)) {
                    shm_left_vecs[i * thread_dim + threadIdx.x] = left_vecs[k][x1_index];
                } else {
                    shm_left_vecs[i * thread_dim + threadIdx.x] = 0;
                }
            }
            for (int j = 0; j < per_thread; j++) {
                const auto x2_index = n + j * thread_dim + threadIdx.y;
                if (x2_index < end_m) {
                    shm_right_vecs[j * thread_dim + threadIdx.y] = right_vecs[k][x2_index];
                } else {
                    shm_right_vecs[j * thread_dim + threadIdx.y] = 0;
                }
            }
            __syncthreads();
            std::array<float, per_thread> left_vecs_reg;
            std::array<float, per_thread> right_vecs_reg;
            for (int i = 0; i < per_thread; i++) {
                left_vecs_reg[i] = shm_left_vecs[i * thread_dim + threadIdx.x];
                right_vecs_reg[i] = shm_right_vecs[i * thread_dim + threadIdx.y];
            }
            for (int i = 0; i < per_thread; i++) {
                for (int j = 0; j < per_thread; j++) {
                    coefficients[i][j] += left_vecs_reg[i] * right_vecs_reg[j];
                }
            }
            __syncthreads();
        }

        // Load x2 to registers
        std::array<float, per_thread> x2_reg;
        for (int j = 0; j < per_thread; j++) {
            x2_reg[j] = shm_x2[j * thread_dim + threadIdx.y];
        }

        // Compute gradients
        for (int i = 0; i < per_thread; i++) {
            for (int j = 0; j < per_thread; j++) {
                const auto grads = kernel_function_bwd<kernel_function_type, num_params>(
                    x1_reg[i], x2_reg[j], params_reg);
                for (int p = 0; p < num_params; p++) {
                    accumulator[p] += coefficients[i][j] * grads[p];
                }
            }
        }
        __syncthreads();
    }

    // Write output to global memory
    for (int p = 0; p < num_params; p++) {
        params_grad[p][blockIdx.y][blockIdx.x][threadIdx.y][threadIdx.x] = accumulator[p];
    }
}

template <int block_size, int thread_dim, int per_thread, int kernel_function_type, int num_params>
__global__ void kernel_weighted_bilinear_derivative_cuda_kernel(
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> x1,
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> x2,
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> left_vecs,
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> right_vecs,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> params,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> start,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> end,
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> weights,
    torch::PackedTensorAccessor32<float, 6, torch::RestrictPtrTraits> params_grad,
    torch::PackedTensorAccessor32<float, 6, torch::RestrictPtrTraits> weights_grad) {
    // Index calculations
    static_assert(thread_dim * per_thread == block_size,
                  "block_size must be the product of thread_dim and per_thread");
    const auto num_tasks = weights.size(1);
    const auto task1 = blockIdx.z % num_tasks;
    const auto task2 = blockIdx.z / num_tasks;

    // Prepare buffers
    std::array<float, num_params> accumulator = {};
    float weight_accumulator = 0;
    __shared__ std::array<float, block_size> shm_left_vecs;
    __shared__ std::array<float, block_size> shm_right_vecs;
    __shared__ std::array<float, block_size> shm_x2;

    // Load x1 to registers
    std::array<float, per_thread> x1_reg = {};
    for (int i = 0; i < per_thread; i++) {
        const auto x1_index = blockIdx.x * block_size + i * thread_dim + threadIdx.x;
        if (x1_index < x1.size(0)) {
            x1_reg[i] = x1[x1_index];
        } else {
            x1_reg[i] = 0;
        }
    }

    // Load params to registers
    std::array<float, num_params> params_reg = {};
    for (int i = 0; i < num_params; i++) {
        params_reg[i] = params[i][blockIdx.y];
    }

    // Load weight
    const auto weight = weights[blockIdx.y][task1][task2];

    // Loop over x2
    const auto start_m = start[blockIdx.x];
    const auto end_m = end[blockIdx.x];
    for (int n = start_m; n < end_m; n += block_size) {

        // Load x2 to shared memory
        for (int j = 0; j < per_thread; j++) {
            const auto x2_index = n + j * thread_dim + threadIdx.y;
            if (x2_index < end_m) {
                shm_x2[j * thread_dim + threadIdx.y] = x2[x2_index];
            } else {
                shm_x2[j * thread_dim + threadIdx.y] = 0;
            }
        }

        // Compute coefficients from outer product of left and right vectors
        std::array<std::array<float, per_thread>, per_thread> coefficients = {};
        for (int k = 0; k < left_vecs.size(0); k++) {
            for (int i = 0; i < per_thread; i++) {
                const auto x1_index = blockIdx.x * block_size + i * thread_dim + threadIdx.x;
                if (x1_index < x1.size(0)) {
                    shm_left_vecs[i * thread_dim + threadIdx.x] = left_vecs[k][task1][x1_index];
                } else {
                    shm_left_vecs[i * thread_dim + threadIdx.x] = 0;
                }
            }
            for (int j = 0; j < per_thread; j++) {
                const auto x2_index = n + j * thread_dim + threadIdx.y;
                if (x2_index < end_m) {
                    shm_right_vecs[j * thread_dim + threadIdx.y] = right_vecs[k][task2][x2_index];
                } else {
                    shm_right_vecs[j * thread_dim + threadIdx.y] = 0;
                }
            }
            __syncthreads();
            std::array<float, per_thread> left_vecs_reg;
            std::array<float, per_thread> right_vecs_reg;
            for (int i = 0; i < per_thread; i++) {
                left_vecs_reg[i] = shm_left_vecs[i * thread_dim + threadIdx.x];
                right_vecs_reg[i] = shm_right_vecs[i * thread_dim + threadIdx.y];
            }
            for (int i = 0; i < per_thread; i++) {
                for (int j = 0; j < per_thread; j++) {
                    coefficients[i][j] += left_vecs_reg[i] * right_vecs_reg[j];
                }
            }
            __syncthreads();
        }

        // Load x2 to registers
        std::array<float, per_thread> x2_reg;
        for (int j = 0; j < per_thread; j++) {
            x2_reg[j] = shm_x2[j * thread_dim + threadIdx.y];
        }

        // Compute gradients
        for (int i = 0; i < per_thread; i++) {
            for (int j = 0; j < per_thread; j++) {
                const auto value = kernel_function<kernel_function_type, num_params>(
                    x1_reg[i], x2_reg[j], params_reg);
                weight_accumulator += coefficients[i][j] * value;
                const auto grads = kernel_function_bwd<kernel_function_type, num_params>(
                    x1_reg[i], x2_reg[j], params_reg);
                for (int p = 0; p < num_params; p++) {
                    accumulator[p] += coefficients[i][j] * grads[p];
                }
            }
        }
        __syncthreads();
    }

    // Write output to global memory
    for (int p = 0; p < num_params; p++) {
        params_grad[p][blockIdx.z][blockIdx.y][blockIdx.x][threadIdx.y][threadIdx.x] =
            weight * accumulator[p];
    }
    weights_grad[blockIdx.y][task1][task2][blockIdx.x][threadIdx.y][threadIdx.x] =
        weight_accumulator;
}

template <int block_size, int thread_dim, int per_thread, int kernel_function_type, int num_params>
__global__ void kernel_matmul_cuda_kernel_bwd(
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> x1,
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> x2,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> rhs,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> params,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> start,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> end,
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> out_grad,
    torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> params_grad) {
    // This is almost the same as the forward pass, as we have the same matmul structure in the
    // derivative. The main difference is that we can now accumulate the gradients to smaller
    // portions directly, saving registers. For this, we need the output gradient, which we will
    // load from global memory directly and cache in registers. However, we will need significantly
    // more shm and registers as we now need to buffer all four gradients of the kernel function.

    // Index calculations
    static_assert(thread_dim * per_thread == block_size,
                  "block_size must be the product of thread_dim and per_thread");
    static_assert((thread_dim * thread_dim) % 32 == 0,
                  "Thread block must be evenly divisible in warps.");
    const int k_base = blockIdx.x * block_size;
    const int m_base = blockIdx.y * block_size;
    const int b = blockIdx.z;
    const int k_size = rhs.size(1);
    const int m_size = x1.size(0);

    // This is an alternative indexing that is used for loading from global to shared memory to
    // avoid bank conflicts.
    const auto thread_rank = threadIdx.y * thread_dim + threadIdx.x;
    const auto warp_based_x = thread_rank % 32;
    const auto warp_based_y = thread_rank / 32;
    const auto warp_num = thread_dim * thread_dim / 32;
    static_assert(thread_dim % warp_num == 0,
                  "thread_dim must be evenly divisible by the number of warps.");

    // Shared memory buffer
    // kernel_values: block_size, thread_dim
    // rhs: block_size, thread_dim
    extern __shared__ int sdata[];
    const int buffer_size = block_size * thread_dim;
    auto shm_rhs = (float *)sdata;
    auto shm_params_grad = shm_rhs + buffer_size;

    // Register buffer
    float reg_rhs[per_thread];
    float reg_params_grad[num_params][per_thread];
    float reg_out_grad[per_thread][per_thread];
#pragma unroll
    for (int m = 0; m < per_thread; m++) {
#pragma unroll
        for (int k = 0; k < per_thread; k++) {
            const auto m_index = m_base + m * thread_dim + threadIdx.y;
            const auto k_index = k_base + k * thread_dim + threadIdx.x;
            if (m_index < m_size && k_index < k_size) {
                reg_out_grad[m][k] = out_grad[b][m_index][k_index];
            } else {
                reg_out_grad[m][k] = 0;
            }
        }
    }

    // Load parameters to registers
    std::array<float, num_params> reg_params;
#pragma unroll
    for (int i = 0; i < num_params; i++) {
        reg_params[i] = params[i][b];
    }
    const int start_m = start[blockIdx.y];
    const int end_m = end[blockIdx.y];

    // Initialize accumulator for output
    // Each thread accumulates outputs for the entries
    // (m_base + m * thread_dim + threadIdx.y, k_base + k * thread_dim + threadIdx.x).
    // However, we only need a single accumulator per parameter now, as we can
    // multiply with the output gradient directly.
    float accumulator[num_params];
#pragma unroll
    for (int p = 0; p < num_params; p++) {
        accumulator[p] = 0;
    }

    // Outer loop advances blocks along columns of the kernel matrix and rows of the rhs
    for (int n_base = start_m; n_base < end_m; n_base += thread_dim) {
        // Load rhs and kernel matrix blocks into shared memory
        // n is always associated with threadIdx.y here to allow for coalesced access.
        // Trick: We can transpose the kernel matrix at virtually no cost here.
        // We use the warp-based indexing calculated above to avoid shm bank conflicts.
        for (int j = warp_based_y; j < thread_dim; j += warp_num) {
            for (int i = warp_based_x; i < block_size; i += 32) {
                const auto shm_index = j * block_size + i;
                const auto n = n_base + j;
                const auto k = k_base + i;
                const auto m = m_base + i;
                if (k < k_size && n < end_m) {
                    shm_rhs[shm_index] = rhs[n][k];
                } else {
                    shm_rhs[shm_index] = 0;
                }
                if (m < m_size && n < end_m) {
                    const auto grads = kernel_function_bwd<kernel_function_type, num_params>(
                        x1[m], x2[n], reg_params);
#pragma unroll
                    for (int p = 0; p < num_params; p++) {
                        shm_params_grad[p * buffer_size + shm_index] = grads[p];
                    }
                } else {
#pragma unroll
                    for (int p = 0; p < num_params; p++) {
                        shm_params_grad[p * buffer_size + shm_index] = 0;
                    }
                }
            }
        }
        __syncthreads();

        // Outer loop iterates over n.
        // We unroll all inner loops for ILP.
        for (int i = 0; i < thread_dim; i++) {
// Load from shm into registers
#pragma unroll
            for (int j = 0; j < per_thread; j++) {
                const auto shm_index = i * block_size + j * thread_dim;
                reg_rhs[j] = shm_rhs[shm_index + threadIdx.x];
#pragma unroll
                for (int p = 0; p < num_params; p++) {
                    reg_params_grad[p][j] =
                        shm_params_grad[p * buffer_size + shm_index + threadIdx.y];
                }
            }

// Inner loops iterate over m and k.
#pragma unroll
            for (int m = 0; m < per_thread; m++) {
#pragma unroll
                for (int k = 0; k < per_thread; k++) {
#pragma unroll
                    for (int p = 0; p < num_params; p++) {
                        accumulator[p] += reg_out_grad[m][k] * reg_params_grad[p][m] * reg_rhs[k];
                    }
                }
            }
        }
        __syncthreads();
    }

    // Write output to global memory
    const auto m = blockIdx.y * thread_dim + threadIdx.y;
    const auto k = blockIdx.x * thread_dim + threadIdx.x;
    if (m < params_grad.size(2) && k < params_grad.size(3)) {
#pragma unroll
        for (int p = 0; p < num_params; p++) {
            params_grad[p][b][m][k] = accumulator[p];
        }
    }
}

template <int block_size, int thread_dim, int per_thread, int kernel_function_type, int num_params,
          int k_block_size>
__global__ void kernel_matmul_vector_cuda_kernel(
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> x1,
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> x2,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> rhs,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> params,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> start,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> end,
    torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> out) {
    // Index calculations
    const auto b = blockIdx.y;
    const auto m_base = blockIdx.x * block_size + threadIdx.x;
    const auto m_size = x1.size(0);
    const auto k_size = rhs.size(1);

    // Load parameters to registers
    std::array<float, num_params> reg_params;
    for (int i = 0; i < num_params; i++) {
        reg_params[i] = params[i][b];
    }

    // Each thread computes values for entries m_base + i * thread_dim
    std::array<std::array<float, k_block_size>, per_thread> accumulator = {};

    // Load x1 into registers
    std::array<float, per_thread> reg_x1;
    for (int i = 0; i < per_thread; i++) {
        const auto m = m_base + i * thread_dim;
        if (m < m_size) {
            reg_x1[i] = x1[m];
        } else {
            reg_x1[i] = 0;
        }
    }

    // We buffer the x2 and rhs in shm as these are read multiple times
    __shared__ std::array<float, block_size> shm_x2;
    __shared__ std::array<std::array<float, k_block_size>, block_size> shm_rhs;

    // Outer loop advances blocks along columns of the kernel matrix and rows of the rhs
    const int start_m = start[blockIdx.x];
    const int end_m = end[blockIdx.x];
    for (int n_base = start_m; n_base < end_m; n_base += block_size) {

        // Load x2
        for (int i = threadIdx.x; i < block_size; i += thread_dim) {
            const auto n = n_base + i;
            if (n < end_m) {
                shm_x2[i] = x2[n];
            } else {
                shm_x2[i] = 0;
            }
        }

        // Load rhs
        for (int i = threadIdx.x; i < block_size * k_block_size; i += thread_dim) {
            const auto n = i / k_block_size;
            const auto n_global = n_base + n;
            const auto k = i % k_block_size;
            if (n_global < end_m && k < k_size) {
                shm_rhs[n][k] = rhs[n_global][k];
            } else {
                shm_rhs[n][k] = 0;
            }
        }

        __syncthreads();

        // Perform reduction
        for (int i = 0; i < block_size; i++) {
            const auto reg_x2 = shm_x2[i];
            std::array<float, k_block_size> reg_rhs;
            for (int k = 0; k < k_block_size; k++) {
                reg_rhs[k] = shm_rhs[i][k];
            }
            for (int j = 0; j < per_thread; j++) {
                const auto value = kernel_function<kernel_function_type, num_params>(
                    reg_x1[j], reg_x2, reg_params);
                for (int k = 0; k < k_block_size; k++) {
                    accumulator[j][k] += value * reg_rhs[k];
                }
            }
        }
        __syncthreads();
    }

    // Write back to global memory
    for (int m = 0; m < per_thread; m++) {
        const auto m_global = m_base + m * thread_dim;
        for (int k = 0; k < k_block_size; k++) {
            if (m_global < m_size && k < k_size) {
                out[b][m_global][k] = accumulator[m][k];
            }
        }
    }
}

template <int block_size, int num_threads_x1, int num_threads_tasks1, int num_blocks_x2,
          int num_tasks, int rhs_columns, int kernel_function_type, int num_params>
__global__ void lmc_matmul_cuda_kernel(
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> x1,
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> x2,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> rhs,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> params,
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> weights,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> start,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> end,
    torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> out) {
    // threadIdx.x: Blocks of tasks1
    // threadIdx.y: Position in x1
    // blockIdx.x: Sub-block of x1
    // blockIdx.y: Blocks of x1
    // blockIdx.z: Block of x2 (position changes depending on start & end)

    static_assert(block_size % num_threads_x1 == 0);
    const auto tasks1_per_thread = (num_tasks + num_threads_tasks1 - 1) / num_threads_tasks1;
    const auto tasks1_offset = threadIdx.x * tasks1_per_thread;

    // Matrix product accumulator for phase II
    std::array<std::array<float, rhs_columns>, tasks1_per_thread> accumulator = {};

    // Load x1
    const auto m = blockIdx.y * block_size + blockIdx.x * num_threads_x1 + threadIdx.y;
    const auto x1_reg = m < x1.size(0) ? x1[m] : 0.0f;

    // Prepare shared memory for data covariance
    __shared__ float data_covar[num_threads_x1][num_threads_tasks1];

    // Loop over x2
    const auto start_m = start[blockIdx.y];
    const auto end_m = end[blockIdx.y];
    const auto n_block_size = (end_m - start_m + num_blocks_x2 - 1) / num_blocks_x2;
    const auto n_start = start_m + blockIdx.z * n_block_size;
    const auto n_end = min(end_m, start_m + (blockIdx.z + 1) * n_block_size);
    for (int n = n_start; n < n_end; n++) {

        // Phase I: Compute covariance
        std::array<std::array<float, num_tasks>, tasks1_per_thread> covar = {};
        for (int l = 0; l < params.size(1); l += num_threads_tasks1) {
            // Compute data covariance for latent model
            const auto x2_n = x2[n];
            const auto l_global = l + threadIdx.x;
            std::array<float, num_threads_tasks1> data_covar_reg;
            if (num_threads_tasks1 > 1) {
                if (l_global < params.size(1)) {
                    std::array<float, num_params> params_reg;
                    for (int i = 0; i < num_params; i++) {
                        params_reg[i] = params[i][l_global];
                    }
                    data_covar[threadIdx.y][threadIdx.x] =
                        kernel_function<kernel_function_type, num_params>(x1_reg, x2_n, params_reg);
                } else {
                    data_covar[threadIdx.y][threadIdx.x] = 0;
                }
                __syncthreads();
                for (int i = 0; i < num_threads_tasks1; i++) {
                    data_covar_reg[i] = data_covar[threadIdx.y][i];
                }
            } else {
                if (l_global < params.size(1)) {
                    std::array<float, num_params> params_reg;
                    for (int i = 0; i < num_params; i++) {
                        params_reg[i] = params[i][l_global];
                    }
                    data_covar_reg[0] =
                        kernel_function<kernel_function_type, num_params>(x1_reg, x2_n, params_reg);
                } else {
                    data_covar_reg[0] = 0;
                }
            }

            // Load weights and compute product
            for (int i = 0; i < num_threads_tasks1; i++) {
                for (int t1 = 0; t1 < tasks1_per_thread; t1++) {
                    const auto t1_global = tasks1_offset + t1;
                    for (int t2 = 0; t2 < num_tasks; t2++) {
                        if (t1_global < num_tasks && l + i < params.size(1)) {
                            const auto weight = weights[l + i][t1_global][t2];
                            covar[t1][t2] += data_covar_reg[i] * weight;
                        }
                    }
                }
            }
            if (num_threads_tasks1 > 1) {
                __syncthreads();
            }
        }

        // Phase II: Multiply chunk with rhs
        // TODO: For K = 1 (or maybe even K < M), it might be good to merge phases I and II to
        // reduce register usage
        for (int t2 = 0; t2 < num_tasks; t2++) {
            for (int k = 0; k < rhs_columns; k++) {
                const auto rhs_reg = rhs[n * num_tasks + t2][k];
                for (int t1 = 0; t1 < tasks1_per_thread; t1++) {
                    accumulator[t1][k] += covar[t1][t2] * rhs_reg;
                }
            }
        }
    }

    // Write output
    if (m < x1.size(0)) {
        for (int t1 = 0; t1 < tasks1_per_thread; t1++) {
            if (t1 + tasks1_offset < num_tasks) {
                for (int k = 0; k < rhs_columns; k++) {
                    out[blockIdx.z][m][t1 + tasks1_offset][k] = accumulator[t1][k];
                }
            }
        }
    }
}

} // namespace

/*
 * Implicit kernel matmul launching methods
 */

template <int kernel_function_type, int num_params>
torch::Tensor kernel_matmul_cuda(torch::Tensor x1, torch::Tensor x2, torch::Tensor rhs,
                                 torch::Tensor params, torch::Tensor start, torch::Tensor end) {
    const int block_size = KERNEL_MATMUL_BLOCK_SIZE;

    torch::Tensor out;
    const auto out_opts =
        torch::TensorOptions().dtype(x1.dtype()).layout(x1.layout()).device(x1.device());

    if (rhs.size(1) > 40) {
        const int thread_dim = 16;
        const int per_thread = 8;

        const dim3 threads{thread_dim, thread_dim, 1};
        const dim3 blocks{(rhs.size(1) + block_size - 1) / block_size,
                          (x1.size(0) + block_size - 1) / block_size, params.size(1)};
        const auto shared = (int)(rhs.element_size() * 2 * block_size * thread_dim);

#ifdef PRINT_SIZE
        printf("m, n, k, b: (%d, %d, %d, %d)\n", x1.size(0), x2.size(0), rhs.size(1),
               params.size(1));
        printf("threads: (%d, %d, %d)\n", threads.x, threads.y, threads.z);
        printf("blocks: (%d, %d, %d)\n", blocks.x, blocks.y, blocks.z);
        printf("shared: %dK\n", (shared + 1023) / 1024);
#endif

        out = torch::zeros({params.size(1), x1.size(0), rhs.size(1)}, out_opts);

        kernel_matmul_cuda_kernel<block_size, thread_dim, per_thread, kernel_function_type,
                                  num_params><<<blocks, threads, shared>>>(
            x1.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
            x2.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
            rhs.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            params.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            start.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
            end.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
            out.packed_accessor32<float, 3, torch::RestrictPtrTraits>());

    } else {
        const int thread_dim = 64;
        const int per_thread = 2;

        const dim3 threads{thread_dim, 1, 1};
        const dim3 blocks{
            (x1.size(0) + block_size - 1) / block_size,
            params.size(1),
            1,
        };

#ifdef PRINT_SIZE
        printf("m, n, b: (%d, %d, %d)\n", x1.size(0), x2.size(0), params.size(1));
        printf("threads: (%d, %d, %d)\n", threads.x, threads.y, threads.z);
        printf("blocks: (%d, %d, %d)\n", blocks.x, blocks.y, blocks.z);
#endif

        out = torch::zeros({params.size(1), x1.size(0), rhs.size(1)}, out_opts);

        switch (rhs.size(1)) {
        case 1:
            kernel_matmul_vector_cuda_kernel<block_size, thread_dim, per_thread,
                                             kernel_function_type, num_params, 1>
                <<<blocks, threads>>>(
                    x1.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
                    x2.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
                    rhs.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                    params.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                    start.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                    end.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                    out.packed_accessor32<float, 3, torch::RestrictPtrTraits>());
            break;
        case 2:
            kernel_matmul_vector_cuda_kernel<block_size, thread_dim, per_thread,
                                             kernel_function_type, num_params, 2>
                <<<blocks, threads>>>(
                    x1.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
                    x2.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
                    rhs.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                    params.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                    start.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                    end.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                    out.packed_accessor32<float, 3, torch::RestrictPtrTraits>());
            break;
        case 3:
            kernel_matmul_vector_cuda_kernel<block_size, thread_dim, per_thread,
                                             kernel_function_type, num_params, 3>
                <<<blocks, threads>>>(
                    x1.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
                    x2.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
                    rhs.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                    params.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                    start.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                    end.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                    out.packed_accessor32<float, 3, torch::RestrictPtrTraits>());
            break;
        case 4:
            kernel_matmul_vector_cuda_kernel<block_size, thread_dim, per_thread,
                                             kernel_function_type, num_params, 4>
                <<<blocks, threads>>>(
                    x1.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
                    x2.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
                    rhs.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                    params.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                    start.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                    end.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                    out.packed_accessor32<float, 3, torch::RestrictPtrTraits>());
            break;
        case 5:
            kernel_matmul_vector_cuda_kernel<block_size, thread_dim, per_thread,
                                             kernel_function_type, num_params, 5>
                <<<blocks, threads>>>(
                    x1.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
                    x2.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
                    rhs.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                    params.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                    start.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                    end.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                    out.packed_accessor32<float, 3, torch::RestrictPtrTraits>());
            break;
        case 6:
            kernel_matmul_vector_cuda_kernel<block_size, thread_dim, per_thread,
                                             kernel_function_type, num_params, 6>
                <<<blocks, threads>>>(
                    x1.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
                    x2.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
                    rhs.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                    params.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                    start.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                    end.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                    out.packed_accessor32<float, 3, torch::RestrictPtrTraits>());
            break;
        case 7:
            kernel_matmul_vector_cuda_kernel<block_size, thread_dim, per_thread,
                                             kernel_function_type, num_params, 7>
                <<<blocks, threads>>>(
                    x1.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
                    x2.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
                    rhs.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                    params.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                    start.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                    end.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                    out.packed_accessor32<float, 3, torch::RestrictPtrTraits>());
            break;
        case 8:
            kernel_matmul_vector_cuda_kernel<block_size, thread_dim, per_thread,
                                             kernel_function_type, num_params, 8>
                <<<blocks, threads>>>(
                    x1.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
                    x2.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
                    rhs.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                    params.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                    start.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                    end.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                    out.packed_accessor32<float, 3, torch::RestrictPtrTraits>());
            break;
        case 9:
        case 10:
        case 11:
        case 12:
            kernel_matmul_vector_cuda_kernel<block_size, thread_dim, per_thread,
                                             kernel_function_type, num_params, 12>
                <<<blocks, threads>>>(
                    x1.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
                    x2.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
                    rhs.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                    params.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                    start.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                    end.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                    out.packed_accessor32<float, 3, torch::RestrictPtrTraits>());
            break;
        case 13:
        case 14:
        case 15:
        case 16:
            kernel_matmul_vector_cuda_kernel<block_size, thread_dim, per_thread,
                                             kernel_function_type, num_params, 16>
                <<<blocks, threads>>>(
                    x1.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
                    x2.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
                    rhs.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                    params.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                    start.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                    end.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                    out.packed_accessor32<float, 3, torch::RestrictPtrTraits>());
            break;
        case 17:
        case 18:
        case 19:
        case 20:
            kernel_matmul_vector_cuda_kernel<block_size, thread_dim, per_thread,
                                             kernel_function_type, num_params, 20>
                <<<blocks, threads>>>(
                    x1.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
                    x2.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
                    rhs.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                    params.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                    start.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                    end.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                    out.packed_accessor32<float, 3, torch::RestrictPtrTraits>());
            break;
        case 21:
        case 22:
        case 23:
        case 24:
            kernel_matmul_vector_cuda_kernel<block_size, thread_dim, per_thread,
                                             kernel_function_type, num_params, 24>
                <<<blocks, threads>>>(
                    x1.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
                    x2.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
                    rhs.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                    params.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                    start.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                    end.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                    out.packed_accessor32<float, 3, torch::RestrictPtrTraits>());
            break;
        case 25:
        case 26:
        case 27:
        case 28:
            kernel_matmul_vector_cuda_kernel<block_size, thread_dim, per_thread,
                                             kernel_function_type, num_params, 28>
                <<<blocks, threads>>>(
                    x1.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
                    x2.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
                    rhs.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                    params.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                    start.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                    end.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                    out.packed_accessor32<float, 3, torch::RestrictPtrTraits>());
            break;
        case 29:
        case 30:
        case 31:
        case 32:
            kernel_matmul_vector_cuda_kernel<block_size, thread_dim, per_thread,
                                             kernel_function_type, num_params, 32>
                <<<blocks, threads>>>(
                    x1.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
                    x2.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
                    rhs.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                    params.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                    start.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                    end.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                    out.packed_accessor32<float, 3, torch::RestrictPtrTraits>());
            break;
        case 33:
        case 34:
        case 35:
        case 36:
            kernel_matmul_vector_cuda_kernel<block_size, thread_dim, per_thread,
                                             kernel_function_type, num_params, 36>
                <<<blocks, threads>>>(
                    x1.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
                    x2.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
                    rhs.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                    params.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                    start.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                    end.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                    out.packed_accessor32<float, 3, torch::RestrictPtrTraits>());
            break;
        case 37:
        case 38:
        case 39:
        case 40:
            kernel_matmul_vector_cuda_kernel<block_size, thread_dim, per_thread,
                                             kernel_function_type, num_params, 40>
                <<<blocks, threads>>>(
                    x1.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
                    x2.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
                    rhs.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                    params.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                    start.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                    end.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                    out.packed_accessor32<float, 3, torch::RestrictPtrTraits>());
            break;
        default:
            throw std::runtime_error("Unsupported rhs size.");
        }
    }

#ifdef GPU_ASSERT
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#endif

    return out;
}

template <int kernel_function_type, int num_params>
torch::Tensor kernel_bilinear_derivative_cuda(torch::Tensor x1, torch::Tensor x2,
                                              torch::Tensor left_vecs, torch::Tensor right_vecs,
                                              torch::Tensor params, torch::Tensor start,
                                              torch::Tensor end) {
    const int block_size = KERNEL_MATMUL_BLOCK_SIZE;

    const int thread_dim = 16;
    const int per_thread = 8;

    const dim3 threads{thread_dim, thread_dim, 1};
    const dim3 blocks{(x1.size(0) + block_size - 1) / block_size, params.size(1), 1};

    const auto out_opts =
        torch::TensorOptions().dtype(x1.dtype()).layout(x1.layout()).device(x1.device());
    auto out =
        torch::zeros({params.size(0), params.size(1), blocks.x, thread_dim, thread_dim}, out_opts);

#ifdef PRINT_SIZE
    printf("m, n, k, b: (%d, %d, %d, %d)\n", x1.size(0), x2.size(0), left_vecs.size(1),
           params.size(1));
    printf("threads: (%d, %d, %d)\n", threads.x, threads.y, threads.z);
    printf("blocks: (%d, %d, %d)\n", blocks.x, blocks.y, blocks.z);
#endif

    const auto left_vecs_t = left_vecs.transpose(0, 1).contiguous();
    const auto right_vecs_t = right_vecs.transpose(0, 1).contiguous();

    kernel_bilinear_derivative_cuda_kernel<block_size, thread_dim, per_thread, kernel_function_type,
                                           num_params>
        <<<blocks, threads>>>(x1.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
                              x2.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
                              left_vecs_t.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                              right_vecs_t.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                              params.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                              start.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                              end.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                              out.packed_accessor32<float, 5, torch::RestrictPtrTraits>());

#ifdef GPU_ASSERT
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#endif

    return out.sum({2, 3, 4});
}

template <int kernel_function_type, int num_params>
std::array<torch::Tensor, 2> kernel_weighted_bilinear_derivative_cuda(
    torch::Tensor x1, torch::Tensor x2, torch::Tensor left_vecs, torch::Tensor right_vecs,
    torch::Tensor params, torch::Tensor start, torch::Tensor end, torch::Tensor weights) {
    const int block_size = KERNEL_MATMUL_BLOCK_SIZE;
    const int thread_dim = 16;
    const int per_thread = 8;

    const auto num_tasks = weights.size(1);

    const dim3 threads{thread_dim, thread_dim, 1};
    const dim3 blocks{(x1.size(0) + block_size - 1) / block_size, params.size(1),
                      num_tasks * num_tasks};

    const auto out_opts =
        torch::TensorOptions().dtype(x1.dtype()).layout(x1.layout()).device(x1.device());
    auto out = torch::zeros({params.size(0), blocks.z, blocks.y, blocks.x, thread_dim, thread_dim},
                            out_opts);
    auto out_weights =
        torch::zeros({blocks.y, num_tasks, num_tasks, blocks.x, thread_dim, thread_dim}, out_opts);

#ifdef PRINT_SIZE
    printf("m, n, k, b, t: (%d, %d, %d, %d, %d)\n", x1.size(0), x2.size(0), left_vecs.size(1),
           params.size(1), num_tasks);
    printf("threads: (%d, %d, %d)\n", threads.x, threads.y, threads.z);
    printf("blocks: (%d, %d, %d)\n", blocks.x, blocks.y, blocks.z);
#endif

    const auto left_vecs_t = left_vecs.view({x1.size(0), weights.size(1), left_vecs.size(1)})
                                 .permute({2, 1, 0})
                                 .contiguous();
    const auto right_vecs_t = right_vecs.view({x2.size(0), weights.size(2), right_vecs.size(1)})
                                  .permute({2, 1, 0})
                                  .contiguous();

    kernel_weighted_bilinear_derivative_cuda_kernel<block_size, thread_dim, per_thread,
                                                    kernel_function_type, num_params>
        <<<blocks, threads>>>(x1.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
                              x2.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
                              left_vecs_t.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
                              right_vecs_t.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
                              params.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                              start.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                              end.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                              weights.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
                              out.packed_accessor32<float, 6, torch::RestrictPtrTraits>(),
                              out_weights.packed_accessor32<float, 6, torch::RestrictPtrTraits>());

#ifdef GPU_ASSERT
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#endif

    return {out.sum({1, 3, 4, 5}), out_weights.sum({3, 4, 5})};
}

template <int kernel_function_type, int num_params>
torch::Tensor kernel_matmul_cuda_bwd(torch::Tensor x1, torch::Tensor x2, torch::Tensor rhs,
                                     torch::Tensor params, torch::Tensor start, torch::Tensor end,
                                     torch::Tensor out_grad) {
    const int block_size = KERNEL_MATMUL_BLOCK_SIZE;
    const int thread_dim = 16;
    const int per_thread = 8;

    const dim3 threads{thread_dim, thread_dim, 1};
    const dim3 blocks{(rhs.size(1) + block_size - 1) / block_size,
                      (x1.size(0) + block_size - 1) / block_size, params.size(1)};
    const auto shared = (int)(rhs.element_size() * (1 + num_params) * block_size * thread_dim);

#ifdef PRINT_SIZE
    printf("m, n, k, b: (%d, %d, %d, %d)\n", x1.size(0), x2.size(0), rhs.size(1), params.size(1));
    printf("threads: (%d, %d, %d)\n", threads.x, threads.y, threads.z);
    printf("blocks: (%d, %d, %d)\n", blocks.x, blocks.y, blocks.z);
    printf("shared: %dK\n", (shared + 1023) / 1024);
#endif

    const auto out_opts =
        torch::TensorOptions().dtype(x1.dtype()).layout(x1.layout()).device(x1.device());
    const auto out_shape =
        torch::IntArrayRef({num_params, params.size(1), x1.size(0), rhs.size(1)});
    auto params_grad = torch::zeros(out_shape, out_opts);

    kernel_matmul_cuda_kernel_bwd<block_size, thread_dim, per_thread, kernel_function_type,
                                  num_params><<<blocks, threads, shared>>>(
        x1.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        x2.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        rhs.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        params.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        start.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        end.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        out_grad.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        params_grad.packed_accessor32<float, 4, torch::RestrictPtrTraits>());

#ifdef GPU_ASSERT
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#endif

    return params_grad.sum({2, 3});
}

template <int num_threads_x1, int num_threads_tasks1, int num_blocks_x2, int num_tasks,
          int kernel_function_type, int num_params>
torch::Tensor _lmc_matmul_cuda(torch::Tensor x1, torch::Tensor x2, torch::Tensor rhs,
                               torch::Tensor params, torch::Tensor weights, torch::Tensor start,
                               torch::Tensor end) {
    const int block_size = KERNEL_MATMUL_BLOCK_SIZE;

    const auto out_opts =
        torch::TensorOptions().dtype(x1.dtype()).layout(x1.layout()).device(x1.device());
    auto out = torch::zeros({num_blocks_x2, x1.size(0), num_tasks, rhs.size(1)}, out_opts);

    const dim3 threads{num_threads_tasks1, num_threads_x1, 1};
    const dim3 blocks{(block_size + num_threads_x1 - 1) / num_threads_x1,
                      (x1.size(0) + block_size - 1) / block_size, num_blocks_x2};

#ifdef PRINT_SIZE
    printf("m, n, k, l: (%d, %d, %d, %d)\n", x1.size(0), x2.size(0), rhs.size(1), params.size(1));
    printf("threads: (%d, %d, %d)\n", threads.x, threads.y, threads.z);
    printf("blocks: (%d, %d, %d)\n", blocks.x, blocks.y, blocks.z);
#endif

    switch (rhs.size(1)) {
    case 1:
        lmc_matmul_cuda_kernel<block_size, num_threads_x1, num_threads_tasks1, num_blocks_x2,
                               num_tasks, 1, kernel_function_type, num_params>
            <<<blocks, threads>>>(x1.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
                                  x2.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
                                  rhs.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                                  params.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                                  weights.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
                                  start.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                  end.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                  out.packed_accessor32<float, 4, torch::RestrictPtrTraits>());
        break;
    case 11:
        lmc_matmul_cuda_kernel<block_size, num_threads_x1, num_threads_tasks1, num_blocks_x2,
                               num_tasks, 11, kernel_function_type, num_params>
            <<<blocks, threads>>>(x1.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
                                  x2.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
                                  rhs.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                                  params.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
                                  weights.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
                                  start.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                  end.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
                                  out.packed_accessor32<float, 4, torch::RestrictPtrTraits>());
        break;
    default:
        throw std::runtime_error("Unsupported rhs size.");
    }

#ifdef GPU_ASSERT
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#endif

    return out.sum(0).view({x1.size(0) * num_tasks, rhs.size(1)});
}

template <int kernel_function_type, int num_params>
torch::Tensor lmc_matmul_cuda(torch::Tensor x1, torch::Tensor x2, torch::Tensor rhs,
                              torch::Tensor params, torch::Tensor weights, torch::Tensor start,
                              torch::Tensor end) {
    switch (weights.size(1)) {
    case 1:
        return _lmc_matmul_cuda<64, 1, 16, 1, kernel_function_type, num_params>(
            x1, x2, rhs, params, weights, start, end);
    case 7:
        return _lmc_matmul_cuda<32, 4, 16, 7, kernel_function_type, num_params>(
            x1, x2, rhs, params, weights, start, end);
    default:
        throw std::runtime_error("Unsupported number of tasks.");
    }
}

/*
 * Explicit template instantiations
 */

template torch::Tensor kernel_matmul_cuda<LOCALLY_PERIODIC, LOCALLY_PERIODIC_NUM_PARAMS>(
    torch::Tensor x1, torch::Tensor x2, torch::Tensor rhs, torch::Tensor params,
    torch::Tensor start, torch::Tensor end);
template torch::Tensor kernel_matmul_cuda<RBF, RBF_NUM_PARAMS>(torch::Tensor x1, torch::Tensor x2,
                                                               torch::Tensor rhs,
                                                               torch::Tensor params,
                                                               torch::Tensor start,
                                                               torch::Tensor end);
template torch::Tensor
kernel_matmul_cuda<SPECTRAL, SPECTRAL_NUM_PARAMS>(torch::Tensor x1, torch::Tensor x2,
                                                  torch::Tensor rhs, torch::Tensor params,
                                                  torch::Tensor start, torch::Tensor end);

template torch::Tensor
kernel_bilinear_derivative_cuda<LOCALLY_PERIODIC, LOCALLY_PERIODIC_NUM_PARAMS>(
    torch::Tensor x1, torch::Tensor x2, torch::Tensor left_vecs, torch::Tensor right_vecs,
    torch::Tensor params, torch::Tensor start, torch::Tensor end);
template torch::Tensor kernel_bilinear_derivative_cuda<RBF, RBF_NUM_PARAMS>(
    torch::Tensor x1, torch::Tensor x2, torch::Tensor left_vecs, torch::Tensor right_vecs,
    torch::Tensor params, torch::Tensor start, torch::Tensor end);
template torch::Tensor kernel_bilinear_derivative_cuda<SPECTRAL, SPECTRAL_NUM_PARAMS>(
    torch::Tensor x1, torch::Tensor x2, torch::Tensor left_vecs, torch::Tensor right_vecs,
    torch::Tensor params, torch::Tensor start, torch::Tensor end);

template std::array<torch::Tensor, 2>
kernel_weighted_bilinear_derivative_cuda<LOCALLY_PERIODIC, LOCALLY_PERIODIC_NUM_PARAMS>(
    torch::Tensor x1, torch::Tensor x2, torch::Tensor left_vecs, torch::Tensor right_vecs,
    torch::Tensor params, torch::Tensor start, torch::Tensor end, torch::Tensor weights);
template std::array<torch::Tensor, 2> kernel_weighted_bilinear_derivative_cuda<RBF, RBF_NUM_PARAMS>(
    torch::Tensor x1, torch::Tensor x2, torch::Tensor left_vecs, torch::Tensor right_vecs,
    torch::Tensor params, torch::Tensor start, torch::Tensor end, torch::Tensor weights);
template std::array<torch::Tensor, 2>
kernel_weighted_bilinear_derivative_cuda<SPECTRAL, SPECTRAL_NUM_PARAMS>(
    torch::Tensor x1, torch::Tensor x2, torch::Tensor left_vecs, torch::Tensor right_vecs,
    torch::Tensor params, torch::Tensor start, torch::Tensor end, torch::Tensor weights);

template torch::Tensor kernel_matmul_cuda_bwd<LOCALLY_PERIODIC, LOCALLY_PERIODIC_NUM_PARAMS>(
    torch::Tensor x1, torch::Tensor x2, torch::Tensor rhs, torch::Tensor params,
    torch::Tensor start, torch::Tensor end, torch::Tensor out_grad);
template torch::Tensor
kernel_matmul_cuda_bwd<RBF, RBF_NUM_PARAMS>(torch::Tensor x1, torch::Tensor x2, torch::Tensor rhs,
                                            torch::Tensor params, torch::Tensor start,
                                            torch::Tensor end, torch::Tensor out_grad);
template torch::Tensor kernel_matmul_cuda_bwd<SPECTRAL, SPECTRAL_NUM_PARAMS>(
    torch::Tensor x1, torch::Tensor x2, torch::Tensor rhs, torch::Tensor params,
    torch::Tensor start, torch::Tensor end, torch::Tensor out_grad);

template torch::Tensor lmc_matmul_cuda<LOCALLY_PERIODIC, LOCALLY_PERIODIC_NUM_PARAMS>(
    torch::Tensor x1, torch::Tensor x2, torch::Tensor rhs, torch::Tensor params,
    torch::Tensor weights, torch::Tensor start, torch::Tensor end);
template torch::Tensor lmc_matmul_cuda<RBF, RBF_NUM_PARAMS>(torch::Tensor x1, torch::Tensor x2,
                                                            torch::Tensor rhs, torch::Tensor params,
                                                            torch::Tensor weights,
                                                            torch::Tensor start, torch::Tensor end);
template torch::Tensor lmc_matmul_cuda<SPECTRAL, SPECTRAL_NUM_PARAMS>(
    torch::Tensor x1, torch::Tensor x2, torch::Tensor rhs, torch::Tensor params,
    torch::Tensor weights, torch::Tensor start, torch::Tensor end);
