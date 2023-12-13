#include "../common/gpu_assert.cuh"
#include "../common/kernel_function.cuh"
#include "../common/utils.h"
#include "matmul_bwd.h"

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

__global__ void kernel_matmul_cuda_kernel_bwd(
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> x1,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> x2,
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> rhs,
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> params,
    const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> start,
    const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> end,
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> out_grad,
    torch::PackedTensorAccessor32<float, 5, torch::RestrictPtrTraits> params_grad) {
    // This is almost the same as the forward pass, as we have the same matmul structure in the
    // derivative. The main difference is that we can now accumulate the gradients to smaller
    // portions directly, saving registers. For this, we need the output gradient, which we will
    // load from global memory directly and cache in registers. However, we will need significantly
    // more shm and registers as we now need to buffer all gradients of the kernel function.

    // Index calculations
    const int block_size = KM_BLOCK_SIZE;
    const int thread_dim = KM_MATMUL_BWD_THREAD_DIM;
    const int per_thread = KM_MATMUL_BWD_PER_THREAD;
    const int num_params = KM_NUM_PARAMS;
    static_assert(thread_dim * per_thread == block_size,
                  "block_size must be the product of thread_dim and per_thread");
    static_assert((thread_dim * thread_dim) % 32 == 0,
                  "Thread block must be evenly divisible in warps.");
    const int k_base = blockIdx.x * block_size;
    const int m_base = blockIdx.y * block_size;
    const int b = blockIdx.z % params.size(1);
    const int batch = blockIdx.z / params.size(1);
    const int k_size = rhs.size(3);
    const int m_size = x1.size(1);

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
                reg_out_grad[m][k] = out_grad[batch][b][m_index][k_index];
            } else {
                reg_out_grad[m][k] = 0;
            }
        }
    }

    // Load parameters to registers
    std::array<float, num_params> reg_params;
#pragma unroll
    for (int i = 0; i < num_params; i++) {
        reg_params[i] = params[batch][b][i];
    }
    const int start_m = start[batch][blockIdx.y];
    const int end_m = end[batch][blockIdx.y];

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
                    shm_rhs[shm_index] = rhs[batch][b][n][k];
                } else {
                    shm_rhs[shm_index] = 0;
                }
                if (m < m_size && n < end_m) {
                    const auto grads = kernel_function_bwd(x1[batch][m], x2[batch][n], reg_params);
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
    if (m < params_grad.size(3) && k < params_grad.size(4)) {
#pragma unroll
        for (int p = 0; p < num_params; p++) {
            params_grad[batch][b][p][m][k] = accumulator[p];
        }
    }
}

torch::Tensor kernel_matmul_bwd_cuda(torch::Tensor x1, torch::Tensor x2, torch::Tensor rhs,
                                     torch::Tensor params, torch::Tensor start, torch::Tensor end,
                                     torch::Tensor out_grad) {
    const int block_size = KM_BLOCK_SIZE;
    const int thread_dim = KM_MATMUL_BWD_THREAD_DIM;
    const int per_thread = KM_MATMUL_BWD_PER_THREAD;
    const int num_params = KM_NUM_PARAMS;
    const int batch = params.size(0);
    const int b = params.size(1);

    const dim3 threads{thread_dim, thread_dim, 1};
    const dim3 blocks{KM_CEIL_DIV(rhs.size(3), block_size), KM_CEIL_DIV(x1.size(1), block_size),
                      b * batch};
    const auto shared = (int)(rhs.element_size() * (1 + num_params) * block_size * thread_dim);

#ifdef PRINT_SIZE
    printf("m, n, k, b, batch: (%d, %d, %d, %d, %d)\n", x1.size(1), x2.size(1), rhs.size(3), b,
           batch);
    printf("threads: (%d, %d, %d)\n", threads.x, threads.y, threads.z);
    printf("blocks: (%d, %d, %d)\n", blocks.x, blocks.y, blocks.z);
    printf("shared: %dK\n", KM_CEIL_DIV(shared, 1024)));
#endif

    const auto out_opts =
        torch::TensorOptions().dtype(x1.dtype()).layout(x1.layout()).device(x1.device());
    const auto out_shape = torch::IntArrayRef({batch, b, num_params, x1.size(1), rhs.size(3)});
    auto params_grad = torch::zeros(out_shape, out_opts);

    kernel_matmul_cuda_kernel_bwd<<<blocks, threads, shared>>>(
        x1.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        x2.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        rhs.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        params.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        start.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
        end.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
        out_grad.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        params_grad.packed_accessor32<float, 5, torch::RestrictPtrTraits>());

    KM_DO_GPU_ASSERT;

    return params_grad.sum({3, 4});
}
