#include "../common/accessor.cuh"
#include "../common/gpu_assert.cuh"
#include "../common/kernel_function.cuh"
#include "../common/utils.h"
#include "matmul.h"

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

__global__ void
kernel_matmul_vector_cuda_kernel(BatchLayout<KM_BATCH_DIM> batch_layout,
                                 BatchedAccessor<float, KM_BATCH_DIM, 1> x1_batch,
                                 BatchedAccessor<float, KM_BATCH_DIM, 1> x2_batch,
                                 BatchedAccessor<float, KM_BATCH_DIM, 2> rhs_batch,
                                 BatchedAccessor<float, KM_BATCH_DIM, 1> params_batch,
                                 BatchedAccessor<int, KM_BATCH_DIM, 1> start_batch,
                                 BatchedAccessor<int, KM_BATCH_DIM, 1> end_batch,
                                 BatchedAccessor<float, KM_BATCH_DIM, 2> out_batch) {

    // Load batch
    const auto batch = batch_layout.get_batch(blockIdx.z);
    const auto x1 = x1_batch[batch];
    const auto x2 = x2_batch[batch];
    const auto rhs = rhs_batch[batch];
    const auto params = params_batch[batch];
    const auto start = start_batch[batch];
    const auto end = end_batch[batch];
    auto out = out_batch[batch];

    // Index calculations
    const auto m_base = blockIdx.x * KM_BLOCK_SIZE + threadIdx.x;
    const auto m_size = x1.size(0);
    const auto k_base = blockIdx.y * KM_MATMUL_K_BLOCK_SIZE;
    const auto k_size = rhs.size(1);

    // Load parameters to registers
    std::array<float, KM_NUM_PARAMS> reg_params;
    for (int i = 0; i < KM_NUM_PARAMS; i++) {
        reg_params[i] = params[i];
    }

    // Each thread computes values for entries m_base + i * KM_MATMUL_THREADS
    std::array<std::array<float, KM_MATMUL_K_BLOCK_SIZE>, KM_MATMUL_PER_THREAD> accumulator = {};

    // Load x1 into registers
    std::array<float, KM_MATMUL_PER_THREAD> reg_x1;
    for (int i = 0; i < KM_MATMUL_PER_THREAD; i++) {
        const auto m = m_base + i * KM_MATMUL_THREADS;
        if (m < m_size) {
            reg_x1[i] = x1[m];
        } else {
            reg_x1[i] = 0;
        }
    }

    // We buffer the x2 and rhs in shm as these are read multiple times
    __shared__ std::array<float, KM_BLOCK_SIZE> shm_x2;
    __shared__ std::array<std::array<float, KM_MATMUL_K_BLOCK_SIZE>, KM_BLOCK_SIZE> shm_rhs;

    // Outer loop advances blocks along columns of the kernel matrix and rows of the rhs
    const int start_m = start[blockIdx.x];
    const int end_m = end[blockIdx.x];
    for (int n_base = start_m; n_base < end_m; n_base += KM_BLOCK_SIZE) {

        // Load x2
        for (int i = threadIdx.x; i < KM_BLOCK_SIZE; i += KM_MATMUL_THREADS) {
            const auto n = n_base + i;
            if (n < end_m) {
                shm_x2[i] = x2[n];
            } else {
                shm_x2[i] = 0;
            }
        }

        // Load rhs
        for (int i = threadIdx.x; i < KM_BLOCK_SIZE * KM_MATMUL_K_BLOCK_SIZE;
             i += KM_MATMUL_THREADS) {
            const auto n = i / KM_MATMUL_K_BLOCK_SIZE;
            const auto n_global = n_base + n;
            const auto k = i % KM_MATMUL_K_BLOCK_SIZE;
            if (n_global < end_m && k_base + k < k_size) {
                shm_rhs[n][k] = rhs[n_global][k_base + k];
            } else {
                shm_rhs[n][k] = 0;
            }
        }

        __syncthreads();

        // Perform reduction
        for (int i = 0; i < KM_BLOCK_SIZE; i++) {
            const auto reg_x2 = shm_x2[i];
            std::array<float, KM_MATMUL_K_BLOCK_SIZE> reg_rhs;
            for (int k = 0; k < KM_MATMUL_K_BLOCK_SIZE; k++) {
                reg_rhs[k] = shm_rhs[i][k];
            }
            for (int j = 0; j < KM_MATMUL_PER_THREAD; j++) {
                const auto value = kernel_function(reg_x1[j], reg_x2, reg_params);
                for (int k = 0; k < KM_MATMUL_K_BLOCK_SIZE; k++) {
                    accumulator[j][k] += value * reg_rhs[k];
                }
            }
        }
        __syncthreads();
    }

    // Write back to global memory
    for (int m = 0; m < KM_MATMUL_PER_THREAD; m++) {
        const auto m_global = m_base + m * KM_MATMUL_THREADS;
        for (int k = 0; k < KM_MATMUL_K_BLOCK_SIZE; k++) {
            if (m_global < m_size && k < k_size) {
                out[m_global][k_base + k] = accumulator[m][k];
            }
        }
    }
}

torch::Tensor kernel_matmul_cuda(torch::Tensor x1, torch::Tensor x2, torch::Tensor rhs,
                                 torch::Tensor params, torch::Tensor start, torch::Tensor end) {

    const auto batch_layout = BatchLayout<KM_BATCH_DIM>(x1.sizes().data());

    const dim3 threads{KM_MATMUL_THREADS, 1, 1};
    const dim3 blocks{
        KM_CEIL_DIV(x1.size(-1), KM_BLOCK_SIZE),
        KM_CEIL_DIV(rhs.size(-1), KM_MATMUL_K_BLOCK_SIZE),
        batch_layout.num_batches(),
    };

#ifdef KM_DEBUG_PRINT_SIZE
    printf("m, n, k: (%d, %d, %d, %d, %d)\n", x1.size(-1), x2.size(-1), rhs.size(-1));
    printf("threads: (%d, %d, %d)\n", threads.x, threads.y, threads.z);
    printf("blocks: (%d, %d, %d)\n", blocks.x, blocks.y, blocks.z);
#endif

    const auto out_opts =
        torch::TensorOptions().dtype(x1.dtype()).layout(x1.layout()).device(x1.device());
    const auto out_shape = batch_layout.make_shape<2>({x1.size(-1), rhs.size(-1)});
    auto out = torch::zeros(out_shape, out_opts);

    kernel_matmul_vector_cuda_kernel<<<blocks, threads>>>(
        batch_layout, BatchedAccessor<float, KM_BATCH_DIM, 1>(x1),
        BatchedAccessor<float, KM_BATCH_DIM, 1>(x2), BatchedAccessor<float, KM_BATCH_DIM, 2>(rhs),
        BatchedAccessor<float, KM_BATCH_DIM, 1>(params),
        BatchedAccessor<int, KM_BATCH_DIM, 1>(start), BatchedAccessor<int, KM_BATCH_DIM, 1>(end),
        BatchedAccessor<float, KM_BATCH_DIM, 2>(out));

    KM_DO_GPU_ASSERT;

    return out;
}
