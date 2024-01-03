#include "../common/accessor.cuh"
#include "../common/gpu_assert.cuh"
#include "../common/kernel_function.cuh"
#include "../common/utils.h"
#include "dense.h"

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

__global__ void kernel_dense_cuda_kernel(const BatchLayout<KM_BATCH_DIM> batch_layout,
                                         const BatchedAccessor<float, KM_BATCH_DIM, 1> batch_x1,
                                         const BatchedAccessor<float, KM_BATCH_DIM, 1> batch_x2,
                                         const BatchedAccessor<float, KM_BATCH_DIM, 1> batch_params,
                                         const BatchedAccessor<int, KM_BATCH_DIM, 1> batch_start,
                                         const BatchedAccessor<int, KM_BATCH_DIM, 1> batch_end,
                                         BatchedAccessor<float, KM_BATCH_DIM, 2> batch_out) {

    const auto batch = batch_layout.get_batch(blockIdx.z);
    const auto x1 = batch_x1[batch];
    const auto x2 = batch_x2[batch];
    const auto params = batch_params[batch];
    const auto start = batch_start[batch];
    const auto end = batch_end[batch];
    auto out = batch_out[batch];

    const int block_size = KM_BLOCK_SIZE;
    const int thread_dim = KM_DENSE_THREAD_DIM;
    const int num_params = KM_NUM_PARAMS;
    static_assert(block_size % thread_dim == 0, "block_size must be divisible by thread_dim");
    const int per_thread = block_size / thread_dim;
    const int m_base = blockIdx.x * block_size;

    std::array<float, num_params> params_reg;
    for (int i = 0; i < num_params; i++) {
        params_reg[i] = params[i];
    }

    const int start_reg = start[blockIdx.x];
    const int end_reg = end[blockIdx.x];
    for (int n_base = start_reg; n_base < end_reg; n_base += block_size) {
        std::array<float, per_thread> x1_reg;
        for (int i = 0; i < per_thread; i++) {
            const int m = m_base + threadIdx.x + i * thread_dim;
            if (m < x1.size(0)) {
                x1_reg[i] = x1[m];
            } else {
                x1_reg[i] = 0;
            }
        }
        std::array<float, per_thread> x2_reg;
        for (int i = 0; i < per_thread; i++) {
            const int n = n_base + threadIdx.y + i * thread_dim;
            if (n < end_reg) {
                x2_reg[i] = x2[n];
            } else {
                x2_reg[i] = 0;
            }
        }
        for (int i = 0; i < per_thread; i++) {
            for (int j = 0; j < per_thread; j++) {
                const int m = m_base + threadIdx.x + i * thread_dim;
                const int n = n_base + threadIdx.y + j * thread_dim;
                if (m < x1.size(0) && n < end_reg) {
                    out[m][n] = kernel_function(x1_reg[i], x2_reg[j], params_reg);
                }
            }
        }
    }
}

torch::Tensor kernel_dense_cuda(torch::Tensor x1, torch::Tensor x2, torch::Tensor params,
                                torch::Tensor start, torch::Tensor end) {
    const int block_size = KM_BLOCK_SIZE;
    const int thread_dim = KM_DENSE_THREAD_DIM;
    const auto batch_layout = BatchLayout<KM_BATCH_DIM>(x1.sizes().data());

    const auto out_opts =
        torch::TensorOptions().dtype(x1.dtype()).layout(x1.layout()).device(x1.device());
    const auto out_shape = batch_layout.make_shape<2>({x1.size(-1), x2.size(-1)});
    auto out = torch::zeros(out_shape, out_opts);

    const dim3 blocks{KM_CEIL_DIV(x1.size(-1), block_size), 1, batch_layout.num_batches()};
    const dim3 threads{thread_dim, thread_dim, 1};

    kernel_dense_cuda_kernel<<<blocks, threads>>>(
        batch_layout, BatchedAccessor<float, KM_BATCH_DIM, 1>(x1),
        BatchedAccessor<float, KM_BATCH_DIM, 1>(x2),
        BatchedAccessor<float, KM_BATCH_DIM, 1>(params),
        BatchedAccessor<int, KM_BATCH_DIM, 1>(start), BatchedAccessor<int, KM_BATCH_DIM, 1>(end),
        BatchedAccessor<float, KM_BATCH_DIM, 2>(out));

    KM_DO_GPU_ASSERT;
    return out;
}
