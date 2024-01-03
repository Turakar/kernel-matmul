#include "../common/accessor.cuh"
#include "../common/gpu_assert.cuh"
#include "../common/kernel_function.cuh"
#include "../common/utils.h"
#include "row.h"

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

__global__ void kernel_row_cuda_kernel(const BatchLayout<KM_BATCH_DIM> batch_layout,
                                       const BatchedAccessor<float, KM_BATCH_DIM, 1> x1_batch,
                                       const BatchedAccessor<float, KM_BATCH_DIM, 1> x2_batch,
                                       const int type,
                                       const BatchedAccessor<float, KM_BATCH_DIM, 1> params_batch,
                                       const BatchedAccessor<int, KM_BATCH_DIM, 1> start_batch,
                                       const BatchedAccessor<int, KM_BATCH_DIM, 1> end_batch,
                                       BatchedAccessor<float, KM_BATCH_DIM, 1> out_batch) {

    const auto batch = batch_layout.get_batch(blockIdx.z);
    const auto x1 = x1_batch[batch];
    const auto x2 = x2_batch[batch];
    const auto params = params_batch[batch];
    const auto start = start_batch[batch];
    const auto end = end_batch[batch];
    auto out = out_batch[batch];

    const int block_size = KM_BLOCK_SIZE;
    const int thread_dim = KM_ROW_THREAD_DIM;
    const int num_params = KM_NUM_PARAMS;

    std::array<float, num_params> params_reg;
    for (int i = 0; i < num_params; i++) {
        params_reg[i] = params[i];
    }

    int m;
    int n;
    bool is_valid = true;
    if (type == -1) {
        // diagonal
        m = blockIdx.x * thread_dim + threadIdx.x;
        n = m;
        is_valid = (m < x1.size(0));
    } else {
        // row
        m = type;
        const int start_m = start[m / block_size];
        const int end_m = end[m / block_size];
        n = blockIdx.x * thread_dim + threadIdx.x + start_m;
        is_valid = (n < end_m);
    }

    if (is_valid) {
        const auto x1_reg = x1[m];
        const auto x2_reg = x2[n];
        const auto out_reg = kernel_function(x1_reg, x2_reg, params_reg);
        out[n] = out_reg;
    }
}

torch::Tensor kernel_row_cuda(torch::Tensor x1, torch::Tensor x2, int type, torch::Tensor params,
                              torch::Tensor start, torch::Tensor end) {
    const int block_size = KM_BLOCK_SIZE;
    const int thread_dim = KM_ROW_THREAD_DIM;
    const auto batch_layout = BatchLayout<KM_BATCH_DIM>(x1.sizes().data());

    const auto out_opts =
        torch::TensorOptions().dtype(x1.dtype()).layout(x1.layout()).device(x1.device());
    const auto out_shape = batch_layout.make_shape<1>({x2.size(-1)});
    auto out = torch::zeros(out_shape, out_opts);

    const dim3 blocks{KM_CEIL_DIV(x2.size(-1), thread_dim), 1, batch_layout.num_batches()};
    const dim3 threads{thread_dim, 1, 1};

    kernel_row_cuda_kernel<<<blocks, threads>>>(
        batch_layout, BatchedAccessor<float, KM_BATCH_DIM, 1>(x1),
        BatchedAccessor<float, KM_BATCH_DIM, 1>(x2), type,
        BatchedAccessor<float, KM_BATCH_DIM, 1>(params),
        BatchedAccessor<int, KM_BATCH_DIM, 1>(start), BatchedAccessor<int, KM_BATCH_DIM, 1>(end),
        BatchedAccessor<float, KM_BATCH_DIM, 1>(out));

    KM_DO_GPU_ASSERT;
    return out;
}
