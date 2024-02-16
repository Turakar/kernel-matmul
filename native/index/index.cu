#include "../common/accessor.cuh"
#include "../common/gpu_assert.cuh"
#include "../common/kernel_function.cuh"
#include "../common/utils.h"
#include "index.h"

#include <array>
#include <cuda.h>
#include <cuda_runtime.h>
#include <utility>

__global__ void kernel_index_cuda_kernel(
    BatchedAccessor<float, KM_BATCH_DIM, 1> x1_batch,
    BatchedAccessor<float, KM_BATCH_DIM, 1> x2_batch,
    BatchedAccessor<float, KM_BATCH_DIM, 1> params_batch,
    BatchedAccessor<int, KM_BATCH_DIM, 1> start_batch,
    BatchedAccessor<int, KM_BATCH_DIM, 1> end_batch,
    BatchLayout<KM_INDEX_BATCH_DIM> index_batch_layout,
    std::array<BatchedAccessor<int, KM_INDEX_BATCH_DIM, 1>, KM_BATCH_DIM> batch_indices_batch,
    BatchedAccessor<int, KM_INDEX_BATCH_DIM, 1> row_index_batch,
    BatchedAccessor<int, KM_INDEX_BATCH_DIM, 1> col_index_batch,
    BatchedAccessor<float, KM_INDEX_BATCH_DIM, 1> result_batch) {

    // Retrieve the indices we process in the current thread
    const auto index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index >= row_index_batch.size(-1)) {
        return;
    }
    const auto index_batch = index_batch_layout.get_batch(blockIdx.z);
    const auto row_index = row_index_batch[index_batch][index];
    const auto col_index = col_index_batch[index_batch][index];
    std::array<int, KM_BATCH_DIM> batch_indices;
    for (int i = 0; i < KM_BATCH_DIM; i++) {
        batch_indices[i] = batch_indices_batch[i][index_batch][index];
    }
    const auto block = row_index / KM_BLOCK_SIZE;

    // Retrieve the range
    const auto start = start_batch[batch_indices][block];
    const auto end = end_batch[batch_indices][block];

    // Check range
    if (start <= col_index && col_index < end) {

        // Retrieve the inputs
        const auto x1 = x1_batch[batch_indices][row_index];
        const auto x2 = x2_batch[batch_indices][col_index];
        std::array<float, KM_NUM_PARAMS> params_reg;
        const auto params = params_batch[batch_indices];
        for (int i = 0; i < KM_NUM_PARAMS; i++) {
            params_reg[i] = params[i];
        }

        // Compute the kernel function
        result_batch[index_batch][index] = kernel_function(x1, x2, params_reg);
    }
}

// We need this function later to convert our array of batch index Tensors to an array of batch
// accessors without malloc. Here, we use that make_integer_sequence() can be used as a variadic
// template parameter, which can then be used for a parameter pack.
template <int... Is>
std::array<BatchedAccessor<int, KM_INDEX_BATCH_DIM, 1>, sizeof...(Is)>
make_batch_accessors(std::array<torch::Tensor, sizeof...(Is)> batch_indices,
                     std::integer_sequence<int, Is...>) {
    return {{(BatchedAccessor<int, KM_INDEX_BATCH_DIM, 1>(batch_indices[Is]))...}};
}

torch::Tensor kernel_index_cuda(torch::Tensor x1, torch::Tensor x2, torch::Tensor params,
                                torch::Tensor start, torch::Tensor end,
                                std::array<torch::Tensor, KM_BATCH_DIM> batch_indices,
                                torch::Tensor row_index, torch::Tensor col_index) {
    const int thread_dim = KM_INDEX_THREAD_DIM;
    const auto index_batch_layout = BatchLayout<KM_INDEX_BATCH_DIM>(row_index.sizes().data());

    const dim3 blocks{static_cast<uint32_t>(KM_CEIL_DIV(row_index.size(-1), thread_dim)), 1,
                      index_batch_layout.num_batches()};
    const dim3 threads{static_cast<uint32_t>(thread_dim), 1, 1};

    const auto out_opts =
        torch::TensorOptions().dtype(x1.dtype()).layout(x1.layout()).device(x1.device());
    const auto out_shape = index_batch_layout.make_shape<1>({row_index.size(-1)});
    auto out = torch::zeros(out_shape, out_opts);

    const auto batch_indices_batch = make_batch_accessors(
        batch_indices, std::make_integer_sequence<int, static_cast<int>(KM_BATCH_DIM)>{});

    const auto params_transformed = transform_params(params);

    kernel_index_cuda_kernel<<<blocks, threads>>>(
        BatchedAccessor<float, KM_BATCH_DIM, 1>(x1), BatchedAccessor<float, KM_BATCH_DIM, 1>(x2),
        BatchedAccessor<float, KM_BATCH_DIM, 1>(params_transformed),
        BatchedAccessor<int, KM_BATCH_DIM, 1>(start), BatchedAccessor<int, KM_BATCH_DIM, 1>(end),
        index_batch_layout, batch_indices_batch,
        BatchedAccessor<int, KM_INDEX_BATCH_DIM, 1>(row_index),
        BatchedAccessor<int, KM_INDEX_BATCH_DIM, 1>(col_index),
        BatchedAccessor<float, KM_INDEX_BATCH_DIM, 1>(out));

    KM_DO_GPU_ASSERT;
    return out;
}
