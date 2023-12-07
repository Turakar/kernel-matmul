#include "../common/gpu_assert.cuh"
#include "../common/kernel_function.cuh"
#include "../common/utils.h"
#include "dense.h"

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

__global__ void kernel_dense_cuda_kernel(
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> x1,
    const torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> x2, const int type,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> params,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> start,
    const torch::PackedTensorAccessor32<int, 1, torch::RestrictPtrTraits> end,
    torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> out) {

    const int block_size = KM_BLOCK_SIZE;
    const int thread_dim = KM_DENSE_THREAD_DIM;
    const int b = blockIdx.y;
    const int num_params = KM_NUM_PARAMS;

    std::array<float, num_params> params_reg;
    for (int i = 0; i < num_params; i++) {
        params_reg[i] = params[i][b];
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
        out[b][n] = out_reg;
    }
}

torch::Tensor kernel_dense_cuda(torch::Tensor x1, torch::Tensor x2, int type, torch::Tensor params,
                                torch::Tensor start, torch::Tensor end) {
    const int block_size = KM_BLOCK_SIZE;
    const int thread_dim = KM_DENSE_THREAD_DIM;
    const int b = params.size(1);

    const auto out_opts =
        torch::TensorOptions().dtype(x1.dtype()).layout(x1.layout()).device(x1.device());
    auto out = torch::zeros({b, x2.size(0)}, out_opts);

    const dim3 blocks{KM_CEIL_DIV(x2.size(0), thread_dim), b, 1};
    const dim3 threads{thread_dim, 1, 1};

    kernel_dense_cuda_kernel<<<blocks, threads>>>(
        x1.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
        x2.packed_accessor32<float, 1, torch::RestrictPtrTraits>(), type,
        params.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        start.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        end.packed_accessor32<int, 1, torch::RestrictPtrTraits>(),
        out.packed_accessor32<float, 2, torch::RestrictPtrTraits>());

    KM_DO_GPU_ASSERT;
    return out;
}
