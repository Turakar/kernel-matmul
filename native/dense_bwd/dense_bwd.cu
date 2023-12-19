#include "../common/gpu_assert.cuh"
#include "../common/kernel_function.cuh"
#include "../common/utils.h"
#include "dense_bwd.h"

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

__global__ void kernel_dense_bwd_cuda_kernel(
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> x1,
    const torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> x2,
    const torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> params,
    const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> start,
    const torch::PackedTensorAccessor32<int, 2, torch::RestrictPtrTraits> end,
    const torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> out_grad,
    torch::PackedTensorAccessor32<float, 6, torch::RestrictPtrTraits> out) {

    const int block_size = KM_BLOCK_SIZE;
    const int thread_dim = KM_DENSE_BWD_THREAD_DIM;
    const int b = blockIdx.y;
    const int batch = blockIdx.z;
    const int num_params = KM_NUM_PARAMS;
    static_assert(block_size % thread_dim == 0, "block_size must be divisible by thread_dim");
    const int per_thread = block_size / thread_dim;
    const int m_base = blockIdx.x * block_size;

    std::array<float, num_params> params_reg;
    for (int i = 0; i < num_params; i++) {
        params_reg[i] = params[batch][b][i];
    }

    std::array<float, num_params> accumulator = {};

    const int start_reg = start[batch][blockIdx.x];
    const int end_reg = end[batch][blockIdx.x];
    for (int n_base = start_reg; n_base < end_reg; n_base += block_size) {
        std::array<float, per_thread> x1_reg;
        for (int i = 0; i < per_thread; i++) {
            const int m = m_base + threadIdx.y + i * thread_dim;
            if (m < x1.size(1)) {
                x1_reg[i] = x1[batch][m];
            } else {
                x1_reg[i] = 0;
            }
        }
        std::array<float, per_thread> x2_reg;
        for (int i = 0; i < per_thread; i++) {
            const int n = n_base + threadIdx.x + i * thread_dim;
            if (n < end_reg) {
                x2_reg[i] = x2[batch][n];
            } else {
                x2_reg[i] = 0;
            }
        }
        for (int i = 0; i < per_thread; i++) {
            for (int j = 0; j < per_thread; j++) {
                const int m = m_base + threadIdx.y + i * thread_dim;
                const int n = n_base + threadIdx.x + j * thread_dim;
                if (m < x1.size(1) && n < end_reg) {
                    auto out_grad_reg = out_grad[batch][b][m][n];
                    auto grads = kernel_function_bwd(x1_reg[i], x2_reg[j], params_reg);
                    for (int p = 0; p < num_params; p++) {
                        accumulator[p] += out_grad_reg * grads[p];
                    }
                }
            }
        }
    }

    for (int p = 0; p < num_params; p++) {
        out[batch][b][p][blockIdx.x][threadIdx.y][threadIdx.x] = accumulator[p];
    }
}

torch::Tensor kernel_dense_bwd_cuda(torch::Tensor x1, torch::Tensor x2, torch::Tensor params,
                                    torch::Tensor start, torch::Tensor end,
                                    torch::Tensor out_grad) {
    const int block_size = KM_BLOCK_SIZE;
    const int thread_dim = KM_DENSE_BWD_THREAD_DIM;
    const int b = params.size(1);
    const int batch = params.size(0);

    const dim3 blocks{KM_CEIL_DIV(x1.size(1), block_size), b, batch};
    const dim3 threads{thread_dim, thread_dim, 1};

    const auto out_opts =
        torch::TensorOptions().dtype(x1.dtype()).layout(x1.layout()).device(x1.device());
    auto out = torch::zeros({batch, b, params.size(2), blocks.x, threads.y, threads.x}, out_opts);

    kernel_dense_bwd_cuda_kernel<<<blocks, threads>>>(
        x1.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        x2.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
        params.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
        start.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
        end.packed_accessor32<int, 2, torch::RestrictPtrTraits>(),
        out_grad.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
        out.packed_accessor32<float, 6, torch::RestrictPtrTraits>());

    KM_DO_GPU_ASSERT;
    return out.sum({3, 4, 5});
}