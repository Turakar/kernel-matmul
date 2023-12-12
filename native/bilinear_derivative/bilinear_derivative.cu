#include "../common/gpu_assert.cuh"
#include "../common/kernel_function.cuh"
#include "../common/utils.h"
#include "bilinear_derivative.h"

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

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
    const int block_size = KM_BLOCK_SIZE;
    const int thread_dim = KM_BILINEAR_DERIVATIVE_THREAD_DIM;
    const int per_thread = KM_BILINEAR_DERIVATIVE_PER_THREAD;
    const int num_params = KM_NUM_PARAMS;
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
                const auto grads = kernel_function_bwd(x1_reg[i], x2_reg[j], params_reg);
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

torch::Tensor kernel_bilinear_derivative_cuda(torch::Tensor x1, torch::Tensor x2,
                                              torch::Tensor left_vecs, torch::Tensor right_vecs,
                                              torch::Tensor params, torch::Tensor start,
                                              torch::Tensor end) {
    const int block_size = KM_BLOCK_SIZE;

    const int thread_dim = KM_BILINEAR_DERIVATIVE_THREAD_DIM;
    const int per_thread = KM_BILINEAR_DERIVATIVE_PER_THREAD;

    const dim3 threads{thread_dim, thread_dim, 1};
    const dim3 blocks{KM_CEIL_DIV(x1.size(0), block_size), params.size(1), 1};

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

    kernel_bilinear_derivative_cuda_kernel<<<blocks, threads>>>(
        x1.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
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
