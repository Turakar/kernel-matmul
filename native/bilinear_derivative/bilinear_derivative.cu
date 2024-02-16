#include "../common/accessor.cuh"
#include "../common/gpu_assert.cuh"
#include "../common/kernel_function.cuh"
#include "../common/utils.h"
#include "bilinear_derivative.h"

#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

__global__ void kernel_bilinear_derivative_cuda_kernel(
    const BatchLayout<KM_BATCH_DIM> batch_layout,
    const BatchedAccessor<float, KM_BATCH_DIM, 1> x1_batch,
    const BatchedAccessor<float, KM_BATCH_DIM, 1> x2_batch,
    const BatchedAccessor<float, KM_BATCH_DIM, 2> left_vecs_batch,
    const BatchedAccessor<float, KM_BATCH_DIM, 2> right_vecs_batch,
    const BatchedAccessor<float, KM_BATCH_DIM, 1> params_batch,
    const BatchedAccessor<int, KM_BATCH_DIM, 1> start_batch,
    const BatchedAccessor<int, KM_BATCH_DIM, 1> end_batch,
    BatchedAccessor<float, KM_BATCH_DIM, 4> params_grad_batch) {

    // Load batch
    const auto batch = batch_layout.get_batch(blockIdx.z);
    const auto x1 = x1_batch[batch];
    const auto x2 = x2_batch[batch];
    const auto left_vecs = left_vecs_batch[batch];
    const auto right_vecs = right_vecs_batch[batch];
    const auto params = params_batch[batch];
    const auto start = start_batch[batch];
    const auto end = end_batch[batch];
    auto params_grad = params_grad_batch[batch];

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
        params_reg[i] = params[i];
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
        params_grad[p][blockIdx.x][threadIdx.y][threadIdx.x] = accumulator[p];
    }
}

torch::Tensor kernel_bilinear_derivative_cuda(torch::Tensor x1, torch::Tensor x2,
                                              torch::Tensor left_vecs, torch::Tensor right_vecs,
                                              torch::Tensor params, torch::Tensor start,
                                              torch::Tensor end) {
    const int block_size = KM_BLOCK_SIZE;
    const int thread_dim = KM_BILINEAR_DERIVATIVE_THREAD_DIM;
    const int per_thread = KM_BILINEAR_DERIVATIVE_PER_THREAD;
    const auto batch_layout = BatchLayout<KM_BATCH_DIM>(x1.sizes().data());

    const dim3 threads{thread_dim, thread_dim, 1};
    const dim3 blocks{KM_CEIL_DIV(x1.size(-1), block_size), 1, batch_layout.num_batches()};

    const auto out_opts =
        torch::TensorOptions().dtype(x1.dtype()).layout(x1.layout()).device(x1.device());
    const auto out_shape =
        batch_layout.make_shape<4>({params.size(-1), blocks.x, thread_dim, thread_dim});
    auto out = torch::zeros(out_shape, out_opts);

#ifdef PRINT_SIZE
    printf("m, n, k, b, batch: (%d, %d, %d, %d, %d)\n", x1.size(-1), x2.size(-1),
           left_vecs.size(-2), params.size(1), batch);
    printf("threads: (%d, %d, %d)\n", threads.x, threads.y, threads.z);
    printf("blocks: (%d, %d, %d)\n", blocks.x, blocks.y, blocks.z);
#endif

    const auto left_vecs_t = left_vecs.transpose(-2, -1).contiguous();
    const auto right_vecs_t = right_vecs.transpose(-2, -1).contiguous();

    const auto params_transformed = transform_params(params);

    kernel_bilinear_derivative_cuda_kernel<<<blocks, threads>>>(
        batch_layout, BatchedAccessor<float, KM_BATCH_DIM, 1>(x1),
        BatchedAccessor<float, KM_BATCH_DIM, 1>(x2),
        BatchedAccessor<float, KM_BATCH_DIM, 2>(left_vecs_t),
        BatchedAccessor<float, KM_BATCH_DIM, 2>(right_vecs_t),
        BatchedAccessor<float, KM_BATCH_DIM, 1>(params_transformed),
        BatchedAccessor<int, KM_BATCH_DIM, 1>(start), BatchedAccessor<int, KM_BATCH_DIM, 1>(end),
        BatchedAccessor<float, KM_BATCH_DIM, 4>(out));

#ifdef GPU_ASSERT
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#endif

    return transform_params_grad(params, out.sum({-3, -2, -1}));
}
