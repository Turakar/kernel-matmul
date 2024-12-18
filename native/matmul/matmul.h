#pragma once

#ifndef KM_BLOCK_SIZE
#error "KM_BLOCK_SIZE must be defined"
#endif

#ifndef KM_BATCH_DIM
#error "KM_BATCH_DIM must be defined"
#endif

#ifndef KM_MATMUL_THREADS
#error "KM_MATMUL_THREADS must be defined"
#endif

#ifndef KM_MATMUL_PER_THREAD
#error "KM_MATMUL_PER_THREAD must be defined"
#endif

#ifndef KM_MATMUL_K_BLOCK_SIZE
#error "KM_MATMUL_K_BLOCK_SIZE must be defined"
#endif

#ifndef KM_MATMUL_USE_SHM
#error "KM_MATMUL_USE_SHM must be defined"
#endif

#ifndef KM_MATMUL_COL_BLOCKS
#error "KM_MATMUL_COL_BLOCKS must be defined"
#endif

#include <torch/types.h>

torch::Tensor kernel_matmul_cuda(torch::Tensor x1, torch::Tensor x2, torch::Tensor rhs,
                                 torch::Tensor params, torch::Tensor start, torch::Tensor end);
