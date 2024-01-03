#pragma once

#ifndef KM_BLOCK_SIZE
#error "KM_BLOCK_SIZE must be defined"
#endif

#ifndef KM_BATCH_DIM
#error "KM_BATCH_DIM must be defined"
#endif

#ifndef KM_MATMUL_BWD_THREAD_DIM
#error "KM_MATMUL_BWD_THREAD_DIM must be defined"
#endif

#ifndef KM_MATMUL_BWD_PER_THREAD
#error "KM_MATMUL_BWD_PER_THREAD must be defined"
#endif

#include <torch/types.h>

torch::Tensor kernel_matmul_bwd_cuda(torch::Tensor x1, torch::Tensor x2, torch::Tensor rhs,
                                     torch::Tensor params, torch::Tensor start, torch::Tensor end,
                                     torch::Tensor out_grad);
