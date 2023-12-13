#pragma once

#ifndef KM_BLOCK_SIZE
#error "KM_BLOCK_SIZE must be defined"
#endif

#ifndef KM_DENSE_BWD_THREAD_DIM
#error "KM_DENSE_BWD_THREAD_DIM must be defined"
#endif

#include <torch/extension.h>

torch::Tensor kernel_dense_bwd_cuda(torch::Tensor x1, torch::Tensor x2, torch::Tensor params,
                                    torch::Tensor start, torch::Tensor end, torch::Tensor out_grad);
