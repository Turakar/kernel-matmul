#pragma once

#ifndef KM_BLOCK_SIZE
#error "KM_BLOCK_SIZE must be defined"
#endif

#ifndef KM_DENSE_THREAD_DIM
#error "KM_DENSE_THREAD_DIM must be defined"
#endif

#include <torch/extension.h>

torch::Tensor kernel_dense_cuda(torch::Tensor x1, torch::Tensor x2, int type, torch::Tensor params,
                                torch::Tensor start, torch::Tensor end);