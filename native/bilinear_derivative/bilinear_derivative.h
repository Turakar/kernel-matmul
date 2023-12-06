#pragma once

#ifndef KM_BLOCK_SIZE
#error "KM_BLOCK_SIZE must be defined"
#endif

#ifndef KM_BILINEAR_DERIVATIVE_THREAD_DIM
#error "KM_BILINEAR_DERIVATIVE_THREAD_DIM must be defined"
#endif

#ifndef KM_BILINEAR_DERIVATIVE_PER_THREAD
#error "KM_BILINEAR_DERIVATIVE_PER_THREAD must be defined"
#endif

#include <torch/extension.h>

torch::Tensor kernel_bilinear_derivative_cuda(torch::Tensor x1, torch::Tensor x2,
                                              torch::Tensor left_vecs, torch::Tensor right_vecs,
                                              torch::Tensor params, torch::Tensor start,
                                              torch::Tensor end);
