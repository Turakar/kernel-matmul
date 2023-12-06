#pragma once

#include <torch/extension.h>

torch::Tensor kernel_bilinear_derivative_cuda(torch::Tensor x1, torch::Tensor x2,
                                              torch::Tensor left_vecs, torch::Tensor right_vecs,
                                              torch::Tensor params, torch::Tensor start,
                                              torch::Tensor end);
