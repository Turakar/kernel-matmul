#include "bilinear_derivative.h"
#include "../common/entrypoint.h"

#include <torch/extension.h>

torch::Tensor kernel_bilinear_derivative(torch::Tensor x1, torch::Tensor x2,
                                         torch::Tensor left_vecs, torch::Tensor right_vecs,
                                         torch::Tensor params, torch::Tensor start,
                                         torch::Tensor end) {
    CHECK_INPUT(x1);
    CHECK_INPUT(x2);
    CHECK_INPUT(left_vecs);
    CHECK_INPUT(right_vecs);
    CHECK_INPUT(params);
    CHECK_INPUT_INT32(start);
    CHECK_INPUT_INT32(end);
    auto out = kernel_bilinear_derivative_cuda(x1, x2, left_vecs, right_vecs, params, start, end);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("call", &kernel_bilinear_derivative); }
