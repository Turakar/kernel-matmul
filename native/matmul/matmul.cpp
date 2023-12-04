#include "matmul.h"
#include "../common/entrypoint.h"

#include <torch/extension.h>

torch::Tensor kernel_matmul(torch::Tensor x1, torch::Tensor x2, torch::Tensor rhs,
                            torch::Tensor params, torch::Tensor start, torch::Tensor end) {
    CHECK_INPUT(x1);
    CHECK_INPUT(x2);
    CHECK_INPUT(rhs);
    CHECK_INPUT(params);
    CHECK_INPUT_INT32(start);
    CHECK_INPUT_INT32(end);
    return kernel_matmul_cuda(x1, x2, rhs, params, start, end);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("call", &kernel_matmul); }
