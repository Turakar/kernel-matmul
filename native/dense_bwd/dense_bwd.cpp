#include "dense_bwd.h"
#include "../common/entrypoint.h"

#include <torch/extension.h>

torch::Tensor kernel_dense_bwd(torch::Tensor x1, torch::Tensor x2, torch::Tensor params,
                               torch::Tensor start, torch::Tensor end, torch::Tensor out_grad) {
    check_cuda(x1);
    check_cuda(x2);
    check_cuda(params);
    check_cuda_int32(start);
    check_cuda_int32(end);
    auto out = kernel_dense_bwd_cuda(x1, x2, params, start, end, out_grad);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("call", &kernel_dense_bwd); }
