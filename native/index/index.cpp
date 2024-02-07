#include "index.h"
#include "../common/entrypoint.h"

#include <torch/extension.h>

torch::Tensor kernel_index(torch::Tensor x1, torch::Tensor x2, torch::Tensor params,
                           torch::Tensor start, torch::Tensor end,
                           std::array<torch::Tensor, KM_BATCH_DIM> batch_indices,
                           torch::Tensor row_index, torch::Tensor col_index) {
    check_cuda(x1);
    check_cuda(x2);
    check_cuda(params);
    check_cuda_int32(start);
    check_cuda_int32(end);
    for (auto &batch_index : batch_indices) {
        check_cuda_int32(batch_index);
    }
    check_cuda_int32(row_index);
    check_cuda_int32(col_index);
    auto out = kernel_index_cuda(x1, x2, params, start, end, batch_indices, row_index, col_index);
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("call", &kernel_index); }