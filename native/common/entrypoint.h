#include <torch/extension.h>

inline void check_cuda(torch::Tensor x) {
    TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
}

inline void check_int32(torch::Tensor x) {
    TORCH_CHECK(x.dtype() == torch::kInt32, "x must be int32");
}

inline void check_cuda_int32(torch::Tensor x) {
    check_cuda(x);
    check_int32(x);
}
