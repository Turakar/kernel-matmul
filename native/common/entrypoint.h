#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                                             \
    CHECK_CUDA(x);                                                                                 \
    CHECK_CONTIGUOUS(x)
#define CHECK_INT32(x) TORCH_CHECK(x.dtype() == torch::kInt32, #x " must be int32")
#define CHECK_INPUT_INT32(x)                                                                       \
    CHECK_INPUT(x);                                                                                \
    CHECK_INT32(x)
