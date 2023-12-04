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

/*
 * Forward definitions
 */

torch::Tensor mosm_vecmul_cuda(torch::Tensor x1, torch::Tensor params1, torch::Tensor x2,
                               torch::Tensor params2, torch::Tensor rhs, torch::Tensor start,
                               torch::Tensor end, torch::Tensor tasks, torch::Tensor tasks_block);

#ifdef MOSM_MATMUL_K_SIZE
torch::Tensor mosm_matmul_cuda(torch::Tensor x1, torch::Tensor params1, torch::Tensor x2,
                               torch::Tensor params2, torch::Tensor rhs, torch::Tensor start,
                               torch::Tensor end, torch::Tensor tasks, torch::Tensor tasks_block);

std::array<torch::Tensor, 2> mosm_matmul_bwd_cuda(torch::Tensor x1, torch::Tensor params1,
                                                  torch::Tensor x2, torch::Tensor params2,
                                                  torch::Tensor rhs, torch::Tensor start,
                                                  torch::Tensor end, torch::Tensor tasks,
                                                  torch::Tensor tasks_block,
                                                  torch::Tensor out_grad);
#endif

std::array<torch::Tensor, 2>
mosm_bilinear_derivative_cuda(torch::Tensor x1, torch::Tensor params1, torch::Tensor x2,
                              torch::Tensor params2, torch::Tensor start, torch::Tensor end,
                              torch::Tensor tasks, torch::Tensor tasks_block,
                              torch::Tensor left_vectors, torch::Tensor right_vectors);

torch::Tensor mosm_dense_cuda(torch::Tensor x1, torch::Tensor params1, torch::Tensor x2,
                              torch::Tensor params2, torch::Tensor start, torch::Tensor end,
                              torch::Tensor tasks, torch::Tensor tasks_block, torch::Tensor index1,
                              torch::Tensor index2);

std::array<torch::Tensor, 2> mosm_dense_bwd_cuda(torch::Tensor x1, torch::Tensor params1,
                                                 torch::Tensor x2, torch::Tensor params2,
                                                 torch::Tensor start, torch::Tensor end,
                                                 torch::Tensor tasks, torch::Tensor tasks_block,
                                                 torch::Tensor index1, torch::Tensor index2,
                                                 torch::Tensor out_grad);

/*
 * C++ interface
 */

torch::Tensor mosm_vecmul(torch::Tensor x1, torch::Tensor params1, torch::Tensor x2,
                          torch::Tensor params2, torch::Tensor rhs, torch::Tensor start,
                          torch::Tensor end, torch::Tensor tasks, torch::Tensor tasks_block) {
    CHECK_INPUT(x1);
    CHECK_INPUT(params1);
    CHECK_INPUT(x2);
    CHECK_INPUT(params2);
    CHECK_CUDA(rhs);
    CHECK_INPUT_INT32(start);
    CHECK_INPUT_INT32(end);
    CHECK_INPUT_INT32(tasks);
    CHECK_INPUT_INT32(tasks_block);
    auto out = mosm_vecmul_cuda(x1, params1, x2, params2, rhs, start, end, tasks, tasks_block);
    return out;
}

#ifdef MOSM_MATMUL_K_SIZE
torch::Tensor mosm_matmul(torch::Tensor x1, torch::Tensor params1, torch::Tensor x2,
                          torch::Tensor params2, torch::Tensor rhs, torch::Tensor start,
                          torch::Tensor end, torch::Tensor tasks, torch::Tensor tasks_block) {
    CHECK_INPUT(x1);
    CHECK_INPUT(params1);
    CHECK_INPUT(x2);
    CHECK_INPUT(params2);
    CHECK_CUDA(rhs);
    CHECK_INPUT_INT32(start);
    CHECK_INPUT_INT32(end);
    CHECK_INPUT_INT32(tasks);
    CHECK_INPUT_INT32(tasks_block);
    auto out = mosm_matmul_cuda(x1, params1, x2, params2, rhs, start, end, tasks, tasks_block);
    return out;
}

std::array<torch::Tensor, 2> mosm_matmul_bwd(torch::Tensor x1, torch::Tensor params1,
                                             torch::Tensor x2, torch::Tensor params2,
                                             torch::Tensor rhs, torch::Tensor start,
                                             torch::Tensor end, torch::Tensor tasks,
                                             torch::Tensor tasks_block, torch::Tensor out_grad) {
    CHECK_INPUT(x1);
    CHECK_INPUT(params1);
    CHECK_INPUT(x2);
    CHECK_INPUT(params2);
    CHECK_CUDA(rhs);
    CHECK_INPUT_INT32(start);
    CHECK_INPUT_INT32(end);
    CHECK_INPUT_INT32(tasks);
    CHECK_INPUT_INT32(tasks_block);
    CHECK_INPUT(out_grad);
    auto out = mosm_matmul_bwd_cuda(x1, params1, x2, params2, rhs, start, end, tasks, tasks_block,
                                    out_grad);
    return out;
}
#endif

std::array<torch::Tensor, 2>
mosm_bilinear_derivative(torch::Tensor x1, torch::Tensor params1, torch::Tensor x2,
                         torch::Tensor params2, torch::Tensor start, torch::Tensor end,
                         torch::Tensor tasks, torch::Tensor tasks_block, torch::Tensor left_vectors,
                         torch::Tensor right_vectors) {
    CHECK_INPUT(x1);
    CHECK_INPUT(params1);
    CHECK_INPUT(x2);
    CHECK_INPUT(params2);
    CHECK_INPUT_INT32(start);
    CHECK_INPUT_INT32(end);
    CHECK_INPUT_INT32(tasks);
    CHECK_INPUT_INT32(tasks_block);
    CHECK_INPUT(left_vectors);
    CHECK_INPUT(right_vectors);
    auto out = mosm_bilinear_derivative_cuda(x1, params1, x2, params2, start, end, tasks,
                                             tasks_block, left_vectors, right_vectors);
    return out;
}

torch::Tensor mosm_dense(torch::Tensor x1, torch::Tensor params1, torch::Tensor x2,
                         torch::Tensor params2, torch::Tensor start, torch::Tensor end,
                         torch::Tensor tasks, torch::Tensor tasks_block, torch::Tensor index1,
                         torch::Tensor index2) {
    CHECK_INPUT(x1);
    CHECK_INPUT(params1);
    CHECK_INPUT(x2);
    CHECK_INPUT(params2);
    CHECK_INPUT_INT32(start);
    CHECK_INPUT_INT32(end);
    CHECK_INPUT_INT32(tasks);
    CHECK_INPUT_INT32(tasks_block);
    CHECK_INPUT_INT32(index1);
    CHECK_INPUT_INT32(index2);
    auto out =
        mosm_dense_cuda(x1, params1, x2, params2, start, end, tasks, tasks_block, index1, index2);
    return out;
}

std::array<torch::Tensor, 2> mosm_dense_bwd(torch::Tensor x1, torch::Tensor params1,
                                            torch::Tensor x2, torch::Tensor params2,
                                            torch::Tensor start, torch::Tensor end,
                                            torch::Tensor tasks, torch::Tensor tasks_block,
                                            torch::Tensor index1, torch::Tensor index2,
                                            torch::Tensor out_grad) {
    CHECK_INPUT(x1);
    CHECK_INPUT(params1);
    CHECK_INPUT(x2);
    CHECK_INPUT(params2);
    CHECK_INPUT_INT32(start);
    CHECK_INPUT_INT32(end);
    CHECK_INPUT_INT32(tasks);
    CHECK_INPUT_INT32(tasks_block);
    CHECK_INPUT_INT32(index1);
    CHECK_INPUT_INT32(index2);
    CHECK_INPUT(out_grad);
    auto out = mosm_dense_bwd_cuda(x1, params1, x2, params2, start, end, tasks, tasks_block, index1,
                                   index2, out_grad);
    return out;
}

/*
 * Python bindings
 */

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mosm_vecmul", &mosm_vecmul,
          "Performs vecmul with a MOSM kernel without computing the full kernel matrix.");
#ifdef MOSM_MATMUL_K_SIZE
    m.def("mosm_matmul", &mosm_matmul,
          "Performs matmul with a MOSM kernel without computing the full kernel matrix.");
    m.def("mosm_matmul_bwd", &mosm_matmul_bwd, "Backward pass for mosm_matmul().");
#endif
    m.def("mosm_bilinear_derivative", &mosm_bilinear_derivative,
          "Computes the derivative of a bilinear form with respect to the parameters.");
    m.def("mosm_dense", &mosm_dense, "Computes the full MOSM matrix for the given indices.");
    m.def("mosm_dense_bwd", &mosm_dense_bwd, "Backward pass for mosm_dense().");
}
