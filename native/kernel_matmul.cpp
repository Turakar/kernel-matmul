#include <torch/extension.h>

#include "kernel_matmul.h"

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
 * C++ interface
 */

template <int kernel_function_type, int num_params>
torch::Tensor kernel_matmul(torch::Tensor x1, torch::Tensor x2, torch::Tensor rhs,
                            torch::Tensor params, torch::Tensor start, torch::Tensor end) {
    CHECK_INPUT(x1);
    CHECK_INPUT(x2);
    CHECK_INPUT(rhs);
    CHECK_INPUT(params);
    CHECK_INPUT_INT32(start);
    CHECK_INPUT_INT32(end);
    auto out =
        kernel_matmul_cuda<kernel_function_type, num_params>(x1, x2, rhs, params, start, end);
    return out;
}

template <int kernel_function_type, int num_params>
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
    auto out = kernel_bilinear_derivative_cuda<kernel_function_type, num_params>(
        x1, x2, left_vecs, right_vecs, params, start, end);
    return out;
}

template <int kernel_function_type, int num_params>
std::array<torch::Tensor, 2>
kernel_weighted_bilinear_derivative(torch::Tensor x1, torch::Tensor x2, torch::Tensor left_vecs,
                                    torch::Tensor right_vecs, torch::Tensor params,
                                    torch::Tensor start, torch::Tensor end, torch::Tensor weights) {
    CHECK_INPUT(x1);
    CHECK_INPUT(x2);
    CHECK_INPUT(left_vecs);
    CHECK_INPUT(right_vecs);
    CHECK_INPUT(params);
    CHECK_INPUT_INT32(start);
    CHECK_INPUT_INT32(end);
    CHECK_INPUT(weights);
    auto out = kernel_weighted_bilinear_derivative_cuda<kernel_function_type, num_params>(
        x1, x2, left_vecs, right_vecs, params, start, end, weights);
    return out;
}

template <int kernel_function_type, int num_params>
torch::Tensor kernel_matmul_bwd(torch::Tensor x1, torch::Tensor x2, torch::Tensor rhs,
                                torch::Tensor params, torch::Tensor start, torch::Tensor end,
                                torch::Tensor out_grad) {
    CHECK_INPUT(x1);
    CHECK_INPUT(x2);
    CHECK_INPUT(rhs);
    CHECK_INPUT(params);
    CHECK_INPUT_INT32(start);
    CHECK_INPUT_INT32(end);
    CHECK_INPUT(out_grad);
    auto out = kernel_matmul_cuda_bwd<kernel_function_type, num_params>(x1, x2, rhs, params, start,
                                                                        end, out_grad);
    return out;
}

template <int kernel_function_type, int num_params>
torch::Tensor lmc_matmul(torch::Tensor x1, torch::Tensor x2, torch::Tensor rhs,
                         torch::Tensor params, torch::Tensor weights, torch::Tensor start,
                         torch::Tensor end) {
    CHECK_INPUT(x1);
    CHECK_INPUT(x2);
    CHECK_INPUT(rhs);
    CHECK_INPUT(params);
    CHECK_INPUT(weights);
    CHECK_INPUT_INT32(start);
    CHECK_INPUT_INT32(end);
    auto out =
        lmc_matmul_cuda<kernel_function_type, num_params>(x1, x2, rhs, params, weights, start, end);
    return out;
}

/*
 * Python bindings
 */

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "kernel_matmul_locally_periodic",
        &kernel_matmul<LOCALLY_PERIODIC, LOCALLY_PERIODIC_NUM_PARAMS>,
        "Performs matmul with a locally periodic kernel without computing the full kernel matrix.");
    m.def("kernel_matmul_rbf", &kernel_matmul<RBF, RBF_NUM_PARAMS>,
          "Performs matmul with an RBF kernel without computing the full kernel matrix.");
    m.def("kernel_matmul_spectral", &kernel_matmul<SPECTRAL, SPECTRAL_NUM_PARAMS>,
          "Performs matmul with a spectral kernel without computing the full kernel matrix.");

    m.def("kernel_bilinear_derivative_locally_periodic",
          &kernel_bilinear_derivative<LOCALLY_PERIODIC, LOCALLY_PERIODIC_NUM_PARAMS>,
          "Computes the derivative of the bilinear form with respect to the parameters of a "
          "locally periodic kernel.");
    m.def("kernel_bilinear_derivative_rbf", &kernel_bilinear_derivative<RBF, RBF_NUM_PARAMS>,
          "Computes the derivative of the bilinear form with respect to the parameters of an RBF "
          "kernel.");
    m.def("kernel_bilinear_derivative_spectral",
          &kernel_bilinear_derivative<SPECTRAL, SPECTRAL_NUM_PARAMS>,
          "Computes the derivative of the bilinear form with respect to the parameters of a "
          "spectral kernel.");

    m.def("kernel_weighted_bilinear_derivative_locally_periodic",
          &kernel_weighted_bilinear_derivative<LOCALLY_PERIODIC, LOCALLY_PERIODIC_NUM_PARAMS>,
          "Computes the derivative of the bilinear form with respect to the parameters of a "
          "locally periodic kernel with task covariance.");
    m.def("kernel_weighted_bilinear_derivative_rbf",
          &kernel_weighted_bilinear_derivative<RBF, RBF_NUM_PARAMS>,
          "Computes the derivative of the bilinear form with respect to the parameters of an RBF "
          "kernel with task covariance.");
    m.def("kernel_weighted_bilinear_derivative_spectral",
          &kernel_weighted_bilinear_derivative<SPECTRAL, SPECTRAL_NUM_PARAMS>,
          "Computes the derivative of the bilinear form with respect to the parameters of a "
          "spectral kernel with task covariance.");

    m.def("kernel_matmul_locally_periodic_bwd",
          &kernel_matmul_bwd<LOCALLY_PERIODIC, LOCALLY_PERIODIC_NUM_PARAMS>,
          "Backward pass for kernel_matmul_locally_periodic().");
    m.def("kernel_matmul_rbf_bwd", &kernel_matmul_bwd<RBF, RBF_NUM_PARAMS>,
          "Backward pass for kernel_matmul_rbf().");
    m.def("kernel_matmul_spectral_bwd", &kernel_matmul_bwd<SPECTRAL, SPECTRAL_NUM_PARAMS>,
          "Backward pass for kernel_matmul_spectral().");

    m.def("lmc_matmul_locally_periodic", &lmc_matmul<LOCALLY_PERIODIC, LOCALLY_PERIODIC_NUM_PARAMS>,
          "Performs matmul with an LMC model of a locally periodic kernel without computing the "
          "full kernel matrix.");
    m.def("lmc_matmul_rbf", &lmc_matmul<RBF, RBF_NUM_PARAMS>,
          "Performs matmul with an LMC model of an RBF kernel without computing the full kernel "
          "matrix.");
    m.def("lmc_matmul_spectral", &lmc_matmul<SPECTRAL, SPECTRAL_NUM_PARAMS>,
          "Performs matmul with an LMC model of a spectral kernel without computing the full "
          "kernel matrix.");
}
