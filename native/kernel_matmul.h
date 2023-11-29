#include <torch/types.h>

const int LOCALLY_PERIODIC = 0;
const int LOCALLY_PERIODIC_NUM_PARAMS = 4;
const int RBF = 1;
const int RBF_NUM_PARAMS = 2;
const int SPECTRAL = 3;
const int SPECTRAL_NUM_PARAMS = 3;

/*
 * Forward declarations for CUDA functions.
 */

template <int kernel_function_type, int num_params>
torch::Tensor kernel_matmul_cuda(
    torch::Tensor x1,
    torch::Tensor x2,
    torch::Tensor rhs,
    torch::Tensor params,
    torch::Tensor start,
    torch::Tensor end
);

template <int kernel_function_type, int num_params>
torch::Tensor kernel_bilinear_derivative_cuda(
    torch::Tensor x1,
    torch::Tensor x2,
    torch::Tensor left_vecs,
    torch::Tensor right_vecs,
    torch::Tensor params,
    torch::Tensor start,
    torch::Tensor end
);

template <int kernel_function_type, int num_params>
std::array<torch::Tensor, 2> kernel_weighted_bilinear_derivative_cuda(
    torch::Tensor x1,
    torch::Tensor x2,
    torch::Tensor left_vecs,
    torch::Tensor right_vecs,
    torch::Tensor params,
    torch::Tensor start,
    torch::Tensor end,
    torch::Tensor weights
);

template <int kernel_function_type, int num_params>
torch::Tensor kernel_matmul_cuda_bwd(
    torch::Tensor x1,
    torch::Tensor x2,
    torch::Tensor rhs,
    torch::Tensor params,
    torch::Tensor start,
    torch::Tensor end,
    torch::Tensor out_grad
);

template <int kernel_function_type, int num_params>
torch::Tensor lmc_matmul_cuda(
    torch::Tensor x1,
    torch::Tensor x2,
    torch::Tensor rhs,
    torch::Tensor params,
    torch::Tensor weights,
    torch::Tensor start,
    torch::Tensor end
);
