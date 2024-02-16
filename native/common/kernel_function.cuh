#pragma once

#if defined(KM_KERNEL_RBF)
#define KM_NUM_PARAMS 2
#elif defined(KM_KERNEL_LOCALLY_PERIODIC)
#define KM_NUM_PARAMS 4
#elif defined(KM_KERNEL_SPECTRAL)
#define KM_NUM_PARAMS 3
#elif defined(KM_KERNEL_COMPACT)
#ifndef KM_NUM_ORDERS
#error "KM_NUM_ORDERS not defined"
#endif
#define KM_NUM_PARAMS (1 + (KM_NUM_ORDERS) + (KM_NUM_ORDERS) * (KM_NUM_ORDERS))
#else
#error "No kernel function defined"
#endif

#include <array>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#define CONSTANT_PI_F 3.141592654f

__device__ __forceinline__ float kernel_function(const float x1, const float x2,
                                                 const std::array<float, KM_NUM_PARAMS> params) {
#if defined(KM_KERNEL_RBF)

    const float lengthscale = params[0];
    const float outputscale = params[1];

    const float diff = x1 - x2;
    const float diff_sq = diff * diff;
    return outputscale * expf(-diff_sq * lengthscale);

#elif defined(KM_KERNEL_LOCALLY_PERIODIC)

    const float lengthscale_rbf = params[0];
    const float lengthscale_periodic = params[1];
    const float frequency = params[2];
    const float outputscale = params[3];

    const float diff = x1 - x2;
    const float rbf = diff * diff * lengthscale_rbf;
    const float periodic_inner = sinpif(frequency * diff);
    const float periodic = periodic_inner * periodic_inner * lengthscale_periodic;
    return outputscale * exp(-(rbf + periodic));

#elif defined(KM_KERNEL_SPECTRAL)

    const float lengthscale = params[0];
    const float frequency = params[1];
    const float outputscale = params[2];

    const float diff = x1 - x2;
    const float cos_term = cospif(frequency * diff);
    const float value = outputscale * exp(-lengthscale * diff * diff) * cos_term;
    return value;

#elif defined(KM_KERNEL_COMPACT)

    const float cutoff_reciprocal = params[0];
    const float t = fabsf(x1 - x2) * cutoff_reciprocal;
    if (t >= 1.0f) {
        return 0.0f;
    }
    const int num_orders = KM_NUM_ORDERS;
    float result = 0.0f;
    for (int i = 0; i < num_orders; i++) {
        const int order_i = static_cast<int>(params[1 + i]);
        for (int j = 0; j < num_orders; j++) {
            const int order_j = static_cast<int>(params[1 + j]);
            if (i != j) {
                const float cos_term = cospif((order_i + order_j) * t);
                const float sin_term = sinpif((order_i - order_j) * (1 - t));
                const float denom = CONSTANT_PI_F * (order_i - order_j);
                const float base = __fdividef(cos_term * sin_term, denom);
                result += base * params[1 + num_orders + i * num_orders + j];
            } else {
                const float cos_term = cospif((order_i + order_j) * t);
                const float factor = 1 - t;
                const float base = cos_term * factor;
                result += base * params[1 + num_orders + i * num_orders + j];
            }
        }
    }
    return result;

#endif
}

__device__ __forceinline__ std::array<float, KM_NUM_PARAMS>
kernel_function_bwd(const float x1, const float x2, const std::array<float, KM_NUM_PARAMS> params) {

#if defined(KM_KERNEL_RBF)

    const float lengthscale = params[0];
    const float outputscale = params[1];

    // forward pass
    const float diff = x1 - x2;
    const float diff_sq = diff * diff;
    const float rbf = diff_sq * lengthscale;
    const float exp_term = exp(-rbf);
    const float value = outputscale * exp_term;

    // backward pass
    const float lengthscale_grad = -value * diff_sq;
    const float outputscale_grad = exp_term;

    return {lengthscale_grad, outputscale_grad};

#elif defined(KM_KERNEL_LOCALLY_PERIODIC)

    const float lengthscale_rbf = params[0];
    const float lengthscale_periodic = params[1];
    const float frequency = params[2];
    const float outputscale = params[3];

    // forward pass
    const float diff = x1 - x2;
    const float diff_sq = diff * diff;
    const float rbf = diff_sq * lengthscale_rbf;
    const float sin_inner = frequency * diff;
    float sin_value;
    float cos_value;
    sincospif(sin_inner, &sin_value, &cos_value);
    const float sin_value_sq = sin_value * sin_value;
    const float periodic = sin_value_sq * lengthscale_periodic;
    const float exp_term = exp(-(rbf + periodic));
    const float value = outputscale * exp_term;

    // backward pass
    const float lengthscale_rbf_grad = -value * diff_sq;
    const float lengthscale_periodic_grad = -value * sin_value_sq;
    const float frequency_grad =
        -(2.0f * CONSTANT_PI_F) * value * lengthscale_periodic * sin_value * cos_value * diff;
    const float outputscale_grad = exp_term;

    return {lengthscale_rbf_grad, lengthscale_periodic_grad, frequency_grad, outputscale_grad};

#elif defined(KM_KERNEL_SPECTRAL)

    const float lengthscale = params[0];
    const float frequency = params[1];
    const float outputscale = params[2];

    // forward pass
    const float diff = x1 - x2;
    const float diff_sq = diff * diff;
    const float cos_inner = frequency * diff;
    float sin_term;
    float cos_term;
    sincospif(cos_inner, &sin_term, &cos_term);
    const float exp_term = exp(-lengthscale * diff_sq);
    const float unscaled = exp_term * cos_term;
    const float value = outputscale * unscaled;

    // backward pass
    const float lengthscale_grad = -value * diff_sq;
    const float frequency_grad = -CONSTANT_PI_F * outputscale * exp_term * sin_term * diff;
    const float outputscale_grad = unscaled;
    return {lengthscale_grad, frequency_grad, outputscale_grad};

#elif defined(KM_KERNEL_COMPACT)

    const float cutoff_reciprocal = params[0];
    const float t = fabsf(x1 - x2) * cutoff_reciprocal;
    std::array<float, KM_NUM_PARAMS> grads = {};
    if (t >= 1.0f) {
        return grads;
    }
    const int num_orders = KM_NUM_ORDERS;
    float result = 0.0f;
    for (int i = 0; i < num_orders; i++) {
        const int order_i = static_cast<int>(params[1 + i]);
        for (int j = 0; j < num_orders; j++) {
            const int order_j = static_cast<int>(params[1 + j]);
            if (i != j) {
                const float cos_term = cospif((order_i + order_j) * t);
                const float sin_term = sinpif((order_i - order_j) * (1 - t));
                const float denom = CONSTANT_PI_F * (order_i - order_j);
                const float base = __fdividef(cos_term * sin_term, denom);
                grads[1 + num_orders + i * num_orders + j];
            } else {
                const float cos_term = cospif((order_i + order_j) * t);
                const float factor = 1 - t;
                const float base = cos_term * factor;
                grads[1 + num_orders + i * num_orders + j] = base;
            }
        }
    }
    return grads;

#endif
}

torch::Tensor transform_params(const torch::Tensor params) {
    using namespace torch::indexing;
    torch::Tensor transformed = torch::empty_like(params);
#if defined(KM_KERNEL_RBF)
    transformed.index_put_({Ellipsis, 0}, params.index({Ellipsis, 0}).pow(-2.0f).mul(0.5));
    transformed.index_put_({Ellipsis, 1}, params.index({Ellipsis, 1}));
#elif defined(KM_KERNEL_LOCALLY_PERIODIC)
    transformed.index_put_({Ellipsis, 0}, params.index({Ellipsis, 0}).pow(-2.0f).mul(0.5));
    transformed.index_put_({Ellipsis, 1}, params.index({Ellipsis, 1}).reciprocal().mul(2.0f));
    transformed.index_put_({Ellipsis, 2}, params.index({Ellipsis, 2}).reciprocal());
    transformed.index_put_({Ellipsis, 3}, params.index({Ellipsis, 3}));
#elif defined(KM_KERNEL_SPECTRAL)
    transformed.index_put_({Ellipsis, 0}, params.index({Ellipsis, 0}).pow(-2.0f).mul(0.5));
    transformed.index_put_({Ellipsis, 1}, params.index({Ellipsis, 1}).mul(2.0f));
    transformed.index_put_({Ellipsis, 2}, params.index({Ellipsis, 2}));
#elif defined(KM_KERNEL_COMPACT)
    transformed.index_put_({Ellipsis, 0}, params.index({Ellipsis, 0}).reciprocal());
    transformed.index_put_({Ellipsis, Slice(1, None, None)},
                           params.index({Ellipsis, Slice(1, None, None)}));
#endif
    return transformed;
}

torch::Tensor transform_params_grad(const torch::Tensor params, const torch::Tensor grad) {
    using namespace torch::indexing;
    torch::Tensor transformed = torch::empty_like(grad);
#if defined(KM_KERNEL_RBF)
    transformed.index_put_(
        {Ellipsis, 0},
        grad.index({Ellipsis, 0}).mul(params.index({Ellipsis, 0}).pow(-3.0f).mul(-1.0f)));
    transformed.index_put_({Ellipsis, 1}, grad.index({Ellipsis, 1}));
#elif defined(KM_KERNEL_LOCALLY_PERIODIC)
    transformed.index_put_(
        {Ellipsis, 0},
        grad.index({Ellipsis, 0}).mul(params.index({Ellipsis, 0}).pow(-3.0f).mul(-1.0f)));
    transformed.index_put_(
        {Ellipsis, 1},
        grad.index({Ellipsis, 1}).mul(params.index({Ellipsis, 1}).pow(-2.0f).mul(-2.0f)));
    transformed.index_put_(
        {Ellipsis, 2},
        grad.index({Ellipsis, 2}).mul(params.index({Ellipsis, 2}).pow(-2.0f).mul(-1.0f)));
    transformed.index_put_({Ellipsis, 3}, grad.index({Ellipsis, 3}));
#elif defined(KM_KERNEL_SPECTRAL)
    transformed.index_put_(
        {Ellipsis, 0},
        grad.index({Ellipsis, 0}).mul(params.index({Ellipsis, 0}).pow(-3.0f).mul(-1.0f)));
    transformed.index_put_({Ellipsis, 1}, grad.index({Ellipsis, 1}).mul(2.0f));
    transformed.index_put_({Ellipsis, 2}, grad.index({Ellipsis, 2}));
#elif defined(KM_KERNEL_COMPACT)
    transformed.index_put_({Ellipsis, Slice(1 + KM_NUM_ORDERS, None, None)},
                           grad.index({Ellipsis, Slice(1 + KM_NUM_ORDERS, None, None)}));
#endif
    return transformed;
}
