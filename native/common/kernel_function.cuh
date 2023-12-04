#pragma once

#if defined(KM_KERNEL_RBF)
#define KM_NUM_PARAMS 2
#elif defined(KM_KERNEL_LOCALLY_PERIODIC)
#define KM_NUM_PARAMS 4
#elif defined(KM_KERNEL_SPECTRAL)
#define KM_NUM_PARAMS 3
#else
#define KM_KERNEL_RBF
#define KM_NUM_PARAMS 2
#endif

#include <array>
#include <cuda.h>
#include <cuda_runtime.h>

#define CONSTANT_PI_F 3.141592654f

__device__ __forceinline__ float kernel_function(const float x1, const float x2,
                                                 const std::array<float, KM_NUM_PARAMS> params) {
#if defined(KM_KERNEL_RBF)
    const float lengthscale_rbf = params[0];
    const float outputscale = params[1];
    const float diff = x1 - x2;
    const float rbf = diff * diff / lengthscale_rbf / lengthscale_rbf / ((float)2);
    return outputscale * exp(-rbf);

#elif defined(KM_KERNEL_LOCALLY_PERIODIC)
    const float lengthscale_rbf = params[0];
    const float lengthscale_periodic = params[1];
    const float period_length = params[2];
    const float outputscale = params[3];
    const float diff = x1 - x2;
    const float rbf = diff * diff / lengthscale_rbf / lengthscale_rbf / ((float)2);
    const float periodic_inner = sin(CONSTANT_PI_F * diff / period_length);
    const float periodic = ((float)2) * periodic_inner * periodic_inner / lengthscale_periodic;
    return outputscale * exp(-(rbf + periodic));

#elif defined(KM_KERNEL_SPECTRAL)
    const float lengthscale = params[0];
    const float frequency = params[1];
    const float outputscale = params[2];

    const float diff = x1 - x2;
    const float rbf = diff * diff / lengthscale / lengthscale / ((float)2);
    const float cos_term = cos(2.0f * CONSTANT_PI_F * frequency * diff);
    const float value = outputscale * exp(-rbf) * cos_term;
    return value;

#endif
}

__device__ __forceinline__ std::array<float, KM_NUM_PARAMS>
kernel_function_bwd(const float x1, const float x2, const std::array<float, KM_NUM_PARAMS> params) {

#if defined(KM_KERNEL_RBF)
    const float lengthscale_rbf = params[0];
    const float outputscale = params[1];

    // forward pass
    const float diff = x1 - x2;
    const float diff_sq = diff * diff;
    const float rbf = diff_sq / lengthscale_rbf / lengthscale_rbf / 2;
    const float exp_term = exp(-rbf);
    const float value = outputscale * exp_term;

    // backward pass
    const float lengthscale_rbf_diff =
        value * diff_sq / (lengthscale_rbf * lengthscale_rbf * lengthscale_rbf);
    const float outputscale_diff = exp_term;

    return {lengthscale_rbf_diff, outputscale_diff};

#elif defined(KM_KERNEL_LOCALLY_PERIODIC)
    const float lengthscale_rbf = params[0];
    const float lengthscale_periodic = params[1];
    const float period_length = params[2];
    const float outputscale = params[3];

    // forward pass
    const float diff = x1 - x2;
    const float diff_sq = diff * diff;
    const float rbf = diff_sq / lengthscale_rbf / lengthscale_rbf / 2;
    const float periodic_inner = sin(diff / period_length);
    const float periodic = 2 * periodic_inner * periodic_inner / lengthscale_periodic;
    const float exp_term = exp(-(rbf + periodic));
    const float value = outputscale * exp_term;

    // backward pass
    // wolfram alpha:
    // - D[α*Exp[-(0.5 *(x-y)^2/λ + 2 * (Sin[(x-y)/p])^2/μ)],λ]
    // - D[α*Exp[-(0.5 *(x-y)^2/λ + 2 * (Sin[(x-y)/p])^2/μ)],μ]
    // - D[α*Exp[-(0.5 *(x-y)^2/λ + 2 * (Sin[(x-y)/p])^2/μ)],p]
    // - D[α*Exp[-(0.5 *(x-y)^2/λ + 2 * (Sin[(x-y)/p])^2/μ)],α]
    const float lengthscale_rbf_diff =
        value * diff_sq / (lengthscale_rbf * lengthscale_rbf * lengthscale_rbf);
    const float lengthscale_periodic_diff = value * periodic / lengthscale_periodic;
    const float period_length_diff = value * 2 * sin(2 * diff / period_length) * diff /
                                     (lengthscale_periodic * period_length * period_length);
    const float outputscale_diff = exp_term;

    return {lengthscale_rbf_diff, lengthscale_periodic_diff, period_length_diff, outputscale_diff};

#elif defined(KM_KERNEL_SPECTRAL)
    const float lengthscale = params[0];
    const float frequency = params[1];
    const float outputscale = params[2];

    const float diff = x1 - x2;
    const float rbf = diff * diff / lengthscale / lengthscale / 2.0f;
    const float cos_factor = 2.0f * CONSTANT_PI_F * diff;
    const float cos_inner = cos_factor * frequency;
    const float cos_term = cos(cos_inner);
    const float exp_term = exp(-rbf);
    const float value = outputscale * exp_term * cos_term;

    const float outputscale_grad = exp_term * cos_term;
    const float lengthscale_grad = value * diff * diff / (lengthscale * lengthscale * lengthscale);
    const float period_grad = -outputscale * exp_term * sin(cos_inner) * cos_factor;

    return {lengthscale_grad, period_grad, outputscale_grad};

#endif
}
