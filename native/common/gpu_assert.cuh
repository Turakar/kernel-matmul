#pragma once

#include "utils.h"
#include <stdio.h>

#ifdef KM_DEBUG_GPU_ASSERT

#define KM_DO_GPU_ASSERT                                                                           \
    gpuAssert(cudaPeekAtLastError(), __FILE__, __LINE__);                                          \
    gpuAssert(cudaDeviceSynchronize(), __FILE__, __LINE__)

inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        auto message = string_format("GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        throw std::runtime_error(message);
    }
}

#else

#define KM_DO_GPU_ASSERT

#endif
