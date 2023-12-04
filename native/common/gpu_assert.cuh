#pragma once

#include <stdio.h>

#ifdef KM_DEBUG_GPU_ASSERT

#define KM_DO_GPU_ASSERT                                                                           \
    gpuAssert(cudaPeekAtLastError(), __FILE__, __LINE__);                                          \
    gpuAssert(cudaDeviceSynchronize(), __FILE__, __LINE__)

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

#else

#define KM_DO_GPU_ASSERT

#endif
