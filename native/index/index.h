#pragma once

#ifndef KM_BLOCK_SIZE
#error "KM_BLOCK_SIZE must be defined"
#endif

#ifndef KM_BATCH_DIM
#error "KM_BATCH_DIM must be defined"
#endif

#ifndef KM_INDEX_THREAD_DIM
#error "KM_INDEX_THREAD_DIM must be defined"
#endif

#ifndef KM_INDEX_BATCH_DIM
#error "KM_INDEX_BATCH_DIM must be defined"
#endif

#include <array>
#include <torch/extension.h>

torch::Tensor kernel_index_cuda(torch::Tensor x1, torch::Tensor x2, torch::Tensor params,
                                torch::Tensor start, torch::Tensor end,
                                std::array<torch::Tensor, KM_BATCH_DIM> batch_indices,
                                torch::Tensor row_index, torch::Tensor col_index);