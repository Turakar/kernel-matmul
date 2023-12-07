#include "../common/entrypoint.h"
#include "../common/utils.h"

#include <algorithm>
#include <array>
#include <torch/extension.h>

std::array<torch::Tensor, 2> make_ranges(const torch::Tensor x1_any, const torch::Tensor x2_any,
                                         const float cutoff, const int block_size) {
    const auto device = x1_any.device();
    const auto x1 = x1_any.detach().cpu();
    const auto x2 = x2_any.detach().cpu();

    const auto rows = KM_CEIL_DIV(x1.size(0), block_size);
    auto start = torch::zeros({rows}, torch::kInt32);
    auto end = torch::zeros({rows}, torch::kInt32);

    for (int i = 0; i < rows; i++) {
        const int row_first = i * block_size;
        int j;
        if (i == 0) {
            j = 0;
        } else {
            j = start[i - 1].item<int>();
        }
        for (; j < x2.size(0); j++) {
            const auto tau = x1[row_first].item<float>() - x2[j].item<float>();
            if (tau <= cutoff) {
                break;
            }
        }
        start[i] = j;

        const int row_last = std::min(row_first + block_size, (int)x1.size(0)) - 1;
        if (i == 0) {
            j = 0;
        } else {
            j = end[i - 1].item<int>();
        }
        if (j < start[i].item<int>()) {
            j = start[i].item<int>();
        }
        for (; j < x2.size(0); j++) {
            const auto tau = x2[j].item<float>() - x1[row_last].item<float>();
            if (tau > cutoff) {
                break;
            }
        }
        end[i] = j;
    }

    return {start.to(device), end.to(device)};
}

std::array<torch::Tensor, 2> make_ranges_symmetric(const torch::Tensor x_any, const float cutoff,
                                                   const int block_size) {
    const auto device = x_any.device();
    const auto x = x_any.detach().cpu();

    const auto rows = KM_CEIL_DIV(x.size(0), block_size);
    auto start = torch::full({rows}, (int)x.size(0), torch::kInt32);
    start[0] = 0;
    auto end = torch::zeros({rows}, torch::kInt32);

    for (int i = 0; i < rows; i++) {
        const int row_last = std::min((i + 1) * block_size, (int)x.size(0)) - 1;
        int j;
        if (i == 0) {
            j = block_size;
        } else {
            j = end[i - 1].item<int>();
        }
        for (; j < x.size(0); j = std::min(j + block_size, (int)x.size(0))) {
            const auto tau = x[j].item<float>() - x[row_last].item<float>();
            if (tau >= cutoff) {
                break;
            }
        }
        end[i] = j;
        for (int k = i; k < KM_CEIL_DIV(j, block_size); k++) {
            start[k] = std::min(start[k].item<int>(), i * block_size);
        }
    }

    return {start.to(device), end.to(device)};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("make_ranges", &make_ranges);
    m.def("make_ranges_symmetric", &make_ranges_symmetric);
}
