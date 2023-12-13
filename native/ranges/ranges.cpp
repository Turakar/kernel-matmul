#include "../common/entrypoint.h"
#include "../common/utils.h"

#include <algorithm>
#include <array>
#include <torch/extension.h>

std::array<torch::Tensor, 2> make_ranges(const torch::Tensor x1_any, const torch::Tensor x2_any,
                                         const float cutoff, const int block_size) {
    const auto device = x1_any.device();
    const auto x1 = x1_any.detach().cpu();
    const auto x1_ = x1.accessor<float, 2>();
    const auto x2 = x2_any.detach().cpu();
    const auto x2_ = x2.accessor<float, 2>();

    const auto num_batch = x1.size(0);
    const auto rows = KM_CEIL_DIV(x1.size(1), block_size);
    auto start = torch::zeros({num_batch, rows}, torch::kInt32);
    auto start_ = start.accessor<int, 2>();
    auto end = torch::zeros({num_batch, rows}, torch::kInt32);
    auto end_ = end.accessor<int, 2>();

    for (int batch = 0; batch < num_batch; batch++) {
        for (int i = 0; i < rows; i++) {
            const int row_first = i * block_size;
            int j;
            if (i == 0) {
                j = 0;
            } else {
                j = start_[batch][i - 1];
            }
            for (; j < x2.size(1); j++) {
                const auto tau = x1_[batch][row_first] - x2_[batch][j];
                if (tau <= cutoff) {
                    break;
                }
            }
            start_[batch][i] = j;

            const int row_last = std::min(row_first + block_size, (int)x1.size(1)) - 1;
            if (i == 0) {
                j = 0;
            } else {
                j = end_[batch][i - 1];
            }
            if (j < start_[batch][i]) {
                j = start_[batch][i];
            }
            for (; j < x2.size(1); j++) {
                const auto tau = x2_[batch][j] - x1_[batch][row_last];
                if (tau > cutoff) {
                    break;
                }
            }
            end_[batch][i] = j;
        }
    }

    return {start.to(device), end.to(device)};
}

std::array<torch::Tensor, 2> make_ranges_symmetric(const torch::Tensor x_any, const float cutoff,
                                                   const int block_size) {
    const auto device = x_any.device();
    const auto x = x_any.detach().cpu();
    const auto x_ = x.accessor<float, 2>();

    const auto num_batch = x.size(0);
    const auto rows = KM_CEIL_DIV(x.size(1), block_size);
    auto start = torch::full({num_batch, rows}, (int)x.size(1), torch::kInt32);
    auto start_ = start.accessor<int, 2>();
    auto end = torch::zeros({num_batch, rows}, torch::kInt32);
    auto end_ = end.accessor<int, 2>();

    for (int batch = 0; batch < num_batch; batch++) {
        start[batch][0] = 0;
        for (int i = 0; i < rows; i++) {
            const int row_last = std::min((i + 1) * block_size, (int)x_.size(1)) - 1;
            int j;
            if (i == 0) {
                j = block_size;
            } else {
                j = end_[batch][i - 1];
            }
            for (; j < x.size(1); j = std::min(j + block_size, (int)x.size(1))) {
                const auto tau = x_[batch][j] - x_[batch][row_last];
                if (tau >= cutoff) {
                    break;
                }
            }
            end_[batch][i] = j;
            for (int k = i; k < KM_CEIL_DIV(j, block_size); k++) {
                start_[batch][k] = std::min(start_[batch][k], i * block_size);
            }
        }
    }

    return {start.to(device), end.to(device)};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("make_ranges", &make_ranges);
    m.def("make_ranges_symmetric", &make_ranges_symmetric);
}
