#include <torch/extension.h>

#include <cuda.h>

template <typename scalar_t, size_t batch_dim, size_t nonbatch_dim> class BatchedAccessor {
  public:
    BatchedAccessor(torch::Tensor tensor) {
        static_assert(nonbatch_dim > 0, "nonbatch_dim must be at least 1");
        this->data = tensor.mutable_data_ptr<scalar_t>();
        auto strides = tensor.strides().data();
        auto sizes = tensor.sizes().data();
        for (int32_t i = 0; i < batch_dim; i++) {
            this->batch_strides[i] = static_cast<int32_t>(strides[i]);
        }
        for (int32_t i = 0; i < nonbatch_dim; i++) {
            this->nonbatch_sizes[i] = static_cast<int32_t>(sizes[batch_dim + i]);
            this->nonbatch_strides[i] = static_cast<int32_t>(strides[batch_dim + i]);
        }
    }

    __device__ torch::TensorAccessor<scalar_t, nonbatch_dim, torch::RestrictPtrTraits, int32_t>
    operator[](std::array<int32_t, batch_dim> batch) const {
        auto offset = 0;
        for (int32_t i = 0; i < batch_dim; i++) {
            offset += batch[i] * this->batch_strides[i];
        }
        scalar_t *data = this->data + offset;
        return torch::TensorAccessor<scalar_t, nonbatch_dim, torch::RestrictPtrTraits, int32_t>(
            data, this->nonbatch_sizes, this->nonbatch_strides);
    }

  private:
    scalar_t *data;
    int32_t batch_strides[batch_dim];
    int32_t nonbatch_sizes[nonbatch_dim];
    int32_t nonbatch_strides[nonbatch_dim];
};

template <size_t batch_dim> class BatchLayout {
  public:
    BatchLayout(const int64_t *batch_sizes) {
        for (int32_t i = batch_dim - 1; i >= 0; i--) {
            this->batch_sizes[i] = static_cast<int32_t>(batch_sizes[i]);
            if (i == batch_dim - 1) {
                this->batch_strides[i] = 1;
            } else {
                this->batch_strides[i] = this->batch_strides[i + 1] * this->batch_sizes[i + 1];
            }
        }
    }

    __device__ std::array<int32_t, batch_dim> get_batch(int32_t index) const {
        std::array<int32_t, batch_dim> batch;
        for (int32_t i = 0; i < batch_dim; i++) {
            batch[i] = index / this->batch_strides[i];
            index -= batch[i] * this->batch_strides[i];
        }
        return batch;
    }

    uint32_t num_batches() const {
        return static_cast<uint32_t>(this->batch_strides[0] * this->batch_sizes[0]);
    }

    template <size_t nonbatch_dim>
    std::array<int64_t, batch_dim + nonbatch_dim>
    make_shape(std::initializer_list<int64_t> nonbatch_sizes) const {
        std::array<int64_t, batch_dim + nonbatch_dim> shape;
        int32_t i = 0;
        for (; i < batch_dim; i++) {
            shape[i] = this->batch_sizes[i];
        }
        for (auto size : nonbatch_sizes) {
            shape[i] = size;
            i++;
        }
        return shape;
    }

  private:
    int32_t batch_sizes[batch_dim];
    int32_t batch_strides[batch_dim];
};
