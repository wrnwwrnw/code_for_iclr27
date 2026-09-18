#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <vector>

template <typename scalar_t>
__global__ void merge_kernel(
    const scalar_t* keys, const scalar_t* values, const float* positions,
    const scalar_t* recalled, const float* recalled_positions, const int64_t* mapping,
    scalar_t* output_keys, scalar_t* output_values, float* output_positions,
    int64_t rows, int64_t dimension) {
    const int64_t offset = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (offset >= rows * dimension) return;
    const int64_t row = offset / dimension;
    const int64_t column = offset % dimension;
    const int64_t source = mapping[row];
    if (source >= 0) {
        output_keys[offset] = keys[source * dimension + column];
        output_values[offset] = values[source * dimension + column];
        if (column == 0) output_positions[row] = positions[source];
    } else {
        const int64_t restored = -source - 1;
        output_keys[offset] = recalled[(restored * 2) * dimension + column];
        output_values[offset] = recalled[(restored * 2 + 1) * dimension + column];
        if (column == 0) output_positions[row] = recalled_positions[restored];
    }
}

template <typename scalar_t>
__global__ void pack_kernel(
    const scalar_t* keys, const scalar_t* values, const float* positions,
    const int64_t* indices, const int64_t* heads, scalar_t* payload,
    int64_t* metadata, int64_t rows, int64_t dimension, int64_t layer) {
    const int64_t offset = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (offset >= rows * dimension) return;
    const int64_t row = offset / dimension;
    const int64_t column = offset % dimension;
    const int64_t source = indices[row];
    payload[(row * 2) * dimension + column] = keys[source * dimension + column];
    payload[(row * 2 + 1) * dimension + column] = values[source * dimension + column];
    if (column == 0) {
        metadata[row * 3] = layer;
        metadata[row * 3 + 1] = heads[row];
        metadata[row * 3 + 2] = static_cast<int64_t>(positions[source]);
    }
}

void check_recall_tensor(const torch::Tensor& tensor, const torch::Tensor& reference) {
    TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous(), "Expected contiguous CUDA tensor");
    TORCH_CHECK(tensor.device() == reference.device(), "Recall devices differ");
}

std::vector<torch::Tensor> merge_cuda(
    torch::Tensor keys, torch::Tensor values, torch::Tensor positions,
    torch::Tensor recalled, torch::Tensor recalled_positions, torch::Tensor mapping) {
    for (const auto& tensor : {keys, values, positions, recalled, recalled_positions, mapping})
        check_recall_tensor(tensor, keys);
    TORCH_CHECK(keys.dim() == 2 && keys.size(1) > 0 && keys.sizes() == values.sizes(), "Invalid K/V shape");
    TORCH_CHECK(recalled.dim() == 3 && recalled.size(1) == 2 &&
                recalled.size(2) == keys.size(1), "Invalid recalled shape");
    TORCH_CHECK(keys.scalar_type() == values.scalar_type() &&
                keys.scalar_type() == recalled.scalar_type(), "K/V dtypes differ");
    TORCH_CHECK(positions.scalar_type() == at::kFloat && recalled_positions.scalar_type() == at::kFloat &&
                positions.numel() == keys.size(0) && recalled_positions.numel() == recalled.size(0), "Invalid positions");
    TORCH_CHECK(mapping.dim() == 1 && mapping.scalar_type() == at::kLong, "Mapping must be int64");
    c10::cuda::CUDAGuard guard(keys.device());
    const auto rows = mapping.numel();
    const auto dimension = keys.size(1);
    auto output_keys = torch::empty({rows, dimension}, keys.options());
    auto output_values = torch::empty_like(output_keys);
    auto output_positions = torch::empty({rows, 1}, positions.options());
    if (rows) {
        AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, keys.scalar_type(), "recall_merge", [&] {
            merge_kernel<scalar_t><<<(rows * dimension + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
                keys.data_ptr<scalar_t>(), values.data_ptr<scalar_t>(), positions.data_ptr<float>(),
                recalled.data_ptr<scalar_t>(), recalled_positions.data_ptr<float>(), mapping.data_ptr<int64_t>(),
                output_keys.data_ptr<scalar_t>(), output_values.data_ptr<scalar_t>(), output_positions.data_ptr<float>(),
                rows, dimension);
        });
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {output_keys, output_values, output_positions};
}

void pack_cuda(torch::Tensor keys, torch::Tensor values, torch::Tensor positions,
               torch::Tensor indices, torch::Tensor heads, int64_t layer,
               torch::Tensor payload, torch::Tensor metadata) {
    for (const auto& tensor : {keys, values, positions, indices, heads, payload, metadata})
        check_recall_tensor(tensor, keys);
    TORCH_CHECK(keys.dim() == 2 && keys.size(1) > 0 && keys.sizes() == values.sizes(), "Invalid K/V shape");
    TORCH_CHECK(keys.scalar_type() == values.scalar_type() && keys.scalar_type() == payload.scalar_type(), "Dtype mismatch");
    TORCH_CHECK(indices.dim() == 1 && indices.scalar_type() == at::kLong &&
                heads.sizes() == indices.sizes() && heads.scalar_type() == at::kLong, "Invalid gather indices");
    const auto rows = indices.numel();
    const auto dimension = keys.size(1);
    TORCH_CHECK(payload.dim() == 3 && payload.size(0) == rows && payload.size(1) == 2 &&
                payload.size(2) == dimension, "Invalid payload buffer");
    TORCH_CHECK(positions.numel() == keys.size(0) && positions.scalar_type() == at::kFloat, "Invalid positions");
    TORCH_CHECK(metadata.dim() == 2 && metadata.size(0) == rows && metadata.size(1) == 3 &&
                metadata.scalar_type() == at::kLong, "Invalid identity buffer");
    c10::cuda::CUDAGuard guard(keys.device());
    if (!rows) return;
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, keys.scalar_type(), "archive_pack", [&] {
        pack_kernel<scalar_t><<<(rows * dimension + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            keys.data_ptr<scalar_t>(), values.data_ptr<scalar_t>(), positions.data_ptr<float>(),
            indices.data_ptr<int64_t>(), heads.data_ptr<int64_t>(), payload.data_ptr<scalar_t>(),
            metadata.data_ptr<int64_t>(), rows, dimension, layer);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

