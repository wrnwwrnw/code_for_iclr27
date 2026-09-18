#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <algorithm>
#include <vector>

template <typename scalar_t>
__global__ void batch_append_kernel(
    const int64_t* descriptors, const scalar_t* new_keys, const scalar_t* new_values,
    scalar_t* output_keys, scalar_t* output_values, float* output_positions,
    int64_t dimension, int64_t tiles_per_segment) {
    const int64_t segment = blockIdx.x / tiles_per_segment;
    const int64_t item = (blockIdx.x % tiles_per_segment) * blockDim.x + threadIdx.x;
    const auto descriptor = descriptors + segment * 7;
    const int64_t old_length = descriptor[4];
    if (item >= (old_length + 1) * dimension) return;
    const int64_t row = item / dimension;
    const int64_t column = item % dimension;
    const int64_t destination = (descriptor[5] + row) * dimension + column;
    if (row < old_length) {
        const auto keys = reinterpret_cast<const scalar_t*>(descriptor[0]);
        const auto values = reinterpret_cast<const scalar_t*>(descriptor[1]);
        const auto positions = reinterpret_cast<const float*>(descriptor[2]);
        const int64_t source = (descriptor[3] + row) * dimension + column;
        output_keys[destination] = keys[source];
        output_values[destination] = values[source];
        if (column == 0) output_positions[descriptor[5] + row] = positions[descriptor[3] + row];
    } else {
        output_keys[destination] = new_keys[segment * dimension + column];
        output_values[destination] = new_values[segment * dimension + column];
        if (column == 0) output_positions[descriptor[5] + row] = static_cast<float>(descriptor[6]);
    }
}

std::vector<torch::Tensor> batch_append_cuda(
    std::vector<torch::Tensor> keys, std::vector<torch::Tensor> values,
    std::vector<torch::Tensor> positions, torch::Tensor new_keys, torch::Tensor new_values,
    std::vector<std::vector<int64_t>> lengths, std::vector<int64_t> logical_positions) {
    TORCH_CHECK(new_keys.is_cuda() && new_keys.is_contiguous() && new_values.is_contiguous(), "Contiguous CUDA states required");
    TORCH_CHECK(new_keys.dim() == 4 && new_keys.size(2) == 1 && new_keys.size(3) > 0, "Only batched one-token append is supported");
    TORCH_CHECK(new_keys.sizes() == new_values.sizes() && new_keys.scalar_type() == new_values.scalar_type() &&
                new_keys.device() == new_values.device(), "New K/V mismatch");
    const int64_t requests = new_keys.size(0), heads = new_keys.size(1), dimension = new_keys.size(3);
    TORCH_CHECK(requests > 0 && heads > 0 && keys.size() == requests && values.size() == requests &&
                positions.size() == requests && lengths.size() == requests && logical_positions.size() == requests,
                "Request metadata mismatch");
    c10::cuda::CUDAGuard guard(new_keys.device());
    auto cpu_descriptors = torch::empty({requests * heads, 7}, torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));
    auto descriptors = cpu_descriptors.data_ptr<int64_t>();
    int64_t destination = 0, maximum = 0;
    for (int64_t request = 0; request < requests; ++request) {
        TORCH_CHECK(keys[request].is_cuda() && keys[request].is_contiguous() && values[request].is_contiguous() &&
                    positions[request].is_contiguous(), "Contiguous CUDA cache required");
        TORCH_CHECK(keys[request].device() == new_keys.device() && values[request].device() == new_keys.device() &&
                    positions[request].device() == new_keys.device(), "Cache devices differ");
        TORCH_CHECK(keys[request].dim() == 2 && keys[request].size(1) == dimension &&
                    keys[request].sizes() == values[request].sizes() && keys[request].scalar_type() == new_keys.scalar_type() &&
                    values[request].scalar_type() == new_keys.scalar_type(), "Cache K/V mismatch");
        TORCH_CHECK(positions[request].scalar_type() == at::kFloat && positions[request].numel() == keys[request].size(0) &&
                    lengths[request].size() == heads && logical_positions[request] >= 0 && logical_positions[request] < (1 << 24),
                    "Invalid positions or head metadata");
        int64_t source = 0;
        for (int64_t head = 0; head < heads; ++head) {
            const int64_t length = lengths[request][head];
            TORCH_CHECK(length >= 0 && source + length <= keys[request].size(0), "Invalid head length");
            const int64_t offset = (request * heads + head) * 7;
            descriptors[offset] = reinterpret_cast<int64_t>(keys[request].data_ptr());
            descriptors[offset + 1] = reinterpret_cast<int64_t>(values[request].data_ptr());
            descriptors[offset + 2] = reinterpret_cast<int64_t>(positions[request].data_ptr());
            descriptors[offset + 3] = source;
            descriptors[offset + 4] = length;
            descriptors[offset + 5] = destination;
            descriptors[offset + 6] = logical_positions[request];
            source += length;
            destination += length + 1;
            maximum = std::max(maximum, length + 1);
        }
        TORCH_CHECK(source == keys[request].size(0), "Head lengths do not cover cache rows");
    }
    auto device_descriptors = cpu_descriptors.to(new_keys.device());
    auto output_keys = torch::empty({destination, dimension}, new_keys.options());
    auto output_values = torch::empty_like(output_keys);
    auto output_positions = torch::empty({destination, 1}, new_keys.options().dtype(torch::kFloat));
    const int64_t tiles = (maximum * dimension + 255) / 256;
    TORCH_CHECK(tiles * requests * heads <= 2147483647LL, "Append grid exceeds CUDA limit");
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, new_keys.scalar_type(), "batch_append", [&] {
        batch_append_kernel<scalar_t><<<tiles * requests * heads, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            device_descriptors.data_ptr<int64_t>(), new_keys.data_ptr<scalar_t>(), new_values.data_ptr<scalar_t>(),
            output_keys.data_ptr<scalar_t>(), output_values.data_ptr<scalar_t>(), output_positions.data_ptr<float>(), dimension, tiles);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {output_keys, output_values, output_positions};
}
