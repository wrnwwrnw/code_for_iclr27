#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>

void check_cuda(const torch::Tensor& tensor, const torch::Tensor& reference) {
    TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous(), "Expected contiguous CUDA tensor");
    TORCH_CHECK(tensor.device() == reference.device(), "CUDA devices differ");
}

template <int Threads, bool Maximum>
__device__ float block_reduce(float value, float* scratch) {
    scratch[threadIdx.x] = value;
    __syncthreads();
    for (int stride = Threads / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] = Maximum
                ? fmaxf(scratch[threadIdx.x], scratch[threadIdx.x + stride])
                : scratch[threadIdx.x] + scratch[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float result = scratch[0];
    __syncthreads();
    return result;
}

template <typename scalar_t>
__global__ void probabilities_kernel(
    const scalar_t* logits, const float* projected_values,
    const int64_t* query_positions, const float* key_positions,
    float* probabilities, float* output_projection,
    int groups, int length, float scale) {
    __shared__ float scratch[256];
    const int row = blockIdx.x;
    const int group = row % groups;
    const int query = row / groups;
    float maximum = -INFINITY;
    for (int position = threadIdx.x; position < length; position += blockDim.x) {
        const float rounded = static_cast<float>(static_cast<scalar_t>(
            static_cast<float>(logits[row * length + position]) * scale));
        const float value = key_positions[position] <= query_positions[query] ? rounded : -INFINITY;
        probabilities[row * length + position] = value;
        maximum = fmaxf(maximum, value);
    }
    maximum = block_reduce<256, true>(maximum, scratch);
    float denominator = 0.0f;
    for (int position = threadIdx.x; position < length; position += blockDim.x) {
        const float weight = expf(probabilities[row * length + position] - maximum);
        probabilities[row * length + position] = weight;
        denominator += weight;
    }
    denominator = block_reduce<256, false>(denominator, scratch);
    float projection = 0.0f;
    for (int position = threadIdx.x; position < length; position += blockDim.x) {
        const float probability = probabilities[row * length + position] / denominator;
        probabilities[row * length + position] = probability;
        projection += probability * projected_values[group * length + position];
    }
    projection = block_reduce<256, false>(projection, scratch);
    if (threadIdx.x == 0) output_projection[row] = projection;
}

__device__ double softplus64(double value) {
    return value > 20.0 ? value : log1p(exp(value));
}

__device__ double risk64(double margin, double probability, double base_softplus, float drop) {
    const double compared = margin - static_cast<double>(drop);
    return fmax(0.0, probability * (margin - compared) - base_softplus + softplus64(compared));
}

__global__ void risk_kernel(
    const float* probabilities, const float* output_projection,
    const float* projected_values, const float* margin, float* result,
    int window, int groups, int length, int first, int count, float epsilon) {
    __shared__ float minimums[256];
    __shared__ float maximums[256];
    __shared__ double base[3];
    if (threadIdx.x == 0) {
        base[0] = static_cast<double>(margin[0]);
        base[1] = 1.0 / (1.0 + exp(-base[0]));
        base[2] = softplus64(base[0]);
    }
    const int candidate_lane = threadIdx.x % 32;
    const int query_lane = threadIdx.x / 32;
    const int candidate = blockIdx.x * 32 + candidate_lane;
    const int position = first + candidate;
    float minimum = INFINITY;
    float maximum = -INFINITY;
    for (int query = query_lane; candidate < count && query < window; query += 8) {
        float drop = 0.0f;
        for (int group = 0; group < groups; ++group) {
            const int row = query * groups + group;
            const float probability = probabilities[row * length + position];
            const float ratio = probability / fmaxf(1.0f - probability, epsilon);
            drop += ratio * (projected_values[group * length + position] - output_projection[row]);
        }
        if (!isfinite(drop)) {
            minimum = NAN;
            maximum = NAN;
            break;
        }
        minimum = fminf(minimum, drop);
        maximum = fmaxf(maximum, drop);
    }
    minimums[threadIdx.x] = minimum;
    maximums[threadIdx.x] = maximum;
    __syncthreads();
    if (query_lane == 0 && candidate < count) {
        for (int lane = 1; lane < 8; ++lane) {
            const float other_minimum = minimums[lane * 32 + candidate_lane];
            const float other_maximum = maximums[lane * 32 + candidate_lane];
            if (isnan(minimum) || isnan(other_minimum)) {
                result[candidate] = NAN;
                return;
            }
            minimum = fminf(minimum, other_minimum);
            maximum = fmaxf(maximum, other_maximum);
        }
        result[candidate] = static_cast<float>(fmax(
            risk64(base[0], base[1], base[2], minimum), risk64(base[0], base[1], base[2], maximum)));
    }
}

template <typename scalar_t>
__global__ void append_kernel(
    const scalar_t* keys, const scalar_t* values, const float* positions,
    const scalar_t* new_keys, const scalar_t* new_values, const int* cumulative,
    scalar_t* output_keys, scalar_t* output_values, float* output_positions,
    int* output_cumulative, int* output_lengths,
    int heads, int dimension, int added, int64_t logical_start) {
    const int head = blockIdx.y;
    const int old_start = cumulative[head];
    const int old_length = cumulative[head + 1] - old_start;
    const int new_start = old_start + head * added;
    const int element = blockIdx.x * blockDim.x + threadIdx.x;
    const int position = element / dimension;
    const int channel = element % dimension;
    if (element == 0) {
        output_cumulative[head] = new_start;
        output_lengths[head] = old_length + added;
        if (head == heads - 1) output_cumulative[heads] = cumulative[heads] + heads * added;
    }
    if (position >= old_length + added) return;
    const int destination = (new_start + position) * dimension + channel;
    if (position < old_length) {
        const int source = (old_start + position) * dimension + channel;
        output_keys[destination] = keys[source];
        output_values[destination] = values[source];
        if (channel == 0) output_positions[new_start + position] = positions[old_start + position];
    } else {
        const int source = (head * added + position - old_length) * dimension + channel;
        output_keys[destination] = new_keys[source];
        output_values[destination] = new_values[source];
        if (channel == 0) output_positions[new_start + position] = static_cast<float>(logical_start + position - old_length);
    }
}

template <typename scalar_t>
__global__ void compact_kernel(
    const scalar_t* keys, const scalar_t* values, const float* positions,
    const int64_t* removed, scalar_t* output_keys, scalar_t* output_values,
    float* output_positions, int rows, int dimension, int removed_count) {
    const int element = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = element / dimension;
    if (row >= rows) return;
    int lower = 0;
    int upper = removed_count;
    const bool warp_aligned = dimension % 32 == 0;
    if (!warp_aligned || threadIdx.x % 32 == 0) {
        while (lower < upper) {
            const int middle = lower + (upper - lower) / 2;
            if (removed[middle] - middle <= row) lower = middle + 1;
            else upper = middle;
        }
    }
    if (warp_aligned) lower = __shfl_sync(0xffffffff, lower, 0);
    const int source_row = row + lower;
    const int channel = element % dimension;
    output_keys[element] = keys[source_row * dimension + channel];
    output_values[element] = values[source_row * dimension + channel];
    if (channel == 0) output_positions[row] = positions[source_row];
}

__global__ void pool_kernel(const float* scores, const int64_t* bounds,
                            float* output, int width) {
    const int segment = blockIdx.y;
    const int begin = bounds[segment];
    const int end = bounds[segment + 1];
    const int position = begin + blockIdx.x * blockDim.x + threadIdx.x;
    if (position >= end) return;
    float value = -INFINITY;
    for (int offset = -width / 2; offset <= width / 2; ++offset) {
        const int neighbor = position + offset;
        if (neighbor >= begin && neighbor < end) value = fmaxf(value, scores[neighbor]);
    }
    output[position] = value;
}

std::vector<torch::Tensor> temporal_probabilities_cuda(
    torch::Tensor logits, torch::Tensor projected_values,
    torch::Tensor query_positions, torch::Tensor key_positions, double scale) {
    check_cuda(logits, logits);
    check_cuda(projected_values, logits);
    check_cuda(query_positions, logits);
    check_cuda(key_positions, logits);
    TORCH_CHECK(logits.dim() == 3, "logits must be [window, groups, positions]");
    TORCH_CHECK(logits.scalar_type() == at::kFloat || logits.scalar_type() == at::kHalf
        || logits.scalar_type() == at::kBFloat16, "Only FP32/FP16/BF16 scores are supported");
    const int window = logits.size(0), groups = logits.size(1), length = logits.size(2);
    TORCH_CHECK(window > 0 && groups > 0 && length > 0, "Empty attention dimensions");
    TORCH_CHECK(projected_values.scalar_type() == at::kFloat && projected_values.dim() == 2
        && projected_values.size(0) == groups && projected_values.size(1) == length, "Invalid projected values");
    TORCH_CHECK(query_positions.scalar_type() == at::kLong && query_positions.numel() == window, "Invalid query positions");
    TORCH_CHECK(key_positions.scalar_type() == at::kFloat && key_positions.numel() == length, "Invalid key positions");
    c10::cuda::CUDAGuard guard(logits.device());
    auto probabilities = torch::empty(logits.sizes(), logits.options().dtype(at::kFloat));
    auto output = torch::empty({window, groups}, probabilities.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, logits.scalar_type(), "lazy_recall_probabilities", [&] {
        probabilities_kernel<scalar_t><<<window * groups, 256, 0, stream>>>(
            logits.data_ptr<scalar_t>(), projected_values.data_ptr<float>(),
            query_positions.data_ptr<int64_t>(), key_positions.data_ptr<float>(),
            probabilities.data_ptr<float>(), output.data_ptr<float>(), groups, length, static_cast<float>(scale));
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {probabilities, output};
}

torch::Tensor temporal_risk_cuda(
    torch::Tensor probabilities, torch::Tensor output_projection,
    torch::Tensor projected_values, torch::Tensor margin,
    int64_t first, int64_t last, double epsilon) {
    for (const auto& tensor : {probabilities, output_projection, projected_values, margin}) {
        check_cuda(tensor, probabilities);
        TORCH_CHECK(tensor.scalar_type() == at::kFloat, "Risk inputs must be float32");
    }
    TORCH_CHECK(probabilities.dim() == 3, "Invalid probability dimensions");
    const int window = probabilities.size(0), groups = probabilities.size(1), length = probabilities.size(2);
    TORCH_CHECK(window > 0 && groups > 0 && first >= 0 && last > first && last <= length && epsilon > 0, "Invalid risk interval");
    TORCH_CHECK(margin.numel() == 1 && output_projection.numel() == window * groups
        && projected_values.numel() == groups * length, "Risk shapes differ");
    c10::cuda::CUDAGuard guard(probabilities.device());
    auto result = torch::empty({last - first}, probabilities.options());
    risk_kernel<<<(last - first + 31) / 32, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        probabilities.data_ptr<float>(), output_projection.data_ptr<float>(), projected_values.data_ptr<float>(),
        margin.data_ptr<float>(), result.data_ptr<float>(), window, groups, length, first, last - first, epsilon);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return result;
}

void check_cache(torch::Tensor keys, torch::Tensor values, torch::Tensor positions) {
    for (const auto& tensor : {keys, values, positions}) check_cuda(tensor, keys);
    TORCH_CHECK(keys.dim() == 2 && keys.size(1) > 0 && keys.sizes() == values.sizes()
        && keys.scalar_type() == values.scalar_type(), "Invalid KV layout");
    TORCH_CHECK(positions.scalar_type() == at::kFloat && positions.numel() == keys.size(0), "Invalid positions");
}

std::vector<torch::Tensor> append_cuda(
    torch::Tensor keys, torch::Tensor values, torch::Tensor positions,
    torch::Tensor new_keys, torch::Tensor new_values, torch::Tensor cumulative,
    int64_t logical_start, int64_t maximum_length) {
    check_cache(keys, values, positions);
    for (const auto& tensor : {new_keys, new_values, cumulative}) check_cuda(tensor, keys);
    TORCH_CHECK(new_keys.dim() == 4 && new_keys.size(0) == 1 && new_keys.sizes() == new_values.sizes()
        && new_keys.scalar_type() == keys.scalar_type() && new_values.scalar_type() == keys.scalar_type(), "Invalid appended KV");
    const int heads = new_keys.size(1), added = new_keys.size(2), dimension = keys.size(1);
    TORCH_CHECK(heads > 0 && added > 0 && new_keys.size(3) == dimension && maximum_length >= 0, "Invalid appended shape");
    TORCH_CHECK(cumulative.scalar_type() == at::kInt && cumulative.numel() == heads + 1, "Invalid cumulative lengths");
    c10::cuda::CUDAGuard guard(keys.device());
    const int rows = keys.size(0) + heads * added;
    auto output_keys = torch::empty({rows, dimension}, keys.options());
    auto output_values = torch::empty_like(output_keys);
    auto output_positions = torch::empty({rows, 1}, positions.options());
    auto output_cumulative = torch::empty_like(cumulative);
    auto output_lengths = torch::empty({heads}, cumulative.options());
    dim3 grid(((maximum_length + added) * dimension + 255) / 256, heads);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, keys.scalar_type(), "lazy_recall_append", [&] {
        append_kernel<scalar_t><<<grid, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            keys.data_ptr<scalar_t>(), values.data_ptr<scalar_t>(), positions.data_ptr<float>(),
            new_keys.data_ptr<scalar_t>(), new_values.data_ptr<scalar_t>(), cumulative.data_ptr<int>(),
            output_keys.data_ptr<scalar_t>(), output_values.data_ptr<scalar_t>(), output_positions.data_ptr<float>(),
            output_cumulative.data_ptr<int>(), output_lengths.data_ptr<int>(), heads, dimension, added, logical_start);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return {output_keys, output_values, output_positions, output_cumulative, output_lengths};
}

std::vector<torch::Tensor> compact_cuda(
    torch::Tensor keys, torch::Tensor values, torch::Tensor positions, torch::Tensor removed) {
    check_cache(keys, values, positions);
    check_cuda(removed, keys);
    TORCH_CHECK(removed.scalar_type() == at::kLong && removed.dim() == 1 && removed.numel() <= keys.size(0), "Invalid removals");
    c10::cuda::CUDAGuard guard(keys.device());
    const int rows = keys.size(0) - removed.numel(), dimension = keys.size(1);
    auto output_keys = torch::empty({rows, dimension}, keys.options());
    auto output_values = torch::empty_like(output_keys);
    auto output_positions = torch::empty({rows, 1}, positions.options());
    if (rows > 0) {
        AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, keys.scalar_type(), "lazy_recall_compact", [&] {
            compact_kernel<scalar_t><<<(rows * dimension + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
                keys.data_ptr<scalar_t>(), values.data_ptr<scalar_t>(), positions.data_ptr<float>(), removed.data_ptr<int64_t>(),
                output_keys.data_ptr<scalar_t>(), output_values.data_ptr<scalar_t>(), output_positions.data_ptr<float>(),
                rows, dimension, removed.numel());
        });
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return {output_keys, output_values, output_positions};
}

torch::Tensor pool_cuda(torch::Tensor scores, torch::Tensor bounds, int64_t maximum_count, int64_t width) {
    check_cuda(scores, scores);
    check_cuda(bounds, scores);
    TORCH_CHECK(scores.scalar_type() == at::kFloat && scores.dim() == 1
        && bounds.scalar_type() == at::kLong && bounds.dim() == 1 && bounds.numel() >= 2, "Invalid pool layout");
    TORCH_CHECK(width > 0 && width % 2 == 1 && maximum_count > 0, "Invalid pool configuration");
    c10::cuda::CUDAGuard guard(scores.device());
    auto output = torch::empty_like(scores);
    dim3 grid((maximum_count + 127) / 128, bounds.numel() - 1);
    pool_kernel<<<grid, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
        scores.data_ptr<float>(), bounds.data_ptr<int64_t>(), output.data_ptr<float>(), width);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
