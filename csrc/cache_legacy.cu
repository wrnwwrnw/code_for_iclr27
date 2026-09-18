#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <algorithm>
#include <vector>

namespace {

template <typename scalar_t>
__global__ void append_flattened_cache_kernel(
    scalar_t* output,
    const scalar_t* cache,
    const scalar_t* states,
    const int* head_lens,
    const int* cu_head_lens,
    int head_dim,
    int new_length) {
  const int head = blockIdx.y;
  const int old_length = head_lens[head];
  const int output_start = cu_head_lens[head] + head * new_length;
  const int elements = (old_length + new_length) * head_dim;
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < elements;
       index += blockDim.x * gridDim.x) {
    const int row = index / head_dim;
    const int column = index % head_dim;
    if (row < old_length) {
      output[(output_start + row) * head_dim + column] =
          cache[(cu_head_lens[head] + row) * head_dim + column];
    } else {
      const int state_row = head * new_length + row - old_length;
      output[(output_start + row) * head_dim + column] =
          states[state_row * head_dim + column];
    }
  }
}

template <typename scalar_t>
__global__ void compact_flattened_kv_kernel(
    scalar_t* output_keys,
    scalar_t* output_values,
    const scalar_t* keys,
    const scalar_t* values,
    const int64_t* keep_indices,
    int64_t rows,
    int head_dim) {
  const int64_t total = rows * head_dim;
  for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
       index < total;
       index += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t output_row = index / head_dim;
    const int column = index % head_dim;
    const int64_t input_row = keep_indices[output_row];
    output_keys[index] = keys[input_row * head_dim + column];
    output_values[index] = values[input_row * head_dim + column];
  }
}

__device__ double stable_sigmoid(double value) {
  if (value >= 0.0) {
    return 1.0 / (1.0 + exp(-value));
  }
  const double exponential = exp(value);
  return exponential / (1.0 + exponential);
}

__device__ double stable_softplus(double value) {
  if (value > 20.0) {
    return value + log1p(exp(-value));
  }
  if (value < -20.0) {
    return log1p(exp(value));
  }
  return log1p(exp(value));
}

template <typename scalar_t>
__global__ void decision_risk_scores_kernel(
    float* scores,
    const float* probabilities,
    const scalar_t* values,
    const float* head_outputs,
    const float* gradients,
    int groups,
    int candidates,
    int head_dim,
    double base_margin,
    float denominator_epsilon) {
  const int candidate = blockIdx.x;
  if (candidate >= candidates) {
    return;
  }

  float partial = 0.0f;
  const int terms = groups * head_dim;
  for (int index = threadIdx.x; index < terms; index += blockDim.x) {
    const int group = index / head_dim;
    const int dimension = index % head_dim;
    const float alpha = probabilities[group * candidates + candidate];
    const float denominator = fmaxf(1.0f - alpha, denominator_epsilon);
    const float value = static_cast<float>(
        values[candidate * head_dim + dimension]);
    const float local_delta =
        alpha / denominator *
        (value - head_outputs[group * head_dim + dimension]);
    partial += gradients[group * head_dim + dimension] * local_delta;
  }

  __shared__ float reduction[256];
  reduction[threadIdx.x] = partial;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      reduction[threadIdx.x] += reduction[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    const double compared_margin = base_margin - reduction[0];
    const double probability = stable_sigmoid(base_margin);
    double kl =
        probability * (base_margin - compared_margin)
        - stable_softplus(base_margin)
        + stable_softplus(compared_margin);
    if (kl < 0.0) {
      kl = 0.0;
    }
    scores[candidate] = static_cast<float>(kl);
  }
}

void check_flat_matrix(const torch::Tensor& tensor, const char* name) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(tensor.dim() == 2, name, " must be rank two");
}

}  // namespace

torch::Tensor append_flattened_cache(
    const torch::Tensor& cache,
    const torch::Tensor& states,
    const torch::Tensor& head_lens,
    const torch::Tensor& cu_head_lens) {
  check_flat_matrix(cache, "cache");
  TORCH_CHECK(states.is_cuda() && states.is_contiguous(),
              "states must be contiguous CUDA");
  TORCH_CHECK(states.dim() == 4, "states must have shape [B,H,N,D]");
  TORCH_CHECK(cache.scalar_type() == states.scalar_type(),
              "cache and states must share dtype");
  TORCH_CHECK(head_lens.scalar_type() == torch::kInt32,
              "head_lens must be int32");
  TORCH_CHECK(cu_head_lens.scalar_type() == torch::kInt32,
              "cu_head_lens must be int32");
  TORCH_CHECK(head_lens.is_cuda() && cu_head_lens.is_cuda(),
              "length metadata must be CUDA");
  TORCH_CHECK(head_lens.is_contiguous() && cu_head_lens.is_contiguous(),
              "length metadata must be contiguous");
  TORCH_CHECK(cache.device() == states.device() &&
                  cache.device() == head_lens.device() &&
                  cache.device() == cu_head_lens.device(),
              "append inputs must share one CUDA device");
  TORCH_CHECK(cache.size(1) == states.size(3), "head_dim mismatch");

  c10::cuda::CUDAGuard guard(cache.device());
  const int total_heads = states.size(0) * states.size(1);
  const int new_length = states.size(2);
  const int head_dim = cache.size(1);
  TORCH_CHECK(head_lens.numel() == total_heads, "head count mismatch");
  TORCH_CHECK(cu_head_lens.numel() == total_heads + 1,
              "cumulative length count mismatch");

  auto output = torch::empty(
      {cache.size(0) + total_heads * new_length, head_dim}, cache.options());
  constexpr int threads = 256;
  dim3 grid(32, total_heads);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, cache.scalar_type(),
      "append_flattened_cache", [&] {
        append_flattened_cache_kernel<scalar_t><<<grid, threads, 0, stream>>>(
            output.data_ptr<scalar_t>(),
            cache.data_ptr<scalar_t>(),
            states.data_ptr<scalar_t>(),
            head_lens.data_ptr<int>(),
            cu_head_lens.data_ptr<int>(),
            head_dim,
            new_length);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

std::vector<torch::Tensor> compact_flattened_kv(
    const torch::Tensor& keys,
    const torch::Tensor& values,
    const torch::Tensor& keep_indices) {
  check_flat_matrix(keys, "keys");
  check_flat_matrix(values, "values");
  TORCH_CHECK(keys.sizes() == values.sizes(), "K/V shape mismatch");
  TORCH_CHECK(keys.scalar_type() == values.scalar_type(), "K/V dtype mismatch");
  TORCH_CHECK(keep_indices.is_cuda() && keep_indices.is_contiguous(),
              "keep_indices must be contiguous CUDA");
  TORCH_CHECK(keep_indices.scalar_type() == torch::kInt64,
              "keep_indices must be int64");
  TORCH_CHECK(keep_indices.dim() == 1, "keep_indices must be rank one");
  TORCH_CHECK(keys.device() == values.device() &&
                  keys.device() == keep_indices.device(),
              "compaction inputs must share one CUDA device");

  c10::cuda::CUDAGuard guard(keys.device());
  const int64_t rows = keep_indices.numel();
  const int head_dim = keys.size(1);
  auto output_keys = torch::empty({rows, head_dim}, keys.options());
  auto output_values = torch::empty({rows, head_dim}, values.options());
  if (rows == 0) {
    return {output_keys, output_values};
  }
  constexpr int threads = 256;
  const int64_t blocks_needed = (rows * head_dim + threads - 1) / threads;
  const int blocks = static_cast<int>(std::min<int64_t>(blocks_needed, 65535));
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, keys.scalar_type(),
      "compact_flattened_kv", [&] {
        compact_flattened_kv_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
            output_keys.data_ptr<scalar_t>(),
            output_values.data_ptr<scalar_t>(),
            keys.data_ptr<scalar_t>(),
            values.data_ptr<scalar_t>(),
            keep_indices.data_ptr<int64_t>(),
            rows,
            head_dim);
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {output_keys, output_values};
}

torch::Tensor decision_risk_scores(
    const torch::Tensor& probabilities,
    const torch::Tensor& values,
    const torch::Tensor& head_outputs,
    const torch::Tensor& gradients,
    double base_margin,
    double denominator_epsilon) {
  check_flat_matrix(probabilities, "probabilities");
  check_flat_matrix(values, "values");
  check_flat_matrix(head_outputs, "head_outputs");
  check_flat_matrix(gradients, "gradients");
  TORCH_CHECK(probabilities.scalar_type() == torch::kFloat32,
              "probabilities must be float32");
  TORCH_CHECK(head_outputs.scalar_type() == torch::kFloat32,
              "head_outputs must be float32");
  TORCH_CHECK(gradients.scalar_type() == torch::kFloat32,
              "gradients must be float32");
  TORCH_CHECK(probabilities.size(0) == head_outputs.size(0),
              "mapped-query-head mismatch");
  TORCH_CHECK(head_outputs.sizes() == gradients.sizes(),
              "head output/gradient mismatch");
  TORCH_CHECK(probabilities.size(1) == values.size(0),
              "candidate count mismatch");
  TORCH_CHECK(values.size(1) == head_outputs.size(1), "head_dim mismatch");
  TORCH_CHECK(probabilities.device() == values.device() &&
                  probabilities.device() == head_outputs.device() &&
                  probabilities.device() == gradients.device(),
              "score inputs must share one CUDA device");
  TORCH_CHECK(denominator_epsilon > 0.0 && denominator_epsilon < 1.0,
              "denominator_epsilon must be in (0,1)");

  c10::cuda::CUDAGuard guard(values.device());
  const int groups = probabilities.size(0);
  const int candidates = probabilities.size(1);
  const int head_dim = values.size(1);
  auto scores = torch::empty({candidates}, probabilities.options());
  if (candidates == 0) {
    return scores;
  }
  constexpr int threads = 256;
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::kHalf, at::kBFloat16, values.scalar_type(),
      "decision_risk_scores", [&] {
        decision_risk_scores_kernel<scalar_t><<<candidates, threads, 0, stream>>>(
            scores.data_ptr<float>(),
            probabilities.data_ptr<float>(),
            values.data_ptr<scalar_t>(),
            head_outputs.data_ptr<float>(),
            gradients.data_ptr<float>(),
            groups,
            candidates,
            head_dim,
            base_margin,
            static_cast<float>(denominator_epsilon));
      });
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return scores;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("append_flattened_cache", &append_flattened_cache);
  module.def("compact_flattened_kv", &compact_flattened_kv);
  module.def("decision_risk_scores", &decision_risk_scores);
}

