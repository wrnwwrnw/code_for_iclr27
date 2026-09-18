#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> temporal_probabilities_cuda(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, double);
torch::Tensor temporal_risk_cuda(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, double);
std::vector<torch::Tensor> append_cuda(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t);
std::vector<torch::Tensor> compact_cuda(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);
torch::Tensor pool_cuda(torch::Tensor, torch::Tensor, int64_t, int64_t);
std::vector<torch::Tensor> merge_cuda(
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor);
void pack_cuda(torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
               torch::Tensor, int64_t, torch::Tensor, torch::Tensor);
std::vector<torch::Tensor> batch_append_cuda(
    std::vector<torch::Tensor>, std::vector<torch::Tensor>, std::vector<torch::Tensor>,
    torch::Tensor, torch::Tensor, std::vector<std::vector<int64_t>>, std::vector<int64_t>);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("temporal_probabilities", &temporal_probabilities_cuda);
    module.def("temporal_risk", &temporal_risk_cuda);
    module.def("append", &append_cuda);
    module.def("compact", &compact_cuda);
    module.def("pool", &pool_cuda);
    module.def("merge", &merge_cuda);
    module.def("pack", &pack_cuda);
    module.def("batch_append", &batch_append_cuda);
}
