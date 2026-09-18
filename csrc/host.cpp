#include <torch/extension.h>
#include <cuda_runtime_api.h>
#include <cstdint>

void register_host(uint64_t address, uint64_t bytes) {
    auto status = cudaHostRegister(reinterpret_cast<void*>(address), bytes, cudaHostRegisterDefault);
    TORCH_CHECK(status == cudaSuccess, "cudaHostRegister failed: ", cudaGetErrorString(status));
}

void unregister_host(uint64_t address) {
    auto status = cudaHostUnregister(reinterpret_cast<void*>(address));
    TORCH_CHECK(status == cudaSuccess, "cudaHostUnregister failed: ", cudaGetErrorString(status));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("register_host", &register_host);
    module.def("unregister_host", &unregister_host);
}

