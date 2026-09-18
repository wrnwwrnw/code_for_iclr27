import os
from pathlib import Path
import site

from setuptools import setup

extensions, commands = [], {}
if os.environ.get("LAZY_RECALL_BUILD_CUDA", "1") != "0":
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

    include_dirs = []
    if CUDA_HOME:
        include_dirs.append(str(Path(CUDA_HOME) / "include"))
    for directory in site.getsitepackages():
        include_dirs.extend(str(path) for path in (Path(directory) / "nvidia").glob("*/include"))
    flags = {"cxx": ["-O3", "-std=c++17"],
             "nvcc": ["-O3", "-std=c++17", "--fmad=false",
                      "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
                      "-U__CUDA_NO_BFLOAT16_CONVERSIONS__"]}
    extensions = [
        CUDAExtension("lazy_recall._C", ["csrc/bindings.cpp", "csrc/scoring.cu", "csrc/recall.cu", "csrc/batch_append.cu"],
                      include_dirs=include_dirs, extra_compile_args=flags),
        CUDAExtension("lazy_recall._legacy_C", ["csrc/cache_legacy.cu"],
                      include_dirs=include_dirs, extra_compile_args=flags),
        CUDAExtension("lazy_recall._host", ["csrc/host.cpp"], include_dirs=include_dirs,
                      extra_compile_args={"cxx": ["-O3", "-std=c++17"]}),
    ]
    commands = {"build_ext": BuildExtension}
setup(ext_modules=extensions, cmdclass=commands)
