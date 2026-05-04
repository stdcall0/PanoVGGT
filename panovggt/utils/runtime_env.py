import ctypes
import os
import sys
import tempfile
from typing import Optional


_ENV_READY_FLAG = "PANOVGGT_GSPLAT_ENV_READY"
_LIBS_PRELOADED_FLAG = "PANOVGGT_GSPLAT_LIBS_PRELOADED"
_ARCH_LIST_READY_FLAG = "PANOVGGT_GSPLAT_ARCH_LIST_READY"


def _prepend_env_path(name: str, value: str) -> None:
    if not value or not os.path.exists(value):
        return
    current = [path for path in os.environ.get(name, "").split(os.pathsep) if path]
    if value not in current:
        os.environ[name] = os.pathsep.join([value] + current)


def loaded_libstdcpp_path() -> Optional[str]:
    try:
        with open("/proc/self/maps", "r", encoding="utf-8") as handle:
            for line in handle:
                if "libstdc++.so.6" not in line:
                    continue
                parts = line.strip().split()
                if parts:
                    return parts[-1]
    except OSError:
        return None
    return None


def configure_gsplat_build_env() -> None:
    if os.environ.get(_ENV_READY_FLAG) == "1":
        return

    prefix = os.environ.get("CONDA_PREFIX")
    if not prefix:
        return

    nvcc_path = os.path.join(prefix, "bin", "nvcc")
    if os.path.isfile(nvcc_path):
        os.environ.setdefault("CUDA_HOME", prefix)
        os.environ.setdefault("CUDA_PATH", prefix)
        os.environ.setdefault("CUDACXX", nvcc_path)
        _prepend_env_path("PATH", os.path.join(prefix, "bin"))

    include_candidates = [
        os.path.join(prefix, "targets", "x86_64-linux", "include"),
        os.path.join(prefix, "include"),
    ]
    lib_candidates = [
        os.path.join(prefix, "targets", "x86_64-linux", "lib"),
        os.path.join(prefix, "lib"),
    ]

    for path in include_candidates:
        _prepend_env_path("CPATH", path)
        _prepend_env_path("CPLUS_INCLUDE_PATH", path)

    for path in lib_candidates:
        _prepend_env_path("LIBRARY_PATH", path)
        _prepend_env_path("LD_LIBRARY_PATH", path)

    os.environ.setdefault(
        "TORCH_EXTENSIONS_DIR", os.path.join(tempfile.gettempdir(), "torch_extensions")
    )
    os.environ[_ENV_READY_FLAG] = "1"


def preload_conda_runtime_libraries() -> None:
    if os.environ.get(_LIBS_PRELOADED_FLAG) == "1":
        return

    prefix = os.environ.get("CONDA_PREFIX")
    if not prefix:
        return

    for library_name in ("libgcc_s.so.1", "libstdc++.so.6"):
        library_path = os.path.join(prefix, "lib", library_name)
        if os.path.isfile(library_path):
            ctypes.CDLL(library_path, mode=ctypes.RTLD_GLOBAL)

    os.environ[_LIBS_PRELOADED_FLAG] = "1"


def configure_gsplat_cuda_arch_list() -> None:
    if os.environ.get(_ARCH_LIST_READY_FLAG) == "1":
        return

    current_arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST")
    if current_arch_list:
        unsafe_legacy_arch = False
        for raw_arch in current_arch_list.replace(" ", ";").split(";"):
            arch = raw_arch.strip().replace("+PTX", "")
            if not arch:
                continue
            try:
                major = int(arch.split(".", 1)[0])
            except ValueError:
                continue
            unsafe_legacy_arch = unsafe_legacy_arch or major < 7
        if not unsafe_legacy_arch:
            os.environ[_ARCH_LIST_READY_FLAG] = "1"
            return

    default_arch_list = os.environ.get("PANOVGGT_DEFAULT_CUDA_ARCH_LIST", "9.0")
    torch_module = sys.modules.get("torch")
    if torch_module is None:
        # This bootstrap is often called before importing torch so that gsplat
        # sees the environment early.  Avoid PyTorch's broad default JIT arch
        # list, which includes legacy SMs where gsplat's cooperative-groups
        # kernels do not compile.
        os.environ["TORCH_CUDA_ARCH_LIST"] = default_arch_list
        os.environ[_ARCH_LIST_READY_FLAG] = "1"
        return

    try:
        if not torch_module.cuda.is_available():
            os.environ.setdefault("TORCH_CUDA_ARCH_LIST", default_arch_list)
            os.environ[_ARCH_LIST_READY_FLAG] = "1"
            return
        capabilities = {
            torch_module.cuda.get_device_capability(device_idx)
            for device_idx in range(torch_module.cuda.device_count())
        }
    except Exception:
        os.environ["TORCH_CUDA_ARCH_LIST"] = default_arch_list
        os.environ[_ARCH_LIST_READY_FLAG] = "1"
        return

    # gsplat 1.5.x uses cooperative_groups::labeled_partition in kernels, which
    # does not compile for legacy architectures included by PyTorch's default
    # JIT arch list.  Restrict the JIT build to the actually visible GPUs.
    capabilities = sorted(cap for cap in capabilities if cap[0] >= 7)
    if not capabilities:
        return

    os.environ["TORCH_CUDA_ARCH_LIST"] = ";".join(
        f"{major}.{minor}" for major, minor in capabilities
    )
    os.environ[_ARCH_LIST_READY_FLAG] = "1"


def bootstrap_gsplat_runtime(preload_runtime_libs: bool = True) -> None:
    configure_gsplat_build_env()
    if preload_runtime_libs:
        preload_conda_runtime_libraries()
    configure_gsplat_cuda_arch_list()
