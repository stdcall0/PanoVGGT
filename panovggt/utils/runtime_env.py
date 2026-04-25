import ctypes
import os
import tempfile
from typing import Optional


_ENV_READY_FLAG = "PANOVGGT_GSPLAT_ENV_READY"
_LIBS_PRELOADED_FLAG = "PANOVGGT_GSPLAT_LIBS_PRELOADED"


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


def bootstrap_gsplat_runtime(preload_runtime_libs: bool = True) -> None:
    configure_gsplat_build_env()
    if preload_runtime_libs:
        preload_conda_runtime_libraries()
