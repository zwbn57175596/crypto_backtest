"""Crypto backtest engine for USDT-margined perpetual futures."""

__version__ = "0.1.0"


def _bootstrap_cuda_paths() -> None:
    """Make pip-installed ``nvidia-cuda-*-cu12`` discoverable by Numba on Windows.

    Numba 0.65.x only auto-detects CUDA inside conda environments. With a pip
    install, cudart/nvrtc DLLs land in separate ``site-packages/nvidia/cuda_*``
    dirs that Numba's ``$CUDA_HOME/bin`` search never visits. We consolidate them
    into ``cuda_nvcc/bin`` once and register DLL dirs for the Windows loader.
    Silent no-op when not on Windows or the nvidia packages aren't installed.
    """
    import os
    import sys
    from pathlib import Path

    if sys.platform != "win32":
        return

    nvcc = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia" / "cuda_nvcc"
    if not nvcc.exists():
        return

    nvidia = nvcc.parent
    nvcc_bin = nvcc / "bin"
    nvvm_bin = nvcc / "nvvm" / "bin"

    for src_dir in (nvidia / "cuda_runtime" / "bin", nvidia / "cuda_nvrtc" / "bin"):
        if not src_dir.exists():
            continue
        for dll in src_dir.glob("*.dll"):
            dst = nvcc_bin / dll.name
            if not dst.exists():
                try:
                    dst.write_bytes(dll.read_bytes())
                except OSError:
                    pass

    os.environ.setdefault("CUDA_HOME", str(nvcc))
    for d in (nvcc_bin, nvvm_bin):
        if d.exists():
            try:
                os.add_dll_directory(str(d))
            except OSError:
                pass


_bootstrap_cuda_paths()
