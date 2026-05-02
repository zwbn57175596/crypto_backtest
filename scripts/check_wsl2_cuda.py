#!/usr/bin/env python3
"""Validate CUDA availability for this project inside WSL2.

This script focuses on the path this repository actually uses:
Numba + numpy + CUDA from inside WSL2.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import traceback


def print_section(title: str) -> None:
    print(f"\n== {title} ==")


def print_ok(message: str) -> None:
    print(f"[OK] {message}")


def print_warn(message: str) -> None:
    print(f"[WARN] {message}")


def print_fail(message: str) -> None:
    print(f"[FAIL] {message}")


def run_command(cmd: list[str]) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return 127, "", f"command not found: {cmd[0]}"
    return proc.returncode, proc.stdout.strip(), proc.stderr.strip()


def detect_wsl2() -> bool:
    if "WSL_DISTRO_NAME" in os.environ:
        return True
    try:
        with open("/proc/version", encoding="utf-8") as f:
            version = f.read().lower()
        return "microsoft" in version or "wsl2" in version
    except OSError:
        return False


def check_system() -> None:
    print_section("System")
    print(f"Python:   {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"Kernel:   {platform.release()}")

    if sys.platform != "linux":
        print_warn("This script is intended for Linux/WSL2, but the current platform is not Linux.")
    else:
        print_ok("Linux platform detected.")

    if detect_wsl2():
        print_ok("WSL environment detected.")
    else:
        print_warn("WSL2 markers were not detected. CUDA may still work on native Linux.")


def check_nvidia_smi() -> None:
    print_section("GPU Driver")

    candidates = [
        shutil.which("nvidia-smi"),
        "/usr/lib/wsl/lib/nvidia-smi",
    ]

    for candidate in candidates:
        if not candidate:
            continue
        code, out, err = run_command([candidate])
        if code == 0:
            print_ok(f"nvidia-smi is available at: {candidate}")
            lines = [line for line in out.splitlines() if line.strip()]
            for line in lines[:8]:
                print(line)
            return
        print_warn(f"Found {candidate}, but it failed with exit code {code}.")
        if err:
            print(err)

    print_fail("Could not run nvidia-smi from PATH or /usr/lib/wsl/lib/nvidia-smi.")
    print("Hint: On WSL2, install/update the NVIDIA Windows driver and then run `wsl --update` on Windows.")


def check_python_packages() -> tuple[object | None, object | None]:
    print_section("Python Packages")

    try:
        import numpy as np  # type: ignore

        print_ok(f"numpy import OK ({np.__version__})")
    except Exception as exc:  # pragma: no cover - diagnostic path
        print_fail(f"numpy import failed: {exc}")
        np = None

    try:
        import numba  # type: ignore
        from numba import cuda  # type: ignore

        print_ok(f"numba import OK ({numba.__version__})")
        print_ok("numba.cuda import OK")
    except Exception as exc:  # pragma: no cover - diagnostic path
        print_fail(f"numba/cuda import failed: {exc}")
        cuda = None

    return np, cuda


def check_numba_cuda(np: object | None, cuda: object | None) -> int:
    print_section("Numba CUDA")

    if np is None or cuda is None:
        print_fail("Skipping CUDA runtime checks because numpy/numba is unavailable.")
        return 2

    try:
        available = cuda.is_available()
        print(f"cuda.is_available(): {available}")
        if not available:
            print_fail("Numba cannot access CUDA in this environment.")
            try:
                cuda.detect()
            except Exception as detect_exc:
                print("numba.cuda.detect() raised:")
                print(detect_exc)
            return 2

        device = cuda.get_current_device()
        print_ok(f"Current device: {device.name.decode() if isinstance(device.name, bytes) else device.name}")
        print(f"Compute capability: {device.compute_capability}")
        print(f"Multiprocessors:    {device.MULTIPROCESSOR_COUNT}")
        print(f"Max threads/block:  {device.MAX_THREADS_PER_BLOCK}")

        free_bytes, total_bytes = cuda.current_context().get_memory_info()
        print(f"Memory free/total:  {free_bytes / 1024**3:.2f} GiB / {total_bytes / 1024**3:.2f} GiB")
    except Exception as exc:
        print_fail(f"CUDA runtime query failed: {exc}")
        traceback.print_exc()
        return 2

    try:
        # Small end-to-end kernel check: if this passes, the project's CUDA path
        # has the essentials it needs.
        @cuda.jit
        def add_kernel(a, b, out):
            idx = cuda.grid(1)
            if idx < out.size:
                out[idx] = a[idx] + b[idx]

        a = np.arange(1024, dtype=np.float32)
        b = np.arange(1024, dtype=np.float32) * 2
        out = np.zeros_like(a)

        d_a = cuda.to_device(a)
        d_b = cuda.to_device(b)
        d_out = cuda.to_device(out)

        threads = 128
        blocks = (out.size + threads - 1) // threads
        add_kernel[blocks, threads](d_a, d_b, d_out)
        cuda.synchronize()

        result = d_out.copy_to_host()
        expected = a + b
        if not np.allclose(result, expected):
            print_fail("CUDA kernel executed, but the result verification failed.")
            return 2

        print_ok("CUDA kernel launch and result verification succeeded.")
        return 0
    except Exception as exc:
        print_fail(f"CUDA kernel smoke test failed: {exc}")
        traceback.print_exc()
        return 2


def main() -> int:
    print("WSL2 CUDA validation for crypto_backtest")
    check_system()
    check_nvidia_smi()
    np, cuda = check_python_packages()
    rc = check_numba_cuda(np, cuda)

    print_section("Summary")
    if rc == 0:
        print_ok("CUDA looks ready for this project's Numba-based GPU path.")
        print("Next step: run the project's optimizer with `--method cuda-grid`.")
    else:
        print_warn("CUDA is not fully ready yet for this project.")
        print("Typical fixes:")
        print("  1. Verify `nvidia-smi` works inside WSL2.")
        print("  2. Install Python deps: python3 -m pip install numpy numba")
        print("  3. If needed, install the CUDA toolkit for WSL Ubuntu without Linux GPU drivers.")

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
