"""Compile a .cu file into a shared library with nvcc and load it with ctypes.

Why not torch.utils.cpp_extension? On Windows it needs ninja plus a full
PyTorch C++ build environment. Plain nvcc -> DLL -> ctypes keeps the kernels as
ordinary CUDA C++ with no PyTorch headers, and Python just passes raw device
pointers (tensor.data_ptr()) and PyTorch's current stream.

Builds are cached in kernels/cuda/build/, keyed by a hash of the sources and
flags, so editing a kernel triggers a rebuild on the next import and a DLL that
is already loaded (and locked by Windows) is never overwritten.
"""
from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

CUDA_DIR = Path(__file__).resolve().parent
BUILD_DIR = CUDA_DIR / "build"
SHARED_HEADERS = [CUDA_DIR / "common.cuh"]
IS_WINDOWS = sys.platform == "win32"


@lru_cache(maxsize=None)
def _msvc_env() -> dict[str, str] | None:
    """Environment with MSVC (cl.exe) on PATH, which nvcc needs on Windows."""
    if not IS_WINDOWS or shutil.which("cl"):
        return None
    vswhere = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / (
        r"Microsoft Visual Studio\Installer\vswhere.exe"
    )
    if not vswhere.exists():
        raise RuntimeError("cl.exe not on PATH and vswhere.exe not found; install VS Build Tools")
    install = subprocess.check_output(
        [str(vswhere), "-latest", "-products", "*", "-property", "installationPath"], text=True
    ).strip()
    vcvars = Path(install) / r"VC\Auxiliary\Build\vcvars64.bat"
    out = subprocess.check_output(f'cmd /s /c ""{vcvars}" >nul 2>nul && set"', text=True, shell=True)
    env = dict(line.split("=", 1) for line in out.splitlines() if "=" in line)
    return env


def _arch() -> str:
    import torch

    major, minor = torch.cuda.get_device_capability()
    return f"sm_{major}{minor}"


def _nvcc_flags() -> list[str]:
    flags = ["-O3", "-lineinfo", "-std=c++17", f"-arch={_arch()}", "-shared", f"-I{CUDA_DIR}"]
    if IS_WINDOWS:
        flags += ["-Xcompiler", "/wd4819"]  # silence code-page warnings in CUDA headers
    else:
        flags += ["-Xcompiler", "-fPIC"]
    return flags


@lru_cache(maxsize=None)
def load(source: str | os.PathLike) -> ctypes.CDLL:
    """Compile `source` (if needed) and return the loaded library."""
    src = Path(source)
    if not src.is_absolute():
        src = CUDA_DIR / src
    flags = _nvcc_flags()

    h = hashlib.sha256()
    for f in [src, *SHARED_HEADERS]:
        h.update(f.read_bytes())
    h.update(" ".join(flags).encode())
    ext = ".dll" if IS_WINDOWS else ".so"
    lib_path = BUILD_DIR / f"{src.stem}_{h.hexdigest()[:12]}{ext}"

    if not lib_path.exists():
        BUILD_DIR.mkdir(exist_ok=True)
        nvcc = shutil.which("nvcc") or "nvcc"
        cmd = [nvcc, *flags, "-o", str(lib_path), str(src)]
        print(f"[build] nvcc {src.relative_to(CUDA_DIR)} -> {lib_path.name}", file=sys.stderr)
        proc = subprocess.run(cmd, env=_msvc_env(), capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"nvcc failed:\n{' '.join(cmd)}\n{proc.stdout}\n{proc.stderr}")

    return ctypes.CDLL(str(lib_path))
