"""ctypes bindings to libgba_env, the emulator's C ABI.

Deliberately ctypes rather than a compiled extension: there is nothing to build
on the Python side, and the buffers the library hands back can be wrapped as
NumPy views without copying.
"""

import ctypes
import os
import sys

_LIB_NAMES = {
    "darwin": "libgba_env.dylib",
    "win32": "gba_env.dll",
}
_LIB_NAME = _LIB_NAMES.get(sys.platform, "libgba_env.so")


def _candidate_paths():
    """Where the shared library might be, most specific first."""
    env = os.environ.get("GBA_ENV_LIB")
    if env:
        yield env
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, "..", ".."))
    for build in ("build", "build/Release", "build/RelWithDebInfo", "cmake-build-release"):
        yield os.path.join(repo, build, _LIB_NAME)
    yield os.path.join(here, _LIB_NAME)
    yield _LIB_NAME  # fall back to the loader's own search path


def load():
    tried = []
    for path in _candidate_paths():
        if path != _LIB_NAME and not os.path.exists(path):
            tried.append(path)
            continue
        try:
            return ctypes.CDLL(path), path
        except OSError as exc:  # pragma: no cover - platform specific
            tried.append(f"{path} ({exc})")
    raise RuntimeError(
        "cannot find "
        + _LIB_NAME
        + ". Build the emulator first (cmake --build build), or set GBA_ENV_LIB.\n"
        + "Looked in:\n  " + "\n  ".join(tried)
    )


lib, lib_path = load()

_u8p = ctypes.POINTER(ctypes.c_uint8)
_u32p = ctypes.POINTER(ctypes.c_uint32)
_u16p = ctypes.POINTER(ctypes.c_uint16)

lib.gba_env_create.argtypes = [ctypes.c_char_p, ctypes.c_uint32, ctypes.c_uint32]
lib.gba_env_create.restype = ctypes.c_void_p
lib.gba_env_destroy.argtypes = [ctypes.c_void_p]
lib.gba_env_last_error.restype = ctypes.c_char_p
lib.gba_env_num_instances.argtypes = [ctypes.c_void_p]
lib.gba_env_num_instances.restype = ctypes.c_uint32
lib.gba_env_obs_width.restype = ctypes.c_uint32
lib.gba_env_obs_height.restype = ctypes.c_uint32
lib.gba_env_capture_reset_point.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
lib.gba_env_capture_reset_point.restype = ctypes.c_int
lib.gba_env_save_reset_point.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
lib.gba_env_save_reset_point.restype = ctypes.c_int
lib.gba_env_load_reset_point.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
lib.gba_env_load_reset_point.restype = ctypes.c_int
lib.gba_env_reset.argtypes = [ctypes.c_void_p, _u32p, ctypes.c_uint32]
lib.gba_env_reset_all.argtypes = [ctypes.c_void_p]
lib.gba_env_step.argtypes = [ctypes.c_void_p, _u16p, ctypes.c_uint32]
lib.gba_env_observations.argtypes = [ctypes.c_void_p]
lib.gba_env_observations.restype = _u8p
lib.gba_env_read_probes.argtypes = [ctypes.c_void_p, _u32p, ctypes.c_uint32, _u32p]
lib.gba_env_framebuffer.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
lib.gba_env_framebuffer.restype = _u32p
