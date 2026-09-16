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


_BUILD_DIRS = ("build", "build/Release", "build/RelWithDebInfo", "cmake-build-release")


def _candidate_paths():
    """Where the shared library might be, most specific first.

    This package and the emulator are separate repositories, so the library is
    not simply "next to us". In order: an explicit path, an emulator checkout
    named by GBA_EMULATOR_ROOT, a sibling checkout next to this one, this
    package's own directory, and finally whatever the dynamic loader can find
    (which covers `cmake --install`).
    """
    explicit = os.environ.get("GBA_ENV_LIB")
    if explicit:
        yield explicit

    roots = []
    named = os.environ.get("GBA_EMULATOR_ROOT")
    if named:
        roots.append(named)

    here = os.path.dirname(os.path.abspath(__file__))
    package_root = os.path.abspath(os.path.join(here, ".."))
    # One level up is the emulator repository when this package lives inside it,
    # and the directory holding both checkouts when it does not. Try it as a
    # root, then look inside it for a checkout by name.
    neighbourhood = os.path.abspath(os.path.join(package_root, ".."))
    roots.append(neighbourhood)
    for name in ("gba", "gba-gpu", "gba-emulator"):
        roots.append(os.path.join(neighbourhood, name))

    for root in roots:
        for build in _BUILD_DIRS:
            yield os.path.join(root, build, _LIB_NAME)

    yield os.path.join(here, _LIB_NAME)
    yield _LIB_NAME  # the dynamic loader's own search path


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


# The version of gba_env.h these bindings were written against. The emulator
# and this package live in separate repositories, so a mismatch is a real
# possibility rather than a theoretical one, and it would otherwise show up as
# a crash or as silently wrong data.
EXPECTED_ABI = 1

lib, lib_path = load()

if hasattr(lib, "gba_env_abi_version"):
    lib.gba_env_abi_version.restype = ctypes.c_uint32
    found = lib.gba_env_abi_version()
    if found != EXPECTED_ABI:
        raise RuntimeError(
            f"{lib_path} speaks gba_env ABI {found}, these bindings expect "
            f"{EXPECTED_ABI}. Rebuild the emulator, or use bindings matching it."
        )
else:
    raise RuntimeError(
        f"{lib_path} predates gba_env_abi_version and is too old for these "
        f"bindings (which expect ABI {EXPECTED_ABI}). Rebuild the emulator."
    )

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
