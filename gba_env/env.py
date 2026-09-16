"""The vectorised environment itself."""

import ctypes
import enum
from typing import Iterable, Optional, Sequence

import numpy as np

from . import _native

_lib = _native.lib


class Button(enum.IntFlag):
    """Buttons, in the order the hardware numbers them.

    An action is the bitwise OR of the buttons held. The register's active-low
    encoding is handled inside the emulator, so nothing here has to invert.
    """

    NONE = 0
    A = 1 << 0
    B = 1 << 1
    SELECT = 1 << 2
    START = 1 << 3
    RIGHT = 1 << 4
    LEFT = 1 << 5
    UP = 1 << 6
    DOWN = 1 << 7
    R = 1 << 8
    L = 1 << 9


NO_SAVE = 1
"""Creation flag: skip save memory, reclaiming 128 KiB per instance.

A quarter of the footprint, and a rollout that never saves does not need it.
Some games refuse to run without it: Pokemon Emerald identifies its flash chip
on boot and gives up if there is none.
"""


class GbaVecEnv:
    """Many independent GBA machines stepped together."""

    def __init__(self, rom_path: str, num_instances: int = 64, flags: int = 0):
        handle = _lib.gba_env_create(rom_path.encode(), num_instances, flags)
        if not handle:
            raise RuntimeError(_lib.gba_env_last_error().decode())
        self._handle = ctypes.c_void_p(handle)
        self.num_instances = _lib.gba_env_num_instances(self._handle)
        self.obs_width = _lib.gba_env_obs_width()
        self.obs_height = _lib.gba_env_obs_height()
        self._actions = np.zeros(self.num_instances, dtype=np.uint16)

    # -- lifetime ----------------------------------------------------------

    def close(self) -> None:
        if getattr(self, "_handle", None):
            _lib.gba_env_destroy(self._handle)
            self._handle = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # -- episodes ----------------------------------------------------------

    def capture_reset_point(self, instance: int = 0) -> None:
        """Captures one instance's whole machine as the point resets return to.

        Drive a game to wherever episodes should begin -- past the title screen,
        into a battle -- then capture. Memory is included, so a reset really
        does start the episode over rather than leaving the game mid-scene.
        """
        if not _lib.gba_env_capture_reset_point(self._handle, instance):
            raise RuntimeError(_lib.gba_env_last_error().decode())

    def save_reset_point(self, path: str) -> None:
        if not _lib.gba_env_save_reset_point(self._handle, path.encode()):
            raise RuntimeError(_lib.gba_env_last_error().decode())

    def load_reset_point(self, path: str) -> None:
        if not _lib.gba_env_load_reset_point(self._handle, path.encode()):
            raise RuntimeError(_lib.gba_env_last_error().decode())

    def reset(self, instances: Optional[Iterable[int]] = None) -> np.ndarray:
        """Resets some or all instances. Others keep running untouched."""
        if instances is None:
            _lib.gba_env_reset_all(self._handle)
        else:
            idx = np.ascontiguousarray(list(instances), dtype=np.uint32)
            if idx.size:
                _lib.gba_env_reset(
                    self._handle,
                    idx.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
                    idx.size,
                )
        return self.observations

    # -- stepping ----------------------------------------------------------

    def step(self, actions: Optional[Sequence[int]] = None, frames: int = 4) -> np.ndarray:
        """Holds each action for `frames` frames, the usual action repeat.

        Only the final frame is rendered: rendering does not affect the emulated
        machine, so skipping it on repeated frames costs nothing in accuracy.
        """
        if actions is not None:
            np.copyto(self._actions, np.asarray(actions, dtype=np.uint16).reshape(-1))
        _lib.gba_env_step(
            self._handle,
            self._actions.ctypes.data_as(ctypes.POINTER(ctypes.c_uint16)),
            frames,
        )
        return self.observations

    # -- reading state out --------------------------------------------------

    @property
    def observations(self) -> np.ndarray:
        """(N, height, width) uint8 grayscale, a view rather than a copy.

        Valid until the next step or reset; copy it if you need to keep it.
        """
        ptr = _lib.gba_env_observations(self._handle)
        count = self.num_instances * self.obs_height * self.obs_width
        flat = np.ctypeslib.as_array(ptr, shape=(count,))
        return flat.reshape(self.num_instances, self.obs_height, self.obs_width)

    def probe(self, addresses: Sequence[int]) -> np.ndarray:
        """Reads GBA addresses from every instance: (len(addresses), N) uint32.

        This is how a reward is read -- a score or a counter in the game's RAM.
        At 120x80 the observation cannot resolve text, so for a dialogue-driven
        game this, not the pixels, is where the signal is.
        """
        addr = np.ascontiguousarray(addresses, dtype=np.uint32)
        out = np.zeros(addr.size * self.num_instances, dtype=np.uint32)
        _lib.gba_env_read_probes(
            self._handle,
            addr.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
            addr.size,
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_uint32)),
        )
        return out.reshape(addr.size, self.num_instances)

    def framebuffer(self, instance: int = 0) -> np.ndarray:
        """One instance's full 240x160 screen as (160, 240, 3) uint8 RGB.

        For looking at, not for training: the observation is what the model
        should see.
        """
        ptr = _lib.gba_env_framebuffer(self._handle, instance)
        if not ptr:
            raise IndexError(f"instance {instance} out of range")
        words = np.ctypeslib.as_array(ptr, shape=(240 * 160 // 2,))
        px = np.empty(240 * 160, dtype=np.uint16)
        px[0::2] = words & 0xFFFF
        px[1::2] = words >> 16
        r = (px & 31).astype(np.uint16)
        g = ((px >> 5) & 31).astype(np.uint16)
        b = ((px >> 10) & 31).astype(np.uint16)
        rgb = np.stack([(c << 3) | (c >> 2) for c in (r, g, b)], axis=-1).astype(np.uint8)
        return rgb.reshape(160, 240, 3)
