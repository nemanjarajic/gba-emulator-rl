"""A mosaic of instance screens, written as a PNG for watching a run."""

import os
import struct
import time
import zlib

import numpy as np

from gba_env import GbaVecEnv


def screens(env: GbaVecEnv, count: int) -> np.ndarray:
    """(rows*160, cols*240, 3) RGB grid of the first `count` instances.

    Full-colour 240x160 framebuffers rather than the grayscale observation, so
    the game is recognisable; the policy sees the observation, not this.
    Instances are spread evenly across the pool rather than taking the first
    few, which all tend to behave alike early in training.
    """
    count = max(1, min(count, env.num_instances))
    cols = int(np.ceil(np.sqrt(count)))
    rows = int(np.ceil(count / cols))
    picks = np.linspace(0, env.num_instances - 1, count).astype(int)
    grid = np.zeros((rows * 160, cols * 240, 3), dtype=np.uint8)
    for k, i in enumerate(picks):
        r, c = divmod(k, cols)
        grid[r * 160:(r + 1) * 160, c * 240:(c + 1) * 240] = env.framebuffer(int(i))
    # A one-pixel dark border between screens.
    grid[::160] = 24
    grid[:, ::240] = 24
    return grid


def write_png(path: str, rgb: np.ndarray) -> None:
    """Writes atomically, so a viewer never reads a half-written file."""
    h, w, _ = rgb.shape
    raw = np.empty((h, w * 3 + 1), dtype=np.uint8)
    raw[:, 0] = 0
    raw[:, 1:] = rgb.reshape(h, w * 3)

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw.tobytes(), 1)) + chunk(b"IEND", b""))
    # Windows refuses to replace a file another process has open, and the
    # viewer opens this one every time it changes. A screenshot is never worth
    # stopping a run for, so the rename is retried briefly and then given up on.
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(png)
    for attempt in range(5):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.05 * (attempt + 1))
    try:
        os.remove(tmp)
    except OSError:
        pass
