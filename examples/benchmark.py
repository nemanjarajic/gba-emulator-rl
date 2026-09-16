"""Measures the throughput a training loop would actually see.

Reports agent steps per second -- actions taken, not frames emulated -- since
that is what sets wall-clock training time. With the usual action repeat of 4,
one agent step is four emulated frames.

    rl/.venv/bin/python rl/examples/benchmark.py <rom.gba> [instances] [steps]
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gba_env import GbaVecEnv  # noqa: E402


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    rom = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 20
    frames = 4

    with GbaVecEnv(rom, num_instances=n) as env:
        rng = np.random.default_rng(0)
        env.reset()
        env.step(np.zeros(n, dtype=np.uint16), frames=1)  # warm up

        t0 = time.perf_counter()
        for _ in range(steps):
            actions = rng.integers(0, 1 << 10, size=n, dtype=np.uint16)
            obs = env.step(actions, frames=frames)
            obs.mean()  # touch the data, as a real loop would
        secs = time.perf_counter() - t0

        agent_steps = steps * n
        emulated_frames = agent_steps * frames
        print(f"{os.path.basename(rom)}: {n} instances, action repeat {frames}")
        print(f"  {agent_steps / secs:10,.0f} agent steps/s")
        print(f"  {emulated_frames / secs:10,.0f} emulated frames/s "
              f"({emulated_frames / secs / 59.7:.0f}x realtime)")
        print(f"  {secs / steps * 1000:10.1f} ms per batched step")
    return 0


if __name__ == "__main__":
    sys.exit(main())
