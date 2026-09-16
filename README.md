# Reinforcement learning interface

A vectorised Game Boy Advance environment: thousands of independent machines
running in one GPU compute shader, each with its own controller input,
observation and episode lifetime.

**There is no model here, by design.** This directory holds the environment and
nothing else; a training script or agent belongs alongside it, not inside the
emulator.

## Setup

Needs the emulator built, which produces `libgba_env`:

```sh
cd /path/to/gba && cmake --build build

python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python examples/smoke_test.py
```

The Python side is ctypes, so there is nothing to compile. It finds the shared
library through `GBA_ENV_LIB`, `GBA_EMULATOR_ROOT`, a sibling checkout, or an
installed location -- see `SPLITTING.md`, which also covers moving this
directory into a repository of its own.

The interface is versioned: loading a library built against a different
`GBA_ENV_ABI_VERSION` raises immediately with both numbers rather than
crashing or returning wrong data.

## Use

```python
from gba_env import Button, GbaVecEnv
import numpy as np

with GbaVecEnv("game.gba", num_instances=4096) as env:
    env.reset()

    # Drive the game to wherever episodes should begin, then capture. The
    # snapshot is a whole machine, memory included, so a reset genuinely
    # restarts the episode rather than leaving the game mid-scene.
    for _ in range(200):
        env.step(np.zeros(env.num_instances, dtype=np.uint16))
    env.capture_reset_point(0)
    env.save_reset_point("start.state")

    obs = env.step(actions, frames=4)     # (N, 40, 60) uint8, action repeat 4
    score = env.probe([0x02024EA4])[0]    # (N,) uint32, straight from game RAM
    env.reset(np.flatnonzero(done))       # only the finished episodes
```

`observations` is a **view into the emulator's buffer, not a copy**. It changes
under any reference held across a step; copy it if you need to keep it. (The
smoke test checks this, having originally got it wrong.)

## Throughput

Pokemon Emerald, action repeat 4, measured through the Python API on an
Apple M4:

| instances | agent steps/s | emulated frames/s | × realtime |
|---|---|---|---|
| 256 | 43 | 172 | 3 |
| 1024 | 168 | 672 | 11 |
| 4096 | **397** | 1588 | 27 |

Two things to be clear about before planning a training run:

- **397 steps/s is modest.** Roughly 1.4M agent steps an hour, so a run of the
  size normally quoted for Atari would take days. Atari vectorised environments
  reach tens of thousands of steps a second, but a GBA is a far heavier machine
  than an Atari and the comparison is not fair; it does set expectations.
- **On this Mac the GPU is about 8x one CPU core, on a 10-core machine.** A
  properly threaded CPU emulator would be roughly break-even here. The GPU's
  real arguments are that it leaves the CPU free for the network, and that the
  ratio should be considerably better on a discrete card. Measure before
  assuming the GPU is the right answer for your hardware.

## Things worth knowing

**The observation cannot resolve text.** It is the 240x160 screen reduced 4x to
60x40 grayscale, so an 8x8 pixel character becomes 2x2. For a dialogue-driven
game like Pokemon the signal is in `probe()`, reading the game's own RAM, not in
the pixels. Finding those addresses is game-specific work: a RAM map, or a
memory search while watching a value change.

**Save memory is optional and usually wanted.** `flags=NO_SAVE` reclaims 128 KiB
an instance, a quarter of the footprint. But some games refuse to run without
it: Emerald identifies its flash chip on boot and, finding none, sets its main
callback to NULL and draws nothing forever. Check the game boots first.

**Instances are deterministic and identical.** Every one starts from the same
state and only the actions differ, so exploration has to come from the policy.
There is no per-instance RNG seeding or sticky-action support yet.

## Not implemented

- Frame stacking, configurable observation size, colour observations. The
  observation is fixed at 60x40 grayscale; changing it means editing `OBS_W`
  and `OBS_H` in `src/core/memmap.h` and rebuilding.
- Episode termination. Nothing detects when an episode is over; that is the
  caller's job, usually from a probe.
- A Gymnasium or EnvPool adapter.
