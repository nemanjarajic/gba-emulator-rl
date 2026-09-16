"""Checks the environment end to end, from Python.

Uses build/roms/input_echo.gba, a ROM that publishes its controller state to
EWRAM word 0 every loop. That makes the action path verifiable rather than
merely plausible: whatever action is sent must come back out of the emulated
machine's own memory.

    rl/.venv/bin/python rl/examples/smoke_test.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gba_env import Button, GbaVecEnv  # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ROM = os.path.join(REPO, "build", "roms", "input_echo.gba")

failures = 0


def check(ok, what):
    global failures
    print(f"  {what:<54} {'ok' if ok else 'FAIL'}")
    if not ok:
        failures += 1


def main():
    if not os.path.exists(ROM):
        print(f"missing {ROM}\nrun: python3 tools/make_bench_roms.py")
        return 2

    n = 64
    with GbaVecEnv(ROM, num_instances=n) as env:
        print(f"{env.num_instances} instances, observations "
              f"{env.obs_height}x{env.obs_width}, library {__import__('gba_env')._native.lib_path}")
        print()

        obs = env.reset()
        check(obs.shape == (n, env.obs_height, env.obs_width), "reset returns (N, H, W)")
        check(obs.dtype == np.uint8, "observations are uint8")

        # observations is a view into the emulator's buffer, not a copy, so it
        # changes underneath any reference held across a step. Copy to keep it.
        # An earlier version of this test held the array across a step and then
        # asserted it was still blank, which it was not.
        at_reset = obs.copy()
        check(int(at_reset.max()) == 0, "nothing has been drawn before the first step")

        # Every instance gets a different action; the ROM echoes each one back
        # to its own EWRAM, so the whole input path is checked from Python.
        rng = np.random.default_rng(0)
        actions = rng.integers(0, 1 << 10, size=n, dtype=np.uint16)
        env.step(actions, frames=4)

        echoed = env.probe([0x02000000])[0]
        expected = 0x3FF & ~actions.astype(np.uint32)
        check(np.array_equal(echoed, expected),
              "every instance read back the action it was given")

        # Distinct actions must produce distinct machines.
        check(len(np.unique(echoed)) > n // 2, "instances diverge under different actions")

        # The ROM paints a white band, so the picture is really arriving.
        obs = env.step(actions, frames=1)
        check(int(obs[0].max()) == 255, "observation contains the ROM's white band")
        check(int(at_reset.max()) == 0, "the copy taken at reset was not overwritten")
        check(obs.ctypes.data == env.observations.ctypes.data,
              "observations is a live view, not a fresh copy each call")

        # A reset point is a whole machine, memory included.
        env.step(np.full(n, int(Button.A | Button.START), dtype=np.uint16), frames=2)
        env.capture_reset_point(0)
        before = env.probe([0x02000000])[0][0]

        env.step(np.zeros(n, dtype=np.uint16), frames=4)
        changed = env.probe([0x02000000])[0][0]
        check(changed != before, "stepping changes the machine")

        env.reset([0])
        after = env.probe([0x02000000])[0][0]
        check(after == before, "reset restores the captured machine")

        others = env.probe([0x02000000])[0][1:]
        check(np.all(others == changed), "resetting one instance leaves the others alone")

        # Round-trip a reset point through a file.
        path = os.path.join(REPO, "build", "smoke.state")
        env.save_reset_point(path)
        env.step(np.zeros(n, dtype=np.uint16), frames=4)
        env.load_reset_point(path)
        env.reset([0])
        check(env.probe([0x02000000])[0][0] == before, "reset point survives a save and load")
        os.remove(path)

        fb = env.framebuffer(0)
        check(fb.shape == (160, 240, 3), "framebuffer is (160, 240, 3) RGB")

    print()
    if failures:
        print(f"SMOKE TEST FAILED: {failures} checks failed")
        return 1
    print("SMOKE TEST PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
