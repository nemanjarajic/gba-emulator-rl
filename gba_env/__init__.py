"""A vectorised Game Boy Advance environment backed by a GPU emulator.

Thousands of independent machines run in one compute shader, each with its own
controller input, observation and episode lifetime.

    from gba_env import GbaVecEnv, Button

    env = GbaVecEnv("game.gba", num_instances=1024)
    obs = env.reset()                     # (N, 80, 120) uint8
    obs = env.step(actions, frames=4)     # actions: (N,) uint16 button masks
    score = env.probe([0x02024EA4])       # (1, N) uint32
"""

from .env import Button, GbaVecEnv

__all__ = ["GbaVecEnv", "Button"]
