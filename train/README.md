# Training: Pokemon Emerald exploration with PPO

An agent that learns to explore Pokemon Emerald, trained with PPO directly
against `GbaVecEnv`. This lives beside the environment package, not in it:
`gba_env/` stays model-free.

## Setup

```sh
.venv/bin/pip install -r train/requirements.txt   # CUDA PyTorch
```

The emulator must be built, including `make_state`, from a version with the
Thumb format 8 and PPU VRAM fixes. Emerald crashes when it enters its first map
without them.

## 1. Build the start state

Every episode starts from a reset point: the player standing in the moving
truck at the end of the intro. It is made on the emulator's CPU core, because
getting through an intro is sequential work and a single GPU instance runs far
below real time:

```sh
<emulator>/build/make_state <emerald.gba> train/states/emerald_start.txt train/states/emerald_start.state \
    --png start.png
```

That takes about 40 seconds. `emerald_start.txt` is the input script; the
`.state` file is not committed, since it contains the game's memory.

## 2. Train

```sh
python -m train.ppo --rom <emerald.gba> --state train/states/emerald_start.state --run runs/emerald
```

It writes `runs/emerald/log.csv` after every update and `latest.pt` every ten;
`--resume` continues a run. Every option is listed by `--help`.

## What the agent sees, does and is paid for

- **Observation:** the last 3 screens at 120x80 grayscale, plus the current map
  number from RAM (the screen cannot tell many maps apart).
- **Actions:** UP, DOWN, LEFT, RIGHT, A, B, each held for 16 frames, which walks
  one tile. START and SELECT are left out; menus cost many steps and lead
  nowhere for exploration.
- **Reward**, all read from game RAM (`train/emerald.py`):

  | event | reward |
  |---|---|
  | standing on a tile not yet visited this episode | 0.02 |
  | entering a map not yet visited this episode | 1 |
  | the count of story flags set reaches a new high | 0.5 each |
  | the party's total level reaches a new high | 1 per level |
  | a badge | 20 |

  Progress rewards pay only for new highs, so nothing can be farmed by losing
  and regaining it.
- **Episodes:** 2048 steps, all instances together, every one from the start
  state.

## RAM addresses

For Emerald (USA/Europe), from the pokeemerald decompilation. The save block
moves to a random address, so it is found through `gSaveBlock1Ptr` every step.

| what | where | checked |
|---|---|---|
| save block 1 | pointer at `0x03005D8C` | yes |
| x, y | save block + 0, + 2 (s16) | yes: walking moves them |
| map group, number | save block + 4, + 5 | yes: truck is 25.40, Littleroot 0.9 |
| flags | save block + `0x1270`, bit per flag | yes: the game's own flag test indexes there |
| badges | flags `0x867`..`0x86E` | not yet |
| party count, party | `0x020244E9`, `0x020244EC` (100 bytes each) | count reads 0 before a starter |
| level | party slot + `0x54` | not yet |

## Throughput on an RTX 5060 Ti (8 GB)

About 340 agent steps/s at 4096 instances with PPO updates included, roughly
1.2M steps an hour. The emulator's 16 frames per step take most of it.

GPU memory is the limit. The emulator holds about 2.4 GB at 4096 instances,
PyTorch peaks near 1.9 GB, and the desktop takes the rest. If the two exceed
the card, Windows pages PyTorch's memory rather than failing and training slows
by two orders of magnitude, so watch `torch ... GiB` in the log line. Fewer
instances (`--num-envs`) or a smaller `--infer-chunk` and `--minibatch-size`
bring it down.

## Expectations

Pokemon is a long-horizon game and exploration is slow to learn. Comparable
work on Pokemon Red needed hundreds of millions of steps to earn a badge; at
this throughput that is days to weeks. The first signs of learning to look for
in `log.csv` are `tiles` and `maps` rising over the first few million steps.
