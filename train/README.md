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

**Train from `emerald_outside.state`**, built from `emerald_clockset.state`,
which is built from `emerald_start.state`. Each one exists because training
could not pass the gate in front of it.

**`emerald_outside.state`** starts in Littleroot Town with the map open. Three
gates sit between the bedroom and the front door and 16M steps never passed
any of them: the staircase is a single tile entered from its side; coming down
it triggers a scene where Mom calls the player to the television, locking
movement until about forty pages of dialogue are dismissed; and the door is in
the bottom wall row, reached by walking into what looks like wall.

**`emerald_clockset.state`**, built the same way with
`--from train/states/emerald_start.state`. Emerald keeps Littleroot's exits shut
until the player sets the wall clock upstairs, and the clock's "Is this the
correct time?" prompt starts on NO: pressing A there returns to the clock, so a
policy that mashes A loops forever and needs UP-then-A to pass. A first run from
the truck confirmed it, plateauing at exactly 4 maps -- the truck, the town and
both floors of the house -- with about 20 story flags and nothing further, for
2.9M steps. `emerald_clockset.state` starts where the game first leaves the
player alone, with the town open and Route 101 and the starter reachable.

## 2. Train

```sh
python -m train.ppo --rom <emerald.gba> --state train/states/emerald_start.state --run runs/emerald
```

It writes `runs/emerald/log.csv` after every update (with the current episode's
progress so far), `episodes.csv` when an episode ends, and `latest.pt` every ten;
`--resume` continues a run. Every option is listed by `--help`.

To watch the instances play, in a second terminal while training runs:

```sh
python -m train.watch --run runs/emerald            # --scale 0.5 for a smaller window
```

The trainer writes 36 instances' screens, spread across the pool, to
`runs/emerald/grid.png` after every step (`--grid` sets how many, 0 turns it
off; it costs about 40 ms against a step of several seconds). The window shows
that grid with the latest progress line, and only reads files, so it can be
opened and closed at any time.

## What the agent sees, does and is paid for

- **Observation:** the last 3 screens at 120x80 grayscale, plus the current map
  number from RAM (the screen cannot tell many maps apart).
- **Actions:** UP, DOWN, LEFT, RIGHT, A, B, each held for 16 frames, which walks
  one tile. START and SELECT are left out; menus cost many steps and lead
  nowhere for exploration.
- **Reward**, all read from game RAM (`train/emerald.py`, weights in `RewardWeights`):

  | event | reward |
  |---|---|
  | standing on a tile this instance has never stood on | +0.05 |
  | entering a map this instance has never entered | +3 |
  | trying a direction from a tile this instance has never tried there | +0.01 |
  | a page of text this instance has not read | +0.5, at most +3 an episode |
  | the count of story flags set reaches a new high | +0.25 each |
  | the party's total level reaches a new high | +0.2 per level |
  | a badge | +100 |
  | the party reaches a new size (the starter, a catch) | +5 per Pokemon |
  | the Pokedex owned count reaches a new high | +2 per species |
  | party HP restored, the party otherwise unchanged | +1 per whole party's worth, at most +5 an episode |
  | a page of text this instance has read before | -0.25 |
  | a step on a tile already visited this episode, unless it was a first attempt in that direction | -0.002 |
  | each step once no new tile has been found for 200 steps | -0.01 |
  | a party Pokemon faints | -2 |
  | the whole party faints | -10 |

  **Novelty is per instance and lasts the whole run.** Two designs failed first,
  both instructively. Novelty that reset each episode paid for walking the same
  rooms forever: two runs plateaued within a room of where they started, 2.9M
  and 3.6M steps. Sharing one count between instances with a 1/sqrt(visits)
  decay then killed exploration outright -- 4096 instances cover the nearby
  tiles within seconds, so walking paid 0.007 an episode, dialogue was the only
  live reward left, and agents stood at the bedroom PC reading menus for 7M
  steps while entropy collapsed to 0.015. Per instance, the signal is dense for
  everyone and the starting room stops paying after one visit; the dialogue cap
  keeps menus from out-earning walking.

  **Walking into walls has to pay.** The doors out of Emerald's houses sit in
  the bottom wall row, so leaving means walking into what looks like solid
  wall: no tile is gained, and the revisit penalty used to be charged for it,
  which taught agents away from the one move that opens the map. A first
  attempt in a given direction from a given tile now pays a little and is
  exempt from the revisit penalty and the stuck timer, so probing edges is
  worth doing once and cannot be farmed.

  Tiles live in a 2 KiB bitmap per instance rather than a set, which would run
  to hundreds of megabytes at this scale; a hash collision costs one unpaid
  tile.

  Dialogue is judged only when the text changes, so a box left open is not
  charged every step. A message is identified by a hash of the expanded text at
  `gStringVar4`, which distinguishes messages built at run time (anything
  containing the player's name).

  Progress rewards pay only for new highs, so nothing can be farmed by losing
  and regaining it; healing is capped for the same reason, and a level-up or the
  recovery after a blackout does not count as healing.

  Battle wins are not rewarded directly: the battle-outcome address could not be
  verified, and levels, fainting and blackouts carry the signal from party data.
  Every party and Pokedex value is range-checked, so a wrong address reads as
  nothing rather than paying out. `episodes.csv` and `log.csv` break the return
  down by term (`r_new_tile`, `r_repeat_dialog`, ...), so a term that dominates
  shows. `python -m train.test_rewards` checks every term against scripted RAM.
- **Episodes:** 512 steps, all instances together.
- **The start moves forward.** At the end of each episode the instance that has
  seen the most maps becomes the start state for the next one, saved as
  `anchor.state` in the run directory so it survives a restart. A game like this
  is a chain of gates, and without an anchor an agent that passes one only
  occasionally has to pass it again every episode. Instances inside a text box
  are skipped, and the anchor only ever moves forward. `--no-anchor` turns it
  off.

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
| HP, max HP | party slot + `0x56`, + `0x58` | not yet |
| save block 2 | pointer at `0x03005D90` | not yet |
| Pokedex owned | save block 2 + `0x28`, 52 bytes | not yet |
| a text box is open | `0x03000F2C` | yes: 1 in seven dialogue snapshots, 0 in three without |
| the message on screen | `gStringVar4` at `0x02021FC4` | yes: found "clock" there in the game's text encoding |

## Throughput on an RTX 5060 Ti (8 GB)

About 340 agent steps/s at 4096 instances with PPO updates included, roughly
1.2M steps an hour. The emulator's 16 frames per step take most of it.

GPU memory is the limit. The emulator holds about 2.4 GB at 4096 instances,
PyTorch peaks near 1.9 GB, and the desktop takes the rest. If the two exceed
the card, Windows pages PyTorch's memory rather than failing and training slows
by two orders of magnitude, so watch `torch ... GiB` in the log line. Fewer
instances (`--num-envs`) or a smaller `--infer-chunk` and `--minibatch-size`
bring it down.

### Sharing the GPU with the rest of the machine

The emulator would otherwise hold the GPU continuously, which makes the desktop
stutter however few instances run: the GPU reads 99% busy with four. So the
trainer defaults to `--gpu-duty 0.5`, which sleeps after every emulator dispatch
for as long as the dispatch took, leaving half the GPU's time to everything
else. That halves throughput, to about 150 agent steps/s. Use `--gpu-duty 1` when
the machine is otherwise idle.

Lowering `--num-envs` does not help here: the load stays near 100% and only
throughput falls. It does lower GPU memory.

## Expectations

Pokemon is a long-horizon game and exploration is slow to learn. Comparable
work on Pokemon Red needed hundreds of millions of steps to earn a badge; at
this throughput that is days to weeks. The first signs of learning to look for
in `log.csv` are `tiles` and `maps` rising over the first few million steps.
