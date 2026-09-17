"""Pokemon Emerald as an exploration task: game RAM, rewards and episodes.

Everything the reward needs is read from the game's own memory through
GbaVecEnv.probe(); the pixels are only for the policy.

Addresses are for Emerald (USA/Europe), from the pokeemerald decompilation.
Position, map and the save-block pointer were checked against the running game
(walking moves the coordinates; leaving the truck lands on map 0.9, Littleroot
Town). The flag base was checked against an instruction trace of the game's
own flag test. Party and badge offsets follow the decompilation but could not
be checked before the player has a Pokemon.
"""

from dataclasses import dataclass

import numpy as np

from gba_env import Button, GbaVecEnv

SAVEBLOCK1_PTR = 0x03005D8C   # gSaveBlock1Ptr; Emerald moves the block at random
PARTY_COUNT = 0x020244E9      # gPlayerPartyCount (u8)
PARTY = 0x020244EC            # gPlayerParty, 100 bytes per Pokemon
PARTY_SLOT = 100
LEVEL_OFFSET = 0x54           # struct Pokemon.level (u8)
FLAGS_OFFSET = 0x1270         # SaveBlock1.flags, one bit per flag
FLAG_BYTES = 300
BADGE_FLAGS = range(0x867, 0x86F)  # FLAG_BADGE01_GET .. FLAG_BADGE08_GET
# Flags below this are temporary (cleared on every map load) and would reward
# walking in and out of a door.
FIRST_STORY_FLAG = 0x20

# The policy's buttons. START and SELECT are left out: they open menus that
# cost many steps and lead nowhere for exploration.
ACTIONS = np.array(
    [Button.UP, Button.DOWN, Button.LEFT, Button.RIGHT, Button.A, Button.B],
    dtype=np.uint16,
)


@dataclass
class RewardWeights:
    new_tile: float = 0.02      # a map tile this episode has not stood on
    new_map: float = 1.0        # a map this episode has not entered
    story_flag: float = 0.5     # the number of story flags set rises
    level: float = 1.0          # the party's total level rises
    badge: float = 20.0


def _bits(data: np.ndarray) -> np.ndarray:
    """(W, N) little-endian uint32 words -> (W*32, N) bits, lowest bit first."""
    as_bytes = np.ascontiguousarray(data.T, dtype="<u4").view(np.uint8).reshape(data.shape[1], -1)
    return np.unpackbits(as_bytes, axis=1, bitorder="little").T


class EmeraldExplore:
    """Wraps GbaVecEnv with Emerald's RAM, per-episode novelty and rewards.

    All instances start every episode from the same reset point and run for a
    fixed number of steps; exploration comes from the policy, not the start.
    """

    def __init__(self, rom: str, state: str, num_envs: int, frames_per_action: int = 16,
                 episode_steps: int = 2048, weights: RewardWeights = RewardWeights()):
        self.env = GbaVecEnv(rom, num_instances=num_envs)
        self.env.load_reset_point(state)
        self.n = self.env.num_instances
        self.frames = frames_per_action
        self.episode_steps = episode_steps
        self.w = weights
        self.flag_words = [FLAGS_OFFSET + 4 * i for i in range(FLAG_BYTES // 4)]
        self.ram: dict = {}
        self._start_episode()

    def map_ids(self) -> np.ndarray:
        """(N,) int64 map identifier, group * 256 + number, for the policy."""
        return (self.ram["group"].astype(np.int64) << 8) | self.ram["num"]

    # -- RAM --------------------------------------------------------------

    def _gather(self, offsets: list[int]) -> np.ndarray:
        """Reads words at saveblock1 + offset for every instance: (len, N).

        The save block's address can differ between instances, and probe()
        reads one address everywhere, so instances are grouped by pointer.
        Groups are few -- usually one.
        """
        ptr = self.env.probe([SAVEBLOCK1_PTR])[0]
        out = np.zeros((len(offsets), self.n), dtype=np.uint32)
        for p in np.unique(ptr):
            idx = np.flatnonzero(ptr == p)
            out[:, idx] = self.env.probe([int(p) + o for o in offsets])[:, idx]
        return out

    def read(self) -> dict:
        head = self._gather([0, 4])
        x = (head[0] & 0xFFFF).astype(np.int32)
        y = ((head[0] >> 16) & 0xFFFF).astype(np.int32)
        group = (head[1] & 0xFF).astype(np.int32)
        num = ((head[1] >> 8) & 0xFF).astype(np.int32)

        flag_bits = _bits(self._gather(self.flag_words))
        story = flag_bits[FIRST_STORY_FLAG:].sum(axis=0).astype(np.int32)
        badges = flag_bits[list(BADGE_FLAGS)].sum(axis=0).astype(np.int32)

        count = np.minimum(self.env.probe([PARTY_COUNT])[0] & 0xFF, 6).astype(np.int32)
        level_words = self.env.probe([PARTY + i * PARTY_SLOT + LEVEL_OFFSET for i in range(6)])
        levels = (level_words & 0xFF).astype(np.int32)
        levels = np.where(np.arange(6)[:, None] < count[None, :], levels, 0)
        return {"x": x, "y": y, "group": group, "num": num, "story": story,
                "badges": badges, "level_sum": levels.sum(axis=0), "party": count}

    # -- episodes ---------------------------------------------------------

    def _start_episode(self) -> None:
        self.env.reset()
        self.env.step(np.zeros(self.n, dtype=np.uint16), frames=1)
        self.t = 0
        ram = self.ram = self.read()
        self.seen_tiles = [set() for _ in range(self.n)]
        self.seen_maps = [set() for _ in range(self.n)]
        self._note_novelty(ram)
        # Progress rewards pay for new highs only, so nothing can be farmed by
        # losing a level or clearing a flag and regaining it.
        self.best_story = ram["story"].copy()
        self.best_level = ram["level_sum"].copy()
        self.best_badges = ram["badges"].copy()
        self.episode_return = np.zeros(self.n, dtype=np.float64)

    def _note_novelty(self, ram: dict) -> tuple[np.ndarray, np.ndarray]:
        new_tile = np.zeros(self.n, dtype=bool)
        new_map = np.zeros(self.n, dtype=bool)
        for i in range(self.n):
            m = (int(ram["group"][i]) << 8) | int(ram["num"][i])
            tile = (m << 16) | ((int(ram["x"][i]) & 0xFF) << 8) | (int(ram["y"][i]) & 0xFF)
            if tile not in self.seen_tiles[i]:
                self.seen_tiles[i].add(tile)
                new_tile[i] = True
            if m not in self.seen_maps[i]:
                self.seen_maps[i].add(m)
                new_map[i] = True
        return new_tile, new_map

    def observe(self) -> np.ndarray:
        """(N, H, W) uint8 -- a copy, since the environment's buffer is reused."""
        return self.env.observations.copy()

    def step(self, action_idx: np.ndarray):
        """Returns (obs, reward, done, info). Episodes end together."""
        self.env.step(ACTIONS[action_idx], frames=self.frames)
        self.t += 1
        ram = self.ram = self.read()
        new_tile, new_map = self._note_novelty(ram)

        story_gain = np.maximum(ram["story"] - self.best_story, 0)
        level_gain = np.maximum(ram["level_sum"] - self.best_level, 0)
        badge_gain = np.maximum(ram["badges"] - self.best_badges, 0)
        self.best_story = np.maximum(self.best_story, ram["story"])
        self.best_level = np.maximum(self.best_level, ram["level_sum"])
        self.best_badges = np.maximum(self.best_badges, ram["badges"])

        reward = (self.w.new_tile * new_tile + self.w.new_map * new_map
                  + self.w.story_flag * story_gain + self.w.level * level_gain
                  + self.w.badge * badge_gain)
        self.episode_return += reward

        info = {}
        done = self.t >= self.episode_steps
        if done:
            info = self.stats()
            self._start_episode()
        return self.observe(), reward.astype(np.float32), done, info

    def stats(self) -> dict:
        """The current episode so far, averaged over instances."""
        ram = self.ram
        return {
            "episode_step": self.t,
            "return": float(self.episode_return.mean()),
            "tiles": float(np.mean([len(s) for s in self.seen_tiles])),
            "maps": float(np.mean([len(s) for s in self.seen_maps])),
            "max_maps": int(max(len(s) for s in self.seen_maps)),
            "party": float(ram["party"].mean()),
            "level_sum": float(ram["level_sum"].mean()),
            "badges": float(ram["badges"].mean()),
            "story_flags": float(ram["story"].mean()),
        }

    def close(self) -> None:
        self.env.close()
