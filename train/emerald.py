"""Pokemon Emerald as an exploration task: game RAM, rewards and episodes.

Everything the reward needs is read from the game's own memory through
GbaVecEnv.probe(); the pixels are only for the policy.

Addresses are for Emerald (USA/Europe), from the pokeemerald decompilation.
Position, map and the save-block pointer were checked against the running game
(walking moves the coordinates; leaving the truck lands on map 0.9, Littleroot
Town). The flag base was checked against an instruction trace of the game's
own flag test. Party, Pokedex and badge offsets follow the decompilation but
could not be checked before the player has a Pokemon, so every value read from
them is range-checked: a wrong address reads as nothing rather than paying out.

Battle outcome is deliberately not rewarded directly: its address could not be
verified. Levels, fainting and blacking out carry the same signal from party
data, which can.
"""

from dataclasses import dataclass

import numpy as np

from gba_env import Button, GbaVecEnv

SAVEBLOCK1_PTR = 0x03005D8C   # gSaveBlock1Ptr; Emerald moves the block at random
SAVEBLOCK2_PTR = 0x03005D90   # gSaveBlock2Ptr; likewise
PARTY_COUNT = 0x020244E9      # gPlayerPartyCount (u8)
PARTY = 0x020244EC            # gPlayerParty, 100 bytes per Pokemon
PARTY_SLOT = 100
LEVEL_OFFSET = 0x54           # struct Pokemon.level (u8)
HP_OFFSET = 0x56              # struct Pokemon.hp (u16), then maxHP (u16) at 0x58
POKEDEX_OWNED_OFFSET = 0x28   # SaveBlock2.pokedex.owned, one bit per species
POKEDEX_BYTES = 52
FLAGS_OFFSET = 0x1270         # SaveBlock1.flags, one bit per flag
FLAG_BYTES = 300
BADGE_FLAGS = range(0x867, 0x86F)  # FLAG_BADGE01_GET .. FLAG_BADGE08_GET
# Flags below this are temporary (cleared on every map load) and would reward
# walking in and out of a door.
FIRST_STORY_FLAG = 0x20

# Every reward term, in the order logs report them.
REWARD_TERMS = ("new_tile", "new_map", "story_flag", "level", "badge", "party_member", "species_owned",
                "heal", "revisit", "stuck", "faint", "blackout")

# The policy's buttons. START and SELECT are left out: they open menus that
# cost many steps and lead nowhere for exploration.
ACTIONS = np.array(
    [Button.UP, Button.DOWN, Button.LEFT, Button.RIGHT, Button.A, Button.B],
    dtype=np.uint16,
)


@dataclass
class RewardWeights:
    # -- positive -----------------------------------------------------------
    new_tile: float = 0.02        # a map tile this episode has not stood on
    new_map: float = 1.0          # a map this episode has not entered
    story_flag: float = 0.5       # per story flag, when the count reaches a new high
    level: float = 0.2            # per level, when the party's total reaches a new high
    badge: float = 100.0          # per badge
    party_member: float = 5.0     # per Pokemon, when the party reaches a new size
    species_owned: float = 2.0    # per species, when the Pokedex owned count reaches a new high
    heal: float = 1.0             # per whole party's worth of HP restored...
    heal_cap: float = 5.0         # ...up to this much per episode, so healing cannot be farmed
    # -- negative -----------------------------------------------------------
    revisit: float = -0.002       # a step spent on a tile already visited this episode
    stuck: float = -0.01          # each step once no new tile has been found for stuck_after steps
    stuck_after: int = 200
    faint: float = -2.0           # per party Pokemon whose HP falls to zero
    blackout: float = -10.0       # the whole party has fainted


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
        self.dex_words = [POKEDEX_OWNED_OFFSET + 4 * i for i in range(POKEDEX_BYTES // 4)]
        self.party_words = [PARTY + i * PARTY_SLOT + o for i in range(6) for o in (LEVEL_OFFSET, HP_OFFSET + 2)]
        self.ram: dict = {}
        self._start_episode()

    def map_ids(self) -> np.ndarray:
        """(N,) int64 map identifier, group * 256 + number, for the policy."""
        return (self.ram["group"].astype(np.int64) << 8) | self.ram["num"]

    # -- RAM --------------------------------------------------------------

    def _gather(self, pointer: int, offsets: list[int]) -> np.ndarray:
        """Reads words at *pointer + offset for every instance: (len, N).

        The save blocks' addresses can differ between instances, and probe()
        reads one address everywhere, so instances are grouped by pointer.
        Groups are few -- usually one.
        """
        ptr = self.env.probe([pointer])[0]
        out = np.zeros((len(offsets), self.n), dtype=np.uint32)
        for p in np.unique(ptr):
            idx = np.flatnonzero(ptr == p)
            out[:, idx] = self.env.probe([int(p) + o for o in offsets])[:, idx]
        return out

    def read(self) -> dict:
        head = self._gather(SAVEBLOCK1_PTR, [0, 4])
        x = (head[0] & 0xFFFF).astype(np.int32)
        y = ((head[0] >> 16) & 0xFFFF).astype(np.int32)
        group = (head[1] & 0xFF).astype(np.int32)
        num = ((head[1] >> 8) & 0xFF).astype(np.int32)

        flag_bits = _bits(self._gather(SAVEBLOCK1_PTR, self.flag_words))
        story = flag_bits[FIRST_STORY_FLAG:].sum(axis=0).astype(np.int32)
        badges = flag_bits[list(BADGE_FLAGS)].sum(axis=0).astype(np.int32)

        owned = _bits(self._gather(SAVEBLOCK2_PTR, self.dex_words)).sum(axis=0).astype(np.int32)
        owned = np.where(owned <= 386, owned, 0)  # more than every species: not a Pokedex

        count = np.minimum(self.env.probe([PARTY_COUNT])[0] & 0xFF, 6).astype(np.int32)
        words = self.env.probe(self.party_words).reshape(6, 2, self.n)
        level = (words[:, 0] & 0xFF).astype(np.int32)
        hp = ((words[:, 0] >> 16) & 0xFFFF).astype(np.int32)
        max_hp = (words[:, 1] & 0xFFFF).astype(np.int32)
        # A slot counts only if it is inside the party and looks like a Pokemon.
        valid = ((np.arange(6)[:, None] < count[None, :]) & (level >= 1) & (level <= 100)
                 & (max_hp >= 1) & (max_hp <= 999) & (hp <= max_hp))
        hp = np.where(valid, hp, 0)
        max_hp = np.where(valid, max_hp, 0)
        return {"x": x, "y": y, "group": group, "num": num, "story": story, "badges": badges,
                "owned": owned, "party": valid.sum(axis=0).astype(np.int32),
                "level_sum": np.where(valid, level, 0).sum(axis=0), "valid": valid,
                "hp": hp, "max_hp": max_hp}

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
        self.best = {key: ram[key].copy() for key in ("story", "level_sum", "badges", "party", "owned")}
        self.since_new_tile = np.zeros(self.n, dtype=np.int32)
        self.healed = np.zeros(self.n, dtype=np.float64)
        self.episode_return = np.zeros(self.n, dtype=np.float64)
        # Reward term -> per-instance total this episode, so logs can show which
        # term a return came from.
        self.components = {key: np.zeros(self.n) for key in REWARD_TERMS}

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
        prev, ram = self.ram, self.read()
        self.ram = ram
        w = self.w
        new_tile, new_map = self._note_novelty(ram)

        def gain(key):
            g = np.maximum(ram[key] - self.best[key], 0)
            self.best[key] = np.maximum(self.best[key], ram[key])
            return g

        # Fainting: a slot that held a conscious Pokemon now holds a fainted one.
        fainted = (prev["valid"] & (prev["hp"] > 0) & ram["valid"] & (ram["hp"] == 0)).sum(axis=0)
        all_down = (ram["party"] > 0) & (ram["hp"].sum(axis=0) == 0)
        was_down = (prev["party"] > 0) & (prev["hp"].sum(axis=0) == 0)
        blackout = all_down & ~was_down

        # Healing: the party's share of its HP rises with the party itself
        # unchanged, so neither a level-up (max HP grows too) nor a new member
        # counts. The recovery after a blackout does not count either.
        same_party = (ram["party"] == prev["party"]) & (ram["max_hp"].sum(axis=0) == prev["max_hp"].sum(axis=0))
        max_total = np.maximum(ram["max_hp"].sum(axis=0), 1)
        restored = (ram["hp"].sum(axis=0) - prev["hp"].sum(axis=0)) / max_total
        restored = np.where(same_party & ~was_down & (ram["party"] > 0), np.maximum(restored, 0.0), 0.0)
        heal = np.minimum(w.heal * restored, np.maximum(w.heal_cap - self.healed, 0.0))
        self.healed += heal

        self.since_new_tile = np.where(new_tile, 0, self.since_new_tile + 1)

        terms = {
            "new_tile": w.new_tile * new_tile,
            "new_map": w.new_map * new_map,
            "story_flag": w.story_flag * gain("story"),
            "level": w.level * gain("level_sum"),
            "badge": w.badge * gain("badges"),
            "party_member": w.party_member * gain("party"),
            "species_owned": w.species_owned * gain("owned"),
            "heal": heal,
            "revisit": w.revisit * ~new_tile,
            "stuck": w.stuck * (self.since_new_tile > w.stuck_after),
            "faint": w.faint * fainted,
            "blackout": w.blackout * blackout,
        }
        reward = sum(np.asarray(v, dtype=np.float64) for v in terms.values())
        assert tuple(terms) == REWARD_TERMS
        for key, value in terms.items():
            self.components[key] = self.components[key] + value
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
            "owned": float(ram["owned"].mean()),
            "level_sum": float(ram["level_sum"].mean()),
            "badges": float(ram["badges"].mean()),
            "story_flags": float(ram["story"].mean()),
            # Where the return came from, so a term that dominates is visible.
            **{f"r_{key}": float(np.mean(value)) for key, value in self.components.items()},
        }

    def close(self) -> None:
        self.env.close()
