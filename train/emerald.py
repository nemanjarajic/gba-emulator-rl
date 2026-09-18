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
DIALOG_TEXT = 0x02021FC4      # gStringVar4: the message being displayed, expanded
DIALOG_WORDS = 16             # enough of it to tell messages apart
DIALOG_OPEN = 0x03000F2C      # non-zero exactly while a text box is on screen
FLAGS_OFFSET = 0x1270         # SaveBlock1.flags, one bit per flag
FLAG_BYTES = 300
BADGE_FLAGS = range(0x867, 0x86F)  # FLAG_BADGE01_GET .. FLAG_BADGE08_GET
# Flags below this are temporary (cleared on every map load) and would reward
# walking in and out of a door.
FIRST_STORY_FLAG = 0x20

# Bits per instance in the visited-tile bitmap: 16384 bits is 2 KiB each, which
# holds a few thousand tiles before collisions start costing rewards.
TILE_BITS = 1 << 14

# Every reward term, in the order logs report them.
REWARD_TERMS = ("new_tile", "new_map", "new_try", "new_dialog", "story_flag", "level", "badge",
                "party_member", "species_owned", "heal", "repeat_dialog", "revisit", "stuck",
                "faint", "blackout")

# The policy's buttons. START and SELECT are left out: they open menus that
# cost many steps and lead nowhere for exploration.
ACTIONS = np.array(
    [Button.UP, Button.DOWN, Button.LEFT, Button.RIGHT, Button.A, Button.B],
    dtype=np.uint16,
)


@dataclass
class RewardWeights:
    # -- positive -----------------------------------------------------------
    # Novelty is per instance and lasts for the whole run: an instance is paid
    # the first time it stands somewhere and never again, in this episode or a
    # later one. Sharing the count between instances instead, with a
    # 1/sqrt(visits) decay, looked reasonable and killed exploration outright:
    # 4096 instances cover every nearby tile within seconds, so the reward
    # decayed to 0.007 an episode and walking stopped paying at all.
    new_tile: float = 0.05
    new_map: float = 3.0
    # Trying a direction from a tile for the first time, whether or not it
    # moves the player. Doors in Emerald's houses sit in the bottom wall row:
    # leaving means walking into what looks like a wall, which earns no tile
    # and used to be charged the revisit penalty, so the agent was being taught
    # away from the one move that opens the map.
    new_try: float = 0.01
    new_dialog: float = 0.5       # a page of text this instance has not read before
    dialog_cap: float = 3.0       # ...up to this much an episode: menus print
                                  # endless unread text, and with walking no
                                  # longer paying, agents stood at the PC
                                  # reading it instead of leaving the bedroom
    story_flag: float = 0.25      # per story flag, when the count reaches a new high
    level: float = 0.2            # per level, when the party's total reaches a new high
    badge: float = 100.0          # per badge
    party_member: float = 5.0     # per Pokemon, when the party reaches a new size
    species_owned: float = 2.0    # per species, when the Pokedex owned count reaches a new high
    heal: float = 1.0             # per whole party's worth of HP restored...
    heal_cap: float = 5.0         # ...up to this much per episode, so healing cannot be farmed
    # -- negative -----------------------------------------------------------
    repeat_dialog: float = -0.25  # a page this instance has read before
    revisit: float = -0.002       # a step spent on a tile visited before
    stuck: float = -0.01          # each step once no new tile has been found for stuck_after steps
    stuck_after: int = 200
    faint: float = -2.0           # per party Pokemon whose HP falls to zero
    blackout: float = -10.0       # the whole party has fainted


def _hash(words: np.ndarray) -> np.ndarray:
    """(W, N) words -> (N,) uint64 FNV-1a hash, one per instance."""
    h = np.full(words.shape[1], np.uint64(0xCBF29CE484222325), dtype=np.uint64)
    prime = np.uint64(0x100000001B3)
    with np.errstate(over="ignore"):  # hashing is meant to wrap
        for row in words:
            h = (h ^ row.astype(np.uint64)) * prime
    return h


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
        self.dialog_words = [DIALOG_OPEN] + [DIALOG_TEXT + 4 * i for i in range(DIALOG_WORDS)]
        # What each instance has ever seen, kept across episodes.
        #
        # Tiles live in a bitmap indexed by a hash of the tile, one per
        # instance: a set per instance would be hundreds of megabytes at this
        # scale, while this is 2 KiB each. A collision costs one unpaid tile,
        # which is a fair trade. Maps and messages are few enough for sets.
        self.tile_seen = np.zeros((self.n, TILE_BITS), dtype=bool)
        self.try_seen = np.zeros((self.n, TILE_BITS), dtype=bool)
        self.seen_maps_ever: list[set[int]] = [set() for _ in range(self.n)]
        self.seen_dialog: list[set[int]] = [set() for _ in range(self.n)]
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

        # The message on screen, identified by a hash of the expanded text
        # rather than a pointer, so messages built at run time (anything with
        # the player's name in it) still tell themselves apart.
        dialog = self.env.probe(self.dialog_words)
        dialog_hash = _hash(dialog[1:])
        dialog_open = dialog[0] != 0

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
                "dialog_open": dialog_open, "dialog_hash": dialog_hash,
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
        self.last_dialog = np.zeros(self.n, dtype=np.uint64)
        self.prev_tile = np.zeros(self.n, dtype=np.int64)
        self._note_novelty(ram)
        self.last_dialog = np.zeros(self.n, dtype=np.uint64)  # judged again after a reset
        self._note_dialog(ram)
        # Progress rewards pay for new highs only, so nothing can be farmed by
        # losing a level or clearing a flag and regaining it.
        self.best = {key: ram[key].copy() for key in ("story", "level_sum", "badges", "party", "owned")}
        self.since_new_tile = np.zeros(self.n, dtype=np.int32)
        self.healed = np.zeros(self.n, dtype=np.float64)
        self.dialog_paid = np.zeros(self.n, dtype=np.float64)
        self.episode_return = np.zeros(self.n, dtype=np.float64)
        # Reward term -> per-instance total this episode, so logs can show which
        # term a return came from.
        self.components = {key: np.zeros(self.n) for key in REWARD_TERMS}

    def _note_tries(self, ram: dict, action_idx: np.ndarray) -> np.ndarray:
        """Which instances just tried a direction they have never tried here.

        Keyed on where the player stood *before* the action, so a move that is
        blocked still counts as tried: that is the whole point.
        """
        prev = self.prev_tile
        h = ((prev.astype(np.uint64) * np.uint64(6) + action_idx.astype(np.uint64))
             * np.uint64(0x9E3779B97F4A7C15))
        idx = ((h >> np.uint64(40)) & np.uint64(TILE_BITS - 1)).astype(np.int64)
        rows = np.arange(self.n)
        first = ~self.try_seen[rows, idx]
        self.try_seen[rows, idx] = True
        return first

    def _note_novelty(self, ram: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Novelty of where each instance now stands.

        Returns which instances are somewhere they have never been, which are on
        a map they have never entered, and which are back on a tile they already
        stood on this episode -- what the revisit penalty charges for. The
        per-episode sets feed the logs.
        """
        m = (ram["group"].astype(np.int64) << 8) | ram["num"]
        tile = (m << 16) | ((ram["x"].astype(np.int64) & 0xFF) << 8) | (ram["y"].astype(np.int64) & 0xFF)
        # Mix the tile id before using its low bits as an index, so neighbouring
        # tiles do not land on neighbouring bits.
        h = (tile.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15))
        idx = ((h >> np.uint64(40)) & np.uint64(TILE_BITS - 1)).astype(np.int64)
        rows = np.arange(self.n)
        new_tile = ~self.tile_seen[rows, idx]
        self.tile_seen[rows, idx] = True

        self.prev_tile = tile
        new_map = np.zeros(self.n, dtype=bool)
        again = np.zeros(self.n, dtype=bool)
        for i in range(self.n):
            t, mi = int(tile[i]), int(m[i])
            again[i] = t in self.seen_tiles[i]
            self.seen_tiles[i].add(t)
            self.seen_maps[i].add(mi)
            if mi not in self.seen_maps_ever[i]:
                self.seen_maps_ever[i].add(mi)
                new_map[i] = True
        return new_tile, new_map, again

    def _note_dialog(self, ram: dict) -> tuple[np.ndarray, np.ndarray]:
        """A page of text just appeared: is it one never shown before?

        Judged only when the text changes, so a box left open for many steps
        counts once rather than every step.
        """
        new = np.zeros(self.n, dtype=bool)
        repeat = np.zeros(self.n, dtype=bool)
        shown = np.where(ram["dialog_open"], ram["dialog_hash"], np.uint64(0))
        for i in np.flatnonzero(ram["dialog_open"] & (shown != self.last_dialog)):
            h = int(shown[i])
            if h in self.seen_dialog[i]:
                repeat[i] = True
            else:
                self.seen_dialog[i].add(h)
                new[i] = True
        self.last_dialog = shown
        return new, repeat

    def observe(self) -> np.ndarray:
        """(N, H, W) uint8 -- a copy, since the environment's buffer is reused."""
        return self.env.observations.copy()

    def step(self, action_idx: np.ndarray):
        """Returns (obs, reward, done, info). Episodes end together."""
        # Where the player stood before the action, for the probing bonus.
        tried = self._note_tries(self.ram, np.asarray(action_idx))
        self.env.step(ACTIONS[action_idx], frames=self.frames)
        self.t += 1
        prev, ram = self.ram, self.read()
        self.ram = ram
        w = self.w
        new_tile, new_map, again = self._note_novelty(ram)
        new_dialog, repeat_dialog = self._note_dialog(ram)
        # Dialogue is capped per episode the way healing is: a menu prints as
        # much unread text as an agent cares to open.
        dialog_pay = np.minimum(w.new_dialog * new_dialog,
                                np.maximum(w.dialog_cap - self.dialog_paid, 0.0))
        self.dialog_paid += dialog_pay

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

        self.since_new_tile = np.where(again & ~tried, self.since_new_tile + 1, 0)

        terms = {
            "new_tile": w.new_tile * new_tile,
            "new_map": w.new_map * new_map,
            "new_try": w.new_try * tried,
            "new_dialog": dialog_pay,
            "story_flag": w.story_flag * gain("story"),
            "level": w.level * gain("level_sum"),
            "badge": w.badge * gain("badges"),
            "party_member": w.party_member * gain("party"),
            "species_owned": w.species_owned * gain("owned"),
            "heal": heal,
            "repeat_dialog": w.repeat_dialog * repeat_dialog,
            "revisit": w.revisit * (again & ~tried),
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
            "tiles_ever": float(self.tile_seen.sum(axis=1).mean()),
            "tries_ever": float(self.try_seen.sum(axis=1).mean()),
            "maps_ever": float(np.mean([len(m) for m in self.seen_maps_ever])),
            "max_maps_ever": int(max(len(m) for m in self.seen_maps_ever)),
            "dialog_ever": float(np.mean([len(d) for d in self.seen_dialog])),
            # Where the return came from, so a term that dominates is visible.
            **{f"r_{key}": float(np.mean(value)) for key, value in self.components.items()},
        }

    def novelty_state(self) -> dict:
        """What each instance has ever seen, for saving beside a checkpoint."""
        return {"tile_seen": np.packbits(self.tile_seen, axis=1),
                "try_seen": np.packbits(self.try_seen, axis=1),
                "seen_maps": [np.fromiter(m, dtype=np.int64) for m in self.seen_maps_ever],
                "seen_dialog": [np.fromiter(d, dtype=np.uint64) for d in self.seen_dialog]}

    def load_novelty_state(self, state: dict) -> None:
        """Restores it, so a resumed run does not pay for the same ground twice."""
        packed = state.get("tile_seen")
        if packed is not None and packed.shape[0] == self.n:
            self.tile_seen = np.unpackbits(packed, axis=1, count=TILE_BITS).astype(bool)
        tries = state.get("try_seen")
        if tries is not None and tries.shape[0] == self.n:
            self.try_seen = np.unpackbits(tries, axis=1, count=TILE_BITS).astype(bool)
        maps, dialog = state.get("seen_maps", []), state.get("seen_dialog", [])
        for i in range(self.n):
            self.seen_maps_ever[i] = {int(v) for v in maps[i]} if i < len(maps) else set()
            self.seen_dialog[i] = {int(v) for v in dialog[i]} if i < len(dialog) else set()

    def close(self) -> None:
        self.env.close()
