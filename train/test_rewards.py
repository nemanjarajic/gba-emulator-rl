"""Checks each reward term against scripted RAM readings, without the emulator.

    python -m train.test_rewards
"""
import sys
import types

import numpy as np

if "gba_env" not in sys.modules:
    # The rewards only need Button; stubbing the package avoids loading the
    # emulator library, so this runs anywhere in a second.
    stub = types.ModuleType("gba_env")
    stub.Button = type("Button", (), dict(NONE=0, A=1, B=2, SELECT=4, START=8, RIGHT=16, LEFT=32,
                                          UP=64, DOWN=128, R=256, L=512))
    stub.GbaVecEnv = object
    sys.modules["gba_env"] = stub

from train.emerald import REWARD_TERMS, EmeraldExplore, RewardWeights  # noqa: E402


def ram(x=0, y=0, party=None, story=10, badges=0, owned=0, dialog=None):
    """One instance. party: list of (level, hp, max_hp); dialog: text id or None."""
    party = party or []
    valid = np.zeros((6, 1), bool)
    hp = np.zeros((6, 1), int)
    max_hp = np.zeros((6, 1), int)
    for i, (lv, h, m) in enumerate(party):
        valid[i], hp[i], max_hp[i] = True, h, m
    a = lambda v: np.array([v])  # noqa: E731
    return {"x": a(x), "y": a(y), "group": a(0), "num": a(9), "story": a(story), "badges": a(badges),
            "dialog_open": np.array([dialog is not None]),
            "dialog_hash": np.array([dialog or 0], dtype=np.uint64),
            "owned": a(owned), "party": a(len(party)), "level_sum": a(sum(p[0] for p in party)),
            "valid": valid, "hp": hp, "max_hp": max_hp}


class Fake:
    num_instances = 1
    def reset(self): pass
    def step(self, *a, **k): pass
    observations = np.zeros((1, 80, 120), np.uint8)


def make(first, weights=RewardWeights(), shared=None):
    """A one-instance environment. `shared` reuses another's memory of what it
    has seen, which is how novelty now works: across instances and episodes."""
    e = EmeraldExplore.__new__(EmeraldExplore)
    e.env, e.n, e.frames, e.episode_steps, e.w = Fake(), 1, 16, 10**9, weights
    e.tile_visits, e.map_visits = (shared.tile_visits, shared.map_visits) if shared else ({}, {})
    e.seen_dialog = [set()]
    e._script = [first]
    e.read = lambda: e._script.pop(0)
    e.ram = {}
    e._start_episode()
    return e


def run(e, reading):
    e._script.append(reading)
    _, r, _, _ = e.step(np.array([0]))
    return float(r[0]), {k: float(v[0]) for k, v in e.components.items()}


failures = 0
def check(what, got, want):
    global failures
    ok = abs(got - want) < 1e-5  # rewards are float32
    failures += not ok
    print(f"  {'ok  ' if ok else 'FAIL'} {what}: got {got:+.4f} want {want:+.4f}")


w = RewardWeights()
starter = [(5, 20, 20)]

e = make(ram(0, 0))
r, _ = run(e, ram(1, 0)); check("a tile nobody has visited", r, w.new_tile)
r, _ = run(e, ram(1, 0)); check("standing still is a revisit", r, w.revisit)
r, _ = run(e, ram(1, 0, story=11)); check("story flag (plus revisit)", r, w.story_flag + w.revisit)
r, _ = run(e, ram(1, 0, story=10)); check("losing a flag pays nothing more", r, w.revisit)
r, _ = run(e, ram(1, 0, story=11)); check("regaining it pays nothing", r, w.revisit)

e = make(ram(0, 0))
r, _ = run(e, ram(0, 0, party=starter, owned=1))
check("starter: party member + species + level", r, w.party_member + w.species_owned + 5 * w.level + w.revisit)
r, _ = run(e, ram(0, 0, party=[(5, 8, 20)])); check("taking damage costs nothing", r, w.revisit)
r, _ = run(e, ram(0, 0, party=[(5, 20, 20)])); check("healing 60%", r, 0.6 * w.heal + w.revisit)
r, _ = run(e, ram(0, 0, party=[(6, 23, 23)])); check("level-up is not healing", r, w.level + w.revisit)
r, _ = run(e, ram(0, 0, party=[(6, 0, 23)])); check("faint = blackout with one Pokemon", r, w.faint + w.blackout + w.revisit)
r, _ = run(e, ram(0, 0, party=[(6, 0, 23)])); check("still down: no second penalty", r, w.revisit)
r, _ = run(e, ram(0, 0, party=[(6, 23, 23)])); check("recovery after blackout is not healing", r, w.revisit)

e = make(ram(0, 0, party=[(5, 20, 20), (5, 20, 20)]))
r, _ = run(e, ram(0, 0, party=[(5, 0, 20), (5, 20, 20)])); check("one of two faints: no blackout", r, w.faint + w.revisit)

e = make(ram(0, 0, party=starter))
for _ in range(10):
    run(e, ram(0, 0, party=[(5, 10, 20)]))
    run(e, ram(0, 0, party=[(5, 20, 20)]))
_, c = run(e, ram(0, 0, party=[(5, 20, 20)]))
check("heal capped per episode", c["heal"], w.heal_cap)

e = make(ram(0, 0))
for _ in range(w.stuck_after):
    run(e, ram(0, 0))
r, _ = run(e, ram(0, 0)); check("stuck after stuck_after steps", r, w.revisit + w.stuck)
r, _ = run(e, ram(5, 5)); check("a new tile clears stuck", r, w.new_tile)
r, _ = run(e, ram(6, 5)); check("still paid while standing on fresh ground", r, w.new_tile)

e = make(ram(0, 0))
r, _ = run(e, ram(1, 0, dialog=111)); check("a message this instance has not read", r, w.new_tile + w.new_dialog)
r, _ = run(e, ram(1, 0, dialog=111)); check("the same page still open: not charged again", r, w.revisit)
r, _ = run(e, ram(1, 0, dialog=222)); check("the next page of it", r, w.revisit + w.new_dialog)
r, _ = run(e, ram(1, 0)); check("box closed", r, w.revisit)
r, _ = run(e, ram(1, 0, dialog=111)); check("reading it again is charged", r, w.revisit + w.repeat_dialog)
other = make(ram(0, 0), shared=e)
r, _ = run(other, ram(9, 9, dialog=111))
check("another instance still gets paid for it", r, w.new_tile + w.new_dialog)

e2 = make(ram(0, 0), shared=e)
r, _ = run(e2, ram(1, 0))
check("a tile two instances have now visited pays w/sqrt(2)", r, w.new_tile / 2 ** 0.5)
e3 = make(ram(0, 0), shared=e2)
r, _ = run(e3, ram(1, 0))
check("a third visit pays w/sqrt(3)", r, w.new_tile / 3 ** 0.5)

e = make(ram(0, 0))
r, _ = run(e, ram(0, 0, badges=1)); check("badge", r, w.badge + w.revisit)
check("components cover every term", float(set(e.components) == set(REWARD_TERMS)), 1.0)

print("REWARD TESTS", "FAIL" if failures else "PASS", f"({failures} failures)")
sys.exit(1 if failures else 0)
