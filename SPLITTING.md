# Moving this into its own repository

This directory is written to stand alone. Nothing in it includes emulator
source; it talks to `libgba_env` through the C ABI in `src/api/gba_env.h`, and
that interface is versioned so a mismatch is reported rather than crashing.

## Doing the split, keeping history

```sh
cd /path/to/gba
git subtree split --prefix=rl -b rl-only

mkdir ../gba-rl && cd ../gba-rl
git init
git pull ../gba rl-only
git branch -M main
```

The new repository then contains only what was under `rl/`, with the commits
that touched it. Delete the branch afterwards with `git branch -D rl-only` in
the emulator repository, and remove `rl/` there.

Without history, `cp -r rl ../gba-rl && cd ../gba-rl && git init` is enough.

## Finding the emulator afterwards

The bindings look for the shared library in this order:

1. `GBA_ENV_LIB` -- a full path to the library itself.
2. `GBA_EMULATOR_ROOT` -- an emulator checkout; its `build/` is searched.
3. A checkout named `gba`, `gba-gpu` or `gba-emulator` beside this repository.
4. This package's own directory.
5. Wherever the dynamic loader looks, which covers `cmake --install`.

So two checkouts side by side need no configuration:

```
projects/
  gba/          <- the emulator, built
  gba-rl/       <- this
```

Otherwise:

```sh
export GBA_EMULATOR_ROOT=/path/to/gba
# or
export GBA_ENV_LIB=/path/to/gba/build/libgba_env.dylib
```

## What to watch

`GBA_ENV_ABI_VERSION` in `src/api/gba_env.h` and `EXPECTED_ABI` in
`gba_env/_native.py` must agree. Loading a library that reports a different
version raises immediately with both numbers, rather than crashing or returning
wrong data. Bump both together whenever the header changes shape.

The emulator repository tests the ABI itself (`build/api_test`), so a break
there is caught without this repository present.
