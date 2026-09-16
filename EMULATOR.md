# The emulator dependency

This repository contains no emulator source. It talks to `libgba_env` through
the C ABI declared in the emulator's `src/api/gba_env.h`, loaded with ctypes.

The emulator lives at <https://github.com/> — wherever you keep the `gba`
checkout. Build it once and this repository finds it.

## Finding it

The bindings look for the shared library in this order:

1. `GBA_ENV_LIB` — a full path to the library itself.
2. `GBA_EMULATOR_ROOT` — an emulator checkout; its `build/` is searched.
3. A checkout named `gba`, `gba-gpu` or `gba-emulator` beside this repository.
4. This package's own directory.
5. Wherever the dynamic loader looks, which covers `cmake --install`.

So two checkouts side by side need no configuration at all:

```
projects/
  gba/        <- the emulator, built
  gba-rl/     <- this
```

Otherwise:

```sh
export GBA_EMULATOR_ROOT=/path/to/gba
# or
export GBA_ENV_LIB=/path/to/gba/build/libgba_env.dylib
```

The same search finds the emulator checkout itself, which is where the test
ROMs used by `examples/smoke_test.py` are generated
(`python3 tools/make_bench_roms.py`, run there).

## The version contract

`GBA_ENV_ABI_VERSION` in the emulator's `src/api/gba_env.h` and `EXPECTED_ABI`
in `gba_env/_native.py` must agree. Loading a library that reports a different
version raises immediately, naming both numbers, rather than crashing or
returning wrong data. Bump them together whenever the header changes shape.

The emulator repository tests that interface itself with `build/api_test`, so a
break there is caught without this repository being present.
