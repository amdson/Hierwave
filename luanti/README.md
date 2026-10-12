# Hierwave in Luanti

Walk around an infinite tile world (`castlegen/infinite.py`, C port
`castlegen/csrc/hwsampler.c`) with every tile built from an 8 x 8 x 24 block
template.  The contract (coordinates, palette, `.hwt` / `.hwm` files, C API)
is `notes/luanti_plan.md`.

## One-time setup

```sh
uv run python luanti/tools/make_textures.py      # textures + hierwave_nodes/palette.lua (already committed)
luanti/build.sh                                  # luanti/build/libhwdummy.dylib (FFI smoke test)
luanti/setup_app.sh                              # luanti/build/luanti.app (see "FFI" below)
```
The sampler library `cache/libhwsampler.dylib`, the model files
`games/hierwave/mods/hierwave_gen/data/{biomes,city}.hwm` and the templates
`data/{biomes,city}.hwt` are generated (gitignored).  From the repo root:
```sh
PYTHONPATH=. uv run python -m castlegen.export_models   # data/*.hwm
PYTHONPATH=. uv run python -m castlegen.prefabs         # data/*.hwt
PYTHONPATH=. uv run python -c "import castlegen.hwsampler as h; h.lib()"   # cache/libhwsampler.dylib
```
Delete `luanti/worlds/<set>-<seed>` after changing any of them.

## Launch

```sh
luanti/run.sh client                 # window, straight into luanti/worlds/biomes-1
HW_SET=city HW_SEED=7 luanti/run.sh client
luanti/run.sh server                 # dedicated server on UDP 30000
luanti/run.sh test 1                 # headless check, see below
```
`run.sh` runs `luanti/build/luanti.app` when it exists, else
`/Applications/luanti.app`, with everything repo-local:
`LUANTI_USER_PATH=luanti/user`, `LUANTI_GAME_PATH=luanti/games`, world
`luanti/worlds/<set>-<seed>`, config `luanti/luanti.conf` (copied to
`luanti/build/<mode>.conf` with the set and seed appended).  Nothing under
`~/Library` is read or written.  The equivalent raw client command:

```sh
LUANTI_USER_PATH=$PWD/luanti/user LUANTI_GAME_PATH=$PWD/luanti/games \
  luanti/build/luanti.app/Contents/MacOS/luanti --go --gameid hierwave \
  --world $PWD/luanti/worlds/biomes-1 --name viewer --config luanti/build/client.conf
```
You spawn at (4, 0.5, -4) with fly, fast, noclip (K fly, J fast, H noclip).
A world remembers what it generated: a new set or seed needs a new world
dir (run.sh names worlds by set and seed); delete `luanti/worlds/<name>`
after the templates or sampler change.

## Settings (`luanti.conf`, `hierwave_gen/settingtypes.txt`)

| setting | default | meaning |
|---|---|---|
| `hierwave_set` | biomes | `biomes` or `city`: `data/<set>.hwm` + `data/<set>.hwt` |
| `hierwave_seed` | 1 | sampler seed |
| `hierwave_provider` | auto | `sampler` (C library via FFI), `placeholder` (Lua hash pattern), `auto` = sampler when available |
| `hierwave_gen_env` | auto | `main` (server env, on_generated + VoxelManip) or `mapgen` (async emerge threads); auto = main for the sampler, mapgen for the placeholder |
| `hierwave_chunk_tiles` | 32 | sampler chunk cache side `C` (`hw_world_new`) |
| `hierwave_lib`, `hierwave_model`, `hierwave_template` | | path overrides |
| `hierwave_log_chunks` | false | per-mapchunk timings |
| `hierwave_test_radius`, `hierwave_test_shutdown`, `hierwave_ffi_selftest` | | test hooks (set by `run.sh test`) |

Missing `data/<set>.hwt` falls back to `hierwave_gen/test_data/placeholder.hwt`
(8 templates, `tools/make_placeholder_hwt.py`).  Mapgen is `singlenode`,
`chunksize = 4` (64^3 mapchunks, origin offset -32), forced by the mod.

## Mapping (contract)

Tile (r, c) covers world x = 8c .. 8c+7 and z = -8r .. -8r-7; template cell
(i, j) is at x = 8c + j, z = -(8r + i); layer k at y = k - 4.  Below y = -4
stone, above y = 19 air.  A 64^3 mapchunk spans 8 tile columns and 9 tile
rows (z is mirrored), fetched with one `hw_world_region(r0, c0, 9, 8)`.

## FFI: what works where (Luanti 5.17.0, LuaJIT 2.1, macOS arm64)

| | `/Applications/luanti.app` (notarized) | `luanti/build/luanti.app` (ad hoc re-signed copy) |
|---|---|---|
| `jit`, LuaJIT | yes | yes |
| main env: `request_insecure_environment()` for a trusted mod, `ie.require("ffi")` | yes | yes |
| main env: `ffi.load` of our own dylib | **no**: library validation, "mapping process and mapped file (non-platform) have different Team IDs" | **yes** (`hw_dummy(20) = 41`, sampler runs) |
| mapgen env, mod security on | no `request_insecure_environment`, `require` disabled: no FFI | same |
| mapgen env, `secure.enable_security = false` | `require("ffi")` works, `ffi.load` blocked as above | works (tested by `ffi_test/run_probe.sh`) |

So the FFI path is: `setup_app.sh` copies the app into `luanti/build/`,
clears the copy's extended attributes (quarantine) and signs it ad hoc
(no hardened runtime, so no library validation).  `/Applications` is only
read.  Mod security stays on; `hierwave_gen` is in `secure.trusted_mods`,
gets FFI from the insecure env at load time, and generates in the server
env (`register_on_generated` + `get_mapgen_object("voxelmanip")`).

Fallback (official app, or no library / model): the placeholder provider
in the async mapgen env, pure Lua; the log says
`sampler unavailable (...); using placeholder provider`.

Not used (option for later): with `secure.enable_security = false` the
mapgen env can `require("ffi")`; `mapgen.lua` then shares the main env's
`hw_world*` (address passed through `core.ipc_set`; the world cache is
mutex-protected) and generation moves to the emerge threads.  Select it
with `hierwave_gen_env = mapgen` (tested: biomes `test 1` passes, 0.46 s).

Server-env generation runs under the environment lock: a slow sampler
chunk stalls the server for that long (city: ~12 s per new 32 x 32 tile
sampler chunk at present, "Maximum lag peaked at 18 s" in the test).

## Headless test

`luanti/run.sh test R` makes a fresh world `luanti/worlds/test-<set>-<seed>`,
emerges mapchunks |cx|, |cz| <= R (y -96..31), samples 2000 random nodes and
compares each against provider + template computed independently in the
main env, logs `TEST PASSED` / `TEST FAILED`, and shuts down.  It also calls
`hw_dummy` from `luanti/build/libhwdummy.dylib`.  Last results:

| run | provider / env | chunks | time | check |
|---|---|---|---|---|
| `test 1` biomes | sampler / server env | 18 | 0.75 s (tiles 26 ms, fill 7 ms per chunk) | 0 / 2000 |
| `test 2` city | sampler / server env | 50 | 37 s (tiles 728 ms, fill 8 ms per chunk; ~12 s per new sampler chunk) | 0 / 2000 |
| `test 1` placeholder | Lua / mapgen env | 18 | 0.11 s | 0 / 2000 |
| `test 1` official app | auto -> placeholder | 18 | | 0 / 2000 |

## Files

- `luanti.conf`, `run.sh`, `setup_app.sh`, `build.sh`
- `games/hierwave/game.conf`
- `games/hierwave/mods/hierwave_nodes/`: `init.lua` (nodes `hierwave:<name>`,
  water / deep_water non-spreading liquids, glass / leaves see-through,
  viewer privileges and spawn), `palette.lua` and `textures/` (generated)
- `games/hierwave/mods/hierwave_gen/`: `init.lua` (settings, FFI, provider
  choice, server-env generation), `mapgen.lua` (async mapgen env),
  `common.lua` (`.hwt` reader, providers, C API cdef, chunk fill),
  `test.lua`, `settingtypes.txt`, `test_data/placeholder.hwt`
- `tools/make_textures.py`, `tools/make_placeholder_hwt.py`
- `ffi_test/dummy.c`, `ffi_test/run_probe.sh` + `ffi_test/games/hwprobe`
  (FFI probe of both envs)
- generated, gitignored: `build/`, `user/`, `worlds/`
