#!/bin/sh
# Launch the Hierwave game with repo-local user data, games, world and config.
#
#   luanti/run.sh client          singleplayer window, straight into the world
#   luanti/run.sh server          dedicated server (connect with a client)
#   luanti/run.sh test [R]        headless: generate mapchunks |cx|,|cz| <= R
#                                 (default 1) in a fresh test world, check, exit
#
# Environment: HW_SET (biomes|city|marian, default biomes), HW_SEED (default 1),
#   LUANTI (binary; default luanti/build/luanti.app if present, else
#   /Applications/luanti.app; only the former can load our dylibs via FFI),
#   HW_EXTRA (extra config lines, e.g. "hierwave_provider = placeholder").
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
MODE="${1:-client}"
SET="${HW_SET:-biomes}"
SEED="${HW_SEED:-1}"
if [ -z "$LUANTI" ]; then
	if [ -x "$HERE/build/luanti.app/Contents/MacOS/luanti" ]; then
		LUANTI="$HERE/build/luanti.app/Contents/MacOS/luanti"
	else
		LUANTI="/Applications/luanti.app/Contents/MacOS/luanti"
	fi
fi
export LUANTI_USER_PATH="$HERE/user"
export LUANTI_GAME_PATH="$HERE/games"
mkdir -p "$HERE/build" "$HERE/worlds" "$HERE/user"

CONF="$HERE/build/$MODE.conf"
{
	cat "$HERE/luanti.conf"
	printf '\nhierwave_set = %s\nhierwave_seed = %s\n' "$SET" "$SEED"
	if [ "$SET" = marian ]; then
		# 3D modules fill y -8 .. 39: spawn above the roofs
		printf 'hierwave_spawn_y = 41.5\nstatic_spawnpoint = 4, 41.5, -4\n'
	fi
	if [ "$MODE" = test ]; then
		printf 'hierwave_test_radius = %s\nhierwave_test_shutdown = true\n' "${2:-1}"
		printf 'hierwave_log_chunks = true\nhierwave_ffi_selftest = true\n'
	fi
	[ -n "$HW_EXTRA" ] && printf '%s\n' "$HW_EXTRA"
} > "$CONF"

case "$MODE" in
client)
	WORLD="$HERE/worlds/$SET-$SEED"
	exec "$LUANTI" --go --gameid hierwave --world "$WORLD" --name viewer \
		--config "$CONF" --logfile "$HERE/user/debug.txt"
	;;
server)
	WORLD="$HERE/worlds/$SET-$SEED"
	exec "$LUANTI" --server --gameid hierwave --world "$WORLD" \
		--config "$CONF" --logfile "$HERE/user/debug.txt"
	;;
test)
	WORLD="$HERE/worlds/test-$SET-$SEED"
	rm -rf "$WORLD"
	exec "$LUANTI" --server --gameid hierwave --world "$WORLD" \
		--config "$CONF" --logfile "" --color never
	;;
*)
	echo "usage: $0 client|server|test [R]" >&2
	exit 2
	;;
esac
