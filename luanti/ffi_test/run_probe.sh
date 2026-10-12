#!/bin/sh
# Headless FFI probe; prints the PROBE lines.  Needs luanti/build.sh first.
#   LUANTI=<binary>       default: luanti/build/luanti.app (re-signed copy)
#   HWPROBE_SECURITY=false to run with mod security off
HERE="$(cd "$(dirname "$0")" && pwd)"
LUANTI="${LUANTI:-$HERE/../build/luanti.app/Contents/MacOS/luanti}"
SEC="${HWPROBE_SECURITY:-true}"
printf 'title = hwprobe\n' > "$HERE/games/hwprobe/game.conf"
printf 'name = hwprobe\n' > "$HERE/games/hwprobe/mods/hwprobe/mod.conf"
printf 'mg_name = singlenode\nchunksize = 4\nsecure.trusted_mods = hwprobe\nsecure.enable_security = %s\n' "$SEC" > "$HERE/probe.conf"
rm -rf "$HERE/world"
LUANTI_USER_PATH="$HERE/user" LUANTI_GAME_PATH="$HERE/games" \
	"$LUANTI" --server --gameid hwprobe --world "$HERE/world" \
	--config "$HERE/probe.conf" --logfile "" 2>&1 | grep -E "PROBE|ERROR"
