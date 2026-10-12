#!/bin/sh
# Make luanti/build/luanti.app: a repo-local copy of /Applications/luanti.app,
# ad hoc signed without hardened runtime, so LuaJIT FFI may dlopen our own
# dylibs (the notarized app has library validation: only Apple or its own
# Team ID libraries load).  Touches only the copy; /Applications is read.
# Idempotent: skips if the copy matches the installed version.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="${LUANTI_APP:-/Applications/luanti.app}"
DST="$HERE/build/luanti.app"
src_ver="$("$SRC/Contents/MacOS/luanti" --version 2>/dev/null | head -1)"
if [ -x "$DST/Contents/MacOS/luanti" ]; then
	dst_ver="$("$DST/Contents/MacOS/luanti" --version 2>/dev/null | head -1 || true)"
	if [ -n "$dst_ver" ] && [ "$dst_ver" = "$src_ver" ]; then
		echo "$DST is up to date ($dst_ver)"
		exit 0
	fi
fi
mkdir -p "$HERE/build"
rm -rf "$DST"
cp -R "$SRC" "$DST"
xattr -cr "$DST"
codesign --force --deep --sign - "$DST"
"$DST/Contents/MacOS/luanti" --version | head -1
echo "made $DST (ad hoc signed copy of $SRC)"
