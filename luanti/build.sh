#!/bin/sh
# Build luanti/build/libhwdummy.dylib (FFI smoke test).  The real sampler,
# cache/libhwsampler.dylib, is built by thread A's Python side.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$HERE/build"
cc -O2 -shared -fPIC -o "$HERE/build/libhwdummy.dylib" "$HERE/ffi_test/dummy.c"
echo "built $HERE/build/libhwdummy.dylib"
