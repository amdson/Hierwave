/* Tiny dylib to check that Luanti's LuaJIT FFI can load and call C.
   Build: luanti/build.sh  ->  luanti/build/libhwdummy.dylib */
#include <stdint.h>

int hw_dummy(int x) { return 2 * x + 1; }

/* Fills out[i] = (y0 + i / W) * 1000 + (x0 + i % W): checks int64 args and
   an int32 output buffer, the shape of hw_world_region. */
int hw_dummy_region(int64_t y0, int64_t x0, int H, int W, int32_t *out) {
    for (int i = 0; i < H * W; i++)
        out[i] = (int32_t)((y0 + i / W) * 1000 + (x0 + i % W));
    return 0;
}
