import os
import time

import numpy as np
import pytest

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
pygame = pytest.importorskip("pygame")

from castlegen import infinite as inf  # noqa: E402
from castlegen import viewer as vw  # noqa: E402


def _viewer(S=16, C=16, R=2, seed=3):
    tsv = vw.biomes_view(S=S)
    pygame.init()
    v = vw.Viewer(tsv, vw.NumbaBackend(tsv.model, tsv.sched, seed), C=C, render_distance=R,
                  size=(320, 240), max_batch=6)
    return tsv, v


def test_headless_smoke_and_exact_chunks():
    tsv, v = _viewer()
    v.run(frames=20, script=lambda f: (1, 1))          # walk south-east, nothing raises
    assert v.py > 0 and v.px_ > 0
    assert v.store.wait_idle(60) and v.store.error is None
    st = v.store
    assert len(st.chunks) >= 25 and st.batches >= 1
    assert all(k in st.chunks for k in st.wanted())
    # every stored chunk equals infinite.generate on its region
    C = v.C
    ys = [k[0] for k in st.chunks]
    xs = [k[1] for k in st.chunks]
    y0, x0 = min(ys), min(xs)
    ref = inf.generate(tsv.model, tsv.sched, 3, y0 * C, x0 * C, (max(ys) - y0 + 1) * C, (max(xs) - x0 + 1) * C)
    for (cy, cx), g in st.chunks.items():
        assert np.array_equal(g, ref[(cy - y0) * C:(cy - y0 + 1) * C, (cx - x0) * C:(cx - x0 + 1) * C])
    screen = pygame.Surface((320, 240))
    v.draw(screen)
    assert "cached" in v.hud_text()


def test_frame_loop_not_blocked_by_generation():
    """The kernel runs without the GIL: the main thread keeps ticking while a
    large batch generates."""
    tsv = vw.biomes_view(S=64)
    b = vw.NumbaBackend(tsv.model, tsv.sched, 0)
    t0 = time.perf_counter()
    st = vw.ChunkStore(b, C=32, render_distance=2, max_batch=25)    # one 160^2 batch, ~0.4 s
    try:
        ticks = 0
        while not st.idle() and time.perf_counter() - t0 < 60:
            time.sleep(0.005)
            ticks += 1
        el = time.perf_counter() - t0
        assert st.idle()
        assert ticks > 0.5 * el / 0.006      # main thread was never held for long
    finally:
        st.stop()


def test_batches_grow_to_rectangles():
    class Rec:
        name = "rec"

        def __init__(self):
            self.calls = []

        def region(self, y0, x0, H, W):
            self.calls.append((y0, x0, H, W))
            return np.zeros((H, W), np.int32)

    b = Rec()
    st = vw.ChunkStore(b, C=8, render_distance=1, max_batch=9)
    try:
        assert st.wait_idle(10)
        assert b.calls == [(-8, -8, 24, 24)]          # 3 x 3 chunks in one batch
        st.set_center(0, 1)                             # step east: one new column strip
        assert st.wait_idle(10)
        assert b.calls[1] == (-8, 16, 24, 8)
    finally:
        st.stop()


def test_c_backend_falls_back():
    tsv = vw.biomes_view(S=4)
    b = vw.make_backend("c", tsv, 0, 8)
    assert hasattr(b, "region")
    if isinstance(b, vw.NumbaBackend):
        assert np.array_equal(b.region(0, 0, 8, 8), inf.generate(tsv.model, tsv.sched, 0, 0, 0, 8, 8))
