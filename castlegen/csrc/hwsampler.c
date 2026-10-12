/* hwsampler.c: infinite tile worlds by annealed checkerboard Gibbs with
 * position-hashed noise, a plain C11 port of castlegen/infinite.py
 * (generate, _anneal, _anneal_sockets, World).  Output equals
 * castlegen.infinite.generate bit for bit: same splitmix hash and key
 * layout, same draw order, same double arithmetic (build with
 * -ffp-contract=off so no multiply-add is fused; exp from libm).
 *
 * No threads inside.  hw_tiles is reentrant (scratch per call); a hw_world
 * may be shared by threads: its chunk map is guarded by a mutex, chunk
 * generation runs outside the lock (two threads may generate the same chunk;
 * the results are identical and the first insert wins).
 *
 * Model file (.hwm, written by castlegen/export_models.py), little endian:
 *   char[4]  "HWM1"
 *   uint32   kind          0 pair tables (TileModel), 1 sockets (SocketModel)
 *   uint32   T             tiles
 *   uint32   S             sweeps of the schedule
 *   uint32   flags         bit 0: wildcard fix pass (socket models; id T)
 *   double   beta[S], gamma[S]          (gamma may be +inf)
 *   double   u[T]
 *   kind 0:  double Eh[T][T], Ev[T][T]; int32 Hh[T][T], Hv[T][T]
 *   kind 1:  double wild_u; int32 sN[T], sE[T], sS[T], sW[T];
 *            double base[S][T]   = exp(-beta[s] (u[t] - min u)) as numpy
 *                                  computes it (stored so no exp differs)
 *
 * Kernels (see infinite.py): step 0 draws every cell of the padded block
 * from the unary at beta[0]; half-step h = 1 .. 2S (+1 for the wildcard
 * pass) updates the cells of global parity (y + x) & 1 == (h - 1) & 1 at
 * margin >= h; each draw takes its uniform from hash(seed, y, x, h).
 *
 * Socket kernel speed-ups that keep the numba result exactly:
 *  - a draw sums its weights once into a prefix array and binary-searches
 *    it (the same additions in the same order as _draw's two loops);
 *  - finite-gamma weights start from bs[t] eg[4] (one row per half-step)
 *    and only tiles sharing a socket with a neighbour are re-multiplied;
 *  - lexicographic half-steps (gamma = inf) draw among the tiles that fit
 *    all four neighbours, found from a per-socket tile list in index order:
 *    the other weights are exactly 0.0, so sums and the search agree.
 * The order-fixed weight sum (T dependent adds per update) is the floor.
 * hw_model_set_fast(m, 1) additionally replaces finite-gamma draws by an
 * equivalent-in-distribution but different-in-bits draw that touches only
 * the tiles sharing a socket with a neighbour; deterministic, not equal to
 * numba (extra uniforms for rejection: hash(seed, y, x, h + k 2^40)).
 *
 * Extra API: hw_model_S, hw_model_pad, hw_model_set_fast, hw_world_chunks,
 * hw_u01.  Calls return 0 on success, -1 on allocation failure. */
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct hw_model {
    int kind, T, S, wild;
    double *beta, *gamma, *u;
    double *Eh, *Ev;
    int32_t *Hh, *Hv;
    double wild_u;
    int32_t *sock[4];          /* sN sE sS sW */
    double *base;              /* [S][T] */
    double *cdf0;              /* prefix of base[0] */
    double *bcum;              /* [S][T + 1] prefix sums of base[s] from 0 (fast path) */
    int nsock;
    int32_t *loff[4], *lidx[4];  /* tiles by socket on each side, index order */
    int fast;
} hw_model;

/* ------------------------------------------------------------------ noise */
static inline uint64_t mix64(uint64_t z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static inline double u01(uint64_t seed, int64_t y, int64_t x, int64_t h) {
    uint64_t z = mix64(seed ^ 0x9E3779B97F4A7C15ULL);
    z = mix64(z ^ (uint64_t)y);
    z = mix64(z ^ ((uint64_t)x * 0xD1B54A32D192ED03ULL));
    z = mix64(z ^ ((uint64_t)h * 0x8CB92BA72F3D8DD7ULL));
    return (double)(z >> 11) * (1.0 / 9007199254740992.0);
}

double hw_u01(uint64_t seed, int64_t y, int64_t x, int64_t h) { return u01(seed, y, x, h); }

/* _draw: index from weights w[:n] with uniform r (reference form) */
static int draw(const double *w, int n, double r) {
    double tot = 0.0;
    for (int t = 0; t < n; t++) tot += w[t];
    r *= tot;
    double acc = 0.0;
    for (int t = 0; t < n; t++) {
        acc += w[t];
        if (r < acc) return t;
    }
    for (int t = n - 1; t >= 0; t--)
        if (w[t] > 0.0) return t;
    return 0;
}

/* first k in [0, n) with c[k] > r (c nondecreasing), n if none */
static inline int upper(const double *c, int n, double r) {
    int lo = 0, hi = n;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (c[mid] <= r) lo = mid + 1; else hi = mid;
    }
    return lo;
}

/* draw from w[:n] whose running sums are c[:n] (c[t] = w[0] + ... + w[t]
 * added left to right): equals draw(w, n, r). */
static inline int draw_prefix(const double *w, const double *c, int n, double r) {
    r *= c[n - 1];
    int k = upper(c, n, r);
    if (k < n) return k;
    for (int t = n - 1; t >= 0; t--)
        if (w[t] > 0.0) return t;
    return 0;
}

/* ------------------------------------------------------------------ pair kernel */
static void anneal_pairs(const hw_model *m, int32_t *grid, int H, int W, int64_t gy0, int64_t gx0,
                         uint64_t seed, double *e, int64_t *v, double *w) {
    const int T = m->T, S = m->S;
    const double *u = m->u, *Eh = m->Eh, *Ev = m->Ev;
    const int32_t *Hh = m->Hh, *Hv = m->Hv;
    double b0 = m->beta[0], umin = u[0];
    for (int t = 1; t < T; t++)
        if (u[t] < umin) umin = u[t];
    for (int t = 0; t < T; t++) w[t] = exp(-b0 * (u[t] - umin));
    for (int64_t i = 0; i < H; i++)
        for (int64_t j = 0; j < W; j++)
            grid[i * W + j] = draw(w, T, u01(seed, gy0 + i, gx0 + j, 0));
    for (int64_t h = 1; h <= 2 * (int64_t)S; h++) {
        int s = (int)((h - 1) >> 1);
        double b = m->beta[s], g = m->gamma[s];
        int hard_inf = isinf(g);
        int64_t par = (h - 1) & 1;
        for (int64_t i = h; i < H - h; i++) {
            int64_t j0 = h + ((par - (gy0 + i + gx0 + h)) & 1);
            for (int64_t j = j0; j < W - h; j += 2) {
                int up = grid[(i - 1) * W + j], dn = grid[(i + 1) * W + j];
                int lf = grid[i * W + j - 1], rt = grid[i * W + j + 1];
                int64_t vmin = 1 << 30;
                for (int t = 0; t < T; t++) {
                    e[t] = u[t] + Eh[lf * T + t] + Eh[t * T + rt] + Ev[up * T + t] + Ev[t * T + dn];
                    v[t] = (int64_t)Hh[lf * T + t] + Hh[t * T + rt] + Hv[up * T + t] + Hv[t * T + dn];
                    if (v[t] < vmin) vmin = v[t];
                }
                double emin = INFINITY;
                for (int t = 0; t < T; t++) {
                    double x;
                    if (hard_inf)
                        x = v[t] == vmin ? b * e[t] : INFINITY;
                    else
                        x = b * e[t] + g * (double)v[t];
                    e[t] = x;
                    if (x < emin) emin = x;
                }
                for (int t = 0; t < T; t++) w[t] = exp(-(e[t] - emin));
                grid[i * W + j] = draw(w, T, u01(seed, gy0 + i, gx0 + j, h));
            }
        }
    }
}

/* ------------------------------------------------------------------ socket kernel */
/* fast path for finite gamma: weight bs[t] eg[v_t]; tiles outside the union
 * U of the four per-socket lists all have v = 4.  With x = r Z: U tiles
 * one by one (list order), else the rest as one block of mass
 * eg[4] (B - sum_U bs), sampled by bs from the prefix sums of base[s] with
 * rejection of U tiles (extra uniforms hash(seed, y, x, h + k 2^40)).
 * Exact in distribution, deterministic, not bitwise numba. */
static int draw_fast(const hw_model *m, int s, const double *bs, const double *eg, const int32_t *a,
                     int32_t *mark, int32_t *ulist, double *uw, double r,
                     uint64_t seed, int64_t gy, int64_t gx, int64_t h) {
    const int T = m->T;
    int nu = 0;
    for (int d = 0; d < 4; d++) {
        int sk = a[d];
        if (sk < 0 || sk >= m->nsock) continue;
        const int32_t *L = m->lidx[d] + m->loff[d][sk];
        int n = m->loff[d][sk + 1] - m->loff[d][sk];
        for (int k = 0; k < n; k++) {
            int t = L[k];
            if (!mark[t]) ulist[nu++] = t;
            mark[t]++;
        }
    }
    double su = 0.0, tu = 0.0;
    for (int k = 0; k < nu; k++) {
        int t = ulist[k];
        su += bs[t];
        uw[k] = bs[t] * eg[4 - mark[t]];
        tu += uw[k];
    }
    const double *cum = m->bcum + (size_t)s * (T + 1);
    double B = cum[T];
    double rest = (B - su) * eg[4];
    if (!(rest > 1e-12 * B * eg[4])) rest = 0.0;     /* U holds (numerically) all the mass */
    double x = r * (tu + rest);
    int out = -1;
    if (x < tu || rest == 0.0) {
        double acc = 0.0;
        for (int k = 0; k < nu; k++) {
            acc += uw[k];
            if (x < acc) { out = ulist[k]; break; }
        }
        if (out < 0) out = nu ? ulist[nu - 1] : 0;
    } else {
        for (int64_t k = 1; k <= 64 && out < 0; k++) {
            int t = upper(cum + 1, T, u01(seed, gy, gx, h + (k << 40)) * B);
            if (t >= T) t = T - 1;
            if (!mark[t] && bs[t] > 0.0) out = t;
        }
        if (out < 0) {                                /* rare: U holds most of the mass */
            double y = u01(seed, gy, gx, h + (65LL << 40)) * (B - su), acc = 0.0;
            for (int t = 0; t < T; t++) {
                if (mark[t] || bs[t] <= 0.0) continue;
                out = t;
                acc += bs[t];
                if (y < acc) break;
            }
            if (out < 0) out = ulist[0];
        }
    }
    for (int k = 0; k < nu; k++) mark[ulist[k]] = 0;
    return out;
}

static void anneal_sockets(const hw_model *m, int32_t *grid, int H, int W, int64_t gy0, int64_t gx0,
                           uint64_t seed, double *w, double *c, uint8_t *v, int32_t *cand,
                           int32_t *mark, int32_t *ulist, double *w4) {
    const int T = m->T, S = m->S;
    const int32_t *sN = m->sock[0], *sE = m->sock[1], *sS = m->sock[2], *sW = m->sock[3];
    const double *cdf = m->cdf0;
    double acc0 = cdf[T - 1];
    for (int64_t i = 0; i < H; i++)
        for (int64_t j = 0; j < W; j++) {
            double r = u01(seed, gy0 + i, gx0 + j, 0) * acc0;
            int k = upper(cdf, T, r);
            grid[i * W + j] = k < T - 1 ? k : T - 1;
        }
    double eg[5];
    int64_t last = 2 * (int64_t)S + (m->wild ? 1 : 0);
    for (int64_t h = 1; h <= last; h++) {
        int s = (int)((h - 1) >> 1);
        if (s > S - 1) s = S - 1;
        int fixp = h > 2 * (int64_t)S;
        double g = fixp ? INFINITY : m->gamma[s];
        int hard_inf = isinf(g);
        for (int k = 0; k < 5; k++) eg[k] = (hard_inf && k > 0) ? 0.0 : exp(-g * k);
        int64_t par = (h - 1) & 1;
        const double *bs = m->base + (size_t)s * T;
        if (!hard_inf)
            for (int t = 0; t < T; t++) w4[t] = bs[t] * eg[4];
        for (int64_t i = h; i < H - h; i++) {
            int64_t j0 = h + ((par - (gy0 + i + gx0 + h)) & 1);
            for (int64_t j = j0; j < W - h; j += 2) {
                int32_t a_up = sS[grid[(i - 1) * W + j]], a_dn = sN[grid[(i + 1) * W + j]];
                int32_t a_lf = sE[grid[i * W + j - 1]], a_rt = sW[grid[i * W + j + 1]];
                double r = u01(seed, gy0 + i, gx0 + j, h);
                int out;
                if (hard_inf) {
                    /* tiles fitting all four sides, from the shortest side list */
                    int32_t a[4] = {a_up, a_rt, a_dn, a_lf};   /* wanted sN, sE, sS, sW */
                    int best = -1, bn = T + 1;
                    for (int d = 0; d < 4; d++) {
                        int n = (a[d] >= 0 && a[d] < m->nsock) ? m->loff[d][a[d] + 1] - m->loff[d][a[d]] : 0;
                        if (n < bn) { bn = n; best = d; }
                    }
                    int nc = 0;
                    if (bn > 0) {
                        const int32_t *L = m->lidx[best] + m->loff[best][a[best]];
                        for (int k = 0; k < bn; k++) {
                            int t = L[k];
                            if (sN[t] == a_up && sE[t] == a_rt && sS[t] == a_dn && sW[t] == a_lf) cand[nc++] = t;
                        }
                    }
                    if (nc > 0) {
                        double acc = 0.0;
                        for (int k = 0; k < nc; k++) {
                            w[k] = bs[cand[k]];
                            acc += w[k];
                            c[k] = acc;
                        }
                        out = cand[draw_prefix(w, c, nc, r)];
                    } else if (fixp) {
                        out = T;
                    } else {
                        int vmin = 4;
                        for (int t = 0; t < T; t++) {
                            int n = (sN[t] != a_up) + (sS[t] != a_dn) + (sW[t] != a_lf) + (sE[t] != a_rt);
                            v[t] = (uint8_t)n;
                            if (n < vmin) vmin = n;
                        }
                        double acc = 0.0;
                        for (int t = 0; t < T; t++) {
                            w[t] = v[t] == vmin ? bs[t] : 0.0;
                            acc += w[t];
                            c[t] = acc;
                        }
                        out = draw_prefix(w, c, T, r);
                    }
                } else if (m->fast) {
                    int32_t a[4] = {a_up, a_rt, a_dn, a_lf};
                    out = draw_fast(m, s, bs, eg, a, mark, ulist, c, r, seed, gy0 + i, gx0 + j, h);
                } else {
                    /* w = bs eg[4] everywhere, then the tiles matching a side
                     * patched to bs eg[4 - matches] (the same products) */
                    memcpy(w, w4, 8 * (size_t)T);
                    int32_t a[4] = {a_up, a_rt, a_dn, a_lf};
                    int nu = 0;
                    for (int d = 0; d < 4; d++) {
                        int sk = a[d];
                        if (sk < 0 || sk >= m->nsock) continue;
                        const int32_t *L = m->lidx[d] + m->loff[d][sk];
                        int n = m->loff[d][sk + 1] - m->loff[d][sk];
                        for (int k = 0; k < n; k++) {
                            int t = L[k];
                            if (!mark[t]) ulist[nu++] = t;
                            mark[t]++;
                        }
                    }
                    for (int k = 0; k < nu; k++) {
                        int t = ulist[k];
                        w[t] = bs[t] * eg[4 - mark[t]];
                        mark[t] = 0;
                    }
                    double acc = 0.0;
                    for (int t = 0; t < T; t++) {
                        acc += w[t];
                        c[t] = acc;
                    }
                    out = draw_prefix(w, c, T, r);
                }
                grid[i * W + j] = out;
            }
        }
    }
}

/* ------------------------------------------------------------------ model */
void hw_model_free(hw_model *m) {
    if (!m) return;
    free(m->beta); free(m->gamma); free(m->u); free(m->Eh); free(m->Ev); free(m->Hh); free(m->Hv);
    for (int d = 0; d < 4; d++) { free(m->sock[d]); free(m->loff[d]); free(m->lidx[d]); }
    free(m->base); free(m->cdf0); free(m->bcum);
    free(m);
}

static int rd(FILE *f, void *p, size_t sz, size_t n) { return fread(p, sz, n, f) == n; }

static void *rdalloc(FILE *f, size_t sz, size_t n, int *ok) {
    void *p = malloc(sz * (n ? n : 1));
    if (!p || !rd(f, p, sz, n)) *ok = 0;
    return p;
}

hw_model *hw_model_load(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    char magic[4];
    uint32_t hd[4];
    if (!rd(f, magic, 1, 4) || memcmp(magic, "HWM1", 4) || !rd(f, hd, 4, 4) || hd[0] > 1 || hd[1] == 0 ||
        hd[2] == 0) {
        fclose(f);
        return NULL;
    }
    hw_model *m = calloc(1, sizeof *m);
    int ok = 1;
    m->kind = (int)hd[0]; m->T = (int)hd[1]; m->S = (int)hd[2]; m->wild = (int)(hd[3] & 1);
    size_t T = (size_t)m->T, S = (size_t)m->S;
    m->beta = rdalloc(f, 8, S, &ok);
    m->gamma = rdalloc(f, 8, S, &ok);
    m->u = rdalloc(f, 8, T, &ok);
    if (m->kind == 0) {
        m->wild = 0;
        m->Eh = rdalloc(f, 8, T * T, &ok);
        m->Ev = rdalloc(f, 8, T * T, &ok);
        m->Hh = rdalloc(f, 4, T * T, &ok);
        m->Hv = rdalloc(f, 4, T * T, &ok);
    } else {
        ok = ok && rd(f, &m->wild_u, 8, 1);
        for (int d = 0; d < 4; d++) m->sock[d] = rdalloc(f, 4, T, &ok);
        m->base = rdalloc(f, 8, S * T, &ok);
    }
    fclose(f);
    if (!ok) { hw_model_free(m); return NULL; }
    if (m->kind == 1) {
        m->cdf0 = malloc(8 * T);
        double acc = 0.0;
        for (size_t t = 0; t < T; t++) { acc += m->base[t]; m->cdf0[t] = acc; }
        m->bcum = malloc(8 * S * (T + 1));
        for (size_t s = 0; s < S; s++) {
            double *cm = m->bcum + s * (T + 1);
            cm[0] = 0.0;
            for (size_t t = 0; t < T; t++) cm[t + 1] = cm[t] + m->base[s * T + t];
        }
        int ns = 0;
        for (int d = 0; d < 4; d++)
            for (size_t t = 0; t < T; t++) {
                if (m->sock[d][t] < 0) { hw_model_free(m); return NULL; }
                if (m->sock[d][t] + 1 > ns) ns = m->sock[d][t] + 1;
            }
        m->nsock = ns;
        for (int d = 0; d < 4; d++) {
            m->loff[d] = calloc((size_t)ns + 1, 4);
            m->lidx[d] = malloc(4 * T);
            for (size_t t = 0; t < T; t++) m->loff[d][m->sock[d][t] + 1]++;
            for (int k = 0; k < ns; k++) m->loff[d][k + 1] += m->loff[d][k];
            int32_t *pos = malloc(4 * (size_t)ns);
            memcpy(pos, m->loff[d], 4 * (size_t)ns);
            for (size_t t = 0; t < T; t++) m->lidx[d][pos[m->sock[d][t]]++] = (int32_t)t;
            free(pos);
        }
    }
    return m;
}

int hw_model_T(const hw_model *m) { return m->T; }
int hw_model_S(const hw_model *m) { return m->S; }
int hw_model_pad(const hw_model *m) { return 2 * m->S + (m->kind == 1 && m->wild); }
void hw_model_set_fast(hw_model *m, int on) { m->fast = on ? 1 : 0; }

/* ------------------------------------------------------------------ generate */
int hw_tiles(const hw_model *m, uint64_t seed, int64_t y0, int64_t x0, int H, int W, int32_t *out) {
    if (H <= 0 || W <= 0) return 0;
    int P = hw_model_pad(m);
    int GH = H + 2 * P, GW = W + 2 * P;
    size_t T = (size_t)m->T;
    int32_t *grid = calloc((size_t)GH * GW, 4);
    double *w = malloc(8 * (T + 1)), *c = malloc(8 * (T + 1));
    int64_t *v64 = m->kind == 0 ? malloc(8 * T) : NULL;
    uint8_t *v8 = m->kind == 1 ? malloc(T) : NULL;
    int32_t *cand = m->kind == 1 ? malloc(4 * T) : NULL;
    int32_t *mark = m->kind == 1 ? calloc(T + 1, 4) : NULL;
    int32_t *ulist = m->kind == 1 ? malloc(4 * T) : NULL;
    double *w4 = m->kind == 1 ? malloc(8 * T) : NULL;
    int ok = grid && w && c && (m->kind == 0 ? v64 != NULL : (v8 && cand && mark && ulist && w4));
    if (ok) {
        if (m->kind == 0)
            anneal_pairs(m, grid, GH, GW, y0 - P, x0 - P, seed, c, v64, w);
        else
            anneal_sockets(m, grid, GH, GW, y0 - P, x0 - P, seed, w, c, v8, cand, mark, ulist, w4);
        for (int i = 0; i < H; i++) memcpy(out + (size_t)i * W, grid + (size_t)(i + P) * GW + P, 4 * (size_t)W);
    }
    free(grid); free(w); free(c); free(v64); free(v8); free(cand); free(mark); free(ulist); free(w4);
    return ok ? 0 : -1;
}

/* ------------------------------------------------------------------ world */
typedef struct { int64_t cy, cx; int32_t *data; } hw_slot;

typedef struct hw_world {
    const hw_model *m;
    uint64_t seed;
    int C;
    pthread_mutex_t mu;
    hw_slot *tab;
    size_t cap, n;
} hw_world;

static inline size_t slot_hash(int64_t cy, int64_t cx) {
    return (size_t)mix64((uint64_t)cy * 0x9E3779B97F4A7C15ULL ^ (uint64_t)cx);
}

static int32_t *wfind(const hw_world *wd, int64_t cy, int64_t cx) {
    size_t mask = wd->cap - 1;
    for (size_t k = slot_hash(cy, cx) & mask;; k = (k + 1) & mask) {
        const hw_slot *s = &wd->tab[k];
        if (!s->data) return NULL;
        if (s->cy == cy && s->cx == cx) return s->data;
    }
}

static void wput_raw(hw_slot *tab, size_t cap, int64_t cy, int64_t cx, int32_t *data) {
    size_t mask = cap - 1;
    size_t k = slot_hash(cy, cx) & mask;
    while (tab[k].data) k = (k + 1) & mask;
    tab[k].cy = cy; tab[k].cx = cx; tab[k].data = data;
}

static int wput(hw_world *wd, int64_t cy, int64_t cx, int32_t *data) {
    if (2 * (wd->n + 1) > wd->cap) {
        size_t nc = wd->cap * 2;
        hw_slot *nt = calloc(nc, sizeof *nt);
        if (!nt) return -1;
        for (size_t k = 0; k < wd->cap; k++)
            if (wd->tab[k].data) wput_raw(nt, nc, wd->tab[k].cy, wd->tab[k].cx, wd->tab[k].data);
        free(wd->tab);
        wd->tab = nt; wd->cap = nc;
    }
    wput_raw(wd->tab, wd->cap, cy, cx, data);
    wd->n++;
    return 0;
}

hw_world *hw_world_new(const hw_model *m, uint64_t seed, int C) {
    if (!m || C <= 0) return NULL;
    hw_world *wd = calloc(1, sizeof *wd);
    if (!wd) return NULL;
    wd->m = m; wd->seed = seed; wd->C = C;
    wd->cap = 64;
    wd->tab = calloc(wd->cap, sizeof *wd->tab);
    pthread_mutex_init(&wd->mu, NULL);
    return wd;
}

void hw_world_free(hw_world *wd) {
    if (!wd) return;
    for (size_t k = 0; k < wd->cap; k++) free(wd->tab[k].data);
    free(wd->tab);
    pthread_mutex_destroy(&wd->mu);
    free(wd);
}

static inline int64_t fdiv(int64_t a, int64_t b) { int64_t q = a / b; return (a % b != 0 && a < 0) ? q - 1 : q; }

int hw_world_chunks(hw_world *wd) {
    pthread_mutex_lock(&wd->mu);
    int n = (int)wd->n;
    pthread_mutex_unlock(&wd->mu);
    return n;
}

int hw_world_region(hw_world *wd, int64_t y0, int64_t x0, int H, int W, int32_t *out) {
    if (H <= 0 || W <= 0) return 0;
    const int64_t C = wd->C;
    int64_t cy0 = fdiv(y0, C), cy1 = fdiv(y0 + H - 1, C) + 1;
    int64_t cx0 = fdiv(x0, C), cx1 = fdiv(x0 + W - 1, C) + 1;
    /* missing chunks -> bounding rectangle (World.ensure) */
    int64_t my0 = INT64_MAX, my1 = INT64_MIN, mx0 = INT64_MAX, mx1 = INT64_MIN;
    pthread_mutex_lock(&wd->mu);
    for (int64_t cy = cy0; cy < cy1; cy++)
        for (int64_t cx = cx0; cx < cx1; cx++)
            if (!wfind(wd, cy, cx)) {
                if (cy < my0) my0 = cy;
                if (cy + 1 > my1) my1 = cy + 1;
                if (cx < mx0) mx0 = cx;
                if (cx + 1 > mx1) mx1 = cx + 1;
            }
    pthread_mutex_unlock(&wd->mu);
    if (my0 != INT64_MAX) {
        int BH = (int)((my1 - my0) * C), BW = (int)((mx1 - mx0) * C);
        int32_t *g = malloc((size_t)BH * BW * 4);
        if (!g || hw_tiles(wd->m, wd->seed, my0 * C, mx0 * C, BH, BW, g)) { free(g); return -1; }
        int rc = 0;
        pthread_mutex_lock(&wd->mu);
        for (int64_t cy = my0; cy < my1 && !rc; cy++)
            for (int64_t cx = mx0; cx < mx1; cx++) {
                if (wfind(wd, cy, cx)) continue;
                int32_t *d = malloc((size_t)C * C * 4);
                if (!d) { rc = -1; break; }
                for (int64_t i = 0; i < C; i++)
                    memcpy(d + i * C, g + ((cy - my0) * C + i) * BW + (cx - mx0) * C, 4 * (size_t)C);
                if (wput(wd, cy, cx, d)) { free(d); rc = -1; break; }
            }
        pthread_mutex_unlock(&wd->mu);
        free(g);
        if (rc) return rc;
    }
    pthread_mutex_lock(&wd->mu);
    for (int64_t cy = cy0; cy < cy1; cy++)
        for (int64_t cx = cx0; cx < cx1; cx++) {
            const int32_t *d = wfind(wd, cy, cx);
            if (!d) { pthread_mutex_unlock(&wd->mu); return -1; }
            int64_t ya = cy * C > y0 ? cy * C : y0, yb = (cy + 1) * C < y0 + H ? (cy + 1) * C : y0 + H;
            int64_t xa = cx * C > x0 ? cx * C : x0, xb = (cx + 1) * C < x0 + W ? (cx + 1) * C : x0 + W;
            for (int64_t y = ya; y < yb; y++)
                memcpy(out + (y - y0) * W + (xa - x0), d + (y - cy * C) * C + (xa - cx * C), 4 * (size_t)(xb - xa));
        }
    pthread_mutex_unlock(&wd->mu);
    return 0;
}
