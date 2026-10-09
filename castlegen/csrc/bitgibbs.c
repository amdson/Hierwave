/* Bit-sliced tile Gibbs (castlegen/bitgibbs.py documents the model).

   A site's candidates are the bits of NW 64-bit words (tiles t < 64 NW).
   A factor is a stack of FB level planes plus a forbid plane: tile t sits at
   level sum_b bit_b(t) 2^b of it, or is forbidden.  The site's factors are
   added by a bit-sliced ripple adder into an AB-bit accumulator over all
   tiles at once (a carry out of the top bit forbids the tile), the
   accumulator split into its NL = 2^AB level masks, and the tile drawn
   exactly from weights cw[level]: a level by one integer draw over
   Z = sum n_l cw[l] (n_l = popcount), then a uniform tile of that level by
   the remainder (r / cw[l]) and a select of that set bit. */
#include <math.h>
#include <stdint.h>
#include <string.h>

typedef uint64_t u64;
typedef unsigned __int128 u128;
#define MAXW 4
#define AB 5
#define NL (1 << AB)

static inline u64 mix(u64 x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static inline double u01(u64 r) { return (r >> 11) * (1.0 / 9007199254740992.0); }

static inline int pc64(u64 w) { return __builtin_popcountll(w); }

static inline int select64(u64 w, int k) { /* position of the k-th set bit (k < popcount) */
    int pos = 0, c = pc64(w & 0xffffffffULL);
    if (k >= c) { k -= c; w >>= 32; pos += 32; }
    c = pc64(w & 0xffff);
    if (k >= c) { k -= c; w >>= 16; pos += 16; }
    c = pc64(w & 0xff);
    if (k >= c) { k -= c; w >>= 8; pos += 8; }
    for (;; w >>= 1, pos++)
        if (w & 1) { if (!k) return pos; k--; }
}

typedef struct { u64 a[AB][MAXW]; u64 bad[MAXW]; } Acc;

static inline void acc_init(Acc *A, int NW) {
    for (int b = 0; b < AB; b++) for (int w = 0; w < NW; w++) A->a[b][w] = 0;
    for (int w = 0; w < NW; w++) A->bad[w] = 0;
}

static inline void acc_add(Acc *A, int NW, int FB, const u64 *x) {
    for (int w = 0; w < NW; w++) {
        u64 c = 0;
        for (int b = 0; b < AB; b++) {
            u64 xb = b < FB ? x[b * NW + w] : 0, a = A->a[b][w];
            A->a[b][w] = a ^ xb ^ c;
            c = (a & xb) | (c & (a ^ xb));
        }
        A->bad[w] |= c | x[FB * NW + w];
    }
}

static inline void acc_addc(Acc *A, int NW, int FB, int L, const u64 *M) { /* level L on the tiles of M */
    u64 x[(AB + 1) * MAXW];
    for (int b = 0; b <= FB; b++)
        for (int w = 0; w < NW; w++)
            x[b * NW + w] = (b < FB ? (L >> b) & 1 : L >= (1 << FB)) ? M[w] : 0;
    acc_add(A, NW, FB, x);
}

static u64 acc_levels(const Acc *A, int NW, const u64 *legal, const u64 *cw, u64 lev[NL][MAXW], u64 *n) {
    /* level masks (legal, not forbidden) and their counts; returns Z = sum n_l cw[l] */
    for (int l = 0; l < NL; l++) n[l] = 0;
    for (int w = 0; w < NW; w++) {
        u64 m[2 * NL];
        m[1] = legal[w] & ~A->bad[w];
        for (int b = AB - 1; b >= 0; b--) {
            int lo = 1 << (AB - 1 - b);
            u64 ab = A->a[b][w];
            for (int i = lo; i < 2 * lo; i++) { m[2 * i] = m[i] & ~ab; m[2 * i + 1] = m[i] & ab; }
        }
        for (int l = 0; l < NL; l++) { lev[l][w] = m[NL + l]; n[l] += pc64(m[NL + l]); }
    }
    u64 Z = 0;
    for (int l = 0; l < NL; l++) Z += n[l] * cw[l];
    return Z;
}

static int acc_pick(int NW, u64 lev[NL][MAXW], const u64 *n, const u64 *cw, u64 r) { /* r < Z */
    for (int l = 0; l < NL; l++) {
        u64 wl = n[l] * cw[l];
        if (r < wl) {
            u64 k = r / cw[l];
            for (int w = 0; w < NW; w++) {
                u64 c = pc64(lev[l][w]);
                if (k < c) return w * 64 + select64(lev[l][w], (int)k);
                k -= c;
            }
        }
        r -= wl;
    }
    return -1;
}

static int acc_draw(const Acc *A, int NW, const u64 *legal, const u64 *cw, u64 rnd) {
    u64 lev[NL][MAXW], n[NL];
    u64 Z = acc_levels(A, NW, legal, cw, lev, n);
    if (!Z) return -1;
    return acc_pick(NW, lev, n, cw, (u64)(((u128)rnd * Z) >> 64));
}

static inline int acc_level(const Acc *A, int t) { /* tile t's accumulated level */
    int l = 0;
    for (int b = 0; b < AB; b++) l |= (int)((A->a[b][t >> 6] >> (t & 63)) & 1) << b;
    return l;
}

/* One draw from F factor stacks (stk: F (FB + 1) NW words, contiguous). */
int bg_sample(int NW, int FB, int F, const u64 *stk, const u64 *legal, const u64 *cw, u64 seed) {
    Acc A;
    acc_init(&A, NW);
    for (int f = 0; f < F; f++) acc_add(&A, NW, FB, stk + (u64)f * (FB + 1) * NW);
    return acc_draw(&A, NW, legal, cw, mix(seed));
}

void bg_sample_many(int NW, int FB, int F, const u64 *stk, const u64 *legal, const u64 *cw, u64 seed, int n,
                    int32_t *out) {
    for (int i = 0; i < n; i++) out[i] = bg_sample(NW, FB, F, stk, legal, cw, seed * 0x100000001b3ULL + i);
}

/* Pair MRF on an H x W torus: tile t at p with neighbour n on side d costs
   pair[d][n] (stacks (4, S, FB + 1, NW)), plus unary (one stack or NULL).
   Checkerboard order (exact in parallel); a site with no legal tile keeps
   its tile.  Ex (4, S, S) [d][n][t] and ux (S,) float energies, if given:
   the quantized draw is a proposal, accepted with exp(-(dE) + unit dlevel)
   (Metropolis-Hastings, exact for the float energies, cw = exp(-unit l)). */
void bg_sweep_pairs(int H, int W, int NW, int FB, int S, int32_t *T, const u64 *pair, const u64 *unary,
                    const u64 *legal, const u64 *cw, u64 seed, int sweeps, const double *Ex, const double *ux,
                    double unit, int64_t *acc) {
    const int st = (FB + 1) * NW;
    const int DY[4] = {-1, 0, 1, 0}, DX[4] = {0, 1, 0, -1};
    u64 ctr = 0;
    for (int s = 0; s < sweeps; s++)
        for (int c = 0; c < 2; c++)
            for (int y = 0; y < H; y++)
                for (int x = (y + c) & 1; x < W; x += 2) {
                    Acc A;
                    int n[4];
                    acc_init(&A, NW);
                    for (int d = 0; d < 4; d++) {
                        n[d] = T[((y + DY[d] + H) % H) * W + (x + DX[d] + W) % W];
                        acc_add(&A, NW, FB, pair + ((u64)d * S + n[d]) * st);
                    }
                    if (unary) acc_add(&A, NW, FB, unary);
                    int t = acc_draw(&A, NW, legal, cw, mix(seed ^ mix(++ctr)));
                    if (t < 0) continue;
                    if (Ex) {
                        int t0 = T[y * W + x];
                        double e1 = ux[t], e0 = ux[t0];
                        for (int d = 0; d < 4; d++) {
                            e1 += Ex[((u64)d * S + n[d]) * S + t];
                            e0 += Ex[((u64)d * S + n[d]) * S + t0];
                        }
                        double la = -(e1 - e0) + unit * (acc_level(&A, t) - acc_level(&A, t0));
                        acc[1]++;
                        if (la < 0 && u01(mix(seed ^ mix(++ctr))) >= exp(la)) continue;
                        acc[0]++;
                    }
                    T[y * W + x] = t;
                }
}

/* ------------------------------------------------------------------ fields
   An H x W map of k x k blocks with blockfield.py's auxiliary fields (g, d),
   per site a joint (t, g) | d draw (bit-sliced, one level count per g) then
   (g, d) | t exactly.  Witnesses and dependants only inside the site's own
   block (each block certifies its own connectivity); tiles see their real
   neighbours across block seams (off the map: wall).  Validity is hard (LV
   -> infinity): no (t, g) or (g, d) is offered that leaves the site or a
   neighbour in its block invalid.  Energy: pair stacks, unary[site] (-logz +
   parent term), gcost[g~] (mu g~ on nodes, eps g~ elsewhere, g~ = g with
   INF = G + 1), lport per opening on a closed port, lport for no opening on
   an open port whose other cells hold none, lport on a sink block's root
   cell while it is not a node (labs[b][8] = root index in the block, -1
   none), lseam per seam rule broken across a block seam (seambad), and
   (g, d) weight exp(-(mu|eps) g~ - delta d). */
#define BIT(M, t) (((M)[(t) >> 6] >> ((t) & 63)) & 1)

typedef struct {
    int H, W, k, NW, S, G, D;
    const int32_t *T, *Gf, *Df;
    const u64 *J, *exits;   /* J (4, S, NW): t joined to n on t's side d; exits (H W, NW) */
} Fld;

static const int DY4[4] = {-1, 0, 1, 0}, DX4[4] = {0, 1, 0, -1};

static inline int joined(const Fld *F, int t, int n, int d) { return BIT(F->J + ((u64)d * F->S + n) * F->NW, t); }
static inline int is_exit(const Fld *F, int p, int t) { return BIT(F->exits + (u64)p * F->NW, t); }
static inline int same_block(const Fld *F, int y, int x, int ny, int nx) {
    return ny >= 0 && ny < F->H && nx >= 0 && nx < F->W && ny / F->k == y / F->k && nx / F->k == x / F->k;
}

static int valid_skip(const Fld *F, int y, int x, int skip) { /* blockfield._valid */
    int W = F->W, p = y * W + x, g = F->Gf[p], dd = F->Df[p], t = F->T[p], INF = F->G + 1;
    if (g >= INF || is_exit(F, p, t)) return 1;
    for (int d = 0; d < 4; d++) {
        if (d == skip) continue;
        int ny = y + DY4[d], nx = x + DX4[d];
        if (!same_block(F, y, x, ny, nx)) continue;
        int q = ny * W + nx, gq = F->Gf[q];
        if (gq >= INF || F->Df[q] >= dd) continue;
        if (gq + !joined(F, t, F->T[q], d) <= g) return 1;
    }
    return 0;
}

static inline int port_of(int y, int x, int d, int k) { /* block-local (y, x) */
    int hh = k / 2;
    if (d == 0 && y == 0) return x / hh;
    if (d == 1 && x == k - 1) return 2 + y / hh;
    if (d == 2 && y == k - 1) return 4 + x / hh;
    if (d == 3 && x == 0) return 6 + y / hh;
    return -1;
}

void bg_sweep_field(int H, int W, int k, int NW, int FB, int S, int32_t *T, int32_t *Gf, int32_t *Df,
                    const u64 *pair, const u64 *unary, const u64 *gcost, const u64 *J, const u64 *legal,
                    const u64 *exits, const u64 *openout, const u64 *all, const u64 *seambad, const int32_t *labs,
                    int lport, int lseam, const u64 *node, int wall, int G, int D, double mu, double eps,
                    double delta, const u64 *cw, u64 seed, int sweeps) {
    const int st = (FB + 1) * NW, INF = G + 1, hh = k / 2, BC = W / k;
    Fld F = {H, W, k, NW, S, G, D, T, Gf, Df, J, exits};
    u64 ctr = 0;
    double lw[64];
    int los[64], his[64];
    for (int s = 0; s < sweeps; s++)
        for (int c = 0; c < 5; c++)
            for (int y = 0; y < H; y++)
                for (int x = 0; x < W; x++) {
                    if ((x + 2 * y) % 5 != c) continue;
                    int p = y * W + x, g = Gf[p], dp = Df[p], ly = y % k, lx = x % k;
                    const int32_t *lab = labs + ((y / k) * BC + x / k) * 9;
                    int nt[4], inb[4], inm[4], dep[4];
                    for (int d = 0; d < 4; d++) {
                        int ny = y + DY4[d], nx = x + DX4[d];
                        inm[d] = ny >= 0 && ny < H && nx >= 0 && nx < W;
                        inb[d] = same_block(&F, y, x, ny, nx);
                        nt[d] = inm[d] ? T[ny * W + nx] : wall;
                        dep[d] = 0;
                        if (inb[d]) {
                            int q = ny * W + nx;
                            dep[d] = Gf[q] < INF && !is_exit(&F, q, T[q]) && !valid_skip(&F, ny, nx, (d + 2) & 3);
                        }
                    }
                    /* ---- energy of the tile, g aside */
                    Acc A;
                    acc_init(&A, NW);
                    for (int d = 0; d < 4; d++) {
                        acc_add(&A, NW, FB, pair + ((u64)d * S + nt[d]) * st);
                        if (inm[d] && !inb[d]) acc_addc(&A, NW, FB, lseam, seambad + ((u64)d * S + nt[d]) * NW);
                    }
                    acc_add(&A, NW, FB, unary + (u64)p * st);
                    if (lab[8] == ly * k + lx) {
                        u64 M[MAXW];
                        for (int w = 0; w < NW; w++) M[w] = all[w] & ~node[w];
                        acc_addc(&A, NW, FB, lport, M);
                    }
                    for (int d = 0; d < 4; d++) {
                        int q = port_of(ly, lx, d, k);
                        if (q < 0) continue;
                        if (lab[q] == 0) { acc_addc(&A, NW, FB, lport, openout + (u64)d * NW); continue; }
                        int cnt = 0;                                   /* openings on the port's other cells */
                        for (int i = 0; i < hh; i++) {
                            int yy = d == 0 ? 0 : d == 2 ? k - 1 : (q & 1) * hh + i;
                            int xx = d == 3 ? 0 : d == 1 ? k - 1 : (q & 1) * hh + i;
                            int pp = (y - ly + yy) * W + x - lx + xx;
                            if (pp != p) cnt += BIT(openout + (u64)d * NW, T[pp]);
                        }
                        if (!cnt) {
                            u64 M[MAXW];
                            for (int w = 0; w < NW; w++) M[w] = all[w] & ~openout[d * NW + w];
                            acc_addc(&A, NW, FB, lport, M);
                        }
                    }
                    /* ---- joint (t, g) | d: per g' the tiles that keep p and its dependants valid */
                    u64 levg[NL][MAXW], ng[NL], Zg[64], Ztot = 0;
                    int pick = -1;
                    u64 rr = mix(seed ^ mix(++ctr));
                    for (int pass = 0; pass < 2 && pick < 0; pass++) {
                        u64 r = pass ? (u64)(((u128)rr * Ztot) >> 64) : 0;
                        if (pass && !Ztot) break;
                        for (int gg = 0; gg <= INF; gg++) {
                            if (pass && r >= Zg[gg]) { r -= Zg[gg]; continue; }
                            u64 L[MAXW];
                            for (int w = 0; w < NW; w++) L[w] = legal[(u64)p * NW + w];
                            if (gg < INF) {
                                u64 V[MAXW];
                                int any = 0;
                                for (int w = 0; w < NW; w++) V[w] = exits[(u64)p * NW + w];
                                for (int d = 0; d < 4; d++) {
                                    if (!inb[d]) continue;
                                    int q = (y + DY4[d]) * W + x + DX4[d], gq = Gf[q];
                                    if (gq >= INF || Df[q] >= dp) continue;
                                    if (gq + 1 <= gg) any = 1;
                                    else if (gq == gg)
                                        for (int w = 0; w < NW; w++) V[w] |= J[((u64)d * S + nt[d]) * NW + w];
                                }
                                if (!any) for (int w = 0; w < NW; w++) L[w] &= V[w];
                            }
                            for (int d = 0; d < 4; d++) {
                                if (!dep[d]) continue;
                                int q = (y + DY4[d]) * W + x + DX4[d], gq = Gf[q];
                                if (gg >= INF || dp >= Df[q] || gg > gq) { for (int w = 0; w < NW; w++) L[w] = 0; }
                                else if (gg == gq) for (int w = 0; w < NW; w++) L[w] &= J[((u64)d * S + nt[d]) * NW + w];
                            }
                            Acc B = A;
                            acc_add(&B, NW, FB, gcost + (u64)gg * st);
                            u64 Z = acc_levels(&B, NW, L, cw, levg, ng);
                            if (!pass) { Zg[gg] = Z; Ztot += Z; continue; }
                            pick = acc_pick(NW, levg, ng, cw, r);
                            if (pick >= 0) { T[p] = pick; Gf[p] = g = gg; }
                            break;
                        }
                    }
                    int t = T[p];
                    /* ---- (g, d) | tiles */
                    int ex = is_exit(&F, p, t), nd = BIT(node, t);
                    double cst = nd ? mu : eps, mx = -INFINITY;
                    int wg[4], wd[4], sq[4];
                    for (int d = 0; d < 4; d++) {
                        wg[d] = INF, wd[d] = 0, sq[d] = 1;
                        if (!inb[d]) continue;
                        int q = (y + DY4[d]) * W + x + DX4[d];
                        wg[d] = Gf[q], wd[d] = Df[q], sq[d] = !joined(&F, t, nt[d], d);
                    }
                    for (int gg = 0; gg <= INF; gg++) {
                        int lo = 0, hi = D, ok = 1;
                        if (gg < INF && !ex) {
                            lo = D + 1;
                            for (int d = 0; d < 4; d++)
                                if (wg[d] < INF && wg[d] + sq[d] <= gg && wd[d] + 1 < lo) lo = wd[d] + 1;
                        }
                        for (int d = 0; d < 4; d++)
                            if (dep[d]) {
                                if (gg >= INF || gg + sq[d] > wg[d]) ok = 0;
                                else if (wd[d] - 1 < hi) hi = wd[d] - 1;
                            }
                        los[gg] = lo, his[gg] = hi;
                        lw[gg] = -INFINITY;
                        if (ok && lo <= hi) {
                            double n = hi - lo + 1;
                            lw[gg] = -cst * gg - delta * lo + (delta > 0 ? log1p(-exp(-delta * n)) - log1p(-exp(-delta)) : log(n));
                            if (lw[gg] > mx) mx = lw[gg];
                        }
                    }
                    if (mx == -INFINITY) continue;                     /* no valid (g, d): keep */
                    double tot = 0;
                    for (int gg = 0; gg <= INF; gg++) tot += exp(lw[gg] - mx);
                    double u = u01(mix(seed ^ mix(++ctr))) * tot;
                    int gg = 0;
                    for (; gg < INF; gg++) { u -= exp(lw[gg] - mx); if (u < 0) break; }
                    while (lw[gg] == -INFINITY) gg--;                  /* rounding at the end */
                    int lo = los[gg], n = his[gg] - lo + 1;
                    double v = u01(mix(seed ^ mix(++ctr)));
                    int dd = delta > 0 ? (int)floor(-log1p(-v * -expm1(-delta * n)) / delta) : (int)(v * n);
                    Gf[p] = gg;
                    Df[p] = lo + (dd < n ? dd : n - 1);
                }
}
