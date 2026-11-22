// Lane kernels of dfsparse.cpp: refactorisation, solves and residuals for LANES systems at once.
// Included by dfsparse.cpp once per instruction set, each time inside its own namespace (see
// there); it relies on the definitions that precede the inclusion (LU, Group, recip, LANES).
// No #include here: everything in this file is compiled for the namespace's target.

// One value per lane: GCC/Clang vector extension, lowered to SSE2, AVX2 or AVX-512 as available.
// The kernels below are written with these types because the auto-vectoriser left the
// equivalent per-lane loops scalar. Every lane computes the same expression, in the same
// order, as the scalar code. With -ffp-contract=off the results are bit-identical.
typedef double vlane __attribute__((vector_size(LANES * sizeof(double))));
typedef double vlane_u __attribute__((vector_size(LANES * sizeof(double)), aligned(sizeof(double)), may_alias));
static inline __attribute__((always_inline)) vlane vld(const double* p) { return *reinterpret_cast<const vlane_u*>(p); }
static inline __attribute__((always_inline)) void vst(double* p, vlane v) { *reinterpret_cast<vlane_u*>(p) = v; }
static inline __attribute__((always_inline)) vlane vsplat(double x) { vlane v; for (int l = 0; l < LANES; ++l) v[l] = x; return v; }

void refactor_lanes(const LU& ref, const int* colptr, const int* rowind, const cplx* const* values,
                    const std::vector<int>& q, double tol, Group& g, LaneVec& xr_buf, LaneVec& xi_buf) {
    constexpr int W = LANES;
    const int n = ref.n;
    const size_t nnz = static_cast<size_t>(colptr[n]);
    const size_t nl = ref.Li.size(), nu = ref.Ui.size();
    g.lr.alloc(nl * W); g.li.alloc(nl * W); g.ur.alloc(nu * W); g.ui.alloc(nu * W);
    g.dr.alloc(static_cast<size_t>(n) * W); g.di.alloc(static_cast<size_t>(n) * W);
    g.ar.alloc(nnz * W); g.ai.alloc(nnz * W);
    double* __restrict ar = g.ar.data();
    double* __restrict ai = g.ai.data();
    for (size_t p = 0; p < nnz; ++p)
        for (int l = 0; l < W; ++l) {
            const cplx v = values[l][p];
            ar[p * W + l] = v.re;
            ai[p * W + l] = v.im;
        }
    xr_buf.assign(static_cast<size_t>(n) * W, 0.0);
    xi_buf.assign(static_cast<size_t>(n) * W, 0.0);
    double* __restrict xr = xr_buf.data();
    double* __restrict xi = xi_buf.data();
    double* __restrict lr = g.lr.data();
    double* __restrict li = g.li.data();
    double* __restrict ur = g.ur.data();
    double* __restrict ui = g.ui.data();
    double* __restrict dr = g.dr.data();
    double* __restrict di = g.di.data();
    const int *pinv = ref.pinv.data(), *Lp = ref.Lp.data(), *Li = ref.Li.data(), *Up = ref.Up.data(),
              *Ui = ref.Ui.data();
    bool lane_ok[W];
    for (int l = 0; l < W; ++l) lane_ok[l] = true;
    const double tol2 = tol * tol;
    const vlane zero = vsplat(0.0), one = vsplat(1.0);
    for (int k = 0; k < n; ++k) {
        const int col = q[k];
        for (int p = colptr[col]; p < colptr[col + 1]; ++p) {
            const size_t t = static_cast<size_t>(pinv[rowind[p]]) * W, s = static_cast<size_t>(p) * W;
            vst(xr + t, vld(ar + s));
            vst(xi + t, vld(ai + s));
        }
        const int d = Up[k + 1] - 1;
        for (int p = Up[k]; p < d; ++p) {  // stored in topological order
            const int jc = Ui[p];
            const size_t j = static_cast<size_t>(jc) * W, up = static_cast<size_t>(p) * W;
            const vlane ujr = vld(xr + j), uji = vld(xi + j);
            vst(ur + up, ujr);
            vst(ui + up, uji);
            vst(xr + j, zero);
            vst(xi + j, zero);
            for (int r = Lp[jc] + 1; r < Lp[jc + 1]; ++r) {
                const size_t t = static_cast<size_t>(Li[r]) * W, m = static_cast<size_t>(r) * W;
                const vlane a = vld(lr + m), b = vld(li + m);
                vst(xr + t, vld(xr + t) - (a * ujr - b * uji));
                vst(xi + t, vld(xi + t) - (a * uji + b * ujr));
            }
        }
        const size_t kk = static_cast<size_t>(k) * W, dd = static_cast<size_t>(d) * W;
        vst(ur + dd, vld(xr + kk));
        vst(ui + dd, vld(xi + kk));
        vst(xr + kk, zero);
        vst(xi + kk, zero);
        vlane amax2 = zero;
        for (int r = Lp[k] + 1; r < Lp[k + 1]; ++r) {
            const size_t t = static_cast<size_t>(Li[r]) * W;
            const vlane a = vld(xr + t), b = vld(xi + t);
            const vlane a2 = a * a + b * b;
            amax2 = a2 > amax2 ? a2 : amax2;
        }
        double invr_s[W], invi_s[W];
        for (int l = 0; l < W; ++l) {
            const double pr = ur[dd + l], pi = ui[dd + l];
            recip(pr, pi, invr_s[l], invi_s[l]);
            dr[kk + l] = invr_s[l];
            di[kk + l] = invi_s[l];
            const double p2 = pr * pr + pi * pi;
            const bool good = p2 > 0 && std::isfinite(p2) && p2 >= tol2 * amax2[l] && std::isfinite(invr_s[l]) &&
                              std::isfinite(invi_s[l]);
            lane_ok[l] = lane_ok[l] && good;
        }
        const vlane invr = vld(invr_s), invi = vld(invi_s);
        const size_t l0 = static_cast<size_t>(Lp[k]) * W;
        vst(lr + l0, one);
        vst(li + l0, zero);
        for (int r = Lp[k] + 1; r < Lp[k + 1]; ++r) {
            const size_t t = static_cast<size_t>(Li[r]) * W, m = static_cast<size_t>(r) * W;
            const vlane a = vld(xr + t), b = vld(xi + t);
            vst(lr + m, a * invr - b * invi);
            vst(li + m, a * invi + b * invr);
            vst(xr + t, zero);
            vst(xi + t, zero);
        }
    }
    for (int l = 0; l < W; ++l) g.ok[l] = l < g.lanes && lane_ok[l];
}

// x = op(A)^{-1} b for all lanes of a group (vectors (n, LANES)); trans 0 N, 1 T, 2 H.
void solve_lanes(const LU& ref, const Group& g, const std::vector<int>& q, int trans, const double* br,
                 const double* bi, double* xr, double* xi, double* tr, double* ti) {
    constexpr int W = LANES;
    const int n = ref.n;
    const int *pinv = ref.pinv.data(), *Lp = ref.Lp.data(), *Li = ref.Li.data(), *Up = ref.Up.data(),
              *Ui = ref.Ui.data();
    const double *lr = g.lr.data(), *li = g.li.data(), *ur = g.ur.data(), *ui = g.ui.data(), *dr = g.dr.data(),
                 *di = g.di.data();
    if (trans == 0) {
        for (int i = 0; i < n; ++i) {
            const size_t t = static_cast<size_t>(pinv[i]) * W, s = static_cast<size_t>(i) * W;
            vst(tr + t, vld(br + s));
            vst(ti + t, vld(bi + s));
        }
        for (int j = 0; j < n; ++j) {  // L t = P b
            const size_t jj = static_cast<size_t>(j) * W;
            const vlane vr = vld(tr + jj), vi = vld(ti + jj);
            for (int p = Lp[j] + 1; p < Lp[j + 1]; ++p) {
                const size_t t = static_cast<size_t>(Li[p]) * W, m = static_cast<size_t>(p) * W;
                const vlane a = vld(lr + m), b = vld(li + m);
                vst(tr + t, vld(tr + t) - (a * vr - b * vi));
                vst(ti + t, vld(ti + t) - (a * vi + b * vr));
            }
        }
        for (int j = n - 1; j >= 0; --j) {  // U t = t
            const size_t jj = static_cast<size_t>(j) * W;
            const vlane a = vld(tr + jj), b = vld(ti + jj), er = vld(dr + jj), ei = vld(di + jj);
            const vlane vr = a * er - b * ei, vi = a * ei + b * er;
            vst(tr + jj, vr);
            vst(ti + jj, vi);
            const int d = Up[j + 1] - 1;
            for (int p = Up[j]; p < d; ++p) {
                const size_t t = static_cast<size_t>(Ui[p]) * W, m = static_cast<size_t>(p) * W;
                const vlane c = vld(ur + m), e = vld(ui + m);
                vst(tr + t, vld(tr + t) - (c * vr - e * vi));
                vst(ti + t, vld(ti + t) - (c * vi + e * vr));
            }
        }
        for (int k = 0; k < n; ++k) {
            const size_t t = static_cast<size_t>(q[k]) * W, s = static_cast<size_t>(k) * W;
            vst(xr + t, vld(tr + s));
            vst(xi + t, vld(ti + s));
        }
        return;
    }
    const vlane c = vsplat(trans == 2 ? -1.0 : 1.0);  // sign of the imaginary part of L, U (conjugate for H)
    for (int k = 0; k < n; ++k) {
        const size_t t = static_cast<size_t>(k) * W, s = static_cast<size_t>(q[k]) * W;
        vst(tr + t, vld(br + s));
        vst(ti + t, vld(bi + s));
    }
    for (int j = 0; j < n; ++j) {  // U^T t = t (U^H: conjugate)
        const size_t jj = static_cast<size_t>(j) * W;
        vlane sr = vld(tr + jj), si = vld(ti + jj);
        const int d = Up[j + 1] - 1;
        for (int p = Up[j]; p < d; ++p) {
            const size_t t = static_cast<size_t>(Ui[p]) * W, m = static_cast<size_t>(p) * W;
            const vlane mr = vld(ur + m), mi = c * vld(ui + m), xr_ = vld(tr + t), xi_ = vld(ti + t);
            sr -= mr * xr_ - mi * xi_;
            si -= mr * xi_ + mi * xr_;
        }
        const vlane er = vld(dr + jj), ei = c * vld(di + jj);
        vst(tr + jj, sr * er - si * ei);
        vst(ti + jj, sr * ei + si * er);
    }
    for (int j = n - 1; j >= 0; --j) {  // L^T t = t (L^H: conjugate)
        const size_t jj = static_cast<size_t>(j) * W;
        vlane sr = vld(tr + jj), si = vld(ti + jj);
        for (int p = Lp[j] + 1; p < Lp[j + 1]; ++p) {
            const size_t t = static_cast<size_t>(Li[p]) * W, m = static_cast<size_t>(p) * W;
            const vlane mr = vld(lr + m), mi = c * vld(li + m), xr_ = vld(tr + t), xi_ = vld(ti + t);
            sr -= mr * xr_ - mi * xi_;
            si -= mr * xi_ + mi * xr_;
        }
        vst(tr + jj, sr);
        vst(ti + jj, si);
    }
    for (int i = 0; i < n; ++i) {
        const size_t t = static_cast<size_t>(i) * W, s = static_cast<size_t>(pinv[i]) * W;
        vst(xr + t, vld(tr + s));
        vst(xi + t, vld(ti + s));
    }
}

// r = b - op(A) x for all lanes, with the group's original A values.
void residual_lanes(int n, const int* colptr, const int* rowind, const Group& g, int trans, const double* br,
                    const double* bi, const double* xr, const double* xi, double* rr, double* ri) {
    constexpr int W = LANES;
    const double *ar = g.ar.data(), *ai = g.ai.data();
    std::memcpy(rr, br, sizeof(double) * n * W);
    std::memcpy(ri, bi, sizeof(double) * n * W);
    if (trans == 0) {
        for (int j = 0; j < n; ++j) {
            const size_t jj = static_cast<size_t>(j) * W;
            const vlane vr = vld(xr + jj), vi = vld(xi + jj);
            for (int p = colptr[j]; p < colptr[j + 1]; ++p) {
                const size_t t = static_cast<size_t>(rowind[p]) * W, m = static_cast<size_t>(p) * W;
                const vlane a = vld(ar + m), b = vld(ai + m);
                vst(rr + t, vld(rr + t) - (a * vr - b * vi));
                vst(ri + t, vld(ri + t) - (a * vi + b * vr));
            }
        }
        return;
    }
    const vlane c = vsplat(trans == 2 ? -1.0 : 1.0);
    for (int j = 0; j < n; ++j) {
        const size_t jj = static_cast<size_t>(j) * W;
        vlane sr = vld(rr + jj), si = vld(ri + jj);
        for (int p = colptr[j]; p < colptr[j + 1]; ++p) {
            const size_t t = static_cast<size_t>(rowind[p]) * W, m = static_cast<size_t>(p) * W;
            const vlane mr = vld(ar + m), mi = c * vld(ai + m), xr_ = vld(xr + t), xi_ = vld(xi + t);
            sr -= mr * xr_ - mi * xi_;
            si -= mr * xi_ + mi * xr_;
        }
        vst(rr + jj, sr);
        vst(ri + jj, si);
    }
}

// Complex product as NumPy computes it (re = fma(ar, br, -ai bi), im = fma(ar, bi, ai br) on FMA
// hardware; separately rounded products otherwise), so the result matches NumPy bit for bit.
static inline __attribute__((always_inline)) void cmul_np(double ar, double ai, double br, double bi, double& re,
                                                          double& im) {
#if DFS_CMUL_FMA
    re = __builtin_fma(ar, br, -(ai * bi));
    im = __builtin_fma(ar, bi, ai * br);
#else
    re = ar * br - ai * bi;
    im = ar * bi + ai * br;
#endif
}

// Adjoint of the sparse solves w.r.t. the matrix entries (one pass, no temporaries):
//   out[f, j] = (-bb[f, sr[j]]) * x[f, sc[j]] - sum_d conj(w[f, d, sr[j]]) * y[f, sc[j], d]
// bb, x: (F, n); w: (F, D, n); y: (F, n, D); out: rows of `out_stride` complex elements.
// The operations and their order are those of the NumPy expression in host._backward.
void adjoint_entries(int F, int nnz, int n, int D, const long long* sr, const long long* sc, const double* bb,
                     const double* x, const double* w, const double* y, double* out, long long out_stride) {
    for (int f = 0; f < F; ++f) {
        const double* bbf = bb + 2 * static_cast<size_t>(f) * n;
        const double* xf = x + 2 * static_cast<size_t>(f) * n;
        const double* wf = w + 2 * static_cast<size_t>(f) * D * n;
        const double* yf = y + 2 * static_cast<size_t>(f) * n * D;
        double* of = out + 2 * static_cast<size_t>(f) * out_stride;
        for (int j = 0; j < nnz; ++j) {
            const size_t r = static_cast<size_t>(sr[j]), c = static_cast<size_t>(sc[j]);
            double tr, ti;
            cmul_np(-bbf[2 * r], -bbf[2 * r + 1], xf[2 * c], xf[2 * c + 1], tr, ti);
            double sr_ = 0.0, si_ = 0.0;
            for (int d = 0; d < D; ++d) {
                const size_t wi = 2 * (static_cast<size_t>(d) * n + r), yi = 2 * (c * D + d);
                double pr, pi;
                cmul_np(wf[wi], -wf[wi + 1], yf[yi], yf[yi + 1], pr, pi);
                if (d == 0) { sr_ = pr; si_ = pi; } else { sr_ += pr; si_ += pi; }
            }
            of[2 * j] = tr - sr_;
            of[2 * j + 1] = ti - si_;
        }
    }
}
