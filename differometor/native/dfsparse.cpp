// dfsparse: batched sparse LU for Differometor's host path (plain C ABI, loaded with ctypes).
//
// Every frequency of the public workload has the same sparsity pattern (751 x 751,
// ~2.6k nonzeros), so the pattern is analysed once (fill-reducing column order on
// A + A^T) and the F numeric factorisations then run back to back in one call.
//
// Factorisation: left-looking Gilbert-Peierls LU with threshold partial pivoting
// (A Q = P^T L U, L unit lower triangular), the algorithm used by CSparse/KLU.
// Each frequency pivots independently, so the result does not depend on the batch.
// Solves: A x = b, A^T x = b and A^H x = b for several right-hand sides, with
// optional fixed-precision iterative refinement (residual from the original A),
// matching `differometor.host.RefinedLU`.
//
// A system whose pivot column is exactly zero is reported singular; its solves
// return NaN (the dense solver's behaviour), other systems are unaffected.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>
#include <algorithm>

namespace {

struct cplx {
    double re, im;
};

inline cplx operator+(cplx a, cplx b) { return {a.re + b.re, a.im + b.im}; }
inline cplx operator-(cplx a, cplx b) { return {a.re - b.re, a.im - b.im}; }
inline cplx operator*(cplx a, cplx b) { return {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re}; }
inline cplx conj(cplx a) { return {a.re, -a.im}; }
inline double abs1(cplx a) { return std::hypot(a.re, a.im); }

// Smith's algorithm: avoids overflow/underflow for badly scaled entries.
inline cplx div(cplx a, cplx b) {
    if (std::fabs(b.re) >= std::fabs(b.im)) {
        double r = b.im / b.re, d = b.re + b.im * r;
        return {(a.re + a.im * r) / d, (a.im - a.re * r) / d};
    }
    double r = b.re / b.im, d = b.re * r + b.im;
    return {(a.re * r + a.im) / d, (a.im * r - a.re) / d};
}

// ---------------------------------------------------------------- ordering
// Minimum degree on the symmetric pattern of A + A^T with an explicit elimination
// graph stored as bitsets (n is small; this runs once per pattern).
// gram = false: graph of A + A^T.  gram = true: graph of A^T A (columns sharing a row),
// whose fill bounds that of LU with any row pivoting (the COLAMD idea).
std::vector<int> minimum_degree(int n, const int* colptr, const int* rowind, bool gram) {
    const int words = (n + 63) / 64;
    std::vector<uint64_t> adj(static_cast<size_t>(n) * words, 0);
    auto row = [&](int i) { return adj.data() + static_cast<size_t>(i) * words; };
    auto set = [&](int i, int j) { row(i)[j >> 6] |= uint64_t(1) << (j & 63); };
    if (gram) {
        std::vector<std::vector<int>> cols_of_row(n);
        for (int j = 0; j < n; ++j)
            for (int p = colptr[j]; p < colptr[j + 1]; ++p) cols_of_row[rowind[p]].push_back(j);
        for (auto& cs : cols_of_row)
            for (int a : cs)
                for (int b : cs)
                    if (a != b) set(a, b);
    } else {
        for (int j = 0; j < n; ++j)
            for (int p = colptr[j]; p < colptr[j + 1]; ++p) {
                int i = rowind[p];
                if (i != j) { set(i, j); set(j, i); }
            }
    }
    std::vector<int> degree(n), order;
    std::vector<char> done(n, 0);
    order.reserve(n);
    auto popcount = [&](int i) {
        int c = 0;
        const uint64_t* r = row(i);
        for (int w = 0; w < words; ++w) c += __builtin_popcountll(r[w]);
        return c;
    };
    for (int i = 0; i < n; ++i) degree[i] = popcount(i);
    std::vector<uint64_t> alive(words, 0);
    for (int i = 0; i < n; ++i) alive[i >> 6] |= uint64_t(1) << (i & 63);
    std::vector<int> neighbours;
    for (int step = 0; step < n; ++step) {
        int v = -1;
        for (int i = 0; i < n; ++i)
            if (!done[i] && (v < 0 || degree[i] < degree[v])) v = i;
        done[v] = 1;
        order.push_back(v);
        alive[v >> 6] &= ~(uint64_t(1) << (v & 63));
        uint64_t* rv = row(v);
        neighbours.clear();
        for (int w = 0; w < words; ++w) {
            uint64_t bits = rv[w] & alive[w];
            while (bits) {
                neighbours.push_back(w * 64 + __builtin_ctzll(bits));
                bits &= bits - 1;
            }
        }
        // Neighbours of v become a clique; v leaves the graph.
        for (int u : neighbours) {
            uint64_t* ru = row(u);
            for (int w = 0; w < words; ++w) ru[w] = (ru[w] | rv[w]) & alive[w];
            ru[u >> 6] &= ~(uint64_t(1) << (u & 63));
            degree[u] = popcount(u);
        }
    }
    return order;
}

// ---------------------------------------------------------------- one LU
struct LU {
    int n = 0;
    bool singular = false;
    std::vector<int> Lp, Li, Up, Ui;  // CSC; L has unit diagonal stored first, U diagonal stored last
    std::vector<cplx> Lx, Ux;
    std::vector<int> pinv;  // row i of A is row pinv[i] of L U
};

struct Workspace {
    std::vector<cplx> x;
    std::vector<int> xi, stack, pstack;
    std::vector<char> marked;
    void resize(int n) {
        x.assign(n, {0, 0});
        xi.assign(n, 0);
        stack.assign(n, 0);
        pstack.assign(n, 0);
        marked.assign(n, 0);
    }
};

// Reach of column `col` of A in the graph of L (rows mapped through pinv); returns top of xi.
int reach(const LU& f, const int* colptr, const int* rowind, int col, Workspace& w) {
    int n = f.n, top = n;
    for (int p = colptr[col]; p < colptr[col + 1]; ++p) {
        int start = rowind[p];
        if (w.marked[start]) continue;
        // iterative DFS
        int head = 0;
        w.stack[0] = start;
        while (head >= 0) {
            int j = w.stack[head];
            int J = f.pinv[j];
            if (!w.marked[j]) {
                w.marked[j] = 1;
                w.pstack[head] = (J < 0) ? 0 : f.Lp[J] + 1;  // skip unit diagonal
            }
            bool done = true;
            int end = (J < 0) ? 0 : f.Lp[J + 1];
            for (int q = w.pstack[head]; q < end; ++q) {
                int i = f.Li[q];
                if (w.marked[i]) continue;
                w.pstack[head] = q + 1;
                w.stack[++head] = i;
                done = false;
                break;
            }
            if (done) {
                --head;
                w.xi[--top] = j;
            }
        }
    }
    for (int p = top; p < n; ++p) w.marked[w.xi[p]] = 0;
    return top;
}

void factor_one(LU& f, int n, const int* colptr, const int* rowind, const cplx* values,
                const std::vector<int>& q, double tol, Workspace& w, size_t lnz_hint, size_t unz_hint) {
    f.n = n;
    f.singular = false;
    f.pinv.assign(n, -1);
    f.Lp.assign(n + 1, 0);
    f.Up.assign(n + 1, 0);
    f.Li.clear(); f.Lx.clear(); f.Ui.clear(); f.Ux.clear();
    f.Li.reserve(lnz_hint); f.Lx.reserve(lnz_hint); f.Ui.reserve(unz_hint); f.Ux.reserve(unz_hint);
    for (int k = 0; k < n; ++k) {
        f.Lp[k] = static_cast<int>(f.Li.size());
        f.Up[k] = static_cast<int>(f.Ui.size());
        int col = q[k];
        int top = reach(f, colptr, rowind, col, w);
        for (int p = top; p < n; ++p) w.x[w.xi[p]] = {0, 0};
        for (int p = colptr[col]; p < colptr[col + 1]; ++p) w.x[rowind[p]] = values[p];
        // x = L \ A(:, col) restricted to the reach (topological order)
        for (int px = top; px < n; ++px) {
            int j = w.xi[px];
            int J = f.pinv[j];
            if (J < 0) continue;
            cplx xj = w.x[j];  // L has unit diagonal
            for (int p = f.Lp[J] + 1; p < f.Lp[J + 1]; ++p) w.x[f.Li[p]] = w.x[f.Li[p]] - f.Lx[p] * xj;
        }
        // choose pivot among not-yet-pivotal rows; prefer the diagonal (threshold tol)
        int ipiv = -1;
        double amax = -1;
        for (int px = top; px < n; ++px) {
            int i = w.xi[px];
            if (f.pinv[i] < 0) {
                double a = abs1(w.x[i]);
                if (a > amax) { amax = a; ipiv = i; }
            } else {
                f.Ui.push_back(f.pinv[i]);
                f.Ux.push_back(w.x[i]);
            }
        }
        if (ipiv < 0 || !(amax > 0)) {  // structurally or numerically singular column (or NaN)
            f.singular = true;
            return;
        }
        if (f.pinv[col] < 0 && abs1(w.x[col]) >= tol * amax) ipiv = col;
        cplx pivot = w.x[ipiv];
        f.Ui.push_back(k);
        f.Ux.push_back(pivot);
        f.pinv[ipiv] = k;
        f.Li.push_back(ipiv);
        f.Lx.push_back({1, 0});
        for (int px = top; px < n; ++px) {
            int i = w.xi[px];
            if (f.pinv[i] < 0) {
                f.Li.push_back(i);
                f.Lx.push_back(div(w.x[i], pivot));
            }
            w.x[i] = {0, 0};
        }
    }
    f.Lp[n] = static_cast<int>(f.Li.size());
    f.Up[n] = static_cast<int>(f.Ui.size());
    for (auto& i : f.Li) i = f.pinv[i];  // L rows in pivot order
}

// Numeric refactorisation of `values` with the pivot order and L/U pattern of `ref`
// (KLU-style). Fails (returns false) if any pivot is zero or smaller than
// `tol` times the largest entry below it in its column, i.e. if the reused pivot
// order would not satisfy threshold partial pivoting for this system.
bool refactor_one(LU& f, const LU& ref, const int* colptr, const int* rowind, const cplx* values,
                  const std::vector<int>& q, double tol, std::vector<cplx>& x) {
    const int n = ref.n;
    f.n = n;
    f.singular = false;
    f.pinv = ref.pinv;
    f.Lp = ref.Lp; f.Li = ref.Li; f.Up = ref.Up; f.Ui = ref.Ui;
    f.Lx.resize(ref.Lx.size());
    f.Ux.resize(ref.Ux.size());
    const int* pinv = ref.pinv.data();
    for (int k = 0; k < n; ++k) {
        const int col = q[k];
        for (int p = colptr[col]; p < colptr[col + 1]; ++p) x[pinv[rowind[p]]] = values[p];
        const int d = f.Up[k + 1] - 1;
        for (int p = f.Up[k]; p < d; ++p) {  // stored in topological order
            const int j = f.Ui[p];
            const cplx ujk = x[j];
            x[j] = {0, 0};
            f.Ux[p] = ujk;
            if (ujk.re == 0 && ujk.im == 0) continue;
            for (int r = f.Lp[j] + 1; r < f.Lp[j + 1]; ++r) x[f.Li[r]] = x[f.Li[r]] - f.Lx[r] * ujk;
        }
        const cplx pivot = x[k];
        x[k] = {0, 0};
        const double apiv = abs1(pivot);
        double amax = 0;
        for (int r = f.Lp[k] + 1; r < f.Lp[k + 1]; ++r) amax = std::max(amax, abs1(x[f.Li[r]]));
        bool ok = apiv > 0 && apiv >= tol * amax && std::isfinite(apiv);
        f.Ux[d] = pivot;
        f.Lx[f.Lp[k]] = {1, 0};
        for (int r = f.Lp[k] + 1; r < f.Lp[k + 1]; ++r) {
            f.Lx[r] = ok ? div(x[f.Li[r]], pivot) : cplx{0, 0};
            x[f.Li[r]] = {0, 0};
        }
        if (!ok) return false;  // x is clean: column k's pattern was zeroed above
    }
    return true;
}

// ---------------------------------------------------------------- solves
// All vectors are dense length n, in pivot coordinates where noted.
void lsolve(const LU& f, cplx* x) {  // L x = b (unit diagonal)
    for (int j = 0; j < f.n; ++j) {
        cplx xj = x[j];
        if (xj.re == 0 && xj.im == 0) continue;
        for (int p = f.Lp[j] + 1; p < f.Lp[j + 1]; ++p) x[f.Li[p]] = x[f.Li[p]] - f.Lx[p] * xj;
    }
}
void usolve(const LU& f, cplx* x) {  // U x = b (diagonal last in each column)
    for (int j = f.n - 1; j >= 0; --j) {
        int d = f.Up[j + 1] - 1;
        x[j] = div(x[j], f.Ux[d]);
        cplx xj = x[j];
        if (xj.re == 0 && xj.im == 0) continue;
        for (int p = f.Up[j]; p < d; ++p) x[f.Ui[p]] = x[f.Ui[p]] - f.Ux[p] * xj;
    }
}
void ltsolve(const LU& f, cplx* x, bool c) {  // L^T x = b  (L^H if c)
    for (int j = f.n - 1; j >= 0; --j) {
        cplx s = x[j];
        for (int p = f.Lp[j] + 1; p < f.Lp[j + 1]; ++p) {
            cplx l = c ? conj(f.Lx[p]) : f.Lx[p];
            s = s - l * x[f.Li[p]];
        }
        x[j] = s;
    }
}
void utsolve(const LU& f, cplx* x, bool c) {  // U^T x = b  (U^H if c)
    for (int j = 0; j < f.n; ++j) {
        cplx s = x[j];
        int d = f.Up[j + 1] - 1;
        for (int p = f.Up[j]; p < d; ++p) {
            cplx u = c ? conj(f.Ux[p]) : f.Ux[p];
            s = s - u * x[f.Ui[p]];
        }
        x[j] = div(s, c ? conj(f.Ux[d]) : f.Ux[d]);
    }
}

// x = op(A)^{-1} b with A Q = P^T L U.  trans: 0 = N, 1 = T, 2 = H.
void solve_one(const LU& f, const std::vector<int>& q, int trans, const cplx* b, cplx* x, cplx* t) {
    int n = f.n;
    if (trans == 0) {
        for (int i = 0; i < n; ++i) t[f.pinv[i]] = b[i];  // t = P b
        lsolve(f, t);
        usolve(f, t);
        for (int k = 0; k < n; ++k) x[q[k]] = t[k];  // x = Q t
    } else {
        bool c = trans == 2;
        for (int k = 0; k < n; ++k) t[k] = b[q[k]];  // t = Q^T b
        utsolve(f, t, c);
        ltsolve(f, t, c);
        for (int i = 0; i < n; ++i) x[i] = t[f.pinv[i]];  // x = P^T t
    }
}

// r = b - op(A) x using the original CSC values.
void residual(int n, const int* colptr, const int* rowind, const cplx* values, int trans,
              const cplx* b, const cplx* x, cplx* r) {
    std::memcpy(r, b, sizeof(cplx) * n);
    if (trans == 0) {
        for (int j = 0; j < n; ++j) {
            cplx xj = x[j];
            for (int p = colptr[j]; p < colptr[j + 1]; ++p) r[rowind[p]] = r[rowind[p]] - values[p] * xj;
        }
    } else {
        bool c = trans == 2;
        for (int j = 0; j < n; ++j) {
            cplx s = r[j];
            for (int p = colptr[j]; p < colptr[j + 1]; ++p) {
                cplx a = c ? conj(values[p]) : values[p];
                s = s - a * x[rowind[p]];
            }
            r[j] = s;
        }
    }
}

struct Batch {
    int n = 0, F = 0, nnz = 0;
    std::vector<int> colptr, rowind, q;
    std::vector<cplx> values;  // (F, nnz) CSC order
    std::vector<LU> lus;
    int refactored = 0;
};

}  // namespace

extern "C" {

int dfs_version() { return 1; }

// Factorise F systems sharing one CSC pattern. values: (F, nnz) complex128, CSC order.
// order: 0 = natural, 1 = minimum degree on A + A^T, 2 = minimum degree on A^T A. tol: diagonal pivot threshold (1 = partial pivoting).
// singular_out[f] = 1 if system f is singular. Returns an opaque handle.
// reuse: 1 = refactorise systems 1..F-1 with the previous pivot order when every reused pivot is
// at least reuse_tol times the largest candidate in its column (else pivot that system afresh).
void* dfs_factor(int n, int F, const int* colptr, const int* rowind, const double* values,
                 int order, double tol, int reuse, double reuse_tol, int* singular_out) {
    auto* b = new Batch;
    b->n = n;
    b->F = F;
    b->nnz = colptr[n];
    b->colptr.assign(colptr, colptr + n + 1);
    b->rowind.assign(rowind, rowind + b->nnz);
    b->values.resize(static_cast<size_t>(F) * b->nnz);
    std::memcpy(b->values.data(), values, sizeof(cplx) * b->values.size());
    if (order == 1 || order == 2) {
        b->q = minimum_degree(n, colptr, rowind, order == 2);
    } else {
        b->q.resize(n);
        for (int i = 0; i < n; ++i) b->q[i] = i;
    }
    b->lus.resize(F);
    Workspace w;
    w.resize(n);
    std::vector<cplx> xr(n, {0, 0});
    size_t lnz = 4 * static_cast<size_t>(b->nnz), unz = lnz;
    int ref = -1;  // last pivoted factorisation, whose pivot order the next systems reuse
    b->refactored = 0;
    for (int f = 0; f < F; ++f) {
        LU& lu = b->lus[f];
        const cplx* vals = b->values.data() + static_cast<size_t>(f) * b->nnz;
        if (reuse && ref >= 0 &&
            refactor_one(lu, b->lus[ref], b->colptr.data(), b->rowind.data(), vals, b->q, reuse_tol, xr)) {
            ++b->refactored;
            singular_out[f] = 0;
            continue;
        }
        factor_one(lu, n, b->colptr.data(), b->rowind.data(), vals, b->q, tol, w, lnz, unz);
        if (lu.singular) {
            w.resize(n);  // workspace may hold partial state
        } else {
            lnz = lu.Li.size() + lu.Li.size() / 8;
            unz = lu.Ui.size() + lu.Ui.size() / 8;
            ref = f;
        }
        singular_out[f] = lu.singular ? 1 : 0;
    }
    return b;
}

// Fill-in statistics: total nonzeros of L and U over all systems.
int dfs_refactored(void* handle) { return static_cast<Batch*>(handle)->refactored; }

void dfs_stats(void* handle, long long* lnz, long long* unz) {
    auto* b = static_cast<Batch*>(handle);
    *lnz = 0; *unz = 0;
    for (auto& lu : b->lus) { *lnz += lu.Li.size(); *unz += lu.Ui.size(); }
}

// Solve op(A_f) X_f = B_f for every system f. rhs: (F, n, k) complex128, C order, overwritten
// with the solution. trans: 0 N, 1 T, 2 H. refine: number of iterative-refinement steps.
void dfs_solve(void* handle, int trans, int k, double* rhs, int refine) {
    auto* b = static_cast<Batch*>(handle);
    int n = b->n;
    std::vector<cplx> rb(n), x(n), t(n), r(n), d(n);
    cplx* data = reinterpret_cast<cplx*>(rhs);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (int f = 0; f < b->F; ++f) {
        const LU& lu = b->lus[f];
        const cplx* vals = b->values.data() + static_cast<size_t>(f) * b->nnz;
        cplx* block = data + static_cast<size_t>(f) * n * k;
        for (int c = 0; c < k; ++c) {
            if (lu.singular) {
                for (int i = 0; i < n; ++i) block[static_cast<size_t>(i) * k + c] = {nan, nan};
                continue;
            }
            for (int i = 0; i < n; ++i) rb[i] = block[static_cast<size_t>(i) * k + c];
            solve_one(lu, b->q, trans, rb.data(), x.data(), t.data());
            for (int s = 0; s < refine; ++s) {
                residual(n, b->colptr.data(), b->rowind.data(), vals, trans, rb.data(), x.data(), r.data());
                solve_one(lu, b->q, trans, r.data(), d.data(), t.data());
                for (int i = 0; i < n; ++i) x[i] = x[i] + d[i];
            }
            for (int i = 0; i < n; ++i) block[static_cast<size_t>(i) * k + c] = x[i];
        }
    }
}

void dfs_free(void* handle) { delete static_cast<Batch*>(handle); }

}  // extern "C"
