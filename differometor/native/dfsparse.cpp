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
#include <memory>
#include <cstdlib>
#include <mutex>
#include <new>
#ifdef __linux__
#include <sys/mman.h>
#endif

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

// ---------------------------------------------------------------- batched (SIMD across systems)
// Wang et al. (arXiv:2101.02270) batch many same-pattern systems so that every sparse update is
// one vector instruction across systems. Here up to LANES frequencies are refactorised together
// with the pivot order and L/U pattern of a pivoted reference system (KLU-style reuse), and
// are also *solved* together: values live lane-interleaved (entry-major, system-minor, real and
// imaginary parts separately) and are never unpacked, so a group shares one copy of the pattern.
// Per lane the arithmetic is that of refactor_one / solve_one, except that divisions by a pivot
// are multiplications by its (scaled) reciprocal.
constexpr int LANES = 8;

// 64-byte aligned storage for lane-interleaved vectors, so that a vector of LANES doubles never
// straddles two cache lines.
template <class T>
struct AlignedAllocator {
    using value_type = T;
    AlignedAllocator() = default;
    template <class U>
    AlignedAllocator(const AlignedAllocator<U>&) {}
    T* allocate(size_t n) {
        const size_t bytes = (n * sizeof(T) + 63) / 64 * 64;
        void* p = std::aligned_alloc(64, bytes ? bytes : 64);
        if (!p) throw std::bad_alloc();
        return static_cast<T*>(p);
    }
    void deallocate(T* p, size_t) { std::free(p); }
    template <class U>
    bool operator==(const AlignedAllocator<U>&) const { return true; }
    template <class U>
    bool operator!=(const AlignedAllocator<U>&) const { return false; }
};
using LaneVec = std::vector<double, AlignedAllocator<double>>;


struct Buf {  // uninitialised double array, owned or a slice of the batch's arena
    std::unique_ptr<double[]> own;
    double* ptr = nullptr;
    size_t size = 0;
    void alloc(size_t n) { if (n > size) { own.reset(new double[n]); ptr = own.get(); size = n; } }
    void attach(double* p, size_t n) { own.reset(); ptr = p; size = n; }
    double* data() { return ptr; }
    const double* data() const { return ptr; }
};

// One allocation for all groups' values. Fresh memory is first touched while the values are
// written; with transparent huge pages (madvise mode on Linux) that is a few 2 MB faults
// instead of thousands of 4 KB ones.
// The most recently released arena is kept for the next factorisation (repeated evaluations
// then reuse memory that is already mapped); every value is written before it is read.
std::mutex spare_mutex;
void* spare_arena = nullptr;
size_t spare_bytes = 0;

struct Arena {
    double* base = nullptr;
    size_t size = 0, used = 0, bytes = 0;
    void reserve(size_t doubles) {
        const size_t huge = size_t(1) << 21;
        const size_t want = (doubles * sizeof(double) + huge - 1) / huge * huge;
        void* p = nullptr;
        {
            std::lock_guard<std::mutex> lock(spare_mutex);
            if (spare_arena && spare_bytes >= want) {
                p = spare_arena;
                bytes = spare_bytes;
                spare_arena = nullptr;
            }
        }
        if (!p) {
            p = std::aligned_alloc(huge, want);
            if (!p) return;
            bytes = want;
#ifdef __linux__
            madvise(p, bytes, MADV_HUGEPAGE);
#endif
        }
        base = static_cast<double*>(p);
        size = bytes / sizeof(double);
        used = 0;
    }
    double* take(size_t doubles) {
        doubles = (doubles + 7) / 8 * 8;
        if (!base || used + doubles > size) return nullptr;
        double* p = base + used;
        used += doubles;
        return p;
    }
    ~Arena() {
        if (!base) return;
        void* release = base;
        {
            std::lock_guard<std::mutex> lock(spare_mutex);
            if (!spare_arena || spare_bytes < bytes) {
                std::swap(release, spare_arena);
                spare_bytes = bytes;
            }
        }
        std::free(release);  // the smaller of the two, or nothing
    }
};

struct Group {
    int ref = -1;            // system whose LU provides the pattern and pivot order
    int first = 0, lanes = 0;
    bool ok[LANES] = {};     // lane passed the pivot check (else that system has its own LU)
    Buf lr, li, ur, ui;      // L and U values, (nnz, LANES)
    Buf dr, di;              // reciprocal of the U diagonal, (n, LANES)
    Buf ar, ai;              // the systems' A values, (nnz(A), LANES), for refinement residuals
};

inline void recip(double re, double im, double& rr, double& ri) {
    const double s = std::fmax(std::fabs(re), std::fabs(im));
    const double a = re / s, b = im / s;
    const double den = s * (a * a + b * b);
    rr = a / den;
    ri = -b / den;
}

// ---------------------------------------------------------------- lane kernels per instruction set
// The kernels are compiled for the baseline target and, on x86-64 with GCC/Clang, also for AVX2+FMA
// and AVX-512F; the best one the CPU supports is chosen at run time. FMA contraction is off
// (build.py: -ffp-contract=off), so every variant computes bit-identical results.
// DIFFEROMETOR_NATIVE_ISA=generic|avx2|avx512 forces one (if supported).
#define DFS_CMUL_FMA 0
namespace lanes_generic {
#include "dfsparse_lanes.h"
}
#undef DFS_CMUL_FMA
#define DFS_CMUL_FMA 1

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#define DFS_X86_DISPATCH 1
#if defined(__clang__)
#pragma clang attribute push(__attribute__((target("avx2,fma"))), apply_to = function)
#else
#pragma GCC push_options
#pragma GCC target("avx2,fma")
#endif
namespace lanes_avx2 {
#include "dfsparse_lanes.h"
}
#if defined(__clang__)
#pragma clang attribute pop
#pragma clang attribute push(__attribute__((target("avx512f,avx2,fma"))), apply_to = function)
#else
#pragma GCC pop_options
#pragma GCC push_options
#pragma GCC target("avx512f,avx2,fma")
#endif
namespace lanes_avx512 {
#include "dfsparse_lanes.h"
}
#if defined(__clang__)
#pragma clang attribute pop
#else
#pragma GCC pop_options
#endif
#endif

struct LaneKernels {
    decltype(&lanes_generic::refactor_lanes) refactor;
    decltype(&lanes_generic::solve_lanes) solve;
    decltype(&lanes_generic::residual_lanes) residual;
    decltype(&lanes_generic::adjoint_entries) adjoint;
    int level;  // 0 generic, 1 avx2, 2 avx512
};

#ifdef DFS_X86_DISPATCH
// CPUID/XGETBV directly (not __builtin_cpu_supports, which some toolchains, e.g. zig's, leave
// unimplemented). A feature counts only if the OS also saves the corresponding register state.
void cpuid(unsigned leaf, unsigned sub, unsigned r[4]) {
    __asm__ volatile("cpuid" : "=a"(r[0]), "=b"(r[1]), "=c"(r[2]), "=d"(r[3]) : "a"(leaf), "c"(sub));
}
int x86_level() {  // 0 baseline, 1 AVX2+FMA, 2 AVX-512F
    unsigned r[4];
    cpuid(0, 0, r);
    const unsigned max_leaf = r[0];
    if (max_leaf < 7) return 0;
    cpuid(1, 0, r);
    const bool osxsave = (r[2] >> 27) & 1, avx = (r[2] >> 28) & 1, fma = (r[2] >> 12) & 1;
    if (!(osxsave && avx && fma)) return 0;
    unsigned lo, hi;
    __asm__ volatile("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    const unsigned long long xcr0 = (static_cast<unsigned long long>(hi) << 32) | lo;
    if ((xcr0 & 0x6) != 0x6) return 0;  // SSE and AVX state
    cpuid(7, 0, r);
    const bool avx2 = (r[1] >> 5) & 1, avx512f = (r[1] >> 16) & 1;
    if (!avx2) return 0;
    return (avx512f && (xcr0 & 0xE6) == 0xE6) ? 2 : 1;  // + opmask, ZMM_Hi256, Hi16_ZMM state
}
#endif

LaneKernels choose_kernels() {
    const LaneKernels generic = {lanes_generic::refactor_lanes, lanes_generic::solve_lanes,
                                 lanes_generic::residual_lanes, lanes_generic::adjoint_entries, 0};
#ifdef DFS_X86_DISPATCH
    const char* want = std::getenv("DIFFEROMETOR_NATIVE_ISA");
    const bool any = !want || !*want;
    const int level = x86_level();
    const bool avx2 = level >= 1;
    const bool avx512 = level >= 2;
    if (avx512 && (any || std::strcmp(want, "avx512") == 0))
        return {lanes_avx512::refactor_lanes, lanes_avx512::solve_lanes, lanes_avx512::residual_lanes,
                lanes_avx512::adjoint_entries, 2};
    if (avx2 && (any || std::strcmp(want, "avx2") == 0 || std::strcmp(want, "avx512") == 0))
        return {lanes_avx2::refactor_lanes, lanes_avx2::solve_lanes, lanes_avx2::residual_lanes,
                lanes_avx2::adjoint_entries, 1};
#endif
    return generic;
}

const LaneKernels& kernels() {
    static const LaneKernels k = choose_kernels();
    return k;
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
    std::vector<LU> lus;              // systems with their own factorisation
    std::vector<Group> groups;        // systems refactorised together (SIMD across systems)
    std::vector<int> group_of;        // group index of system f, or -1 if lus[f] holds its LU
    Arena arena;                      // storage for the groups' values
    int refactored = 0;
};

}  // namespace

extern "C" {

int dfs_version() { return 2; }

// Instruction set of the lane kernels in use: 0 generic, 1 AVX2+FMA, 2 AVX-512F.
int dfs_isa() { return kernels().level; }

// See adjoint_entries in dfsparse_lanes.h.
void dfs_adjoint_entries(int F, int nnz, int n, int D, const long long* sr, const long long* sc, const double* bb,
                         const double* x, const double* w, const double* y, double* out, long long out_stride) {
    kernels().adjoint(F, nnz, n, D, sr, sc, bb, x, w, y, out, out_stride);
}

// Factorise F systems sharing one CSC pattern. values: complex128, system f's entries start at
// values + 2 * f * stride (stride in complex elements). Entry p of the CSC order is perm[p] of
// a system's entries, or entry p itself if perm is null.
// order: 0 = natural, 1 = minimum degree on A + A^T, 2 = minimum degree on A^T A. tol: diagonal pivot threshold (1 = partial pivoting).
// singular_out[f] = 1 if system f is singular. Returns an opaque handle.
// reuse: 1 = refactorise systems 1..F-1 with the previous pivot order when every reused pivot is
// at least reuse_tol times the largest candidate in its column (else pivot that system afresh);
// 2 = the same, LANES systems at a time with SIMD across systems (refactor_lanes).
void* dfs_factor(int n, int F, const int* colptr, const int* rowind, const double* values, long long stride,
                 const int* perm, int order, double tol, int reuse, double reuse_tol, int* singular_out) {
    auto* b = new Batch;
    b->n = n;
    b->F = F;
    b->nnz = colptr[n];
    b->colptr.assign(colptr, colptr + n + 1);
    b->rowind.assign(rowind, rowind + b->nnz);
    b->values.resize(static_cast<size_t>(F) * b->nnz);
    for (int f = 0; f < F; ++f) {  // gather into CSC order (one pass; no copy on the Python side)
        const cplx* src = reinterpret_cast<const cplx*>(values) + static_cast<size_t>(f) * stride;
        cplx* dst = b->values.data() + static_cast<size_t>(f) * b->nnz;
        if (perm) {
            for (int p = 0; p < b->nnz; ++p) dst[p] = src[perm[p]];
        } else {
            std::memcpy(dst, src, sizeof(cplx) * b->nnz);
        }
    }
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
    auto pivot_afresh = [&](int f, bool update_ref = true) {
        LU& lu = b->lus[f];
        factor_one(lu, n, b->colptr.data(), b->rowind.data(), b->values.data() + static_cast<size_t>(f) * b->nnz,
                   b->q, tol, w, lnz, unz);
        if (lu.singular) {
            w.resize(n);  // workspace may hold partial state
        } else {
            lnz = lu.Li.size() + lu.Li.size() / 8;
            unz = lu.Ui.size() + lu.Ui.size() / 8;
            if (update_ref) ref = f;
        }
        singular_out[f] = lu.singular ? 1 : 0;
    };
    LaneVec lane_xr, lane_xi;
    b->group_of.assign(F, -1);
    b->groups.reserve(F / LANES + 1);
    int f = 0;
    while (f < F) {
        if (!reuse || ref < 0) {
            pivot_afresh(f++);
            continue;
        }
        if (reuse == 2 && F - f >= 2) {  // batched: up to LANES systems per pass
            const int lanes = std::min(LANES, F - f);
            b->groups.emplace_back();
            Group& g = b->groups.back();
            g.ref = ref; g.first = f; g.lanes = lanes;
            {
                const LU& r = b->lus[ref];
                const size_t W = LANES, nl = r.Li.size(), nu = r.Ui.size(), nn = n, na = b->nnz;
                if (!b->arena.base) {
                    const size_t per_group = W * (2 * nl + 2 * nu + 2 * nn + 2 * na) + 64;
                    b->arena.reserve(per_group * static_cast<size_t>((F - f + LANES - 1) / LANES));
                }
                const size_t sizes[8] = {nl * W, nl * W, nu * W, nu * W, nn * W, nn * W, na * W, na * W};
                Buf* bufs[8] = {&g.lr, &g.li, &g.ur, &g.ui, &g.dr, &g.di, &g.ar, &g.ai};
                for (int i = 0; i < 8; ++i)
                    if (double* p = b->arena.take(sizes[i])) bufs[i]->attach(p, sizes[i]);
            }
            const cplx* vals[LANES];
            for (int l = 0; l < LANES; ++l)
                vals[l] = b->values.data() + static_cast<size_t>(f + (l < lanes ? l : 0)) * b->nnz;
            kernels().refactor(b->lus[ref], b->colptr.data(), b->rowind.data(), vals, b->q, reuse_tol, g, lane_xr,
                           lane_xi);
            const int gi = static_cast<int>(b->groups.size()) - 1;
            int failed = 0, last_fresh = -1;
            for (int l = 0; l < lanes; ++l) {
                if (g.ok[l]) {
                    b->group_of[f + l] = gi;
                    ++b->refactored;
                    singular_out[f + l] = 0;
                } else {
                    // this system gets its own pivoted LU; an isolated outlier does not become the
                    // reference for later groups, but if most lanes failed the pivot order has drifted
                    pivot_afresh(f + l, false);
                    ++failed;
                    if (!b->lus[f + l].singular) last_fresh = f + l;
                }
            }
            if (2 * failed > lanes && last_fresh >= 0) ref = last_fresh;
            f += lanes;
            continue;
        }
        LU& lu = b->lus[f];
        const cplx* vals = b->values.data() + static_cast<size_t>(f) * b->nnz;
        if (refactor_one(lu, b->lus[ref], b->colptr.data(), b->rowind.data(), vals, b->q, reuse_tol, xr)) {
            ++b->refactored;
            singular_out[f] = 0;
            ++f;
            continue;
        }
        pivot_afresh(f++);
    }
    return b;
}

// Fill-in statistics: total nonzeros of L and U over all systems.
int dfs_refactored(void* handle) { return static_cast<Batch*>(handle)->refactored; }

void dfs_stats(void* handle, long long* lnz, long long* unz) {
    auto* b = static_cast<Batch*>(handle);
    *lnz = 0; *unz = 0;
    for (int f = 0; f < b->F; ++f) {
        const LU& lu = b->group_of[f] >= 0 ? b->lus[b->groups[b->group_of[f]].ref] : b->lus[f];
        *lnz += lu.Li.size(); *unz += lu.Ui.size();
    }
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
        if (b->group_of[f] >= 0) continue;  // solved with its group below
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
    if (b->groups.empty()) return;
    constexpr int W = LANES;
    const size_t len = static_cast<size_t>(n) * W;
    LaneVec buf(10 * len);
    double *br = buf.data(), *bi = br + len, *xr = bi + len, *xi = xr + len, *tr = xi + len, *ti = tr + len,
           *rr = ti + len, *ri = rr + len, *dr = ri + len, *di = dr + len;
    for (const Group& g : b->groups) {
        const LU& ref = b->lus[g.ref];
        for (int c = 0; c < k; ++c) {
            for (int i = 0; i < n; ++i)
                for (int l = 0; l < W; ++l) {
                    const cplx v = (l < g.lanes && g.ok[l])
                                       ? data[(static_cast<size_t>(g.first + l) * n + i) * k + c]
                                       : cplx{0, 0};
                    br[static_cast<size_t>(i) * W + l] = v.re;
                    bi[static_cast<size_t>(i) * W + l] = v.im;
                }
            kernels().solve(ref, g, b->q, trans, br, bi, xr, xi, tr, ti);
            for (int s = 0; s < refine; ++s) {
                kernels().residual(n, b->colptr.data(), b->rowind.data(), g, trans, br, bi, xr, xi, rr, ri);
                kernels().solve(ref, g, b->q, trans, rr, ri, dr, di, tr, ti);
                for (size_t i = 0; i < len; ++i) { xr[i] += dr[i]; xi[i] += di[i]; }
            }
            for (int l = 0; l < g.lanes; ++l) {
                if (!g.ok[l]) continue;
                for (int i = 0; i < n; ++i)
                    data[(static_cast<size_t>(g.first + l) * n + i) * k + c] = {xr[static_cast<size_t>(i) * W + l],
                                                                                xi[static_cast<size_t>(i) * W + l]};
            }
        }
    }
}

void dfs_free(void* handle) { delete static_cast<Batch*>(handle); }

}  // extern "C"
