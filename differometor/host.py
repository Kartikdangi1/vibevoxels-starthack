"""
Host-side (NumPy/SciPy) execution of the sparse simulation with an exact adjoint.

Why: after switching the linear algebra to sparse LU, most of the remaining
first-run time was XLA compilation of the matrix-assembly graph (component
functions, scatters, conjugation and their reverse-mode graph). This module runs
the entire simulation on the host inside a single `jax.custom_vjp`, so XLA only
compiles two tiny callback stubs.

Exactness of derivatives: component and linking functions are *not* re-written.
Each one is traced once to a jaxpr (its forward, and its JAX VJP), and those
jaxprs are evaluated with NumPy by a small interpreter (`eval_jaxpr`). The
gradients therefore follow JAX's own differentiation rules and complex
conventions. The remaining glue (parameter insertion/expansion, carrier
scaling, conjugation, detector selections, sparse solves, quantum-noise
quadratic form) has a hand-written adjoint mirroring `simulate_dense` step by
step; it is validated against JAX autodiff of the in-graph implementation.

If any traced function uses a primitive the interpreter does not support,
`simulate` falls back to the in-graph sparse implementation.

JAX complex cotangent convention used throughout: for a real loss L and complex
z = x + iy, ct(z) = dL/dx - i dL/dy. Hence for y = a*z: ct(z) += a*ct(y);
y = conj(z): ct(z) += conj(ct(y)); y = Re(z): ct(z) += ct(y).
"""

from collections import OrderedDict
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax.extend import core as jcore

import scipy.sparse as sp
import scipy.sparse.linalg as spl

from differometor.components import FUNCTIONS, UNIT_VACUUM, H_PLANCK, F0
from differometor.build import LINKING_FUNCTIONS

NOISE_SCALE = 2 * UNIT_VACUUM * H_PLANCK * F0 * 0.25


# ======================================================================
# Minimal NumPy jaxpr interpreter
# ======================================================================

def _broadcast_in_dim(x, shape, broadcast_dimensions, **_):
    x = np.asarray(x)
    expanded = [1] * len(shape)
    for operand_axis, out_axis in enumerate(broadcast_dimensions):
        expanded[out_axis] = x.shape[operand_axis]
    return np.broadcast_to(x.reshape(expanded), shape)


def _slice(x, start_indices, limit_indices, strides=None, **_):
    strides = strides or (1,) * len(start_indices)
    return x[tuple(slice(s, l, st) for s, l, st in zip(start_indices, limit_indices, strides))]


def _pad(x, padding_value, padding_config, **_):
    x = np.asarray(x)
    pv = np.asarray(padding_value).item()
    # interior padding
    if any(interior for _, _, interior in padding_config):
        shape = [d + max(d - 1, 0) * interior for d, (_, _, interior) in zip(x.shape, padding_config)]
        out = np.full(shape, pv, dtype=x.dtype)
        out[tuple(slice(None, None, interior + 1) for _, _, interior in padding_config)] = x
        x = out
    # negative low/high = cropping
    crop = tuple(slice(max(-lo, 0), x.shape[i] - max(-hi, 0)) for i, (lo, hi, _) in enumerate(padding_config))
    x = x[crop]
    widths = [(max(lo, 0), max(hi, 0)) for lo, hi, _ in padding_config]
    return np.pad(x, widths, mode="constant", constant_values=pv)


def _convert(x, new_dtype, **_):
    x = np.asarray(x)
    new_dtype = np.dtype(new_dtype)
    if np.iscomplexobj(x) and not np.issubdtype(new_dtype, np.complexfloating):
        x = x.real
    return x.astype(new_dtype)


def _integer_div(x, y):
    if np.issubdtype(np.asarray(x).dtype, np.integer):
        return np.trunc(np.true_divide(x, y)).astype(np.asarray(x).dtype)
    return np.true_divide(x, y)


def _split(x, sizes, axis, **_):
    return np.split(x, np.cumsum(sizes)[:-1], axis=axis)


def _select_n(which, *cases):
    which = np.asarray(which)
    if which.dtype == np.bool_:
        which = which.astype(np.int32)
    return np.choose(which, np.broadcast_arrays(*cases)) if len(cases) > 2 else np.where(which, cases[1], cases[0])


def _dot_general(x, y, dimension_numbers, **_):
    (lc, rc), (lb, rb) = dimension_numbers
    if lb or rb:
        raise NotImplementedError
    return np.tensordot(x, y, axes=(lc, rc))


_PRIMITIVES = {
    "mul": lambda x, y: np.multiply(x, y),
    "add": lambda x, y: np.add(x, y),
    "add_any": lambda x, y: np.add(x, y),
    "sub": lambda x, y: np.subtract(x, y),
    "div": _integer_div,
    "neg": lambda x: np.negative(x),
    "exp": lambda x, **_: np.exp(x),
    "log": lambda x, **_: np.log(x),
    "sin": lambda x, **_: np.sin(x),
    "cos": lambda x, **_: np.cos(x),
    "tan": lambda x, **_: np.tan(x),
    "asin": lambda x, **_: np.arcsin(x),
    "acos": lambda x, **_: np.arccos(x),
    "atan": lambda x, **_: np.arctan(x),
    "atan2": lambda x, y, **_: np.arctan2(x, y),
    "sinh": lambda x, **_: np.sinh(x),
    "cosh": lambda x, **_: np.cosh(x),
    "tanh": lambda x, **_: np.tanh(x),
    "sqrt": lambda x, **_: np.sqrt(x),
    "rsqrt": lambda x, **_: 1.0 / np.sqrt(x),
    "square": lambda x, **_: np.square(x),
    "integer_pow": lambda x, y, **_: x ** y,
    "pow": lambda x, y, **_: np.power(x, y),
    "abs": lambda x: np.abs(x),
    "sign": lambda x: np.sign(x),
    "conj": lambda x, **_: np.conj(x),
    "real": lambda x: np.real(x),
    "imag": lambda x: np.imag(x),
    "complex": lambda x, y: x + 1j * y,
    "max": lambda x, y: np.maximum(x, y),
    "min": lambda x, y: np.minimum(x, y),
    "lt": lambda x, y: np.less(x, y),
    "le": lambda x, y: np.less_equal(x, y),
    "gt": lambda x, y: np.greater(x, y),
    "ge": lambda x, y: np.greater_equal(x, y),
    "eq": lambda x, y: np.equal(x, y),
    "ne": lambda x, y: np.not_equal(x, y),
    "and": lambda x, y: np.logical_and(x, y) if np.asarray(x).dtype == np.bool_ else np.bitwise_and(x, y),
    "or": lambda x, y: np.logical_or(x, y) if np.asarray(x).dtype == np.bool_ else np.bitwise_or(x, y),
    "not": lambda x: np.logical_not(x) if np.asarray(x).dtype == np.bool_ else np.invert(x),
    "select_n": _select_n,
    "broadcast_in_dim": _broadcast_in_dim,
    "reshape": lambda x, new_sizes, **_: np.reshape(x, new_sizes),
    "squeeze": lambda x, dimensions: np.squeeze(x, axis=tuple(dimensions)),
    "expand_dims": lambda x, dimensions: np.expand_dims(x, tuple(dimensions)),
    "slice": _slice,
    "pad": _pad,
    "concatenate": lambda *xs, dimension: np.concatenate(xs, axis=dimension),
    "split": _split,
    "transpose": lambda x, permutation: np.transpose(x, permutation),
    "reduce_sum": lambda x, axes, **_: np.sum(x, axis=tuple(axes)),
    "reduce_max": lambda x, axes, **_: np.max(x, axis=tuple(axes)),
    "reduce_min": lambda x, axes, **_: np.min(x, axis=tuple(axes)),
    "convert_element_type": _convert,
    "copy": lambda x: np.array(x),
    "copy_p": lambda x: np.array(x),
    "stop_gradient": lambda x: x,
    "dot_general": _dot_general,
}
_CALL_PRIMITIVES = {"pjit", "jit", "closed_call", "core_call", "custom_jvp_call", "custom_vjp_call", "remat", "checkpoint"}


def _sub_jaxpr(eqn):
    for key in ("jaxpr", "call_jaxpr", "fun_jaxpr"):
        if key in eqn.params:
            sub = eqn.params[key]
            return (sub.jaxpr, sub.consts) if hasattr(sub, "consts") else (sub, ())
    raise NotImplementedError(str(eqn.primitive))


# Structural primitives whose axis parameters shift when a leading batch axis is present.
def _batched_structural(name, x, params, batched_flags, batch_size):
    """Apply a structural primitive to operands that may carry a leading batch axis."""
    b = batched_flags[0]
    if name == "broadcast_in_dim":
        if not b:
            return _broadcast_in_dim(x[0], **params), False
        shape = (batch_size, *params["shape"])
        dims = (0, *[d + 1 for d in params["broadcast_dimensions"]])
        return _broadcast_in_dim(x[0], shape, dims), True
    if not any(batched_flags):
        return _PRIMITIVES[name](*x, **params), False
    if name == "slice":
        st = params.get("strides") or (1,) * len(params["start_indices"])
        return _slice(x[0], (0, *params["start_indices"]), (x[0].shape[0], *params["limit_indices"]), (1, *st)), True
    if name == "squeeze":
        return np.squeeze(x[0], axis=tuple(d + 1 for d in params["dimensions"])), True
    if name == "expand_dims":
        return np.expand_dims(x[0], tuple(d + 1 for d in params["dimensions"])), True
    if name == "reshape":
        if params.get("dimensions") is not None:
            raise NotImplementedError("reshape with dimensions")
        return np.reshape(x[0], (batch_size, *params["new_sizes"])), True
    if name == "pad":
        return _pad(x[0], x[1], ((0, 0, 0), *params["padding_config"])), True
    if name == "concatenate":
        parts = [xi if bi else np.broadcast_to(xi, (batch_size, *np.shape(xi))) for xi, bi in zip(x, batched_flags)]
        return np.concatenate(parts, axis=params["dimension"] + 1), True
    if name == "split":
        return _split(x[0], params["sizes"], params["axis"] + 1), True
    if name == "transpose":
        return np.transpose(x[0], (0, *[p + 1 for p in params["permutation"]])), True
    if name in ("reduce_sum", "reduce_max", "reduce_min"):
        reducer = {"reduce_sum": np.sum, "reduce_max": np.max, "reduce_min": np.min}[name]
        return reducer(x[0], axis=tuple(a + 1 for a in params["axes"])), True
    raise NotImplementedError(name)


_STRUCTURAL = {"broadcast_in_dim", "slice", "squeeze", "expand_dims", "reshape", "pad", "concatenate",
               "split", "transpose", "reduce_sum", "reduce_max", "reduce_min"}
# Primitives that are not elementwise and have no batched rule here.
_UNBATCHABLE = {"dot_general"}


def unsupported_primitives(jaxpr, found=None):
    """Return the set of primitive names in `jaxpr` the batched interpreter cannot run."""
    found = set() if found is None else found
    for eqn in jaxpr.eqns:
        name = str(eqn.primitive)
        if name in _CALL_PRIMITIVES:
            try:
                unsupported_primitives(_sub_jaxpr(eqn)[0], found)
            except NotImplementedError:
                found.add(name)
        elif name not in _PRIMITIVES or name in _UNBATCHABLE:
            found.add(name)
        elif name == "reshape" and eqn.params.get("dimensions") is not None:
            found.add("reshape(dimensions)")
    return found


def eval_jaxpr_batched(jaxpr, consts, args, batched, batch_size):
    """Evaluate an *unbatched* jaxpr on inputs that may carry a leading batch axis.

    Equivalent to evaluating `jax.vmap` of the function: elementwise primitives
    broadcast naturally (unbatched operands have the per-example shape), and
    structural primitives shift their axis parameters by one.
    Returns (outputs, batched_flags).
    """
    env, flags = {}, {}

    def read(v):
        if isinstance(v, jcore.Literal):
            return v.val, False
        return env[v], flags[v]

    for v, x in zip(jaxpr.constvars, consts):
        env[v], flags[v] = np.asarray(x), False
    for v, x, b in zip(jaxpr.invars, args, batched):
        env[v], flags[v] = x, b
    for eqn in jaxpr.eqns:
        values, bflags = zip(*[read(v) for v in eqn.invars]) if eqn.invars else ((), ())
        name = str(eqn.primitive)
        if name in _CALL_PRIMITIVES:
            sub, sub_consts = _sub_jaxpr(eqn)
            n_in = len(sub.invars)
            outputs, out_flags = eval_jaxpr_batched(sub, sub_consts, values[len(values) - n_in:],
                                                    bflags[len(bflags) - n_in:], batch_size)
        elif name in _STRUCTURAL:
            out, flag = _batched_structural(name, values, eqn.params, bflags, batch_size)
            outputs, out_flags = (out if eqn.primitive.multiple_results else [out]), None
            out_flags = [flag] * len(outputs)
        else:
            out = _PRIMITIVES[name](*values, **eqn.params)
            outputs = out if eqn.primitive.multiple_results else [out]
            out_flags = [any(bflags)] * len(outputs)
        for v, x, b in zip(eqn.outvars, outputs, out_flags):
            env[v], flags[v] = np.asarray(x), b
    results = [read(v) for v in jaxpr.outvars]
    return [r[0] for r in results], [r[1] for r in results]


def _dce(jaxpr, used):
    try:
        from jax._src.interpreters import partial_eval as pe
        return pe.dce_jaxpr(jaxpr, used)[0]
    except Exception:  # pragma: no cover - private API changed; keep the full jaxpr
        return None


class TracedFunction:
    """Forward and VJP jaxprs of a per-example function, evaluated batched with NumPy.

    One trace of x -> (f(x), vjp(f, x)(ct)) provides both programs; dead-code
    elimination splits off the forward-only part.
    """

    def __init__(self, function, in_shape, dtype):
        example = jax.ShapeDtypeStruct(in_shape, dtype)
        out = jax.eval_shape(function, example)
        self.out_shape, self.out_dtype = out.shape, out.dtype

        def both(x, ct):
            y, pullback = jax.vjp(function, x)
            return y, pullback(ct)[0]

        closed = jax.make_jaxpr(both)(example, jax.ShapeDtypeStruct(out.shape, out.dtype))
        self.consts = closed.consts
        self.backward = closed.jaxpr
        self.forward = _dce(closed.jaxpr, [True, False])
        self.forward_has_ct = True  # dce keeps invars; ct input is simply unused
        if self.forward is None:
            self.forward = closed.jaxpr
        self.unsupported = unsupported_primitives(self.backward)

    def _run(self, jaxpr, x, ct):
        x = np.asarray(x)
        batch = x.shape[0]
        if ct is None:
            ct = np.zeros(self.out_shape, np.dtype(self.out_dtype))
            ct_batched = False
        else:
            ct_batched = True
        outs, flags = eval_jaxpr_batched(jaxpr, self.consts, [x, ct], [True, ct_batched], batch)
        return [o if f else np.broadcast_to(o, (batch, *np.shape(o))) for o, f in zip(outs, flags)]

    def __call__(self, x):
        """Batched forward: x (B, *in_shape) -> (B, *out_shape)."""
        return self._run(self.forward, x, None)[0]

    def vjp(self, x, ct):
        """Batched VJP: x (B, *in_shape), ct (B, *out_shape) -> (B, *in_shape)."""
        return self._run(self.backward, x, np.asarray(ct))[1]


# ======================================================================
# Static plan (shapes and traced functions), built at JAX trace time
# ======================================================================

class Plan:
    """Everything shape-dependent the host program needs; created while tracing `simulate`."""

    def __init__(self, shapes, groups_meta, n_links):
        self.shapes = shapes
        self.traced = {}
        self.unsupported = set()
        for groups in groups_meta.values():
            for key in groups:
                fid = int(key[1:])
                if fid not in self.traced:
                    t = TracedFunction(FUNCTIONS[fid], (7,), jnp.complex128)
                    self.traced[fid] = t
                    self.unsupported |= t.unsupported
        self.links = []
        if n_links:
            for function in LINKING_FUNCTIONS:
                t = TracedFunction(function, (), jnp.complex128)
                self.links.append(t)
                self.unsupported |= t.unsupported

    @property
    def supported(self):
        return not self.unsupported


# ======================================================================
# Host program
# ======================================================================

_RESIDUALS = OrderedDict()
_RESIDUAL_SLOTS = 2
_TOKEN = [0]


def _expand(array, indices, values):
    """NumPy mirror of `expand_parameters`; also returns the source row of every output row."""
    reps = values.shape[1]
    out = np.tile(array, (reps, 1))
    source = np.tile(np.arange(array.shape[0]), reps)
    rows = np.repeat(np.arange(reps), len(indices))
    cols = np.tile(indices, reps)
    out[rows, cols] = values.T.reshape(-1)
    return out, source, (rows, cols)


def _expand_backward(ct_out, source, set_positions, n_source_rows):
    ct = np.array(ct_out)
    ct[set_positions] = 0
    ct_in = np.zeros((n_source_rows, ct.shape[1]), ct.dtype)
    np.add.at(ct_in, source, ct)
    return ct_in


def _assemble(plan, parameters, base, groups):
    """Evaluate grouped component functions into compressed values (R, nnz)."""
    rows_count = parameters.shape[0]
    values = np.broadcast_to(base, (rows_count, base.shape[0])).copy()
    saved = []
    for key, g in groups.items():
        fid = int(key[1:])
        inputs = parameters[:, g["inputs"]].reshape(-1, 7)  # (R*K, 7)
        traced = plan.traced[fid]
        out = traced(inputs).reshape(rows_count, g["inputs"].shape[0], -1)
        values[:, g["k"]] = out[:, g["rows"], g["slots"]]
        saved.append((key, inputs, out.shape))
    return values.astype(np.complex128), saved


def _assemble_backward(plan, ct_values, groups, saved, parameters_shape):
    """Adjoint of `_assemble` w.r.t. parameters (R, P). ct_values is consumed."""
    ct_parameters = np.zeros(parameters_shape, np.complex128)
    for (key, inputs, out_shape) in saved:
        g = groups[key]
        fid = int(key[1:])
        ct_out = np.zeros(out_shape, np.complex128)
        np.add.at(ct_out, (slice(None), g["rows"], g["slots"]), ct_values[:, g["k"]])
        ct_in = plan.traced[fid].vjp(inputs, ct_out.reshape(inputs.shape[0], -1))
        ct_in = np.asarray(ct_in).reshape(out_shape[0], g["inputs"].shape[0], 7)
        rows = np.arange(out_shape[0])[:, None, None]
        np.add.at(ct_parameters, (np.broadcast_to(rows, ct_in.shape), np.broadcast_to(g["inputs"][None], ct_in.shape)), ct_in)
    return ct_parameters


def _factor(values, rows, cols, n):
    return spl.splu(sp.csc_matrix((values, (rows, cols)), shape=(n, n)))


def _forward(plan, a):
    """Full forward pass. `a` holds NumPy arrays. Returns outputs and residuals."""
    r = {}
    p = np.asarray(a["optimized_parameters"])
    P = np.array(a["parameters"], dtype=np.complex128)
    P[0, a["optimized_parameter_indices"]] = p[a["optimized_value_indices"]]
    link_inputs = P[0, a["linked_indices"]]
    link_outputs = np.empty(len(a["linked_indices"]), np.complex128)
    for i, (fi, x) in enumerate(zip(a["linking_function_indices"], link_inputs)):
        link_outputs[i] = np.asarray(plan.links[int(fi)](np.array([x]))).reshape(-1)[0]
    P[0, a["indices_to_link"]] = link_outputs
    r["P0"], r["link_inputs"] = P.copy(), link_inputs
    qhd = np.exp(1j * (P[0, a["qhd_parameter_indices"]] * (np.pi / 180)))

    # carrier
    Pc, src_c, set_c = _expand(P, a["carrier_changing_parameter_indices"], a["carrier_changing_values"])
    cvals, saved_c = _assemble(plan, Pc, a["carrier_values"], a["carrier_groups"])
    n_c = a["carrier_rows"].shape[0]
    CN = a["carrier_size_marker"].shape[0]
    F_c = cvals.shape[0]
    cb = np.zeros((F_c, CN), np.complex128)
    cb[:, a["carrier_rhs_rows"]] = cvals[:, n_c:]
    lus_c = [_factor(cvals[f, :n_c], a["carrier_rows"], a["carrier_cols"], CN) for f in range(F_c)]
    carrier = np.stack([lu.solve(cb[f]) for f, lu in enumerate(lus_c)])  # (CV, CN)

    # signal
    Ps, src_s, set_s = _expand(Pc, a["signal_changing_parameter_indices"], a["signal_changing_values"])
    svals0, saved_s = _assemble(plan, Ps, a["signal_values"], a["signal_groups"])
    pk = a["signal_place_k"]
    sci = a["signal_carrier_indices"].reshape(-1)
    placed_before = svals0[:, pk].copy()
    svals = svals0
    svals[:, pk] = -placed_before * carrier[:, sci]
    mask = a["signal_conj_mask"]
    svals = np.where(mask, np.conj(svals), svals)
    SN = a["noise_selection_vectors"].shape[2]
    n_s = a["signal_rows"].shape[0]
    F = svals.shape[0]
    sb = np.zeros((F, SN), np.complex128)
    sb[:, a["signal_rhs_rows"]] = svals[:, n_s:]

    # detector selections
    nsv0 = np.array(a["noise_selection_vectors"], dtype=np.complex128)
    qpos = a["qhd_placing_indices"].reshape(-1, 3)
    nsv = nsv0.copy()
    if len(qpos):
        nsv[qpos[:, 0], qpos[:, 1], qpos[:, 2]] *= qhd
    CV = carrier.shape[0]
    nsv_t = np.tile(nsv, (1, CV, 1))  # (D, CV, SN)
    sel = nsv_t.copy()
    sel[:, :, :CN] = nsv_t[:, :, :CN] * carrier
    sel[:, :, CN:2 * CN] = np.conj(nsv_t[:, :, CN:2 * CN] * carrier)
    D = sel.shape[0]
    c = np.ascontiguousarray(np.transpose(np.broadcast_to(sel, (D, F, SN)), (1, 0, 2)))  # (F, D, SN)

    lus_s = [_factor(svals[f, :n_s], a["signal_rows"], a["signal_cols"], SN) for f in range(F)]
    signal = np.empty((F, SN), np.complex128)
    w = np.empty((F, D, SN), np.complex128)
    for f, lu in enumerate(lus_s):
        signal[f] = lu.solve(sb[f])
        w[f] = lu.solve(np.ascontiguousarray(c[f].T), trans="H").T

    # quantum noise
    nvals, saved_n = _assemble(plan, Ps, a["noise_values"], a["noise_groups"])
    nr, nc = a["noise_rows"], a["noise_cols"]
    wr_conj = np.conj(w[:, :, nr])
    wc = w[:, :, nc]
    q = np.sum(np.real(nvals[:, None, :] * wr_conj * wc), axis=2)  # (F, D)
    noise = np.sqrt(NOISE_SCALE * q.T)  # (D, F)

    r.update(Pc=Pc, src_c=src_c, set_c=set_c, saved_c=saved_c, lus_c=lus_c, carrier=carrier,
             Ps=Ps, src_s=src_s, set_s=set_s, saved_s=saved_s, placed_before=placed_before,
             svals=svals, nsv=nsv, nsv_t=nsv_t, qhd=qhd, lus_s=lus_s, signal=signal, w=w,
             nvals=nvals, saved_n=saved_n, wr_conj=wr_conj, wc=wc, noise=noise)
    return carrier.T, signal.T, noise, r


def _backward(plan, a, r, ct_carrier_T, ct_signal_T, ct_noise):
    """Adjoint of `_forward` w.r.t. optimized_parameters (real)."""
    carrier, w, noise = r["carrier"], r["w"], r["noise"]
    CV, CN = carrier.shape
    F, D, SN = w.shape
    ct_carrier = np.array(ct_carrier_T.T, dtype=np.complex128)
    ct_signal = np.asarray(ct_signal_T.T, dtype=np.complex128)
    ct_noise = np.asarray(ct_noise, dtype=np.float64).reshape(noise.shape)

    # quantum noise: noise = sqrt(s q^T)  -> ct(q) = ct(noise) * s / (2 noise)
    with np.errstate(divide="ignore", invalid="ignore"):
        ct_q = (ct_noise * NOISE_SCALE / (2 * noise)).T  # (F, D)
    ct_q = np.where(np.isfinite(ct_q), ct_q, 0.0) if not np.all(noise > 0) else ct_q
    nr, nc = a["noise_rows"], a["noise_cols"]
    nvals, wr_conj, wc = r["nvals"], r["wr_conj"], r["wc"]
    ct_s = ct_q[:, :, None]  # ct of the complex sum terms (real)
    ct_nvals = np.sum(ct_s * wr_conj * wc, axis=1)  # (F, nnzN)
    ct_w = np.zeros((F, D, SN), np.complex128)
    np.add.at(ct_w, (slice(None), slice(None), nr), np.conj(ct_s * nvals[:, None, :] * wc))
    np.add.at(ct_w, (slice(None), slice(None), nc), ct_s * nvals[:, None, :] * wr_conj)

    # sparse solves (signal x and noise weights w share one factorisation)
    svals = r["svals"]
    n_s = a["signal_rows"].shape[0]
    sr, sc = a["signal_rows"], a["signal_cols"]
    ct_svals = np.zeros(svals.shape, np.complex128)
    ct_c = np.empty((F, D, SN), np.complex128)
    signal = r["signal"]
    for f, lu in enumerate(r["lus_s"]):
        bb = lu.solve(ct_signal[f], trans="T")
        y = lu.solve(np.ascontiguousarray(np.conj(ct_w[f]).T)).T  # (D, SN)
        ct_c[f] = np.conj(y)
        ct_svals[f, :n_s] = -bb[sr] * signal[f][sc] - np.sum(np.conj(w[f][:, sr]) * y[:, sc], axis=0)
        ct_svals[f, n_s:] = bb[a["signal_rhs_rows"]]

    # detector selections -> carrier (and qhd phases)
    ct_sel_b = np.transpose(ct_c, (1, 0, 2))  # (D, F, SN)
    ct_sel = ct_sel_b.sum(axis=1, keepdims=True) if CV == 1 and F != 1 else ct_sel_b  # broadcast_to adjoint
    nsv_t = r["nsv_t"]
    ct_carrier += np.sum(nsv_t[:, :, :CN] * ct_sel[:, :, :CN], axis=0)
    ct_lower = np.conj(ct_sel[:, :, CN:2 * CN])
    ct_carrier += np.sum(nsv_t[:, :, CN:2 * CN] * ct_lower, axis=0)
    qpos = a["qhd_placing_indices"].reshape(-1, 3)
    ct_P0 = np.zeros(r["P0"].shape, np.complex128)
    if len(qpos):
        ct_nsv_t = ct_sel.copy()
        ct_nsv_t[:, :, :CN] = carrier * ct_sel[:, :, :CN]
        ct_nsv_t[:, :, CN:2 * CN] = carrier * ct_lower
        ct_nsv = ct_nsv_t.sum(axis=1, keepdims=True)  # tile adjoint (second axis was 1)
        nsv0 = np.asarray(a["noise_selection_vectors"], dtype=np.complex128)
        ct_qhd = ct_nsv[qpos[:, 0], qpos[:, 1], qpos[:, 2]] * nsv0[qpos[:, 0], qpos[:, 1], qpos[:, 2]]
        dq = 1j * np.pi / 180 * r["qhd"]
        np.add.at(ct_P0, (0, a["qhd_parameter_indices"]), ct_qhd * dq)

    # conjugation mask and carrier scaling of the signal matrix
    mask = a["signal_conj_mask"]
    ct_svals = np.where(mask, np.conj(ct_svals), ct_svals)
    pk = a["signal_place_k"]
    sci = a["signal_carrier_indices"].reshape(-1)
    ct_placed = ct_svals[:, pk].copy()
    ct_svals[:, pk] = -carrier[:, sci] * ct_placed
    ct_carrier_terms = -r["placed_before"] * ct_placed  # (F, M)
    ct_carrier_terms = ct_carrier_terms.sum(axis=0, keepdims=True) if CV == 1 and F != 1 else ct_carrier_terms
    for row in range(ct_carrier_terms.shape[0]):
        np.add.at(ct_carrier[row], sci, ct_carrier_terms[row])

    # signal + noise assembly -> signal-expanded parameters
    ct_Ps = _assemble_backward(plan, ct_svals, a["signal_groups"], r["saved_s"], r["Ps"].shape)
    ct_Ps += _assemble_backward(plan, ct_nvals, a["noise_groups"], r["saved_n"], r["Ps"].shape)
    ct_Pc = _expand_backward(ct_Ps, r["src_s"], r["set_s"], r["Pc"].shape[0])

    # carrier solve
    cvals_ct = np.zeros((CV, a["carrier_values"].shape[0]), np.complex128)
    n_c = a["carrier_rows"].shape[0]
    crr, ccc = a["carrier_rows"], a["carrier_cols"]
    for f, lu in enumerate(r["lus_c"]):
        bb = lu.solve(ct_carrier[f], trans="T")
        cvals_ct[f, :n_c] = -bb[crr] * carrier[f][ccc]
        cvals_ct[f, n_c:] = bb[a["carrier_rhs_rows"]]
    ct_Pc += _assemble_backward(plan, cvals_ct, a["carrier_groups"], r["saved_c"], r["Pc"].shape)
    ct_P0 += _expand_backward(ct_Pc, r["src_c"], r["set_c"], r["P0"].shape[0])

    # linking (outputs overwrite, inputs read the row before linking)
    link_idx = a["indices_to_link"]
    if len(link_idx):
        ct_links = ct_P0[0, link_idx].copy()
        ct_P0[0, link_idx] = 0
        for i, (fi, x) in enumerate(zip(a["linking_function_indices"], r["link_inputs"])):
            ct_x = np.asarray(plan.links[int(fi)].vjp(np.array([x]), np.array([ct_links[i]]))).reshape(-1)[0]
            ct_P0[0, a["linked_indices"][i]] += ct_x

    # optimized parameter insertion (real inputs: real part of the cotangent)
    p = np.asarray(a["optimized_parameters"])
    ct_p = np.zeros(p.shape, np.float64)
    np.add.at(ct_p, a["optimized_value_indices"], np.real(ct_P0[0, a["optimized_parameter_indices"]]))
    return ct_p


# ======================================================================
# JAX glue
# ======================================================================

def _to_numpy(tree):
    return jax.tree_util.tree_map(np.asarray, tree)


def _host_forward(plan, flat_arrays, treedef):
    a = _to_numpy(jax.tree_util.tree_unflatten(treedef, flat_arrays))
    carrier_T, signal_T, noise, residuals = _forward(plan, a)
    _TOKEN[0] += 1
    token = _TOKEN[0]
    _RESIDUALS[token] = (residuals, a["optimized_parameters"].copy())
    while len(_RESIDUALS) > _RESIDUAL_SLOTS:
        _RESIDUALS.popitem(last=False)
    return carrier_T, signal_T, noise.reshape(plan.shapes["noise"]), np.int64(token)


def _host_backward(plan, token, flat_arrays, treedef, ct_carrier, ct_signal, ct_noise):
    a = _to_numpy(jax.tree_util.tree_unflatten(treedef, flat_arrays))
    cached = _RESIDUALS.get(int(token))
    if cached is not None and np.array_equal(cached[1], a["optimized_parameters"]):
        residuals = cached[0]
    else:  # recompute deterministically
        residuals = _forward(plan, a)[3]
    return _backward(plan, a, residuals, np.asarray(ct_carrier), np.asarray(ct_signal), np.asarray(ct_noise))


def make_host_simulate(arrays):
    """Return (simulate_fn, plan) for these (traced) arrays, or (None, plan) if unsupported."""
    carrier_rows_count = arrays["carrier_changing_values"].shape[1]
    signal_rows_count = arrays["signal_changing_values"].shape[1] * carrier_rows_count
    groups_meta = {
        "carrier": {k: (g["inputs"].shape[0], carrier_rows_count) for k, g in arrays["carrier_groups"].items()},
        "signal": {k: (g["inputs"].shape[0], signal_rows_count) for k, g in arrays["signal_groups"].items()},
        "noise": {k: (g["inputs"].shape[0], signal_rows_count) for k, g in arrays["noise_groups"].items()},
    }
    CN = arrays["carrier_size_marker"].shape[0]
    SN = arrays["noise_selection_vectors"].shape[2]
    D = arrays["noise_selection_vectors"].shape[0]
    F = signal_rows_count
    shapes = dict(carrier=(CN, carrier_rows_count), signal=(SN, F), noise=jax.eval_shape(
        lambda: jnp.zeros((D, F)).squeeze()).shape)
    plan = Plan(shapes, groups_meta, arrays["linked_indices"].shape[0])
    if not plan.supported:
        return None, plan

    out_shapes = (jax.ShapeDtypeStruct(shapes["carrier"], jnp.complex128),
                  jax.ShapeDtypeStruct(shapes["signal"], jnp.complex128),
                  jax.ShapeDtypeStruct(shapes["noise"], jnp.float64),
                  jax.ShapeDtypeStruct((), jnp.int64))

    def run_forward(optimized_parameters, rest):
        tree = dict(rest, optimized_parameters=optimized_parameters)
        flat, treedef = jax.tree_util.tree_flatten(tree)
        return jax.pure_callback(partial(_host_forward_flat, plan, treedef), out_shapes, *flat,
                                 vmap_method="sequential")

    @jax.custom_vjp
    def host_simulate(optimized_parameters, rest):
        return run_forward(optimized_parameters, rest)[:3]

    def fwd(optimized_parameters, rest):
        carrier_T, signal_T, noise, token = run_forward(optimized_parameters, rest)
        return (carrier_T, signal_T, noise), (token, optimized_parameters, rest)

    def bwd(residuals, cotangents):
        token, optimized_parameters, rest = residuals
        tree = dict(rest, optimized_parameters=optimized_parameters)
        flat, treedef = jax.tree_util.tree_flatten(tree)
        ct_p = jax.pure_callback(partial(_host_backward_flat, plan, treedef, len(flat)),
                                 jax.ShapeDtypeStruct(optimized_parameters.shape, jnp.float64),
                                 token, *flat, *cotangents, vmap_method="sequential")
        zeros = jax.tree_util.tree_map(_zero_cotangent, rest)
        return ct_p.astype(optimized_parameters.dtype), zeros

    host_simulate.defvjp(fwd, bwd)
    return host_simulate, plan


def _zero_cotangent(x):
    x = jnp.asarray(x)
    if jnp.issubdtype(x.dtype, jnp.inexact):
        return jnp.zeros_like(x)
    return np.zeros(x.shape, dtype=jax.dtypes.float0)


def _host_forward_flat(plan, treedef, *flat):
    return _host_forward(plan, flat, treedef)


def _host_backward_flat(plan, treedef, n_flat, token, *rest):
    flat, cts = rest[:n_flat], rest[n_flat:]
    return _host_backward(plan, token, flat, treedef, *cts)
