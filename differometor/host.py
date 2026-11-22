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

import os
import sys
from collections import OrderedDict
from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
from jax.extend import core as jcore

from differometor.components import FUNCTIONS, UNIT_VACUUM, H_PLANCK, F0
from differometor.build import LINKING_FUNCTIONS
from differometor.native_kernels import NativeFunction, native_function, native_link
from differometor.simulate import gc_paused
from differometor.native import NativeBatchedLU

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


def _integer_div(x, y, **_):
    if np.issubdtype(np.asarray(x).dtype, np.integer):
        return np.trunc(np.true_divide(x, y)).astype(np.asarray(x).dtype)
    return np.true_divide(x, y)


def _split(x, sizes, axis, **_):
    return np.split(x, np.cumsum(sizes)[:-1], axis=axis)


def _select_n(which, *cases, **_):
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
    "mul": lambda x, y, **_: np.multiply(x, y),
    "add": lambda x, y, **_: np.add(x, y),
    "add_any": lambda x, y, **_: np.add(x, y),
    "sub": lambda x, y, **_: np.subtract(x, y),
    "div": _integer_div,
    "neg": lambda x, **_: np.negative(x),
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
    "one_minus_square": lambda x, **_: 1 - np.square(x),
    "integer_pow": lambda x, y, **_: x ** y,
    "pow": lambda x, y, **_: np.power(x, y),
    "abs": lambda x, **_: np.abs(x),
    "sign": lambda x, **_: np.sign(x),
    "conj": lambda x, **_: np.conj(x),
    "real": lambda x, **_: np.real(x),
    "imag": lambda x, **_: np.imag(x),
    "complex": lambda x, y, **_: x + 1j * y,
    "max": lambda x, y, **_: np.maximum(x, y),
    "min": lambda x, y, **_: np.minimum(x, y),
    "lt": lambda x, y, **_: np.less(x, y),
    "le": lambda x, y, **_: np.less_equal(x, y),
    "gt": lambda x, y, **_: np.greater(x, y),
    "ge": lambda x, y, **_: np.greater_equal(x, y),
    "eq": lambda x, y, **_: np.equal(x, y),
    "ne": lambda x, y, **_: np.not_equal(x, y),
    "and": lambda x, y, **_: np.logical_and(x, y) if np.asarray(x).dtype == np.bool_ else np.bitwise_and(x, y),
    "or": lambda x, y, **_: np.logical_or(x, y) if np.asarray(x).dtype == np.bool_ else np.bitwise_or(x, y),
    "not": lambda x, **_: np.logical_not(x) if np.asarray(x).dtype == np.bool_ else np.invert(x),
    "select_n": _select_n,
    "broadcast_in_dim": _broadcast_in_dim,
    "reshape": lambda x, new_sizes, **_: np.reshape(x, new_sizes),
    "squeeze": lambda x, dimensions, **_: np.squeeze(x, axis=tuple(dimensions)),
    "expand_dims": lambda x, dimensions, **_: np.expand_dims(x, tuple(dimensions)),
    "slice": _slice,
    "pad": _pad,
    "concatenate": lambda *xs, dimension, **_: np.concatenate(xs, axis=dimension),
    "split": _split,
    "transpose": lambda x, permutation, **_: np.transpose(x, permutation),
    "reduce_sum": lambda x, axes, **_: np.sum(x, axis=tuple(axes)),
    "reduce_max": lambda x, axes, **_: np.max(x, axis=tuple(axes)),
    "reduce_min": lambda x, axes, **_: np.min(x, axis=tuple(axes)),
    "convert_element_type": _convert,
    "copy": lambda x, **_: np.array(x),
    "copy_p": lambda x, **_: np.array(x),
    "stop_gradient": lambda x, **_: x,
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


def _eval_eqn(eqn, values, bflags, batch_size):
    """Evaluate one equation on (possibly batched) NumPy values; returns (outputs, flags)."""
    name = str(eqn.primitive)
    if name in _CALL_PRIMITIVES:
        sub, sub_consts = _sub_jaxpr(eqn)
        n_in = len(sub.invars)
        return eval_jaxpr_batched(sub, sub_consts, values[len(values) - n_in:],
                                  bflags[len(bflags) - n_in:], batch_size)
    if name in _STRUCTURAL:
        out, flag = _batched_structural(name, values, eqn.params, bflags, batch_size)
        outputs = out if eqn.primitive.multiple_results else [out]
        return [np.asarray(o) for o in outputs], [flag] * len(outputs)
    if any(bflags):
        # Elementwise primitives may mix per-example ranks (e.g. mul c128[4] c128[]).
        # Right-aligned NumPy broadcasting is only correct per example, so batched
        # operands get singleton axes between the batch axis and their own shape.
        rank = max(len(v.aval.shape) for v in eqn.invars)
        values = [_align(x, b, v.aval.shape, rank) for x, b, v in zip(values, bflags, eqn.invars)]
    out = _PRIMITIVES[name](*values, **eqn.params)
    outputs = out if eqn.primitive.multiple_results else [out]
    return [np.asarray(o) for o in outputs], [any(bflags)] * len(outputs)


def _align(x, batched, shape, rank):
    """Reshape a batched value (B, *shape) to (B, 1, ..., 1, *shape) with `rank` per-example dims."""
    if not batched or len(shape) == rank:
        return x
    x = np.asarray(x)
    return x.reshape((x.shape[0],) + (1,) * (rank - len(shape)) + tuple(shape))


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
        outputs, out_flags = _eval_eqn(eqn, values, bflags, batch_size)
        for v, x, b in zip(eqn.outvars, outputs, out_flags):
            env[v], flags[v] = x, b
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
        # DCE drops the unused ct invar; x stays first, so evaluation passes [x, ct] safely.
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

    def forward_with_cache(self, x):
        return self(x), x

    def vjp(self, x, ct):
        """Batched VJP: x (B, *in_shape), ct (B, *out_shape) -> (B, *in_shape)."""
        ct = np.asarray(ct)
        if not np.issubdtype(np.dtype(self.out_dtype), np.complexfloating):
            ct = np.real(ct).astype(self.out_dtype)  # JAX passes real cotangents to real outputs
        return self._run(self.backward, x, ct)[1]

    def vjp_from_cache(self, cache, ct):
        return self.vjp(cache, ct)




class _Unsupported(Exception):
    pass


def _sum_to(ct, shape):
    """Sum a batched cotangent (B, *out) down to the batched operand shape (B, *s).

    The per-example shape s is right-aligned with out (elementwise broadcasting):
    extra leading per-example axes are summed, size-1 axes are summed with keepdims.
    The batch axis is never summed.
    """
    ct = np.asarray(ct)
    extra = ct.ndim - len(shape)
    if extra > 0:
        ct = ct.sum(axis=tuple(range(1, 1 + extra)))
    for axis in range(1, len(shape)):
        if shape[axis] == 1 and ct.shape[axis] != 1:
            ct = ct.sum(axis=axis, keepdims=True)
    return np.broadcast_to(ct, shape)


def _transpose_eqn(eqn, cts_out, primal, tangent_in, batch):
    """Transpose one *linear* (tangent) equation of a JVP jaxpr, JAX complex convention.

    Returns {invar index: cotangent} for the tangent inputs. Shapes carry a leading batch axis.
    """
    name = str(eqn.primitive)
    p = eqn.params
    shapes = [tuple(v.aval.shape) for v in eqn.invars]
    full = lambda i: (batch, *shapes[i])
    tangent_idx = [i for i, t in enumerate(tangent_in) if t]
    rank = max(len(sh) for sh in shapes) if shapes else 0
    raw_primal = primal
    primal = lambda i: _align(*raw_primal(i), shapes[i], rank)  # aligned like the forward op
    if name == "split":
        pieces = [np.zeros((batch, *v.aval.shape), np.dtype(v.aval.dtype)) if c is None else c
                  for v, c in zip(eqn.outvars, cts_out)]
        return {0: np.concatenate(pieces, axis=p["axis"] + 1)}
    ct = cts_out[0]
    if name in ("add", "add_any"):
        return {i: _sum_to(ct, full(i)) for i in tangent_idx}
    if name == "sub":
        return {i: _sum_to(ct if i == 0 else -ct, full(i)) for i in tangent_idx}
    if name == "neg":
        return {0: -ct}
    if name in ("copy", "copy_p"):
        return {0: ct}
    if name == "mul":
        if len(tangent_idx) != 1:
            raise _Unsupported("nonlinear mul")
        i = tangent_idx[0]
        return {i: _sum_to(ct * primal(1 - i), full(i))}
    if name == "div":
        if tangent_idx != [0]:
            raise _Unsupported("nonlinear div")
        return {0: _sum_to(ct / primal(1), full(0))}
    if name == "convert_element_type":
        src = np.dtype(eqn.invars[0].aval.dtype)
        if np.issubdtype(src, np.complexfloating):
            return {0: np.asarray(ct).astype(src)}
        return {0: np.real(ct).astype(src)}
    if name == "conj":
        src = np.dtype(eqn.invars[0].aval.dtype)
        return {0: np.conj(ct) if np.issubdtype(src, np.complexfloating) else np.real(ct).astype(src)}
    if name == "real":
        return {0: np.asarray(ct).astype(np.complex128)}
    if name == "imag":
        return {0: -1j * np.asarray(ct)}
    if name == "complex":
        out = {}
        if tangent_in[0]:
            out[0] = _sum_to(np.real(ct), full(0))
        if tangent_in[1]:
            out[1] = _sum_to(-np.imag(ct), full(1))
        return out
    if name == "select_n":
        if tangent_in[0]:
            raise _Unsupported("tangent predicate")
        which = np.asarray(primal(0))
        ct = np.asarray(ct)
        which = which.astype(np.int32) if which.dtype == np.bool_ else which
        return {i: _sum_to(np.where(which == i - 1, ct, 0), full(i)) for i in tangent_idx}
    if name == "broadcast_in_dim":
        dims = p["broadcast_dimensions"]
        out_shape = p["shape"]
        summed = tuple(k + 1 for k in range(len(out_shape)) if k not in dims)
        r = np.sum(ct, axis=summed) if summed else ct
        for k, d in enumerate(dims):
            if shapes[0][k] == 1 and out_shape[d] != 1:
                r = np.sum(r, axis=k + 1, keepdims=True)
        return {0: np.reshape(r, full(0))}
    if name in ("reshape", "squeeze", "expand_dims"):
        if name == "reshape" and p.get("dimensions") is not None:
            raise _Unsupported("reshape dimensions")
        return {0: np.reshape(ct, full(0))}
    if name == "slice":
        st = p.get("strides") or (1,) * len(p["start_indices"])
        z = np.zeros(full(0), np.asarray(ct).dtype)
        z[(slice(None), *[slice(a, b, c) for a, b, c in zip(p["start_indices"], p["limit_indices"], st)])] = ct
        return {0: z}
    if name == "pad":
        if tangent_in[1]:
            raise _Unsupported("tangent padding value")
        r = np.asarray(ct)
        for k, (lo, hi, interior) in enumerate(p["padding_config"]):
            n_in = shapes[0][k]
            pos = lo + np.arange(n_in) * (interior + 1)
            n_out = r.shape[k + 1]
            valid = (pos >= 0) & (pos < n_out)
            r = np.take(r, np.clip(pos, 0, max(n_out - 1, 0)), axis=k + 1)
            mask_shape = [1] * r.ndim
            mask_shape[k + 1] = n_in
            r = r * valid.reshape(mask_shape)
        return {0: r}
    if name == "concatenate":
        axis = p["dimension"] + 1
        bounds = np.cumsum([s[p["dimension"]] for s in shapes])[:-1]
        pieces = np.split(ct, bounds, axis=axis)
        return {i: pieces[i] for i in tangent_idx}
    if name == "reduce_sum":
        axes = tuple(sorted(a + 1 for a in p["axes"]))
        return {0: np.broadcast_to(np.expand_dims(ct, axes), full(0))}
    if name == "transpose":
        inverse = np.argsort(p["permutation"])
        return {0: np.transpose(ct, (0, *[int(q) + 1 for q in inverse]))}
    raise _Unsupported(name)


_TRANSPOSABLE = {"add", "add_any", "sub", "neg", "copy", "copy_p", "mul", "div", "convert_element_type",
                 "conj", "real", "imag", "complex", "select_n", "broadcast_in_dim", "reshape", "squeeze",
                 "expand_dims", "slice", "pad", "concatenate", "split", "reduce_sum", "transpose"}


class JvpTransposedFunction:
    """Per-example function with its VJP obtained by transposing JAX's JVP program.

    Tracing `jax.jvp` is ~3x cheaper than tracing `jax.vjp` (no partial evaluation
    or JAX-side transposition). Derivative formulas still come from JAX's JVP
    rules; only the linear tangent equations are transposed here, which needs
    rules for linear primitives only. Raises _Unsupported at construction if the
    tangent program contains anything else (caller falls back to TracedFunction).
    """

    def __init__(self, function, in_shape, dtype):
        example = jax.ShapeDtypeStruct(in_shape, dtype)
        closed = jax.make_jaxpr(lambda x, t: jax.jvp(function, (x,), (t,)))(example, example)
        jaxpr = closed.jaxpr
        self.consts = closed.consts
        self.in_shape, self.in_dtype = tuple(in_shape), np.dtype(dtype)
        y, ydot = jaxpr.outvars
        self.out_shape, self.out_dtype = tuple(y.aval.shape), np.dtype(y.aval.dtype)
        x_var, t_var = jaxpr.invars
        tangent = {t_var}
        self.primal_eqns, self.tangent_eqns = [], []
        for eqn in jaxpr.eqns:
            flags = [(not isinstance(v, jcore.Literal)) and v in tangent for v in eqn.invars]
            if any(flags):
                name = str(eqn.primitive)
                if name not in _TRANSPOSABLE:
                    raise _Unsupported(name)
                self.tangent_eqns.append((eqn, flags))
                tangent.update(eqn.outvars)
            else:
                self.primal_eqns.append(eqn)
        if (not isinstance(y, jcore.Literal)) and y in tangent:
            raise _Unsupported("primal output depends on tangent")
        self.jaxpr, self.x_var, self.t_var, self.y_var, self.ydot_var = jaxpr, x_var, t_var, y, ydot
        self.ydot_is_tangent = (not isinstance(ydot, jcore.Literal)) and ydot in tangent
        primal_only = jcore.Jaxpr(jaxpr.constvars, [x_var], [y], self.primal_eqns)
        self.unsupported = unsupported_primitives(primal_only)

    def _primal_env(self, x):  # returns env, batched flags, batch size
        x = np.asarray(x)
        batch = x.shape[0]
        env, flags = {}, {}
        for v, c in zip(self.jaxpr.constvars, self.consts):
            env[v], flags[v] = np.asarray(c), False
        env[self.x_var], flags[self.x_var] = x, True
        with np.errstate(all="ignore"):
            for eqn in self.primal_eqns:
                values = [v.val if isinstance(v, jcore.Literal) else env[v] for v in eqn.invars]
                bflags = [False if isinstance(v, jcore.Literal) else flags[v] for v in eqn.invars]
                outs, oflags = _eval_eqn(eqn, values, bflags, batch)
                for v, o, f in zip(eqn.outvars, outs, oflags):
                    env[v], flags[v] = o, f
        return env, flags, batch

    def __call__(self, x):
        return self.forward_with_cache(x)[0]

    def forward_with_cache(self, x):
        env, flags, batch = self._primal_env(x)
        y = self.y_var.val if isinstance(self.y_var, jcore.Literal) else env[self.y_var]
        f = False if isinstance(self.y_var, jcore.Literal) else flags[self.y_var]
        out = y if f else np.broadcast_to(y, (batch, *np.shape(y)))
        return out, (env, flags, batch)

    def vjp(self, x, ct):
        return self.vjp_from_cache(self._primal_env(x), ct)

    def vjp_from_cache(self, cache, ct):
        env, flags, batch = cache
        zero = np.zeros((batch, *self.in_shape), np.complex128 if np.issubdtype(self.in_dtype, np.complexfloating) else self.in_dtype)
        if not self.ydot_is_tangent:
            return zero
        ct = np.asarray(ct)
        if not np.issubdtype(self.out_dtype, np.complexfloating):
            ct = np.real(ct).astype(self.out_dtype)
        cts = {self.ydot_var: ct}
        with np.errstate(all="ignore"):
            for eqn, tflags in reversed(self.tangent_eqns):
                outs = [cts.pop(v, None) for v in eqn.outvars]
                if all(o is None for o in outs):
                    continue
                primal = lambda i, eqn=eqn: ((eqn.invars[i].val, False) if isinstance(eqn.invars[i], jcore.Literal)
                                             else (env[eqn.invars[i]], flags[eqn.invars[i]]))
                for i, c in _transpose_eqn(eqn, outs, primal, tflags, batch).items():
                    v = eqn.invars[i]
                    cts[v] = c if v not in cts else cts[v] + c
        result = cts.get(self.t_var)
        return zero if result is None else np.asarray(result).astype(zero.dtype).reshape(zero.shape)


def _close(a, b, tol=1e-9):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    both_nan = np.isnan(a) & np.isnan(b)
    scale = max(float(np.nanmax(np.abs(b), initial=0.0)), 1e-300)
    diff = np.where(both_nan, 0.0, np.abs(a - b))
    return bool(np.all(np.isfinite(diff) | both_nan) and np.nanmax(diff, initial=0.0) <= tol * scale)


def _self_check(traced, in_shape, dtype):
    """Cheap NumPy-only consistency checks run once per traced function.

    1. Batched evaluation (batch 11) equals batch-1 evaluation of sampled rows (forward and VJP):
       catches batching/broadcasting errors of the interpreter.
    2. For JVP-transposed functions, the dot-product test
       Re<ct, J t> == Re<J^T ct, t> against JAX's JVP program, evaluated
       independently: catches transposition errors.
    Returns None if all checks pass, otherwise a reason string.
    """
    rng = np.random.default_rng(1234)
    batch = 11
    shape = (batch, *in_shape)
    if np.issubdtype(np.dtype(dtype), np.complexfloating):
        x = (rng.uniform(0.3, 1.7, shape) + 0j).astype(dtype)
        t = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(dtype)
    else:
        x = rng.uniform(0.3, 1.7, shape).astype(dtype)
        t = rng.normal(size=shape).astype(dtype)
    try:
        with np.errstate(all="ignore"):
            y = np.asarray(traced(x))
            out_shape = (batch, *traced.out_shape)
            ct = rng.normal(size=out_shape) + (1j * rng.normal(size=out_shape)
                                               if np.issubdtype(np.dtype(traced.out_dtype), np.complexfloating) else 0)
            g = np.asarray(traced.vjp(x, ct))
            for i in (0, batch - 4):  # batch-1 evaluation is immune to batch-axis misalignment
                if not _close(np.asarray(traced(x[i:i + 1]))[0], y[i]):
                    return "batched forward mismatch"
                if not _close(np.asarray(traced.vjp(x[i:i + 1], ct[i:i + 1]))[0], g[i]):
                    return "batched vjp mismatch"
            if isinstance(traced, JvpTransposedFunction):
                outs, flags = eval_jaxpr_batched(traced.jaxpr, traced.consts, [x, t], [True, True], batch)
                ydot = outs[1] if flags[1] else np.broadcast_to(outs[1], out_shape)
                lhs = np.real(np.sum(np.reshape(ct, (batch, -1)) * np.reshape(ydot, (batch, -1)), axis=1))
                rhs = np.real(np.sum(np.reshape(g, (batch, -1)) * np.reshape(t, (batch, -1)), axis=1))
                if not _close(lhs, rhs, 1e-8):
                    return "dot-product test failed"
    except Exception as error:  # any interpreter failure -> unsupported
        return f"self-check error: {type(error).__name__}"
    return None


_TRACE_CACHE = {}


def trace_function(function, in_shape, dtype):
    """JVP-transposed tracing when possible (cheaper), else forward+VJP tracing.

    Every traced function is self-checked once and cached by (function, in_shape, dtype);
    failures are recorded in `.unsupported` so that `simulate` falls back to the
    in-graph JAX implementation.
    """
    key = (function, tuple(in_shape), np.dtype(dtype))
    cached = _TRACE_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        traced = JvpTransposedFunction(function, in_shape, dtype)
        if _self_check(traced, in_shape, dtype) is not None:
            raise _Unsupported("self-check")
        return traced  # already self-checked above; checking it again would repeat identical work
    except _Unsupported:
        traced = TracedFunction(function, in_shape, dtype)
    if not traced.unsupported:
        reason = _self_check(traced, in_shape, dtype)
        if reason is not None:
            traced.unsupported = set(traced.unsupported) | {reason}
    _TRACE_CACHE[key] = traced
    return traced


# ======================================================================
# Static plan (shapes and traced functions), built at JAX trace time
# ======================================================================

class Plan:
    """Everything shape-dependent the host program needs; created while tracing `simulate`."""

    def __init__(self, shapes, groups_meta, link_ids):
        self.shapes = shapes
        self.traced = {}
        self.unsupported = set()
        self._last_a = None
        for groups in groups_meta.values():
            for key in groups:
                fid = int(key[1:])
                if fid not in self.traced:
                    t = native_function(FUNCTIONS[fid]) or trace_function(FUNCTIONS[fid], (7,), jnp.complex128)
                    self.traced[fid] = t
                    self.unsupported |= t.unsupported
        self.links = {}
        for index in link_ids:  # only the linking functions this setup references
            t = native_link(LINKING_FUNCTIONS[index]) or trace_function(LINKING_FUNCTIONS[index], (), jnp.complex128)
            self.links[index] = t
            self.unsupported |= t.unsupported

    @property
    def supported(self):
        return not self.unsupported


# ======================================================================
# Host program
# ======================================================================

_RESIDUALS = OrderedDict()
_RESIDUAL_SLOTS = 8
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
    if n_source_rows == 1:
        return ct.sum(axis=0, keepdims=True)
    ct_in = np.zeros((n_source_rows, ct.shape[1]), ct.dtype)
    np.add.at(ct_in, source, ct)
    return ct_in


def _relevant_parameters(a, n_params):
    """Parameters whose cotangent reaches the gradient: the optimized ones and the outputs of
    links whose input is optimized (links read the values before linking, so no chaining)."""
    relevant = np.zeros(n_params, bool)
    relevant[a["optimized_parameter_indices"]] = True
    link_out, link_in = np.asarray(a["indices_to_link"]), np.asarray(a["linked_indices"])
    if link_out.size:
        relevant[link_out[relevant[link_in]]] = True
    return relevant


def _group_columns(relevant, inputs):
    """Input columns (of 7) of a group that read a relevant parameter, or None if none do."""
    needed = relevant[inputs]
    if not needed.any():
        return None
    columns = np.zeros(7, bool)
    columns[:inputs.shape[1]] = needed.any(axis=0)
    return columns


def _assemble(plan, parameters, base, groups, relevant=None):
    """Evaluate grouped component functions into compressed values (R, nnz).

    With `relevant` (see _relevant_parameters), native kernels also keep the partials the
    gradient needs, so that the backward does not evaluate them again.
    """
    rows_count = parameters.shape[0]
    values = np.broadcast_to(base, (rows_count, base.shape[0])).copy()
    saved = []
    for key, g in groups.items():
        fid = int(key[1:])
        # 'c' groups read no swept parameter: all rows agree, evaluate row 0 once.
        evaluated_rows = 1 if key[0] == "c" else rows_count
        gathered = parameters[:evaluated_rows, g["inputs"]]  # (R, K, width)
        if gathered.shape[-1] < 7:  # narrow groups (e.g. carrier-side `space` reads 3): functions only read their own width
            gathered = np.pad(gathered, ((0, 0), (0, 0), (0, 7 - gathered.shape[-1])))
        inputs = gathered.reshape(-1, 7)  # (R*K, 7)
        traced = plan.traced[fid]
        columns = None
        if relevant is not None and isinstance(traced, NativeFunction):
            columns = _group_columns(relevant, np.asarray(g["inputs"]))
        if columns is not None:
            raw_out, cache = traced.forward_with_cache(inputs, columns)
        else:
            raw_out, cache = traced.forward_with_cache(inputs)
        out = raw_out.reshape(evaluated_rows, g["inputs"].shape[0], -1)
        values[:, g["k"]] = out[:, g["rows"], g["slots"]]
        saved.append((key, cache, inputs.shape[0], out.shape))
    return values.astype(np.complex128, copy=False), saved


def _scatter_add(index, weights, size):
    """Complex `np.add.at(zeros(size), index, weights)` for flat indices, via np.bincount.

    np.bincount accumulates in input order starting from zero, exactly as np.add.at does, so
    the result is bit-identical; it is much faster for complex data.
    """
    weights = np.asarray(weights)
    out = np.empty(size, np.complex128)
    out.real = np.bincount(index, weights.real, size)
    out.imag = np.bincount(index, weights.imag, size)
    return out


def _assemble_backward(plan, ct_values, groups, saved, parameters_shape, relevant=None):
    """Adjoint of `_assemble` w.r.t. parameters (R, P).

    `relevant` (bool per parameter, optional): cotangents are only needed for these parameters.
    Native kernels then skip the partials of input columns that read none of them; the
    cotangents of the relevant parameters are unchanged (see NativeFunction.vjp).
    """
    n_rows, n_params = parameters_shape
    flat_index, flat_weights = [], []
    for (key, cache, n_batch, out_shape) in saved:
        g = groups[key]
        fid = int(key[1:])
        ct_group = ct_values[:, g["k"]]
        if key[0] == "c":  # broadcast over rows -> sum of row cotangents
            ct_group = ct_group.sum(axis=0, keepdims=True)
        R, K, S = out_shape
        out_index = g.get("_out_index")
        if out_index is None:  # flat (k, slot) position of every write, cached on the group
            out_index = g["_out_index"] = np.asarray(g["rows"]) * S + np.asarray(g["slots"])
        index = (np.arange(R)[:, None] * (K * S) + out_index[None, :]).reshape(-1)
        ct_out = _scatter_add(index, ct_group.reshape(-1), R * K * S)
        traced = plan.traced[fid]
        inputs = np.asarray(g["inputs"])
        width = inputs.shape[1]
        if relevant is not None:
            needed = relevant[inputs]  # (K, width)
            if not needed.any():
                continue
        if relevant is not None and isinstance(traced, NativeFunction):
            ct_in = traced.vjp_from_cache(cache, ct_out.reshape(n_batch, -1), _group_columns(relevant, inputs))
        else:
            ct_in = traced.vjp_from_cache(cache, ct_out.reshape(n_batch, -1))
        ct_in = np.asarray(ct_in).reshape(R, K, 7)[..., :width]
        if relevant is None:
            flat_index.append((np.arange(R)[:, None, None] * n_params + inputs[None]).reshape(-1))
            flat_weights.append(ct_in.reshape(-1))
        else:
            # Only entries that land on relevant parameters; their accumulation order is unchanged.
            sel = np.flatnonzero(needed.reshape(-1))
            flat_index.append((np.arange(R)[:, None] * n_params + inputs.reshape(-1)[sel][None, :]).reshape(-1))
            flat_weights.append(ct_in.reshape(R, K * width)[:, sel].reshape(-1))
    if not flat_index:
        return np.zeros(parameters_shape, np.complex128)
    # One scatter for all groups keeps np.add.at's group-by-group accumulation order.
    return _scatter_add(np.concatenate(flat_index), np.concatenate(flat_weights),
                        n_rows * n_params).reshape(parameters_shape)


_SUPERLU = []
_DLL_DIRS = []


def _superlu():
    """SciPy's SuperLU extension module (`scipy.sparse.linalg._dsolve._superlu`).

    Importing `scipy.sparse.linalg` costs ~0.5 s, almost all of it unrelated to
    SuperLU. The compiled extension only needs NumPy, so it is loaded directly and
    registered under its real name (a later `import scipy.sparse.linalg` reuses it).
    Falls back to the regular import if the layout differs.
    """
    if not _SUPERLU:
        name = "scipy.sparse.linalg._dsolve._superlu"
        module = sys.modules.get(name)
        if module is None:
            try:
                import glob
                import importlib.util
                scipy_dir = os.path.dirname(importlib.util.find_spec("scipy").origin)
                if os.name == "nt":
                    for candidate in (os.path.join(os.path.dirname(scipy_dir), "scipy.libs"),
                                      os.path.join(scipy_dir, ".libs")):
                        if os.path.isdir(candidate):
                            _DLL_DIRS.append(os.add_dll_directory(candidate))
                (path,) = glob.glob(os.path.join(scipy_dir, "sparse", "linalg", "_dsolve", "_superlu*.so")) or \
                    glob.glob(os.path.join(scipy_dir, "sparse", "linalg", "_dsolve", "_superlu*.pyd"))
                spec = importlib.util.spec_from_file_location(name, path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                sys.modules[name] = module
            except Exception:
                from scipy.sparse.linalg._dsolve import _superlu as module
        _SUPERLU.append(module)
    return _SUPERLU[0]


def _csc_construct(*args, **kwargs):  # only used if SuperLU.L / .U are requested
    from scipy.sparse import csc_array
    return csc_array(*args, **kwargs)


class SparsePattern:
    """Fixed sparsity pattern (rows, cols) of an n x n matrix in sorted CSC order."""

    def __init__(self, rows, cols, n, order=None, indices=None, indptr=None):
        rows = np.asarray(rows, np.int64)
        cols = np.asarray(cols, np.int64)
        self.n, self.rows, self.cols = n, rows, cols
        if order is None:
            self.order = np.lexsort((rows, cols))  # by column, then row (canonical CSC)
            self.indices = np.ascontiguousarray(rows[self.order], dtype=np.intc)
            self.indptr = np.ascontiguousarray(
                np.concatenate([[0], np.cumsum(np.bincount(cols, minlength=n))]), dtype=np.intc)
        else:
            self.order, self.indices, self.indptr = order, indices, indptr

    def matvec(self, values, x, trans):
        """A x (trans 'N'), A^T x ('T') or A^H x ('H') for x of shape (n,) or (n, k)."""
        if trans == "N":
            out_index, in_index, v = self.rows, self.cols, values
        else:
            out_index, in_index = self.cols, self.rows
            v = values if trans == "T" else np.conj(values)
        if x.ndim == 1:
            terms = v * x[in_index]
            return (np.bincount(out_index, terms.real, self.n)
                    + 1j * np.bincount(out_index, terms.imag, self.n))
        return np.stack([self.matvec(values, x[:, k], trans) for k in range(x.shape[1])], axis=1)


_PATTERNS = {}


def _pattern(rows, cols, n):
    rows = np.asarray(rows)
    cols = np.asarray(cols)
    key = (n, rows.shape[0], hash(rows.tobytes()), hash(cols.tobytes()))
    pattern = _PATTERNS.get(key)
    if pattern is None:
        pattern = _PATTERNS[key] = SparsePattern(rows, cols, n)
    return pattern


_SPARSETOOLS = []


def _sparsetools():
    if not _SPARSETOOLS:
        name = "scipy.sparse._sparsetools"
        module = sys.modules.get(name)
        if module is None:
            try:
                import glob
                import importlib.util
                scipy_dir = os.path.dirname(importlib.util.find_spec("scipy").origin)
                _superlu()  # ensures DLL directory is registered on Windows
                (path,) = glob.glob(os.path.join(scipy_dir, "sparse", "_sparsetools*.so")) or \
                    glob.glob(os.path.join(scipy_dir, "sparse", "_sparsetools*.pyd"))
                spec = importlib.util.spec_from_file_location(name, path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                sys.modules[name] = module
            except Exception:
                from scipy.sparse import _sparsetools as module
        _SPARSETOOLS.append(module)
    return _SPARSETOOLS[0]


class RefinedLU:
    """SuperLU factorisation of a sparse A with one step of iterative refinement.

    The optical matrices are badly scaled (condition numbers up to ~1e31), and
    some outputs (e.g. the quantum-noise quadratic form) depend on components
    of the solution far below its norm. A fill-reducing sparse pivot order is
    normwise backward stable but can lose those small components; one step of
    fixed-precision refinement (residual computed with the sparse matrix)
    restores componentwise accuracy. Validated against 60-digit references.
    A singular matrix yields NaN (as the dense solver would), not an exception.
    Factorisation options are SuperLU's defaults, exactly as `scipy.sparse.linalg.splu`.
    """

    # DIFFEROMETOR_REFINE=0 skips refinement (about 0.1 s faster per warm run on the benchmark setup, results
    # agree to ~5e-8); the default keeps it because other, worse-conditioned setups need it.
    def __init__(self, values, rows, cols, n, refine=int(os.environ.get("DIFFEROMETOR_REFINE", 1)), pattern=None):
        self.pattern = _pattern(rows, cols, n) if pattern is None else pattern
        self.values = np.asarray(values, dtype=np.complex128)
        self.refine = refine
        self.data = np.ascontiguousarray(self.values[self.pattern.order])
        try:
            self.lu = _superlu().gstrf(
                n, self.data.shape[0], self.data, self.pattern.indices, self.pattern.indptr,
                csc_construct_func=_csc_construct, ilu=False,
                options=dict(DiagPivotThresh=None, ColPerm=None, PanelSize=None, Relax=None))
        except RuntimeError:  # exactly singular
            self.lu = None

    def _matvec(self, x, trans):
        st = _sparsetools()
        n = self.pattern.n
        indptr, indices, data = self.pattern.indptr, self.pattern.indices, self.data
        if x.ndim == 1:
            out = np.zeros(n, dtype=np.complex128)
            if trans == "N":
                st.csc_matvec(n, n, indptr, indices, data, np.ascontiguousarray(x), out)
            elif trans == "T":
                st.csr_matvec(n, n, indptr, indices, data, np.ascontiguousarray(x), out)
            else:
                st.csr_matvec(n, n, indptr, indices, data, np.ascontiguousarray(np.conj(x)), out)
                np.conj(out, out=out)
            return out
        return np.stack([self._matvec(x[:, k], trans) for k in range(x.shape[1])], axis=1)

    def solve(self, b, trans="N"):
        b = np.asarray(b, dtype=np.complex128)
        if self.lu is None:
            return np.full(b.shape, np.nan + 1j * np.nan)
        x = self.lu.solve(b, trans=trans)
        for _ in range(self.refine):
            x = x + self.lu.solve(b - self._matvec(x, trans), trans=trans)
        return x


def _factor(values, rows, cols, n):
    return RefinedLU(values, rows, cols, n)


_BATCHED_PATTERNS = {}


def BatchedLU(values, rows, cols, n):
    """Factorise F same-pattern systems: the native C++ solver if built, else SuperLU.

    Both refine every solve once and give NaN only for singular systems. The native
    solver (`differometor/native`) also avoids loading SciPy's SuperLU extension.
    `DIFFEROMETOR_NATIVE=0` forces SuperLU.
    """
    from differometor import native
    if native.available():
        return native.NativeBatchedLU(values, _pattern(rows, cols, n))
    return SuperLUBatchedLU(values, rows, cols, n)


class SuperLUBatchedLU:
    """Factorise F same-pattern systems at once as one block-diagonal sparse matrix.

    Block-diagonal LU is exactly the per-block LU (pivoting cannot cross blocks), but
    one SuperLU call replaces F calls and every solve handles all F right-hand sides
    together, removing per-call overhead. If any block is singular the systems are
    re-factorised one by one so that only the singular frequencies become NaN.
    values: (F, nnz). solve(b): b of shape (F, n) or (F, n, k).
    """

    def __init__(self, values, rows, cols, n):
        values = np.asarray(values, dtype=np.complex128)
        self.F, self.n = values.shape[0], n
        small = _pattern(rows, cols, n)
        if self.F == 1:
            self.parts = None
            self.lu = RefinedLU(values[0], rows, cols, n, pattern=small)
            if self.lu.lu is None:
                self.parts = [self.lu]
            return
        key = (id(small), self.F)
        big_pat = _BATCHED_PATTERNS.get(key)
        if big_pat is None:
            nnz = small.order.shape[0]
            offsets = (np.arange(self.F, dtype=np.int64) * n)[:, None]
            big_rows = (small.rows[None, :] + offsets).reshape(-1)
            big_cols = (small.cols[None, :] + offsets).reshape(-1)
            order = (small.order[None, :] + (np.arange(self.F, dtype=np.int64) * nnz)[:, None]).reshape(-1)
            f_intc = np.arange(self.F, dtype=np.intc)[:, None]
            indices = np.ascontiguousarray((small.indices[None, :] + f_intc * n).reshape(-1), dtype=np.intc)
            indptr = np.empty(self.F * n + 1, dtype=np.intc)
            indptr[0] = 0
            indptr[1:] = (small.indptr[1:][None, :] + f_intc * nnz).reshape(-1)
            big_pat = _BATCHED_PATTERNS[key] = SparsePattern(big_rows, big_cols, self.F * n, order, indices, indptr)
        self.parts = None
        self.lu = RefinedLU(values.reshape(-1), big_pat.rows, big_pat.cols, self.F * n, pattern=big_pat)
        if self.lu.lu is None:
            self.parts = [RefinedLU(values[f], rows, cols, n, pattern=small) for f in range(self.F)]

    def solve(self, b, trans="N", overwrite_b=False):  # overwrite_b: accepted for interface parity
        b = np.asarray(b, dtype=np.complex128)
        if b.ndim == 3 and b.shape[2] == 0:
            return np.zeros(b.shape, np.complex128)
        if self.parts is not None:
            return np.stack([lu.solve(b[f], trans) for f, lu in enumerate(self.parts)])
        flat = b.reshape(self.F * self.n, *b.shape[2:]) if self.F > 1 else b[0]
        return self.lu.solve(np.ascontiguousarray(flat), trans).reshape(b.shape)


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
    # Only cotangents of these parameters reach the gradient (see the end of _backward).
    relevant = _relevant_parameters(a, P.shape[1])
    qhd = np.exp(1j * (P[0, a["qhd_parameter_indices"]] * (np.pi / 180)))

    # carrier
    Pc, src_c, set_c = _expand(P, a["carrier_changing_parameter_indices"], a["carrier_changing_values"])
    cvals, saved_c = _assemble(plan, Pc, a["carrier_values"], a["carrier_groups"], relevant)
    n_c = a["carrier_rows"].shape[0]
    CN = a["carrier_size_marker"].shape[0]
    F_c = cvals.shape[0]
    cb = np.zeros((F_c, CN), np.complex128)
    cb[:, a["carrier_rhs_rows"]] = cvals[:, n_c:]
    lus_c = BatchedLU(cvals[:, :n_c], a["carrier_rows"], a["carrier_cols"], CN)
    carrier = lus_c.solve(cb, overwrite_b=True)  # (CV, CN); cb is a temporary

    # signal
    Ps, src_s, set_s = _expand(Pc, a["signal_changing_parameter_indices"], a["signal_changing_values"])
    svals0, saved_s = _assemble(plan, Ps, a["signal_values"], a["signal_groups"], relevant)
    pk = a["signal_place_k"]
    sci = a["signal_carrier_indices"].reshape(-1)
    placed_before = svals0[:, pk].copy()
    svals = svals0
    svals[:, pk] = -placed_before * carrier[:, sci]
    mask = a["signal_conj_mask"]
    conj_k = np.flatnonzero(mask)
    svals[:, conj_k] = np.conj(svals[:, conj_k])  # same values as np.where(mask, conj, svals), in place
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

    lus_s = BatchedLU(svals[:, :n_s], a["signal_rows"], a["signal_cols"], SN)
    signal = lus_s.solve(sb, overwrite_b=True)  # (F, SN); sb is a temporary
    w = np.ascontiguousarray(np.transpose(lus_s.solve(np.transpose(c, (0, 2, 1)), trans="H"), (0, 2, 1)))  # (F, D, SN)

    # quantum noise
    noise_const = all(k[0] == "c" for k in a["noise_groups"])
    Pn = P if noise_const else Ps
    nvals, saved_n = _assemble(plan, Pn, a["noise_values"], a["noise_groups"], relevant)
    nr, nc = a["noise_rows"], a["noise_cols"]
    wr_conj = np.conj(w[:, :, nr])
    wc = w[:, :, nc]
    q = np.sum(np.real(nvals[:, None, :] * wr_conj * wc), axis=2)  # (F, D)
    noise = np.sqrt(NOISE_SCALE * q.T)  # (D, F)

    r.update(relevant=relevant, Pc=Pc, src_c=src_c, set_c=set_c, saved_c=saved_c, lus_c=lus_c, carrier=carrier,
             Ps=Ps, src_s=src_s, set_s=set_s, saved_s=saved_s, placed_before=placed_before,
             svals=svals, nsv=nsv, nsv_t=nsv_t, qhd=qhd, lus_s=lus_s, signal=signal, w=w,
             nvals=nvals, saved_n=saved_n, noise_const=noise_const, wr_conj=wr_conj, wc=wc, noise=noise)
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
    ct_nvals = np.sum(ct_s * wr_conj * wc, axis=(0, 1))[None, :] if r["noise_const"] else np.sum(ct_s * wr_conj * wc, axis=1)
    # ct_w[f, d, nr] += conj(ct_s nvals wc); then ct_w[f, d, nc] += ct_s nvals wr_conj (np.add.at order)
    M = nr.shape[0]
    base = (np.arange(F * D) * SN)[:, None]
    index = np.concatenate([(base + nr[None, :]).reshape(-1), (base + nc[None, :]).reshape(-1)])
    weights = np.concatenate([np.conj(ct_s * nvals[:, None, :] * wc).reshape(F * D, M).reshape(-1),
                              (ct_s * nvals[:, None, :] * wr_conj).reshape(-1)])
    ct_w = _scatter_add(index, weights, F * D * SN).reshape(F, D, SN)

    # sparse solves (signal x and noise weights w share one factorisation)
    svals = r["svals"]
    n_s = a["signal_rows"].shape[0]
    sr, sc = a["signal_rows"], a["signal_cols"]
    signal = r["signal"]
    native_lu = isinstance(r["lus_s"], NativeBatchedLU)
    # The native solver keeps its own copy of the matrix values and the backward needs only the
    # shape of svals, so its (already mapped) memory is reused for the cotangent.
    ct_svals = svals if native_lu else np.empty(svals.shape, np.complex128)
    bb = r["lus_s"].solve(ct_signal, trans="T")  # (F, SN)
    y_nd = r["lus_s"].solve(np.ascontiguousarray(np.transpose(np.conj(ct_w), (0, 2, 1))), overwrite_b=True)  # (F, SN, D)
    y = np.transpose(y_nd, (0, 2, 1))  # (F, D, SN) view
    ct_c = np.conj(y)
    # ct(A) = -bb[sr] x[sc] - sum_d conj(w_d[sr]) y_d[sc]
    terms = ct_svals[:, :n_s]
    if native_lu:  # one pass in C++, same operations and order as the NumPy expression below
        from differometor.native import adjoint_entries
        adjoint_entries(np.ascontiguousarray(sr), np.ascontiguousarray(sc), np.ascontiguousarray(bb),
                        np.ascontiguousarray(signal), np.ascontiguousarray(w), y_nd, terms)
    else:
        terms[...] = -bb[:, sr] * signal[:, sc]
        terms -= np.sum(np.conj(w[:, :, sr]) * y[:, :, sc], axis=1)
    ct_svals[:, n_s:] = bb[:, a["signal_rhs_rows"]]

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
    conj_k = np.flatnonzero(a["signal_conj_mask"])
    ct_svals[:, conj_k] = np.conj(ct_svals[:, conj_k])
    pk = a["signal_place_k"]
    sci = a["signal_carrier_indices"].reshape(-1)
    ct_placed = ct_svals[:, pk].copy()
    ct_svals[:, pk] = -carrier[:, sci] * ct_placed
    ct_carrier_terms = -r["placed_before"] * ct_placed  # (F, M)
    ct_carrier_terms = ct_carrier_terms.sum(axis=0, keepdims=True) if CV == 1 and F != 1 else ct_carrier_terms
    for row in range(ct_carrier_terms.shape[0]):
        np.add.at(ct_carrier[row], sci, ct_carrier_terms[row])

    # Only cotangents of optimized and linked parameters reach the result (see the end of
    # this function); assembly adjoints may skip inputs that read none of them.
    relevant = r["relevant"]  # as in the forward, whose cached kernel partials match it

    # signal + noise assembly -> signal-expanded parameters
    ct_Ps = _assemble_backward(plan, ct_svals, a["signal_groups"], r["saved_s"], r["Ps"].shape, relevant)
    if r["noise_const"]:
        ct_P0 += _assemble_backward(plan, ct_nvals, a["noise_groups"], r["saved_n"], r["P0"].shape, relevant)
    else:
        ct_Ps += _assemble_backward(plan, ct_nvals, a["noise_groups"], r["saved_n"], r["Ps"].shape, relevant)
    ct_Pc = _expand_backward(ct_Ps, r["src_s"], r["set_s"], r["Pc"].shape[0])

    # carrier solve
    cvals_ct = np.zeros((CV, a["carrier_values"].shape[0]), np.complex128)
    n_c = a["carrier_rows"].shape[0]
    crr, ccc = a["carrier_rows"], a["carrier_cols"]
    bb = r["lus_c"].solve(ct_carrier, trans="T")  # (CV, CN)
    cvals_ct[:, :n_c] = -bb[:, crr] * carrier[:, ccc]
    cvals_ct[:, n_c:] = bb[:, a["carrier_rhs_rows"]]
    ct_Pc += _assemble_backward(plan, cvals_ct, a["carrier_groups"], r["saved_c"], r["Pc"].shape, relevant)
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


def _host_forward(plan, rebuild, optimized_parameters, *operands):
    with gc_paused():
        return _host_forward_impl(plan, rebuild, optimized_parameters, *operands)


def _host_forward_impl(plan, rebuild, optimized_parameters, *operands):
    a = _to_numpy(rebuild([np.asarray(x) for x in operands]))
    a["optimized_parameters"] = np.asarray(optimized_parameters)
    plan._last_a = a
    carrier_T, signal_T, noise, residuals = _forward(plan, a)
    _TOKEN[0] += 1
    token = _TOKEN[0]
    _RESIDUALS[token] = (residuals, a)
    while len(_RESIDUALS) > _RESIDUAL_SLOTS:
        _RESIDUALS.popitem(last=False)
    return carrier_T, signal_T, noise.reshape(plan.shapes["noise"]), np.int64(token)


def _host_backward_token(plan, token, optimized_parameters, ct_carrier, ct_signal, ct_noise):
    with gc_paused():
        return _host_backward_impl(plan, token, optimized_parameters, ct_carrier, ct_signal, ct_noise)


def _host_backward_impl(plan, token, optimized_parameters, ct_carrier, ct_signal, ct_noise):
    p = np.asarray(optimized_parameters)
    cached = _RESIDUALS.get(int(token))
    if cached is not None and np.array_equal(cached[1]["optimized_parameters"], p):
        residuals, a = cached
    else:  # recompute deterministically
        a = dict(plan._last_a, optimized_parameters=p)
        residuals = _forward(plan, a)[3]
    return _backward(plan, a, residuals, np.asarray(ct_carrier), np.asarray(ct_signal), np.asarray(ct_noise))


def make_host_simulate(source):
    """Return (simulate_fn, plan) for these (traced) arrays, or (None, plan) if unsupported.

    `source`: the compressed arrays other than optimized_parameters, as a `PackedArrays` (from
    `run_build_step`: a few buffers become the callback's operands) or as a dict of arrays.
    """
    from differometor.simulate import PackedArrays
    if isinstance(source, PackedArrays):
        arrays = source.shapes()
        operands = list(source.buffers)
        layout = source.layout

        def rebuild(buffers):
            return PackedArrays(buffers, layout).unpack()
    else:
        arrays = source
        operands, treedef = jax.tree_util.tree_flatten(source)

        def rebuild(leaves):
            return jax.tree_util.tree_unflatten(treedef, leaves)
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
    shapes = dict(carrier=(CN, carrier_rows_count), signal=(SN, F), noise=(F,) if D == 1 and F > 1 else (
        () if D == 1 and F == 1 else (D,) if F == 1 else (D, F)))
    link_ids = sorted(int(key[1:]) for key in arrays["linking_function_ids"])
    plan = Plan(shapes, groups_meta, link_ids)
    if not plan.supported:
        return None, plan

    out_shapes = (jax.ShapeDtypeStruct(shapes["carrier"], jnp.complex128),
                  jax.ShapeDtypeStruct(shapes["signal"], jnp.complex128),
                  jax.ShapeDtypeStruct(shapes["noise"], jnp.float64),
                  jax.ShapeDtypeStruct((), jnp.int64))

    def run_forward(optimized_parameters):
        return jax.pure_callback(partial(_host_forward, plan, rebuild), out_shapes, optimized_parameters,
                                 *operands, vmap_method="sequential")

    @jax.custom_vjp
    def host_simulate(optimized_parameters):
        return run_forward(optimized_parameters)[:3]

    def fwd(optimized_parameters):
        carrier_T, signal_T, noise, token = run_forward(optimized_parameters)
        return (carrier_T, signal_T, noise), (token, optimized_parameters)

    def bwd(residuals, cotangents):
        token, optimized_parameters = residuals
        ct_p = jax.pure_callback(partial(_host_backward_token, plan),
                                 jax.ShapeDtypeStruct(optimized_parameters.shape, jnp.float64),
                                 token, optimized_parameters, *cotangents, vmap_method="sequential")
        return (ct_p.astype(optimized_parameters.dtype),)

    host_simulate.defvjp(fwd, bwd)
    return host_simulate, plan
