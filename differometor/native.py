"""
Native NumPy kernels for the built-in component functions (value + exact VJP).

Why: the host path previously traced every component function with JAX
(`jax.make_jaxpr` of its JVP) and ran a self-check on the result, which cost
~80 ms of the first run on an M2 for the public setup. The built-in component
functions are small closed-form expressions, so each one is re-expressed here
once, in the same operation order as `differometor.components`, and
differentiated by a sparse forward-mode "Wirtinger dual" number:

    y = f(x),   dy = sum_i A_i dx_i + B_i conj(dx_i)

`A` holds holomorphic partials, `B` the anti-holomorphic ones (only `conj`
produces them). The VJP in JAX's complex convention (ct(z) = dL/dx - i dL/dy)
is then  ct(x_i) = sum_k A_ki ct(y_k) + conj(B_ki ct(y_k)),  which is exactly
what JAX computes for these functions (validated in tests/check_native.py
against `jax.vjp` on random complex inputs, including the conj branches).

A kernel is used only when `FUNCTIONS[fid]` is still the original function
object from `differometor.components`; a replaced or user-added function is
traced with JAX as before (`differometor.host.trace_function`).
"""

import numpy as np

from differometor import components as _c

C_LIGHT, F0, X_SCALE, LAMBDA = _c.C_LIGHT, _c.F0, _c.X_SCALE, _c.LAMBDA
EPSILON_0C, UNIT_VACUUM = _c.EPSILON_0C, _c.UNIT_VACUUM
_DEG = np.pi / 180


# ----------------------------------------------------------------------
# Sparse Wirtinger dual number
# ----------------------------------------------------------------------

def _scale(d, s):
    return {i: v * s for i, v in d.items()}


def _sum(d1, d2):
    if not d1:
        return dict(d2)
    if not d2:
        return dict(d1)
    out = dict(d1)
    for i, v in d2.items():
        out[i] = out[i] + v if i in out else v
    return out


class Dual:
    """value `v` (B,), holomorphic partials `a` {input: (B,)}, anti-holomorphic `b`."""
    __slots__ = ("v", "a", "b")
    __array_priority__ = 1000
    __array_ufunc__ = None  # ndarray (op) Dual defers to Dual's reflected operator

    def __init__(self, v, a=None, b=None):
        self.v, self.a, self.b = v, a or {}, b or {}

    def _chain(self, value, derivative):  # holomorphic unary op with f'(self.v) = derivative
        return Dual(value, _scale(self.a, derivative), _scale(self.b, derivative))

    def __neg__(self):
        return Dual(-self.v, _scale(self.a, -1), _scale(self.b, -1))

    def __add__(self, o):
        if isinstance(o, Dual):
            return Dual(self.v + o.v, _sum(self.a, o.a), _sum(self.b, o.b))
        return Dual(self.v + o, self.a, self.b)

    __radd__ = __add__

    def __sub__(self, o):
        return self + (-o)

    def __rsub__(self, o):
        return (-self) + o

    def __mul__(self, o):
        if isinstance(o, Dual):
            return Dual(self.v * o.v, _sum(_scale(self.a, o.v), _scale(o.a, self.v)),
                        _sum(_scale(self.b, o.v), _scale(o.b, self.v)))
        return Dual(self.v * o, _scale(self.a, o), _scale(self.b, o))

    __rmul__ = __mul__

    def __truediv__(self, o):
        if isinstance(o, Dual):
            q = self.v / o.v
            return Dual(q, _sum(_scale(self.a, 1 / o.v), _scale(o.a, -q / o.v)),
                        _sum(_scale(self.b, 1 / o.v), _scale(o.b, -q / o.v)))
        return Dual(self.v / o, _scale(self.a, 1 / o), _scale(self.b, 1 / o))

    def __rtruediv__(self, o):  # o / self
        q = o / self.v
        return self._chain(q, -q / self.v)

    def __pow__(self, n):
        if n != 2:
            raise NotImplementedError
        return self._chain(self.v * self.v, 2 * self.v)


class _DualOps:
    @staticmethod
    def exp(x):
        if not isinstance(x, Dual):
            return np.exp(x)
        e = np.exp(x.v)
        return x._chain(e, e)

    @staticmethod
    def sqrt(x):
        if not isinstance(x, Dual):
            return np.sqrt(x)
        s = np.sqrt(x.v)
        return x._chain(s, 0.5 / s)

    @staticmethod
    def sin(x):
        return x._chain(np.sin(x.v), np.cos(x.v)) if isinstance(x, Dual) else np.sin(x)

    @staticmethod
    def cos(x):
        return x._chain(np.cos(x.v), -np.sin(x.v)) if isinstance(x, Dual) else np.cos(x)

    @staticmethod
    def sinh(x):
        return x._chain(np.sinh(x.v), np.cosh(x.v)) if isinstance(x, Dual) else np.sinh(x)

    @staticmethod
    def cosh(x):
        return x._chain(np.cosh(x.v), np.sinh(x.v)) if isinstance(x, Dual) else np.cosh(x)

    @staticmethod
    def arcsin(x):
        if not isinstance(x, Dual):
            return np.arcsin(x)
        return x._chain(np.arcsin(x.v), 1 / np.sqrt(1 - x.v * x.v))

    @staticmethod
    def conj(x):
        if not isinstance(x, Dual):
            return np.conj(x)
        # d conj(y) = conj(A) conj(dx) + conj(B) dx
        return Dual(np.conj(x.v), {i: np.conj(v) for i, v in x.b.items()},
                    {i: np.conj(v) for i, v in x.a.items()})


class _ValueOps:
    exp, sqrt, sin, cos, sinh, cosh, arcsin, conj = (
        np.exp, np.sqrt, np.sin, np.cos, np.sinh, np.cosh, np.arcsin, np.conj)


# ----------------------------------------------------------------------
# Kernels: same expressions and operation order as differometor.components
# ----------------------------------------------------------------------

def _radians(x):
    return x * _DEG


def _ial(m, alpha):  # incidence_angle_left
    return m.cos(_radians(alpha))


def _iar(m, alpha, n_left, n_right):  # incidence_angle_right
    return m.cos(m.arcsin(n_left / n_right * m.sin(_radians(alpha))))


def _phase_shift(tuning, frequency, n):
    return 2 * _radians(tuning) * n * (1 + frequency / F0)


def _refl_left(m, r, tuning, frequency, n_left, alpha):
    return -m.sqrt(r) * m.exp(1j * _phase_shift(tuning, frequency, n_left) * _ial(m, alpha))


def _refl_right(m, r, tuning, frequency, n_left, n_right, alpha):
    return -m.sqrt(r) * m.exp(-1j * _phase_shift(tuning, frequency, n_right) * _iar(m, alpha, n_right, n_left))


def _transmissivity(m, t, tuning, frequency, n_left, n_right, alpha):
    left = _phase_shift(tuning, frequency, n_left) * _ial(m, alpha)
    right = _phase_shift(tuning, frequency, n_right) * _iar(m, alpha, n_left, n_right)
    return -m.sqrt(t) * m.exp(1j * (np.pi / 2 + 0.5 * (left - right)))


def k_space(m, p):
    return [-m.exp(-1j * 2 * np.pi * p[0] * p[1] * p[2] / C_LIGHT)]


def k_space_lower(m, p):
    return k_space(m, [-p[0], p[1], p[2]])


def k_laser(m, p):
    return [m.sqrt(2 * p[0] / EPSILON_0C) * m.exp(1j * _radians(p[1]))]


def k_squeezer(m, p):
    vacuum_unit = UNIT_VACUUM / 2
    s = p[0] / (20 * np.log10(np.e))
    phase = m.exp(1j * 2 * _radians(p[1]))
    upper_diagonal = vacuum_unit * m.cosh(2 * s)
    upper_off = vacuum_unit * m.sinh(2 * s) * phase
    return [upper_diagonal, upper_off, m.conj(upper_diagonal), m.conj(upper_off)]


def k_signal_function(m, p):
    return [p[0] * m.exp(1j * _radians(p[1]))]


def k_constant(value):
    def kernel(m, p):
        return [value]
    return kernel


def k_laser_frequency_modulation(m, p):
    return [EPSILON_0C * 0.5 / p[0]]


def k_laser_frequency_modulation_lower(m, p):
    return k_laser_frequency_modulation(m, [-p[0]])


def k_space_modulation(m, p):
    w_g = 2 * np.pi * p[0]
    w_0 = 2 * np.pi * F0
    m_g = -0.5 * w_0 / w_g * m.sin(w_g * p[1] * p[2] / 2 / C_LIGHT)
    phi_sb = -w_g * p[1] * p[2] / 2 / C_LIGHT
    return [1j * m_g * m.exp(1j * phi_sb)]


def k_space_modulation_lower(m, p):
    return k_space_modulation(m, [-p[0], p[1], p[2]])


def k_susceptibility(m, p):
    factor = -1 / (p[1] * (2 * np.pi * p[0]) ** 2)
    return [-factor]


def k_force_left(m, p):
    return [-(_ial(m, p[0]) / (C_LIGHT * X_SCALE))]


def k_force_right(m, p):
    return [_iar(m, p[0], p[1], p[2]) / (C_LIGHT * X_SCALE)]


def _tuning_correction(m, p):
    return m.exp(1j * _radians(p[1]) * p[0] / F0)


def k_corrected_om_left(m, p):
    r = (1 - p[3]) * p[2]
    om = 1j * X_SCALE * 2 * np.pi / LAMBDA * _ial(m, p[5])
    factor = _tuning_correction(m, p) * om * _refl_left(m, r, p[1], p[0], p[4], p[5])
    return [-factor]


def k_corrected_om_right(m, p):
    r = (1 - p[3]) * p[2]
    om = 1j * X_SCALE * 2 * np.pi / LAMBDA * _iar(m, p[6], p[4], p[5])
    factor = m.conj(_tuning_correction(m, p)) * -1 * om * _refl_right(m, r, p[1], p[0], p[4], p[5], p[6])
    return [-factor]


def k_surface(m, p):
    r = (1 - p[0]) * p[1]
    t = (1 - p[0]) * (1 - p[1])
    return [_refl_left(m, r, p[2], p[3], p[4], p[6]),
            _transmissivity(m, t, p[2], p[3], p[4], p[5], p[6]),
            _refl_right(m, r, p[2], p[3], p[4], p[5], p[6])]


KERNELS = {
    _c.force_calculation_right: k_force_right,
    _c.laser: k_laser,
    _c.surface: k_surface,
    _c.space: k_space,
    _c.space_modulation: k_space_modulation,
    _c.signal_function: k_signal_function,
    _c.dummy_function: k_constant(1.0),
    _c.vacuum_quantum_noise: k_constant(UNIT_VACUUM / 2),
    _c.loss_quantum_noise: lambda m, p: [p[0] / 2],
    _c.space_lower: k_space_lower,
    _c.laser_amplitude_modulation: k_constant(EPSILON_0C * 0.5),
    _c.laser_frequency_modulation: k_laser_frequency_modulation,
    _c.susceptibility: k_susceptibility,
    _c.force_calculation_left: k_force_left,
    _c.corrected_optomechanical_phase_left: k_corrected_om_left,
    _c.corrected_optomechanical_phase_right: k_corrected_om_right,
    _c.space_modulation_lower: k_space_modulation_lower,
    _c.squeezer: k_squeezer,
    _c.laser_frequency_modulation_lower: k_laser_frequency_modulation_lower,
}


class NativeFunction:
    """Same interface as host.TracedFunction for a kernel f: C^7 -> C^4 (batched)."""

    out_shape = (4,)
    out_dtype = np.dtype(np.complex128)
    unsupported = frozenset()

    def __init__(self, kernel, in_shape=(7,)):
        self.kernel = kernel
        self.in_shape = tuple(in_shape)

    def _columns(self, x):
        x = np.asarray(x, dtype=np.complex128)
        return x, [x[:, i] for i in range(self.in_shape[0])]

    def __call__(self, x):
        x, cols = self._columns(x)
        with np.errstate(all="ignore"):
            outs = self.kernel(_ValueOps, cols)
        y = np.zeros((x.shape[0], 4), np.complex128)
        for k, o in enumerate(outs):
            y[:, k] = o
        return y

    def vjp(self, x, ct):
        x, cols = self._columns(x)
        ct = np.asarray(ct, dtype=np.complex128)
        duals = [Dual(c, {i: 1.0}) for i, c in enumerate(cols)]
        with np.errstate(all="ignore"):
            outs = self.kernel(_DualOps, duals)
        g = np.zeros(x.shape, np.complex128)
        for k, o in enumerate(outs):
            if not isinstance(o, Dual):
                continue
            ck = ct[:, k]
            for i, d in o.a.items():
                g[:, i] += d * ck
            for i, d in o.b.items():
                g[:, i] += np.conj(d * ck)
        return g


def native_function(function):
    """NativeFunction for an unmodified built-in component function, else None.

    DIFFEROMETOR_NATIVE=none disables the kernels (JAX tracing for everything);
    a comma-separated list of function names restricts them (used for ablations).
    """
    import os
    allowed = os.environ.get("DIFFEROMETOR_NATIVE", "all")
    if allowed != "all" and getattr(function, "__name__", None) not in allowed.split(","):
        return None
    try:
        kernel = KERNELS.get(function)
    except TypeError:  # unhashable callable
        return None
    return None if kernel is None else NativeFunction(kernel)
