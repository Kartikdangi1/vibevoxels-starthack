"""
JAX-based optical simulation runner.

This module builds linear systems from a `differometor` `Setup` and solves:
- a carrier (steady-state) system,
- a signal (sideband) system,
- and a quantum noise propagation system.

The main public entry points are:
- `run`: build once and simulate once
- `run_setups`: run multiple setups over a shared frequency grid
- `run_with_parameter_sets`: run the same setup over many optimized-parameter sets

Conventions used throughout:
- Arrays are JAX arrays (`jax.numpy`).
- Shapes are documented explicitly for every function.
- All indices are integer arrays that index into parameter vectors and matrix entries.
"""

import os
import jax
from time import time
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from jax import lax, vmap, jit

from differometor.setups import Setup
from differometor.build import build, pairs_to_arrays, prepare_arrays, LINKING_FUNCTIONS
from differometor.components import FUNCTIONS, UNIT_VACUUM, H_PLANCK, F0


jax.config.update("jax_enable_x64", True)


def solve(matrix: jnp.ndarray, right_hand_side: jnp.ndarray) -> jnp.ndarray:
    """
    Solve a dense linear system for each batch item.

    Parameters
    ----------
    matrix : jnp.ndarray
        System matrix with shape (..., system_size, system_size).
    right_hand_side : jnp.ndarray
        Right-hand side vector with shape (..., system_size).

    Returns
    -------
    solution : jnp.ndarray
        Solution vector with shape (..., system_size).

    Notes
    -----
    Both inputs are cast to complex128 to reduce numerical issues during solving.

    The function intentionally returns the raw complex solution rather than derived
    quantities (such as phase or amplitude), because some downstream transforms
    (notably `jnp.angle`) can produce invalid gradients in some configurations.
    """
    right_hand_side = right_hand_side.astype(jnp.complex128)
    matrix = matrix.astype(jnp.complex128)
    return jnp.linalg.solve(matrix, right_hand_side)


def update(
    parameters: jnp.ndarray,
    matrix: jnp.ndarray,
    function_input_indices: jnp.ndarray,
    output_indices: tuple[jnp.ndarray, jnp.ndarray],
    matrix_indices: tuple[jnp.ndarray, jnp.ndarray],
    function_indices: jnp.ndarray,
) -> jnp.ndarray:
    """
    Update selected entries of a system matrix by evaluating component functions.

    This function takes a parameter vector, selects the subset that serves as
    inputs to component transfer functions, evaluates those functions, selects
    certain outputs, and writes them into `matrix` at specified indices.

    Parameters
    ----------
    parameters : jnp.ndarray
        Parameter values for one simulation instance, shape (P,).
    matrix : jnp.ndarray
        Base matrix to update, shape (N, N + 1) for augmented systems, or (N, N).
        The function writes entries in-place (functionally) using `.at[...].set(...)`.
    function_input_indices : jnp.ndarray
        Indices into `parameters` selecting inputs to the component functions,
        shape (K,).
    output_indices : tuple[jnp.ndarray, jnp.ndarray]
        Pair of index arrays selecting entries from the evaluated output matrix.
        Each element has shape (M,). Together they define M output positions.
    matrix_indices : tuple[jnp.ndarray, jnp.ndarray]
        Pair of index arrays selecting where to write into `matrix`.
        Each element has shape (M,).
    function_indices : jnp.ndarray
        Indices selecting which function to use for each input, shape (K,).
        Values index into the global `FUNCTIONS` list.

    Returns
    -------
    updated_matrix : jnp.ndarray
        Matrix with updated entries, same shape as `matrix`.

    Notes
    -----
    `lax.switch` selects functions from the global `FUNCTIONS` list. In JAX, a
    Python list of callables referenced as a global is treated as static during
    tracing, which keeps the function jittable as long as the list is stable.
    """
    function_inputs = parameters[function_input_indices]
    output_matrix = vmap(lambda i, x: lax.switch(i, FUNCTIONS, x))(function_indices, function_inputs)
    outputs = output_matrix[output_indices[0], output_indices[1]]
    return matrix.at[matrix_indices[0], matrix_indices[1]].set(outputs)


def expand_parameters(array: jnp.ndarray, indices: jnp.ndarray, values: jnp.ndarray) -> jnp.ndarray:
    """
    Expand a single parameter row into multiple rows by replacing selected entries.

    Parameters
    ----------
    array : jnp.ndarray
        Base parameter array with shape (1, P).
    indices : jnp.ndarray
        Column indices to replace, shape (V,).
    values : jnp.ndarray
        Replacement values stacked row-wise for each index, shape (V, R),
        where R is the number of expanded rows.

        Example:
        - array shape (1, 5)
        - indices shape (2,)
        - values shape (2, 3)
        Output has shape (3, 5), where the two indexed columns are filled from
        `values[:, row]`.

    Returns
    -------
    expanded : jnp.ndarray
        Expanded parameter array with shape (R, P).

    Notes
    -----
    This implementation uses indexed updates that remain compatible with JAX
    transformations and compilation.
    """
    tiled_array = jnp.tile(array, (values.shape[1], 1))
    row_indices = jnp.repeat(jnp.arange(values.shape[1]), len(indices))
    column_indices = jnp.tile(indices, values.shape[1])
    flat_values = values.T.flatten()
    return tiled_array.at[row_indices, column_indices].set(flat_values)


def simulate_dense(
    # prepared arrays
    optimized_parameters: jnp.ndarray,
    optimized_parameter_indices: jnp.ndarray,
    optimized_value_indices: jnp.ndarray,
    carrier_changing_parameter_indices: jnp.ndarray,
    carrier_changing_values: jnp.ndarray,
    signal_changing_parameter_indices: jnp.ndarray,
    signal_changing_values: jnp.ndarray,
    # matrices
    parameters: jnp.ndarray,
    carrier_matrix: jnp.ndarray,
    signal_matrix: jnp.ndarray,
    noise_matrix: jnp.ndarray,
    noise_selection_vectors: jnp.ndarray,
    qhd_parameter_indices: jnp.ndarray,
    qhd_placing_indices: jnp.ndarray,
    linked_indices: jnp.ndarray,
    linking_function_indices: jnp.ndarray,
    indices_to_link: jnp.ndarray,
    # carrier arrays
    carrier_function_input_indices: jnp.ndarray,
    carrier_function_indices: jnp.ndarray,
    carrier_output_indices: tuple[jnp.ndarray, jnp.ndarray],
    carrier_matrix_indices: tuple[jnp.ndarray, jnp.ndarray],
    # signal arrays
    signal_function_input_indices: jnp.ndarray,
    signal_function_indices: jnp.ndarray,
    signal_output_indices: tuple[jnp.ndarray, jnp.ndarray],
    signal_matrix_indices: tuple[jnp.ndarray, jnp.ndarray],
    signal_carrier_indices: jnp.ndarray,
    signal_carrier_placing_indices: tuple[jnp.ndarray, jnp.ndarray],
    # noise arrays
    noise_function_input_indices: jnp.ndarray,
    noise_function_indices: jnp.ndarray,
    noise_output_indices: tuple[jnp.ndarray, jnp.ndarray],
    noise_system_matrix_indices: tuple[jnp.ndarray, jnp.ndarray],
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Run one full simulation pass: carrier, signal, and quantum noise.

    Parameters
    ----------
    optimized_parameters : jnp.ndarray
        Optimized parameter values provided by an optimizer or sampling routine,
        shape (OPV,) or (num_ranges, OPV) in outer vectorization contexts.
    optimized_parameter_indices : jnp.ndarray
        Indices into `parameters` for parameters that are driven by optimized values,
        shape (OP,).
    optimized_value_indices : jnp.ndarray
        Indices into `optimized_parameters` mapping each optimized parameter to
        its source value, shape (OP,). This supports shared optimized values.
    carrier_changing_parameter_indices : jnp.ndarray
        Indices into `parameters` that change for the carrier sweep, shape (CCP,).
    carrier_changing_values : jnp.ndarray
        Values for those carrier-changing parameters, shape (CCP, CV).
    signal_changing_parameter_indices : jnp.ndarray
        Indices into `parameters` that change for the signal sweep, shape (SCP,).
    signal_changing_values : jnp.ndarray
        Values for those signal-changing parameters, shape (SCP, SV).

    parameters : jnp.ndarray
        Base parameter array, shape (1, P). It is expanded internally.
    carrier_matrix : jnp.ndarray
        Base carrier augmented matrix, shape (CN, CN + 1).
    signal_matrix : jnp.ndarray
        Base signal augmented matrix, shape (SN, SN + 1).
    noise_matrix : jnp.ndarray
        Noise source covariance matrix, shape (NN, NN) or broadcastable for vmap.
    noise_selection_vectors : jnp.ndarray
        Detector selection vectors, shape (D, 1, setup_size).
    qhd_parameter_indices : jnp.ndarray
        Indices of quadrature-homodyne-detector phase parameters in `parameters`,
        shape (Q,).
    qhd_placing_indices : jnp.ndarray
        Indices describing where detector phases are applied into
        `noise_selection_vectors`, shape (Q, 3). Each row contains:
        (detector_index, row_index, column_index).
    linked_indices : jnp.ndarray
        Parameter indices used as inputs to linking functions, shape (L,).
    linking_function_indices : jnp.ndarray
        Function selectors for linking functions, shape (L,).
    indices_to_link : jnp.ndarray
        Parameter indices to receive outputs of linking functions, shape (L,).

    carrier_function_input_indices, carrier_function_indices, carrier_output_indices,
    carrier_matrix_indices :
        Indexing and function-selection arrays used by `update` to fill the carrier matrix.

    signal_function_input_indices, signal_function_indices, signal_output_indices,
    signal_matrix_indices :
        Indexing and function-selection arrays used by `update` to fill the signal matrix.

    signal_carrier_indices : jnp.ndarray
        Indices selecting the carrier solution entries used to scale signal connectors.
    signal_carrier_placing_indices : tuple[jnp.ndarray, jnp.ndarray]
        Matrix indices locating the signal connector entries that must be scaled by carrier.

    noise_function_input_indices, noise_function_indices, noise_output_indices,
    noise_system_matrix_indices :
        Indexing and function-selection arrays used by `update` to fill the noise matrix.

    Returns
    -------
    carrier : jnp.ndarray
        Carrier solution, transposed to shape (CN, CV).
    signal : jnp.ndarray
        Signal solution, transposed to shape (SN, SV).
    noise : jnp.ndarray
        Quantum noise amplitude spectral density, shape (D, SV) after squeeze.

    Notes
    -----
    - Linking functions are applied before parameter expansion so they operate on a single row.
    - The signal path is always executed (even if effectively empty) to avoid conditionals
      that would force compilation of branches with empty shapes.
    - Conjugation and block manipulation follow the conventions used for upper and lower
      sidebands in frequency-domain interferometer models.
    """
    # Insert optimized parameter values into the base parameter row.
    parameters = parameters.at[[0], optimized_parameter_indices].set(
        optimized_parameters[optimized_value_indices]
    )

    # Apply parameter linking (derived parameters).
    inputs_to_link = parameters[0, linked_indices]
    outputs_to_link = vmap(lambda i, x: lax.switch(i, LINKING_FUNCTIONS, x))(
        linking_function_indices, inputs_to_link
    )
    parameters = parameters.at[[0], indices_to_link].set(outputs_to_link)

    # Quadrature-homodyne-detector phase factors (unit magnitude complex phase).
    qhd_phase_values = jnp.exp(1j * jnp.radians(parameters[0, qhd_parameter_indices]))

    # --------------------
    # Carrier solve
    # --------------------
    parameters = expand_parameters(parameters, carrier_changing_parameter_indices, carrier_changing_values)
    carrier_matrix = vmap(update, in_axes=(0, None, None, None, None, None))(
        parameters,
        carrier_matrix,
        carrier_function_input_indices,
        carrier_output_indices,
        carrier_matrix_indices,
        carrier_function_indices,
    )
    carrier = vmap(solve)(carrier_matrix[:, :, :-1], carrier_matrix[:, :, -1])

    # --------------------
    # Signal solve
    # --------------------
    parameters = expand_parameters(parameters, signal_changing_parameter_indices, signal_changing_values)
    signal_matrix = vmap(update, in_axes=(0, None, None, None, None, None))(
        parameters,
        signal_matrix,
        signal_function_input_indices,
        signal_output_indices,
        signal_matrix_indices,
        signal_function_indices,
    )

    # Scale signal connector entries by the carrier solution at the respective spaces.
    # The minus sign matches the sign convention used by the referenced modeling tools.
    signal_entries = (
        -signal_matrix[:, signal_carrier_placing_indices[0], signal_carrier_placing_indices[1]]
        * carrier[:, signal_carrier_indices.flatten()]
    )
    signal_matrix = signal_matrix.at[:, signal_carrier_placing_indices[0], signal_carrier_placing_indices[1]].set(
        signal_entries
    )

    # Apply conjugation rules to enforce upper and lower sideband symmetry.
    carrier_size = carrier.shape[1]
    signal_matrix = signal_matrix.at[:, carrier_size : carrier_size * 2, carrier_size : carrier_size * 2].set(
        jnp.conjugate(signal_matrix[:, carrier_size : carrier_size * 2, carrier_size : carrier_size * 2])
    )
    signal_matrix = signal_matrix.at[:, carrier_size * 2 :, :carrier_size].set(
        jnp.conjugate(signal_matrix[:, carrier_size * 2 :, :carrier_size])
    )
    signal_matrix = signal_matrix.at[:, carrier_size : carrier_size * 2, carrier_size * 2 :].set(
        jnp.conjugate(signal_matrix[:, carrier_size : carrier_size * 2, carrier_size * 2 :])
    )

    signal = vmap(solve)(signal_matrix[:, :, :-1], signal_matrix[:, :, -1])

    # --------------------
    # Quantum noise solve
    # --------------------
    # Apply quadrature-homodyne-detector phases into the selection vectors.
    # `lax.scatter_mul` is used (rather than `.at[...].set`) so empty index arrays
    # remain valid without conditionals.
    noise_selection_vectors = lax.scatter_mul(
        noise_selection_vectors,
        qhd_placing_indices.reshape(-1, 3),
        qhd_phase_values,
        lax.ScatterDimensionNumbers(
            update_window_dims=(),
            inserted_window_dims=(0, 1, 2),
            scatter_dims_to_operand_dims=(0, 1, 2),
        ),
        unique_indices=True,
    )

    # Broadcast selection vectors across the carrier sweep dimension.
    noise_selection_vectors = jnp.tile(noise_selection_vectors, (1, carrier.shape[0], 1))

    # Select necessary carrier entries and build upper and lower sideband selections.
    upper_carrier_selections = noise_selection_vectors[:, :, : carrier.shape[1]] * carrier
    upper_carrier_selections = noise_selection_vectors.at[:, :, : carrier.shape[1]].set(upper_carrier_selections)

    lower_carrier_selections = jnp.conjugate(
        upper_carrier_selections[:, :, carrier.shape[1] : carrier.shape[1] * 2] * carrier
    )
    carrier_selections = upper_carrier_selections.at[
        :, :, carrier.shape[1] : carrier.shape[1] * 2
    ].set(lower_carrier_selections)

    # Back-propagate detector weights by solving the transposed conjugated signal system.
    transposed_conjugated_signal_matrix = jnp.transpose(jnp.conjugate(signal_matrix[:, :, :-1]), (0, 2, 1))
    carrier_selections = jnp.broadcast_to(
        carrier_selections,
        (carrier_selections.shape[0], transposed_conjugated_signal_matrix.shape[0], carrier_selections.shape[2]),
    )
    noise_weights = vmap(vmap(solve), in_axes=(None, 0))(transposed_conjugated_signal_matrix, carrier_selections)

    # Update noise source matrix (may depend on current parameters).
    noise_matrix = vmap(update, in_axes=(0, None, None, None, None, None))(
        parameters,
        noise_matrix,
        noise_function_input_indices,
        noise_output_indices,
        noise_system_matrix_indices,
        noise_function_indices,
    )

    def matmul_single_batch(matrix_batch: jnp.ndarray, vector_batch: jnp.ndarray) -> jnp.ndarray:
        """
        Multiply a batch of matrices with a batch of vectors.

        Parameters
        ----------
        matrix_batch : jnp.ndarray
            Shape (B, N, N).
        vector_batch : jnp.ndarray
            Shape (B, N).

        Returns
        -------
        product : jnp.ndarray
            Shape (B, N).
        """
        return jnp.einsum("bij,bj->bi", matrix_batch, vector_batch)

    # Project noise sources into the system and propagate through the signal matrix.
    noise_sources = vmap(matmul_single_batch, in_axes=(None, 0))(noise_matrix, noise_weights)
    # Convert propagated covariances into amplitude spectral density.
    temporary_sum = jnp.sum(
    jnp.real(noise_sources * jnp.conjugate(noise_weights)),
    axis=2
    )    
    noise = jnp.sqrt(2 * temporary_sum * UNIT_VACUUM * H_PLANCK * F0 * 0.25)

    return carrier.T, signal.T, noise.squeeze()



# =====================================================================
# Sparse path (default): compressed matrix entries + SuperLU with custom VJP
# =====================================================================
#
# The optical matrices are extremely sparse (public UIFO: signal 751x751 with
# 2537 nonzeros, carrier 286x286 with 680, noise 751x751 with 272). Instead of
# materialising dense (F, N, N+1) arrays and running dense LU in XLA, the build
# step compresses every structurally relevant matrix position to a flat index;
# `simulate` only computes those entries, and the linear systems are solved by
# SuperLU on the host through `jax.pure_callback`. Reverse mode is a hand-written
# adjoint (custom VJP) that reuses the same factorisation:
#   x = A^{-1} b        ->  b_bar = A^{-T} x_bar,            A_bar -= b_bar x^T
#   w = A^{-H} c        ->  c_bar = conj(A^{-1} conj(w_bar)), A_bar -= conj(w) (A^{-1} conj(w_bar))^T
# Component functions are grouped by type at build time (static dict keys), so
# no `lax.switch` over all component functions is traced, and entries whose
# inputs do not reference swept parameters are evaluated once, not per frequency.

from collections import OrderedDict


def _compress(base, write_sets):
    """Map each structurally relevant (row, col) of `base` to a flat index.

    Returns rows, cols, base values and, for each write set, the flat indices
    of its positions. Negative (wrap-around) indices are normalised.
    """
    base = np.asarray(base)
    rows, cols = np.nonzero(base)
    extra = [np.asarray(ws).reshape(2, -1) % np.array(base.shape)[:, None] for ws in write_sets]
    all_rows = np.concatenate([rows] + [e[0] for e in extra])
    all_cols = np.concatenate([cols] + [e[1] for e in extra])
    ncols = base.shape[1]
    keys = np.unique(all_rows * ncols + all_cols)
    r, c = keys // ncols, keys % ncols
    return r, c, base[r, c], [np.searchsorted(keys, e[0] * ncols + e[1]) for e in extra]


def _matrix_first(r, c, v, ks, n):
    """Reorder so that system-matrix entries (col < n) precede RHS entries (col == n)."""
    order = np.argsort(c >= n, kind="stable")
    inverse = np.empty_like(order)
    inverse[order] = np.arange(order.size)
    return r[order], c[order], v[order], [inverse[k] for k in ks], int(np.sum(c < n))


def _group_functions(function_input_indices, function_indices, output_indices, flat_indices, swept):
    """Group component-function evaluations by function type and sweep dependence.

    Returns {key: dict(inputs, rows, slots, k)} where key = '<c|v><function index>'.
    'c' groups do not read any swept parameter and are evaluated once.
    Later writes to the same flat index win, as in the original scatter order.
    """
    fin = np.asarray(function_input_indices).reshape(-1, np.asarray(function_input_indices).shape[-1]) \
        if np.asarray(function_input_indices).size else np.zeros((0, 7), np.int64)
    fidx = np.asarray(function_indices).reshape(-1)
    out = np.asarray(output_indices).reshape(2, -1)
    k = np.asarray(flat_indices).reshape(-1)
    # keep only the last write to each flat index
    _, last = np.unique(k[::-1], return_index=True)
    keep = np.sort(k.size - 1 - last)
    out, k = out[:, keep], k[keep]
    varying = np.isin(fin, np.asarray(swept)).any(axis=1) if fin.size else np.zeros(0, bool)
    groups = {}
    for g in np.unique(fidx):
        for tag in ("c", "v"):
            rows_g = np.nonzero((fidx == g) & (varying == (tag == "v")))[0]
            if rows_g.size == 0:
                continue
            sel = np.isin(out[0], rows_g)
            if not sel.any():
                continue
            local = np.searchsorted(rows_g, out[0, sel])
            groups[f"{tag}{int(g):02d}"] = dict(
                inputs=jnp.asarray(fin[rows_g]), rows=jnp.asarray(local),
                slots=jnp.asarray(out[1, sel]), k=jnp.asarray(k[sel]),
            )
    return groups


def compress_arrays(arrays):
    """Convert dense-matrix arrays from the builder into the sparse `simulate` inputs."""
    a = dict(arrays)
    swept = np.concatenate([np.asarray(a["carrier_changing_parameter_indices"]).reshape(-1),
                            np.asarray(a["signal_changing_parameter_indices"]).reshape(-1)])
    out = {}

    carrier_matrix = np.asarray(a["carrier_matrix"])
    cn = carrier_matrix.shape[0]
    r, c, v, ks, na = _matrix_first(*_compress(carrier_matrix, [a["carrier_matrix_indices"]]), cn)
    out.update(
        carrier_values=jnp.asarray(v.astype(np.complex128)), carrier_rows=jnp.asarray(r[:na]),
        carrier_cols=jnp.asarray(c[:na]), carrier_rhs_rows=jnp.asarray(r[na:]),
        carrier_size_marker=jnp.asarray(np.zeros(cn, np.int8)),
        carrier_groups=_group_functions(a["carrier_function_input_indices"], a["carrier_function_indices"],
                                        a["carrier_output_indices"], ks[0], swept),
    )

    signal_matrix = np.asarray(a["signal_matrix"])
    sn = signal_matrix.shape[0]
    r, c, v, ks, na = _matrix_first(*_compress(
        signal_matrix, [a["signal_matrix_indices"], a["signal_carrier_placing_indices"]]), sn)
    conj = (((r >= cn) & (r < 2 * cn) & (c >= cn) & (c < 2 * cn))
            | ((r >= 2 * cn) & (c < cn))
            | ((r >= cn) & (r < 2 * cn) & (c >= 2 * cn)))
    out.update(
        signal_values=jnp.asarray(v.astype(np.complex128)), signal_rows=jnp.asarray(r[:na]),
        signal_cols=jnp.asarray(c[:na]), signal_rhs_rows=jnp.asarray(r[na:]),
        signal_place_k=jnp.asarray(ks[1]), signal_conj_mask=jnp.asarray(conj),
        signal_groups=_group_functions(a["signal_function_input_indices"], a["signal_function_indices"],
                                       a["signal_output_indices"], ks[0], swept),
    )

    noise_matrix = np.asarray(a["noise_matrix"])
    r, c, v, ks = _compress(noise_matrix, [a["noise_system_matrix_indices"]])
    out.update(
        noise_values=jnp.asarray(v.astype(np.complex128)), noise_rows=jnp.asarray(r), noise_cols=jnp.asarray(c),
        noise_groups=_group_functions(a["noise_function_input_indices"], a["noise_function_indices"],
                                      a["noise_output_indices"], ks[0], swept),
    )

    keep = ("optimized_parameters", "optimized_parameter_indices", "optimized_value_indices",
            "carrier_changing_parameter_indices", "carrier_changing_values",
            "signal_changing_parameter_indices", "signal_changing_values", "parameters",
            "noise_selection_vectors", "qhd_parameter_indices", "qhd_placing_indices",
            "linked_indices", "linking_function_indices", "indices_to_link", "signal_carrier_indices")
    return {**{key: (None if a[key] is None else jnp.asarray(a[key])) for key in keep}, **out}


# ---------------------------------------------------------------- host solver
_LU_CACHE = OrderedDict()
_LU_CACHE_SIZE = 4
_LU_TOKEN = [0]


def _factor(values, rows, cols, n):
    return spl.splu(sp.csc_matrix((values, (rows, cols)), shape=(n, n)))


def _factor_all(values, rows, cols, n):
    return [_factor(values[f], rows, cols, n) for f in range(values.shape[0])]


def _forward_host(values, rows, cols, b, c):
    values, rows, cols, b, c = (np.asarray(x) for x in (values, rows, cols, b, c))
    n = b.shape[1]
    lus = _factor_all(values, rows, cols, n)
    x = np.empty(b.shape, np.complex128)
    w = np.empty(c.shape, np.complex128)
    for f, lu in enumerate(lus):
        x[f] = lu.solve(b[f])
        if c.shape[1]:
            w[f] = lu.solve(np.ascontiguousarray(c[f].T), trans="H").T
    _LU_TOKEN[0] += 1
    token = _LU_TOKEN[0]
    _LU_CACHE[token] = (lus, values)
    while len(_LU_CACHE) > _LU_CACHE_SIZE:
        _LU_CACHE.popitem(last=False)
    return x, w, np.int64(token)


def _backward_host(token, values, rows, cols, x, w, xb, wb):
    values, rows, cols, x, w, xb, wb = (np.asarray(v) for v in (values, rows, cols, x, w, xb, wb))
    n = x.shape[1]
    cached = _LU_CACHE.get(int(token))
    # Reuse the forward factorisation only if it belongs to exactly these values.
    lus = cached[0] if cached is not None and np.array_equal(cached[1], values) else _factor_all(values, rows, cols, n)
    vb = np.empty(values.shape, np.complex128)
    bb = np.empty(x.shape, np.complex128)
    cb = np.empty(w.shape, np.complex128)
    for f, lu in enumerate(lus):
        bb[f] = lu.solve(xb[f], trans="T")
        g = -bb[f][rows] * x[f][cols]
        if w.shape[1]:
            y = lu.solve(np.ascontiguousarray(np.conj(wb[f]).T)).T  # (D, n)
            cb[f] = np.conj(y)
            g = g - np.sum(np.conj(w[f][:, rows]) * y[:, cols], axis=0)
        vb[f] = g
    return vb, bb, cb


def _solve_shapes(b, c):
    return (jax.ShapeDtypeStruct(b.shape, jnp.complex128), jax.ShapeDtypeStruct(c.shape, jnp.complex128),
            jax.ShapeDtypeStruct((), jnp.int64))


@jax.custom_vjp
def sparse_solve(values, rows, cols, b, c):
    """Per batch item f: x = A^{-1} b and w = A^{-H} c (one factorisation).

    values (F, nnz) complex: entries of A at (rows, cols); b (F, n); c (F, D, n).
    """
    x, w, _ = jax.pure_callback(_forward_host, _solve_shapes(b, c), values, rows, cols, b, c,
                                vmap_method="sequential")
    return x, w


def _sparse_solve_fwd(values, rows, cols, b, c):
    x, w, token = jax.pure_callback(_forward_host, _solve_shapes(b, c), values, rows, cols, b, c,
                                    vmap_method="sequential")
    return (x, w), (token, values, rows, cols, x, w)


def _sparse_solve_bwd(residuals, cotangents):
    token, values, rows, cols, x, w = residuals
    xb, wb = cotangents
    shapes = (jax.ShapeDtypeStruct(values.shape, jnp.complex128), jax.ShapeDtypeStruct(x.shape, jnp.complex128),
              jax.ShapeDtypeStruct(w.shape, jnp.complex128))
    vb, bb, cb = jax.pure_callback(_backward_host, shapes, token, values, rows, cols, x, w, xb, wb,
                                   vmap_method="sequential")
    return vb, None, None, bb, cb


sparse_solve.defvjp(_sparse_solve_fwd, _sparse_solve_bwd)


# ---------------------------------------------------------------- assembly
def _assemble(parameters, base_values, groups):
    """Evaluate grouped component functions and write them into compressed values.

    parameters (R, P); returns (R, nnz) complex values.
    """
    rows_count = parameters.shape[0]
    values = jnp.broadcast_to(base_values, (rows_count, base_values.shape[0]))
    const_k, const_out, var_k, var_out = [], [], [], []
    for key, group in groups.items():
        function = FUNCTIONS[int(key[1:])]
        if key[0] == "c":
            evaluated = vmap(function)(parameters[0][group["inputs"]])  # (K, 4)
            const_k.append(group["k"])
            const_out.append(evaluated[group["rows"], group["slots"]])
        else:
            evaluated = vmap(vmap(function))(parameters[:, group["inputs"]])  # (R, K, 4)
            var_k.append(group["k"])
            var_out.append(evaluated[:, group["rows"], group["slots"]])
    if const_k:
        values = values.at[:, jnp.concatenate(const_k)].set(jnp.concatenate(const_out)[None, :])
    if var_k:
        values = values.at[:, jnp.concatenate(var_k)].set(jnp.concatenate(var_out, axis=1))
    return values.astype(jnp.complex128)


def simulate_sparse_jax(
    # prepared arrays
    optimized_parameters, optimized_parameter_indices, optimized_value_indices,
    carrier_changing_parameter_indices, carrier_changing_values,
    signal_changing_parameter_indices, signal_changing_values,
    parameters, noise_selection_vectors, qhd_parameter_indices, qhd_placing_indices,
    linked_indices, linking_function_indices, indices_to_link, signal_carrier_indices,
    # compressed carrier system
    carrier_values, carrier_rows, carrier_cols, carrier_rhs_rows, carrier_size_marker, carrier_groups,
    # compressed signal system
    signal_values, signal_rows, signal_cols, signal_rhs_rows, signal_place_k, signal_conj_mask, signal_groups,
    # compressed noise source matrix
    noise_values, noise_rows, noise_cols, noise_groups,
):
    """Sparse equivalent of `simulate_dense`: returns (carrier.T, signal.T, noise)."""
    parameters = parameters.at[[0], optimized_parameter_indices].set(optimized_parameters[optimized_value_indices])
    inputs_to_link = parameters[0, linked_indices]
    outputs_to_link = vmap(lambda i, x: lax.switch(i, LINKING_FUNCTIONS, x))(linking_function_indices, inputs_to_link)
    parameters = parameters.at[[0], indices_to_link].set(outputs_to_link)
    qhd_phase_values = jnp.exp(1j * jnp.radians(parameters[0, qhd_parameter_indices]))

    # Carrier
    parameters = expand_parameters(parameters, carrier_changing_parameter_indices, carrier_changing_values)
    cvals = _assemble(parameters, carrier_values, carrier_groups)
    carrier_size = carrier_size_marker.shape[0]
    n_carrier = carrier_rows.shape[0]
    carrier_rhs = jnp.zeros((cvals.shape[0], carrier_size), jnp.complex128).at[:, carrier_rhs_rows].set(cvals[:, n_carrier:])
    carrier, _ = sparse_solve(cvals[:, :n_carrier], carrier_rows, carrier_cols, carrier_rhs,
                              jnp.zeros((cvals.shape[0], 0, carrier_size), jnp.complex128))

    # Signal
    parameters = expand_parameters(parameters, signal_changing_parameter_indices, signal_changing_values)
    svals = _assemble(parameters, signal_values, signal_groups)
    svals = svals.at[:, signal_place_k].set(-svals[:, signal_place_k] * carrier[:, signal_carrier_indices.flatten()])
    svals = jnp.where(signal_conj_mask, jnp.conjugate(svals), svals)
    signal_size = noise_selection_vectors.shape[2]
    n_signal = signal_rows.shape[0]
    signal_rhs = jnp.zeros((svals.shape[0], signal_size), jnp.complex128).at[:, signal_rhs_rows].set(svals[:, n_signal:])

    # Detector selection vectors (same as simulate_dense)
    selections = lax.scatter_mul(
        noise_selection_vectors, qhd_placing_indices.reshape(-1, 3), qhd_phase_values,
        lax.ScatterDimensionNumbers(update_window_dims=(), inserted_window_dims=(0, 1, 2),
                                    scatter_dims_to_operand_dims=(0, 1, 2)),
        unique_indices=True)
    selections = jnp.tile(selections, (1, carrier.shape[0], 1))
    upper = selections.at[:, :, :carrier_size].set(selections[:, :, :carrier_size] * carrier)
    lower = jnp.conjugate(upper[:, :, carrier_size:2 * carrier_size] * carrier)
    selections = upper.at[:, :, carrier_size:2 * carrier_size].set(lower)
    selections = jnp.broadcast_to(selections, (selections.shape[0], svals.shape[0], signal_size))

    # One factorisation per frequency: signal x = A^{-1} b and noise weights w = A^{-H} c
    signal, weights = sparse_solve(svals[:, :n_signal], signal_rows, signal_cols, signal_rhs,
                                   jnp.transpose(selections, (1, 0, 2)).astype(jnp.complex128))

    # Quantum noise: Re(w^H N w) with N in compressed form (adjoint quadratic-form identity)
    nvals = _assemble(parameters, noise_values, noise_groups)
    quadratic = jnp.sum(jnp.real(nvals[:, None, :] * jnp.conjugate(weights[:, :, noise_rows])
                                 * weights[:, :, noise_cols]), axis=2)  # (F, D)
    noise = jnp.sqrt(2 * quadratic.T * UNIT_VACUUM * H_PLANCK * F0 * 0.25)
    return carrier.T, signal.T, noise.squeeze()



def simulate(
    # prepared arrays
    optimized_parameters, optimized_parameter_indices, optimized_value_indices,
    carrier_changing_parameter_indices, carrier_changing_values,
    signal_changing_parameter_indices, signal_changing_values,
    parameters, noise_selection_vectors, qhd_parameter_indices, qhd_placing_indices,
    linked_indices, linking_function_indices, indices_to_link, signal_carrier_indices,
    # compressed carrier system
    carrier_values, carrier_rows, carrier_cols, carrier_rhs_rows, carrier_size_marker, carrier_groups,
    # compressed signal system
    signal_values, signal_rows, signal_cols, signal_rhs_rows, signal_place_k, signal_conj_mask, signal_groups,
    # compressed noise source matrix
    noise_values, noise_rows, noise_cols, noise_groups,
):
    """
    Run one full simulation pass (carrier, signal, quantum noise) on compressed arrays.

    By default the whole pass executes on the host (NumPy/SciPy SuperLU) inside a
    single custom VJP (see `differometor.host`), which removes nearly all XLA
    compilation. If a component/linking function uses a primitive the host
    interpreter does not support, or DIFFEROMETOR_BACKEND=jax is set, the
    in-graph implementation `simulate_sparse_jax` is used instead.

    Note: the host path differentiates with respect to `optimized_parameters`
    only (the other inputs are treated as constants), which is what optimisation
    and the benchmark require. Use DIFFEROMETOR_BACKEND=jax for gradients with
    respect to other inputs.
    """
    arrays = dict(locals())
    if os.environ.get("DIFFEROMETOR_BACKEND", "host") != "jax":
        from differometor.host import make_host_simulate
        host_simulate, _ = make_host_simulate(arrays)
        if host_simulate is not None:
            rest = {key: value for key, value in arrays.items() if key != "optimized_parameters"}
            return host_simulate(optimized_parameters, rest)
    return simulate_sparse_jax(**arrays)


def run_build_step(
    setup: Setup,
    changing_pairs: list | None = None,
    changing_values: jnp.ndarray | None = None,
    optimization_pairs: list | None = None,
    timeit: bool = False,
) -> tuple:
    """
    Build a `Setup` into simulation-ready arrays and matrices.

    Parameters
    ----------
    setup : Setup
        The optical setup to build.
    changing_pairs : list | None
        Each element is a (component_name, parameter_name) tuple describing which
        parameters are swept during simulation. If None, no sweep is applied.
    changing_values : jnp.ndarray | None
        Values for the parameters described by `changing_pairs`.
        Expected shape is (num_changing_parameters, num_values).
    optimization_pairs : list | None
        Each element is a (component_name, parameter_name) tuple describing which
        parameters are treated as optimized inputs (external values injected into
        the parameter vector before solving). If None, no optimized parameters exist.
    timeit : bool
        If True, return an additional timing measurement (seconds) for the build.

    Returns
    -------
    If `timeit` is False
        simulation_arrays : dict[str, jnp.ndarray]
            Mapping from `simulate` argument name to the array value.
        metadata : tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
            (detector_indices, mirror_indices, beamsplitter_indices, isolator_indices)

    If `timeit` is True
        build_seconds : float
            Time spent building and preparing arrays.
        (simulation_arrays, metadata) : tuple
            The same outputs as the non-timed case, packed into a tuple.

    Notes
    -----
    The returned dictionary keys are derived from the `simulate` function signature.
    """
    start = time()

    instructions, matrices, metadata = build(setup)
    arrays_to_prepare, carrier_arrays, signal_arrays, noise_arrays = pairs_to_arrays(
        *instructions, optimization_pairs, changing_pairs
    )
    prepared_arrays = prepare_arrays(*arrays_to_prepare, changing_values, parameters=matrices[0])

    arg_count = simulate_dense.__code__.co_argcount
    arg_names = simulate_dense.__code__.co_varnames[:arg_count]

    all_arrays = (*prepared_arrays, *matrices, *carrier_arrays, *signal_arrays, *noise_arrays)
    if len(arg_names) != len(all_arrays):
        raise ValueError(
            "simulate signature does not match the number of prepared arrays and matrices. "
            f"simulate expects {len(arg_names)} arguments, but build produced {len(all_arrays)} arrays."
        )

    simulation_arrays = compress_arrays({arg_names[i]: array for i, array in enumerate(all_arrays)})
    expected = simulate.__code__.co_varnames[:simulate.__code__.co_argcount]
    if set(simulation_arrays) != set(expected):
        raise ValueError("compressed arrays do not match the sparse simulate signature")

    if timeit:
        for array in simulation_arrays.values():
            jax.block_until_ready(array)
        return time() - start, (simulation_arrays, metadata)

    return simulation_arrays, *metadata


def run_simulation_step(
    simulation_arrays: dict,
    jit_simulation: bool = False,
    timeit: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray] | tuple[float, tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]]:
    """
    Run `simulate` using a pre-built set of arrays.

    Parameters
    ----------
    simulation_arrays : dict[str, jnp.ndarray]
        Dictionary mapping argument names to arrays. This is the first return value
        of `run_build_step`.
    jit_simulation : bool
        If True, compile `simulate` using `jax.jit` and run one warmup call to
        trigger compilation before measuring runtime.
    timeit : bool
        If True, return the simulation runtime (seconds) alongside the results.

    Returns
    -------
    If `timeit` is False
        carrier : jnp.ndarray
        signal : jnp.ndarray
        noise : jnp.ndarray

    If `timeit` is True
        simulation_seconds : float
            Time spent executing the simulation (excluding warmup compilation).
        results : tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
            (carrier, signal, noise)

    Notes
    -----
    The warmup run uses the same inputs and forces compilation when `jit_simulation`
    is enabled.
    """
    simulation_function = simulate

    if jit_simulation:
        simulation_function = jit(simulate)
        _ = simulation_function(**simulation_arrays)  # warmup compilation

    start = time()
    results = simulation_function(**simulation_arrays)

    if timeit:
        for result in results:
            jax.block_until_ready(result)
        return time() - start, results

    return results


def run(
    setup: Setup,
    changing_pairs: list | None = None,
    changing_values: jnp.ndarray | None = None,
    optimization_pairs: list | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Convenience wrapper: build a setup and immediately run the simulation.

    Parameters
    ----------
    setup : Setup
        The setup to simulate.
    changing_pairs : list | None
        Sweep definition as (component_name, parameter_name) pairs.
    changing_values : jnp.ndarray | None
        Sweep values, shape (num_changing_parameters, num_values).
    optimization_pairs : list | None
        Optimized parameter definition as (component_name, parameter_name) pairs.

    Returns
    -------
    carrier : jnp.ndarray
        Carrier solution, shape (num_ports_carrier, num_carrier_values).
    signal : jnp.ndarray
        Signal solution, shape (num_ports_signal, num_signal_values).
    noise : jnp.ndarray
        Noise amplitude spectral density, shape (num_detectors, num_signal_values) after squeeze.
    detector_indices : jnp.ndarray
    mirror_indices : jnp.ndarray
    beamsplitter_indices : jnp.ndarray
    isolator_indices : jnp.ndarray
        Port indices in the order defined by the setup.
    """
    simulation_arrays, *metadata = run_build_step(setup, changing_pairs, changing_values, optimization_pairs)
    carrier, signal, noise = run_simulation_step(simulation_arrays)
    return carrier, signal, noise, *metadata


def run_setups(setups: list[Setup], frequencies: jnp.ndarray):
    """
    Run a list of setups over a shared frequency sweep.

    Parameters
    ----------
    setups : list[Setup]
        Setups to simulate.
    frequencies : jnp.ndarray
        Frequency values. This is passed as the sweep values for the pair ("f", "frequency").
        Expected shape is (1, num_frequencies) or broadcastable to the builder expectations.

    Returns
    -------
    results : list
        Each element is the return value of `run(setup, [("f", "frequency")], frequencies)`.
    """
    results = []
    for setup in setups:
        results.append(run(setup, [("f", "frequency")], frequencies))
    return results
