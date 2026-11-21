"""Hashable, compile-time simulation metadata built from a Setup."""

from dataclasses import dataclass

import jax
import numpy as np


@dataclass(frozen=True)
class FunctionGroup:
    function_id: int
    input_indices: tuple[tuple[int, ...], ...]
    output_rows: tuple[int, ...]
    output_columns: tuple[int, ...]
    matrix_rows: tuple[int, ...]
    matrix_columns: tuple[int, ...]


@dataclass(frozen=True)
class MatrixUpdatePlan:
    groups: tuple[FunctionGroup, ...]
    static_groups: tuple[FunctionGroup, ...] = ()
    frequency_groups: tuple[FunctionGroup, ...] = ()


@jax.tree_util.register_static
@dataclass(frozen=True)
class SimulationPlan:
    carrier: MatrixUpdatePlan
    signal: MatrixUpdatePlan
    noise: MatrixUpdatePlan
    linked_groups: tuple[tuple[object, tuple[int, ...], tuple[int, ...]], ...]
    qhd_parameter_indices: tuple[int, ...]
    qhd_placing_indices: tuple[tuple[int, int, int], ...]
    linked_indices: tuple[int, ...]
    indices_to_link: tuple[int, ...]
    signal_carrier_indices: tuple[int, ...]
    signal_carrier_rows: tuple[int, ...]
    signal_carrier_columns: tuple[int, ...]
    optimized_parameter_indices: tuple[int, ...]
    optimized_value_indices: tuple[int, ...]
    carrier_changing_parameter_indices: tuple[int, ...]
    signal_changing_parameter_indices: tuple[int, ...]


def _matrix_update_plan(
    function_inputs, function_ids, output_indices, matrix_indices,
    frequency_parameter_indices=(),
):
    inputs = np.asarray(function_inputs, dtype=np.int32)
    functions = np.asarray(function_ids, dtype=np.int32)
    out_rows, out_cols = np.asarray(output_indices, dtype=np.int32)
    matrix_rows, matrix_cols = np.asarray(matrix_indices, dtype=np.int32)

    frequency_parameter_indices = set(map(int, frequency_parameter_indices))
    groups = []
    static_groups = []
    frequency_groups = []
    for function_id in dict.fromkeys(functions.tolist()):
        function_calls = np.flatnonzero(functions == function_id)
        is_frequency_dependent = [
            any(int(index) in frequency_parameter_indices for index in inputs[call])
            for call in function_calls
        ]
        for dependent in (False, True):
            calls = [
                int(call) for call, is_dependent in zip(function_calls, is_frequency_dependent)
                if is_dependent == dependent
            ]
            if not calls:
                continue
            call_lookup = {call: index for index, call in enumerate(calls)}
            selected_outputs = [
                index for index, call in enumerate(out_rows.tolist())
                if int(call) in call_lookup and int(functions[int(call)]) == int(function_id)
            ]
            group = FunctionGroup(
                function_id=int(function_id),
                input_indices=tuple(tuple(map(int, inputs[index])) for index in calls),
                output_rows=tuple(call_lookup[int(out_rows[index])] for index in selected_outputs),
                output_columns=tuple(int(out_cols[index]) for index in selected_outputs),
                matrix_rows=tuple(int(matrix_rows[index]) for index in selected_outputs),
                matrix_columns=tuple(int(matrix_cols[index]) for index in selected_outputs),
            )
            groups.append(group)
            (frequency_groups if dependent else static_groups).append(group)
    return MatrixUpdatePlan(tuple(groups), tuple(static_groups), tuple(frequency_groups))


def make_simulation_plan(
    carrier_arrays, signal_arrays, noise_arrays, matrices, link_functions,
    prepared_arrays, frequency_parameter_indices=(),
):
    carrier = _matrix_update_plan(*carrier_arrays[:4])
    signal = _matrix_update_plan(*signal_arrays[:4], frequency_parameter_indices)
    noise = _matrix_update_plan(*noise_arrays[:4], frequency_parameter_indices)

    linked_indices = tuple(map(int, np.asarray(matrices[7]).reshape(-1)))
    function_ids = np.asarray(matrices[8], dtype=np.int32).reshape(-1)
    targets = tuple(map(int, np.asarray(matrices[9]).reshape(-1)))
    linked_groups = []
    for function_id in dict.fromkeys(function_ids.tolist()):
        positions = np.flatnonzero(function_ids == function_id)
        linked_groups.append((
            link_functions[int(function_id)],
            tuple(linked_indices[index] for index in positions),
            tuple(targets[index] for index in positions),
        ))

    return SimulationPlan(
        carrier=carrier,
        signal=signal,
        noise=noise,
        linked_groups=tuple(linked_groups),
        qhd_parameter_indices=tuple(map(int, np.asarray(matrices[5]).reshape(-1))),
        qhd_placing_indices=tuple(
            tuple(map(int, row)) for row in np.asarray(matrices[6]).reshape(-1, 3)
        ),
        linked_indices=linked_indices,
        indices_to_link=targets,
        signal_carrier_indices=tuple(map(int, np.asarray(signal_arrays[4]).reshape(-1))),
        signal_carrier_rows=tuple(map(int, np.asarray(signal_arrays[5])[0].reshape(-1))),
        signal_carrier_columns=tuple(map(int, np.asarray(signal_arrays[5])[1].reshape(-1))),
        optimized_parameter_indices=tuple(map(int, np.asarray(prepared_arrays[1]).reshape(-1))),
        optimized_value_indices=tuple(map(int, np.asarray(prepared_arrays[2]).reshape(-1))),
        carrier_changing_parameter_indices=tuple(map(int, np.asarray(prepared_arrays[3]).reshape(-1))),
        signal_changing_parameter_indices=tuple(map(int, np.asarray(prepared_arrays[5]).reshape(-1))),
    )
