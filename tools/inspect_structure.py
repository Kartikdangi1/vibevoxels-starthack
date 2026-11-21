"""Inspect the built signal matrix and verify a cell-interior Schur partition.

Run with the project environment from the repository root:
    python tools/inspect_structure.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from differometor.setups import Setup
from differometor.simulate import run_build_step


CELL_NODE = re.compile(r"(?:center|ml|mr|mt|mb)(\d+)(\d+)$")


def cell_id(name: str):
    match = CELL_NODE.fullmatch(name)
    return tuple(map(int, match.groups())) if match else None


def main():
    repo = Path(__file__).resolve().parents[1]
    data_dir = repo / "examples" / "data"
    setup_data = json.loads((data_dir / "uifo_800_3000.json").read_text())
    reference = json.loads((data_dir / "uifo_800_3000_test_data.json").read_text())
    setup = Setup.from_data(setup_data)
    arrays, *_metadata = run_build_step(
        setup, [("f", "frequency")], np.asarray(reference["frequencies"])
    )
    port_to_index = _metadata[4]
    signal_matrix = np.asarray(arrays["signal_matrix"])
    plan = arrays["update_plan"]
    system_size = np.asarray(arrays["carrier_matrix"]).shape[0]
    total_size = signal_matrix.shape[0]
    signal_start = system_size * 2

    # Structural support includes constant entries already in the base matrix
    # plus every location that a parameter-dependent update may write.
    support = set(zip(*np.nonzero(signal_matrix[:, :-1])))
    for group in plan.signal.groups:
        support.update(zip(group.matrix_rows, group.matrix_columns))
    support.update(zip(plan.signal_carrier_rows, plan.signal_carrier_columns))

    block_names = ("U-U", "U-L", "L-U", "L-L", "U-S", "S-U", "L-S", "S-L", "S-S")
    bounds = {
        "U": (0, system_size),
        "L": (system_size, system_size * 2),
        "S": (signal_start, total_size),
    }
    block_order = (("U", "U"), ("U", "L"), ("L", "U"), ("L", "L"),
                   ("U", "S"), ("S", "U"), ("L", "S"), ("S", "L"), ("S", "S"))
    for label, (row_block, col_block) in zip(block_names, block_order):
        r0, r1 = bounds[row_block]
        c0, c1 = bounds[col_block]
        count = sum(r0 <= row < r1 and c0 <= col < c1 for row, col in support)
        print(f"{label}: {count} structural entries")

    noise_plan = arrays["update_plan"].noise
    noise_groups = noise_plan.static_groups + noise_plan.frequency_groups
    noise_pairs = [
        pair
        for group in noise_groups
        for pair in zip(group.matrix_rows, group.matrix_columns)
    ]
    duplicate_noise_entries = len(noise_pairs) - len(set(noise_pairs))
    noise_base_nnz = int(np.count_nonzero(np.asarray(arrays["noise_matrix"])))
    print(
        f"Noise sparsity: base_nnz={noise_base_nnz}, "
        f"update_entries={len(noise_pairs)}, duplicate_positions={duplicate_noise_entries}"
    )

    # Optical unknowns are assigned to their geometric cell. Every endpoint
    # used by an edge that crosses a cell boundary is promoted to the interface.
    per_cell: dict[tuple[int, int], set[int]] = {}
    for key, index in port_to_index.items():
        node, _port, _direction = key.rsplit(".", 2)
        cell = cell_id(node)
        if cell is not None:
            per_cell.setdefault(cell, set()).add(int(index))

    interface = set()
    for source, target, edge_data in setup.edges(data=True):
        source_cell, target_cell = cell_id(source), cell_id(target)
        if source_cell == target_cell:
            continue
        for node, cell, port in (
            (source, source_cell, edge_data.get("source_port", "right")),
            (target, target_cell, edge_data.get("target_port", "left")),
        ):
            if cell is None:
                continue
            for direction in ("in", "out"):
                key = f"{node}.{port}.{direction}"
                if key in port_to_index:
                    interface.add(int(port_to_index[key]))

    interiors = {}
    for cell, indices in per_cell.items():
        local = indices - interface
        interiors[cell] = local | {index + system_size for index in local}

    # Explicit signal and mechanical unknowns remain in the global interface.
    interior_owner = {
        index: cell for cell, indices in interiors.items() for index in indices
    }
    cross_cell = []
    for row, col in support:
        row_cell, col_cell = interior_owner.get(row), interior_owner.get(col)
        if row_cell is not None and col_cell is not None and row_cell != col_cell:
            cross_cell.append((row, col, row_cell, col_cell))

    print(f"Signal matrix: {total_size} x {total_size}; carrier block: {system_size}")
    print(f"Cells: {len(interiors)}; cell interiors: {[len(v) for v in interiors.values()]}")
    print(f"Cell optical interface unknowns: {len(interface) * 2}")
    print(f"Other signal/mechanical unknowns kept in interface: {total_size - 2 * system_size}")
    print(f"Cross-cell interior couplings: {len(cross_cell)}")
    if cross_cell:
        for entry in cross_cell[:20]:
            print("  ", entry)
        raise SystemExit("FAIL: proposed cell interiors are not independent")
    print("PASS: cell interiors have no direct cross-cell structural couplings")


if __name__ == "__main__":
    main()
