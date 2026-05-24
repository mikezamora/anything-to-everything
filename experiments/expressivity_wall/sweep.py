"""E30 — Expressivity wall depth/breadth/χ sweep (spec §13.5 follow-on).

Scans the (depth, breadth, χ) cube on a representative hole-bearing corpus
and records reconstruction residual <H> = ham.total_energy(final_state)
after imaginary-time MERA evolution. The "expressivity wall" is the locus
in this cube where the substrate physically cannot represent the
target — for the hole-binding family that means χ < (#candidate binders),
which the encoder loudly refuses via §1.1 ("binding = entanglement, never
classical lookup"). Above the wall the state converges to <H> ≈ 0.

Corpus
------
Hole-bearing programs of the form ``\\a:Int...\\k:Int. (Hole + 1 + 1 + ...)``
where ``Hole`` is a ``HoleVar`` with the k binders as candidates. The
encoder produces a genuinely entangled MERA between the hole bid leaf and
the candidate binder bid leaves — exactly the §5.3 / §9.4 binding state
whose §13.5 area-law representability depends on χ.

Axes
----
- breadth  ∈ {2, 3, 4, 5, 6}   — #candidate binders (drives required χ)
- depth    ∈ {0, 1, 2}         — body Bin nesting (drives chain length)
- χ        ∈ {2, 3, 4, 6, 8}   — substrate bond dimension

Outputs
-------
- ``experiments/results/expressivity_wall/sweep.csv`` — raw rows
- ``experiments/results/expressivity_wall/sweep.json`` — same data + wall loci

Run directly::

    uv run python -m experiments.expressivity_wall.sweep
"""
from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

from src.qft_pcn.logic.ast import (
    Bin, HoleVar, IntLit, Lam, TInt,
)
from src.qft_pcn.logic.mera_encoder import encode_mera
from src.qft_pcn.logic.mera_evaluation_hamiltonian import MeraEvalHamiltonian
from src.qft_pcn.logic.mera_evolution_logic import (
    mera_imaginary_evolve_state,
)


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------
def build_hole_program(n_binders: int, body_depth: int):
    """Construct ``\\a:Int...\\k:Int. ((Hole + 1) + 1 + ... + 1)`` with
    ``n_binders`` lambda binders and ``body_depth`` ``+1`` Bin layers.

    The single ``HoleVar`` carries all ``n_binders`` names as candidates,
    forcing the encoder to produce a rank-``n_binders`` entangled MERA
    between the hole bid leaf and the binder bid leaves.
    """
    if n_binders < 1:
        raise ValueError("n_binders must be >= 1")
    names = [chr(ord("a") + i) for i in range(n_binders)]
    h = HoleVar(candidates=list(names))
    body = h
    for _ in range(body_depth):
        body = Bin(op="+", lhs=body, rhs=IntLit(1))
    ast = body
    for nm in reversed(names):
        ast = Lam(param=nm, param_ty=TInt(), body=ast)
    return ast


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
@dataclass
class SweepRow:
    breadth: int           # #candidate binders
    depth: int             # body Bin nesting
    chi: int               # bond dimension
    residual: float        # final <H>; +inf if encoder refused (wall)
    accepted: bool         # residual < threshold (i.e. converged)
    refused: bool          # encoder/runtime raised — substrate cannot represent
    error_kind: str        # exception class name, or ""
    error_message: str     # exception message, or ""
    seconds: float


def run_one(
    breadth: int,
    depth: int,
    chi: int,
    *,
    dt: float = 0.1,
    steps: int = 20,
    threshold: float = 1e-2,
) -> SweepRow:
    """Run a single (breadth, depth, χ) cell. Refusal (any exception during
    encode/evolve) is recorded as the wall surfacing — residual=+inf,
    refused=True. NaN residual is also treated as a wall."""
    t0 = time.time()
    try:
        ast = build_hole_program(breadth, depth)
        state, meta = encode_mera(ast, chi_layer=chi)
        H = MeraEvalHamiltonian(meta)
        _, final = mera_imaginary_evolve_state(
            state, H, dt=dt, steps=steps, chi_layer=chi,
        )
        res = float(H.total_energy(final))
    except Exception as exc:  # noqa: BLE001 — sweep records any refusal
        return SweepRow(
            breadth=breadth, depth=depth, chi=chi,
            residual=float("inf"),
            accepted=False,
            refused=True,
            error_kind=type(exc).__name__,
            error_message=str(exc),
            seconds=time.time() - t0,
        )
    if math.isnan(res):
        return SweepRow(
            breadth=breadth, depth=depth, chi=chi,
            residual=float("inf"),
            accepted=False,
            refused=True,
            error_kind="NaNResidual",
            error_message="evolution produced NaN energy",
            seconds=time.time() - t0,
        )
    return SweepRow(
        breadth=breadth, depth=depth, chi=chi,
        residual=res,
        accepted=(res < threshold),
        refused=False,
        error_kind="",
        error_message="",
        seconds=time.time() - t0,
    )


def run_sweep(
    breadths: Iterable[int],
    depths: Iterable[int],
    chis: Iterable[int],
    *,
    dt: float = 0.1,
    steps: int = 20,
    threshold: float = 1e-2,
) -> list[SweepRow]:
    rows: list[SweepRow] = []
    for b in breadths:
        for d in depths:
            for c in chis:
                rows.append(
                    run_one(b, d, c, dt=dt, steps=steps, threshold=threshold)
                )
    return rows


# ---------------------------------------------------------------------------
# Wall locus analysis
# ---------------------------------------------------------------------------
def wall_loci_by_breadth(
    rows: list[SweepRow], depth: int,
) -> dict[int, int | None]:
    """For each breadth (at a fixed depth), find the smallest χ at which the
    program is accepted. None means no χ in the sweep accepts it."""
    out: dict[int, int | None] = {}
    breadths = sorted({r.breadth for r in rows})
    for b in breadths:
        accepted_chis = sorted(
            r.chi for r in rows
            if r.breadth == b and r.depth == depth and r.accepted
        )
        out[b] = accepted_chis[0] if accepted_chis else None
    return out


def is_monotone_transition(
    rows: list[SweepRow], breadth: int, depth: int,
) -> bool:
    """At fixed (breadth, depth), check that residual is monotone
    non-increasing as χ grows — i.e. once χ crosses the wall, it stays
    accepting. (Refusals = +inf, so they sit above any accepted cell.)"""
    cell = sorted(
        (r for r in rows if r.breadth == breadth and r.depth == depth),
        key=lambda r: r.chi,
    )
    last = float("inf")
    for r in cell:
        if r.residual > last + 1e-9:
            return False
        last = r.residual
    return True


# ---------------------------------------------------------------------------
# CLI / artifact writer
# ---------------------------------------------------------------------------
DEFAULT_BREADTHS = (2, 3, 4, 5, 6)
DEFAULT_DEPTHS = (0, 1)
DEFAULT_CHIS = (2, 3, 4, 6, 8)


def default_results_dir() -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "experiments" / "results" / "expressivity_wall"
    )


def write_artifacts(
    rows: list[SweepRow],
    out_dir: Path,
    *,
    config: dict,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "sweep.csv"
    json_path = out_dir / "sweep.json"

    fields = list(SweepRow.__dataclass_fields__.keys())
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            row = asdict(r)
            # CSV-safe inf
            if math.isinf(row["residual"]):
                row["residual"] = "inf"
            w.writerow(row)

    loci_by_depth = {
        str(d): {str(b): chi for b, chi in
                 wall_loci_by_breadth(rows, d).items()}
        for d in sorted({r.depth for r in rows})
    }
    monotone_by_cell = {
        f"breadth={b},depth={d}": is_monotone_transition(rows, b, d)
        for b in sorted({r.breadth for r in rows})
        for d in sorted({r.depth for r in rows})
    }

    payload = {
        "spec_section": "§13.5 expressivity wall — depth/breadth/χ sweep",
        "extension_id": "E30",
        "config": config,
        "rows": [
            {**asdict(r),
             "residual": ("inf" if math.isinf(r.residual) else r.residual)}
            for r in rows
        ],
        "wall_loci_min_chi_by_breadth": loci_by_depth,
        "monotone_transition_by_cell": monotone_by_cell,
    }
    json_path.write_text(json.dumps(payload, indent=2))
    return csv_path, json_path


def main() -> int:
    config = {
        "breadths": list(DEFAULT_BREADTHS),
        "depths": list(DEFAULT_DEPTHS),
        "chis": list(DEFAULT_CHIS),
        "dt": 0.1,
        "steps": 20,
        "threshold": 1e-2,
    }
    rows = run_sweep(
        DEFAULT_BREADTHS, DEFAULT_DEPTHS, DEFAULT_CHIS,
        dt=config["dt"], steps=config["steps"],
        threshold=config["threshold"],
    )
    csv_path, json_path = write_artifacts(
        rows, default_results_dir(), config=config,
    )
    print(f"wrote {csv_path}")
    print(f"wrote {json_path}")
    # Quick stdout summary
    for d in sorted({r.depth for r in rows}):
        loci = wall_loci_by_breadth(rows, d)
        print(f"  depth={d}: min-accepting χ by breadth = {loci}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
