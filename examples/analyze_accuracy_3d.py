"""Validate case JSON, retain failures, and select matched-error configurations."""

import argparse
import csv
import json
import statistics
from itertools import groupby
from operator import itemgetter
from pathlib import Path

import numpy as np


def relative_error(field, reference, weights):
    return float(
        np.sqrt(
            np.sum(weights * abs(field - reference) ** 2)
            / np.sum(weights * abs(reference) ** 2)
        )
    )


def certified(info):
    stats = info.get("gmres_stats", info)
    residual = stats["final_rel_res"]
    if stats["true_res_history"][-1] != residual:
        raise ValueError("final residual disagrees with stored history")
    converged = bool(np.isfinite(residual) and residual <= 1e-8)
    flag = info["converged"] if "converged" in info else info["info"] == 0
    if flag != converged:
        raise ValueError("convergence flag disagrees with true residual")
    return converged


def analyze(files, reference_controls):
    radial_changes = {
        r["kappa"]: max(
            value
            for key, value in r["changes"].items()
            if key != "legacy_reference"
        )
        for r in reference_controls
    }
    raw = []
    failures = []
    for path in files:
        row = json.loads(path.read_text())
        row["case"] = path.stem
        if row.get("failed"):
            failures.append(row)
        else:
            raw.append(row)
    if not raw:
        raise ValueError("no successful case files")
    for row in raw:
        np.testing.assert_array_equal(row["targets"], raw[0]["targets"])
        np.testing.assert_array_equal(row["weights"], raw[0]["weights"])
        values = np.asarray(row["field_real"]) + 1j * np.asarray(
            row["field_imag"]
        )
        assert np.isfinite(values).all()
        row["field"] = values
        row["certified"] = certified(row["first_info"]) and all(
            certified(rep["info"]) for rep in row["repeats"]
        )
    weights = np.asarray(raw[0]["weights"])
    references = {}
    for kappa in sorted({r["parameters"]["kappa"] for r in raw}):
        candidates = sorted(
            [
                r
                for r in raw
                if r["parameters"]["kind"] == "phantom"
                and r["parameters"]["solver"] == "fft"
                and r["parameters"]["kappa"] == kappa
                and r["certified"]
            ],
            key=lambda r: r["parameters"]["n"],
        )
        if len(candidates) >= 3:
            earlier, previous, fine = candidates[-3:]
            references[kappa] = dict(
                row=fine,
                n=fine["parameters"]["n"],
                previous_n=previous["parameters"]["n"],
                earlier_n=earlier["parameters"]["n"],
                relative_change=relative_error(
                    previous["field"], fine["field"], weights
                ),
                previous_relative_change=relative_error(
                    earlier["field"], previous["field"], weights
                ),
            )
    rows = []
    for row in raw:
        p = row["parameters"]
        info = row["first_info"]
        stats = info.get("gmres_stats", info)
        repeats = row["repeats"]
        timings = [r["seconds"] for r in repeats]
        uncertainty = (
            radial_changes.get(p["kappa"]) if p["kind"] == "radial" else None
        )
        field_error = row.get("field_relative_error")
        if p["kind"] == "phantom" and p["kappa"] in references:
            reference = references[p["kappa"]]
            field_error = relative_error(
                row["field"], reference["row"]["field"], weights
            )
            uncertainty = reference["relative_change"]
        rows.append(
            dict(
                case=row["case"],
                kind=p["kind"],
                kappa=p["kappa"],
                method="FFT"
                if p["solver"] == "fft"
                else (
                    "HPS Jacobi" if p["precond"] == "jacobi" else "HPS coarse"
                ),
                resolution=f"n={p['n']}"
                if p["solver"] == "fft"
                else (f"q={p['q']},L={p['L']},p={p['p'] or p['q'] + 4}"),
                n=p["n"] if p["solver"] == "fft" else None,
                q=p["q"] if p["solver"] == "hps" else None,
                p=(p["p"] or p["q"] + 4) if p["solver"] == "hps" else None,
                L=p["L"] if p["solver"] == "hps" else None,
                leaf_batch_size=p.get("leaf_batch_size"),
                unknowns=row["unknowns"],
                converged=bool(row["certified"]),
                max_true_residual=max(
                    [stats["final_rel_res"]]
                    + [
                        r["info"].get("gmres_stats", r["info"])[
                            "final_rel_res"
                        ]
                        for r in repeats
                    ]
                ),
                field_error=field_error,
                reference_change=uncertainty,
                setup_seconds=row["setup_seconds"],
                first_solve_seconds=row["first_solve_seconds"],
                cold_seconds=row["cold_setup_solve_seconds"],
                repeat_min_seconds=min(timings) if timings else None,
                repeat_median_seconds=(
                    statistics.median(timings) if timings else None
                ),
                repeat_max_seconds=max(timings) if timings else None,
                evaluation_seconds=row["evaluation_seconds"],
                matvecs=stats["n_matvec"],
                inner_matvecs=info.get("inner_stats", {}).get("n_matvec", 0),
                peak_live_gib=row["gpu_memory"]["peak_bytes_in_use"] / 2**30,
                peak_pool_gib=row["gpu_memory"]["peak_pool_bytes"] / 2**30,
                host_peak_gib=row["host_peak_rss_bytes"] / 2**30,
            )
        )
    matches = []
    ordered = sorted(rows, key=itemgetter("kind", "kappa", "method"))
    for (kind, kappa), cases in groupby(
        ordered, key=itemgetter("kind", "kappa")
    ):
        methods = {
            method: list(group)
            for method, group in groupby(cases, key=itemgetter("method"))
        }
        for ceiling in (0.01, 0.001, 0.0001):
            for method, group in methods.items():
                eligible = [
                    r
                    for r in group
                    if r["converged"]
                    and r["field_error"] is not None
                    and r["reference_change"] is not None
                    and r["field_error"] + r["reference_change"] <= ceiling
                    and r["reference_change"] <= ceiling / 10
                ]
                for metric in ("cold_seconds", "repeat_median_seconds"):
                    best = min(
                        (r for r in eligible if r[metric] is not None),
                        key=itemgetter(metric),
                        default=None,
                    )
                    matches.append(
                        dict(
                            kind=kind,
                            kappa=kappa,
                            ceiling=ceiling,
                            method=method,
                            metric=metric,
                            best_case=best["case"] if best else None,
                            seconds=best[metric] if best else None,
                        )
                    )
    return dict(
        rows=rows,
        matches=matches,
        failures=failures,
        phantom_references={
            k: {key: v for key, v in r.items() if key != "row"}
            for k, r in references.items()
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference-controls", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(
        [p for d in args.directories for p in sorted(d.glob("*.json"))],
        json.loads(args.reference_controls.read_text()),
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(result, indent=2))
    with (args.out / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(result["rows"][0]))
        writer.writeheader()
        writer.writerows(result["rows"])
    print(
        f"{len(result['rows'])} completed; {len(result['failures'])} failed processes"
    )


if __name__ == "__main__":
    main()
