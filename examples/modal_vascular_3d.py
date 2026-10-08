"""Sequential A100 refinement with a 15-minute GPU-work budget; no deployment."""

import json
import subprocess
import time
from pathlib import Path

import modal

from benchmark_io import save_result

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("jaxhps-vascular-a100")
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("jax[cuda12]==0.6.2", "numpy==2.2.6", "scipy==1.15.3")
    .env(
        {
            "PYTHONPATH": "/root/jaxhps/src:/root/jaxhps/examples",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "1",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        }
    )
    .add_local_dir(ROOT / "src", "/root/jaxhps/src")
    .add_local_dir(
        ROOT / "examples",
        "/root/jaxhps/examples",
        ignore=["**/__pycache__/**"],
    )
)


def case_grid(suite):
    if suite == "smoke":
        return [("vascular", 10, 64)]
    if suite == "high-frequency":
        return [("dense", 80, n) for n in (192, 224, 256)]
    if suite == "refinement":
        return [
            (variant, waves, n)
            for waves, grids in [
                (10, (96, 128, 160)),
                (20, (128, 160, 192)),
                (40, (192, 224, 256)),
            ]
            for variant in ("vascular", "dense")
            for n in grids
        ]
    raise ValueError("suite must be smoke, refinement, or high-frequency")


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=4,
    memory=32768,
    timeout=1020,
    max_containers=1,
    min_containers=0,
    scaledown_window=2,
)
def measure(suite, budget_seconds, case_seconds):
    if not 1 <= budget_seconds <= 900 or not 1 <= case_seconds <= 180:
        raise ValueError("budget must be <=900s and each case <=180s")
    gpu = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv",
        ],
        text=True,
    ).strip()
    deadline = time.monotonic() + budget_seconds
    print(gpu, flush=True)
    for variant, waves, n in case_grid(suite):
        remaining = deadline - time.monotonic()
        if remaining < 15:
            yield dict(
                skipped=True,
                reason="GPU work budget exhausted",
                variant=variant,
                waves=waves,
                n=n,
            )
            continue
        output = Path("/root/vascular-case.json")
        output.unlink(missing_ok=True)
        command = [
            "python",
            "-u",
            "/root/jaxhps/examples/vascular_benchmark_3d.py",
            "--variant",
            variant,
            "--waves",
            str(waves),
            "--n",
            str(n),
            "--out",
            str(output),
        ]
        print(
            f"CASE {variant} waves={waves} n={n}; limit={min(case_seconds, remaining):.0f}s",
            flush=True,
        )
        t0 = time.monotonic()
        failure = None
        try:
            subprocess.run(
                command, check=True, timeout=min(case_seconds, remaining)
            )
        except (
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
        ) as exc:
            failure = str(exc)
        row = json.loads(output.read_text()) if output.exists() else {}
        row.update(
            gpu=gpu,
            process_seconds=time.monotonic() - t0,
            variant=variant,
            waves=waves,
            n=n,
        )
        if failure:
            row.update(failed=True, error=failure)
        yield row


@app.local_entrypoint()
def main(
    suite: str = "smoke",
    output: str = "data/examples/vascular-a100",
    budget_seconds: int = 900,
    case_seconds: int = 150,
):
    case_grid(suite)
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    if list(directory.glob("*.json")):
        raise ValueError("use an empty output directory")
    for row in measure.remote_gen(suite, budget_seconds, case_seconds):
        name = f"{row['variant']}-w{row['waves']}-n{row['n']}"
        save_result(directory / f"{name}.json", row)
        print(
            f"SAVED {name} failed={row.get('failed', False)} skipped={row.get('skipped', False)}",
            flush=True,
        )
