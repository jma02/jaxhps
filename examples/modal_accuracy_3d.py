"""Sequential fresh-process H100 benchmark cases (one container maximum).

Run with ``modal run examples/modal_accuracy_3d.py --suite smoke`` first.
The JSON files are written as each case finishes; failed cases are retained.
"""

import json
import shlex
import subprocess
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("jaxhps-matched-accuracy")
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
    .add_local_dir(
        ROOT / "data/examples/SD_3D", "/root/jaxhps/data/examples/SD_3D"
    )
)


@app.function(
    image=image,
    gpu="H100",
    cpu=4,
    memory=32768,
    timeout=2400,
    max_containers=1,
)
def measure(arguments):
    output = Path("/root/case.json")
    output.unlink(missing_ok=True)
    command = [
        "python",
        "-u",
        "/root/jaxhps/examples/accuracy_benchmark_3d.py",
        *arguments,
        "--out",
        str(output),
    ]
    gpu = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv",
        ],
        text=True,
    ).strip()
    try:
        subprocess.run(command, check=True, timeout=2300)
        return dict(json.loads(output.read_text()), gpu=gpu)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        return dict(failed=True, command=command, error=str(exc), gpu=gpu)


@app.local_entrypoint()
def main(
    suite: str = "smoke",
    output: str = "data/examples/accuracy",
    case: str = "",
    start: int = 0,
):
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    if case:
        commands = [case]
    elif suite == "smoke":
        commands = [
            "--solver fft --kappa 4 --n 32",
            "--solver hps --kappa 4 --q 4 --L 1",
            "--solver hps --kind phantom --kappa 4 --q 6 --L 1",
        ]
    elif suite == "radial":
        commands = [
            f"--solver fft --kappa {k} --n {n}"
            for k in (4, 8, 12)
            for n in (16, 24, 32, 48, 64)
        ] + [
            f"--solver hps --kappa {k} --q {q} --L {level} --precond {pc}"
            for k in (4, 8, 12)
            for q, level in ((8, 1), (10, 1), (12, 1), (6, 2), (8, 2))
            for pc in ("jacobi", "shift-coarse:0.1")
        ]
    elif suite == "phantom":
        commands = [
            f"--solver fft --kind phantom --kappa {k} --n {n}"
            for k in (4, 8)
            for n in (32, 48, 64, 96, 128, 160, 192)
        ] + [
            f"--solver hps --kind phantom --kappa {k} --q {q} --L {level}"
            for k in (4, 8)
            for q, level in (
                (6, 1),
                (8, 1),
                (10, 1),
                (12, 1),
                (4, 2),
                (6, 2),
                (8, 2),
            )
        ]
    elif suite == "phantom-p":
        commands = [
            f"--solver hps --kind phantom --kappa {k} --q 8 --L 2 --p {p} --leaf-batch-size 4"
            for k in (4, 8)
            for p in (16, 20)
        ]
    else:
        raise ValueError("suite must be smoke, radial, phantom, or phantom-p")
    for index, command in enumerate(commands[start:], start=start):
        print(f"CASE {index}: {command}", flush=True)
        result = measure.remote(shlex.split(command))
        result["invocation"] = command
        (directory / f"{suite}-{index:03d}.json").write_text(
            json.dumps(result, indent=2)
        )
        print(
            f"SAVED {suite}-{index:03d} failed={result.get('failed', False)}",
            flush=True,
        )
