"""Bounded TT rank sweep on one A100; evaluation uses fresh GPU processes."""

import json
import os
import subprocess
import time
from pathlib import Path

import modal

from modal_vascular_3d import image

app = modal.App("jaxhps-born-tensor-train")


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=4,
    memory=65536,
    timeout=5520,
    max_containers=1,
    min_containers=0,
    scaledown_window=2,
    retries=0,
)
def measure(n, sources, receivers, wavelength_mm, ranks, budget_seconds):
    if not 1 <= budget_seconds <= 5400:
        raise ValueError("budget must be between 1 and 5400 seconds")
    directory = Path("/root/tt-born")
    directory.mkdir(exist_ok=True)
    common = [
        "python",
        "-u",
        "/root/jaxhps/examples/tensor_train_benchmark_3d.py",
    ]
    commands = [
        [
            *common,
            "prepare",
            "--directory",
            str(directory),
            "--n",
            str(n),
            "--sources",
            str(sources),
            "--receivers",
            str(receivers),
            "--wavelength-mm",
            str(wavelength_mm),
            "--ranks",
            *map(str, ranks),
        ]
    ]
    commands += [
        [
            *common,
            "evaluate",
            "--directory",
            str(directory),
            "--rank",
            str(rank),
        ]
        for rank in ranks
    ]
    deadline = time.monotonic() + budget_seconds
    results = []
    for command in commands:
        print("RUN " + " ".join(command), flush=True)
        # Preallocate only for the dense-cache setup; compressed applications
        # should expose their smaller actual driver allocation as well.
        env = dict(
            os.environ,
            JAX_ENABLE_COMPILATION_CACHE="false",
            XLA_PYTHON_CLIENT_PREALLOCATE="true"
            if "prepare" in command
            else "false",
            XLA_PYTHON_CLIENT_MEM_FRACTION="0.95",
        )
        with subprocess.Popen(command, env=env) as process:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    process.kill()
                    process.wait()
                    results.append("timeout")
                    break
                try:
                    results.append(process.wait(timeout=min(30, remaining)))
                    break
                except subprocess.TimeoutExpired:
                    print(
                        subprocess.check_output(
                            [
                                "nvidia-smi",
                                "--query-gpu=name,memory.used,utilization.gpu",
                                "--format=csv,noheader",
                            ],
                            text=True,
                            timeout=10,
                        ).strip(),
                        flush=True,
                    )
        if results[-1] != 0:
            break
    return dict(
        returncodes=results,
        completed=len(results) == len(commands)
        and all(code == 0 for code in results),
        reports={
            p.name: json.loads(p.read_text()) for p in directory.glob("*.json")
        },
    )


@app.local_entrypoint()
def main(
    out: str,
    n: int = 288,
    sources: int = 24,
    receivers: int = 256,
    wavelength_mm: float = 1.25,
    ranks: str = "32,64,128",
    budget_seconds: int = 5400,
):
    destination = Path(out)
    if destination.exists():
        raise ValueError("Use a new output file")
    row = measure.remote(
        n,
        sources,
        receivers,
        wavelength_mm,
        [int(r) for r in ranks.split(",")],
        budget_seconds,
    )
    destination.write_text(json.dumps(row, indent=2))
    if not row["completed"]:
        raise RuntimeError("TT benchmark failed; partial results saved")
