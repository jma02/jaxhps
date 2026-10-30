"""Bounded decomposition comparison, sequentially on one A100."""

import json
import os
import subprocess
import time
from pathlib import Path

import modal

from modal_vascular_3d import image

app = modal.App("jaxhps-born-compression-comparison")


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
def measure(
    n: int,
    sources: int,
    receivers: int,
    wavelength_mm: float,
    cases: list[str],
    budget_seconds: int,
    iterations: int,
) -> dict:
    if not 1 <= budget_seconds <= 5400:
        raise ValueError("budget must be between 1 and 5400 seconds")
    directory = Path("/root/born-compression")
    directory.mkdir(exist_ok=True)
    common = [
        "python",
        "-u",
        "/root/jaxhps/examples/compression_benchmark_3d.py",
    ]
    shared = ["--directory", str(directory)]
    commands = [
        [
            *common,
            "prepare",
            *shared,
            "--n",
            str(n),
            "--sources",
            str(sources),
            "--receivers",
            str(receivers),
            "--wavelength-mm",
            str(wavelength_mm),
        ]
    ]
    commands.append([*common, "screen", *shared, "--block-size", "8"])
    for case in cases:
        parts = case.split("-")
        method, rank = parts[0], int(parts[-1])
        options = [
            *shared,
            "--method",
            method,
            "--rank",
            str(rank),
            "--label",
            case,
            "--iterations",
            str(iterations),
        ]
        if method == "block":
            options += ["--block-size", parts[1]]
        commands += [
            [*common, stage, *options] for stage in ("compress", "evaluate")
        ]
    deadline = time.monotonic() + budget_seconds
    results = []
    for command in commands:
        print("RUN " + " ".join(command), flush=True)
        env = dict(
            os.environ,
            JAX_ENABLE_COMPILATION_CACHE="false",
            XLA_PYTHON_CLIENT_PREALLOCATE="true"
            if command[3] in ("prepare", "compress")
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
    cases: str = "tucker-128,block-4-32,cp-256",
    budget_seconds: int = 5400,
    iterations: int = 50,
):
    destination = Path(out)
    if destination.exists():
        raise ValueError("Use a new output file")
    result = measure.remote(
        n,
        sources,
        receivers,
        wavelength_mm,
        cases.split(","),
        budget_seconds,
        iterations,
    )
    destination.write_text(json.dumps(result, indent=2))
    if not result["completed"]:
        raise RuntimeError("Comparison failed; partial results saved")
