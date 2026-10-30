"""Bounded streamed spectral-compression experiment on one A100."""

import json
import os
import subprocess
import time
from pathlib import Path

import modal

from modal_vascular_3d import image

app = modal.App("jaxhps-spectral-born")


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
    tolerances: list[float],
    batches: list[int],
    budget_seconds: int,
    basis: str,
    taper_floor: float,
    tucker_rank: int,
) -> dict:
    if not 1 <= budget_seconds <= 5400:
        raise ValueError("budget must be between 1 and 5400 seconds")
    directory = Path("/root/spectral-born")
    directory.mkdir(exist_ok=True)
    common = ["python", "-u", "/root/jaxhps/examples/spectral_benchmark_3d.py"]
    shared = [
        "--directory",
        str(directory),
        "--basis",
        basis,
        "--taper-floor",
        str(taper_floor),
        "--tucker-rank",
        str(tucker_rank),
    ]
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
        ],
        [*common, "dense", *shared],
    ]
    if tucker_rank:
        commands.append([*common, "tucker", *shared])
    for tolerance in tolerances:
        options = [
            *shared,
            "--tolerance",
            str(tolerance),
            "--label",
            f"{basis}-{tolerance}",
        ]
        commands.append([*common, "compress", *options])
        commands.extend(
            [*common, "evaluate", *options, "--source-batch-size", str(batch)]
            for batch in batches
        )
    deadline = time.monotonic() + budget_seconds
    results = []
    for command in commands:
        print("RUN " + " ".join(command), flush=True)
        env = dict(
            os.environ,
            JAX_ENABLE_COMPILATION_CACHE="false",
            XLA_PYTHON_CLIENT_PREALLOCATE="true"
            if command[3] == "dense"
            else "false",
            XLA_PYTHON_CLIENT_MEM_FRACTION="0.95"
            if command[3] == "dense"
            else "0.45",
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
    tolerances: str = "0.01,0.003",
    batches: str = "2",
    budget_seconds: int = 5400,
    basis: str = "dct",
    taper_floor: float = 0.01,
    tucker_rank: int = 0,
):
    destination = Path(out)
    if destination.exists():
        raise ValueError("Use a new output file")
    result = measure.remote(
        n,
        sources,
        receivers,
        wavelength_mm,
        [float(x) for x in tolerances.split(",")],
        [int(x) for x in batches.split(",")],
        budget_seconds,
        basis,
        taper_floor,
        tucker_rank,
    )
    destination.write_text(json.dumps(result, indent=2))
    if not result["completed"]:
        raise RuntimeError("Spectral benchmark failed; partial results saved")
