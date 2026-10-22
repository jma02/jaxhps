"""Compare a git baseline and working tree on one A100, in fresh processes."""

import json
import os
import subprocess
from collections.abc import Iterator
from pathlib import Path

import modal

from modal_vascular_3d import ROOT, image

app = modal.App("jaxhps-fft-gpu-benchmark")


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=4,
    memory=65536,
    timeout=1800,
    max_containers=1,
    min_containers=0,
    scaledown_window=2,
)
def measure(sources: dict[str, str], n: int, waves: float) -> Iterator[dict]:
    for name, source in sources.items():
        Path(f"/root/{name}").write_text(source)
    env = dict(
        os.environ,
        JAX_ENABLE_COMPILATION_CACHE="false",
        XLA_PYTHON_CLIENT_PREALLOCATE="true",
        XLA_PYTHON_CLIENT_MEM_FRACTION=".95",
        PYTHONPATH="/root:/root/jaxhps/src:/root/jaxhps/examples",
    )
    gpu = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv",
        ],
        text=True,
    )
    print(gpu, flush=True)
    for repeat in range(1, 4):
        for baseline in (True, False) if repeat % 2 else (False, True):
            name = f"n{n}-w{waves}-r{repeat}-{'baseline' if baseline else 'optimized'}"
            out = Path(f"/root/{name}.json")
            command = [
                "python",
                "-u",
                "/root/jaxhps/examples/fft_gpu_benchmark_3d.py",
                "--n",
                str(n),
                "--waves",
                str(waves),
                "--reference",
                "/root/reference.npy",
                "--out",
                str(out),
            ]
            if baseline:
                command.append("--baseline")
            print("START", name, flush=True)
            result = subprocess.run(
                command,
                env=env,
                timeout=240,
                capture_output=True,
                text=True,
            )
            print(result.stdout, result.stderr[-1200:], flush=True)
            row = json.loads(out.read_text()) if out.exists() else {}
            row.update(
                name=name,
                repeat=repeat,
                gpu=gpu,
                returncode=result.returncode,
                log=result.stdout + "\n" + result.stderr,
            )
            yield row
            if result.returncode:
                raise RuntimeError(f"benchmark failed: {name}")


@app.local_entrypoint()
def main(
    baseline_ref: str, output: str, n: int = 288, waves: float = 80
) -> None:
    directory = Path(output)
    directory.mkdir(parents=True, exist_ok=True)
    if list(directory.glob("*.json")):
        raise ValueError("use an empty output directory")
    sources = {
        name: subprocess.check_output(
            ["git", "show", f"{baseline_ref}:{path}"],
            cwd=ROOT,
            text=True,
        )
        for name, path in (
            ("baseline_fft.py", "examples/fft_volume_3d.py"),
            ("baseline_krylov.py", "src/jaxhps/_krylov.py"),
        )
    }
    sources["baseline_fft.py"] = sources["baseline_fft.py"].replace(
        "from jaxhps._krylov import fgmres",
        "from baseline_krylov import fgmres",
    )
    for row in measure.remote_gen(sources, n, waves):
        name = row["name"]
        (directory / f"{name}.log").write_text(row.pop("log"))
        row["baseline_ref"] = baseline_ref
        (directory / f"{name}.json").write_text(json.dumps(row, indent=2))
        print("SAVED", name, row.get("cold_seconds"), flush=True)
