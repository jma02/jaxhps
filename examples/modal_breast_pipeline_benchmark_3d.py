"""Compare data-generation paths sequentially on one A100 in fresh processes."""

import json
import os
import subprocess
import threading
from collections.abc import Iterator
from pathlib import Path

import modal

from modal_vascular_3d import ROOT, image

app = modal.App("jaxhps-breast-pipeline-benchmark")


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=4,
    memory=65536,
    timeout=1020,
    max_containers=1,
    min_containers=0,
    scaledown_window=2,
)
def measure(
    baseline_source: str, n: int, wavelength_mm: float
) -> Iterator[dict]:
    Path("/root/baseline_inverse.py").write_text(baseline_source)
    env = dict(
        os.environ,
        PYTHONPATH="/root:/root/jaxhps/src:/root/jaxhps/examples",
        JAX_ENABLE_COMPILATION_CACHE="false",
    )
    for name in ("baseline", "optimized"):
        out = Path(f"/root/{name}.json")
        command = [
            "python",
            "-u",
            "/root/jaxhps/examples/breast_pipeline_benchmark_3d.py",
            "--n",
            str(n),
            "--wavelength-mm",
            str(wavelength_mm),
            "--reference",
            "/root/reference.npz",
            "--out",
            str(out),
        ]
        if name == "baseline":
            command.append("--baseline")
        print(f"START {name}", flush=True)
        lines = []
        with subprocess.Popen(
            command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        ) as process:
            timer = threading.Timer(480, process.kill)
            timer.start()
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    lines.append(line)
                status = process.wait()
            finally:
                timer.cancel()
        row = json.loads(out.read_text()) if out.exists() else {}
        yield dict(name=name, returncode=status, log="".join(lines), **row)
        if status:
            raise RuntimeError(f"{name} failed with exit {status}")


@app.local_entrypoint()
def main(
    baseline_ref: str, out: str, n: int = 128, wavelength_mm: float = 5
) -> None:
    destination = Path(out)
    destination.mkdir(parents=True, exist_ok=False)
    source = subprocess.check_output(
        ["git", "show", f"{baseline_ref}:examples/phantom_inverse_3d.py"],
        cwd=ROOT,
        text=True,
    )
    for row in measure.remote_gen(source, n, wavelength_mm):
        (destination / f"{row['name']}.log").write_text(row.pop("log"))
        (destination / f"{row['name']}.json").write_text(
            json.dumps(row, indent=2)
        )
        print(f"SAVED {row['name']}", flush=True)
