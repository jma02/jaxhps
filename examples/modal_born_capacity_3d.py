"""Run one bounded A100 capacity experiment, including driver-memory sampling."""

import json
import os
import subprocess
import threading
import time
from pathlib import Path

import modal

from modal_vascular_3d import image

app = modal.App("jaxhps-born-288-capacity")


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=4,
    memory=65536,
    timeout=1920,
    max_containers=1,
    min_containers=0,
    scaledown_window=2,
    retries=0,
)
def measure() -> dict:
    out = Path("/root/born-capacity.json")
    samples = []
    stopped = threading.Event()
    start = time.monotonic()

    def monitor() -> None:
        while not stopped.is_set():
            raw = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                timeout=10,
            )
            used, utilization = map(int, raw.strip().split(","))
            sample = dict(
                seconds=time.monotonic() - start,
                memory_mib=used,
                utilization=utilization,
            )
            samples.append(sample)
            if len(samples) % 15 == 1:
                print(f"GPU {json.dumps(sample)}", flush=True)
            stopped.wait(2)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        process = subprocess.run(
            [
                "python",
                "-u",
                "/root/jaxhps/examples/born_capacity_benchmark_3d.py",
                "--out",
                str(out),
            ],
            env=dict(
                os.environ,
                JAX_ENABLE_COMPILATION_CACHE="false",
                XLA_PYTHON_CLIENT_PREALLOCATE="true",
                XLA_PYTHON_CLIENT_MEM_FRACTION="0.95",
            ),
            timeout=1800,
            check=False,
        )
        status = process.returncode
    except subprocess.TimeoutExpired:
        status = "timeout after 1800 seconds"
    finally:
        stopped.set()
        thread.join(timeout=15)
    row = json.loads(out.read_text()) if out.exists() else {}
    row.update(returncode=status, driver_memory_samples=samples)
    row["allocator"] = dict(preallocate=True, memory_fraction=0.95)
    row["peak_sampled_driver_memory_bytes"] = (
        max(s["memory_mib"] for s in samples) * 1024**2
    )
    return row


@app.local_entrypoint()
def main(out: str) -> None:
    destination = Path(out)
    if destination.exists():
        raise ValueError("Use a new output file")
    row = measure.remote()
    destination.write_text(json.dumps(row, indent=2))
    print(
        f"Saved {destination}; completed={row.get('completed', False)}",
        flush=True,
    )
    if row["returncode"] != 0 or not row.get("completed", False):
        raise RuntimeError("Capacity benchmark failed; partial results saved")
