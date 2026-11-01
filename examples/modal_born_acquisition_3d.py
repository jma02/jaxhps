"""A bounded acquisition comparison on one A100; no persistent deployment."""

import io
import os
import subprocess
import tarfile
import time
from pathlib import Path

import modal

from modal_vascular_3d import image

app = modal.App("jaxhps-born-acquisition")


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
def measure(smoke: bool, budget_seconds: int) -> bytes:
    if not 1 <= budget_seconds <= 5400:
        raise ValueError("Budget must be between 1 and 5400 seconds")
    out = Path("/root/acquisition")
    out.mkdir(exist_ok=True)
    common = ["python", "-u", "/root/jaxhps/examples/born_acquisition_3d.py"]
    shared = ["--directory", str(out)]
    options = (
        [
            "--n",
            "32",
            "--wavelength-mm",
            "10",
            "--sources",
            "8",
            "--receivers",
            "16",
            "--held-sources",
            "2",
            "--held-receivers",
            "4",
        ]
        if smoke
        else []
    )
    counts = (
        [(4, 8), (8, 8), (8, 16)]
        if smoke
        else [(24, 256), (64, 256), (64, 512)]
    )
    commands = [
        [*common, "prepare", *shared, *options],
        *[
            [
                *common,
                "evaluate",
                *shared,
                "--sources",
                str(s),
                "--receivers",
                str(r),
            ]
            for s, r in counts
        ],
        [*common, "heldout", *shared],
    ]
    deadline = time.monotonic() + budget_seconds
    returns = []
    for command in commands:
        print("RUN " + " ".join(command), flush=True)
        with subprocess.Popen(
            command,
            env=dict(
                os.environ,
                JAX_ENABLE_COMPILATION_CACHE="false",
                XLA_PYTHON_CLIENT_PREALLOCATE="true",
                XLA_PYTHON_CLIENT_MEM_FRACTION="0.90",
            ),
        ) as process:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    process.kill()
                    process.wait()
                    returns.append("timeout")
                    break
                try:
                    returns.append(process.wait(timeout=min(30, remaining)))
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
        if returns[-1] != 0:
            break
    import json

    (out / "run.json").write_text(
        json.dumps(
            dict(
                returncodes=returns,
                completed=len(returns) == len(commands)
                and all(x == 0 for x in returns),
            )
        )
    )
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path in [
            *out.glob("*.json"),
            *out.glob("*-slices.npz"),
            *out.glob("geometry.npz"),
        ]:
            archive.add(path, arcname=path.name)
    return buffer.getvalue()


@app.local_entrypoint()
def main(out: str, smoke: bool = False, budget_seconds: int = 5400) -> None:
    destination = Path(out)
    if destination.exists():
        raise ValueError("Use a new output archive")
    destination.write_bytes(measure.remote(smoke, budget_seconds))
    print(f"Saved {destination}")
