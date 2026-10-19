"""One A100, one bounded reconstruction process; no persistent deployment."""

import io
import subprocess
import zipfile
from pathlib import Path

import modal

from modal_vascular_3d import image

app = modal.App("jaxhps-phantom-inverse-fit")


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=4,
    memory=49152,
    timeout=3300,
    max_containers=1,
    min_containers=0,
    scaledown_window=2,
)
def fit(smoke: bool) -> bytes:
    output = Path("/root/inverse-results")
    output.mkdir(exist_ok=True)
    command = [
        "python",
        "-u",
        "/root/jaxhps/examples/phantom_inverse_3d.py",
        "--out",
        str(output),
        "--seconds",
        "3000",
    ]
    if smoke:
        command += [
            "--n",
            "32",
            "--sources",
            "8",
            "--receivers",
            "64",
            "--waves",
            "2",
            "4",
            "--iterations",
            "10",
        ]
    with (output / "run.log").open("w") as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        status = process.wait()
    (output / "exit-status.txt").write_text(str(status))
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zipped:
        for path in output.iterdir():
            zipped.write(path, path.name)
    return archive.getvalue()


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=4,
    memory=49152,
    timeout=900,
    max_containers=1,
    min_containers=0,
    scaledown_window=2,
)
def validate(payload: bytes, waves: float) -> str:
    source = Path("/root/fitted.npz")
    source.write_bytes(payload)
    output = Path("/root/validation.json")
    subprocess.run(
        [
            "python",
            "-u",
            "/root/jaxhps/examples/check_phantom_inverse_3d.py",
            "--fit",
            str(source),
            "--out",
            str(output),
            "--waves",
            str(waves),
        ],
        check=True,
        timeout=840,
    )
    return output.read_text()


@app.local_entrypoint()
def main(
    out: str, smoke: bool = False, fitted: str = "", waves: float = 12.0
) -> None:
    destination = Path(out)
    destination.mkdir(parents=True, exist_ok=True)
    if fitted:
        result = validate.remote(Path(fitted).read_bytes(), waves)
        (destination / "validation.json").write_text(result)
        print(result)
        return
    payload = fit.remote(smoke)
    (destination / "results.zip").write_bytes(payload)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        archive.extractall(destination)
    print(f"Saved results to {destination}")
