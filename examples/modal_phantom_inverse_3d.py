"""One A100, one bounded reconstruction process; no persistent deployment."""

import io
import subprocess
import zipfile
from pathlib import Path

import modal

from modal_vascular_3d import image

app = modal.App("jaxhps-phantom-inverse-fit")


def _run(script: str, extra: list[str], payload: bytes = b"") -> bytes:
    output = Path("/root/inverse-results")
    output.mkdir(exist_ok=True)
    command = [
        "python",
        "-u",
        f"/root/jaxhps/examples/{script}",
        "--out",
        str(output),
        *extra,
    ]
    if payload:
        Path("/root/images.npz").write_bytes(payload)
        command += ["--images", "/root/images.npz"]
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
    timeout=3300,
    max_containers=1,
    min_containers=0,
    scaledown_window=2,
)
def fit(script: str, extra: list[str], payload: bytes = b"") -> bytes:
    return _run(script, extra, payload)


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=4,
    memory=49152,
    timeout=11400,
    max_containers=1,
    min_containers=0,
    scaledown_window=2,
)
def fit_long(script: str, extra: list[str]) -> bytes:
    return _run(script, extra)


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
    out: str,
    smoke: bool = False,
    learn: bool = False,
    images: str = "",
    breast: bool = False,
    spheres: bool = False,
    fitted: str = "",
    waves: float = 12.0,
) -> None:
    destination = Path(out)
    destination.mkdir(parents=True, exist_ok=True)
    if fitted:
        result = validate.remote(Path(fitted).read_bytes(), waves)
        (destination / "validation.json").write_text(result)
        print(result)
        return
    if spheres:
        extra = ["--n", "24", "--wavelength-mm", "30", "--train", "6"]
        extra += ["--test", "2", "--fits", "1", "--sources", "4"]
        extra += ["--receivers", "32", "--epochs", "2", "--iterations", "3"]
        payload = fit_long.remote(
            "breast_spheres_learn_3d.py", extra if smoke else []
        )
    elif breast:
        extra = ["--waves", "4", "--grids", "32", "--seconds", "300"]
        extra += ["--sources", "4", "--receivers", "64", "--iterations", "3"]
        payload = fit_long.remote(
            "breast_inverse_3d.py", extra if smoke else []
        )
    elif images:
        script, extra = "phantom_dataset_fit_3d.py", ["--seconds", "3000"]
        if smoke:
            extra = ["--count", "2", "--sources", "6", "--receivers", "32"]
            extra += ["--waves", "2", "3", "--iterations", "5"]
        payload = fit.remote(script, extra, Path(images).read_bytes())
    elif learn:
        script, extra = "phantom_dataset_learn_3d.py", ["--seconds", "3000"]
        if smoke:
            extra = ["--n", "16", "--train", "8", "--test", "4"]
            extra += ["--sources", "6", "--receivers", "32", "--waves", "3"]
            extra += ["--epochs", "3", "--minimum-core", "1"]
        payload = fit.remote(script, extra)
    else:
        script, extra = "phantom_inverse_3d.py", ["--seconds", "3000"]
        if smoke:
            extra = ["--n", "32", "--sources", "8", "--receivers", "64"]
            extra += ["--waves", "2", "4", "--iterations", "10"]
        payload = fit.remote(script, extra)
    (destination / "results.zip").write_bytes(payload)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        archive.extractall(destination)
    print(f"Saved results to {destination}")
