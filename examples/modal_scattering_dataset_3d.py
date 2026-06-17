r"""Run the free-space 3D Helmholtz scattering dataset generator on Modal.

This wraps :mod:`scattering_dataset_3d` so the HPS+BIE forward solve runs on a
single Modal GPU container.  The precomputed single/double-layer matrix lives in
a Modal Volume (``jaxhps-sd``); the Hugging Face token is a Modal Secret
(``hf-token``).  Shards are streamed to the Hub as they finish, and the run is
resumable: shards already present in the target repo are skipped, so a restart
picks up where it left off.

Usage (from a shell authenticated to Modal)::

    # tiny GPU smoke test (no upload): time a few solves
    modal run examples/modal_scattering_dataset_3d.py --mode smoke --n 3

    # upload the dataset card + metadata only
    modal run examples/modal_scattering_dataset_3d.py --mode setup

    # full run on one GPU container (resumable, streams shards to the Hub)
    modal run --detach examples/modal_scattering_dataset_3d.py --mode run
"""

import os

import modal

REPO_ID = "jma02/helmholtz-scattering-3d"
SD_FILE = "/sd/SD_k4_q8_L2_a1.25.npz"

# Sample / sensor configuration (mirrors scattering_dataset_3d.py defaults).
N_TX, N_RX, RHO = 128, 128, 2.5
N_MAX = 3
R_RANGE = (0.15, 0.35)
A_RANGE = (-0.5, -0.1)
C_MAX = 1.1
P = None  # interior Chebyshev order; None -> q + 4
SEED = 0
N_SAMPLES = 20000
SHARD = 50
GPU = "H100"

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = modal.App("helmholtz-scattering-3d")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "jax[cuda12]==0.10.1",
        "numpy==2.4.6",
        "scipy==1.17.1",
        "pyarrow",
        "huggingface_hub",
    )
    .add_local_dir(
        REPO_ROOT,
        remote_path="/root/jaxhps",
        ignore=[
            "data/**",
            ".git/**",
            "**/__pycache__/**",
            "**/*.pyc",
            "**/*.npz",
        ],
        copy=True,
    )
    .run_commands("pip install --no-deps /root/jaxhps")
)

vol = modal.Volume.from_name("jaxhps-sd")
hf_secret = modal.Secret.from_name("hf-token")


def _add_examples_to_path():
    import sys

    p = "/root/jaxhps/examples"
    if p not in sys.path:
        sys.path.insert(0, p)


@app.function(
    image=image,
    gpu=GPU,
    volumes={"/sd": vol},
    secrets=[hf_secret],
    timeout=900,
)
def smoke(n: int = 3):
    """Solve the first ``n`` samples on GPU; report timing + |M| stats."""
    _add_examples_to_path()
    os.environ["JAXHPS_TIMING"] = "1"  # per-stage breakdown
    import time

    import jax
    import numpy as np
    from scattering_dataset_3d import make_sensors, sample_config, solve_sample
    from wave_scattering_utils_3D import (
        build_cartesian_ctx,
        load_SD_matrices_3D,
    )

    print("jax devices:", jax.devices(), flush=True)
    sd = load_SD_matrices_3D(SD_FILE)
    tx, rx = make_sensors(N_TX, N_RX, RHO)
    ctx = build_cartesian_ctx(sd, tx, p=P)
    out = []
    for idx in range(n):
        rng = np.random.default_rng([SEED, idx])
        ns, c, r, a = sample_config(rng, N_MAX, R_RANGE, A_RANGE, C_MAX)
        t0 = time.time()
        M = solve_sample(sd, c, r, a, tx, rx, P, ctx=ctx)
        dt = time.time() - t0
        out.append((idx, ns, float(np.abs(M).max()), complex(M[0, 0]), dt))
        print(
            f"idx {idx} n={ns} |M|max={np.abs(M).max():.3e} "
            f"M[0,0]={M[0, 0]:.3e} dt={dt:.1f}s",
            flush=True,
        )
    return out


@app.function(
    image=image, volumes={"/sd": vol}, secrets=[hf_secret], timeout=1800
)
def setup_repo():
    """Create the HF dataset repo and upload ``metadata.json`` + ``README.md``."""
    _add_examples_to_path()
    import types

    from huggingface_hub import HfApi
    from scattering_dataset_3d import make_sensors, write_card, write_metadata
    from wave_scattering_utils_3D import load_SD_matrices_3D

    sd = load_SD_matrices_3D(SD_FILE)
    tx, rx = make_sensors(N_TX, N_RX, RHO)
    args = types.SimpleNamespace(
        p=P,
        n_tx=N_TX,
        n_rx=N_RX,
        rho=RHO,
        n_max=N_MAX,
        R_range=list(R_RANGE),
        A_range=list(A_RANGE),
        c_max=C_MAX,
        seed=SEED,
        n_samples=N_SAMPLES,
    )
    os.makedirs("/tmp/meta", exist_ok=True)
    meta = write_metadata("/tmp/meta", sd, args, tx, rx)
    write_card("/tmp/meta", meta)
    rp = "/tmp/meta/README.md"
    with open(rp) as f:
        txt = f.read().replace("REPO_ID", REPO_ID)
    with open(rp, "w") as f:
        f.write(txt)
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(REPO_ID, repo_type="dataset", exist_ok=True)
    api.upload_file(
        path_or_fileobj="/tmp/meta/metadata.json",
        path_in_repo="metadata.json",
        repo_id=REPO_ID,
        repo_type="dataset",
    )
    api.upload_file(
        path_or_fileobj=rp,
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="dataset",
    )
    print("repo ready:", REPO_ID, flush=True)


@app.function(
    image=image,
    gpu=GPU,
    volumes={"/sd": vol},
    secrets=[hf_secret],
    timeout=86400,
)
def run(lo: int = 0, hi: int = N_SAMPLES, shard: int = SHARD):
    """Generate samples ``[lo, hi)`` on one GPU container, streaming shards to HF.

    Resumable: shards already present in the repo (matched by file name) are
    skipped.  Each shard is written to ``data/train-{lo:07d}-{hi:07d}.parquet``.
    """
    _add_examples_to_path()
    import time

    import jax
    import numpy as np
    from huggingface_hub import HfApi
    from scattering_dataset_3d import (
        make_sensors,
        sample_config,
        solve_sample,
        write_shard,
    )
    from wave_scattering_utils_3D import (
        build_cartesian_ctx,
        load_SD_matrices_3D,
    )

    print("jax devices:", jax.devices(), flush=True)
    api = HfApi(token=os.environ["HF_TOKEN"])
    api.create_repo(REPO_ID, repo_type="dataset", exist_ok=True)
    existing = set(api.list_repo_files(REPO_ID, repo_type="dataset"))
    sd = load_SD_matrices_3D(SD_FILE)
    tx, rx = make_sensors(N_TX, N_RX, RHO)
    ctx = build_cartesian_ctx(sd, tx, p=P)
    os.makedirs("/tmp/ds", exist_ok=True)

    s_lo = lo
    while s_lo < hi:
        s_hi = min(s_lo + shard, hi)
        fname = f"train-{s_lo:07d}-{s_hi:07d}.parquet"
        repo_path = f"data/{fname}"
        if repo_path in existing:
            print(f"skip {repo_path} (exists)", flush=True)
            s_lo = s_hi
            continue
        rows = []
        t0 = time.time()
        for idx in range(s_lo, s_hi):
            rng = np.random.default_rng([SEED, idx])
            ns, c, r, a = sample_config(rng, N_MAX, R_RANGE, A_RANGE, C_MAX)
            M = solve_sample(sd, c, r, a, tx, rx, P, ctx=ctx)
            rows.append(dict(n=ns, centers=c, radii=r, amps=a, M=M, idx=idx))
        path = f"/tmp/ds/{fname}"
        write_shard(path, rows, N_MAX, N_TX, N_RX)
        api.upload_file(
            path_or_fileobj=path,
            path_in_repo=repo_path,
            repo_id=REPO_ID,
            repo_type="dataset",
        )
        os.remove(path)
        dt = time.time() - t0
        print(
            f"done {repo_path}: {len(rows)} samples in {dt:.0f}s "
            f"({dt / len(rows):.1f}s/sample)",
            flush=True,
        )
        s_lo = s_hi
    return "ok"


@app.local_entrypoint()
def main(
    mode: str = "smoke",
    n: int = 3,
    lo: int = 0,
    hi: int = N_SAMPLES,
    shard: int = SHARD,
):
    if mode == "smoke":
        print(smoke.remote(n))
    elif mode == "setup":
        setup_repo.remote()
    elif mode == "run":
        setup_repo.remote()
        print(run.remote(lo, hi, shard))
    else:
        raise SystemExit(f"unknown mode: {mode!r} (use smoke|setup|run)")
