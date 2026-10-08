"""Shared fmm3dbie/JAX image for the two Modal forward solves."""

import os

import modal

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

fmm_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install(
        "gfortran",
        "libopenblas-dev",
        "make",
        "git",
        "curl",
    )
    .pip_install(
        "numpy<2",
        "setuptools<60",
        "fmm3dpy",
        "scipy",
        "jax[cuda12]",
        "charset_normalizer",
    )
    .run_commands(
        # Clone and build fmm3dbie
        "git clone --recurse-submodules https://github.com/fastalgorithms/fmm3dbie.git /opt/fmm3dbie",
        "cd /opt/fmm3dbie && git checkout ddc93f53e60181b79928fb896a678b49865810aa && git submodule update --recursive",
        # Patch setup.py typo
        "sed -i \"s|'../src/stok_wrappers/stok_comb_vel.f'|'../src/stok_wrappers/stok_comb_vel.f90'|\" /opt/fmm3dbie/python/setup.py",
        # Fix non-ASCII chars in Fortran sources (f2py encoding issue)
        "find /opt/fmm3dbie/src -name '*.f90' -exec sed -i 's/[^[:print:]\\t]//g' {} +",
        # Build static lib
        "cd /opt/fmm3dbie && cp make.inc.linux.gnu.openblas make.inc && make -j$(nproc) lib",
        # Build and install Python wrapper
        "cd /opt/fmm3dbie/python && FMMBIE_LIBS='-fopenmp -lopenblas' "
        "FFLAGS='-fallow-argument-mismatch -fPIC -O3 -funroll-loops -std=legacy -w' "
        "python setup.py install",
        # Verify
        "python -c 'import fmm3dbie; print(\"fmm3dbie OK\")'",
    )
    .add_local_dir(
        REPO_ROOT,
        remote_path="/root/jaxhps",
        ignore=["data/**", ".git/**", "**/__pycache__/**", "**/*.npz"],
        copy=True,
    )
    .run_commands("pip install --no-deps /root/jaxhps")
)
