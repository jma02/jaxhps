#!/usr/bin/env bash
# Reproducible build of fmm3dbie's Python interface, used by examples/gen_SD_3D.py
# to pre-compute Helmholtz single- and double-layer matrices on the cube boundary.
#
# Mirrors the 2D workflow that uses chunkIE (MATLAB) to generate S, D matrices,
# saved as .mat and consumed by examples/wave_scattering_utils.py:load_SD_matrices.
# The 3D analog generates .npz files; see examples/wave_scattering_utils_3D.py.
#
# Requirements: conda (for toolchain), curl/git, ~2 GB free disk.
# Produces:
#   ${ENV_DIR}         -- conda env with gfortran, openblas, py3.10, fmm3dpy
#   ${BIE_DIR}         -- cloned + built fmm3dbie source tree
# After running:
#   PATH=${ENV_DIR}/bin:$PATH  LD_LIBRARY_PATH=${ENV_DIR}/lib python examples/gen_SD_3D.py ...

set -euo pipefail

# Paths (override via env if you want them somewhere persistent).
: "${ENV_DIR:=/tmp/fmm3d-env310}"
: "${BIE_DIR:=/tmp/fmm3dbie}"
: "${CONDA:=$HOME/miniconda3/bin/conda}"

if [[ ! -x "${CONDA}" ]]; then
    echo "conda not found at ${CONDA}.  Set CONDA=/path/to/conda before running." >&2
    exit 1
fi

# 1. Toolchain env (Python 3.10 -- numpy.distutils, on which fmm3dbie's
#    setup.py still relies, is fully removed in 3.12; setuptools<60 keeps
#    distutils.msvccompiler addressable; numpy<2 keeps numpy.distutils alive).
echo ">>> creating env at ${ENV_DIR}"
"${CONDA}" create -p "${ENV_DIR}" -c conda-forge -y \
    python=3.10 "numpy<2" "setuptools<60" gfortran openblas pip charset-normalizer
"${ENV_DIR}/bin/pip" install --quiet fmm3dpy

# 2. fmm3dbie source (with FMM3D submodule).  Pinned to a known-working
# commit; bump deliberately when fmm3dbie changes break this build.
FMM3DBIE_COMMIT="ddc93f53e60181b79928fb896a678b49865810aa"
echo ">>> cloning fmm3dbie into ${BIE_DIR} at ${FMM3DBIE_COMMIT}"
rm -rf "${BIE_DIR}"
git clone --recurse-submodules https://github.com/fastalgorithms/fmm3dbie.git "${BIE_DIR}"
(cd "${BIE_DIR}" && git checkout "${FMM3DBIE_COMMIT}" && git submodule update --recursive)

# 3. Patch upstream typo: setup.py references stok_comb_vel.f, but the file is .f90.
sed -i "s|'../src/stok_wrappers/stok_comb_vel.f'|'../src/stok_wrappers/stok_comb_vel.f90'|" \
    "${BIE_DIR}/python/setup.py"

# 4. Build static + dynamic libfmm3dbie.
echo ">>> building libfmm3dbie"
cp "${BIE_DIR}/make.inc.linux.gnu.openblas" "${BIE_DIR}/make.inc"
(
    cd "${BIE_DIR}"
    PATH="${ENV_DIR}/bin:${PATH}" \
    LIBRARY_PATH="${ENV_DIR}/lib" \
    LD_LIBRARY_PATH="${ENV_DIR}/lib" \
    CPATH="${ENV_DIR}/include" \
    make -j"$(nproc)" lib
)

# 5. Build + install the Python wrapper.  --fallow-argument-mismatch is required
# because helm_comb_dir.f passes a scalar where an array is declared at one call
# site; gfortran 10+ errors on this by default.
echo ">>> building fmm3dbie Python wrapper"
(
    cd "${BIE_DIR}/python"
    rm -rf build
    PATH="${ENV_DIR}/bin:${PATH}" \
    LIBRARY_PATH="${ENV_DIR}/lib" \
    LD_LIBRARY_PATH="${ENV_DIR}/lib" \
    CPATH="${ENV_DIR}/include" \
    FMMBIE_LIBS="-fopenmp -lopenblas -fopenmp" \
    FFLAGS="-fallow-argument-mismatch -fPIC -O3 -funroll-loops -std=legacy -w" \
    "${ENV_DIR}/bin/python" setup.py install
)

# 6. Verify the wrapper imports and exposes the routines we need.
echo ">>> verifying build"
LD_LIBRARY_PATH="${ENV_DIR}/lib" "${ENV_DIR}/bin/python" - <<'PY'
import fmm3dbie, fmm3dpy
print("fmm3dpy", getattr(fmm3dpy, "__version__", "?"))
need = {"helm_comb_dir_fds_block_mem",
        "helm_comb_dir_fds_block_init",
        "helm_comb_dir_fds_block_matgen"}
have = set(dir(fmm3dbie))
missing = need - have
assert not missing, f"fmm3dbie missing routines: {missing}"
print("OK")
PY

echo ""
echo "build complete.  Run gen_SD_3D.py with:"
echo "  LD_LIBRARY_PATH=${ENV_DIR}/lib ${ENV_DIR}/bin/python examples/gen_SD_3D.py --q 8 --L 0 --kappa 4.0 --a 0.5 --out data/examples/SD_3D/SD_k4_q8_L0.npz"
