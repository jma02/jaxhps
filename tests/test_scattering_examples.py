"""CPU checks for example assembly and dispatch without a native FMM backend."""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import jax
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import gen_nearfield_3D as nearfield  # noqa: E402
import wave_scattering_utils_3D as scattering  # noqa: E402


@pytest.mark.parametrize("bulk", [False, True])
@pytest.mark.parametrize("near_ratio", [0.5, 4.0])
def test_generate_nearfield(tmp_path, monkeypatch, bulk, near_ratio):
    _, _, _, src, _ = nearfield.build_cube_srcvals(1.25, 2, 1)
    pts, normals = src[:3].T, src[9:12].T
    n = len(pts)
    wts = np.full(n, 1.0 / n)
    rng = np.random.default_rng(0)
    S, D = rng.normal(size=(2, n, n)) + 1j * rng.normal(size=(2, n, n))
    calls = []

    def matgen(*args):
        zpars = args[7]
        rows, cols = args[-3] - 1, args[-2] - 1
        assert zpars[0] == 4.0
        assert tuple(zpars[1:]) in ((1, 0), (0, 1))
        assert np.all((0 <= rows) & (rows < n))
        assert np.all((0 <= cols) & (cols < n))
        calls.append(tuple(zpars[1:]))
        return (zpars[1] * S + zpars[2] * D)[np.ix_(rows, cols)]

    monkeypatch.setitem(
        sys.modules,
        "fmm3dbie",
        SimpleNamespace(
            surf_vals_to_coefs=lambda *args: np.zeros_like(args[-1]),
            get_qwts=lambda *args: wts,
            helm_comb_dir_fds_block_mem=lambda *args: (1, 0, 1),
            helm_comb_dir_fds_block_init=lambda *args: (
                np.zeros(1),
                np.zeros(1),
            ),
            helm_comb_dir_fds_block_matgen=matgen,
        ),
    )
    out = tmp_path / "nearfield.npz"
    nearfield.generate_nearfield(
        out, 1.25, 2, 1, 4.0, near_ratio=near_ratio, bulk=bulk
    )
    actual = scattering.load_nearfield_correction(out)
    expected = scattering.build_nearfield_correction(
        {"S": S, "D": D},
        pts,
        normals,
        wts,
        2,
        1,
        4.0,
        near_ratio=near_ratio,
    )
    for key in ("S_corr", "D_corr"):
        np.testing.assert_array_equal(
            actual[key].toarray(), expected[key].toarray()
        )
    for key, value in (
        ("boundary_points", pts),
        ("normals", normals),
        ("wts", wts),
    ):
        np.testing.assert_array_equal(actual[key], value)
    assert actual["n_near_pairs"] == expected["n_near_pairs"]
    assert len(calls) == 2 * (1 if bulk else expected["n_near_pairs"])


@pytest.mark.parametrize("jax_tdtn", [False, True])
@pytest.mark.parametrize("n_src", [1, 3])
def test_fmm_solver_dispatch(monkeypatch, jax_tdtn, n_src):
    n = 5
    T = np.diag(np.linspace(0.1, 0.5, n)).astype(complex)
    S, D = 0.2 * np.eye(n), 0.1 * np.eye(n)
    rng = np.random.default_rng(1)
    shape = (n,) if n_src == 1 else (n, n_src)
    uin = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    uin_dn = 2 * uin
    monkeypatch.setattr(scattering, "fmm_matvec_S", lambda v, *args: S @ v)
    monkeypatch.setattr(scattering, "fmm_matvec_D", lambda v, *args: D @ v)
    x64_enabled = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    try:
        imp, field, derivative, info = scattering.solve_bie_gmres_fmm(
            T,
            np.zeros((n, 3)),
            np.zeros((n, 3)),
            np.ones(n),
            4.0,
            4.0,
            uin,
            uin_dn,
            {"S_corr": None, "D_corr": None, "fmm_eps": 1e-7},
            tol=1e-12,
            use_gpu_tdtn=jax_tdtn,
        )
    finally:
        jax.config.update("jax_enable_x64", x64_enabled)
    expected = np.linalg.solve(
        0.5 * np.eye(n) - D + S @ T, S @ (uin_dn - T @ uin)
    )
    expected = expected.reshape(n, n_src)
    expected_dn = T @ (expected + uin.reshape(n, n_src)) - uin_dn.reshape(
        n, n_src
    )
    np.testing.assert_allclose(field, expected, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(derivative, expected_dn, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(
        imp, expected_dn + 4j * expected, atol=1e-12, rtol=1e-12
    )
    assert info == {
        "gmres_info": [0] * n_src,
        "n_src": n_src,
        "converged": True,
    }


@pytest.mark.parametrize("push_only", [False, True])
@pytest.mark.parametrize("publish", [False, True])
def test_dataset_upload_modes(tmp_path, monkeypatch, push_only, publish):
    import scattering_dataset_3d as dataset

    api = Mock()
    monkeypatch.setitem(sys.modules, "huggingface_hub", api)
    monkeypatch.setenv("HF_TOKEN", "test-token")
    load = Mock(return_value={"a": 0.5, "kappa": 4.0, "q": 2, "L": 1})
    monkeypatch.setattr(dataset, "load_SD_matrices_3D", load)
    for name in ("build_cartesian_ctx", "solve_sample", "write_shard"):
        monkeypatch.setattr(dataset, name, Mock())
    argv = ["dataset", "--out", str(tmp_path), "--n_samples", "1"]
    if push_only:
        argv.append("--push_only")
    if publish:
        argv.extend(["--push_to_hub", "test/dataset"])
    monkeypatch.setattr(sys, "argv", argv)
    if push_only and not publish:
        with pytest.raises(SystemExit, match="--push_only requires"):
            dataset.main()
    else:
        dataset.main()
    assert load.call_count == int(not push_only)
    assert dataset.solve_sample.call_count == int(not push_only)
    assert dataset.write_shard.call_count == int(not push_only)
    if publish:
        api.HfApi.assert_called_once_with(token="test-token")
        api.HfApi.return_value.create_repo.assert_called_once_with(
            "test/dataset", repo_type="dataset", exist_ok=True
        )
        api.HfApi.return_value.upload_folder.assert_called_once_with(
            folder_path=str(tmp_path),
            repo_id="test/dataset",
            repo_type="dataset",
        )
    else:
        api.HfApi.assert_not_called()
