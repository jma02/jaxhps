"""Guard benchmark admission against residual and reference uncertainty."""

import json

import pytest

from examples.analyze_accuracy_3d import analyze, certified


def case(tmp_path, name, *, kind="radial", n=16, value=1.0, residual=1e-9):
    stats = dict(
        final_rel_res=residual,
        true_res_history=[1.0, residual],
        n_matvec=7,
        info=int(residual > 1e-8),
    )
    row = dict(
        parameters=dict(kind=kind, solver="fft", kappa=4, n=n),
        targets=[[2.5, 0, 0]],
        weights=[1.0],
        field_real=[value],
        field_imag=[0.0],
        field_relative_error=1e-5,
        first_info=stats,
        repeats=[dict(seconds=1.0, info=stats)] * 3,
        unknowns=n**3,
        setup_seconds=1.0,
        first_solve_seconds=1.0,
        cold_setup_solve_seconds=2.0,
        evaluation_seconds=1.0,
        gpu_memory=dict(peak_bytes_in_use=1024, peak_pool_bytes=2048),
        host_peak_rss_bytes=4096,
        git_commit="test",
        git_dirty=False,
    )
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(row))
    return path


def test_missing_reference_controls_prevent_a_matched_error_claim(tmp_path):
    result = analyze([case(tmp_path, "radial")], [])
    assert result["rows"][0]["reference_change"] is None
    assert all(m["best_case"] is None for m in result["matches"])


def test_nonconverged_case_is_excluded_despite_small_field_error(tmp_path):
    path = case(tmp_path, "unconverged", residual=0.1)
    result = analyze([path], [dict(kappa=4, changes=dict(all_tight=1e-12))])
    assert not result["rows"][0]["converged"]
    assert all(m["best_case"] is None for m in result["matches"])


def test_phantom_finest_grid_is_not_treated_as_exact(tmp_path):
    paths = [
        case(tmp_path, str(n), kind="phantom", n=n, value=value)
        for n, value in ((16, 1.002), (24, 1.0005), (32, 1.0))
    ]
    result = analyze(paths, [])
    assert result["phantom_references"][4]["relative_change"] == pytest.approx(
        0.0005
    )
    assert all(
        m["best_case"] is not None
        for m in result["matches"]
        if m["ceiling"] == 0.01
    )
    assert all(
        m["best_case"] is None
        for m in result["matches"]
        if m["ceiling"] < 0.01
    )


def test_inconsistent_residual_history_is_rejected():
    with pytest.raises(ValueError, match="history"):
        certified(
            dict(final_rel_res=1e-9, true_res_history=[1.0, 0.1], info=0)
        )
    with pytest.raises(ValueError, match="flag"):
        certified(dict(final_rel_res=0.1, true_res_history=[1.0, 0.1], info=0))
