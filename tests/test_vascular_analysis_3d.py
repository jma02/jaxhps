"""Prevent incomplete or incompatible solves from entering refinement results."""

import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from analyze_vascular_3d import summarize  # noqa: E402


def measurement(n, field):
    return dict(
        variant="dense",
        waves=10,
        n=n,
        phase="complete",
        converged=True,
        solve={"final_rel_res": 1e-9},
        field_real=field,
        field_imag=[0, 0],
        receivers=[[2.5, 0, 0], [0, 0, 2.5]],
        parameters=dict(seed=7, a=1.25, kappa=8 * np.pi),
    )


@pytest.mark.parametrize(
    "failure",
    [
        {"failed": True},
        {"skipped": True},
        {"phase": "evaluation"},
        {"converged": False},
        {"solve": {"final_rel_res": 1e-4}},
    ],
)
def test_unusable_case_does_not_become_reference(failure):
    coarse = measurement(64, [1, 2])
    invalid = measurement(96, [10, 20])
    invalid.update(failure)
    fine = measurement(128, [1, 2.1])
    rows = summarize([fine, invalid, coarse])
    assert rows[1]["receiver_change"] is None
    assert rows[2]["previous_n"] == 64
    assert rows[2]["receiver_change"] == pytest.approx(0.1 / np.hypot(1, 2.1))
    assert rows[2]["peak_device_gib"] is None


@pytest.mark.parametrize("parameter", ["seed", "a", "kappa"])
def test_incompatible_problem_is_rejected(parameter):
    coarse = measurement(64, [1, 2])
    fine = deepcopy(coarse)
    fine["n"] = 128
    fine["parameters"][parameter] += 1
    with pytest.raises(ValueError, match="incompatible"):
        summarize([coarse, fine])


def test_incompatible_receivers_are_rejected():
    coarse = measurement(64, [1, 2])
    fine = measurement(128, [1, 2])
    fine["receivers"][0][0] = 3
    with pytest.raises(ValueError, match="receiver sets"):
        summarize([coarse, fine])
