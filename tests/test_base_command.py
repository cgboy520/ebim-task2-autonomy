"""Pedal-token reduction of a commanded base twist.

The bridge only understands exclusive String tokens at fixed velocities
(FWD/BACK/A/B at 0.5 m/s, A+C / B+C at 1.2 rad/s), so the mapping from a
continuous twist must pick the dominant axis by its own full-scale and
fall back to STOP below half scale. Demos record the applied (token)
velocities themselves, so half-scale thresholds recover the tokens
exactly.
"""

from __future__ import annotations

import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ebim_task2.base_command import pedal_token  # noqa: E402


@pytest.mark.parametrize(
    "twist,token",
    [
        ((0.5, 0.0, 0.0), "FWD"),
        ((-0.5, 0.0, 0.0), "BACK"),
        ((0.0, 0.5, 0.0), "A"),
        ((0.0, -0.5, 0.0), "B"),
        ((0.0, 0.0, 1.2), "A+C"),
        ((0.0, 0.0, -1.2), "B+C"),
        ((0.0, 0.0, 0.0), "STOP"),
    ],
)
def test_exact_token_velocities_round_trip(twist, token):
    assert pedal_token(*twist) == token


def test_below_half_scale_is_stop():
    # 0.24 m/s < half of the 0.5 m/s token velocity; 0.5 rad/s < half of 1.2.
    assert pedal_token(0.24, 0.0, 0.0) == "STOP"
    assert pedal_token(0.0, -0.24, 0.0) == "STOP"
    assert pedal_token(0.0, 0.0, 0.55) == "STOP"


def test_dominant_axis_is_normalised_by_its_own_full_scale():
    # 0.3/0.5 = 0.6 beats 0.6/1.2 = 0.5 — the linear axis wins even though
    # the raw yaw number is bigger.
    assert pedal_token(0.3, 0.0, 0.6) == "FWD"
    # 1.2/1.2 = 1.0 beats 0.3/0.5 = 0.6.
    assert pedal_token(0.3, 0.0, -1.2) == "B+C"


def test_noisy_policy_output_still_resolves():
    # A trained policy emits approximate velocities; the dominant axis and
    # sign must still resolve to one token.
    assert pedal_token(0.41, 0.07, -0.1) == "FWD"
    assert pedal_token(-0.05, 0.38, 0.2) == "A"
