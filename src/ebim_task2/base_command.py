"""Mobile-base pedal command mapping.

The Isaac bridge drives the mobile base from string tokens on
``/pedal/state`` (std_msgs/String): FWD/BACK = ±x at 0.5 m/s, A/B = ±y at
0.5 m/s, A+C / B+C = ±yaw at 1.2 rad/s, STOP (or one sim-second of
silence) = zero. Tokens are exclusive — the bridge cannot mix axes — so a
commanded base twist is reduced to its dominant axis, normalised by that
axis's full-scale: half scale or more maps to the token, below maps to
STOP.

Publish at <= 10 Hz (the bridge's pedal subscription has queue depth 10).
"""

from __future__ import annotations

#: Velocity the bridge applies for FWD/BACK (±x) and A/B (±y), m/s.
FULL_SCALE_XY = 0.5
#: Yaw rate the bridge applies for A+C / B+C, rad/s.
FULL_SCALE_YAW = 1.2


def pedal_token(vx: float, vy: float, omega: float) -> str:
    """Reduce a base twist to the pedal token the bridge understands."""
    nx = abs(vx) / FULL_SCALE_XY
    ny = abs(vy) / FULL_SCALE_XY
    nw = abs(omega) / FULL_SCALE_YAW
    best = max(nx, ny, nw)
    if best < 0.5:
        return "STOP"
    if nx == best:
        return "FWD" if vx > 0 else "BACK"
    if ny == best:
        return "A" if vy > 0 else "B"
    return "A+C" if omega > 0 else "B+C"
