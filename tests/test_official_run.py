"""Pure-logic tests for the self-hunting official entry (no ROS needed)."""

from types import SimpleNamespace

import numpy as np
import pytest

from ebim_task2.official_run import (
    FROZEN_ARGS,
    attempts_left_in_budget,
    RESET_COST_S,
    RESET_SETTLE_S,
    build_chain_args,
    judge_attempt,
    parse_loose_label_map,
    select_target_bbox_px,
    should_start_attempt,
    stop_threshold,
    wait_for_reset_endpoints,
)
from ebim_task2.official_run import (
    ABSENT_ID,
    CONTINUATION_VALUE,
    ids_from_semantic_payload,
    parse_official_eval_reply,
    target_bbox_is_barebone_shaped,
)
from ebim_task2.perception import predict_official_verdict


# -- stop_threshold ---------------------------------------------------------
def test_short_horizon_ladder_is_the_continuation_value():
    # With five attempts the threshold is min(ramp, continuation value):
    # early steps bind on the value table, the tail steps on the ramp's
    # linear fall (pivot max(3, total//2) = 3).
    for idx in range(1, 6):
        rem = 5 - idx
        ramp = 0.80 if rem >= 3 else 0.80 * rem / 3
        assert stop_threshold(idx, 5) == pytest.approx(
            min(ramp, CONTINUATION_VALUE[rem]) if rem else 0.0)
    assert stop_threshold(5, 5) == 0.0
    # ...and it is a real relaxation, not a re-labelled hold.
    from ebim_task2.official_run import LADDER_HOLD as _hold
    assert stop_threshold(1, 5) < _hold


def test_shipped_cap_holds_early_then_relaxes_all_the_way_down():
    # At the shipped cap the hold binds first (the continuation value is
    # above it while 20+ attempts remain), then the ramp takes over and
    # drives the threshold to zero — the shape that governs run-level
    # zeros.
    assert stop_threshold(1, 24) == 0.80
    assert stop_threshold(24, 24) == 0.0
    seq = [stop_threshold(i, 24) for i in range(1, 25)]
    assert seq == sorted(seq, reverse=True)
    assert all(t < 0.80 for t in seq[12:])
    # The ramp is the cap: the threshold never exceeds it, so an
    # over-estimated continuation value cannot make the run hold out
    # longer than the ramp alone would have.
    for i in range(1, 25):
        rem = 24 - i
        ramp = 0.80 if rem >= 12 else 0.80 * rem / 12
        assert stop_threshold(i, 24) <= ramp + 1e-12


def test_ladder_is_monotone_within_a_run():
    for total in (1, 2, 3, 5, 8):
        seq = [stop_threshold(i, total) for i in range(1, total + 1)]
        assert seq == sorted(seq, reverse=True)
        assert seq[-1] == 0.0  # the last attempt accepts any valid lay


def test_single_attempt_accepts_any_valid_lay():
    assert stop_threshold(1, 1) == 0.0


# -- should_start_attempt ---------------------------------------------------
def test_first_attempt_always_starts():
    assert should_start_attempt(1, elapsed_s=1e9, budget_s=1.0,
                                avg_attempt_s=1e9)


def test_no_start_past_budget():
    assert not should_start_attempt(2, elapsed_s=100.0, budget_s=100.0,
                                    avg_attempt_s=10.0)


def test_start_needs_room_for_most_of_an_average_attempt():
    # remaining 100s vs avg 500s -> 0.8*avg = 400 > 100: skip
    assert not should_start_attempt(2, elapsed_s=0.0, budget_s=100.0,
                                    avg_attempt_s=500.0)
    # remaining 500s vs avg 500s -> 400 < 500: go
    assert should_start_attempt(2, elapsed_s=0.0, budget_s=500.0,
                                avg_attempt_s=500.0)


def test_unknown_average_starts_within_budget():
    assert should_start_attempt(3, elapsed_s=10.0, budget_s=20.0,
                                avg_attempt_s=None)


def test_reset_cost_charge_closes_the_self_zero_window():
    # The hole the charge closes: at judge time the raw lookahead says the
    # next attempt fits, so a valid sub-threshold lay gets reset away — but
    # the reset itself pushes the loop-start check past the governor and
    # the run ends on the freshly-emptied scene (final score 0). Charging
    # RESET_COST_S at judge time makes the two checks agree: the lay is
    # accepted instead of destroyed.
    assert RESET_COST_S > RESET_SETTLE_S
    budget, avg = 2700.0, 480.0
    elapsed_at_judge = budget - 0.8 * avg - RESET_COST_S / 2  # inside window
    assert should_start_attempt(5, elapsed_at_judge, budget, avg)
    assert not should_start_attempt(
        5, elapsed_at_judge + RESET_COST_S, budget, avg)


# -- build_chain_args -------------------------------------------------------
def test_frozen_recipe_is_preserved_and_extras_append():
    assert build_chain_args(None) == list(FROZEN_ARGS)
    out = build_chain_args(["--film"])
    assert out[: len(FROZEN_ARGS)] == list(FROZEN_ARGS)
    assert out[-1] == "--film"


# -- judge_attempt ----------------------------------------------------------
def test_no_verdict_never_accepts():
    assert judge_attempt(True, None, 1, 5, None) == (False, 0.0, False)


def test_good_lay_accepted_against_ladder():
    accept, gated, futile = judge_attempt(
        True, ("liner_only", 0.82), 1, 5, None)
    assert accept and gated == 0.82 and not futile
    # A lay is banked iff it beats what playing on is worth. With 4 further
    # attempts that is CONTINUATION_VALUE[4] ~ 0.774, so 0.78 is banked...
    assert judge_attempt(True, ("liner_only", 0.78), 1, 5, None)[0]
    # ...while the same lay on a 24-attempt run is not: there, continuing
    # really is worth more than 0.78 (the 0.80 hold binds).
    assert not judge_attempt(True, ("liner_only", 0.78), 1, 24, None)[0]


def test_mid_lay_held_early_accepted_late():
    early = judge_attempt(True, ("both_liner_dominant", 0.55), 1, 24, None)
    late = judge_attempt(True, ("both_liner_dominant", 0.55), 20, 24, None)
    assert not early[0] and late[0]


def test_wrong_face_raw_iou_is_gated_to_zero_but_retryable():
    # The dev evaluator reports a raw IoU for a wrong-face lay; the
    # official orientation gate zeroes it. NOT futile: a scene_reset
    # restores the liner-up spawn.
    accept, gated, futile = judge_attempt(
        True, ("thermalpad_only", 0.469), 5, 5, None)
    assert not accept and gated == 0.0 and not futile
    assert judge_attempt(True, ("both_thermalpad_dominant", 0.3), 1, 5,
                         None)[2] is False


def test_sideways_is_zero_but_retryable():
    accept, gated, futile = judge_attempt(True, ("sideways", 0.0), 2, 5, None)
    assert not accept and gated == 0.0 and not futile


def test_incomplete_chain_never_accepts_even_with_a_verdict():
    accept, _, futile = judge_attempt(
        False, ("liner_only", 0.65), 5, 5, None)
    assert not accept and not futile


def test_stop_iou_override_wins_over_ladder():
    assert judge_attempt(True, ("liner_only", 0.45), 1, 5, 0.40)[0]
    assert not judge_attempt(True, ("liner_only", 0.45), 5, 5, 0.60)[0]


def test_last_attempt_keeps_any_scoring_lay():
    accept, gated, _ = judge_attempt(True, ("liner_only", 0.05), 5, 5, None)
    assert accept and gated == 0.05


# -- live loose-bbox target helpers ----------------------------------------
def _det(cid, score, cx, cy, sx, sy, nested=True):
    """A vision_msgs-shaped detection stub; nested=False mimics the older
    bridge generation (xy on center, hypothesis fields on the result)."""
    hyp = SimpleNamespace(class_id=cid, score=score)
    result = SimpleNamespace(hypothesis=hyp) if nested else hyp
    center = (SimpleNamespace(position=SimpleNamespace(x=cx, y=cy))
              if nested else SimpleNamespace(x=cx, y=cy))
    bbox = SimpleNamespace(center=center, size_x=sx, size_y=sy)
    return SimpleNamespace(results=[result], bbox=bbox)


def test_loose_label_map_parses_evaluator_payload():
    payload = ('{"0":{"class":"board"},"1":{"class":"target"},'
               '"2":{"class":"liner"},"time_stamp":{"sec":129}}')
    assert parse_loose_label_map(payload) == {
        "board": 0, "target": 1, "liner": 2}
    assert parse_loose_label_map("not json") == {}


def test_select_target_bbox_matches_by_id_and_name_and_converts():
    # Numbers from the a4 evaluator artifact: the loose target
    # detection center (616, 381.5) size 82x17 -> x1y1x2y2 (575,373,657,390).
    msg = SimpleNamespace(detections=[
        _det("0", 1.0, 100.0, 50.0, 10.0, 10.0),
        _det("1", 1.0, 616.0, 381.5, 82.0, 17.0),
    ])
    assert select_target_bbox_px(msg, 1) == (373, 390, 575, 657)
    msg2 = SimpleNamespace(detections=[
        _det("target", 1.0, 616.0, 381.5, 82.0, 17.0, nested=False)])
    assert select_target_bbox_px(msg2, None) == (373, 390, 575, 657)
    assert select_target_bbox_px(SimpleNamespace(detections=[]), 1) is None


def test_select_target_bbox_prefers_best_score_and_skips_empty():
    msg = SimpleNamespace(detections=[
        _det("target", 0.4, 100.0, 100.0, 10.0, 10.0),
        _det("target", 0.9, 200.0, 200.0, 20.0, 10.0),
        _det("target", 1.0, 300.0, 300.0, 0.0, 10.0),   # degenerate: skipped
    ])
    assert select_target_bbox_px(msg, None) == (195, 205, 190, 210)


# -- reset handshake --------------------------------------------------------
def _counters(req_after: int, evt_after: int):
    """Fakes whose match counts flip to 1 after N spins (discovery delay)."""
    ticks = {"n": 0, "t": 0.0}

    def spin():
        ticks["n"] += 1
        ticks["t"] += 0.1

    return (lambda: 1 if ticks["n"] >= req_after else 0,
            lambda: 1 if ticks["n"] >= evt_after else 0,
            spin,
            lambda: ticks["t"])


def test_reset_handshake_waits_for_both_halves_then_returns():
    req, evt, spin, now = _counters(req_after=3, evt_after=7)
    assert wait_for_reset_endpoints(req, evt, spin, now, timeout_s=5.0) == (
        True, True)
    # It must not return on the request half alone: the ack subscription
    # matching LATE is the failure that loses the event.
    assert evt() == 1


def test_reset_handshake_reports_which_half_never_matched():
    req, evt, spin, now = _counters(req_after=1, evt_after=10_000)
    assert wait_for_reset_endpoints(req, evt, spin, now, timeout_s=1.0) == (
        True, False)


def test_reset_handshake_returns_immediately_when_already_matched():
    calls = {"spin": 0}
    assert wait_for_reset_endpoints(
        lambda: 1, lambda: 2,
        lambda: calls.__setitem__("spin", calls["spin"] + 1),
        lambda: 0.0, timeout_s=5.0) == (True, True)
    assert calls["spin"] == 0


# -- semantic-labels id source ---------------------------------------------
LIVE_PAYLOAD = ('{"0":{"class":"BACKGROUND"},"1":{"class":"UNLABELLED"},'
                '"2":{"class":"target"},"3":{"class":"thermalpad"},'
                '"4":{"class":"board"},"5":{"class":"liner"},'
                '"time_stamp":{"nanosec":733333528,"sec":3}}')
# Observed live after in-place scene resets: thermalpad drops out entirely.
DEGRADED_PAYLOAD = ('{"0":{"class":"BACKGROUND"},"1":{"class":"UNLABELLED"},'
                    '"2":{"class":"target"},"4":{"class":"board"},'
                    '"5":{"class":"liner"},"time_stamp":{"sec":3}}')


def test_semantic_payload_yields_the_evaluators_raw_ids():
    assert ids_from_semantic_payload(LIVE_PAYLOAD) == {
        "thermalpad": 3, "target": 2, "liner": 5}


def test_degraded_payload_reproduces_the_evaluators_zero_pad_regime():
    ids = ids_from_semantic_payload(DEGRADED_PAYLOAD)
    assert ids == {"thermalpad": ABSENT_ID, "target": 2, "liner": 5}
    # ABSENT_ID must count zero pixels on any real mask, so a both-present
    # lay resolves liner_only exactly as the official evaluator does.
    mask = np.array([[5, 5, 2], [5, 3, 2], [0, 0, 0]], dtype=np.int32)
    case, _ = predict_official_verdict(
        mask, thermalpad_id=ids["thermalpad"], target_id=ids["target"],
        liner_id=ids["liner"], target_bbox=(0, 1, 2, 2))
    assert case == "liner_only"
    # With the complete map the same frame is a dominance decision instead.
    full = ids_from_semantic_payload(LIVE_PAYLOAD)
    case_full, _ = predict_official_verdict(
        mask, thermalpad_id=full["thermalpad"], target_id=full["target"],
        liner_id=full["liner"], target_bbox=(0, 1, 2, 2))
    assert case_full == "sideways"


def test_payload_without_target_or_liner_is_not_an_id_source():
    assert ids_from_semantic_payload('{"0":{"class":"board"}}') is None
    assert ids_from_semantic_payload('{"2":{"class":"target"}}') is None
    assert ids_from_semantic_payload("not json") is None
    assert ids_from_semantic_payload("[1,2,3]") is None


# -- scene shape check ------------------------------------------------------
def test_barebone_plate_bbox_is_wide():
    # A barebone startup bbox (x 575..648, y 283..321) in the
    # (y0, y1, x0, x1) convention.
    ok, why = target_bbox_is_barebone_shaped((283, 321, 575, 648))
    assert ok, why


def test_room_scene_plate_bbox_is_flagged():
    # Same plate, room layout: the 0.12 m axis runs along world Y, so the
    # box is the barebone one transposed.
    ok, why = target_bbox_is_barebone_shaped((283, 356, 575, 613))
    assert not ok and "ROOM" in why


def test_degenerate_and_absurd_bboxes_are_flagged():
    assert not target_bbox_is_barebone_shaped((300, 300, 500, 500))[0]
    assert not target_bbox_is_barebone_shaped((0, 700, 0, 1200))[0]


# -- degraded-label-map hedge ----------------------------------------------
def test_strict_reading_is_reported_but_does_not_block_by_default():
    # Live map lost 'thermalpad' -> evaluator sees liner_only and a good
    # IoU; a repaired evaluator would call the same frame sideways. The
    # default is to bank what the evaluator will actually score today.
    accept, gated, _ = judge_attempt(True, ("liner_only", 0.85), 1, 24, None,
                                     strict_verdict=("sideways", 0.0))
    assert accept and gated == 0.85


def test_strict_reading_blocks_an_early_stop_when_opted_in():
    accept, gated, _ = judge_attempt(True, ("liner_only", 0.85), 1, 24, None,
                                     strict_verdict=("sideways", 0.0),
                                     require_strict=True)
    assert not accept and gated == 0.85


def test_strict_reading_that_also_passes_still_accepts():
    accept, gated, _ = judge_attempt(True, ("liner_only", 0.85), 1, 24, None,
                                     strict_verdict=("both_liner_dominant", 0.84),
                                     require_strict=True)
    assert accept and gated == 0.85


def test_relaxed_end_of_the_ladder_ignores_the_strict_reading():
    # Last attempt (threshold 0.0): a lay that scores TODAY beats walking
    # away with a freshly reset scene.
    accept, gated, _ = judge_attempt(True, ("liner_only", 0.31), 24, 24, None,
                                     strict_verdict=("sideways", 0.0),
                                     require_strict=True)
    assert accept and gated == 0.31
    # ...and the same exemption applies to an explicit --stop-iou 0.
    accept, _, _ = judge_attempt(True, ("liner_only", 0.31), 5, 24, 0.0,
                                 strict_verdict=("sideways", 0.0),
                                 require_strict=True)
    assert accept


def test_no_strict_reading_leaves_the_ladder_unchanged():
    accept, _, _ = judge_attempt(True, ("liner_only", 0.85), 1, 24, None)
    assert accept


# -- ladder horizon vs the wall budget -------------------------------------
def test_budget_horizon_shrinks_the_ladder_when_the_wall_binds():
    # 30-minute budget, ~370 s attempts + 40 s reset: about 3 more fit.
    n = attempts_left_in_budget(elapsed_s=370.0, budget_s=1800.0,
                                avg_attempt_s=370.0, cap_left=23)
    assert n == 3
    # So attempt 1 of a nominal 24 is really attempt 1 of 4, and the ladder
    # must already be ramping rather than holding 0.80 to the bitter end.
    assert stop_threshold(2, 1 + n) < 0.80


def test_budget_horizon_leaves_a_roomy_run_alone():
    # 180 min at the same pace: the 24-attempt cap binds first, not the wall.
    n = attempts_left_in_budget(elapsed_s=370.0, budget_s=10800.0,
                                avg_attempt_s=370.0, cap_left=23)
    assert n == 23
    # The wall did not shrink the horizon, so the threshold is the full
    # 24-attempt ladder's own top: min(hold, V[22]) — V since the pooled
    # 56-draw table puts V[22] a shade under the 0.80 hold.
    assert stop_threshold(2, 1 + n) == pytest.approx(
        min(0.80, CONTINUATION_VALUE[22]))
    assert stop_threshold(2, 1 + n) > 0.67


def test_budget_horizon_never_exceeds_the_cap_or_goes_negative():
    assert attempts_left_in_budget(0.0, 10800.0, 370.0, 0) == 0
    assert attempts_left_in_budget(10800.0, 10800.0, 370.0, 24) == 0
    # No timing history yet -> assume the cap is the binding horizon.
    assert attempts_left_in_budget(0.0, 1800.0, None, 24) == 24


# -- parse_official_eval_reply ---------------------------------------------
def test_eval_reply_parses_the_node_message_format():
    # Byte-for-byte the f-string scripts/evaluation/task2/node.py appends:
    #   f" eval_iou={iou:.4f} orientation={'correct'|'wrong'}[{case}]"
    msg = ("Saved [/output/evaluate/eval_camera_rgb_20260804_101010_000001.jpg, "
           "/output/evaluate/eval_camera_iou_20260804_101010_000001.json] "
           "frame_id=eval_camera eval_iou=0.6955 "
           "orientation=correct[both_liner_dominant]")
    assert parse_official_eval_reply(msg) == ("both_liner_dominant", 0.6955)


def test_eval_reply_wrong_face_gates_to_zero_through_judge_attempt():
    # The official case vocabulary flows into the SAME gate the replica
    # uses — a wrong-face verdict must never be accepted, whatever its raw
    # IoU says.
    v = parse_official_eval_reply(
        " eval_iou=0.3000 orientation=wrong[both_thermalpad_dominant]")
    assert v == ("both_thermalpad_dominant", 0.3)
    accept, gated, futile = judge_attempt(True, v, 1, 24, None)
    assert not accept and gated == 0.0 and not futile


def test_eval_reply_correct_verdict_accepts_like_a_replica_one():
    v = parse_official_eval_reply(
        " eval_iou=0.8100 orientation=correct[liner_only]")
    accept, gated, _ = judge_attempt(True, v, 1, 24, None)
    assert accept and gated == pytest.approx(0.81)


def test_eval_reply_none_on_service_side_failures():
    # success=False replies carry no verdict fields; the caller must fall
    # back to the replica rather than treat them as 0-IoU verdicts.
    assert parse_official_eval_reply(
        "No image received yet on /isaac/eval_camera/image_raw") is None
    assert parse_official_eval_reply(
        "Evaluation requires bbox_2d_tight and a label map "
        "(bbox_2d_tight_labels or semantic_labels)") is None
    assert parse_official_eval_reply("") is None


# -- live-pool shrinkage ----------------------------------------------------
def test_live_draws_pull_the_ladder_toward_this_run():
    """Live draws re-weight the static pool toward this run's own
    distribution. Two aborts must lower the bar the next lay has to clear,
    or a decent lay gets reset away and the run ends on an abort."""
    from ebim_task2.official_run import continuation_values

    assert continuation_values(()) == CONTINUATION_VALUE
    dry = stop_threshold(3, 6, (0.0, 0.0))
    assert dry < stop_threshold(3, 6, ())
    # ... and a good draw pushes it back up, so a lucky scene is not
    # talked into accepting its next mediocre lay
    rich = stop_threshold(3, 6, (0.7, 0.8))
    assert rich > stop_threshold(3, 6, ())


def test_live_pool_never_breaks_the_ladder_invariants():
    for obs in ((), (0.0,), (0.0, 0.0, 0.0), (0.9, 0.9), (0.0, 0.35, 0.7)):
        seq = [stop_threshold(i, 6, obs) for i in range(1, 7)]
        assert seq[-1] == 0.0            # last attempt takes any valid lay
        assert all(a >= b - 1e-12 for a, b in zip(seq, seq[1:]))
        from ebim_task2.official_run import LADDER_HOLD
        assert all(0.0 <= v <= LADDER_HOLD + 1e-12 for v in seq)


def test_live_pool_shrinkage_is_gradual():
    """One draw must not hand the run's whole policy to a single sample."""
    from ebim_task2.official_run import LIVE_POOL_HALF_WEIGHT

    one = stop_threshold(1, 6, (0.0,))
    banked = stop_threshold(1, 6, ())
    assert one > banked * (LIVE_POOL_HALF_WEIGHT
                           / (1.0 + LIVE_POOL_HALF_WEIGHT)) - 1e-9
    many = stop_threshold(1, 6, (0.0,) * 12)
    assert many < one
