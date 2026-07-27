"""Balance test: stamina drain/regen sanity checks over realistic
match-length durations. A 90-minute half (well, full match here) is
obviously too long to simulate every tick in a test, so we validate over
shorter but still meaningful windows (a sustained sprint, then a recovery
jog) and check the *shape* of the behaviour: stamina drains under sustained
sprinting, drains faster for low-stamina-attribute players, and recovers
when resting - with the speed penalty (up to 65%) kicking in as stamina
depletes.
"""
from __future__ import annotations

import random

from footballcoach.engine.movement import MovementParams, drain_stamina, regen_stamina, stamina_multiplier

RNG_REDUCTION = 0.3


def test_stamina_drains_over_sustained_sprint(balance_recorder):
    params = MovementParams.from_config()
    stamina = 1.0
    dt = 1.0
    samples = {}
    for second in range(1, 121):
        stamina = drain_stamina(params, stamina, stamina_attr=0.5, effort=1.0, dt_s=dt)
        if second in (10, 30, 60, 90, 120):
            samples[f"t={second}s"] = round(stamina, 3)
    balance_recorder.report("stamina_drain_over_120s_sprint_attr_0.5", samples)
    assert samples["t=120s"] < samples["t=10s"]
    assert samples["t=120s"] < 1.0


def test_higher_stamina_attribute_drains_slower_over_time(balance_recorder):
    # A continuous 45s sprint is used here (rather than 90s) so that neither
    # player has fully bottomed out at 0 by the end - a full 90s continuous
    # sprint drains even a high-stamina-attribute player to 0, which would
    # make this differentiation check meaningless (0 vs 0).
    params = MovementParams.from_config()
    low_attr_stamina = 1.0
    high_attr_stamina = 1.0
    for _ in range(45):
        low_attr_stamina = drain_stamina(params, low_attr_stamina, stamina_attr=0.1, effort=1.0, dt_s=1.0)
        high_attr_stamina = drain_stamina(params, high_attr_stamina, stamina_attr=0.9, effort=1.0, dt_s=1.0)
    stats = {
        "remaining_after_45s_low_attr_0.1": round(low_attr_stamina, 3),
        "remaining_after_45s_high_attr_0.9": round(high_attr_stamina, 3),
    }
    balance_recorder.report("stamina_drain_attribute_comparison", stats)
    assert high_attr_stamina > low_attr_stamina


def test_stamina_regenerates_when_resting(balance_recorder):
    params = MovementParams.from_config()
    stamina = 0.2  # exhausted
    samples = {"t=0s": stamina}
    for second in range(1, 61):
        stamina = regen_stamina(params, stamina, stamina_attr=0.5, dt_s=1.0)
        if second in (10, 30, 60):
            samples[f"t={second}s"] = round(stamina, 3)
    balance_recorder.report("stamina_regen_over_60s_rest_attr_0.5", samples)
    assert samples["t=60s"] > samples["t=0s"]


def test_stamina_speed_penalty_shape(balance_recorder):
    """Reports the speed multiplier at various stamina levels, confirming
    the up-to-65%-reduction design spec (multiplier at stamina=0 should be
    1 - 0.65 = 0.35)."""
    params = MovementParams.from_config()
    table = {
        f"stamina={s}": round(stamina_multiplier(params, s), 3)
        for s in (1.0, 0.75, 0.5, 0.25, 0.0)
    }
    balance_recorder.report("stamina_speed_multiplier_table", table)
    assert table["stamina=1.0"] == 1.0
    assert abs(table["stamina=0.0"] - 0.35) < 1e-9


def test_low_stamina_attribute_player_exhausts_and_recovers_realistically(balance_recorder):
    """Full drain-then-rest cycle for a low-stamina-attribute player: sprints
    for 60s, then rests for 60s, reporting the stamina trajectory and the
    resulting speed penalty at the point of exhaustion."""
    params = MovementParams.from_config()
    stamina = 1.0
    for _ in range(60):
        stamina = drain_stamina(params, stamina, stamina_attr=0.1, effort=1.0, dt_s=1.0)
    exhausted_stamina = stamina
    exhausted_speed_mult = stamina_multiplier(params, stamina)

    for _ in range(60):
        stamina = regen_stamina(params, stamina, stamina_attr=0.1, dt_s=1.0)
    recovered_stamina = stamina

    stats = {
        "stamina_after_60s_sprint": round(exhausted_stamina, 3),
        "speed_multiplier_at_exhaustion": round(exhausted_speed_mult, 3),
        "stamina_after_60s_further_rest": round(recovered_stamina, 3),
    }
    balance_recorder.report("stamina_full_cycle_low_attr_player", stats)
    assert exhausted_stamina < 1.0
    assert recovered_stamina > exhausted_stamina
