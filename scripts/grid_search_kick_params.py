"""Grid search over kicking error parameters.

Targets:
  Penalties (running, power=0.8):
    - precision=0.5 centre:   >95% scored
    - precision=0.5 corner:   50-80% scored
    - precision=0.8 corner:   85-95% scored

  Passes (stationary passer, auto-pace):
    - 10m  precision=0.5:    >80%
    - 10m  precision=0.15:   >70%    (was failing before)
    - 25m  precision=0.5:    >50%
    - 25m  precision=0.9:    >85%

N=20 per scenario. Results are quick so we can scan many combinations.
Run with:  uv run python scripts/grid_search_kick_params.py
"""
from __future__ import annotations

import math
import random
import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import footballcoach.config.loader as loader_mod

from footballcoach.engine.kicking import KickingParams, PassingParams, pass_speed_mps, max_kick_speed_mps
from footballcoach.engine.match import Match
from footballcoach.engine.movement import MovementParams, effective_top_speed
from footballcoach.entities import Ball, Pitch, Team
from footballcoach.mathutils import Vector3
from footballcoach.orders import KickOrder

RNG_REDUCTION = 0.3
N = 70  # trials per scenario

# --------------------------------------------------------------------------
# Parameter grid  - pass calibration only, kicking params fixed at known-good values
# --------------------------------------------------------------------------
ANGLE_BASE   = [0.0055]
ANGLE_SCALE  = [0.04]
PWR_SCALE    = [1.4]
PWR_EXP      = [2.0]
PREC_EXP     = [0.87]
AIM_HEIGHT   = [0.11]

# Best calibrated params (O10=1.17, O60=1.30, dexp=1.0 → base=1.144, drag=0.0026).
# These mirror physics.json. To explore alternatives, add more values to the lists.
TARGET_O10 = [1.17]   # overshoot at 10m
TARGET_O60 = [1.30]   # overshoot at 60m
DRAG_EXP   = [1.0]    # linear curve

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def make_kicking_params(base, scale, pwr_scale, pwr_exp, prec_exp=1.0):
    return KickingParams(
        angle_error_base_rad=base,
        angle_error_scale_rad=scale,
        power_error_scale=pwr_scale,
        power_error_exponent=pwr_exp,
        precision_exponent=prec_exp,
        power_base_mps=11.47,
        power_scale_mps=15.29,
        firsttime_precision_weight=0.8,
        running_power_coefficient=0.7,
    )


def make_passing_params(overshoot, drag_adj=0.0, drag_exp=1.0, aim_height=0.11):
    return PassingParams(
        power_overshoot_factor=overshoot,
        overshoot_drag_factor=drag_adj,
        overshoot_drag_exponent=drag_exp,
        min_speed_mps=2.0,
        max_speed_mps=25.0,
        pass_aim_height_m=aim_height,
    )


def make_passing_params_from_targets(o10: float, o60: float, exp: float,
                                     aim_height: float = 0.11) -> "PassingParams | None":
    """Derive (base, drag) analytically so overshoot(10m)=o10 and overshoot(60m)=o60.
    Returns None when base < 1.0 (ball would undershoot at short distances).
    """
    d10 = 10.0 ** exp
    d60 = 60.0 ** exp
    drag = (o60 - o10) / (d60 - d10)
    base = o10 - drag * d10
    if base < 1.0 or drag <= 0.0:
        return None
    return PassingParams(
        power_overshoot_factor=base,
        overshoot_drag_factor=drag,
        overshoot_drag_exponent=exp,
        min_speed_mps=2.0,
        max_speed_mps=25.0,
        pass_aim_height_m=aim_height,
    )


_BP_CACHE = None


def _bp():
    global _BP_CACHE
    if _BP_CACHE is None:
        from footballcoach.engine.ball_physics import BallPhysicsParams
        _BP_CACHE = BallPhysicsParams.from_config()
    return _BP_CACHE


def fmt_speeds(pp: PassingParams) -> str:
    """Launch speed and effective overshoot at 10/25/40/60/75m."""
    bp = _bp()
    parts = []
    for d in [10, 25, 40, 60, 75]:
        spd = pass_speed_mps(pp, d, bp.gravity_mps2, bp.rolling_friction_coefficient)
        ov = pp.power_overshoot_factor + pp.overshoot_drag_factor * (d ** pp.overshoot_drag_exponent)
        parts.append(f"{d}m:{spd:.1f}(x{ov:.2f})")
    return "  ".join(parts)


def _make_player(player_id, team, position, kick_precision=0.5, kick_power=0.7,
                 top_speed=0.7, acceleration=0.7):
    """Thin helper - bypasses make_player from conftest to avoid config loading."""
    from footballcoach.entities.player import Player, PlayerState
    from footballcoach.entities.attributes import PlayerAttributes
    attrs = PlayerAttributes(
        top_speed=top_speed, acceleration=acceleration, stamina=0.7,
        kick_precision=kick_precision, kick_power=kick_power,
        dribbling=0.5, ball_control=0.5, tackling=0.5,
    )
    return Player(
        player_id=player_id, team=team, attributes=attrs,
        position=position, radius_m=0.3, height_m=1.8,
    )


def run_penalty_trial(kp: KickingParams, precision: float,
                      aim_y: float, aim_z: float, seed: int) -> bool:
    pitch = Pitch.standard()
    spot = pitch.penalty_spot(left=False)
    goal_centre = pitch.right_goal_centre
    mvmt = MovementParams.from_config()

    kicker = _make_player("k", Team.LEFT, spot,
                          kick_precision=precision, kick_power=0.7,
                          top_speed=0.7, acceleration=0.7)
    v_run = effective_top_speed(mvmt, kicker.attributes.top_speed, 1.0,
                                has_ball=True,
                                ball_control_attr=kicker.attributes.ball_control)
    kicker.velocity = Vector3(v_run, 0.0, 0.0)

    ball = Ball.at_rest(spot)
    ball.possessed_by = "k"
    match = Match(pitch=pitch, players=[kicker], ball=ball,
                  rng_reduction=RNG_REDUCTION, rng=random.Random(seed),
                  kicking_params=kp)

    aim = goal_centre + Vector3(0, aim_y, aim_z)
    kicker.current_order = KickOrder(aim_point=aim, power_fraction=0.8, spin=Vector3.zero())

    for _ in range(150):
        match.step()
        if match.scoreboard.left_goals > 0:
            return True
        if match.ball.position.x > pitch.half_length + 2.0:
            return False
        if match.ball.velocity.length() < 0.1 and match.ball.position.x < pitch.half_length - 1.0:
            return False
    return False


def run_pass_trial(kp: KickingParams, pp: PassingParams,
                   precision: float, distance: float, seed: int) -> bool:
    from footballcoach.engine.match import Match
    import footballcoach.engine.kicking as kicking_mod

    pitch = Pitch.standard()
    # Centre the pair at midfield so long passes don't cross a goal/touchline.
    origin = Vector3(-distance / 2.0, 0.0, 0.0)
    target = Vector3(distance / 2.0, 0.0, 0.0)
    passer = _make_player("p", Team.LEFT, origin, kick_precision=precision)
    receiver = _make_player("r", Team.LEFT, target)

    # Passer runs at half their top speed toward the receiver (realistic passing scenario)
    mvmt = MovementParams.from_config()
    top_speed = effective_top_speed(mvmt, passer.attributes.top_speed, 1.0,
                                    has_ball=True, ball_control_attr=passer.attributes.ball_control)
    passer.velocity = Vector3(top_speed * 0.5, 0.0, 0.0)

    ball = Ball.at_rest(passer.position)
    ball.possessed_by = "p"

    match = Match(pitch=pitch, players=[passer, receiver], ball=ball,
                  rng_reduction=RNG_REDUCTION, rng=random.Random(seed),
                  kicking_params=kp, passing_params=pp)

    from footballcoach.orders import PassOrder
    passer.current_order = PassOrder(target_position=target)

    # Timeout scales with distance: 30 ticks/s, a 60m ball at ~9m/s takes ~20s to stop
    max_ticks = max(400, int(distance * 20))
    for _ in range(max_ticks):
        match.step()
        if ball.possessed_by == "r":
            return True
        if ball.velocity.length() < 0.05 and ball.possessed_by is None and receiver.state.name != "CONTROLLING_BALL":
            return False
    return False


def score(kp: KickingParams, pp: PassingParams) -> dict:
    pas_10_5  = sum(run_pass_trial(kp, pp, 0.5,  10.0, s) for s in range(N)) / N
    pas_10_15 = sum(run_pass_trial(kp, pp, 0.15, 10.0, s) for s in range(N)) / N
    pas_25_5  = sum(run_pass_trial(kp, pp, 0.5,  25.0, s) for s in range(N)) / N
    pas_25_9  = sum(run_pass_trial(kp, pp, 0.9,  25.0, s) for s in range(N)) / N
    pas_40_5  = sum(run_pass_trial(kp, pp, 0.5,  40.0, s) for s in range(N)) / N
    pas_40_9  = sum(run_pass_trial(kp, pp, 0.9,  40.0, s) for s in range(N)) / N
    pas_50_9  = sum(run_pass_trial(kp, pp, 0.9,  50.0, s) for s in range(N)) / N
    pas_60_9  = sum(run_pass_trial(kp, pp, 0.9,  60.0, s) for s in range(N)) / N
    pas_75_9  = sum(run_pass_trial(kp, pp, 0.9,  75.0, s) for s in range(N)) / N

    return dict(
        pas_10_5=pas_10_5, pas_10_15=pas_10_15,
        pas_25_5=pas_25_5, pas_25_9=pas_25_9,
        pas_40_5=pas_40_5, pas_40_9=pas_40_9,
        pas_50_9=pas_50_9, pas_60_9=pas_60_9,
        pas_75_9=pas_75_9,
    )


def passes(r: dict) -> bool:
    """Pass targets for passing calibration."""
    return (
        r["pas_10_5"]  > 0.80 and
        r["pas_10_15"] > 0.75 and
        r["pas_25_5"]  > 0.50 and
        r["pas_25_9"]  > 0.85 and
        r["pas_40_5"]  > 0.25 and   # long ball with average player: hard but not 0
        r["pas_40_9"]  > 0.65 and   # good player should manage most long balls
        r["pas_50_9"]  > 0.40 and   # 50m: hard but achievable for elite players
        r["pas_60_9"]  > 0.20 and   # 60m: very unreliable, just not 0
        r["pas_75_9"]  > 0.20       # 75m: elite player should still manage ~1 in 5
    )


def fmt(r: dict) -> str:
    return (
        f"pas10_5={r['pas_10_5']:.0%}  "
        f"pas10_15={r['pas_10_15']:.0%}  "
        f"| pas25_5={r['pas_25_5']:.0%}  "
        f"pas25_9={r['pas_25_9']:.0%}  "
        f"| pas40_5={r['pas_40_5']:.0%}  "
        f"pas40_9={r['pas_40_9']:.0%}  "
        f"| pas50_9={r['pas_50_9']:.0%}  "
        f"pas60_9={r['pas_60_9']:.0%}  "
        f"pas75_9={r['pas_75_9']:.0%}"
    )


# --------------------------------------------------------------------------
# Debug: trace a single pass trial tick-by-tick
# --------------------------------------------------------------------------
def debug_pass_trial(kp: KickingParams, pp: PassingParams,
                     precision: float, distance: float, seed: int) -> None:
    from footballcoach.engine.kicking import pass_speed_mps, max_kick_speed_mps
    from footballcoach.engine.ball_physics import BallPhysicsParams

    pitch = Pitch.standard()
    origin = Vector3(-distance / 2.0, 0.0, 0.0)
    target = Vector3(distance / 2.0, 0.0, 0.0)
    passer = _make_player("p", Team.LEFT, origin, kick_precision=precision)
    receiver = _make_player("r", Team.LEFT, target)

    ball = Ball.at_rest(passer.position)
    ball.possessed_by = "p"

    match = Match(pitch=pitch, players=[passer, receiver], ball=ball,
                  rng_reduction=RNG_REDUCTION, rng=random.Random(seed),
                  kicking_params=kp, passing_params=pp)

    # Compute what pass_ball will use
    bp = BallPhysicsParams.from_config()
    auto_spd = pass_speed_mps(pp, distance, bp.gravity_mps2, bp.rolling_friction_coefficient)
    mk = max_kick_speed_mps(kp, passer.attributes.kick_power)
    eff_pf = auto_spd / max(mk, 1e-6)
    from footballcoach.engine.kicking import kick_sigma_rad
    sigma = kick_sigma_rad(kp, precision, eff_pf, RNG_REDUCTION)
    print(f"  auto_speed={auto_spd:.3f}m/s  max_kick={mk:.1f}m/s  eff_pf={eff_pf:.4f}  sigma={sigma:.5f}rad ({math.degrees(sigma):.3f}deg)")
    print(f"  lateral_std_at_{distance}m = {math.tan(sigma)*distance:.3f}m  pickup_radius=0.4m")

    from footballcoach.orders import PassOrder
    passer.current_order = PassOrder(target_position=target)

    launch_logged = False
    prev_vz = 0.0
    prev_vxy = 0.0
    for tick in range(400):
        match.step()
        bx, by, bz = round(ball.position.x, 3), round(ball.position.y, 3), round(ball.position.z, 3)
        vxy = ball.velocity.length_xy()
        vz = ball.velocity.z
        bvx = round(vxy, 3)
        bvz = round(vz, 3)
        rx = round(receiver.position.x, 2)
        state = ball.possessed_by or f"loose(v={bvx})"

        if tick == 0 and not launch_logged:
            launch_logged = True
            total_v = ball.velocity.length()
            pitch_angle = math.degrees(math.atan2(vz, vxy)) if vxy > 0.01 else 0
            print(f"  LAUNCH: vx={round(ball.velocity.x,3)} vy={round(ball.velocity.y,3)} vz={bvz}  |vxy|={bvx}  |v|={round(total_v,3)}  pitch_angle={pitch_angle:.2f}deg  ball_z={bz}")

        # Detect bounce: vz flips from negative to positive
        if prev_vz < -0.05 and vz > 0.0:
            print(f"  BOUNCE tick={tick:3d}  ball=({bx},{by},z={bz})  vz_before={round(prev_vz,3)}->{bvz}  |vxy|_before={round(prev_vxy,3)}->{bvx}")

        # Detect landing (ball hits z~0 with near-zero vz, transitions to rolling)
        if prev_vz < -0.1 and abs(vz) < 0.02 and bz < 0.15:
            print(f"  GROUNDED tick={tick:3d}  ball=({bx},{by},z={bz})  now rolling at |vxy|={bvx}")

        # Detect large sudden speed drop (>30% in one tick) - might indicate something odd
        if prev_vxy > 0.5 and vxy < prev_vxy * 0.7:
            print(f"  SPEED_DROP tick={tick:3d}  |vxy|: {round(prev_vxy,3)} -> {bvx}  ball=({bx},{by},z={bz})")

        if tick < 3 or tick % 30 == 0 or (bvx < 0.15 and bvx > 0):
            print(f"  tick={tick:3d}  ball=({bx},{by},z={bz})  |vxy|={bvx}  vz={bvz}  state={state}  recv={receiver.state.name}")

        prev_vz = vz
        prev_vxy = vxy

        if ball.possessed_by == "r":
            print(f"  => RECEIVED at tick {tick}")
            return
        if ball.velocity.length() < 0.05 and ball.possessed_by is None and receiver.state.name != "CONTROLLING_BALL":
            print(f"  => BALL STOPPED at ({bx},{by},z={bz}), receiver at ({rx},0). lateral_miss={abs(by):.3f}m  short_by={distance-bx:.2f}m")
            return
    print(f"  => TIMEOUT")


# --------------------------------------------------------------------------
# Main grid search
# --------------------------------------------------------------------------
def _default_kp_pp():
    """Current best/default params (mirrors physics.json)."""
    kp = make_kicking_params(ANGLE_BASE[0], ANGLE_SCALE[0], PWR_SCALE[0], PWR_EXP[0], PREC_EXP[0])
    # O10=1.17, O60=1.30, dexp=1.0  →  base=1.144, drag=0.0026
    pp = make_passing_params_from_targets(1.17, 1.30, 1.0)
    return kp, pp


def run_pass_trial_with_velocity(kp, pp, precision, distance, velocity_fraction, seed):
    """Like run_pass_trial but lets the caller set the passer's velocity fraction."""
    pitch = Pitch.standard()
    origin = Vector3(-distance / 2.0, 0.0, 0.0)
    target = Vector3(distance / 2.0, 0.0, 0.0)
    passer = _make_player("p", Team.LEFT, origin, kick_precision=precision)
    receiver = _make_player("r", Team.LEFT, target)
    mvmt = MovementParams.from_config()
    top_speed = effective_top_speed(mvmt, passer.attributes.top_speed, 1.0,
                                    has_ball=True, ball_control_attr=passer.attributes.ball_control)
    passer.velocity = Vector3(top_speed * velocity_fraction, 0.0, 0.0)
    ball = Ball.at_rest(passer.position)
    ball.possessed_by = "p"
    match = Match(pitch=pitch, players=[passer, receiver], ball=ball,
                  rng_reduction=RNG_REDUCTION, rng=random.Random(seed),
                  kicking_params=kp, passing_params=pp)
    from footballcoach.orders import PassOrder
    passer.current_order = PassOrder(target_position=target)
    max_ticks = max(400, int(distance * 20))
    for _ in range(max_ticks):
        match.step()
        if ball.possessed_by == "r":
            return True
        if ball.velocity.length() < 0.05 and ball.possessed_by is None and receiver.state.name != "CONTROLLING_BALL":
            return False
    return False


if __name__ == "__main__":
    import sys
    debug_mode = "--debug" in sys.argv
    running_mode = "--running" in sys.argv

    if debug_mode:
        kp, pp = _default_kp_pp()
        print("=== Default params debug ===")
        print(f"  kp: base={kp.angle_error_base_rad} scale={kp.angle_error_scale_rad}")
        print(f"  pp: {fmt_speeds(pp)}")
        print()
        print("=== 40m pass, precision=0.9, seed=0 ===")
        debug_pass_trial(kp, pp, precision=0.9, distance=40.0, seed=0)
        print()
        print("=== 60m pass, precision=0.9, seed=0 ===")
        debug_pass_trial(kp, pp, precision=0.9, distance=60.0, seed=0)

    elif running_mode:
        # ----------------------------------------------------------
        # Verify run_mult normalisation: pass success rate should be
        # approximately invariant to passer running speed.
        # ----------------------------------------------------------
        kp, pp = _default_kp_pp()
        VEL_FRACTIONS = [0.0, 0.25, 0.5, 0.75, 1.0]
        DISTANCES = [10, 25, 40, 60]
        PRECISIONS = [0.5, 0.9]
        N_RUN = 80

        print(f"Running-speed invariance test  (N={N_RUN} per cell)")
        print(f"Params: {fmt_speeds(pp)}")
        print()

        for precision in PRECISIONS:
            print(f"  precision={precision}")
            header = f"  {'dist':>6}  " + "  ".join(f"v={f:.2f}".rjust(8) for f in VEL_FRACTIONS)
            print(header)
            for dist in DISTANCES:
                rates = []
                for vf in VEL_FRACTIONS:
                    rate = sum(run_pass_trial_with_velocity(kp, pp, precision, dist, vf, s)
                               for s in range(N_RUN)) / N_RUN
                    rates.append(rate)
                row = f"  {dist:>5}m  " + "  ".join(f"{r:>7.0%}" for r in rates)
                spread = max(rates) - min(rates)
                print(f"{row}   spread={spread:.0%}")
            print()

    else:
        total_combos = len(TARGET_O10) * len(TARGET_O60) * len(DRAG_EXP) * len(AIM_HEIGHT)
        print(f"Grid: up to {total_combos} combinations × {N} trials each (invalid base<1.0 skipped)")
        print(f"Targets: pas10_5>80%  pas10_15>75%  pas25_5>50%  pas25_9>85%  "
              f"pas40_5>25%  pas40_9>65%  pas50_9>40%  pas60_9>20%  pas75_9>20%")
        print()

        kp = make_kicking_params(ANGLE_BASE[0], ANGLE_SCALE[0], PWR_SCALE[0], PWR_EXP[0], PREC_EXP[0])
        candidates = []
        run_idx = 0
        for (o10, o60, dexp, ah) in product(TARGET_O10, TARGET_O60, DRAG_EXP, AIM_HEIGHT):
            pp = make_passing_params_from_targets(o10, o60, dexp, ah)
            if pp is None:
                print(f"  SKIP  O10={o10} O60={o60} dexp={dexp}  (base<1.0 or drag<=0)")
                continue
            run_idx += 1
            r = score(kp, pp)
            ok = passes(r)
            marker = " *** CANDIDATE ***" if ok else ""
            speeds_str = fmt_speeds(pp)
            params_str = (f"O10={o10:.3f} O60={o60:.1f} dexp={dexp}  "
                          f"base={pp.power_overshoot_factor:.4f} drag={pp.overshoot_drag_factor:.6f}")
            print(f"[{run_idx:3d}] {params_str}  |  {speeds_str}  |  {fmt(r)}{marker}")
            if ok:
                candidates.append((o10, o60, dexp, ah, pp, r))

        print()
        print(f"=== {len(candidates)} candidates passed all targets ===")
        for (o10, o60, dexp, ah, pp, r) in candidates:
            params_str = (f"O10={o10:.3f} O60={o60:.1f} dexp={dexp}  "
                          f"base={pp.power_overshoot_factor:.4f} drag={pp.overshoot_drag_factor:.6f}")
            print(f"  {params_str}  |  {fmt_speeds(pp)}  |  {fmt(r)}")
