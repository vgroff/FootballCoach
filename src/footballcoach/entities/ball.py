"""The ball entity: position, velocity, spin, and possession state."""
from __future__ import annotations

from dataclasses import dataclass

from footballcoach.config import load_physics_config
from footballcoach.mathutils import Vector3


@dataclass
class Ball:
    position: Vector3 = None  # type: ignore[assignment]
    velocity: Vector3 = None  # type: ignore[assignment]
    spin: Vector3 = None  # type: ignore[assignment]
    radius_m: float = 0.11
    mass_kg: float = 0.43

    # Player id currently in possession (ball "stuck" to them), or None if loose.
    possessed_by: str | None = None

    # Player id of whoever most recently GAINED possession, kept even after
    # the ball goes loose again. None only until the very first possession
    # of the episode. Lets loose-ball behaviour (e.g. Match._run_get_
    # possession_behaviour's boundary-braking) and the observation encoder
    # (BallFeatures.last_touch_team_direction) ask "who/which team touched
    # this ball last" without needing a separate possession-history log.
    #
    # Two write-paths, both required to keep this authoritative:
    #   - Match._set_possession() -- every possession change during live
    #     play (a real gain via pickup/tackle).
    #   - set_initial_possession() below -- a scenario builder pre-assigning
    #     the ball to a player's feet at construction time, before any
    #     Match exists to call _set_possession() through. A scenario that
    #     wrote ball.possessed_by directly instead (bypassing both) used to
    #     leave this field None despite the ball genuinely starting held --
    #     a real, confirmed divergence between this field and ScenarioLoop's
    #     own (now-removed) separate toucher-tracking copy. Every scenario
    #     builder in ui/scenarios.py that starts a trial with the ball
    #     already possessed goes through set_initial_possession() for
    #     exactly this reason -- never assign possessed_by directly.
    last_touched_by_player_id: str | None = None

    # Countdown timer set to just_bounced_display_duration_s whenever the
    # ball makes a genuine bounce (incoming vz exceeds BOUNCE_THRESHOLD_MPS).
    # Decayed by dt each tick in ball_physics.step_ball. Used by the renderer
    # to show a visual "just bounced" indicator. Zero when not recently bounced.
    just_bounced_timer_s: float = 0.0

    def __post_init__(self) -> None:
        if self.position is None:
            self.position = Vector3.zero()
        if self.velocity is None:
            self.velocity = Vector3.zero()
        if self.spin is None:
            self.spin = Vector3.zero()

    @staticmethod
    def at_rest(position: Vector3 | None = None) -> "Ball":
        cfg = load_physics_config()["ball"]
        return Ball(
            position=position or Vector3.zero(),
            velocity=Vector3.zero(),
            spin=Vector3.zero(),
            radius_m=cfg["radius_m"],
            mass_kg=cfg["mass_kg"],
        )

    def set_initial_possession(self, player_id: str) -> None:
        """Assign possession at scenario-BUILD time (before any Match.step()
        has run, so Match._set_possession's on_possession_gained-callback/
        match_logger machinery don't apply yet -- there's no Match to own
        them). Sets possessed_by AND last_touched_by_player_id together, so
        the two can never diverge the way a direct `ball.possessed_by = ...`
        assignment used to allow -- see last_touched_by_player_id's own
        docstring."""
        self.possessed_by = player_id
        self.last_touched_by_player_id = player_id

    @property
    def is_loose(self) -> bool:
        return self.possessed_by is None

    @property
    def height_m(self) -> float:
        return self.position.z

    def is_grounded(self, epsilon: float = 1e-6) -> bool:
        return self.position.z <= self.radius_m + epsilon
