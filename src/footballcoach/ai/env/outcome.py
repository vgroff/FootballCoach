"""Shared trial-end detection for ScenarioLoop (UI) and ScenarioEnv (training).

Both consumers need to answer the same question every tick -- "has this
trial ended, and why?" -- from the same Match state. Keeping the detection
logic in one place means the UI and training code can never silently drift
apart on what counts as a goal/miss/save/dispossession/box-possession, and
outcome labels shown in the UI always match the ones used for training
rewards/termination.

Outcome vocabulary (no catch-all "other" bucket -- every outcome is named
for what actually happened):
    "miss"          - ball left the pitch (out of bounds). ScenarioEnv
                      (phase 1 only) further splits this into "miss" (some
                      player last touched/kicked the ball -- that player is
                      penalised) vs "invalid" (nobody touched it at all this
                      episode -- nobody's fault, no penalty) -- see
                      ScenarioEnv.step(). This module only ever returns the
                      undifferentiated "miss".
    "goal"          - scoreboard changed.
    "saved"         - initial carrier's opponent goalkeeper took the ball.
    "dispossessed"  - initial carrier's opponent outfield player took the ball.
    "received"      - a TEAMMATE of the initial carrier took the ball; only
                      returned when ``teammate_reception_ends_trial`` is set
                      (passing drills). Otherwise a teammate's reception does
                      not end the trial.
    "box_possession" - a player dribbled the ball into the opponent's box.
                      When ``phase1_trainee_player_id`` is given, this
                      specifically means the TRAINEE reached the opponent's
                      box; a non-trainee player doing so is reported as
                      "opponent_box_possession" instead (see
                      detect_phase1_box_terminal()). Without it (every
                      pre-existing caller), this is the undifferentiated
                      "any player, any box" check -- unaware of who's who.
    "opponent_box_possession" - only ever returned when
                      ``phase1_trainee_player_id`` is given: a non-trainee
                      player (with a real controller -- see
                      can_score_box_terminal()) reached the TRAINEE's own
                      box with the ball.
    "course_complete" - a player's AI reports it finished its course (e.g.
                      SprintWaypointAI reaching its final waypoint).
    "timeout"       - trial_tick reached timeout_ticks with no other outcome.
"""
from __future__ import annotations

from footballcoach.engine.match import Match
from footballcoach.entities.player import Team

# Canonical, ordered list of every raw outcome key detect_trial_outcome() can
# return (see the vocabulary docstring above), plus a short display label for
# each. This is the single source of truth for "what outcomes exist" -- UI
# tallies (ScenarioLoop.outcomes / app.py) iterate this instead of hardcoding
# their own key list, so adding a new outcome here is enough to make it show
# up everywhere automatically.
RAW_OUTCOME_KEYS: tuple[str, ...] = (
    "goal",
    "saved",
    "miss",
    "dispossessed",
    "received",
    "box_possession",
    "opponent_box_possession",
    "course_complete",
    "timeout",
)

RAW_OUTCOME_LABELS: dict[str, str] = {
    "goal": "Goals",
    "saved": "Saved",
    "miss": "Miss",
    "dispossessed": "Disp",
    "received": "Recv",
    "box_possession": "Box",
    "opponent_box_possession": "Opp box",
    "course_complete": "Done",
    "timeout": "Timeout",
}


def can_score_box_terminal(player_obj) -> bool:
    """True iff `player_obj` has a real controller and can therefore
    legitimately trigger a box-terminal event (reaching a scoring box with
    possession).

    Single source of truth for this gate -- shared by every phase-1
    box-terminal check (trainee win, trainee loss, and each secondary
    player's own reward bonus). Before this was consolidated, each call
    site independently decided whether to exclude a player with no real
    controller (e.g. an immobile opponent that never chases or holds a
    defensive line), and at least one site missed the gate entirely --
    letting a ball that settled near an immobile player by pure physics
    coincidence re-fire a one-time reward every tick for as long as the
    fluke held (confirmed: 64 consecutive ticks in one real recorded
    episode) instead of once. `player_obj.ai is None` means no controller
    was ever assigned (the immobile-opponent build path) -- a player with
    no controller can never intentionally "reach" anything, so the ball
    settling near them isn't a real terminal event.
    """
    return player_obj.ai is not None


def detect_phase1_box_terminal(match: Match, *, trainee_player_id: str) -> str | None:
    """Returns "box_possession" if the trainee just reached the opponent's
    box with the ball, "opponent_box_possession" if some OTHER player
    (gated via can_score_box_terminal -- excludes players with no real
    controller) just reached the trainee's own box with it, else None.

    This is THE single source of truth for "who won a phase-1 1v1 trial" --
    used by both ScenarioEnv (training/eval) and, via detect_trial_outcome's
    phase1_trainee_player_id param, ScenarioLoop (UI/debug scripts). Before
    this was extracted, ScenarioEnv computed this inline (with the
    can_score_box_terminal gate) while ScenarioLoop/detect_trial_outcome
    used a separate, UNGATED, trainee-unaware "any player in any box" check
    -- confirmed to genuinely disagree on real episodes: a UI/debug-script
    playback could end (and declare the trainee a winner) purely because
    the ball first neared box while the trainee held it, even on a seed
    where eval's own check -- immune to that false trigger -- correctly
    scored the SAME trajectory as a loss once the opponent took the ball
    back moments later.
    """
    ball = match.ball
    carrier_id = ball.possessed_by
    if carrier_id is None:
        return None
    try:
        carrier = match.player_by_id(carrier_id)
        trainee = match.player_by_id(trainee_player_id)
    except KeyError:
        return None

    if carrier_id == trainee_player_id:
        in_opponent_box = match.pitch.is_in_box(
            ball.position, left=(trainee.team == Team.RIGHT),  # opponent's box
        )
        if in_opponent_box and can_score_box_terminal(carrier):
            return "box_possession"
        return None

    # "Trainee's own box" is relative to the TRAINEE's own team, not the
    # carrier's -- do not swap this for carrier.team (they're not
    # necessarily opposite teams once a scenario has more than 2 players).
    in_trainee_box = match.pitch.is_in_box(
        ball.position, left=(trainee.team == Team.LEFT),  # trainee's own box
    )
    if in_trainee_box and can_score_box_terminal(carrier):
        return "opponent_box_possession"
    return None


def detect_trial_outcome(
    match: Match,
    *,
    initial_scoreboard: tuple[int, int],
    initial_carrier_id: str | None,
    ball_released: bool,
    box_possession_terminal: bool,
    trial_tick: int,
    timeout_ticks: int,
    phase1_trainee_player_id: str | None = None,
    teammate_reception_ends_trial: bool = False,
) -> tuple[str | None, bool]:
    """Returns (outcome_key, is_half_linger) if the trial is over, else (None, False).

    ``is_half_linger`` tells the caller whether this outcome should use a
    shorter linger (out-of-bounds events) or the full linger (everything
    else) -- callers decide the actual seconds themselves.

    ``phase1_trainee_player_id``: when given, the box-possession check below
    uses detect_phase1_box_terminal() instead of the generic "any player,
    any box" loop -- returns the correctly trainee/opponent-differentiated,
    can_score_box_terminal()-gated "box_possession"/"opponent_box_possession"
    instead of an undifferentiated "box_possession" regardless of who
    reached which box. ``None`` (default, every pre-existing caller) keeps
    the original generic behaviour unchanged.
    """
    pitch = match.pitch
    ball = match.ball
    scoreboard = match.scoreboard

    # A goal must be checked before "out": Match scores it the tick the ball's
    # centre crosses the goal line inside the mouth, but a hard shot moves
    # ~0.7 m per tick, so the same tick can already put the centre more than a
    # ball radius past the line.
    if (scoreboard.left_goals, scoreboard.right_goals) != initial_scoreboard:
        return "goal", False

    # Laws of the Game, Law 9: the ball is out of play when the WHOLE ball has
    # crossed the WHOLE line (on the ground or in the air), i.e. its centre is
    # more than one ball radius beyond the painted line.
    out_margin_m = ball.radius_m
    if (
        abs(ball.position.x) > pitch.half_length + out_margin_m
        or abs(ball.position.y) > pitch.half_width + out_margin_m
    ):
        match.notify_ball_out()
        return "miss", True

    # The defence winning the ball ends the trial: their goalkeeper = "saved",
    # any other defender = "dispossessed" (tackle, interception or clearance).
    # Team-based, so it holds for any number of players: a TEAMMATE of the
    # initial carrier receiving the ball (a pass, a lay-off) does not end the
    # trial, and a later loss to the defence is still caught. Until 2026-11-16
    # a teammate's reception fell through to "saved", ending a pass move as if
    # the keeper had the ball.
    if ball_released and ball.possessed_by is not None and ball.possessed_by != initial_carrier_id:
        repossessor = match.player_by_id(ball.possessed_by)        # a missing id is a bug: raise
        attacking_team = match.player_by_id(initial_carrier_id).team
        if repossessor.team != attacking_team:
            return ("saved" if repossessor.is_goalkeeper else "dispossessed"), False
        if teammate_reception_ends_trial:
            return "received", False

    # Box possession. Skipped when box_possession_terminal is False (e.g.
    # 1v2, where the attacker is meant to enter the box and play to a
    # natural end).
    if box_possession_terminal:
        if phase1_trainee_player_id is not None:
            phase1_outcome = detect_phase1_box_terminal(
                match, trainee_player_id=phase1_trainee_player_id,
            )
            if phase1_outcome is not None:
                return phase1_outcome, False
        else:
            # Generic, trainee-unaware version: any player dribbled the ball
            # into ITS OWN opponent's box. Team.LEFT attacks +x so their
            # opponent box is the right box (left=False).
            for player in match.players:
                if ball.possessed_by == player.player_id:
                    in_opp_box = pitch.is_in_box(
                        ball.position,
                        left=(player.team == Team.RIGHT),  # opponent box for LEFT; mirrored for RIGHT
                    )
                    if in_opp_box:
                        return "box_possession", False

    # Course completion: any player whose AI exposes course_complete() (e.g.
    # SprintWaypointAI) that reports having finished its waypoint course.
    for player in match.players:
        is_complete = getattr(player.ai, "course_complete", None)
        if is_complete is not None and is_complete(player):
            return "course_complete", False

    if trial_tick >= timeout_ticks:
        return "timeout", False

    return None, False


def remap_phase1_outcome(
    outcome: str,
    *,
    last_ball_toucher_id: str | None,
    box_terminal: bool,
    opponent_box_terminal: bool,
    timeout: bool,
) -> str:
    """Apply phase-1 outcome label transforms shared by ScenarioEnv and ScenarioLoop.

    Call this after detect_trial_outcome returns a non-None outcome, passing
    the same box_terminal / opponent_box_terminal / timeout flags used to
    decide ``done``.  Returns the final label to record.
    """
    if box_terminal:
        return "box_possession"
    if opponent_box_terminal:
        return "opponent_box_possession"
    if timeout:
        return "timeout"
    if outcome == "goal":
        outcome = "miss"
    if outcome == "miss" and last_ball_toucher_id is None:
        return "invalid"
    return outcome


# ---------------------------------------------------------------------------
# Attack-vs-defence phases (phase 2: 1v2, and any later N-vs-M variant)
# ---------------------------------------------------------------------------

# Every outcome classify_attack_defence_outcome() can return, in display order.
# Reward tables (ai_config.json reward.<phase>.terminal[role]) must list each.
ATTACK_DEFENCE_OUTCOME_KEYS: tuple[str, ...] = (
    "goal",
    "saved",
    "defence_possession",
    "shot_off_target",
    "attacker_out",
    "timeout",
)

# The raw detect_trial_outcome() keys that end an attack-vs-defence trial
# (ScenarioLoop.terminal_outcomes); timeout is the env's own.
ATTACK_DEFENCE_TERMINAL_RAW_OUTCOMES: frozenset = frozenset({"goal", "saved", "dispossessed", "miss"})


def classify_attack_defence_outcome(raw_outcome: str, final_match: Match, *, attacking_team: Team) -> str:
    """Map a trial's raw outcome and its final tick to the attack-vs-defence
    vocabulary (ATTACK_DEFENCE_OUTCOME_KEYS). Team-based, not id-based, so it
    holds for any number of attackers and defenders.

    - "goal": the scoreboard changed.
    - "saved" / "defence_possession": a defending goalkeeper / outfield player
      has the ball. A clearance always starts with the defender taking
      possession, so it lands here before the ball can go out.
    - Ball out: if a defender was the last to possess it, the same split as
      above. Otherwise the attacking side put it out: "shot_off_target" when
      the ball is loose and over the goal line being attacked, "attacker_out"
      for anything else (dribbled out, kicked over a touchline).
    - "timeout" is passed through (the env's own clock).

    "Last to possess" is Ball.last_touched_by_player_id, which only possession
    writes; a deflection off a grounded player does not change it, so such a
    ball going out is attributed to whoever last had it.

    Raises ValueError for a raw outcome this vocabulary has no answer for
    (including "received", a drills-only outcome), and for an attacking-team
    player holding the ball at a "saved"/"dispossessed" end, which
    detect_trial_outcome() no longer produces (since 2026-11-16 a teammate's
    reception does not end the trial).
    """
    if raw_outcome in ("goal", "timeout"):
        return raw_outcome
    ball = final_match.ball
    if raw_outcome in ("saved", "dispossessed"):
        holder = final_match.player_by_id_or_none(ball.possessed_by)
        if holder is None or holder.team == attacking_team:
            raise ValueError(
                f"raw outcome {raw_outcome!r} with the ball held by {ball.possessed_by!r}, "
                "not a defending player"
            )
        return "saved" if holder.is_goalkeeper else "defence_possession"
    if raw_outcome == "miss":
        toucher = final_match.player_by_id_or_none(ball.last_touched_by_player_id)
        if toucher is None:
            raise ValueError("ball went out but nobody has possessed it this trial")
        if toucher.team != attacking_team:
            return "saved" if toucher.is_goalkeeper else "defence_possession"
        half_length = final_match.pitch.half_length
        over_attacked_goal_line = (
            ball.position.x > half_length if attacking_team == Team.LEFT else ball.position.x < -half_length
        )
        if over_attacked_goal_line and ball.possessed_by is None:
            return "shot_off_target"
        return "attacker_out"
    raise ValueError(f"no attack/defence outcome for raw outcome {raw_outcome!r}")
