"""Match's tackle resolution: armed (intentional) tackles, the head-on collision fallback, contact checks, goalkeeper tackle immunity, and tackle-result logging."""
from __future__ import annotations

from footballcoach.engine.tackling import apply_tackle_result, attempt_tackle, tackle_angle_modifier
from footballcoach.entities.player import Player, PlayerState, Team



class MatchTacklesMixin:
    """Tackle-resolution methods of Match (see engine/match.py for the tick order that calls them)."""

    def _apply_gk_immune_penalty(self, player: Player) -> None:
        """Apply the auto-fail penalty to a tackler who charged into a GK protected in their own box.
        The GK is untouched; only the tackler is penalised.
        """
        player.velocity = player.velocity * self.tackling_params.tackle_attempt_tackler_speed_mult
        player.state = PlayerState.INACTIVE_TACKLED
        player.state_timer_s = self.tackling_params.tackle_cooldown_s

    def _check_armed_tackles(self) -> None:
        """Resolve tackles for any player who armed tackle_armed this tick
        and is now in contact range.  Fires on_tackle so BC records the event.
        Runs after movement so players that sprinted into range are caught here
        before _check_head_on_tackles (the collision-only fallback) can fire."""
        carrier = self.ball_carrier()
        if carrier is None or carrier.is_inactive:
            return
        # Opportunity accounting FIRST, for every player whether or not it armed a tackle and
        # before any tackle this tick resolves (entities/action_opportunity.py): the tackle
        # bit must not influence whether an opportunity is recorded.
        for player in self.players:
            if self._tackle_contact_possible(player, carrier):
                player.opportunity.note_tack_opp(self.tick_index)
        for player in self.players:
            if not player.tackle_armed:
                continue
            if self._tackle_contact_possible(player, carrier):
                player.tackle_fire_count += 1
                self._attempt_tackle_contact(player, carrier)
                return  # one tackle per tick

    def _tackle_contact_possible(self, player: Player, carrier: Player) -> bool:
        """The armed-tackle resolution predicate: an opposing carrier within auto-tackle overlap
        range of *player*, both available. Shared by resolution and opportunity accounting so
        the two can never disagree."""
        if player.player_id == carrier.player_id:
            return False
        if player.team == carrier.team:
            return False
        # Use the same overlap radius as auto-tackle so tunnelled approaches
        # are caught here; skip can_tackle's are_touching to avoid re-checking
        # the tighter 0.65m threshold that was already passed during movement.
        dist = player.position.xy().distance_to(carrier.position.xy())
        overlap_threshold = self.tackling_params.auto_tackle_overlap_factor * (player.radius_m + carrier.radius_m)
        return dist < overlap_threshold and player.is_available_to_tackle() and carrier.is_available_to_tackle()

    def _check_head_on_tackles(self) -> None:
        """Automatically triggers a tackle when two players from opposite teams
        are in serious overlap (distance < auto_tackle_overlap_factor * combined_radius)
        and have a net closing velocity above auto_tackle_min_closing_mps.

        This replaces the old requirement that BOTH players charge head-on at
        >= 1 m/s, which was too restrictive and never fired when one player was
        slower or approaching from an angle.  The new condition fires whenever
        the players are deeply overlapping and moving toward each other at all.

        Only one such tackle is resolved per tick (the first qualifying pair).
        The resulting tackle uses the same skill-check, inactivity penalties,
        and dribble-speed-penalty model as regular tackles.
        """
        carrier = self.ball_carrier()
        if carrier is None or carrier.is_inactive:
            return

        overlap_factor = self.tackling_params.auto_tackle_overlap_factor
        min_closing = self.tackling_params.auto_tackle_min_closing_mps

        for other in self.players:
            if other.player_id == carrier.player_id:
                continue
            if other.is_inactive or other.team == carrier.team:
                continue

            # Direction from carrier to other player.
            delta = other.position.xy() - carrier.position.xy()
            dist = delta.length()
            if dist < 1e-9:
                continue

            # Trigger only on serious overlap (beyond mere touching).
            min_distance = carrier.radius_m + other.radius_m
            if dist >= overlap_factor * min_distance:
                continue

            direction = delta / dist  # unit vector carrier -> other

            # Net closing speed: sum of each player's velocity component toward
            # the other.  Positive = getting closer together.
            carrier_speed_towards = carrier.velocity.xy().dot(direction)
            other_speed_towards = -(other.velocity.xy().dot(direction))
            closing_speed = carrier_speed_towards + other_speed_towards

            if closing_speed < min_closing:
                continue

            # Head-on: resolve as a tackle (other player is the tackler).
            # Aerial control: if the carrier is mid first-touch on a high ball,
            # the tackle is blocked (nothing to poke away at foot level yet).
            if (
                carrier.state == PlayerState.CONTROLLING_BALL
                and self.ball.position.z > self.control_time_params.control_tackle_immune_height_m
            ):
                self._log_info(
                    f"{other.player_id} head-on tackle on {carrier.player_id} blocked [aerial control]"
                )
                return
            if self._gk_immune_from_tackle(carrier):
                # Phase B: GK in own box with ball is untackleable —
                # the onrushing player is penalised; carrier is untouched.
                other.velocity = other.velocity * self.tackling_params.tackle_attempt_tackler_speed_mult
                other.state = PlayerState.INACTIVE_TACKLED
                other.state_timer_s = self.tackling_params.tackle_cooldown_s
                self._log_info(f"{other.player_id} head-on tackle on {carrier.player_id} auto-failed [GK in own box]")
                if self.match_logger is not None:
                    self.match_logger.notify_tackle_attempt(self.time_s, self.ball.position, other, carrier)
                return
            angle_mod = tackle_angle_modifier(
                carrier.heading_rad, carrier.position, other.position, self.tackling_params
            )
            gk_outside_box = self._gk_outside_own_box(other)  # Phase B: GK penalty
            result = attempt_tackle(
                other.attributes.tackling,
                carrier.attributes.dribbling,
                self.rng_reduction,
                self.rng,
                self.tackling_params,
                is_goalkeeper_tackle=other.is_goalkeeper,
                angle_modifier=angle_mod,
                gk_outside_box=gk_outside_box,
            )
            self._log_tackle_result(
                other.player_id, carrier.player_id, result,
                other.attributes.tackling, carrier.attributes.dribbling,
                other.is_goalkeeper, angle_mod, gk_outside_box, "head-on",
            )
            if result.tackler_won:
                self._set_possession(other.player_id)
            apply_tackle_result(result, other, carrier, self.tackling_params)
            # NOTE: on_tackle/on_tackle_result deliberately do NOT fire here,
            # mirroring on_tackle's existing scope — this is the collision-
            # based auto-tackle fallback, not a recordable AI-intended attempt
            # (see test_tackle.py's auto-tackle-vs-armed-path tests). Firing
            # on_tackle_result here with no matching on_tackle "attempt" would
            # corrupt success/fail stats derived from on_tackle counts.
            # on_auto_tackle_result is the separate, dedicated callback for
            # callers that DO want to see/count these events.
            if other.on_auto_tackle_result is not None:
                other.on_auto_tackle_result(other, result.tackler_won, True)
            if carrier.on_auto_tackle_result is not None:
                carrier.on_auto_tackle_result(carrier, result.tackler_won, False)
            return  # only one head-on tackle per tick

    def _log_tackle_result(
        self,
        tackler_id: str,
        dribbler_id: str,
        result: "TackleResult",  # type: ignore[name-defined]  # noqa: F821
        tackling_attr: float,
        dribbling_attr: float,
        is_goalkeeper_tackle: bool,
        angle_modifier: float,
        gk_outside_box: bool,
        modifier_notes: str = "",
    ) -> None:
        """Emit tackle outcome to log_callback (if set) as ONE INFO entry
        with a hover-detail breakdown of the skill check attached (see
        ui/gamelog.py's LogEntry.detail) -- attribute values, boost/angle
        modifiers, and the actual rolls, replacing what used to be a
        separate DEBUG-only line.

        ``tackling_attr``/``dribbling_attr``/``is_goalkeeper_tackle``/
        ``angle_modifier``/``gk_outside_box`` are the exact values the
        caller passed into ``attempt_tackle()`` for this contest;
        ``result.effective_boost`` is the boost ``attempt_tackle`` already
        derived from them (see ``TackleResult``'s docstring) -- passed
        through rather than recomputed here.
        """
        if self.log_callback is None:
            return
        outcome = "tackled" if result.tackler_won else "failed tackle on"
        msg = f"{tackler_id} {outcome} {dribbler_id}"

        boost_kind = "GK" if is_goalkeeper_tackle else "outfield"
        if is_goalkeeper_tackle and gk_outside_box:
            boost_kind += " (outside-box penalty applied)"
        lines = [
            f"Tackler {tackler_id}: tackling attr={tackling_attr:.2f}, boost={boost_kind},"
            f" angle modifier={angle_modifier:+.2f} -> effective boost={result.effective_boost:.2f}",
            f"  roll = {result.tackler_roll:.3f}  (rng_reduction={self.rng_reduction:.2f})",
            f"Dribbler {dribbler_id}: dribbling attr={dribbling_attr:.2f}",
            f"  roll = {result.dribbler_roll:.3f}",
            f"Winner: {tackler_id if result.tackler_won else dribbler_id}"
            f" (margin {abs(result.tackler_roll - result.dribbler_roll):.3f})",
        ]
        if modifier_notes:
            lines.append(f"Detected via: {modifier_notes}")
        self._log_info(msg, detail="\n".join(lines))

    # ── Phase B helpers ────────────────────────────────────────────────────

    def _defends_left(self, player: Player) -> bool:
        """True if this player's team defends the left goal (Team.LEFT)."""
        return player.team == Team.LEFT

    def _gk_immune_from_tackle(self, gk_candidate: Player) -> bool:
        """True when a goalkeeper is in possession inside their own penalty
        box — in that case all tackle attempts auto-fail (GK is legally
        handling the ball in a protected area)."""
        return (
            gk_candidate.is_goalkeeper
            and self.ball.possessed_by == gk_candidate.player_id
            and self.pitch.is_in_box(
                gk_candidate.position, left=self._defends_left(gk_candidate)
            )
        )

    def _gk_outside_own_box(self, tackler: Player) -> bool:
        """True when the tackler is a goalkeeper making a tackle from outside
        their own penalty box (triggers the -40% tackle boost penalty)."""
        return (
            tackler.is_goalkeeper
            and not self.pitch.is_in_box(
                tackler.position, left=self._defends_left(tackler)
            )
        )

    def _effective_dribbling(self, target: Player) -> float:
        """Returns the target's dribbling attribute, with up to 25% penalty
        if they are mid first-touch control (CONTROLLING_BALL state).

        penalty_frac = min(1, state_timer_s / control_time_penalty_reference_s)
        dribbling_eff = dribbling_attr * (1 - 0.25 * penalty_frac)

        A player who just picked the ball up (near max timer) has the full
        25% penalty; one about to finish control (near 0 s remaining) has
        essentially no penalty.
        """
        if target.state != PlayerState.CONTROLLING_BALL:
            return target.attributes.dribbling
        ref_s = self.tackling_params.control_time_penalty_reference_s
        penalty_frac = min(1.0, target.state_timer_s / ref_s)
        return target.attributes.dribbling * (1.0 - 0.25 * penalty_frac)

    def _target_has_or_controls_ball(self, target: Player) -> bool:
        """True if the target has possession OR is mid first-touch control
        (ball frozen at their feet). Used to decide ball transfer on a
        tackle win against a CONTROLLING_BALL player."""
        return (
            self.ball.possessed_by == target.player_id
            or target.state == PlayerState.CONTROLLING_BALL
        )

    def _attempt_tackle_contact(self, player: Player, target: Player) -> None:
        """Resolve one tackle attempt between *player* (tackler) and *target*
        (ball carrier).  Caller must have already confirmed the two players are
        touching and that ``target.is_available_to_tackle()`` is True.

        Fires ``player.on_tackle`` before physics resolution so BC demonstration
        recording captures the event regardless of which order type triggered
        the tackle (``ChaseTackleOrder`` or ``GetPossessionOrder``).
        """
        _delta = target.position - player.position
        _delta_len = _delta.length()
        player.last_tackle_direction = (_delta * (1.0 / _delta_len)) if _delta_len > 1e-6 else None
        if player.on_tackle is not None:
            player.on_tackle(player)
        if self.match_logger is not None:
            self.match_logger.notify_tackle_attempt(self.time_s, self.ball.position, player, target)
        # A player controlling an aerial ball (above waist height) cannot be tackled —
        # the ball isn't on the ground yet so there's nothing to poke away.
        if (
            target.state == PlayerState.CONTROLLING_BALL
            and self.ball.position.z > self.control_time_params.control_tackle_immune_height_m
        ):
            self._log_info(
                f"{player.player_id} tackle on {target.player_id} blocked [aerial control]"
            )
            return
        if self._gk_immune_from_tackle(target):
            self._apply_gk_immune_penalty(player)
            self._log_info(
                f"{player.player_id} tackle on {target.player_id} auto-failed [GK in own box]"
            )
        else:
            angle_mod = tackle_angle_modifier(
                target.heading_rad, target.position, player.position, self.tackling_params
            )
            gk_outside_box = self._gk_outside_own_box(player)
            effective_dribbling = self._effective_dribbling(target)
            result = attempt_tackle(
                player.attributes.tackling,
                effective_dribbling,
                self.rng_reduction,
                self.rng,
                self.tackling_params,
                is_goalkeeper_tackle=player.is_goalkeeper,
                angle_modifier=angle_mod,
                gk_outside_box=gk_outside_box,
            )
            self._log_tackle_result(
                player.player_id, target.player_id, result,
                player.attributes.tackling, effective_dribbling,
                player.is_goalkeeper, angle_mod, gk_outside_box,
            )
            if result.tackler_won and self._target_has_or_controls_ball(target):
                self._set_possession(player.player_id)
            apply_tackle_result(result, player, target, self.tackling_params)
            if player.on_tackle_result is not None:
                player.on_tackle_result(player, result.tackler_won, True)
            if target.on_tackle_result is not None:
                target.on_tackle_result(target, result.tackler_won, False)
