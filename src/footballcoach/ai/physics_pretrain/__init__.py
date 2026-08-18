"""Self-supervised ball-dynamics pretraining.

See agent_plans/ball_physics_pretrain_plan.md for the full design. This
package is entirely standalone (no dependency on the live PPO/BC policy
networks) -- it produces a frozen encoder checkpoint
(``checkpoints/physics_pretrain/ball_encoder.pt``) that a *future*,
separately-approved change would load into ``DecisionNetwork``/
``ExecutionNetwork`` (see the plan's section 8). Nothing in this package
currently touches ``ai/models/``.
"""
