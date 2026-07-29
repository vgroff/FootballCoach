"""Rollout buffer and GAE(lambda) advantage estimation for PPO.

Stores transitions collected during environment interaction, then computes
advantages and returns using Generalized Advantage Estimation (GAE).

See ai_design_doc.md sections 9.1 and 9.3 for design details.

IMPORTANT: GAE correctness depends on correct `dones` alignment:
  - dones[t] = 1.0 means the episode ENDED at step t (terminal state).
  - next_value used in delta must be the value for state t+1, not state t.
  See the unit test in tests/ai_unit/test_gae.py for a hand-computed example
  that validates this implementation against known correct outputs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional  # noqa: F401 - used in add() signature

import numpy as np
import torch


@dataclass
class RolloutBuffer:
    """Stores transitions for one PPO rollout (N decision steps).

    All numpy arrays are populated step-by-step via ``add()``, then
    ``compute_gae()`` is called once before passing to the PPO trainer.

    Observation tensors are stored as numpy arrays (dict keyed by obs key).
    Actions are stored as flat numpy arrays (the sampled raw values from the
    policy distributions, before squashing, as required for log_prob).
    """

    # Raw observations at each step, keyed by obs tensor name
    # e.g. {"self_feat": ..., "other_feat": ..., "exists_mask": ..., ...}
    obs: list[dict[str, np.ndarray]] = field(default_factory=list)

    # Raw actions (pre-squash) - stored as flat numpy arrays; layout matches
    # what combined_log_prob expects.  Includes decision and execution actions.
    actions: list[dict[str, np.ndarray]] = field(default_factory=list)

    # log_prob of the action taken at each step under the policy that took it
    # (old policy; fixed during the PPO epochs over this rollout)
    log_probs: list[float] = field(default_factory=list)

    # Critic value estimate at each step
    values: list[float] = field(default_factory=list)

    # Scalar reward this step
    rewards: list[float] = field(default_factory=list)

    # 1.0 if this step ended the episode, else 0.0
    dones: list[float] = field(default_factory=list)

    # Optional BC supervision labels (flat float32 arrays of length BC_LABEL_DIM).
    # Each entry is the output of BCLabel.to_array() for the corresponding step.
    # An all-zeros array with valid=0 is used as a placeholder when no label fn
    # is provided (bc_labels[t][_I_VALID] == 0.0 → skip BC loss for this step).
    bc_labels: list[np.ndarray] = field(default_factory=list)

    def add(
        self,
        obs: dict[str, np.ndarray],
        action: dict[str, np.ndarray],
        log_prob: float,
        value: float,
        reward: float,
        done: float,
        bc_label: Optional[np.ndarray] = None,
    ) -> None:
        from footballcoach.ai.ppo.bc import BC_LABEL_DIM
        self.obs.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
        self.bc_labels.append(
            bc_label if bc_label is not None
            else np.zeros(BC_LABEL_DIM, dtype=np.float32)
        )

    def __len__(self) -> int:
        return len(self.rewards)

    def compute_gae(
        self,
        gamma: float,
        lam: float,
        last_value: float,
    ) -> tuple[list[float], list[float]]:
        """Compute GAE(lambda) advantages and value targets (returns).

        Args:
            gamma: Discount factor.
            lam: GAE lambda (trade-off between bias and variance).
            last_value: Critic's value estimate for the state AFTER the last
                stored step (the bootstrap value for the final step; should
                be 0.0 if the episode ended exactly at the last stored step,
                otherwise the next-state value estimate).

        Returns:
            (advantages, returns) - both lists of floats, same length as
            the buffer.  ``returns[t] = advantages[t] + values[t]``.

        Common silent bug: off-by-one on next_value / dones.  The ``delta``
        at step t uses the value at t+1 (after transitioning), not t.
        ``next_non_terminal`` is 0 when dones[t]=1 so no bootstrapping
        happens across episode boundaries.  See design doc section 9.3.
        """
        n = len(self.rewards)
        advantages = [0.0] * n
        last_gae = 0.0
        # Extend values list with last_value for the n+1 bootstrap
        all_values = self.values + [last_value]

        for t in reversed(range(n)):
            next_value = all_values[t + 1]
            next_non_terminal = 1.0 - self.dones[t]
            delta = (
                self.rewards[t]
                + gamma * next_value * next_non_terminal
                - self.values[t]
            )
            last_gae = delta + gamma * lam * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = [adv + val for adv, val in zip(advantages, self.values)]
        return advantages, returns

    def as_tensors(
        self, advantages: list[float], returns: list[float]
    ) -> dict[str, torch.Tensor]:
        """Pack buffer contents + GAE outputs into a flat dict of tensors.

        Returns a dict with keys:
          obs/{key}    - stacked observation tensors (N, ...)
          action/{key} - stacked action arrays (N, ...)
          log_probs    - (N,)
          values       - (N,)
          rewards      - (N,)
          dones        - (N,)
          advantages   - (N,)
          returns      - (N,)
        """
        result: dict[str, torch.Tensor] = {}

        # Stack observations
        for key in self.obs[0]:
            result[f"obs/{key}"] = torch.from_numpy(
                np.stack([o[key] for o in self.obs], axis=0)
            )

        # Stack actions
        for key in self.actions[0]:
            arr = np.stack([a[key] for a in self.actions], axis=0)
            result[f"action/{key}"] = torch.from_numpy(arr.astype(np.float32))

        result["log_probs"] = torch.tensor(self.log_probs, dtype=torch.float32)
        result["values"] = torch.tensor(self.values, dtype=torch.float32)
        result["rewards"] = torch.tensor(self.rewards, dtype=torch.float32)
        result["dones"] = torch.tensor(self.dones, dtype=torch.float32)
        result["advantages"] = torch.tensor(advantages, dtype=torch.float32)
        result["returns"] = torch.tensor(returns, dtype=torch.float32)

        if self.bc_labels:
            result["bc_labels"] = torch.from_numpy(
                np.stack(self.bc_labels, axis=0).astype(np.float32)
            )

        return result

    def clear(self) -> None:
        self.obs.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()
        self.bc_labels.clear()
