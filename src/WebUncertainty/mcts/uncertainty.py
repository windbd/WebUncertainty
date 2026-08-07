"""ConActU: Confidence-induced Action Uncertainty (paper Section 3.2).

Turns the Reasoning Agent's raw per-candidate confidence scores into two
uncertainty scalars, and turns those into a reward-modulation decision when
the Evaluation Agent's base score misses the acceptance threshold.

    p_i     = c_i / sum(c)                          (Eq. 3, normalized)
    E       = mean(c)                                (Eq. 3, "total evidence proxy")
    H_norm  = -1/log(K) * sum(p_i * log(p_i))         (Eq. 3, normalized entropy)
    EU      = 1 - E                                   (Eq. 4)
    AU      = H_norm * E                               (Eq. 5)

EU is high when the model just isn't confident about anything here
(epistemic — it doesn't know this page). AU is high only when the model
*is* confident overall but that confidence is spread across several
candidates (aleatoric — several options look equally valid). A single
scalar can't tell those apart; that is the whole reason ConActU keeps two.

The paper pins two thresholds from a sensitivity sweep: the planning
switch ``delta = 0.4`` and the evaluation acceptance bar ``tau = 6`` (of
10). It does not give numeric values for what counts as "high" vs. "low"
EU/AU in the reward-modulation table (Section 3.2, "Simulation"); both
naturally live in [0, 1], so ``eu_threshold``/``au_threshold`` default to
0.5 here as a documented, overridable choice — not a value taken from the
paper.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Sequence

# Reward constants from the paper's four-case table (Section 3.2, "Simulation").
STRICT_PENALTY_REWARD = -5.0
RELAXED_PENALTY_REWARD = -1.0
REGENERATE_REWARD = 0.0


class UncertaintyError(ValueError):
    """Raised for malformed confidence scores (empty, negative, non-finite)."""


@dataclass(frozen=True)
class ConActU:
    """Epistemic (EU) and aleatoric (AU) uncertainty for one candidate set."""

    eu: float
    au: float
    mean_confidence: float
    normalized_entropy: float


class RewardCase(Enum):
    """Which cell of the EU/AU reward-modulation table applied."""

    ACCEPTED = "accepted"  # S_base >= tau; EU/AU not consulted.
    STRICT_PENALTY = "strict_penalty"  # High EU, high AU.
    RELAXED_PENALTY = "relaxed_penalty"  # High EU, low AU.
    NORMAL = "normal"  # Low EU, high AU.
    REGENERATE = "regenerate"  # Low EU, low AU.


@dataclass(frozen=True)
class ModulatedReward:
    """The backpropagated reward plus which case produced it, for logging."""

    reward: float
    case: RewardCase


def normalize_confidences(confidences: Sequence[float]) -> list[float]:
    """Normalize raw confidence scores into a probability-like distribution.

    Falls back to a uniform distribution when every score is (numerically)
    zero, matching "no information at all" rather than raising.
    """
    values = _validate_confidences(confidences)
    total = sum(values)
    if total <= 0.0:
        return [1.0 / len(values)] * len(values)
    return [value / total for value in values]


def compute_conactu(confidences: Sequence[float]) -> ConActU:
    """Compute (EU, AU) from a leaf's raw candidate confidence scores."""
    values = _validate_confidences(confidences)
    k = len(values)
    mean_confidence = sum(values) / k

    if k == 1:
        # A set of one candidate carries no choice among alternatives.
        normalized_entropy = 0.0
    else:
        probabilities = normalize_confidences(values)
        raw_entropy = -sum(p * math.log(p) for p in probabilities if p > 0.0)
        normalized_entropy = raw_entropy / math.log(k)

    eu = 1.0 - mean_confidence
    au = normalized_entropy * mean_confidence
    return ConActU(
        eu=eu,
        au=au,
        mean_confidence=mean_confidence,
        normalized_entropy=normalized_entropy,
    )


def classify(
    eu: float,
    au: float,
    *,
    eu_threshold: float = 0.5,
    au_threshold: float = 0.5,
) -> RewardCase:
    """Map an (EU, AU) pair to one of the four below-threshold cases.

    Only meaningful when the base score already missed ``tau`` — call
    :func:`modulate_reward` rather than this directly in search code.
    """
    high_eu = eu >= eu_threshold
    high_au = au >= au_threshold
    if high_eu and high_au:
        return RewardCase.STRICT_PENALTY
    if high_eu and not high_au:
        return RewardCase.RELAXED_PENALTY
    if not high_eu and high_au:
        return RewardCase.NORMAL
    return RewardCase.REGENERATE


def modulate_reward(
    base_score: float,
    *,
    tau: float,
    eu: float,
    au: float,
    eu_threshold: float = 0.5,
    au_threshold: float = 0.5,
) -> ModulatedReward:
    """Apply the paper's acceptance-then-modulation rule to one evaluation.

    ``base_score`` is the Evaluation Agent's ``S_base`` (0-10 scale, same
    scale as ``tau``). ``eu``/``au`` are the ConActU scores for the
    candidate action that produced ``base_score`` — they describe the
    *proposal*, not the outcome, which is exactly why they still carry
    information when the outcome turns out to be a failure.
    """
    if base_score >= tau:
        return ModulatedReward(reward=float(base_score), case=RewardCase.ACCEPTED)

    case = classify(eu, au, eu_threshold=eu_threshold, au_threshold=au_threshold)
    if case is RewardCase.STRICT_PENALTY:
        reward = STRICT_PENALTY_REWARD
    elif case is RewardCase.RELAXED_PENALTY:
        reward = RELAXED_PENALTY_REWARD
    elif case is RewardCase.NORMAL:
        reward = float(base_score)
    else:
        reward = REGENERATE_REWARD
    return ModulatedReward(reward=reward, case=case)


def _validate_confidences(confidences: Sequence[float]) -> list[float]:
    values = [float(value) for value in confidences]
    if not values:
        raise UncertaintyError("confidences must not be empty")
    for value in values:
        if not math.isfinite(value):
            raise UncertaintyError(f"confidence must be finite, got {value!r}")
        if value < 0.0:
            raise UncertaintyError(f"confidence must be non-negative, got {value!r}")
    return values
