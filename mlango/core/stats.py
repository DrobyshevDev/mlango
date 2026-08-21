"""The small amount of statistics the framework does on its own behalf.

In `core` because two layers need it and neither may import the other: model
versions ask whether a new fit is better than the old one, and evaluations ask
whether a new prompt is better than the old one. That is the same question about
the same shape of evidence, and the answer should not depend on which half of
the framework is asking.
"""

from __future__ import annotations

import math
from typing import Any

#: Below this, a difference is reported as real rather than as noise. Nothing
#: about 0.05 is principled; it is the number everyone reads without asking, and
#: the p-value is printed beside the verdict so you can disagree with it.
DEFAULT_ALPHA = 0.05


def significance(fixed: int, broke: int, *, alpha: float = DEFAULT_ALPHA) -> dict[str, Any]:
    """Is the difference between two versions distinguishable from noise?

    Cases both versions get right, and cases both get wrong, say nothing about
    which is better — only the disagreements carry information. That leaves
    ``fixed`` cases the new version rescued and ``broke`` cases it lost, and the
    question becomes whether a coin that came up ``fixed`` heads in
    ``fixed + broke`` tosses was fair.

    This is McNemar's test, computed exactly rather than through the chi-square
    approximation, because promotion decisions are often made on a few hundred
    rows where the approximation is worst.

    A version that fixes 200 rows and breaks 3 is an improvement; one that fixes
    38 and breaks 40 is a coin. Both look like "a regression" to a rule that
    counts broken rows, which is why this exists beside that rule rather than
    instead of it.
    """
    discordant = fixed + broke
    if discordant == 0:
        # The two versions are right and wrong on exactly the same rows. There
        # is no evidence either way, and no amount of data would change that.
        return {
            "discordant": 0,
            "p_value": 1.0,
            "direction": "identical",
            "verdict": "the two versions are right on exactly the same rows",
        }

    smaller = min(fixed, broke)
    tail = sum(math.comb(discordant, i) for i in range(smaller + 1)) / (2**discordant)
    p_value = min(1.0, 2 * tail)

    if fixed > broke:
        direction = "improvement"
    elif broke > fixed:
        direction = "regression"
    else:
        direction = "tie"

    return {
        "discordant": discordant,
        "p_value": p_value,
        "direction": direction,
        "verdict": _verdict(direction, p_value, fixed, broke, alpha),
    }


def _verdict(direction: str, p_value: float, fixed: int, broke: int, alpha: float) -> str:
    if direction == "tie":
        return f"{fixed} fixed against {broke} broken is a coin, not a change"
    if p_value >= alpha:
        return (
            f"{fixed} fixed against {broke} broken is not distinguishable from noise "
            f"(p={p_value:.3f})"
        )
    if direction == "improvement":
        return f"a real improvement: {fixed} fixed against {broke} broken (p={p_value:.3f})"
    return f"a real regression: {broke} broken against {fixed} fixed (p={p_value:.3f})"


__all__ = ["significance", "DEFAULT_ALPHA"]
