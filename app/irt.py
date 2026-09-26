"""IRT math for the adaptive engine.

The item bank uses the two-parameter logistic model (2PL).

For an examinee with ability ``theta`` and an item with discrimination ``a``
and difficulty ``b``, define the logit::

    z = a * (theta - b)

The probability of a correct response and its complement are::

    P(theta) = 1 / (1 + exp(-z))
    Q(theta) = 1 - P  = 1 / (1 + exp(z))

The Fisher information of that item at ``theta`` is::

    I(theta) = a**2 * P(theta) * Q(theta)

Ability is estimated by *one* Newton-Raphson step on the log-likelihood of a
single binary response ``u`` (1 = correct, 0 = wrong).  For one observation::

    d1/dtheta = a * (u - P)     #  a*Q when correct, -a*P when wrong
    d2/dtheta = -a**2 * P * Q = -I(theta)

so the Newton update is::

    theta' = theta - d1/d2 = theta + a * (u - P) / I(theta)

Over a session the standard error of the ability estimate is approximated
from the summed Fisher information of every answered item at the current
theta::

    SE(theta) = 1 / sqrt(sum_j I_j(theta))

When that sum is zero the standard error is undefined (``None``).

Numerical stability.  ``P`` and ``Q`` are each evaluated through the
*small* exponential of their branch (``P = exp(z)/(1+exp(z))`` for
``z < 0``, ``Q = exp(-z)/(1+exp(-z))`` for ``z >= 0``), never as
``1 - <rounded probability>``.  Consequently, when ``P`` rounds to exactly
``1.0`` far out in the right tail, ``Q`` is still a strictly positive float
and ``a**2*P*Q`` does not vanish.  ``z`` is additionally clamped to
``[-60, 60]`` so the exponential always stays finite (with the bank's
legal ranges the extreme logit is exactly 60); the information is positive
and symmetric, ``I(z) == I(-z)``, at both ends.
"""

from __future__ import annotations

import math

THETA_MIN = -6.0
THETA_MAX = 6.0

# Keep exp() arguments finite on every reachable input; with legal theta in
# [-6, 6], a in (0, 5] and b in [-6, 6] the extreme logit is exactly 60.
_Z_LIMIT = 60.0

# Information is tiny when prediction is nearly certain; floor the Newton
# denominator so a single extreme response cannot send theta to infinity.
_DENOM_FLOOR = 1e-4


def clamp_theta(theta: float) -> float:
    """Keep ability estimates inside a numerically sane range."""
    return min(max(float(theta), THETA_MIN), THETA_MAX)


def _logistic_pair(theta: float, a: float, b: float) -> tuple[float, float]:
    """Return ``(P, Q)`` for the 2PL model, stable in both tails.

    ``z = a * (theta - b)`` clamped to ``[-60, 60]``.  The complement on the
    certain side is computed from the small exponential directly, so a
    probability rounded to ``1.0`` never drags ``Q`` -- and hence the Fisher
    information -- to zero.
    """
    z = a * (theta - b)
    if z > _Z_LIMIT:
        z = _Z_LIMIT
    elif z < -_Z_LIMIT:
        z = -_Z_LIMIT
    if z >= 0.0:
        small = math.exp(-z)
        p = 1.0 / (1.0 + small)
        q = small / (1.0 + small)
    else:
        small = math.exp(z)
        p = small / (1.0 + small)
        q = 1.0 / (1.0 + small)
    return p, q


def probability(theta: float, a: float, b: float) -> float:
    """2PL probability of a correct answer ``P = 1/(1+exp(-a(theta-b)))``."""
    return _logistic_pair(theta, a, b)[0]


def information(theta: float, a: float, b: float) -> float:
    """Fisher information of item (a, b) at ability theta: ``a**2 * P * Q``."""
    p, q = _logistic_pair(theta, a, b)
    return (a * a) * p * q


def standard_error(total_info: float) -> float | None:
    """Ability estimate standard error ``1 / sqrt(sum I)``.

    Returns ``None`` when the accumulated information is zero (or not
    positive), in which case the precision stop rule cannot fire.
    """
    if total_info <= 0.0:
        return None
    return 1.0 / math.sqrt(total_info)


def newton_step(
    theta: float, a: float, b: float, score: int
) -> tuple[float, float, float, float]:
    """Apply one Newton-Raphson update for one binary response.

    The gradient is ``a*Q`` for a correct response and ``-a*P`` for a wrong
    one; the denominator keeps the ``_DENOM_FLOOR`` lower bound and the
    resulting ability is clamped to ``[THETA_MIN, THETA_MAX]``.

    Returns ``(new_theta, p_correct, gradient, information)``.
    """
    p, q = _logistic_pair(theta, a, b)
    if score:
        gradient = a * q
    else:
        gradient = -a * p
    info = (a * a) * p * q
    step = gradient / max(info, _DENOM_FLOOR)
    return clamp_theta(theta + step), p, gradient, info
