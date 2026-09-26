"""IRT math for the adaptive engine.

The item bank uses the two-parameter logistic model (2PL).

For an examinee with ability ``theta`` and an item with discrimination ``a``
and difficulty ``b``, the linear predictor is

    z = a * (theta - b)

and the probability of a correct response is::

    P(theta) = 1 / (1 + exp(-z))      = sigmoid(z)
    Q(theta) = 1 - P = P(wrong)       = sigmoid(-z)

The Fisher information of that item at ``theta`` is::

    I(theta) = a**2 * P(theta) * Q(theta)

Ability is estimated by *one* Newton-Raphson step on the log-likelihood of a
single binary response ``u`` (1 = correct, 0 = wrong).  For one observation::

    d1/dtheta = a * (u - P)
    d2/dtheta = -a**2 * P * Q = -I(theta)

so the Newton update is::

    theta' = theta - d1/d2 = theta + a * (u - P) / I(theta)

Writing the gradient by outcome,

    u = 1 (correct): d1/dtheta = a * (1 - P) = a * Q
    u = 0 (wrong):   d1/dtheta = a * (0 - P) = -a * P

keeps the sign numerically safe even when one of P/Q rounds to exactly 0 or 1.

Over a session the standard error of the ability estimate is approximated
from the summed Fisher information of every answered item at the current
theta::

    SE(theta) = 1 / sqrt(sum_j I_j(theta))

When that sum is zero the standard error is undefined (``None``).

Numerical safety
----------------

All three kernels share one helper (:func:`_p_and_q`) and one convention:
the magnitude of the logistic argument is capped at :data:`_Z_LIMIT` (60)
before ``exp`` is ever called, so no input from the legal parameter ranges
can overflow, while each complementary probability is computed from the
*short* side of the logistic::

    z >= 0:  P = 1 / (1 + exp(-z)),  Q = exp(-z) / (1 + exp(-z))
    z <  0:  P = exp(z) / (1 + exp(z)), Q = 1 / (1 + exp(z))

The product ``P * Q`` is evaluated through :func:`_product_pq` as
``e^-|z| / (1 + e^-|z|)^2`` -- an even function of z -- rather than by
multiplying ``P`` and ``(1 - P)`` after P has rounded to 1.0.  Consequently
Fisher information stays strictly positive and is exactly symmetric at
``z = +60`` and ``z = -60`` even though ``probability`` itself rounds to
exactly ``1.0`` (positive tail) or to the sub-1e-26 small tail (negative
tail) there.
"""

from __future__ import annotations

import math

THETA_MIN = -6.0
THETA_MAX = 6.0

# Legal a in (0, 5] and theta/b in [-6, 6] give |z| <= 60; cap the logistic
# argument there so exp can never overflow while information stays positive
# and symmetric for the most extreme legal inputs.
_Z_LIMIT = 60.0

# Information is tiny when prediction is nearly certain; floor the Newton
# denominator so a single extreme response cannot send theta to infinity.
_DENOM_FLOOR = 1e-4


def clamp_theta(theta: float) -> float:
    """Keep ability estimates inside a numerically sane range."""
    return min(max(float(theta), THETA_MIN), THETA_MAX)


def _p_and_q(theta: float, a: float, b: float) -> tuple[float, float]:
    """Return ``(P, Q)`` for the 2PL model, stable in both tails.

    ``z = a * (theta - b)``.  Each complementary probability is taken from
    the branch that stays inside (0, 1): the small-tail value is built as
    ``exp(-|z|) / (1 + exp(-|z|))`` instead of subtracting a rounded 1.0.
    """
    z = max(min(a * (theta - b), _Z_LIMIT), -_Z_LIMIT)
    if z >= 0.0:
        e = math.exp(-z)
        p = 1.0 / (1.0 + e)
        q = e / (1.0 + e)
    else:
        e = math.exp(z)
        p = e / (1.0 + e)
        q = 1.0 / (1.0 + e)
    return p, q


def _product_pq(theta: float, a: float, b: float) -> float:
    """Stable ``P * Q``: even in z and positive even at ``|z| = 60``.

    With ``t = exp(-|z|)`` this is ``t / (1 + t) ** 2``, which never
    subtracts rounded ones and therefore cannot collapse to zero when
    :func:`probability` itself rounds to exactly 0 or 1.
    """
    z = abs(a * (theta - b))
    if z >= _Z_LIMIT:
        z = _Z_LIMIT
    t = math.exp(-z)
    return t / ((1.0 + t) * (1.0 + t))


def probability(theta: float, a: float, b: float) -> float:
    """2PL probability of a correct answer ``1 / (1 + exp(-a(theta - b)))``.

    Overflow-safe; rounds to exactly 1.0 in the far positive tail and
    keeps the true small-tail value (never an exact zero) in the far
    negative tail.
    """
    p, _q = _p_and_q(theta, a, b)
    return p


def information(theta: float, a: float, b: float) -> float:
    """Fisher information ``a**2 * P * Q`` of item (a, b) at theta.

    The ``P * Q`` factor comes from :func:`_product_pq`, so the value stays
    strictly positive and symmetric in the two tails instead of vanishing
    when ``probability`` rounds to 1.0.
    """
    return (a * a) * _product_pq(theta, a, b)


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

    Returns ``(new_theta, p_correct, gradient, information)``.

    The gradient uses the outcome form -- ``a * Q`` for a correct answer
    and ``-a * P`` for a wrong one -- so its magnitude never depends on a
    rounded 1.0 minus P; the Newton denominator still has its
    ``_DENOM_FLOOR`` floor and theta is still clamped to
    ``[THETA_MIN, THETA_MAX]``.
    """
    p, q = _p_and_q(theta, a, b)
    info = (a * a) * _product_pq(theta, a, b)
    if score:
        gradient = a * q
    else:
        gradient = -a * p
    step = gradient / max(info, _DENOM_FLOOR)
    return clamp_theta(theta + step), p, gradient, info
