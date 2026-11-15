"""Named numeric tolerances for the simulation core (engine/, entities/, orders.py,
steering.py).

GEOM_ZERO_EPS is the tolerance for "this length, distance, speed or duration is
numerically zero": the guard before normalizing a vector or dividing by a
magnitude. Use it for any new guard of that kind.

Older literals are deliberately left as they are, because their values differ
and unifying them would change behaviour: several such guards use 1e-6
instead, Vector3.normalized() uses 1e-12 on a length, and a few sites use
1e-12 (or, in engine/interception.py, 1e-9) on a squared length. Ball.is_grounded()'s 1e-6 m is a physical contact
tolerance, not a numeric guard. The AI code (ai/) has its own float32-scale
tolerances, which are unrelated.
"""

GEOM_ZERO_EPS: float = 1e-9
