"""mathutils/tolerances.py: GEOM_ZERO_EPS is the named zero-length/zero-speed guard for
the simulation core. Its value is pinned (changing it changes simulation behaviour),
and bare 1e-9 literals are not allowed back into the files that use it."""
import io
import re
import tokenize
from pathlib import Path

from footballcoach.mathutils.tolerances import GEOM_ZERO_EPS

_SRC = Path(__file__).resolve().parents[2] / "src" / "footballcoach"
_CORE = sorted([*(_SRC / "engine").glob("*.py"), *(_SRC / "entities").glob("*.py"),
                _SRC / "orders.py", _SRC / "steering.py"])
# The one deliberate exception: a squared-length guard (see tolerances.py).
_ALLOWED = {("engine/interception.py", "if d_sq < 1e-9:")}


def test_value_is_pinned():
    assert GEOM_ZERO_EPS == 1e-9


def test_no_bare_1e_9_literals_in_the_simulation_core():
    offenders = []
    for path in _CORE:
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type == tokenize.NUMBER and re.fullmatch(r"1e-0*9", tok.string, re.I):
                rel = path.relative_to(_SRC).as_posix()
                line = lines[tok.start[0] - 1].strip()
                if not any(rel == f and line.startswith(code) for f, code in _ALLOWED):
                    offenders.append(f"{rel}:{tok.start[0]}: {line}")
    assert not offenders, "use GEOM_ZERO_EPS instead:\n" + "\n".join(offenders)
