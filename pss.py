# FossielPythonSimple
# Optional any-type inputs A–J + script; unconnected inputs default to 0.
# Outputs resultA–resultJ (unset in script → 0). Accepts only int/float (bool rejected).
# Sandbox: math + basic builtins, no imports.
# No node registration (handle in __init__.py).

import math
import ast


class _AnyType(str):
    """ComfyUI wildcard type so any link can connect."""

    def __ne__(self, other):
        return False


any_type = _AnyType("*")

_SAFE_BUILTINS = {
    "abs": abs,
    "min": min,
    "max": max,
    "round": round,
    "pow": pow,
    "sum": sum,
    "len": len,
    "int": int,
    "float": float,
    "bool": bool,
    "True": True,
    "False": False,
    "None": None,
}

_SAFE_GLOBALS_BASE = {
    "__builtins__": _SAFE_BUILTINS,
    "math": math,
}

_RESULT_NAMES = tuple(f"result{ch}" for ch in "ABCDEFGHIJ")


def _run_script(script: str, variables: dict) -> dict:
    if not script or not str(script).strip():
        raise ValueError(
            "Script is empty. Assign at least one of resultA–resultJ, "
            "e.g. resultA = A + B"
        )

    try:
        tree = ast.parse(str(script), mode="exec")
    except SyntaxError as e:
        raise ValueError(f"Script syntax error: {e}") from e

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError("Imports are not allowed in this sandbox.")
        if isinstance(node, ast.Attribute) and isinstance(node.attr, str) and node.attr.startswith("__"):
            raise ValueError("Access to dunder attributes is not allowed.")

    env = dict(_SAFE_GLOBALS_BASE)
    env.update(variables)

    try:
        exec(compile(tree, "<FossielPythonSimple>", "exec"), env, env)
    except Exception as e:
        raise RuntimeError(f"Script error: {type(e).__name__}: {e}") from e

    if not any(name in env for name in _RESULT_NAMES):
        raise ValueError(
            "Script must assign at least one of resultA–resultJ, "
            "e.g. resultA = (A + B) / 2"
        )

    return env


def _coerce_number(name, value):
    """Unconnected/None -> 0. Only plain int or float allowed otherwise."""
    if value is None:
        return 0

    if type(value) is bool:
        raise ValueError(
            f"Input {name} received a boolean ({value!r}). "
            f"Only integers or floats are allowed (unconnected inputs default to 0)."
        )

    if type(value) is int or type(value) is float:
        return value

    raise ValueError(
        f"Input {name} received type {type(value).__name__} ({value!r}). "
        f"Only integers or floats are allowed (unconnected inputs default to 0)."
    )


def _coerce_result(name, value):
    """Missing result* -> 0. Must be int or float if present."""
    if value is None:
        return 0

    if type(value) is bool:
        raise ValueError(f"`{name}` must be an int or float, got boolean ({value!r}).")

    if type(value) is int or type(value) is float:
        return value

    try:
        if isinstance(value, int) and not isinstance(value, bool):
            return int(value)
        if isinstance(value, float):
            return float(value)
    except Exception:
        pass

    raise ValueError(f"`{name}` must be an int or float, got {type(value).__name__}: {value!r}")


class FossielPythonSimple:
    @classmethod
    def INPUT_TYPES(cls):
        optional = {name: (any_type,) for name in "ABCDEFGHIJ"}
        return {
            "required": {
                "script": (
                    "STRING",
                    {
                        "default": "resultA = A + B",
                        "multiline": True,
                        "tooltip": (
                            "Use A–J (unconnected = 0). Assign any of resultA–resultJ "
                            "(unset outputs default to 0). math module available."
                        ),
                    },
                ),
            },
            "optional": optional,
        }

    RETURN_TYPES = tuple(any_type for _ in _RESULT_NAMES)
    RETURN_NAMES = tuple(f"Result{ch}" for ch in "ABCDEFGHIJ")
    FUNCTION = "run"
    CATEGORY = "Fossiel/script"

    def run(self, script, A=None, B=None, C=None, D=None, E=None, F=None, G=None, H=None, I=None, J=None):
        variables = {
            "A": _coerce_number("A", A),
            "B": _coerce_number("B", B),
            "C": _coerce_number("C", C),
            "D": _coerce_number("D", D),
            "E": _coerce_number("E", E),
            "F": _coerce_number("F", F),
            "G": _coerce_number("G", G),
            "H": _coerce_number("H", H),
            "I": _coerce_number("I", I),
            "J": _coerce_number("J", J),
        }
        env = _run_script(script, variables)
        return tuple(_coerce_result(name, env.get(name)) for name in _RESULT_NAMES)
