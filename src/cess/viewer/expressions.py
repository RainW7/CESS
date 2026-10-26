"""Small, non-executable array filter language for the optional viewer.

Expressions accept x/y, numeric constants, arithmetic, comparisons, &, | and
an explicit set of pure NumPy functions. No Python eval, arbitrary attributes,
indexing, imports, comprehensions, mutation or user-provided callables are used.
"""
from __future__ import annotations

import ast
import math

import numpy as np


_FUNCTIONS = {
    "isfinite": (np.isfinite, 1, 1),
    "isnan": (np.isnan, 1, 1),
    "isinf": (np.isinf, 1, 1),
    "abs": (np.abs, 1, 1),
    "log10": (np.log10, 1, 1),
    "log": (np.log, 1, 1),
    "sqrt": (np.sqrt, 1, 1),
    "exp": (np.exp, 1, 1),
    "minimum": (np.minimum, 2, 2),
    "maximum": (np.maximum, 2, 2),
    "clip": (np.clip, 3, 3),
    "logical_and": (np.logical_and, 2, 2),
    "logical_or": (np.logical_or, 2, 2),
    "logical_not": (np.logical_not, 1, 1),
    "isclose": (np.isclose, 2, 4),
    "where": (np.where, 3, 3),
}
_BINARY = {
    ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply,
    ast.Div: np.true_divide, ast.FloorDiv: np.floor_divide,
    ast.Mod: np.remainder, ast.Pow: np.power,
    ast.BitAnd: np.bitwise_and, ast.BitOr: np.bitwise_or, ast.BitXor: np.bitwise_xor,
}
_UNARY = {ast.UAdd: np.positive, ast.USub: np.negative, ast.Invert: np.invert}
_COMPARE = {
    ast.Eq: np.equal, ast.NotEq: np.not_equal, ast.Lt: np.less,
    ast.LtE: np.less_equal, ast.Gt: np.greater, ast.GtE: np.greater_equal,
}


def _evaluate(node, values):
    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, bool):
            return value
        if isinstance(value, int) and abs(value) <= 2 ** 64 - 1:
            return value
        if isinstance(value, float) and math.isfinite(value):
            return value
        raise ValueError("Only finite numeric or boolean constants are allowed")
    if isinstance(node, ast.Name) and node.id in values:
        return values[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        return _BINARY[type(node.op)](_evaluate(node.left, values), _evaluate(node.right, values))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_evaluate(node.operand, values))
    if isinstance(node, ast.Compare):
        left = _evaluate(node.left, values)
        result = True
        for operation, right_node in zip(node.ops, node.comparators):
            if type(operation) not in _COMPARE:
                raise ValueError("Only numeric comparison operators are allowed")
            right = _evaluate(right_node, values)
            result = np.logical_and(result, _COMPARE[type(operation)](left, right))
            left = right
        return result
    if isinstance(node, ast.Call):
        function = node.func
        if isinstance(function, ast.Name):
            name = function.id
        elif (isinstance(function, ast.Attribute) and isinstance(function.value, ast.Name)
              and function.value.id in ("np", "numpy")):
            name = function.attr
        else:
            raise ValueError("Only named pure array functions are allowed")
        if name not in _FUNCTIONS or node.keywords:
            raise ValueError("Unsupported function or keyword arguments in filter")
        call, minimum, maximum = _FUNCTIONS[name]
        if not minimum <= len(node.args) <= maximum:
            raise ValueError("Wrong number of arguments for filter function " + name)
        return call(*[_evaluate(argument, values) for argument in node.args])
    if isinstance(node, ast.BoolOp):
        raise ValueError("Use & and | with parenthesized conditions, not and/or")
    raise ValueError("Unsupported expression element: " + type(node).__name__)


def filter_mask(x, y, expression=""):
    """Return a one-dimensional finite-and-expression mask for equal-sized data.

    An empty expression retains finite numeric values (strings are retained).
    A nonempty expression must itself return a boolean array with exactly the
    same shape. Scalar predicates and implicit numeric-to-bool conversion are
    rejected. Function calls cannot pass ``out`` or other mutating arguments.
    """
    x, y = np.asarray(x), np.asarray(y)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError("X and Y must be one-dimensional arrays of equal length")
    mask = np.ones(x.shape, dtype=bool)
    for values in (x, y):
        if values.dtype.kind in "fiubc":
            mask &= np.isfinite(values)
    expression = str(expression).strip()
    if not expression:
        return mask
    if len(expression) > 4096:
        raise ValueError("Filter expression is too long")
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, RecursionError) as error:
        raise ValueError("Invalid filter expression") from error
    if sum(1 for _ in ast.walk(tree)) > 128:
        raise ValueError("Filter expression is too complex")
    try:
        with np.errstate(all="ignore"):
            result = np.asarray(_evaluate(tree.body, {"x": x, "y": y}))
    except (TypeError, OverflowError, FloatingPointError, ZeroDivisionError) as error:
        raise ValueError("Invalid numeric filter operation: " + str(error)) from error
    if result.dtype.kind != "b" or result.shape != x.shape:
        raise ValueError("Filter must return a boolean array matching X and Y")
    return mask & result
