"""Generate the CUDA, Metal and NumPy versions of a scalar kernel block from its Pyccel source.

A block is a small function of scalars shared by the kernels of several packages (a Boris
update, the cell of a position on a grid): one expression per result, written for Pyccel
(``@inline``, no local variables). :class:`KernelBlock` translates it to

* a CUDA ``__device__`` function (double precision), for a header;
* a Metal function (float32), for the ``header=`` of a :class:`MetalKernel`;
* a NumPy function on arrays (or scalars), elementwise.

The translation covers what a block needs: arithmetic (``+ - * / // **`` and unary minus),
comparisons (also chained), ``and``/``or``/``not``, conditional expressions, ``int(x)``
(truncation, as in Python and C) and ``float(x)``, the results as one expression or a
tuple. Subexpressions used more than once are computed once into temporaries in the
generated code (the C compilers would merge them anyway; NumPy would not).
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from collections.abc import Callable
from typing import Any

__all__ = ["KernelBlock"]

_C_TYPES = {
    "cuda": {"float": "double", "int": "long long", "bool": "bool"},
    "metal": {"float": "float", "int": "int", "bool": "bool"},
}
_BINARY = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}
_COMPARE = {
    ast.Lt: "<",
    ast.LtE: "<=",
    ast.Gt: ">",
    ast.GtE: ">=",
    ast.Eq: "==",
    ast.NotEq: "!=",
}


def _annotation(node: ast.expr | None, what: str) -> str:
    """Return ``float``, ``int`` or ``bool`` for a scalar annotation."""
    if isinstance(node, ast.Name) and node.id in ("float", "int", "bool"):
        return node.id
    if isinstance(node, ast.Constant) and node.value in ("float", "int", "bool"):
        return node.value
    raise TypeError(f"{what}: a block takes and returns scalars (float, int, bool)")


def _result_types(node: ast.expr | None) -> list[str]:
    """Return the types of the results from the return annotation."""
    if (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == "tuple"
    ):
        elements = (
            node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        )
        return [_annotation(e, "result") for e in elements]
    return [_annotation(node, "result")]


class KernelBlock:
    """A scalar kernel block, written once for Pyccel, in every backend.

    Parameters
    ----------
    function : callable
        The block: a function of scalars whose body is one ``return`` of an expression
        or a tuple of expressions (an ``@inline`` Pyccel function; the decorator is
        ignored).

    Raises
    ------
    TypeError
        If an argument or result is not a scalar, or the body is not one return.
    NotImplementedError
        If the body uses a construct the translation does not cover.

    Examples
    --------
    In a module (the source is read with :func:`inspect.getsource`)::

        def axpy(a: float, x: float, y: float) -> float:
            return a * x + y

    ``KernelBlock(axpy).cuda()`` is then::

        __device__ inline double axpy(double a, double x, double y) {
            return ((a * x) + y);
        }

    and ``KernelBlock(axpy).numpy()(2.0, 3.0, 1.0)`` is ``7.0``.
    """

    def __init__(self, function: Callable[..., Any]) -> None:
        source = textwrap.dedent(inspect.getsource(function))
        definition = ast.parse(source).body[0]
        if not isinstance(definition, ast.FunctionDef):
            raise TypeError("a block is a function")
        body = [
            s
            for s in definition.body
            if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))
        ]
        if (
            len(body) != 1
            or not isinstance(body[0], ast.Return)
            or body[0].value is None
        ):
            raise TypeError(
                f"{definition.name}: the body of a block is one return statement"
            )
        self.name = definition.name
        self.arguments = [
            (a.arg, _annotation(a.annotation, a.arg)) for a in definition.args.args
        ]
        self.types = dict(self.arguments)
        value = body[0].value
        self.results = list(value.elts) if isinstance(value, ast.Tuple) else [value]
        self.result_types = _result_types(definition.returns)
        if len(self.result_types) != len(self.results):
            raise TypeError(
                f"{self.name}: the return annotation does not match the results"
            )

    # --- common subexpressions -------------------------------------------------------

    def _temporaries(self) -> tuple[list[tuple[str, ast.expr]], dict[str, str]]:
        """Return the subexpressions used more than once, innermost first, and their names."""
        counts: dict[str, int] = {}
        nodes: dict[str, ast.expr] = {}

        def visit(node: ast.expr) -> None:
            if isinstance(node, (ast.Name, ast.Constant)):
                return
            key = ast.dump(node)
            counts[key] = counts.get(key, 0) + 1
            nodes.setdefault(key, node)
            if counts[key] == 1:
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, ast.expr):
                        visit(child)

        for result in self.results:
            visit(result)
        shared = {key for key, count in counts.items() if count > 1}
        # the operands of a conditional are evaluated only on their branch: keep them out
        order: list[str] = []

        def collect(node: ast.expr) -> None:
            if isinstance(node, (ast.Name, ast.Constant)):
                return
            for child in ast.iter_child_nodes(node):
                if isinstance(child, ast.expr):
                    collect(child)
            key = ast.dump(node)
            if key in shared and key not in order:
                order.append(key)

        for result in self.results:
            collect(result)
        names = {key: f"t{i}" for i, key in enumerate(order)}
        return [(names[key], nodes[key]) for key in order], names

    def _kind(self, node: ast.expr) -> str:
        """Return the scalar type of an expression: float, int or bool."""
        if isinstance(node, ast.Name):
            return self.types[node.id]
        if isinstance(node, ast.Constant):
            return (
                "bool"
                if isinstance(node.value, bool)
                else ("int" if isinstance(node.value, int) else "float")
            )
        if isinstance(node, (ast.Compare, ast.BoolOp)) or (
            isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not)
        ):
            return "bool"
        if isinstance(node, ast.Call):
            return (
                node.func.id
                if isinstance(node.func, ast.Name) and node.func.id in ("int", "float")
                else "float"
            )
        if isinstance(node, ast.IfExp):
            kinds = {self._kind(node.body), self._kind(node.orelse)}
            return "float" if "float" in kinds else kinds.pop()
        if isinstance(node, ast.UnaryOp):
            return self._kind(node.operand)
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.Div):
                return "float"
            kinds = {self._kind(node.left), self._kind(node.right)}
            return "float" if "float" in kinds else "int"
        raise NotImplementedError(ast.dump(node))

    # --- C (CUDA and Metal) ----------------------------------------------------------

    def _c_type(self, node: ast.expr, backend: str) -> str:
        return _C_TYPES[backend][self._kind(node)]

    def _c(
        self, node: ast.expr, names: dict[str, str], backend: str, top: bool = False
    ) -> str:
        key = ast.dump(node)
        if not top and key in names:
            return names[key]
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool):
                return "true" if node.value else "false"
            if isinstance(node.value, float):
                text = repr(node.value)
                if "." not in text and "e" not in text and "n" not in text:
                    text += ".0"
                return text + ("f" if backend == "metal" else "")
            return repr(node.value)
        if isinstance(node, ast.UnaryOp):
            operand = self._c(node.operand, names, backend)
            if isinstance(node.op, ast.USub):
                return f"(-{operand})"
            if isinstance(node.op, ast.UAdd):
                return operand
            if isinstance(node.op, ast.Not):
                return f"(!{operand})"
        if isinstance(node, ast.BinOp):
            left = self._c(node.left, names, backend)
            right = self._c(node.right, names, backend)
            if type(node.op) in _BINARY:
                return f"({left} {_BINARY[type(node.op)]} {right})"
            if isinstance(node.op, ast.FloorDiv):
                if self._kind(node) == "int":
                    raise NotImplementedError(
                        "// of integers (Python floors, C truncates)"
                    )
                floor = "metal::floor" if backend == "metal" else "floor"
                return f"{floor}({left} / {right})"
            if isinstance(node.op, ast.Pow):
                power = "metal::pow" if backend == "metal" else "pow"
                return f"{power}({left}, {right})"
        if isinstance(node, ast.Compare):
            parts, left = [], node.left
            for op, right in zip(node.ops, node.comparators, strict=True):
                if type(op) not in _COMPARE:
                    raise NotImplementedError(ast.dump(op))
                parts.append(
                    f"({self._c(left, names, backend)} {_COMPARE[type(op)]} {self._c(right, names, backend)})"
                )
                left = right
            return parts[0] if len(parts) == 1 else "(" + " && ".join(parts) + ")"
        if isinstance(node, ast.BoolOp):
            joint = " && " if isinstance(node.op, ast.And) else " || "
            return (
                "(" + joint.join(self._c(v, names, backend) for v in node.values) + ")"
            )
        if isinstance(node, ast.IfExp):
            return (
                f"({self._c(node.test, names, backend)} ? {self._c(node.body, names, backend)} "
                f": {self._c(node.orelse, names, backend)})"
            )
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and len(node.args) == 1
        ):
            argument = self._c(node.args[0], names, backend)
            target = {
                "int": _C_TYPES[backend]["int"],
                "float": _C_TYPES[backend]["float"],
            }.get(node.func.id)
            if target is not None:
                return f"(({target}){argument})"
            if node.func.id in ("sqrt", "abs"):
                function = {"sqrt": "sqrt", "abs": "fabs"}[node.func.id]
                return (
                    f"{'metal::' if backend == 'metal' else ''}{function}({argument})"
                )
        raise NotImplementedError(f"{self.name}: {ast.unparse(node)}")

    def _c_function(
        self, backend: str, name: str | None, qualifier: str, reference: str
    ) -> str:
        name = name or self.name
        temporaries, names = self._temporaries()
        arguments = [f"{_C_TYPES[backend][kind]} {arg}" for arg, kind in self.arguments]
        lines = []
        for temporary, node in temporaries:
            body_names = {k: v for k, v in names.items() if v != temporary}
            lines.append(
                f"    const {self._c_type(node, backend)} {temporary} = {self._c(node, body_names, backend, top=True)};"
            )
        if len(self.results) == 1:
            result_type = _C_TYPES[backend][self.result_types[0]]
            lines.append(
                f"    return {self._c(self.results[0], names, backend, top=False)};"
            )
            head = f"{qualifier}{result_type} {name}({', '.join(arguments)})"
        else:
            outputs = [
                f"{reference.format(_C_TYPES[backend][kind])}result{i}"
                for i, kind in enumerate(self.result_types)
            ]
            for i, result in enumerate(self.results):
                target = f"*result{i}" if reference.endswith("*") else f"result{i}"
                lines.append(
                    f"    {target} = {self._c(result, names, backend, top=False)};"
                )
            head = f"{qualifier}void {name}({', '.join(arguments + outputs)})"
        return head + " {\n" + "\n".join(lines) + "\n}"

    def cuda(self, name: str | None = None) -> str:
        """Return the block as a CUDA ``__device__`` function (double precision).

        One result is the return value; several are written through pointers
        ``result0, result1, ...`` after the arguments.

        Parameters
        ----------
        name : str, optional
            The function's name; by default the block's.

        Returns
        -------
        str
            The source of the function, for a header or a kernel.
        """
        return self._c_function("cuda", name, "__device__ inline ", "{} *")

    def metal(self, name: str | None = None) -> str:
        """Return the block as a Metal function (float32), for a kernel's ``header=``.

        One result is the return value; several are written through ``thread``
        references ``result0, result1, ...`` after the arguments.

        Parameters
        ----------
        name : str, optional
            The function's name; by default the block's.

        Returns
        -------
        str
            The source of the function.
        """
        return self._c_function("metal", name, "inline ", "thread {}& ")

    # --- NumPy -----------------------------------------------------------------------

    def _py(self, node: ast.expr, names: dict[str, str], top: bool = False) -> str:
        key = ast.dump(node)
        if not top and key in names:
            return names[key]
        if isinstance(node, (ast.Name, ast.Constant)):
            return ast.unparse(node)
        if isinstance(node, ast.UnaryOp):
            operand = self._py(node.operand, names)
            if isinstance(node.op, ast.Not):
                return f"np.logical_not({operand})"
            return f"({'-' if isinstance(node.op, ast.USub) else '+'}{operand})"
        if isinstance(node, ast.BinOp):
            op = {
                ast.Add: "+",
                ast.Sub: "-",
                ast.Mult: "*",
                ast.Div: "/",
                ast.FloorDiv: "//",
                ast.Pow: "**",
            }.get(type(node.op))
            if op is None:
                raise NotImplementedError(ast.unparse(node))
            return f"({self._py(node.left, names)} {op} {self._py(node.right, names)})"
        if isinstance(node, ast.Compare):
            parts, left = [], node.left
            for op, right in zip(node.ops, node.comparators, strict=True):
                parts.append(
                    f"({self._py(left, names)} {_COMPARE[type(op)]} {self._py(right, names)})"
                )
                left = right
            return parts[0] if len(parts) == 1 else "(" + " & ".join(parts) + ")"
        if isinstance(node, ast.BoolOp):
            function = (
                "np.logical_and" if isinstance(node.op, ast.And) else "np.logical_or"
            )
            text = self._py(node.values[0], names)
            for value in node.values[1:]:
                text = f"{function}({text}, {self._py(value, names)})"
            return text
        if isinstance(node, ast.IfExp):
            return f"np.where({self._py(node.test, names)}, {self._py(node.body, names)}, {self._py(node.orelse, names)})"
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and len(node.args) == 1
        ):
            argument = self._py(node.args[0], names)
            function = {
                "int": "np.trunc({}).astype(np.int64)",
                "float": "np.asarray({}, dtype=np.float64)",
                "sqrt": "np.sqrt({})",
                "abs": "np.abs({})",
            }.get(node.func.id)
            if function is not None:
                return function.format(argument)
        raise NotImplementedError(f"{self.name}: {ast.unparse(node)}")

    def numpy_source(self, name: str | None = None) -> str:
        """Return the block as the source of a NumPy function, elementwise on arrays.

        Parameters
        ----------
        name : str, optional
            The function's name; by default the block's.

        Returns
        -------
        str
            The source, which uses ``np`` for NumPy.
        """
        temporaries, names = self._temporaries()
        lines = [
            f"def {name or self.name}({', '.join(arg for arg, _ in self.arguments)}):"
        ]
        for temporary, node in temporaries:
            body_names = {k: v for k, v in names.items() if v != temporary}
            lines.append(f"    {temporary} = {self._py(node, body_names, top=True)}")
        results = [self._py(r, names) for r in self.results]
        lines.append(
            f"    return {results[0]}"
            if len(results) == 1
            else f"    return ({', '.join(results)})"
        )
        return "\n".join(lines) + "\n"

    def numpy(self) -> Callable[..., Any]:
        """Return the block as a NumPy function, elementwise on arrays or scalars.

        Returns
        -------
        callable
            The function, with the block's arguments; scalar arguments give scalar
            results.
        """
        import numpy as np

        namespace: dict[str, Any] = {"np": np}
        exec(compile(self.numpy_source(), f"<block {self.name}>", "exec"), namespace)  # noqa: S102
        function = namespace[self.name]

        def run(*args: Any) -> Any:
            out = function(*args)
            scalar = all(np.ndim(a) == 0 for a in args)
            if not scalar:
                return out
            convert = lambda v: v.item() if hasattr(v, "item") else v
            return (
                tuple(convert(v) for v in out)
                if isinstance(out, tuple)
                else convert(out)
            )

        run.__name__ = self.name
        run.__doc__ = f"The NumPy version of the block ``{self.name}``."
        return run
