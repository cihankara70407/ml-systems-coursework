"""
Automated Differentiation Framework: Tape-Free Directed Acyclic Graph Engine
Author: Cihan Karayigit (Student ID: 70407)
Academic Program: Computer Engineering, Vistula University
Module: Lab 05 - Frameworks and Automatic Differentiation
Reference: Vijay Janapa Reddi, Machine Learning Systems, Chapter 7
"""

from __future__ import annotations
import math
from typing import Tuple, Set, Callable, Optional, Union, List
from collections import deque


class AutoDiffScalar:
    """
    Primitive scalar vertex within a dynamic compute graph.
    Maintains primal state, accumulated derivative, ancestor references,
    and a local gradient backward propagation functor.
    """
    __slots__ = ('data', 'grad', '_ancestors', '_grad_fn', '_op_type', 'annotation')

    def __init__(
        self,
        data: Union[float, int],
        ancestors: Tuple[AutoDiffScalar, ...] = (),
        op_type: str = 'leaf',
        annotation: Optional[str] = None
    ) -> None:
        self.data: float = float(data)
        self.grad: float = 0.0
        self._ancestors: Tuple[AutoDiffScalar, ...] = ancestors
        self._grad_fn: Callable[[], None] = lambda: None
        self._op_type: str = op_type
        self.annotation: Optional[str] = annotation

    def __repr__(self) -> str:
        tag = f":{self.annotation}" if self.annotation else ""
        return f"AutoDiffScalar({self.data:.5f}, grad={self.grad:.5f}{tag})"

    def __add__(self, other: Union[AutoDiffScalar, float, int]) -> AutoDiffScalar:
        rhs = other if isinstance(other, AutoDiffScalar) else AutoDiffScalar(other)
        out = AutoDiffScalar(self.data + rhs.data, (self, rhs), 'add')

        def _backward_add() -> None:
            # Partial derivatives: dz/dx = 1.0, dz/dy = 1.0
            self.grad += 1.0 * out.grad
            rhs.grad += 1.0 * out.grad

        out._grad_fn = _backward_add
        return out

    def __radd__(self, other: Union[float, int]) -> AutoDiffScalar:
        return self.__add__(other)

    def __mul__(self, other: Union[AutoDiffScalar, float, int]) -> AutoDiffScalar:
        rhs = other if isinstance(other, AutoDiffScalar) else AutoDiffScalar(other)
        out = AutoDiffScalar(self.data * rhs.data, (self, rhs), 'mul')

        def _backward_mul() -> None:
            # Partial derivatives: dz/dx = y, dz/dy = x
            self.grad += rhs.data * out.grad
            rhs.grad += self.data * out.grad

        out._grad_fn = _backward_mul
        return out

    def __rmul__(self, other: Union[float, int]) -> AutoDiffScalar:
        return self.__mul__(other)

    def __pow__(self, power: Union[float, int]) -> AutoDiffScalar:
        if not isinstance(power, (int, float)):
            raise TypeError(f"Power index must be float/int, got {type(power).__name__}")
        out = AutoDiffScalar(self.data ** power, (self,), f'pow({power})')

        def _backward_pow() -> None:
            # Power rule: d(u^p)/du = p * u^(p-1)
            self.grad += (power * (self.data ** (power - 1))) * out.grad

        out._grad_fn = _backward_pow
        return out

    def relu(self) -> AutoDiffScalar:
        activated = self.data if self.data > 0.0 else 0.0
        out = AutoDiffScalar(activated, (self,), 'relu')

        def _backward_relu() -> None:
            # Subgradient for ReLU: 1.0 if x > 0 else 0.0
            gate = 1.0 if self.data > 0.0 else 0.0
            self.grad += gate * out.grad

        out._grad_fn = _backward_relu
        return out

    def sigmoid(self) -> AutoDiffScalar:
        # Logistic activation with overflow suppression
        if self.data >= 0.0:
            exp_term = math.exp(-self.data)
            sig = 1.0 / (1.0 + exp_term)
        else:
            exp_term = math.exp(self.data)
            sig = exp_term / (1.0 + exp_term)
        out = AutoDiffScalar(sig, (self,), 'sigmoid')

        def _backward_sigmoid() -> None:
            # Logistic derivative: dsigma/dx = sigma * (1 - sigma)
            self.grad += (sig * (1.0 - sig)) * out.grad

        out._grad_fn = _backward_sigmoid
        return out

    def log(self, eps: float = 1e-15) -> AutoDiffScalar:
        # Clamped logarithm for robust cross-entropy evaluation
        clamped = max(self.data, eps)
        out = AutoDiffScalar(math.log(clamped), (self,), 'log')

        def _backward_log() -> None:
            # Derivative: d(ln x)/dx = 1 / x
            self.grad += (1.0 / clamped) * out.grad

        out._grad_fn = _backward_log
        return out

    def __neg__(self) -> AutoDiffScalar:
        return self * -1.0

    def __sub__(self, other: Union[AutoDiffScalar, float, int]) -> AutoDiffScalar:
        return self + (-other)

    def __rsub__(self, other: Union[float, int]) -> AutoDiffScalar:
        return AutoDiffScalar(other) + (-self)

    def __truediv__(self, other: Union[AutoDiffScalar, float, int]) -> AutoDiffScalar:
        rhs = other if isinstance(other, AutoDiffScalar) else AutoDiffScalar(other)
        return self * (rhs ** -1.0)

    def backward(self) -> None:
        """
        Executes reverse-mode backpropagation via DAG topological sorting.
        Gradients are strictly accumulated using += to preserve multivariate chain rule.
        """
        sorted_nodes: List[AutoDiffScalar] = []
        visited: Set[AutoDiffScalar] = set()

        def dfs_visit(node: AutoDiffScalar) -> None:
            if node not in visited:
                visited.add(node)
                for ancestor in node._ancestors:
                    dfs_visit(ancestor)
                sorted_nodes.append(node)

        dfs_visit(self)

        # Seed loss derivative
        self.grad = 1.0

        # Traverse reverse topological schedule
        for vertex in reversed(sorted_nodes):
            vertex._grad_fn()
