"""Leaf-only stateful adaptive cubature."""

from __future__ import annotations

import heapq
import itertools
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from ._rules import NestedRule, resolve_rule


Kernel = Callable[[np.ndarray], np.ndarray]
PayloadBuilder = Callable[[np.ndarray, np.ndarray], Iterable[Any]]
Evaluator = Callable[[np.ndarray, np.ndarray | list[Any], Any], np.ndarray]
LeafPayload = np.ndarray | tuple[Any, ...]


@dataclass(slots=True)
class IntegrationResult:
    """Result of a single stateful integration call.

    Status values:

    - ``"converged"``: the requested tolerance was met,
    - ``"max_subdivisions"``: refinement stopped after hitting ``max_subdivisions``,
    - ``"not_finite"``: the evaluator produced a non-finite estimate or error.
    """

    estimate: Any
    error: Any
    status: str
    n_kernel_evals: int
    n_evaluator_evals: int
    n_leaves: int
    n_leaf_nodes: int
    subdivisions: int


@dataclass(slots=True)
class _CallStats:
    n_kernel_evals: int = 0
    n_evaluator_evals: int = 0
    subdivisions: int = 0


@dataclass(slots=True)
class _Leaf:
    leaf_id: int
    a: np.ndarray
    b: np.ndarray
    payload: LeafPayload | None = None
    estimate: np.ndarray | np.generic | None = None
    error: np.ndarray | np.generic | None = None
    priority: float = field(default=float("inf"))


class StatefulIntegrator:
    """Adaptive cubature engine with a leaf-only mesh and per-leaf payload cache."""

    def __init__(
        self,
        a: np.ndarray | list[float],
        b: np.ndarray | list[float],
        kernel: Kernel,
        evaluator: Evaluator,
        rule: str = "auto",
        dtype: np.dtype | type = np.float64,
        batch_size: int | None = None,
        payload_builder: PayloadBuilder | None = None,
    ) -> None:
        """Create a stateful integrator on a finite rectangular domain.

        Args:
            a: Lower integration bounds. Reversed bounds are allowed and flip the result sign.
            b: Upper integration bounds. Bounds must be finite.
            kernel: Expensive callback evaluated at rule nodes. It must return a numeric array
                whose leading dimension matches the input batch size.
            evaluator: Cheap callback that combines points, cached payloads, and the parameter
                object passed to ``integrate(...)``. When ``payload_builder`` is ``None``, cached
                payloads are numeric arrays. Otherwise the evaluator receives a Python ``list`` of
                prepared per-node payload objects aligned with ``points``.
            rule: ``"auto"``, ``"gk21"``, or ``"genz_malik"``. ``"auto"`` selects ``"gk21"``
                in 1D and ``"genz_malik"`` otherwise.
            dtype: Floating-point dtype used for the integration domain and rule weights.
            batch_size: Optional maximum number of points passed to ``kernel`` and ``evaluator``
                per callback invocation.
            payload_builder: Optional callback that runs once for newly created rule nodes. It
                receives the flattened node batch and the raw numeric ``kernel`` payloads, and
                must return one prepared payload object per input point.
        """
        self.kernel = kernel
        self.evaluator = evaluator
        self.payload_builder = payload_builder
        self.dtype = np.dtype(dtype)
        self.batch_size = batch_size

        if not np.issubdtype(self.dtype, np.floating):
            raise TypeError("dtype must be a floating-point dtype")
        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be positive when provided")

        a_arr = self._as_point(a)
        b_arr = self._as_point(b)
        if np.any(~np.isfinite(a_arr)) or np.any(~np.isfinite(b_arr)):
            raise ValueError("StatefulIntegrator only supports finite integration bounds")

        self._orientation_sign = -1 if np.count_nonzero(a_arr > b_arr) % 2 else 1
        self.a = np.minimum(a_arr, b_arr)
        self.b = np.maximum(a_arr, b_arr)
        self.ndim = int(self.a.size)

        self.rule_name = self._resolve_rule_name(rule)
        self._rule: NestedRule = resolve_rule(self.rule_name, self.ndim, self.dtype)
        self._n_rule_nodes = self._rule.n_nodes
        self._child_sides = tuple(itertools.product((0, 1), repeat=self.ndim))

        self._payload_shape: tuple[int, ...] | None = None
        self._value_shape: tuple[int, ...] | None = None
        self._next_leaf_id = 0
        self._leaf_heap: list[tuple[float, int]] = []
        self._leaves: dict[int, _Leaf] = {}

        root = self._new_leaf(self.a, self.b)
        self._leaves[root.leaf_id] = root

    @property
    def n_leaves(self) -> int:
        return len(self._leaves)

    @property
    def n_leaf_nodes(self) -> int:
        return self.n_leaves * self._n_rule_nodes

    def replace_evaluator(self, evaluator: Evaluator) -> "StatefulIntegrator":
        """Return a new integrator sharing the current live leaf payloads.

        The clone starts from the current adaptive mesh snapshot, reuses cached leaf payloads, and
        resets only evaluator-dependent state. Future refinement happens independently in the
        original and cloned integrators.
        """

        clone = object.__new__(StatefulIntegrator)
        clone.kernel = self.kernel
        clone.evaluator = evaluator
        clone.payload_builder = self.payload_builder
        clone.dtype = self.dtype
        clone.batch_size = self.batch_size
        clone._orientation_sign = self._orientation_sign
        clone.a = self.a.copy()
        clone.b = self.b.copy()
        clone.ndim = self.ndim
        clone.rule_name = self.rule_name
        clone._rule = self._rule
        clone._n_rule_nodes = self._n_rule_nodes
        clone._child_sides = self._child_sides
        clone._payload_shape = self._payload_shape
        clone._value_shape = None
        clone._next_leaf_id = self._next_leaf_id
        clone._leaf_heap = []
        clone._leaves = {
            leaf_id: _Leaf(
                leaf_id=leaf.leaf_id,
                a=leaf.a,
                b=leaf.b,
                payload=leaf.payload,
            )
            for leaf_id, leaf in self._leaves.items()
        }
        return clone

    def integrate(
        self,
        params: Any = None,
        *,
        atol: float = 0.0,
        rtol: float = 1e-8,
        max_subdivisions: int | None = 10_000,
        **param_kwargs: Any,
    ) -> IntegrationResult:
        """Integrate for one parameter value and update the live adaptive mesh.

        Pass parameters either as ``params`` or as keyword arguments, but not both. When keyword
        arguments are used they are bundled into a dictionary and forwarded to ``evaluator`` as
        the third argument.

        Args:
            params: Parameter payload passed through to the evaluator.
            atol: Absolute tolerance. Must be non-negative.
            rtol: Relative tolerance. Must be non-negative.
            max_subdivisions: Maximum number of refinement steps for this call. ``None`` allows
                unbounded refinement.
            **param_kwargs: Keyword parameters forwarded as a dictionary when ``params`` is not
                provided.
        """

        if param_kwargs:
            if params is not None:
                raise ValueError("pass parameters either as `params` or keyword arguments, not both")
            params = param_kwargs
        if atol < 0 or rtol < 0:
            raise ValueError("atol and rtol must be non-negative")
        if max_subdivisions is not None and max_subdivisions < 0:
            raise ValueError("max_subdivisions must be non-negative")

        stats = _CallStats()
        leaf_ids = list(self._leaves)
        self._ensure_leaf_payloads(leaf_ids, stats)
        estimate, error, is_finite = self._refresh_leaves(leaf_ids, params, stats, rebuild_heap=True)
        signed_estimate = estimate * self._orientation_sign

        while True:
            if not is_finite:
                return self._build_result("not_finite", signed_estimate, error, stats)

            if self._converged(signed_estimate, error, atol=atol, rtol=rtol):
                return self._build_result("converged", signed_estimate, error, stats)

            if max_subdivisions is not None and stats.subdivisions >= max_subdivisions:
                return self._build_result("max_subdivisions", signed_estimate, error, stats)

            parent, child_ids = self._split_worst_leaf()
            self._ensure_leaf_payloads(child_ids, stats)
            child_estimate, child_error, child_finite = self._refresh_leaves(
                child_ids,
                params,
                stats,
                rebuild_heap=False,
            )

            estimate = estimate - np.asarray(parent.estimate) + child_estimate
            error = error - np.asarray(parent.error) + child_error
            signed_estimate = estimate * self._orientation_sign
            is_finite = child_finite and self._is_finite_result(estimate, error)
            stats.subdivisions += 1
            self._push_leaves(child_ids)

    def _as_point(self, value: np.ndarray | list[float]) -> np.ndarray:
        arr = np.asarray(value, dtype=self.dtype)
        if arr.ndim != 1:
            raise ValueError("integration bounds must be 1D arrays")
        arr = np.where(arr == 0, self.dtype.type(0), arr)
        return np.ascontiguousarray(arr)

    def _resolve_rule_name(self, rule: str) -> str:
        if rule == "auto":
            return "gk21" if self.ndim == 1 else "genz_malik"
        if rule not in {"gk21", "genz_malik"}:
            raise ValueError(f"unknown rule {rule!r}")
        return rule

    def _new_leaf(self, a: np.ndarray, b: np.ndarray) -> _Leaf:
        leaf = _Leaf(
            leaf_id=self._next_leaf_id,
            a=np.array(a, copy=True),
            b=np.array(b, copy=True),
        )
        self._next_leaf_id += 1
        return leaf

    def _leaf_batch_size(self) -> int | None:
        if self.batch_size is None:
            return None
        return max(1, self.batch_size // self._n_rule_nodes)

    def _iter_leaf_batches(self, leaf_ids: list[int]):
        batch_size = self._leaf_batch_size()
        if batch_size is None:
            yield leaf_ids
            return
        for start in range(0, len(leaf_ids), batch_size):
            yield leaf_ids[start : start + batch_size]

    def _ensure_leaf_payloads(self, leaf_ids: list[int], stats: _CallStats) -> None:
        pending = [leaf_id for leaf_id in leaf_ids if self._leaves[leaf_id].payload is None]
        if not pending:
            return

        if self.payload_builder is None:
            self._ensure_numeric_leaf_payloads(pending, stats)
            return

        self._ensure_prepared_leaf_payloads(pending, stats)

    def _ensure_numeric_leaf_payloads(self, leaf_ids: list[int], stats: _CallStats) -> None:
        for batch_ids in self._iter_leaf_batches(leaf_ids):
            batch = [self._leaves[leaf_id] for leaf_id in batch_ids]
            centers, half_widths = self._leaf_geometry(batch)
            nodes = self._map_nodes_batch(centers, half_widths)
            payloads = self._call_kernel(nodes.reshape(-1, self.ndim), stats)
            payloads = payloads.reshape((len(batch), self._n_rule_nodes) + self._payload_shape)
            for index, leaf in enumerate(batch):
                leaf.payload = np.array(payloads[index], copy=True)

    def _ensure_prepared_leaf_payloads(self, leaf_ids: list[int], stats: _CallStats) -> None:
        for batch_ids in self._iter_leaf_batches(leaf_ids):
            batch = [self._leaves[leaf_id] for leaf_id in batch_ids]
            centers, half_widths = self._leaf_geometry(batch)
            nodes = self._map_nodes_batch(centers, half_widths)
            flat_nodes = nodes.reshape(-1, self.ndim)
            raw_payloads = self._call_kernel(flat_nodes, stats)
            prepared_payloads = self._call_payload_builder(flat_nodes, raw_payloads)

            for index, leaf in enumerate(batch):
                start = index * self._n_rule_nodes
                stop = start + self._n_rule_nodes
                leaf.payload = tuple(prepared_payloads[start:stop])

    def _refresh_leaves(
        self,
        leaf_ids: list[int],
        params: Any,
        stats: _CallStats,
        *,
        rebuild_heap: bool,
    ) -> tuple[np.ndarray | np.generic, np.ndarray | np.generic, bool]:
        total_estimate: np.ndarray | np.generic | None = None
        total_error: np.ndarray | np.generic | None = None
        heap_entries: list[tuple[float, int]] = []

        for batch_ids in self._iter_leaf_batches(leaf_ids):
            batch = [self._leaves[leaf_id] for leaf_id in batch_ids]
            centers, half_widths = self._leaf_geometry(batch)
            scales = np.prod(half_widths, axis=1, dtype=self.dtype)
            nodes = self._map_nodes_batch(centers, half_widths)

            if self.payload_builder is None:
                values = self._evaluate_numeric_leaf_batch(batch, nodes, params, stats)
            else:
                values = self._evaluate_prepared_leaf_batch(batch, nodes, params, stats)

            estimates, errors = self._estimate_leaf_batch(half_widths, scales, values)
            batch_estimate_total = np.sum(estimates, axis=0)
            batch_error_total = np.sum(errors, axis=0)

            total_estimate = batch_estimate_total if total_estimate is None else total_estimate + batch_estimate_total
            total_error = batch_error_total if total_error is None else total_error + batch_error_total

            for index, leaf in enumerate(batch):
                leaf.estimate = np.array(estimates[index], copy=True)
                leaf.error = np.array(errors[index], copy=True)
                leaf.priority = self._priority_from_error(leaf.error)
                if rebuild_heap:
                    heap_entries.append((-leaf.priority, leaf.leaf_id))

        if total_estimate is None or total_error is None:
            raise RuntimeError("integrator has no active leaves")

        if rebuild_heap:
            self._leaf_heap = heap_entries
            heapq.heapify(self._leaf_heap)

        return total_estimate, total_error, self._is_finite_result(total_estimate, total_error)

    def _evaluate_numeric_leaf_batch(
        self,
        leaves: list[_Leaf],
        nodes: np.ndarray,
        params: Any,
        stats: _CallStats,
    ) -> np.ndarray:
        if self._payload_shape is None:
            raise RuntimeError("kernel payload shape is unknown")

        payloads = np.stack([np.asarray(leaf.payload) for leaf in leaves], axis=0)
        values = self._call_numeric_evaluator(
            nodes.reshape(-1, self.ndim),
            payloads.reshape((len(leaves) * self._n_rule_nodes,) + self._payload_shape),
            params,
            stats,
        )
        return values.reshape((len(leaves), self._n_rule_nodes) + self._value_shape)

    def _evaluate_prepared_leaf_batch(
        self,
        leaves: list[_Leaf],
        nodes: np.ndarray,
        params: Any,
        stats: _CallStats,
    ) -> np.ndarray:
        payloads = self._flatten_prepared_payloads(leaves)
        values = self._call_prepared_evaluator(nodes.reshape(-1, self.ndim), payloads, params, stats)
        return values.reshape((len(leaves), self._n_rule_nodes) + self._value_shape)

    def _leaf_geometry(self, leaves: list[_Leaf]) -> tuple[np.ndarray, np.ndarray]:
        a = np.stack([leaf.a for leaf in leaves], axis=0)
        b = np.stack([leaf.b for leaf in leaves], axis=0)
        half_widths = 0.5 * (b - a)
        centers = 0.5 * (a + b)
        return centers, half_widths

    def _map_nodes_batch(self, centers: np.ndarray, half_widths: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(
            centers[:, None, :] + half_widths[:, None, :] * self._rule.reference_nodes[None, :, :]
        )

    def _estimate_leaf_batch(
        self,
        half_widths: np.ndarray,
        scales: np.ndarray,
        values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        weights_shape = (len(scales), self._n_rule_nodes) + (1,) * (values.ndim - 2)
        high_weights = (scales[:, None] * self._rule.high_weights[None, :]).reshape(weights_shape)
        low_weights = (scales[:, None] * self._rule.low_weights[None, :]).reshape(weights_shape)

        high_estimates = np.sum(values * high_weights, axis=1)
        low_estimates = np.sum(values * low_weights, axis=1)

        if self.ndim == 1:
            errors = self._gauss_kronrod_error_batch(
                half_widths[:, 0],
                high_weights,
                values,
                high_estimates,
                low_estimates,
            )
        else:
            errors = np.abs(high_estimates - low_estimates)
        return high_estimates, errors

    def _gauss_kronrod_error_batch(
        self,
        half_widths: np.ndarray,
        high_weights: np.ndarray,
        values: np.ndarray,
        high_estimates: np.ndarray,
        low_estimates: np.ndarray,
    ) -> np.ndarray:
        err = np.abs(high_estimates - low_estimates)
        denom_shape = (len(half_widths),) + (1,) * (high_estimates.ndim - 1)
        denom = (2.0 * half_widths).reshape(denom_shape)
        center = np.divide(
            high_estimates,
            denom,
            out=np.zeros_like(high_estimates, dtype=np.result_type(high_estimates, self.dtype)),
            where=denom != 0,
        )

        dabs = np.sum(high_weights * np.abs(values - np.expand_dims(center, axis=1)), axis=1)
        ratio = np.divide(
            200.0 * err,
            dabs,
            out=np.zeros_like(np.asarray(err, dtype=np.result_type(err, np.float64))),
            where=dabs != 0,
        )
        err = np.where((dabs != 0) & (err != 0), dabs * np.minimum(1.0, ratio**1.5), err)

        abs_integrand = np.sum(high_weights * np.abs(values), axis=1)
        round_err = np.abs(50.0 * np.finfo(self.dtype).eps * abs_integrand)
        err = np.where(round_err > np.finfo(self.dtype).tiny, np.maximum(err, round_err), err)
        return err

    def _priority_from_error(self, error: np.ndarray | np.generic) -> float:
        error_arr = np.asarray(error)
        if not np.all(np.isfinite(error_arr)):
            return float("inf")
        return float(np.max(np.abs(error_arr)))

    def _push_leaves(self, leaf_ids: list[int]) -> None:
        for leaf_id in leaf_ids:
            leaf = self._leaves[leaf_id]
            heapq.heappush(self._leaf_heap, (-leaf.priority, leaf.leaf_id))

    def _split_worst_leaf(self) -> tuple[_Leaf, list[int]]:
        while self._leaf_heap:
            _, leaf_id = heapq.heappop(self._leaf_heap)
            parent = self._leaves.pop(leaf_id, None)
            if parent is not None:
                break
        else:
            raise RuntimeError("integrator has no active leaves to refine")

        midpoint = 0.5 * (parent.a + parent.b)
        child_ids: list[int] = []
        for side in self._child_sides:
            child_a = parent.a.copy()
            child_b = parent.b.copy()
            for axis, upper_half in enumerate(side):
                if upper_half:
                    child_a[axis] = midpoint[axis]
                else:
                    child_b[axis] = midpoint[axis]
            child = self._new_leaf(child_a, child_b)
            self._leaves[child.leaf_id] = child
            child_ids.append(child.leaf_id)

        return parent, child_ids

    def _call_kernel(self, points: np.ndarray, stats: _CallStats) -> np.ndarray:
        outputs: list[np.ndarray] = []
        for chunk in self._iter_chunks(points):
            values = np.asarray(self.kernel(chunk))
            values = self._validate_batch_output("kernel", values, chunk.shape[0], self._payload_shape)
            if self._payload_shape is None:
                self._payload_shape = values.shape[1:]
            outputs.append(values)
            stats.n_kernel_evals += chunk.shape[0]
        return np.concatenate(outputs, axis=0)

    def _call_payload_builder(self, points: np.ndarray, raw_payloads: np.ndarray) -> list[Any]:
        if self.payload_builder is None:
            raise RuntimeError("payload_builder is not configured")

        try:
            prepared_payloads = list(self.payload_builder(points, raw_payloads))
        except TypeError as exc:
            raise TypeError(
                "payload_builder must return an iterable with one prepared payload per input point"
            ) from exc

        if len(prepared_payloads) != points.shape[0]:
            raise ValueError(
                "payload_builder must return one prepared payload per input point, "
                f"got {len(prepared_payloads)} for {points.shape[0]} points"
            )
        return prepared_payloads

    def _call_numeric_evaluator(
        self,
        points: np.ndarray,
        payloads: np.ndarray,
        params: Any,
        stats: _CallStats,
    ) -> np.ndarray:
        outputs: list[np.ndarray] = []
        for chunk_points, chunk_payloads in zip(
            self._iter_chunks(points),
            self._iter_chunks(payloads),
            strict=True,
        ):
            values = np.asarray(self.evaluator(chunk_points, chunk_payloads, params))
            values = self._validate_batch_output("evaluator", values, chunk_points.shape[0], self._value_shape)
            if self._value_shape is None:
                self._value_shape = values.shape[1:]
            outputs.append(values)
            stats.n_evaluator_evals += chunk_points.shape[0]
        return np.concatenate(outputs, axis=0)

    def _call_prepared_evaluator(
        self,
        points: np.ndarray,
        payloads: list[Any],
        params: Any,
        stats: _CallStats,
    ) -> np.ndarray:
        outputs: list[np.ndarray] = []
        for chunk_points, chunk_payloads in zip(
            self._iter_chunks(points),
            self._iter_payload_chunks(payloads),
            strict=True,
        ):
            values = np.asarray(self.evaluator(chunk_points, chunk_payloads, params))
            values = self._validate_batch_output("evaluator", values, chunk_points.shape[0], self._value_shape)
            if self._value_shape is None:
                self._value_shape = values.shape[1:]
            outputs.append(values)
            stats.n_evaluator_evals += chunk_points.shape[0]
        return np.concatenate(outputs, axis=0)

    def _validate_batch_output(
        self,
        name: str,
        values: np.ndarray,
        npoints: int,
        expected_shape: tuple[int, ...] | None,
    ) -> np.ndarray:
        if values.ndim == 0 or values.shape[0] != npoints:
            raise ValueError(
                f"{name} must return an array with leading dimension equal to the batch size"
            )
        if not np.issubdtype(values.dtype, np.number):
            raise TypeError(f"{name} must return a numeric array")
        if expected_shape is not None and values.shape[1:] != expected_shape:
            raise ValueError(
                f"{name} must return a fixed trailing shape {expected_shape}, got {values.shape[1:]}"
            )
        return values

    def _iter_chunks(self, values: np.ndarray):
        if self.batch_size is None:
            yield values
            return
        for start in range(0, values.shape[0], self.batch_size):
            yield values[start : start + self.batch_size]

    def _iter_payload_chunks(self, values: list[Any]):
        if self.batch_size is None:
            yield values
            return
        for start in range(0, len(values), self.batch_size):
            yield values[start : start + self.batch_size]

    def _flatten_prepared_payloads(self, leaves: list[_Leaf]) -> list[Any]:
        payloads: list[Any] = []
        for leaf in leaves:
            if leaf.payload is None:
                raise RuntimeError("leaf payloads must be available before evaluation")
            payloads.extend(leaf.payload)
        return payloads

    def _is_finite_result(self, estimate: np.ndarray | np.generic, error: np.ndarray | np.generic) -> bool:
        return bool(np.all(np.isfinite(np.asarray(estimate))) and np.all(np.isfinite(np.asarray(error))))

    def _converged(self, estimate: np.ndarray | np.generic, error: np.ndarray | np.generic, *, atol: float, rtol: float) -> bool:
        tolerance = atol + rtol * np.abs(estimate)
        return bool(np.all(error <= tolerance))

    def _build_result(
        self,
        status: str,
        estimate: np.ndarray | np.generic,
        error: np.ndarray | np.generic,
        stats: _CallStats,
    ) -> IntegrationResult:
        return IntegrationResult(
            estimate=self._coerce_result(estimate),
            error=self._coerce_result(error),
            status=status,
            n_kernel_evals=stats.n_kernel_evals,
            n_evaluator_evals=stats.n_evaluator_evals,
            n_leaves=self.n_leaves,
            n_leaf_nodes=self.n_leaf_nodes,
            subdivisions=stats.subdivisions,
        )

    def _coerce_result(self, value: np.ndarray | np.generic) -> Any:
        arr = np.asarray(value)
        if arr.ndim == 0:
            return arr.item()
        return arr
