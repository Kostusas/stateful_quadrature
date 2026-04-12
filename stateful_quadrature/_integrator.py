"""Stateful adaptive quadrature built around exact node reuse."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from ._rules import NestedRule, map_rule, resolve_rule


Kernel = Callable[[np.ndarray], np.ndarray]
Evaluator = Callable[[np.ndarray, np.ndarray, Any], np.ndarray]


@dataclass(slots=True)
class IntegrationResult:
    """Result of a single parameter-specific integration call."""

    estimate: Any
    error: Any
    status: str
    n_kernel_evals: int
    n_evaluator_evals: int
    n_regions: int
    n_cached_nodes: int
    subdivisions: int


@dataclass(slots=True)
class _CallStats:
    n_kernel_evals: int = 0
    n_evaluator_evals: int = 0
    subdivisions: int = 0


@dataclass(slots=True)
class _Region:
    a: np.ndarray
    b: np.ndarray
    node_coords: np.ndarray
    high_weights: np.ndarray
    low_weights: np.ndarray
    node_ids: np.ndarray | None = None
    estimate: np.ndarray | np.generic | None = None
    error: np.ndarray | np.generic | None = None

    @property
    def priority(self) -> float:
        if self.error is None:
            return float("inf")
        error = np.asarray(self.error)
        if not np.all(np.isfinite(error)):
            return float("inf")
        return float(np.max(np.abs(error)))


class StatefulIntegrator:
    """Adaptive quadrature engine that reuses exact cached kernel evaluations."""

    def __init__(
        self,
        a: np.ndarray | list[float],
        b: np.ndarray | list[float],
        kernel: Kernel,
        evaluator: Evaluator,
        rule: str = "auto",
        dtype: np.dtype | type = np.float64,
        batch_size: int | None = None,
        points: list[np.ndarray | list[float] | float] | None = None,
    ) -> None:
        self.kernel = kernel
        self.evaluator = evaluator
        self.dtype = np.dtype(dtype)
        self.batch_size = batch_size

        if not np.issubdtype(self.dtype, np.floating):
            raise TypeError("dtype must be a floating-point dtype")
        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be positive when provided")

        a_arr = self._as_point(a)
        b_arr = self._as_point(b)
        if np.any(~np.isfinite(a_arr)) or np.any(~np.isfinite(b_arr)):
            raise ValueError("v1 only supports finite integration bounds")

        self._orientation_sign = -1 if np.count_nonzero(a_arr > b_arr) % 2 else 1
        self.a = np.minimum(a_arr, b_arr)
        self.b = np.maximum(a_arr, b_arr)

        self.ndim = int(self.a.size)
        self.rule_name = self._resolve_rule_name(rule)
        self._rule: NestedRule = resolve_rule(self.rule_name, self.ndim, self.dtype)

        self._node_index: dict[bytes, int] = {}
        self._node_coords: list[np.ndarray] = []
        self._payloads: list[np.ndarray] = []
        self._payload_shape: tuple[int, ...] | None = None
        self._value_shape: tuple[int, ...] | None = None

        initial_regions = self._split_initial_regions(self.a, self.b, points)
        self._leaf_regions = [self._build_region(left, right) for left, right in initial_regions]

    @property
    def n_cached_nodes(self) -> int:
        return len(self._node_coords)

    @property
    def n_regions(self) -> int:
        return len(self._leaf_regions)

    def integrate(
        self,
        params: Any = None,
        *,
        atol: float = 0.0,
        rtol: float = 1e-8,
        max_subdivisions: int | None = 10_000,
        **param_kwargs: Any,
    ) -> IntegrationResult:
        """Integrate for one parameter value and update the cached adaptive state."""

        if param_kwargs:
            if params is not None:
                raise ValueError("pass parameters either as `params` or keyword arguments, not both")
            params = param_kwargs
        if atol < 0 or rtol < 0:
            raise ValueError("atol and rtol must be non-negative")
        if max_subdivisions is not None and max_subdivisions < 0:
            raise ValueError("max_subdivisions must be non-negative")

        stats = _CallStats()
        self._ensure_region_nodes(self._leaf_regions, stats)
        estimate, error, is_finite = self._evaluate_regions(self._leaf_regions, params, stats)
        signed_estimate = estimate * self._orientation_sign

        while True:
            if not is_finite:
                return self._build_result("not_finite", signed_estimate, error, stats)

            if self._converged(signed_estimate, error, atol=atol, rtol=rtol):
                return self._build_result("converged", signed_estimate, error, stats)

            if max_subdivisions is not None and stats.subdivisions >= max_subdivisions:
                return self._build_result("max_subdivisions", signed_estimate, error, stats)

            parent_region, children = self._refine_worst_region()
            self._ensure_region_nodes(children, stats)
            child_estimate, child_error, child_finite = self._evaluate_regions(children, params, stats)

            estimate = estimate - parent_region.estimate + child_estimate
            error = error - parent_region.error + child_error
            signed_estimate = estimate * self._orientation_sign
            is_finite = child_finite and self._is_finite_result(estimate, error)
            stats.subdivisions += 1

    def _as_point(self, value: np.ndarray | list[float]) -> np.ndarray:
        arr = np.asarray(value, dtype=self.dtype)
        if arr.ndim != 1:
            raise ValueError("integration bounds must be 1D arrays")
        arr = np.where(arr == 0, self.dtype.type(0), arr)
        return np.ascontiguousarray(arr)

    def _resolve_rule_name(self, rule: str) -> str:
        if rule == "auto":
            return "gk21" if self.ndim == 1 else "genz_malik"
        if rule not in {"gk15", "gk21", "trapezoid", "genz_malik"}:
            raise ValueError(f"unknown rule {rule!r}")
        return rule

    def _build_region(self, a: np.ndarray, b: np.ndarray) -> _Region:
        node_coords, high_weights, low_weights = map_rule(self._rule, a, b)
        return _Region(
            a=np.array(a, copy=True),
            b=np.array(b, copy=True),
            node_coords=np.ascontiguousarray(node_coords, dtype=self.dtype),
            high_weights=np.asarray(high_weights, dtype=self.dtype),
            low_weights=np.asarray(low_weights, dtype=self.dtype),
        )

    def _ensure_region_nodes(self, regions: list[_Region], stats: _CallStats) -> None:
        pending_regions = [region for region in regions if region.node_ids is None]
        if not pending_regions:
            return

        all_points = np.concatenate([region.node_coords for region in pending_regions], axis=0)
        node_ids = self._register_nodes(all_points, stats)

        start = 0
        for region in pending_regions:
            stop = start + region.node_coords.shape[0]
            region.node_ids = node_ids[start:stop]
            start = stop

    def _register_nodes(self, points: np.ndarray, stats: _CallStats) -> np.ndarray:
        points = self._canonicalize_points(points)
        if points.shape[0] == 0:
            return np.empty(0, dtype=np.int64)

        node_ids = np.empty(points.shape[0], dtype=np.int64)
        unique_missing_keys: list[bytes] = []
        unique_missing_points: list[np.ndarray] = []
        unresolved: list[tuple[int, bytes]] = []
        pending: dict[bytes, int] = {}

        for idx, point in enumerate(points):
            key = point.tobytes()
            existing = self._node_index.get(key)
            if existing is not None:
                node_ids[idx] = existing
                continue

            if key not in pending:
                pending[key] = len(unique_missing_points)
                unique_missing_keys.append(key)
                unique_missing_points.append(point.copy())
            unresolved.append((idx, key))

        if unique_missing_points:
            payloads = self._call_kernel(np.stack(unique_missing_points, axis=0), stats)
            for key, point, payload in zip(unique_missing_keys, unique_missing_points, payloads, strict=True):
                node_id = len(self._node_coords)
                self._node_index[key] = node_id
                self._node_coords.append(point)
                self._payloads.append(np.array(payload, copy=True))

        for idx, key in unresolved:
            node_ids[idx] = self._node_index[key]

        return node_ids

    def _canonicalize_points(self, points: np.ndarray) -> np.ndarray:
        arr = np.asarray(points, dtype=self.dtype)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2 or arr.shape[1] != self.ndim:
            raise ValueError(f"points must have shape (npoints, {self.ndim})")
        arr = np.where(arr == 0, self.dtype.type(0), arr)
        return np.ascontiguousarray(arr)

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

    def _call_evaluator(
        self,
        points: np.ndarray,
        payloads: np.ndarray,
        params: Any,
        stats: _CallStats,
    ) -> np.ndarray:
        outputs: list[np.ndarray] = []
        for chunk_points, chunk_payloads in zip(
            self._iter_chunks(points), self._iter_chunks(payloads), strict=True
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

    def _evaluate_regions(
        self,
        regions: list[_Region],
        params: Any,
        stats: _CallStats,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        active_ids = sorted({int(node_id) for region in regions for node_id in region.node_ids})
        coords = np.stack([self._node_coords[node_id] for node_id in active_ids], axis=0)
        payloads = np.stack([self._payloads[node_id] for node_id in active_ids], axis=0)
        values = self._call_evaluator(coords, payloads, params, stats)

        lookup = np.full(len(self._node_coords), -1, dtype=np.int64)
        lookup[np.asarray(active_ids, dtype=np.int64)] = np.arange(len(active_ids), dtype=np.int64)

        total_estimate: np.ndarray | np.generic | None = None
        total_error: np.ndarray | np.generic | None = None

        for region in regions:
            region_values = values[lookup[region.node_ids]]
            region.estimate, region.error = self._estimate_region(region, region_values)

            if total_estimate is None:
                total_estimate = np.array(region.estimate, copy=True)
                total_error = np.array(region.error, copy=True)
            else:
                total_estimate = total_estimate + region.estimate
                total_error = total_error + region.error

        if total_estimate is None or total_error is None:
            raise RuntimeError("integrator has no active regions")
        return total_estimate, total_error, self._is_finite_result(total_estimate, total_error)

    def _estimate_region(
        self,
        region: _Region,
        region_values: np.ndarray,
    ) -> tuple[np.ndarray | np.generic, np.ndarray | np.generic]:
        high_estimate = np.tensordot(region.high_weights, region_values, axes=(0, 0))
        low_estimate = np.tensordot(region.low_weights, region_values, axes=(0, 0))

        if self.ndim == 1 and self.rule_name in {"gk15", "gk21"}:
            return high_estimate, self._gauss_kronrod_error(region, region_values, high_estimate, low_estimate)
        if self.ndim == 1 and self.rule_name == "trapezoid":
            return high_estimate, np.abs(high_estimate - low_estimate) / 3.0

        return high_estimate, np.abs(high_estimate - low_estimate)

    def _gauss_kronrod_error(
        self,
        region: _Region,
        region_values: np.ndarray,
        high_estimate: np.ndarray | np.generic,
        low_estimate: np.ndarray | np.generic,
    ) -> np.ndarray | np.generic:
        h = 0.5 * float(region.b[0] - region.a[0])
        err = np.abs(high_estimate - low_estimate)
        if h == 0.0:
            return err

        reshape = (region.high_weights.shape[0],) + (1,) * (region_values.ndim - 1)
        weighted_high = region.high_weights.reshape(reshape)
        center_value = high_estimate / (2.0 * h)
        dabs = np.sum(weighted_high * np.abs(region_values - center_value), axis=0)
        ratio = np.divide(
            200.0 * err,
            dabs,
            out=np.zeros_like(np.asarray(err, dtype=np.result_type(err, np.float64))),
            where=dabs != 0,
        )
        err = np.where((dabs != 0) & (err != 0), dabs * np.minimum(1.0, ratio**1.5), err)

        abs_integrand = np.sum(weighted_high * np.abs(region_values), axis=0)
        round_err = np.abs(50.0 * np.finfo(self.dtype).eps * abs_integrand)
        err = np.where(round_err > np.finfo(self.dtype).tiny, np.maximum(err, round_err), err)
        return err

    def _is_finite_result(self, estimate: np.ndarray | np.generic, error: np.ndarray | np.generic) -> bool:
        return bool(np.all(np.isfinite(np.asarray(estimate))) and np.all(np.isfinite(np.asarray(error))))

    def _converged(self, estimate: np.ndarray, error: np.ndarray, *, atol: float, rtol: float) -> bool:
        tolerance = atol + rtol * np.abs(estimate)
        return bool(np.all(error <= tolerance))

    def _refine_worst_region(self) -> tuple[_Region, list[_Region]]:
        worst_index = max(range(len(self._leaf_regions)), key=lambda idx: self._leaf_regions[idx].priority)
        worst_region = self._leaf_regions.pop(worst_index)
        midpoint = 0.5 * (worst_region.a + worst_region.b)

        children = []
        for side in itertools.product((0, 1), repeat=self.ndim):
            child_a = np.array(worst_region.a, copy=True)
            child_b = np.array(worst_region.b, copy=True)
            for axis, upper_half in enumerate(side):
                if upper_half:
                    child_a[axis] = midpoint[axis]
                else:
                    child_b[axis] = midpoint[axis]
            children.append(self._build_region(child_a, child_b))

        self._leaf_regions.extend(children)
        return worst_region, children

    def _split_initial_regions(
        self,
        a: np.ndarray,
        b: np.ndarray,
        points: list[np.ndarray | list[float] | float] | None,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        if points is None or np.any(a == b):
            return [(a, b)]

        normalized_points = [self._normalize_point(point) for point in points]
        regions = [(a, b)]

        for point in normalized_points:
            new_regions: list[tuple[np.ndarray, np.ndarray]] = []
            for left, right in regions:
                if self._is_strictly_in_region(left, right, point):
                    for child_left, child_right in self._split_subregion(left, right, point):
                        if np.any(child_left == child_right):
                            continue
                        new_regions.append((child_left, child_right))
                else:
                    new_regions.append((left, right))
            regions = new_regions

        return regions

    def _normalize_point(self, point: np.ndarray | list[float] | float) -> np.ndarray:
        arr = np.asarray(point, dtype=self.dtype)
        if self.ndim == 1 and arr.ndim == 0:
            arr = arr.reshape(1)
        if arr.ndim != 1 or arr.shape[0] != self.ndim:
            raise ValueError(f"each point must have shape ({self.ndim},)")
        return np.ascontiguousarray(np.where(arr == 0, self.dtype.type(0), arr))

    def _is_strictly_in_region(self, a: np.ndarray, b: np.ndarray, point: np.ndarray) -> bool:
        if np.all(point == a) or np.all(point == b):
            return False
        return bool(np.all(a <= point) and np.all(point <= b))

    def _split_subregion(
        self,
        a: np.ndarray,
        b: np.ndarray,
        split_at: np.ndarray,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        left = [np.stack((a[i], split_at[i])) for i in range(self.ndim)]
        right = [np.stack((split_at[i], b[i])) for i in range(self.ndim)]

        a_sub = self._cartesian_product(left)
        b_sub = self._cartesian_product(right)
        return [(a_sub[i].copy(), b_sub[i].copy()) for i in range(a_sub.shape[0])]

    def _cartesian_product(self, arrays: list[np.ndarray]) -> np.ndarray:
        arrays_ix = np.meshgrid(*arrays, indexing="ij")
        return np.reshape(np.stack(arrays_ix, axis=-1), (-1, len(arrays)))

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
            n_regions=self.n_regions,
            n_cached_nodes=self.n_cached_nodes,
            subdivisions=stats.subdivisions,
        )

    def _coerce_result(self, value: np.ndarray | np.generic) -> Any:
        arr = np.asarray(value)
        if arr.ndim == 0:
            return arr.item()
        return arr
