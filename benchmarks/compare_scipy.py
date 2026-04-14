"""Benchmark repeated finite cubature sweeps against repeated SciPy cubature solves."""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import tempfile
import time
import tracemalloc
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Callable

import numpy as np

import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from stateful_quadrature import StatefulIntegrator

try:
    from scipy.integrate import cubature as scipy_cubature
except ImportError as exc:  # pragma: no cover - convenience script
    raise SystemExit("SciPy is required for this benchmark. Use `pixi run -e test python benchmarks/compare_scipy.py`.") from exc


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class CaseSpec:
    case_id: str
    label: str
    family: str
    params: dict[str, Any]


@dataclass(slots=True)
class BenchmarkRecord:
    case_id: str
    case_label: str
    impl: str
    median_seconds: float
    peak_python_mib: float
    metrics: dict[str, Any]
    estimate: Any
    delta_max_abs: float | None = None
    timing: dict[str, Any] | None = None
    overhead_breakdown: dict[str, float] | None = None


class KernelCounter:
    def __init__(self, func: Callable[[np.ndarray], np.ndarray]) -> None:
        self._func = func
        self.calls = 0
        self.points = 0

    def __call__(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points)
        self.calls += 1
        self.points += int(points.shape[0])
        return self._func(points)


class TimedKernel(KernelCounter):
    def __init__(self, func: Callable[[np.ndarray], np.ndarray]) -> None:
        super().__init__(func)
        self.seconds = 0.0

    def __call__(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points)
        self.calls += 1
        self.points += int(points.shape[0])
        start = time.perf_counter()
        values = self._func(points)
        self.seconds += time.perf_counter() - start
        return values


class EvaluatorCounter:
    def __init__(self, func: Callable[[np.ndarray, np.ndarray, Any], np.ndarray]) -> None:
        self._func = func
        self.calls = 0
        self.points = 0

    def __call__(self, points: np.ndarray, payload: np.ndarray, params: Any) -> np.ndarray:
        points = np.asarray(points)
        self.calls += 1
        self.points += int(points.shape[0])
        return self._func(points, payload, params)


class TimedEvaluator(EvaluatorCounter):
    def __init__(self, func: Callable[[np.ndarray, np.ndarray, Any], np.ndarray]) -> None:
        super().__init__(func)
        self.seconds = 0.0

    def __call__(self, points: np.ndarray, payload: np.ndarray, params: Any) -> np.ndarray:
        points = np.asarray(points)
        self.calls += 1
        self.points += int(points.shape[0])
        start = time.perf_counter()
        values = self._func(points, payload, params)
        self.seconds += time.perf_counter() - start
        return values


class BatchCounter:
    def __init__(self, func: Callable[[np.ndarray], np.ndarray]) -> None:
        self._func = func
        self.calls = 0
        self.points = 0

    def __call__(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points)
        self.calls += 1
        self.points += int(points.shape[0])
        return self._func(points)


class TimedIntegrand(BatchCounter):
    def __init__(self, func: Callable[[np.ndarray], np.ndarray]) -> None:
        super().__init__(func)
        self.seconds = 0.0

    def __call__(self, points: np.ndarray) -> np.ndarray:
        points = np.asarray(points)
        self.calls += 1
        self.points += int(points.shape[0])
        start = time.perf_counter()
        values = self._func(points)
        self.seconds += time.perf_counter() - start
        return values


class TimedStatefulIntegrator(StatefulIntegrator):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.bucket_seconds = {
            "_ensure_leaf_payloads_seconds": 0.0,
            "_refresh_leaves_seconds": 0.0,
            "leaf_geometry_seconds": 0.0,
            "map_nodes_batch_seconds": 0.0,
            "estimate_leaf_batch_seconds": 0.0,
            "priority_from_error_seconds": 0.0,
            "split_worst_leaf_seconds": 0.0,
            "call_kernel_seconds": 0.0,
            "call_evaluator_seconds": 0.0,
        }
        self.bucket_child_seconds = {key: 0.0 for key in self.bucket_seconds}
        self._bucket_stack: list[str] = []

    def _time_bucket(self, bucket: str, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        parent_bucket = self._bucket_stack[-1] if self._bucket_stack else None
        self._bucket_stack.append(bucket)
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed = time.perf_counter() - start
            self._bucket_stack.pop()
            self.bucket_seconds[bucket] += elapsed
            if parent_bucket is not None:
                self.bucket_child_seconds[parent_bucket] += elapsed

    def _exclusive_seconds(self, bucket: str) -> float:
        return self.bucket_seconds[bucket] - self.bucket_child_seconds[bucket]

    def _ensure_leaf_payloads(self, leaf_ids: list[int], stats: Any) -> None:
        return self._time_bucket("_ensure_leaf_payloads_seconds", super()._ensure_leaf_payloads, leaf_ids, stats)

    def _refresh_leaves(
        self,
        leaf_ids: list[int],
        params: Any,
        stats: Any,
        *,
        rebuild_heap: bool,
    ) -> tuple[np.ndarray | np.generic, np.ndarray | np.generic, bool]:
        return self._time_bucket(
            "_refresh_leaves_seconds",
            super()._refresh_leaves,
            leaf_ids,
            params,
            stats,
            rebuild_heap=rebuild_heap,
        )

    def _leaf_geometry(self, leaves: list[Any]) -> tuple[np.ndarray, np.ndarray]:
        return self._time_bucket("leaf_geometry_seconds", super()._leaf_geometry, leaves)

    def _map_nodes_batch(self, centers: np.ndarray, half_widths: np.ndarray) -> np.ndarray:
        return self._time_bucket("map_nodes_batch_seconds", super()._map_nodes_batch, centers, half_widths)

    def _estimate_leaf_batch(
        self,
        half_widths: np.ndarray,
        scales: np.ndarray,
        values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._time_bucket(
            "estimate_leaf_batch_seconds",
            super()._estimate_leaf_batch,
            half_widths,
            scales,
            values,
        )

    def _priority_from_error(self, error: np.ndarray | np.generic) -> float:
        return self._time_bucket("priority_from_error_seconds", super()._priority_from_error, error)

    def _split_worst_leaf(self) -> tuple[Any, list[int]]:
        return self._time_bucket("split_worst_leaf_seconds", super()._split_worst_leaf)

    def _call_kernel(self, points: np.ndarray, stats: Any) -> np.ndarray:
        return self._time_bucket("call_kernel_seconds", super()._call_kernel, points, stats)

    def _call_evaluator(
        self,
        points: np.ndarray,
        payloads: np.ndarray,
        params: Any,
        stats: Any,
    ) -> np.ndarray:
        return self._time_bucket("call_evaluator_seconds", super()._call_evaluator, points, payloads, params, stats)

    def explicit_overhead_breakdown(
        self,
        *,
        kernel_callback_seconds: float,
        evaluator_callback_seconds: float,
        framework_overhead_seconds: float,
    ) -> dict[str, float]:
        breakdown = {
            "_refresh_leaves_self_seconds": self._exclusive_seconds("_refresh_leaves_seconds"),
            "_ensure_leaf_payloads_self_seconds": self._exclusive_seconds("_ensure_leaf_payloads_seconds"),
            "leaf_geometry_seconds": self._exclusive_seconds("leaf_geometry_seconds"),
            "map_nodes_batch_seconds": self._exclusive_seconds("map_nodes_batch_seconds"),
            "estimate_leaf_batch_seconds": self._exclusive_seconds("estimate_leaf_batch_seconds"),
            "priority_from_error_seconds": self._exclusive_seconds("priority_from_error_seconds"),
            "split_worst_leaf_seconds": self._exclusive_seconds("split_worst_leaf_seconds"),
            "call_kernel_framework_seconds": self._exclusive_seconds("call_kernel_seconds") - kernel_callback_seconds,
            "call_evaluator_framework_seconds": self._exclusive_seconds("call_evaluator_seconds") - evaluator_callback_seconds,
        }
        explicit_seconds = sum(breakdown.values())
        breakdown["other_framework_seconds"] = framework_overhead_seconds - explicit_seconds
        return breakdown

    def instrumented_method_totals(self) -> dict[str, float]:
        return {
            "_ensure_leaf_payloads_seconds": self.bucket_seconds["_ensure_leaf_payloads_seconds"],
            "_refresh_leaves_seconds": self.bucket_seconds["_refresh_leaves_seconds"],
        }


def _jsonify(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonify(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    return value


def _estimate_delta(lhs: Any, rhs: Any) -> float:
    return float(np.max(np.abs(np.asarray(lhs) - np.asarray(rhs))))


def _median_run_index(samples: list[float]) -> int:
    ordered = sorted(range(len(samples)), key=lambda index: (samples[index], index))
    return ordered[len(ordered) // 2]


def _measure(func: Callable[[], dict[str, Any]], repeats: int) -> dict[str, Any]:
    samples: list[float] = []
    payloads: list[dict[str, Any]] = []
    peak_python_bytes = 0

    for _ in range(repeats):
        tracemalloc.start()
        start = time.perf_counter()
        payload = func()
        samples.append(time.perf_counter() - start)
        payloads.append(payload)
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_python_bytes = max(peak_python_bytes, peak_bytes)

    if not payloads:
        raise RuntimeError("benchmark did not produce a payload")

    median_index = _median_run_index(samples)
    return {
        "median_seconds": statistics.median(samples),
        "peak_python_mib": peak_python_bytes / (1024.0 * 1024.0),
        "payload": payloads[median_index],
        "sample_seconds": samples,
    }


def _phase_sweep_stateful(case: CaseSpec) -> dict[str, Any]:
    params = np.asarray(case.params["samples"], dtype=np.float64)
    a = np.asarray(case.params["a"], dtype=np.float64)
    b = np.asarray(case.params["b"], dtype=np.float64)
    rtol = float(case.params["rtol"])
    atol = float(case.params["atol"])

    def kernel(points: np.ndarray) -> np.ndarray:
        return points

    def evaluator(points: np.ndarray, payload: np.ndarray, omega: float) -> np.ndarray:
        phase = 0.9 * payload[:, 0] - 0.4 * payload[:, 1]
        return np.cos(omega * phase)

    kernel_counter = KernelCounter(kernel)
    evaluator_counter = EvaluatorCounter(evaluator)
    integrator = StatefulIntegrator(a=a, b=b, kernel=kernel_counter, evaluator=evaluator_counter, batch_size=136)

    estimates = []
    kernel_evals = 0
    evaluator_evals = 0
    subdivisions = 0
    for omega in params:
        result = integrator.integrate(float(omega), atol=atol, rtol=rtol, max_subdivisions=512)
        estimates.append(result.estimate)
        kernel_evals += result.n_kernel_evals
        evaluator_evals += result.n_evaluator_evals
        subdivisions += result.subdivisions

    return {
        "estimate": np.asarray(estimates),
        "metrics": {
            "kernel_calls": kernel_counter.calls,
            "kernel_point_evals": kernel_counter.points,
            "evaluator_calls": evaluator_counter.calls,
            "evaluator_point_evals": evaluator_counter.points,
            "kernel_evals_reported": kernel_evals,
            "evaluator_evals_reported": evaluator_evals,
            "subdivisions_total": subdivisions,
            "leaves_final": integrator.n_leaves,
            "leaf_nodes_final": integrator.n_leaf_nodes,
        },
    }


def _phase_sweep_scipy(case: CaseSpec) -> dict[str, Any]:
    params = np.asarray(case.params["samples"], dtype=np.float64)
    a = np.asarray(case.params["a"], dtype=np.float64)
    b = np.asarray(case.params["b"], dtype=np.float64)
    rtol = float(case.params["rtol"])
    atol = float(case.params["atol"])

    estimates = []
    counter_calls = 0
    counter_points = 0
    subdivisions = 0
    for omega in params:
        counter = BatchCounter(lambda points, omega=omega: np.cos(omega * (0.9 * points[:, 0] - 0.4 * points[:, 1])))
        result = scipy_cubature(counter, a, b, rule="genz-malik", atol=atol, rtol=rtol)
        estimates.append(result.estimate)
        counter_calls += counter.calls
        counter_points += counter.points
        subdivisions += result.subdivisions

    return {
        "estimate": np.asarray(estimates),
        "metrics": {
            "batch_calls_total": counter_calls,
            "point_evals_total": counter_points,
            "subdivisions_total": subdivisions,
            "samples": len(params),
        },
    }


def _phase_sweep_overhead_stateful(case: CaseSpec) -> dict[str, Any]:
    params = np.asarray(case.params["samples"], dtype=np.float64)
    a = np.asarray(case.params["a"], dtype=np.float64)
    b = np.asarray(case.params["b"], dtype=np.float64)
    rtol = float(case.params["rtol"])
    atol = float(case.params["atol"])
    batch_size = int(case.params["batch_size"])

    def kernel(points: np.ndarray) -> np.ndarray:
        return points

    def evaluator(points: np.ndarray, payload: np.ndarray, omega: float) -> np.ndarray:
        phase = 0.9 * payload[:, 0] - 0.4 * payload[:, 1]
        return np.cos(omega * phase)

    kernel_counter = TimedKernel(kernel)
    evaluator_counter = TimedEvaluator(evaluator)
    integrator = TimedStatefulIntegrator(
        a=a,
        b=b,
        kernel=kernel_counter,
        evaluator=evaluator_counter,
        batch_size=batch_size,
    )

    estimates = []
    sample_timings = []
    status_counts: dict[str, int] = {}
    kernel_evals = 0
    evaluator_evals = 0
    subdivisions = 0
    for omega in params:
        kernel_seconds_before = kernel_counter.seconds
        evaluator_seconds_before = evaluator_counter.seconds
        start = time.perf_counter()
        result = integrator.integrate(float(omega), atol=atol, rtol=rtol, max_subdivisions=512)
        total_seconds = time.perf_counter() - start
        kernel_seconds = kernel_counter.seconds - kernel_seconds_before
        evaluator_seconds = evaluator_counter.seconds - evaluator_seconds_before
        callback_seconds = kernel_seconds + evaluator_seconds
        framework_overhead_seconds = total_seconds - callback_seconds

        estimates.append(result.estimate)
        kernel_evals += result.n_kernel_evals
        evaluator_evals += result.n_evaluator_evals
        subdivisions += result.subdivisions
        status_counts[result.status] = status_counts.get(result.status, 0) + 1
        sample_timings.append(
            {
                "omega": float(omega),
                "total_seconds": total_seconds,
                "callback_seconds": callback_seconds,
                "framework_overhead_seconds": framework_overhead_seconds,
                "kernel_callback_seconds": kernel_seconds,
                "evaluator_callback_seconds": evaluator_seconds,
                "kernel_point_evals": result.n_kernel_evals,
                "evaluator_point_evals": result.n_evaluator_evals,
                "subdivisions": result.subdivisions,
                "status": result.status,
                "leaves": result.n_leaves,
                "leaf_nodes": result.n_leaf_nodes,
            }
        )

    total_seconds = sum(item["total_seconds"] for item in sample_timings)
    callback_seconds = sum(item["callback_seconds"] for item in sample_timings)
    framework_overhead_seconds = total_seconds - callback_seconds
    total_callback_points = kernel_counter.points + evaluator_counter.points
    warm_totals = [item["total_seconds"] for item in sample_timings[1:]]
    breakdown = integrator.explicit_overhead_breakdown(
        kernel_callback_seconds=kernel_counter.seconds,
        evaluator_callback_seconds=evaluator_counter.seconds,
        framework_overhead_seconds=framework_overhead_seconds,
    )
    breakdown_percent = {
        key: (100.0 * value / framework_overhead_seconds if framework_overhead_seconds else 0.0)
        for key, value in breakdown.items()
    }

    return {
        "estimate": np.asarray(estimates),
        "metrics": {
            "samples": len(params),
            "status_counts": dict(sorted(status_counts.items())),
            "kernel_calls": kernel_counter.calls,
            "kernel_point_evals": kernel_counter.points,
            "evaluator_calls": evaluator_counter.calls,
            "evaluator_point_evals": evaluator_counter.points,
            "kernel_evals_reported": kernel_evals,
            "evaluator_evals_reported": evaluator_evals,
            "subdivisions_total": subdivisions,
            "leaves_final": integrator.n_leaves,
            "leaf_nodes_final": integrator.n_leaf_nodes,
        },
        "timing": {
            "total_seconds": total_seconds,
            "callback_seconds": callback_seconds,
            "framework_overhead_seconds": framework_overhead_seconds,
            "first_call_seconds": sample_timings[0]["total_seconds"],
            "warm_median_seconds": statistics.median(warm_totals) if warm_totals else sample_timings[0]["total_seconds"],
            "warm_mean_seconds": statistics.mean(warm_totals) if warm_totals else sample_timings[0]["total_seconds"],
            "framework_overhead_per_1k_points": (
                1000.0 * framework_overhead_seconds / total_callback_points if total_callback_points else 0.0
            ),
            "callback_points_total": total_callback_points,
            "sample_count": len(sample_timings),
            "per_sample": sample_timings,
            "overhead_breakdown_percent": breakdown_percent,
            "instrumented_method_totals": integrator.instrumented_method_totals(),
        },
        "overhead_breakdown": breakdown,
    }


def _phase_sweep_overhead_scipy(case: CaseSpec) -> dict[str, Any]:
    params = np.asarray(case.params["samples"], dtype=np.float64)
    a = np.asarray(case.params["a"], dtype=np.float64)
    b = np.asarray(case.params["b"], dtype=np.float64)
    rtol = float(case.params["rtol"])
    atol = float(case.params["atol"])

    sample_timings = []
    estimates = []
    status_counts: dict[str, int] = {}
    counter_calls = 0
    counter_points = 0
    subdivisions = 0
    for omega in params:
        counter = TimedIntegrand(
            lambda points, omega=omega: np.cos(omega * (0.9 * points[:, 0] - 0.4 * points[:, 1]))
        )
        start = time.perf_counter()
        result = scipy_cubature(counter, a, b, rule="genz-malik", atol=atol, rtol=rtol)
        total_seconds = time.perf_counter() - start
        framework_overhead_seconds = total_seconds - counter.seconds

        estimates.append(result.estimate)
        counter_calls += counter.calls
        counter_points += counter.points
        subdivisions += result.subdivisions
        status_counts[result.status] = status_counts.get(result.status, 0) + 1
        sample_timings.append(
            {
                "omega": float(omega),
                "total_seconds": total_seconds,
                "callback_seconds": counter.seconds,
                "framework_overhead_seconds": framework_overhead_seconds,
                "point_evals": counter.points,
                "batch_calls": counter.calls,
                "subdivisions": result.subdivisions,
                "status": result.status,
            }
        )

    total_seconds = sum(item["total_seconds"] for item in sample_timings)
    callback_seconds = sum(item["callback_seconds"] for item in sample_timings)
    framework_overhead_seconds = total_seconds - callback_seconds
    warm_totals = [item["total_seconds"] for item in sample_timings[1:]]

    return {
        "estimate": np.asarray(estimates),
        "metrics": {
            "samples": len(params),
            "status_counts": dict(sorted(status_counts.items())),
            "batch_calls_total": counter_calls,
            "point_evals_total": counter_points,
            "subdivisions_total": subdivisions,
        },
        "timing": {
            "total_seconds": total_seconds,
            "callback_seconds": callback_seconds,
            "framework_overhead_seconds": framework_overhead_seconds,
            "first_call_seconds": sample_timings[0]["total_seconds"],
            "warm_median_seconds": statistics.median(warm_totals) if warm_totals else sample_timings[0]["total_seconds"],
            "warm_mean_seconds": statistics.mean(warm_totals) if warm_totals else sample_timings[0]["total_seconds"],
            "framework_overhead_per_1k_points": (
                1000.0 * framework_overhead_seconds / counter_points if counter_points else 0.0
            ),
            "callback_points_total": counter_points,
            "sample_count": len(sample_timings),
            "per_sample": sample_timings,
        },
    }


def _mean_field_sweep_stateful(case: CaseSpec) -> dict[str, Any]:
    params = np.asarray(case.params["samples"], dtype=np.float64)
    a = np.asarray(case.params["a"], dtype=np.float64)
    b = np.asarray(case.params["b"], dtype=np.float64)
    rtol = float(case.params["rtol"])
    atol = float(case.params["atol"])
    k_t = float(case.params["k_t"])

    def kernel(points: np.ndarray) -> np.ndarray:
        npoints = points.shape[0]
        hamiltonians = np.zeros((npoints, 2, 2), dtype=np.float64)
        hopping = 1.0 + 0.25 * np.cos(points[:, 0]) + 0.5 * np.sin(points[:, 1]) - 0.2 * points[:, 2]
        onsite = 0.3 * points[:, 0] - 0.2 * points[:, 1] + 0.1 * points[:, 2]
        hamiltonians[:, 0, 0] = onsite
        hamiltonians[:, 1, 1] = -onsite
        hamiltonians[:, 0, 1] = hopping
        hamiltonians[:, 1, 0] = hopping
        eigenvalues, eigenvectors = np.linalg.eigh(hamiltonians)
        return np.concatenate(
            [eigenvalues.reshape(npoints, -1), eigenvectors.reshape(npoints, -1)],
            axis=-1,
        )

    def evaluator(points: np.ndarray, payload: np.ndarray, mu: float) -> np.ndarray:
        eigenvalues = payload[:, :2]
        occupation = 1.0 / (1.0 + np.exp((eigenvalues - mu) / k_t))
        return np.sum(occupation, axis=-1)

    kernel_counter = KernelCounter(kernel)
    evaluator_counter = EvaluatorCounter(evaluator)
    integrator = StatefulIntegrator(a=a, b=b, kernel=kernel_counter, evaluator=evaluator_counter, batch_size=136)

    estimates = []
    kernel_evals = 0
    evaluator_evals = 0
    subdivisions = 0
    for mu in params:
        result = integrator.integrate(float(mu), atol=atol, rtol=rtol, max_subdivisions=512)
        estimates.append(result.estimate)
        kernel_evals += result.n_kernel_evals
        evaluator_evals += result.n_evaluator_evals
        subdivisions += result.subdivisions

    return {
        "estimate": np.asarray(estimates),
        "metrics": {
            "kernel_calls": kernel_counter.calls,
            "kernel_point_evals": kernel_counter.points,
            "evaluator_calls": evaluator_counter.calls,
            "evaluator_point_evals": evaluator_counter.points,
            "kernel_evals_reported": kernel_evals,
            "evaluator_evals_reported": evaluator_evals,
            "subdivisions_total": subdivisions,
            "leaves_final": integrator.n_leaves,
            "leaf_nodes_final": integrator.n_leaf_nodes,
        },
    }


def _mean_field_sweep_scipy(case: CaseSpec) -> dict[str, Any]:
    params = np.asarray(case.params["samples"], dtype=np.float64)
    a = np.asarray(case.params["a"], dtype=np.float64)
    b = np.asarray(case.params["b"], dtype=np.float64)
    rtol = float(case.params["rtol"])
    atol = float(case.params["atol"])
    k_t = float(case.params["k_t"])

    def direct_integrand(points: np.ndarray, mu: float) -> np.ndarray:
        npoints = points.shape[0]
        hamiltonians = np.zeros((npoints, 2, 2), dtype=np.float64)
        hopping = 1.0 + 0.25 * np.cos(points[:, 0]) + 0.5 * np.sin(points[:, 1]) - 0.2 * points[:, 2]
        onsite = 0.3 * points[:, 0] - 0.2 * points[:, 1] + 0.1 * points[:, 2]
        hamiltonians[:, 0, 0] = onsite
        hamiltonians[:, 1, 1] = -onsite
        hamiltonians[:, 0, 1] = hopping
        hamiltonians[:, 1, 0] = hopping
        eigenvalues, _ = np.linalg.eigh(hamiltonians)
        occupation = 1.0 / (1.0 + np.exp((eigenvalues - mu) / k_t))
        return np.sum(occupation, axis=-1)

    estimates = []
    counter_calls = 0
    counter_points = 0
    subdivisions = 0
    for mu in params:
        counter = BatchCounter(lambda points, mu=mu: direct_integrand(points, float(mu)))
        result = scipy_cubature(counter, a, b, rule="genz-malik", atol=atol, rtol=rtol)
        estimates.append(result.estimate)
        counter_calls += counter.calls
        counter_points += counter.points
        subdivisions += result.subdivisions

    return {
        "estimate": np.asarray(estimates),
        "metrics": {
            "batch_calls_total": counter_calls,
            "point_evals_total": counter_points,
            "subdivisions_total": subdivisions,
            "samples": len(params),
        },
    }


RUNNERS = {
    ("phase_sweep_2d", "stateful"): _phase_sweep_stateful,
    ("phase_sweep_2d", "scipy"): _phase_sweep_scipy,
    ("phase_sweep_2d_overhead", "stateful"): _phase_sweep_overhead_stateful,
    ("phase_sweep_2d_overhead", "scipy"): _phase_sweep_overhead_scipy,
    ("mean_field_sweep_3d", "stateful"): _mean_field_sweep_stateful,
    ("mean_field_sweep_3d", "scipy"): _mean_field_sweep_scipy,
}


def _build_cases(mode: str) -> list[CaseSpec]:
    if mode == "overhead-2d":
        return [
            CaseSpec(
                case_id="phase_sweep_2d_overhead",
                label="2D phase sweep overhead",
                family="phase_sweep_2d_overhead",
                params={
                    "a": np.array([0.0, 0.0]),
                    "b": np.array([1.0, 1.0]),
                    "samples": np.linspace(6.0, 12.0, 41),
                    "atol": 1e-8,
                    "rtol": 1e-8,
                    "batch_size": 136,
                },
            )
        ]

    cases = [
        CaseSpec(
            case_id="phase_sweep_2d",
            label="2D phase sweep",
            family="phase_sweep_2d",
            params={
                "a": np.array([0.0, 0.0]),
                "b": np.array([1.0, 1.0]),
                "samples": np.linspace(6.0, 12.0, 9),
                "atol": 1e-8,
                "rtol": 1e-8,
            },
        ),
        CaseSpec(
            case_id="mean_field_sweep_3d",
            label="3D mean-field sweep",
            family="mean_field_sweep_3d",
            params={
                "a": np.array([-np.pi, 0.0, -1.0]),
                "b": np.array([np.pi, np.pi, 1.0]),
                "samples": np.linspace(-0.6, 0.6, 17),
                "atol": 1e-8,
                "rtol": 1e-8,
                "k_t": 0.15,
            },
        ),
    ]

    if mode == "full":
        cases.append(
            CaseSpec(
                case_id="mean_field_sweep_3d_dense",
                label="3D mean-field sweep (dense)",
                family="mean_field_sweep_3d",
                params={
                    "a": np.array([-np.pi, 0.0, -1.0]),
                    "b": np.array([np.pi, np.pi, 1.0]),
                    "samples": np.linspace(-0.8, 0.8, 25),
                    "atol": 1e-9,
                    "rtol": 1e-9,
                    "k_t": 0.1,
                },
            )
        )

    return cases


def _run_case(case: CaseSpec, impl: str, repeats: int) -> BenchmarkRecord:
    measurement = _measure(lambda: RUNNERS[(case.family, impl)](case), repeats)
    payload = measurement["payload"]
    return BenchmarkRecord(
        case_id=case.case_id,
        case_label=case.label,
        impl=impl,
        median_seconds=float(measurement["median_seconds"]),
        peak_python_mib=float(measurement["peak_python_mib"]),
        metrics=_jsonify(payload["metrics"]),
        estimate=_jsonify(payload["estimate"]),
        timing=_jsonify(payload.get("timing")),
        overhead_breakdown=_jsonify(payload.get("overhead_breakdown")),
    )


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3e}"
    if isinstance(value, dict):
        parts = [f"{key}:{_format_value(val)}" for key, val in sorted(value.items())]
        return "{" + ", ".join(parts) + "}"
    return str(value)


def _format_metrics(metrics: dict[str, Any]) -> str:
    parts = []
    for key in sorted(metrics):
        parts.append(f"{key}={_format_value(metrics[key])}")
    return ", ".join(parts)


def _format_status_counts(metrics: dict[str, Any]) -> str:
    status_counts = metrics.get("status_counts", {})
    if not isinstance(status_counts, dict):
        return "-"
    return ",".join(f"{key}:{value}" for key, value in sorted(status_counts.items()))


def _format_overhead_summary(record: BenchmarkRecord) -> str:
    metrics = record.metrics
    if record.impl == "stateful":
        return (
            f"status={_format_status_counts(metrics)}, subdivisions={metrics['subdivisions_total']}, "
            f"kernel_points={metrics['kernel_point_evals']}, evaluator_points={metrics['evaluator_point_evals']}, "
            f"leaves_final={metrics['leaves_final']}"
        )
    return (
        f"status={_format_status_counts(metrics)}, subdivisions={metrics['subdivisions_total']}, "
        f"point_evals={metrics['point_evals_total']}, batch_calls={metrics['batch_calls_total']}"
    )


def _render_default_markdown(records: list[BenchmarkRecord], output_dir: pathlib.Path) -> str:
    lines = [
        "# Stateful Benchmark Results",
        "",
        f"- Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"- Output dir: `{output_dir.relative_to(REPO_ROOT)}`",
        "",
        "| Case | Impl | Time (s) | Python Peak (MiB) | Delta | Metrics |",
        "| --- | --- | ---: | ---: | ---: | --- |",
    ]

    for record in records:
        delta = "-" if record.delta_max_abs is None else f"{record.delta_max_abs:.3e}"
        lines.append(
            f"| {record.case_label} | {record.impl} | {record.median_seconds:.6f} | {record.peak_python_mib:.2f} | {delta} | {_format_metrics(record.metrics)} |"
        )

    return "\n".join(lines) + "\n"


def _render_overhead_markdown(records: list[BenchmarkRecord], output_dir: pathlib.Path) -> str:
    lines = [
        "# Stateful Benchmark Results",
        "",
        f"- Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"- Output dir: `{output_dir.relative_to(REPO_ROOT)}`",
        "- Timing rows below use the representative median-duration run for payload details.",
        "",
        "## Overhead Summary",
        "",
        "| Case | Impl | Median Time (s) | Callback (s) | Framework Overhead (s) | First Call (s) | Warm Median (s) | Warm Mean (s) | Overhead / 1k points | Delta | Summary |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    for record in records:
        if record.timing is None:
            raise RuntimeError("overhead markdown requires timing payloads")
        delta = "-" if record.delta_max_abs is None else f"{record.delta_max_abs:.3e}"
        lines.append(
            "| "
            + " | ".join(
                [
                    record.case_label,
                    record.impl,
                    f"{record.median_seconds:.6f}",
                    f"{record.timing['callback_seconds']:.6f}",
                    f"{record.timing['framework_overhead_seconds']:.6f}",
                    f"{record.timing['first_call_seconds']:.6f}",
                    f"{record.timing['warm_median_seconds']:.6f}",
                    f"{record.timing['warm_mean_seconds']:.6f}",
                    f"{record.timing['framework_overhead_per_1k_points'] * 1e3:.3f} ms",
                    delta,
                    _format_overhead_summary(record),
                ]
            )
            + " |"
        )

    stateful_record = next((record for record in records if record.overhead_breakdown is not None), None)
    if stateful_record is not None and stateful_record.timing is not None:
        framework_overhead_seconds = float(stateful_record.timing["framework_overhead_seconds"])
        lines.extend(
            [
                "",
                "## Stateful Overhead Breakdown",
                "",
                "| Bucket | Seconds | Share |",
                "| --- | ---: | ---: |",
            ]
        )
        for key, value in sorted(stateful_record.overhead_breakdown.items(), key=lambda item: item[1], reverse=True):
            percent = 100.0 * float(value) / framework_overhead_seconds if framework_overhead_seconds else 0.0
            lines.append(f"| {key} | {float(value):.6f} | {percent:.1f}% |")

        method_totals = stateful_record.timing.get("instrumented_method_totals", {})
        if method_totals:
            lines.extend(
                [
                    "",
                    "## Stateful Method Totals",
                    "",
                    "| Method | Inclusive Seconds | Exclusive Self Seconds | Share |",
                    "| --- | ---: | ---: | ---: |",
                ]
            )
            method_rows = [
                ("_refresh_leaves", method_totals.get("_refresh_leaves_seconds"), stateful_record.overhead_breakdown.get("_refresh_leaves_self_seconds")),
                ("_ensure_leaf_payloads", method_totals.get("_ensure_leaf_payloads_seconds"), stateful_record.overhead_breakdown.get("_ensure_leaf_payloads_self_seconds")),
            ]
            for label, inclusive, exclusive in method_rows:
                if inclusive is None or exclusive is None:
                    continue
                percent = 100.0 * float(exclusive) / framework_overhead_seconds if framework_overhead_seconds else 0.0
                lines.append(f"| {label} | {float(inclusive):.6f} | {float(exclusive):.6f} | {percent:.1f}% |")

    return "\n".join(lines) + "\n"


def _render_markdown(records: list[BenchmarkRecord], output_dir: pathlib.Path) -> str:
    if records and all(record.timing is not None for record in records):
        return _render_overhead_markdown(records, output_dir)
    return _render_default_markdown(records, output_dir)


def _create_output_dir(requested: str | None) -> pathlib.Path:
    if requested is not None:
        output_dir = (REPO_ROOT / requested).resolve() if not pathlib.Path(requested).is_absolute() else pathlib.Path(requested)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    temp_root = REPO_ROOT / "tmp"
    temp_root.mkdir(parents=True, exist_ok=True)
    return pathlib.Path(tempfile.mkdtemp(prefix="benchmark_", dir=temp_root))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3, help="timing repeats per case")
    parser.add_argument("--mode", choices=("quick", "full", "overhead-2d"), default="quick", help="benchmark intensity")
    parser.add_argument("--output-dir", type=str, default=None, help="directory for benchmark artifacts")
    args = parser.parse_args()

    output_dir = _create_output_dir(args.output_dir)
    records: list[BenchmarkRecord] = []

    for case in _build_cases(args.mode):
        stateful_record = _run_case(case, "stateful", repeats=args.repeats)
        scipy_record = _run_case(case, "scipy", repeats=args.repeats)
        delta = _estimate_delta(stateful_record.estimate, scipy_record.estimate)
        stateful_record.delta_max_abs = delta
        scipy_record.delta_max_abs = delta
        records.extend([stateful_record, scipy_record])

    markdown = _render_markdown(records, output_dir)
    (output_dir / "results.md").write_text(markdown)
    (output_dir / "results.json").write_text(json.dumps([asdict(record) for record in records], indent=2))
    print(markdown)


if __name__ == "__main__":
    main()
