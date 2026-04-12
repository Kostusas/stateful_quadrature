"""SciPy-inspired benchmark harness for quadrature and cubature."""

from __future__ import annotations

import argparse
import pathlib
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from stateful_quadrature import StatefulIntegrator, cubature, quad_vec

try:
    from scipy.integrate import cubature as scipy_cubature
    from scipy.integrate import quad_vec as scipy_quad_vec
except ImportError as exc:  # pragma: no cover - convenience script
    raise SystemExit("SciPy is required for this benchmark. Use `pixi run -e test python benchmarks/compare_scipy.py`.") from exc


@dataclass(slots=True)
class Timing:
    seconds: float
    payload: Any


def _median_time(func: Callable[[], Any], repeats: int, warmups: int = 1) -> Timing:
    payload = None
    for _ in range(warmups):
        payload = func()

    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        payload = func()
        samples.append(time.perf_counter() - start)

    return Timing(seconds=statistics.median(samples), payload=payload)


def _print_row(case: str, impl: str, seconds: float, metric: str, delta: float | None) -> None:
    delta_text = "-" if delta is None else f"{delta:.3e}"
    print(f"{case:24} {impl:12} {seconds:10.4f}s  {metric:18}  delta={delta_text}")


def _quad_vec_oscillatory_case(fdim: int, rtol: float, repeats: int) -> None:
    case_name = f"quad_vec(fdim={fdim})"

    def integrand(x: float) -> np.ndarray:
        r = np.repeat(0.5, fdim)
        alphas = np.repeat(0.1, fdim)
        return np.cos(2.0 * np.pi * r + alphas * x)

    ours = _median_time(
        lambda: quad_vec(integrand, 0.0, 1.0, epsabs=0.0, epsrel=rtol, full_output=True),
        repeats=repeats,
    )
    scipy = _median_time(
        lambda: scipy_quad_vec(integrand, 0.0, 1.0, epsabs=0.0, epsrel=rtol, full_output=True),
        repeats=repeats,
    )

    our_estimate, _, our_info = ours.payload
    scipy_estimate, _, scipy_info = scipy.payload
    delta = float(np.max(np.abs(np.asarray(our_estimate) - np.asarray(scipy_estimate))))

    _print_row(case_name, "ours", ours.seconds, f"subdiv={our_info.intervals.shape[0]}", delta)
    _print_row(case_name, "scipy", scipy.seconds, f"subdiv={scipy_info.intervals.shape[0]}", None)


def _cubature_sphere_case(rule: str, rtol: float, repeats: int) -> None:
    case_name = f"sphere({rule})"
    a = np.array([0.0, 0.0, 0.0])
    b = np.array([1.0, 2.0 * np.pi, np.pi])

    def integrand(points: np.ndarray) -> np.ndarray:
        radius = points[:, 0]
        phi = points[:, 2]
        return radius**2 * np.sin(phi)

    ours = _median_time(
        lambda: cubature(integrand, a, b, rule=rule, rtol=rtol, atol=0.0),
        repeats=repeats,
    )
    scipy = _median_time(
        lambda: scipy_cubature(integrand, a, b, rule=rule, rtol=rtol, atol=0.0),
        repeats=repeats,
    )

    delta = float(np.max(np.abs(np.asarray(ours.payload.estimate) - np.asarray(scipy.payload.estimate))))
    _print_row(case_name, "ours", ours.seconds, f"subdiv={ours.payload.subdivisions}", delta)
    _print_row(case_name, "scipy", scipy.seconds, f"subdiv={scipy.payload.subdivisions}", None)


def _cubature_oscillatory_case(rule: str, ndim: int, fdim: int, rtol: float, repeats: int) -> None:
    case_name = f"cubature({rule},n={ndim},m={fdim})"
    a = np.zeros(ndim)
    b = np.ones(ndim)

    def integrand(points: np.ndarray) -> np.ndarray:
        npoints = points.shape[0]
        r = np.repeat(0.5, fdim)
        alphas = np.repeat(0.1, fdim * ndim).reshape(fdim, ndim)
        reshaped = points.reshape(npoints, 1, ndim)
        return np.cos(2.0 * np.pi * r + np.sum(alphas[None, :, :] * reshaped, axis=-1))

    ours = _median_time(
        lambda: cubature(integrand, a, b, rule=rule, rtol=rtol, atol=0.0),
        repeats=repeats,
    )
    scipy = _median_time(
        lambda: scipy_cubature(integrand, a, b, rule=rule, rtol=rtol, atol=0.0),
        repeats=repeats,
    )

    delta = float(np.max(np.abs(np.asarray(ours.payload.estimate) - np.asarray(scipy.payload.estimate))))
    _print_row(case_name, "ours", ours.seconds, f"subdiv={ours.payload.subdivisions}", delta)
    _print_row(case_name, "scipy", scipy.seconds, f"subdiv={scipy.payload.subdivisions}", None)


def _stateful_sweep_case(samples: int, repeats: int) -> None:
    case_name = f"stateful-sweep({samples})"
    params = np.linspace(-1.0, 1.0, samples)

    def kernel(points: np.ndarray) -> np.ndarray:
        x = points[:, 0]
        return np.stack([np.sin(x), np.cos(x), np.exp(-x**2)], axis=-1)

    def evaluator(points: np.ndarray, payload: np.ndarray, lam: float) -> np.ndarray:
        return payload[:, 2] * (payload[:, 0] + lam * payload[:, 1])

    def run_stateful() -> tuple[np.ndarray, int, int]:
        integrator = StatefulIntegrator(a=[-3.0], b=[3.0], kernel=kernel, evaluator=evaluator, rule="gk21")
        estimates = []
        kernel_evals = 0
        for lam in params:
            result = integrator.integrate(lam, atol=1e-9, rtol=1e-9)
            estimates.append(result.estimate)
            kernel_evals += result.n_kernel_evals
        return np.asarray(estimates), kernel_evals, integrator.n_cached_nodes

    def run_scipy() -> tuple[np.ndarray, int]:
        estimates = []
        neval = 0
        for lam in params:
            result, _, info = scipy_quad_vec(
                lambda x, lam=lam: np.exp(-x**2) * (np.sin(x) + lam * np.cos(x)),
                -3.0,
                3.0,
                epsabs=1e-9,
                epsrel=1e-9,
                full_output=True,
            )
            estimates.append(result)
            neval += info.neval
        return np.asarray(estimates), neval

    stateful = _median_time(run_stateful, repeats=repeats)
    scipy = _median_time(run_scipy, repeats=repeats)

    stateful_estimates, stateful_kernel_evals, cached_nodes = stateful.payload
    scipy_estimates, scipy_neval = scipy.payload
    delta = float(np.max(np.abs(stateful_estimates - scipy_estimates)))

    _print_row(case_name, "stateful", stateful.seconds, f"k={stateful_kernel_evals}, cache={cached_nodes}", delta)
    _print_row(case_name, "scipy", scipy.seconds, f"neval={scipy_neval}", None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3, help="timing repeats per case")
    parser.add_argument(
        "--mode",
        choices=("quick", "full"),
        default="quick",
        help="quick runs a compact SciPy-inspired suite; full adds heavier oscillatory cases",
    )
    args = parser.parse_args()

    print("Case                     Impl             Median Time  Metric              Notes")
    print("-" * 88)

    _quad_vec_oscillatory_case(fdim=1, rtol=1e-10, repeats=args.repeats)
    _quad_vec_oscillatory_case(fdim=8, rtol=1e-10, repeats=args.repeats)

    for rule in ("gk15", "gk21", "genz-malik"):
        _cubature_sphere_case(rule=rule, rtol=1e-9, repeats=args.repeats)

    oscillatory_cases = [
        ("genz-malik", 3, 1, 1e-10),
        ("gk15", 3, 1, 1e-10),
        ("gk21", 3, 1, 1e-10),
    ]
    if args.mode == "full":
        oscillatory_cases.extend(
            [
                ("genz-malik", 5, 1, 1e-10),
                ("genz-malik", 3, 8, 1e-10),
            ]
        )

    for rule, ndim, fdim, rtol in oscillatory_cases:
        _cubature_oscillatory_case(rule=rule, ndim=ndim, fdim=fdim, rtol=rtol, repeats=args.repeats)

    _stateful_sweep_case(samples=21, repeats=args.repeats)


if __name__ == "__main__":  # pragma: no cover - convenience script
    main()
