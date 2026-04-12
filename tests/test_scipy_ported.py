from __future__ import annotations

import itertools
import math
import unittest

import numpy as np

from stateful_quadrature import StatefulIntegrator


def _gm_f1(points: np.ndarray, r: float, alphas: np.ndarray) -> np.ndarray:
    return np.cos(2.0 * math.pi * r + np.sum(points * alphas, axis=-1))


def _gm_f1_exact(a: np.ndarray, b: np.ndarray, r: float, alphas: np.ndarray) -> float:
    ndim = a.shape[0]
    return (
        (-2.0) ** ndim
        / np.prod(alphas)
        * np.cos(2.0 * math.pi * r + np.sum(alphas * (a + b) * 0.5))
        * np.prod(np.sin(alphas * (a - b) / 2.0))
    )


def _gm_f2(points: np.ndarray, alphas: np.ndarray, betas: np.ndarray) -> np.ndarray:
    return 1.0 / np.prod(alphas**2 + (points - betas) ** 2, axis=-1)


def _gm_f2_exact(a: np.ndarray, b: np.ndarray, alphas: np.ndarray, betas: np.ndarray) -> float:
    ndim = a.shape[0]
    return (
        (-1.0) ** ndim
        / np.prod(alphas)
        * np.prod(np.arctan((a - betas) / alphas) - np.arctan((b - betas) / alphas))
    )


def _gm_f3(points: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    return np.exp(np.sum(alphas * points, axis=-1))


def _gm_f3_exact(a: np.ndarray, b: np.ndarray, alphas: np.ndarray) -> float:
    ndim = a.shape[0]
    return (-1.0) ** ndim / np.prod(alphas) * np.prod(np.exp(alphas * a) - np.exp(alphas * b))


def _gm_f4(points: np.ndarray, alphas: np.ndarray) -> np.ndarray:
    ndim = points.shape[-1]
    return (1.0 + np.sum(alphas * points, axis=-1)) ** (-ndim - 1)


def _gm_f4_exact(a: np.ndarray, b: np.ndarray, alphas: np.ndarray) -> float:
    ndim = a.shape[0]

    def antiderivative(x: np.ndarray) -> float:
        return (-1.0) ** ndim / np.prod(alphas) / math.factorial(ndim) / (1.0 + np.sum(alphas * x))

    total = 0.0
    corners = np.stack([a, b], axis=0)
    for index in itertools.product(range(2), repeat=ndim):
        point = np.array([corners[i, axis] for axis, i in enumerate(index)], dtype=np.float64)
        total += (-1.0) ** (sum(index) + ndim) * antiderivative(point)
    return total


def _gm_f5(points: np.ndarray, alphas: np.ndarray, betas: np.ndarray) -> np.ndarray:
    return np.exp(-np.sum(alphas**2 * (points - betas) ** 2, axis=-1))


def _gm_f5_exact(a: np.ndarray, b: np.ndarray, alphas: np.ndarray, betas: np.ndarray) -> float:
    ndim = a.shape[0]
    erf_terms = np.array(
        [math.erf(alphas[i] * (betas[i] - a[i])) + math.erf(alphas[i] * (b[i] - betas[i])) for i in range(ndim)],
        dtype=np.float64,
    )
    return (0.5**ndim) * (math.pi ** (ndim / 2.0)) / np.prod(alphas) * np.prod(erf_terms)


class PortedQuadVecTests(unittest.TestCase):
    def test_simple_polynomial_family_matches_scipy_style_cases(self) -> None:
        n = np.arange(10, dtype=np.float64)
        exact = 2.0 ** (n + 1.0) / (n + 1.0)

        def kernel(points: np.ndarray) -> np.ndarray:
            return points[:, 0]

        def evaluator(points: np.ndarray, payload: np.ndarray, powers: np.ndarray) -> np.ndarray:
            return payload[:, None] ** powers[None, :]

        for rule in ("gk15", "gk21"):
            for atol in (1e-1, 1e-3, 1e-6):
                with self.subTest(rule=rule, atol=atol, split=False):
                    integrator = StatefulIntegrator(a=[0.0], b=[2.0], kernel=kernel, evaluator=evaluator, rule=rule)
                    result = integrator.integrate(n, atol=atol, rtol=0.0)
                    np.testing.assert_allclose(result.estimate, exact, rtol=0.0, atol=atol)

                with self.subTest(rule=rule, atol=atol, split=True):
                    integrator = StatefulIntegrator(
                        a=[0.0],
                        b=[2.0],
                        kernel=kernel,
                        evaluator=evaluator,
                        rule=rule,
                        points=[0.5, 1.0],
                    )
                    result = integrator.integrate(n, atol=atol, rtol=0.0)
                    np.testing.assert_allclose(result.estimate, exact, rtol=0.0, atol=atol)

    def test_args_style_case_matches_scipy(self) -> None:
        scale = 2.0
        exact = np.array([0.0, 4.0 / 3.0, 8.0 / 3.0])

        def kernel(points: np.ndarray) -> np.ndarray:
            return points[:, 0]

        def evaluator(points: np.ndarray, payload: np.ndarray, a: float) -> np.ndarray:
            return payload[:, None] * (payload + a)[:, None] * np.arange(3.0)[None, :]

        integrator = StatefulIntegrator(a=[0.0], b=[1.0], kernel=kernel, evaluator=evaluator, rule="gk21")
        result = integrator.integrate(scale, atol=1e-8, rtol=0.0)
        np.testing.assert_allclose(result.estimate, exact, rtol=0.0, atol=1e-8)

    def test_break_points_split_initial_regions_like_scipy(self) -> None:
        recorded: list[float] = []

        def kernel(points: np.ndarray) -> np.ndarray:
            recorded.extend(points[:, 0].tolist())
            return points[:, 0]

        def evaluator(points: np.ndarray, payload: np.ndarray, _: None) -> np.ndarray:
            return np.zeros(points.shape[0], dtype=np.float64)

        integrator = StatefulIntegrator(
            a=[0.0],
            b=[1.0],
            kernel=kernel,
            evaluator=evaluator,
            rule="gk15",
            points=[0.25, 0.5, 0.75],
        )
        result = integrator.integrate(None, atol=0.0, rtol=0.0, max_subdivisions=0)

        self.assertEqual(result.status, "converged")
        intervals = [(0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0)]
        self.assertEqual(len(recorded), 15 * len(intervals))
        for group_index, (left, right) in enumerate(intervals):
            group = recorded[group_index * 15 : (group_index + 1) * 15]
            self.assertTrue(all(left < value < right for value in group))


class PortedCubatureInterfaceTests(unittest.TestCase):
    def test_zero_width_limits_return_zero_with_correct_shape(self) -> None:
        n = np.arange(5, dtype=np.float64)

        def kernel(points: np.ndarray) -> np.ndarray:
            return points[:, 0]

        def evaluator(points: np.ndarray, payload: np.ndarray, powers: np.ndarray) -> np.ndarray:
            return payload[:, None] ** powers[None, :]

        integrator = StatefulIntegrator(a=[0.0], b=[0.0], kernel=kernel, evaluator=evaluator, rule="gk21")
        result = integrator.integrate(n, atol=0.0, rtol=0.0)
        np.testing.assert_allclose(result.estimate, np.zeros_like(n), rtol=0.0, atol=0.0)

    def test_reversed_limits_flip_the_sign(self) -> None:
        n = np.arange(5, dtype=np.float64)
        exact = -(2.0 ** (n + 1.0) / (n + 1.0))

        def kernel(points: np.ndarray) -> np.ndarray:
            return points[:, 0]

        def evaluator(points: np.ndarray, payload: np.ndarray, powers: np.ndarray) -> np.ndarray:
            return payload[:, None] ** powers[None, :]

        integrator = StatefulIntegrator(a=[2.0], b=[0.0], kernel=kernel, evaluator=evaluator, rule="gk21")
        result = integrator.integrate(n, atol=1e-10, rtol=0.0)
        np.testing.assert_allclose(result.estimate, exact, rtol=0.0, atol=1e-10)


class PortedGenzMalikProblemTests(unittest.TestCase):
    def test_finite_domain_benchmark_cases(self) -> None:
        problems = [
            (
                _gm_f1,
                _gm_f1_exact,
                np.array([0.0, 0.0]),
                np.array([1.0, 1.0]),
                (0.25, np.array([2.0, 4.0])),
            ),
            (
                _gm_f2,
                _gm_f2_exact,
                np.array([0.0, 0.0, 0.0]),
                np.array([1.0, 1.0, 1.0]),
                (np.array([2.0, 3.0, 4.0]), np.array([2.0, 3.0, 4.0])),
            ),
            (
                _gm_f3,
                _gm_f3_exact,
                np.array([-1.0, -1.0, -1.0]),
                np.array([1.0, 1.0, 1.0]),
                (np.array([1.0, 1.0, 1.0]),),
            ),
            (
                _gm_f4,
                _gm_f4_exact,
                np.array([0.0, 0.0, 0.0]),
                np.array([1.0, 1.0, 1.0]),
                (np.array([1.0, 1.0, 1.0]),),
            ),
            (
                _gm_f5,
                _gm_f5_exact,
                np.array([-1.0, -1.0, -1.0]),
                np.array([1.0, 1.0, 1.0]),
                (np.array([1.0, 1.0, 1.0]), np.array([1.0, 1.0, 1.0])),
            ),
        ]

        def kernel(points: np.ndarray) -> np.ndarray:
            return points

        for function, exact_function, a, b, args in problems:
            with self.subTest(function=function.__name__, ndim=a.shape[0]):
                def evaluator(points: np.ndarray, payload: np.ndarray, packed_args: tuple[object, ...]) -> np.ndarray:
                    return function(payload, *packed_args)

                integrator = StatefulIntegrator(
                    a=a,
                    b=b,
                    kernel=kernel,
                    evaluator=evaluator,
                    rule="genz_malik",
                )
                result = integrator.integrate(args, atol=1e-5, rtol=1e-4)
                exact = exact_function(a, b, *args)

                self.assertEqual(result.status, "converged")
                np.testing.assert_allclose(result.estimate, exact, rtol=1e-4, atol=1e-5)
