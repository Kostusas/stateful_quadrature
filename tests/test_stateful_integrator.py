from __future__ import annotations

import math
import unittest

import numpy as np

from stateful_quadrature import StatefulIntegrator
from stateful_quadrature._rules import map_rule, resolve_rule

try:
    from scipy.integrate import cubature, quad_vec
    from scipy.integrate._quad_vec import _quadrature_gk15, _quadrature_gk21
    from scipy.integrate._rules import GaussKronrodQuadrature, GenzMalikCubature
except ImportError:  # pragma: no cover - exercised in the default pixi env
    cubature = None
    quad_vec = None
    _quadrature_gk15 = None
    _quadrature_gk21 = None
    GaussKronrodQuadrature = None
    GenzMalikCubature = None


class RuleTests(unittest.TestCase):
    def test_gauss_kronrod_polynomials_are_exact_after_remap(self) -> None:
        a = np.array([-1.7], dtype=np.float64)
        b = np.array([2.3], dtype=np.float64)

        for rule_name in ("gk15", "gk21"):
            rule = resolve_rule(rule_name, ndim=1, dtype=np.float64)
            nodes, high_weights, _ = map_rule(rule, a, b)
            x = nodes[:, 0]

            for degree in range(11):
                with self.subTest(rule=rule_name, degree=degree):
                    approx = np.dot(high_weights, x**degree)
                    exact = (b[0] ** (degree + 1) - a[0] ** (degree + 1)) / (degree + 1)
                    np.testing.assert_allclose(approx, exact, rtol=0.0, atol=5e-12)

    def test_genz_malik_preserves_constant_and_linear_moments(self) -> None:
        a = np.array([-0.2, 1.0], dtype=np.float64)
        b = np.array([1.3, 2.5], dtype=np.float64)
        rule = resolve_rule("genz_malik", ndim=2, dtype=np.float64)
        nodes, high_weights, _ = map_rule(rule, a, b)

        ones = np.ones(nodes.shape[0])
        volume = np.prod(b - a)
        x_moment = 0.5 * (b[0] ** 2 - a[0] ** 2) * (b[1] - a[1])
        y_moment = 0.5 * (b[1] ** 2 - a[1] ** 2) * (b[0] - a[0])

        np.testing.assert_allclose(np.dot(high_weights, ones), volume, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(np.dot(high_weights, nodes[:, 0]), x_moment, rtol=0.0, atol=1e-12)
        np.testing.assert_allclose(np.dot(high_weights, nodes[:, 1]), y_moment, rtol=0.0, atol=1e-12)


class ReuseTests(unittest.TestCase):
    def test_second_call_reuses_kernel_values_when_mesh_is_unchanged(self) -> None:
        def kernel(points: np.ndarray) -> np.ndarray:
            x = points[:, 0]
            return np.stack([np.sin(x), np.cos(x)], axis=-1)

        def evaluator(points: np.ndarray, payload: np.ndarray, params: tuple[float, float]) -> np.ndarray:
            alpha, beta = params
            return alpha * payload[:, 0] + beta * payload[:, 1]

        integrator = StatefulIntegrator(
            a=[-1.0],
            b=[1.0],
            kernel=kernel,
            evaluator=evaluator,
            rule="gk21",
            batch_size=8,
        )

        first = integrator.integrate((1.0, -0.25), atol=1e-10, rtol=1e-10)
        cached_after_first = first.n_cached_nodes
        second = integrator.integrate((0.5, 1.5), atol=1e-10, rtol=1e-10)

        np.testing.assert_allclose(first.estimate, -0.5 * math.sin(1.0), atol=1e-11, rtol=0.0)
        np.testing.assert_allclose(second.estimate, 3.0 * math.sin(1.0), atol=1e-11, rtol=0.0)
        self.assertEqual(second.n_kernel_evals, 0)
        self.assertEqual(second.n_cached_nodes, cached_after_first)
        self.assertGreater(second.n_evaluator_evals, 0)

    def test_second_call_only_adds_new_nodes_when_refinement_is_needed(self) -> None:
        def kernel(points: np.ndarray) -> np.ndarray:
            return points[:, 0]

        def evaluator(points: np.ndarray, payload: np.ndarray, omega: float) -> np.ndarray:
            return np.cos(omega * payload)

        integrator = StatefulIntegrator(
            a=[-1.0],
            b=[1.0],
            kernel=kernel,
            evaluator=evaluator,
            rule="gk15",
            batch_size=4,
        )

        easy = integrator.integrate(1.0, atol=1e-8, rtol=1e-8)
        cached_before = easy.n_cached_nodes
        hard = integrator.integrate(25.0, atol=1e-10, rtol=1e-10, max_subdivisions=512)

        self.assertEqual(hard.status, "converged")
        self.assertGreater(hard.n_kernel_evals, 0)
        self.assertEqual(hard.n_cached_nodes - cached_before, hard.n_kernel_evals)
        np.testing.assert_allclose(hard.estimate, 2.0 * math.sin(25.0) / 25.0, atol=1e-9, rtol=0.0)

    def test_complex_outputs_are_supported(self) -> None:
        def kernel(points: np.ndarray) -> np.ndarray:
            return points[:, 0]

        def evaluator(points: np.ndarray, payload: np.ndarray, omega: float) -> np.ndarray:
            return np.exp(1j * omega * payload)

        integrator = StatefulIntegrator(a=[0.0], b=[1.0], kernel=kernel, evaluator=evaluator, rule="gk21")
        result = integrator.integrate(2.5, atol=1e-11, rtol=1e-11)
        exact = (np.exp(2.5j) - 1.0) / (2.5j)
        np.testing.assert_allclose(result.estimate, exact, atol=1e-10, rtol=0.0)

    def test_batched_mean_field_style_kernel_reuses_eigensystems(self) -> None:
        def kernel(points: np.ndarray) -> np.ndarray:
            k = points[:, 0]
            hamiltonians = np.zeros((points.shape[0], 2, 2), dtype=np.float64)
            hopping = 1.0 + np.cos(k)
            hamiltonians[:, 0, 1] = hopping
            hamiltonians[:, 1, 0] = hopping
            eigenvalues, eigenvectors = np.linalg.eigh(hamiltonians)
            return np.concatenate(
                [eigenvalues.reshape(points.shape[0], -1), eigenvectors.reshape(points.shape[0], -1)],
                axis=-1,
            )

        def evaluator(
            points: np.ndarray, payload: np.ndarray, params: tuple[float, float]
        ) -> np.ndarray:
            mu, kT = params
            eigenvalues = payload[:, :2]
            occupation = 1.0 / (1.0 + np.exp((eigenvalues - mu) / kT))
            return np.sum(occupation, axis=-1) / (2.0 * np.pi)

        integrator = StatefulIntegrator(
            a=[-np.pi],
            b=[np.pi],
            kernel=kernel,
            evaluator=evaluator,
            rule="gk21",
            batch_size=16,
        )

        first = integrator.integrate((1.0, 0.2), atol=1e-9, rtol=1e-9)
        second = integrator.integrate((0.0, 0.2), atol=1e-9, rtol=1e-9)

        self.assertEqual(first.status, "converged")
        self.assertEqual(second.status, "converged")
        self.assertEqual(second.n_kernel_evals, 0)
        self.assertLess(second.estimate, first.estimate)


@unittest.skipUnless(cubature is not None and quad_vec is not None, "SciPy is not installed")
class SciPyComparisonTests(unittest.TestCase):
    def test_gk15_single_region_matches_scipy_internal_rule(self) -> None:
        a = np.array([-1.3], dtype=np.float64)
        b = np.array([2.1], dtype=np.float64)
        rule = resolve_rule("gk15", ndim=1, dtype=np.float64)
        nodes, high_weights, low_weights = map_rule(rule, a, b)
        x = nodes[:, 0]
        scipy_rule = GaussKronrodQuadrature(15)

        def f_scalar(value: float) -> float:
            return math.exp(-value * value) * (1.0 + value**3)

        ours_high = np.dot(high_weights, np.array([f_scalar(value) for value in x]))
        ours_low = np.dot(low_weights, np.array([f_scalar(value) for value in x]))

        scipy_high, _, _ = _quadrature_gk15(a[0], b[0], f_scalar, abs)
        scipy_low_nodes = ((scipy_rule.lower_nodes_and_weights[0] + 1.0) * (b - a) * 0.5 + a).reshape(-1)
        scipy_low_weights = scipy_rule.lower_nodes_and_weights[1] * ((b[0] - a[0]) / 2.0)
        scipy_low = np.dot(
            scipy_low_weights,
            np.array([f_scalar(value) for value in scipy_low_nodes]),
        )

        np.testing.assert_allclose(ours_high, scipy_high, rtol=0.0, atol=1e-14)
        np.testing.assert_allclose(ours_low, scipy_low, rtol=0.0, atol=1e-14)

    def test_gk21_single_region_matches_scipy_internal_rule(self) -> None:
        a = np.array([-0.7], dtype=np.float64)
        b = np.array([1.9], dtype=np.float64)
        rule = resolve_rule("gk21", ndim=1, dtype=np.float64)
        nodes, high_weights, low_weights = map_rule(rule, a, b)
        x = nodes[:, 0]
        scipy_rule = GaussKronrodQuadrature(21)

        def f_scalar(value: float) -> float:
            return math.sin(3.0 * value) + value**4 - 0.5 * value

        ours_high = np.dot(high_weights, np.array([f_scalar(value) for value in x]))
        ours_low = np.dot(low_weights, np.array([f_scalar(value) for value in x]))
        scipy_high, _, _ = _quadrature_gk21(a[0], b[0], f_scalar, abs)
        scipy_low_nodes = ((scipy_rule.lower_nodes_and_weights[0] + 1.0) * (b - a) * 0.5 + a).reshape(-1)
        scipy_low_weights = scipy_rule.lower_nodes_and_weights[1] * ((b[0] - a[0]) / 2.0)
        scipy_low = np.dot(
            scipy_low_weights,
            np.array([f_scalar(value) for value in scipy_low_nodes]),
        )

        np.testing.assert_allclose(ours_high, scipy_high, rtol=0.0, atol=1e-14)
        np.testing.assert_allclose(ours_low, scipy_low, rtol=0.0, atol=1e-14)

    def test_genz_malik_single_region_matches_scipy_internal_rule(self) -> None:
        a = np.array([-0.3, 0.2, -0.1], dtype=np.float64)
        b = np.array([0.8, 1.1, 0.4], dtype=np.float64)
        rule = resolve_rule("genz_malik", ndim=3, dtype=np.float64)
        nodes, high_weights, low_weights = map_rule(rule, a, b)
        scipy_rule = GenzMalikCubature(3)

        def f_batch(points: np.ndarray) -> np.ndarray:
            return np.exp(points[:, 0] - 0.5 * points[:, 1]) * (1.0 + points[:, 2] ** 2)

        ours_high = np.dot(high_weights, f_batch(nodes))
        ours_low = np.dot(low_weights, f_batch(nodes))
        scipy_high = scipy_rule.estimate(f_batch, a, b)
        scipy_low_nodes = (scipy_rule.lower_nodes_and_weights[0] + 1.0) * ((b - a) * 0.5) + a
        scipy_low_weights = scipy_rule.lower_nodes_and_weights[1] * np.prod(b - a) / 8.0
        scipy_low = np.dot(scipy_low_weights, f_batch(scipy_low_nodes))

        np.testing.assert_allclose(ours_high, scipy_high, rtol=0.0, atol=1e-13)
        np.testing.assert_allclose(ours_low, scipy_low, rtol=0.0, atol=1e-13)

    def test_matches_quad_vec_for_1d_array_output(self) -> None:
        def kernel(points: np.ndarray) -> np.ndarray:
            x = points[:, 0]
            return np.stack([np.sin(x), np.cos(x), x**2], axis=-1)

        def evaluator(points: np.ndarray, payload: np.ndarray, lam: float) -> np.ndarray:
            return np.stack(
                [payload[:, 0] + lam * payload[:, 1], payload[:, 2] - 0.25 * payload[:, 0]],
                axis=-1,
            )

        integrator = StatefulIntegrator(a=[-1.0], b=[2.0], kernel=kernel, evaluator=evaluator, rule="gk21")
        result = integrator.integrate(0.3, atol=1e-11, rtol=1e-11)

        def scipy_integrand(x: float) -> np.ndarray:
            return np.array([math.sin(x) + 0.3 * math.cos(x), x**2 - 0.25 * math.sin(x)])

        scipy_estimate, _ = quad_vec(scipy_integrand, -1.0, 2.0, epsabs=1e-11, epsrel=1e-11)
        np.testing.assert_allclose(result.estimate, scipy_estimate, atol=1e-10, rtol=1e-10)

    def test_matches_cubature_for_2d_scalar_output(self) -> None:
        a = np.array([-1.0, 0.0])
        b = np.array([0.5, 1.0])

        def kernel(points: np.ndarray) -> np.ndarray:
            return np.exp(points[:, 0] + 0.5 * points[:, 1])

        def evaluator(points: np.ndarray, payload: np.ndarray, scale: float) -> np.ndarray:
            return scale * payload * (1.0 + points[:, 0] * points[:, 1])

        integrator = StatefulIntegrator(a=a, b=b, kernel=kernel, evaluator=evaluator, rule="genz_malik")
        result = integrator.integrate(0.75, atol=1e-9, rtol=1e-9)

        def scipy_integrand(points: np.ndarray) -> np.ndarray:
            payload = np.exp(points[:, 0] + 0.5 * points[:, 1])
            return 0.75 * payload * (1.0 + points[:, 0] * points[:, 1])

        scipy_result = cubature(
            scipy_integrand,
            a,
            b,
            rule="genz-malik",
            atol=1e-9,
            rtol=1e-9,
        )
        np.testing.assert_allclose(result.estimate, scipy_result.estimate, atol=1e-8, rtol=1e-8)

    def test_matches_cubature_for_3d_complex_output(self) -> None:
        a = np.array([-0.2, 0.0, 0.1])
        b = np.array([0.4, 0.8, 0.9])

        def kernel(points: np.ndarray) -> np.ndarray:
            return np.prod(points + 1.0, axis=-1)

        def evaluator(points: np.ndarray, payload: np.ndarray, omega: float) -> np.ndarray:
            phase = np.exp(1j * omega * np.sum(points, axis=-1))
            return payload * phase

        integrator = StatefulIntegrator(a=a, b=b, kernel=kernel, evaluator=evaluator, rule="genz_malik")
        result = integrator.integrate(0.7, atol=1e-8, rtol=1e-8)

        def scipy_integrand(points: np.ndarray) -> np.ndarray:
            payload = np.prod(points + 1.0, axis=-1)
            values = payload * np.exp(1j * 0.7 * np.sum(points, axis=-1))
            return np.stack([values.real, values.imag], axis=-1)

        scipy_result = cubature(
            scipy_integrand,
            a,
            b,
            rule="genz-malik",
            atol=1e-8,
            rtol=1e-8,
        )
        scipy_estimate = scipy_result.estimate[0] + 1j * scipy_result.estimate[1]
        np.testing.assert_allclose(result.estimate, scipy_estimate, atol=5e-8, rtol=5e-8)
