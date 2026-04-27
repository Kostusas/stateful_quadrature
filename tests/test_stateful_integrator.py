from __future__ import annotations

import math
import unittest

import numpy as np

from stateful_quadrature import StatefulIntegrator
from stateful_quadrature._rules import map_rule, resolve_rule


def _complex_linear_antiderivative(x: np.ndarray | float, shift: float, omega: float) -> np.ndarray:
    a = 1j * omega
    return np.exp(a * x) * ((x + shift) / a - 1.0 / (a * a))


class RuleTests(unittest.TestCase):
    def test_gk21_polynomials_are_exact_after_remap(self) -> None:
        a = np.array([-1.7], dtype=np.float64)
        b = np.array([2.3], dtype=np.float64)
        rule = resolve_rule("gk21", ndim=1, dtype=np.float64)
        nodes, high_weights, _ = map_rule(rule, a, b)
        x = nodes[:, 0]

        for degree in range(11):
            with self.subTest(degree=degree):
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


class InterfaceTests(unittest.TestCase):
    def test_auto_rule_selects_supported_rules(self) -> None:
        one_d = StatefulIntegrator(a=[0.0], b=[1.0], kernel=lambda x: x[:, 0], evaluator=lambda x, p, _: p)
        two_d = StatefulIntegrator(a=[0.0, 0.0], b=[1.0, 1.0], kernel=lambda x: x, evaluator=lambda x, p, _: p[:, 0])

        self.assertEqual(one_d.rule_name, "gk21")
        self.assertEqual(two_d.rule_name, "genz_malik")

    def test_removed_rule_variants_raise(self) -> None:
        with self.assertRaises(ValueError):
            StatefulIntegrator(a=[0.0], b=[1.0], kernel=lambda x: x[:, 0], evaluator=lambda x, p, _: p, rule="gk15")

        with self.assertRaises(ValueError):
            StatefulIntegrator(a=[0.0], b=[1.0], kernel=lambda x: x[:, 0], evaluator=lambda x, p, _: p, rule="trapezoid")

        with self.assertRaises(ValueError):
            StatefulIntegrator(
                a=[0.0],
                b=[1.0],
                kernel=lambda x: x[:, 0],
                evaluator=lambda x, p, _: p,
                rule="genz_malik",
            )

    def test_points_are_no_longer_supported(self) -> None:
        with self.assertRaises(TypeError):
            StatefulIntegrator(
                a=[0.0],
                b=[1.0],
                kernel=lambda x: x[:, 0],
                evaluator=lambda x, p, _: p,
                points=[0.5],
            )

    def test_zero_width_limits_return_zero_with_correct_shape(self) -> None:
        powers = np.arange(4.0)

        def kernel(points: np.ndarray) -> np.ndarray:
            return points[:, 0]

        def evaluator(points: np.ndarray, payload: np.ndarray, terms: np.ndarray) -> np.ndarray:
            return payload[:, None] ** terms[None, :]

        integrator = StatefulIntegrator(a=[0.0], b=[0.0], kernel=kernel, evaluator=evaluator)
        result = integrator.integrate(powers, atol=0.0, rtol=0.0)

        np.testing.assert_allclose(result.estimate, np.zeros_like(powers), rtol=0.0, atol=0.0)
        np.testing.assert_allclose(result.error, np.zeros_like(powers), rtol=0.0, atol=0.0)

    def test_reversed_limits_flip_the_sign(self) -> None:
        powers = np.arange(5.0)
        exact = -(2.0 ** (powers + 1.0) / (powers + 1.0))

        def kernel(points: np.ndarray) -> np.ndarray:
            return points[:, 0]

        def evaluator(points: np.ndarray, payload: np.ndarray, terms: np.ndarray) -> np.ndarray:
            return payload[:, None] ** terms[None, :]

        integrator = StatefulIntegrator(a=[2.0], b=[0.0], kernel=kernel, evaluator=evaluator)
        result = integrator.integrate(powers, atol=1e-10, rtol=0.0)
        np.testing.assert_allclose(result.estimate, exact, rtol=0.0, atol=1e-10)

    def test_integrate_accepts_keyword_parameters(self) -> None:
        def kernel(points: np.ndarray) -> np.ndarray:
            return points[:, 0]

        def evaluator(points: np.ndarray, payload: np.ndarray, params: dict[str, float]) -> np.ndarray:
            return params["scale"] * payload + params["shift"]

        integrator = StatefulIntegrator(a=[0.0], b=[1.0], kernel=kernel, evaluator=evaluator)
        result = integrator.integrate(scale=2.0, shift=1.0, atol=1e-10, rtol=1e-10)

        np.testing.assert_allclose(result.estimate, 2.0, rtol=0.0, atol=1e-10)

    def test_integrate_rejects_mixed_params_and_keywords(self) -> None:
        integrator = StatefulIntegrator(
            a=[0.0],
            b=[1.0],
            kernel=lambda points: points[:, 0],
            evaluator=lambda points, payload, params: payload,
        )

        with self.assertRaises(ValueError):
            integrator.integrate({"scale": 2.0}, scale=2.0)

    def test_invalid_tolerances_and_subdivision_limits_raise(self) -> None:
        integrator = StatefulIntegrator(
            a=[0.0],
            b=[1.0],
            kernel=lambda points: points[:, 0],
            evaluator=lambda points, payload, params: payload,
        )

        with self.assertRaises(ValueError):
            integrator.integrate(atol=-1.0)

        with self.assertRaises(ValueError):
            integrator.integrate(rtol=-1.0)

        with self.assertRaises(ValueError):
            integrator.integrate(max_subdivisions=-1)

    def test_non_finite_evaluator_output_returns_not_finite(self) -> None:
        def kernel(points: np.ndarray) -> np.ndarray:
            return points[:, 0]

        def evaluator(points: np.ndarray, payload: np.ndarray, params: object) -> np.ndarray:
            return np.full(points.shape[0], np.inf)

        integrator = StatefulIntegrator(a=[0.0], b=[1.0], kernel=kernel, evaluator=evaluator)
        with np.errstate(invalid="ignore"):
            result = integrator.integrate(atol=0.0, rtol=0.0)

        self.assertEqual(result.status, "not_finite")
        self.assertTrue(np.isinf(result.estimate))

    def test_kernel_output_shape_must_stay_fixed_within_a_call(self) -> None:
        call_shapes = [1, 2]

        def kernel(points: np.ndarray) -> np.ndarray:
            width = call_shapes.pop(0) if call_shapes else 2
            return np.ones((points.shape[0], width))

        integrator = StatefulIntegrator(
            a=[0.0],
            b=[1.0],
            kernel=kernel,
            evaluator=lambda points, payload, params: payload[:, 0],
            batch_size=10,
        )

        with self.assertRaises(ValueError):
            integrator.integrate(atol=0.0, rtol=0.0)

    def test_evaluator_output_shape_must_stay_fixed_within_a_call(self) -> None:
        call_shapes = [0, 1]

        def evaluator(points: np.ndarray, payload: np.ndarray, params: object) -> np.ndarray:
            width = call_shapes.pop(0) if call_shapes else 1
            if width == 0:
                return payload
            return payload[:, None]

        integrator = StatefulIntegrator(
            a=[0.0],
            b=[1.0],
            kernel=lambda points: points[:, 0],
            evaluator=evaluator,
            batch_size=10,
        )

        with self.assertRaises(ValueError):
            integrator.integrate(atol=0.0, rtol=0.0)


class AccuracyTests(unittest.TestCase):
    def test_1d_tensor_output_matches_exact(self) -> None:
        powers = np.arange(6.0)
        exact = 1.0 / (powers + 1.0)

        def kernel(points: np.ndarray) -> np.ndarray:
            return points[:, 0]

        def evaluator(points: np.ndarray, payload: np.ndarray, terms: np.ndarray) -> np.ndarray:
            return payload[:, None] ** terms[None, :]

        integrator = StatefulIntegrator(a=[0.0], b=[1.0], kernel=kernel, evaluator=evaluator, batch_size=8)
        result = integrator.integrate(powers, atol=1e-10, rtol=1e-10)

        self.assertEqual(result.status, "converged")
        np.testing.assert_allclose(result.estimate, exact, rtol=0.0, atol=1e-10)

    def test_2d_scalar_output_matches_exact(self) -> None:
        a = np.array([-0.3, 0.1])
        b = np.array([0.7, 1.2])

        def kernel(points: np.ndarray) -> np.ndarray:
            return np.exp(points[:, 0]) * np.cos(points[:, 1])

        def evaluator(points: np.ndarray, payload: np.ndarray, scale: float) -> np.ndarray:
            return scale * payload

        exact = 1.5 * (math.exp(b[0]) - math.exp(a[0])) * (math.sin(b[1]) - math.sin(a[1]))
        integrator = StatefulIntegrator(a=a, b=b, kernel=kernel, evaluator=evaluator, batch_size=68)
        result = integrator.integrate(1.5, atol=1e-10, rtol=1e-10)

        self.assertEqual(result.status, "converged")
        np.testing.assert_allclose(result.estimate, exact, rtol=0.0, atol=1e-10)

    def test_3d_complex_output_matches_exact(self) -> None:
        a = np.array([-0.2, 0.0, 0.1])
        b = np.array([0.4, 0.8, 0.9])
        omega = 0.7

        def kernel(points: np.ndarray) -> np.ndarray:
            return np.prod(points + np.array([1.0, 2.0, 3.0]), axis=-1)

        def evaluator(points: np.ndarray, payload: np.ndarray, phase_omega: float) -> np.ndarray:
            return payload * np.exp(1j * phase_omega * np.sum(points, axis=-1))

        exact = (
            _complex_linear_antiderivative(b[0], 1.0, omega) - _complex_linear_antiderivative(a[0], 1.0, omega)
        )
        exact *= (
            _complex_linear_antiderivative(b[1], 2.0, omega) - _complex_linear_antiderivative(a[1], 2.0, omega)
        )
        exact *= (
            _complex_linear_antiderivative(b[2], 3.0, omega) - _complex_linear_antiderivative(a[2], 3.0, omega)
        )

        integrator = StatefulIntegrator(a=a, b=b, kernel=kernel, evaluator=evaluator, batch_size=136)
        result = integrator.integrate(omega, atol=1e-8, rtol=1e-8, max_subdivisions=512)

        self.assertEqual(result.status, "converged")
        np.testing.assert_allclose(result.estimate, exact, rtol=0.0, atol=5e-8)

    def test_readme_minimal_example_converges(self) -> None:
        def kernel(points: np.ndarray) -> np.ndarray:
            x = points[:, 0]
            return np.stack([np.sin(x), np.cos(x)], axis=-1)

        def evaluator(points: np.ndarray, payload: np.ndarray, params: dict[str, float]) -> np.ndarray:
            return payload[:, 0] + params["alpha"] * payload[:, 1]

        integrator = StatefulIntegrator(
            a=[-1.0],
            b=[1.0],
            kernel=kernel,
            evaluator=evaluator,
            rule="gk21",
        )

        result = integrator.integrate(alpha=1.5, atol=1e-10, rtol=1e-10)

        self.assertEqual(result.status, "converged")
        np.testing.assert_allclose(result.estimate, 3.0 * math.sin(1.0), rtol=0.0, atol=1e-10)


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
            batch_size=8,
        )

        first = integrator.integrate((1.0, -0.25), atol=1e-10, rtol=1e-10)
        second = integrator.integrate((0.5, 1.5), atol=1e-10, rtol=1e-10)

        np.testing.assert_allclose(first.estimate, -0.5 * math.sin(1.0), atol=1e-11, rtol=0.0)
        np.testing.assert_allclose(second.estimate, 3.0 * math.sin(1.0), atol=1e-11, rtol=0.0)
        self.assertEqual(second.n_kernel_evals, 0)
        self.assertEqual(second.n_leaf_nodes, first.n_leaf_nodes)
        self.assertGreater(second.n_evaluator_evals, 0)

    def test_harder_followup_only_evaluates_new_child_leaves(self) -> None:
        rule_nodes = resolve_rule("gk21", ndim=1).n_nodes

        def kernel(points: np.ndarray) -> np.ndarray:
            return points[:, 0]

        def evaluator(points: np.ndarray, payload: np.ndarray, omega: float) -> np.ndarray:
            return np.cos(omega * payload)

        integrator = StatefulIntegrator(a=[-1.0], b=[1.0], kernel=kernel, evaluator=evaluator, batch_size=16)

        easy = integrator.integrate(1.0, atol=1e-8, rtol=1e-8)
        hard = integrator.integrate(25.0, atol=1e-10, rtol=1e-10, max_subdivisions=512)

        self.assertEqual(hard.status, "converged")
        self.assertGreater(hard.subdivisions, 0)
        self.assertEqual(hard.n_kernel_evals, 2 * rule_nodes * hard.subdivisions)
        self.assertEqual(hard.n_leaf_nodes - easy.n_leaf_nodes, rule_nodes * hard.subdivisions)
        np.testing.assert_allclose(hard.estimate, 2.0 * math.sin(25.0) / 25.0, atol=1e-9, rtol=0.0)

    def test_replace_evaluator_reuses_live_leaf_payloads_for_new_output_shape(self) -> None:
        def kernel(points: np.ndarray) -> np.ndarray:
            x = points[:, 0]
            return np.stack([np.sin(x), np.cos(x)], axis=-1)

        def charge_evaluator(points: np.ndarray, payload: np.ndarray, alpha: float) -> np.ndarray:
            return np.stack([payload[:, 0] + alpha * payload[:, 1], payload[:, 1]], axis=-1)

        def density_evaluator(points: np.ndarray, payload: np.ndarray, alpha: float) -> np.ndarray:
            values = payload[:, 0] + alpha * payload[:, 1]
            return values[:, None, None]

        charge_integrator = StatefulIntegrator(a=[-1.0], b=[1.0], kernel=kernel, evaluator=charge_evaluator)
        charge_result = charge_integrator.integrate(0.5, atol=1e-10, rtol=1e-10)

        density_integrator = charge_integrator.replace_evaluator(density_evaluator)
        density_result = density_integrator.integrate(0.5, atol=1e-10, rtol=1e-10)

        self.assertEqual(density_result.status, "converged")
        self.assertEqual(density_result.n_kernel_evals, 0)
        self.assertEqual(density_result.n_leaves, charge_result.n_leaves)
        self.assertEqual(density_result.n_leaf_nodes, charge_result.n_leaf_nodes)
        self.assertEqual(density_result.estimate.shape, (1, 1))

    def test_replace_evaluator_allows_independent_refinement_after_clone(self) -> None:
        def kernel(points: np.ndarray) -> np.ndarray:
            return points[:, 0]

        def first_evaluator(points: np.ndarray, payload: np.ndarray, omega: float) -> np.ndarray:
            return np.cos(omega * payload)

        def second_evaluator(points: np.ndarray, payload: np.ndarray, omega: float) -> np.ndarray:
            return np.stack([np.cos(omega * payload), np.sin(omega * payload)], axis=-1)

        first = StatefulIntegrator(a=[-1.0], b=[1.0], kernel=kernel, evaluator=first_evaluator)
        first.integrate(20.0, atol=1e-10, rtol=1e-10, max_subdivisions=256)
        second = first.replace_evaluator(second_evaluator)
        result = second.integrate(20.0, atol=1e-10, rtol=1e-10, max_subdivisions=256)

        self.assertEqual(result.status, "converged")
        self.assertEqual(result.n_kernel_evals, 0)
        self.assertEqual(result.estimate.shape[-1], 2)


class PreparedPayloadTests(unittest.TestCase):
    def test_prepared_payloads_support_basic_convergence(self) -> None:
        def kernel(points: np.ndarray) -> np.ndarray:
            return points[:, 0]

        def payload_builder(points: np.ndarray, raw_payloads: np.ndarray) -> list[dict[str, float]]:
            return [{"value": float(value)} for value in raw_payloads]

        def evaluator(points: np.ndarray, payloads: list[dict[str, float]], power: float) -> np.ndarray:
            return np.array([payload["value"] ** power for payload in payloads])

        integrator = StatefulIntegrator(
            a=[0.0],
            b=[1.0],
            kernel=kernel,
            evaluator=evaluator,
            batch_size=8,
            payload_builder=payload_builder,
        )
        result = integrator.integrate(2.0, atol=1e-10, rtol=1e-10)

        self.assertEqual(result.status, "converged")
        np.testing.assert_allclose(result.estimate, 1.0 / 3.0, rtol=0.0, atol=1e-10)

    def test_prepared_payload_builder_is_skipped_when_mesh_is_reused(self) -> None:
        builder_calls = 0
        built_points = 0

        def kernel(points: np.ndarray) -> np.ndarray:
            x = points[:, 0]
            return np.stack([np.sin(x), np.cos(x)], axis=-1)

        def payload_builder(points: np.ndarray, raw_payloads: np.ndarray) -> list[dict[str, float]]:
            nonlocal builder_calls, built_points
            builder_calls += 1
            built_points += points.shape[0]
            return [
                {"sin": float(raw_payload[0]), "cos": float(raw_payload[1])}
                for raw_payload in raw_payloads
            ]

        def evaluator(points: np.ndarray, payloads: list[dict[str, float]], params: tuple[float, float]) -> np.ndarray:
            alpha, beta = params
            return np.array([alpha * payload["sin"] + beta * payload["cos"] for payload in payloads])

        integrator = StatefulIntegrator(
            a=[-1.0],
            b=[1.0],
            kernel=kernel,
            evaluator=evaluator,
            batch_size=8,
            payload_builder=payload_builder,
        )

        first = integrator.integrate((1.0, -0.25), atol=1e-10, rtol=1e-10)
        builder_calls_after_first = builder_calls
        built_points_after_first = built_points
        second = integrator.integrate((0.5, 1.5), atol=1e-10, rtol=1e-10)

        np.testing.assert_allclose(first.estimate, -0.5 * math.sin(1.0), atol=1e-11, rtol=0.0)
        np.testing.assert_allclose(second.estimate, 3.0 * math.sin(1.0), atol=1e-11, rtol=0.0)
        self.assertEqual(second.n_kernel_evals, 0)
        self.assertEqual(builder_calls, builder_calls_after_first)
        self.assertEqual(built_points, built_points_after_first)

    def test_prepared_payload_builder_only_runs_for_new_child_leaves(self) -> None:
        rule_nodes = resolve_rule("gk21", ndim=1).n_nodes
        built_points = 0

        def kernel(points: np.ndarray) -> np.ndarray:
            return points[:, 0]

        def payload_builder(points: np.ndarray, raw_payloads: np.ndarray) -> list[dict[str, float]]:
            nonlocal built_points
            built_points += points.shape[0]
            return [{"value": float(value)} for value in raw_payloads]

        def evaluator(points: np.ndarray, payloads: list[dict[str, float]], omega: float) -> np.ndarray:
            return np.array([math.cos(omega * payload["value"]) for payload in payloads])

        integrator = StatefulIntegrator(
            a=[-1.0],
            b=[1.0],
            kernel=kernel,
            evaluator=evaluator,
            batch_size=16,
            payload_builder=payload_builder,
        )

        easy = integrator.integrate(1.0, atol=1e-8, rtol=1e-8)
        built_points_after_easy = built_points
        hard = integrator.integrate(25.0, atol=1e-10, rtol=1e-10, max_subdivisions=512)

        self.assertEqual(hard.status, "converged")
        self.assertGreater(hard.subdivisions, 0)
        self.assertEqual(hard.n_kernel_evals, 2 * rule_nodes * hard.subdivisions)
        self.assertEqual(hard.n_leaf_nodes - easy.n_leaf_nodes, rule_nodes * hard.subdivisions)
        self.assertEqual(built_points - built_points_after_easy, hard.n_kernel_evals)
        np.testing.assert_allclose(hard.estimate, 2.0 * math.sin(25.0) / 25.0, atol=1e-9, rtol=0.0)

    def test_replace_evaluator_reuses_live_prepared_payloads(self) -> None:
        def kernel(points: np.ndarray) -> np.ndarray:
            x = points[:, 0]
            return np.stack([np.sin(x), np.cos(x)], axis=-1)

        def payload_builder(points: np.ndarray, raw_payloads: np.ndarray) -> list[dict[str, float]]:
            return [
                {"sin": float(raw_payload[0]), "cos": float(raw_payload[1])}
                for raw_payload in raw_payloads
            ]

        def charge_evaluator(points: np.ndarray, payloads: list[dict[str, float]], alpha: float) -> np.ndarray:
            return np.array(
                [[payload["sin"] + alpha * payload["cos"], payload["cos"]] for payload in payloads]
            )

        def density_evaluator(points: np.ndarray, payloads: list[dict[str, float]], alpha: float) -> np.ndarray:
            values = np.array([payload["sin"] + alpha * payload["cos"] for payload in payloads])
            return values[:, None, None]

        charge_integrator = StatefulIntegrator(
            a=[-1.0],
            b=[1.0],
            kernel=kernel,
            evaluator=charge_evaluator,
            payload_builder=payload_builder,
        )
        charge_result = charge_integrator.integrate(0.5, atol=1e-10, rtol=1e-10)

        density_integrator = charge_integrator.replace_evaluator(density_evaluator)
        density_result = density_integrator.integrate(0.5, atol=1e-10, rtol=1e-10)

        self.assertEqual(density_result.status, "converged")
        self.assertEqual(density_result.n_kernel_evals, 0)
        self.assertEqual(density_result.n_leaves, charge_result.n_leaves)
        self.assertEqual(density_result.n_leaf_nodes, charge_result.n_leaf_nodes)
        self.assertEqual(density_result.estimate.shape, (1, 1))

    def test_prepared_payload_mutations_are_visible_after_clone(self) -> None:
        def kernel(points: np.ndarray) -> np.ndarray:
            return points[:, 0]

        def payload_builder(points: np.ndarray, raw_payloads: np.ndarray) -> list[dict[str, Any]]:
            return [{"value": float(value), "history": []} for value in raw_payloads]

        def first_evaluator(points: np.ndarray, payloads: list[dict[str, Any]], scale: float) -> np.ndarray:
            values = []
            for payload in payloads:
                payload["history"].append(("first", scale))
                values.append(scale * payload["value"])
            return np.array(values)

        def second_evaluator(points: np.ndarray, payloads: list[dict[str, Any]], shift: float) -> np.ndarray:
            values = []
            for payload in payloads:
                payload["history"].append(("second", shift))
                values.append(payload["value"] + shift)
            return np.array(values)

        first = StatefulIntegrator(
            a=[0.0],
            b=[1.0],
            kernel=kernel,
            evaluator=first_evaluator,
            payload_builder=payload_builder,
        )
        first_result = first.integrate(2.0, atol=1e-10, rtol=1e-10)
        original_payload = next(iter(first._leaves.values())).payload[0]

        second = first.replace_evaluator(second_evaluator)
        second_result = second.integrate(3.0, atol=1e-10, rtol=1e-10)
        cloned_payload = next(iter(second._leaves.values())).payload[0]

        self.assertEqual(first_result.status, "converged")
        self.assertEqual(second_result.status, "converged")
        self.assertEqual(second_result.n_kernel_evals, 0)
        self.assertIs(original_payload, cloned_payload)
        self.assertEqual(original_payload["history"], [("first", 2.0), ("second", 3.0)])

    def test_payload_builder_must_return_one_payload_per_point(self) -> None:
        def kernel(points: np.ndarray) -> np.ndarray:
            return points[:, 0]

        def payload_builder(points: np.ndarray, raw_payloads: np.ndarray) -> list[dict[str, float]]:
            return [{"value": float(raw_payloads[0])}]

        def evaluator(points: np.ndarray, payloads: list[dict[str, float]], params: object) -> np.ndarray:
            return np.array([payload["value"] for payload in payloads])

        integrator = StatefulIntegrator(
            a=[0.0],
            b=[1.0],
            kernel=kernel,
            evaluator=evaluator,
            payload_builder=payload_builder,
        )

        with self.assertRaisesRegex(ValueError, "payload_builder must return one prepared payload per input point"):
            integrator.integrate(atol=0.0, rtol=0.0)
