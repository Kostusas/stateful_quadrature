from __future__ import annotations

import math
import unittest

import numpy as np

from stateful_quadrature import cubature, quad_vec

try:
    from scipy.integrate import cubature as scipy_cubature
    from scipy.integrate import quad_vec as scipy_quad_vec
except ImportError:  # pragma: no cover - exercised in the default pixi env
    scipy_cubature = None
    scipy_quad_vec = None


def _basic_1d_integrand(points: np.ndarray, powers: np.ndarray) -> np.ndarray:
    return points[:, [0]] ** powers[None, :]


def _basic_1d_exact(powers: np.ndarray) -> np.ndarray:
    return 2.0 ** (powers + 1.0) / (powers + 1.0)


def _basic_nd_integrand(points: np.ndarray, powers: np.ndarray) -> np.ndarray:
    return np.sum(points, axis=-1, keepdims=True) ** powers[None, :]


def _basic_nd_exact(powers: np.ndarray) -> np.ndarray:
    return (-2.0 ** (3 + powers) + 4.0 ** (2 + powers)) / ((1.0 + powers) * (2.0 + powers))


class QuadVecCompatTests(unittest.TestCase):
    def test_simple_cases_follow_scipy_style_quadrature_matrix(self) -> None:
        powers = np.arange(10)
        exact = _basic_1d_exact(powers)

        for quadrature in (None, "gk15", "gk21", "trapezoid"):
            for epsabs in (1e-1, 1e-3, 1e-6):
                if quadrature == "trapezoid" and epsabs < 1e-4:
                    continue

                with self.subTest(quadrature=quadrature, epsabs=epsabs, norm="max"):
                    result, _ = quad_vec(lambda x: x**powers, 0.0, 2.0, norm="max", epsabs=epsabs, quadrature=quadrature)
                    np.testing.assert_allclose(result, exact, rtol=0.0, atol=epsabs)

                with self.subTest(quadrature=quadrature, epsabs=epsabs, norm="2"):
                    result, _ = quad_vec(lambda x: x**powers, 0.0, 2.0, norm="2", epsabs=epsabs, quadrature=quadrature)
                    self.assertLess(np.linalg.norm(result - exact), epsabs)

                with self.subTest(quadrature=quadrature, epsabs=epsabs, points=True):
                    result, _ = quad_vec(
                        lambda x: x**powers,
                        0.0,
                        2.0,
                        norm="max",
                        epsabs=epsabs,
                        quadrature=quadrature,
                        points=(0.5, 1.0),
                    )
                    np.testing.assert_allclose(result, exact, rtol=0.0, atol=epsabs)

                with self.subTest(quadrature=quadrature, epsabs=epsabs, full_output=True):
                    result, _, info = quad_vec(
                        lambda x: x**powers,
                        0.0,
                        2.0,
                        norm="max",
                        epsabs=epsabs,
                        epsrel=1e-8,
                        quadrature=quadrature,
                        full_output=True,
                        limit=10_000,
                    )
                    np.testing.assert_allclose(result, exact, rtol=0.0, atol=epsabs)
                    self.assertTrue(info.success)

    def test_infinite_limits_follow_scipy_style_cases(self) -> None:
        def lorentzian(x: float) -> float:
            return 1.0 / (1.0 + np.float64(x) ** 2)

        for quadrature in (None, "gk15", "gk21", "trapezoid"):
            for epsabs in (1e-1, 1e-3, 1e-6):
                if quadrature == "trapezoid" and epsabs < 1e-4:
                    continue

                kwargs = dict(norm="max", epsabs=epsabs, quadrature=quadrature)

                with self.subTest(quadrature=quadrature, case="0_inf"):
                    result, error = quad_vec(lorentzian, 0.0, np.inf, **kwargs)
                    np.testing.assert_allclose(result, np.pi / 2.0, rtol=0.0, atol=max(epsabs, error))

                with self.subTest(quadrature=quadrature, case="inf_0"):
                    result, error = quad_vec(lorentzian, np.inf, 0.0, **kwargs)
                    np.testing.assert_allclose(result, -np.pi / 2.0, rtol=0.0, atol=max(epsabs, error))

                with self.subTest(quadrature=quadrature, case="minus_inf_inf"):
                    result, error = quad_vec(lorentzian, -np.inf, np.inf, **kwargs)
                    np.testing.assert_allclose(result, np.pi, rtol=0.0, atol=max(epsabs, error))

                with self.subTest(quadrature=quadrature, case="points"):
                    result, error = quad_vec(lorentzian, 0.0, np.inf, points=(1.0, 2.0), **kwargs)
                    np.testing.assert_allclose(result, np.pi / 2.0, rtol=0.0, atol=max(epsabs, error))

    def test_args_num_eval_info_and_nan_status(self) -> None:
        def with_args(x: float, a: float) -> np.ndarray:
            return x * (x + a) * np.arange(3)

        result, _ = quad_vec(with_args, 0.0, 1.0, args=(2.0,))
        np.testing.assert_allclose(result, np.array([0.0, 4.0 / 3.0, 8.0 / 3.0]), rtol=0.0, atol=1e-4)

        count = [0]

        def counted(x: float) -> float:
            count[0] += 1
            return x**5

        _, _, info = quad_vec(counted, 0.0, 1.0, norm="max", full_output=True, quadrature="gk21")
        self.assertEqual(info.neval, count[0])

        _, _, info = quad_vec(lambda x: np.ones((3, 2, 1)), 0.0, 1.0, norm="max", full_output=True)
        self.assertTrue(info.success)
        self.assertEqual(info.status, 0)
        self.assertEqual(info.message, "Target precision reached.")
        self.assertGreater(info.neval, 0)
        self.assertEqual(info.intervals.shape[1], 2)
        self.assertEqual(info.integrals.shape, (info.intervals.shape[0], 3, 2, 1))
        self.assertEqual(info.errors.shape, (info.intervals.shape[0],))

        _, _, info = quad_vec(lambda x: np.nan, 0.0, 1.0, full_output=True)
        self.assertEqual(info.status, 3)

    @unittest.skipUnless(scipy_quad_vec is not None, "SciPy is not installed")
    def test_matches_scipy_quad_vec_on_ported_cases(self) -> None:
        for quadrature in (None, "gk15", "gk21", "trapezoid"):
            with self.subTest(quadrature=quadrature, case="finite"):
                ours = quad_vec(lambda x: np.cos(0.1 * x + np.arange(5)), 0.0, 1.0, epsrel=1e-10, quadrature=quadrature)
                theirs = scipy_quad_vec(lambda x: np.cos(0.1 * x + np.arange(5)), 0.0, 1.0, epsrel=1e-10, quadrature=quadrature)
                np.testing.assert_allclose(ours[0], theirs[0], atol=1e-12, rtol=1e-12)
                np.testing.assert_allclose(ours[1], theirs[1], atol=1e-12, rtol=1e-12)

            with self.subTest(quadrature=quadrature, case="infinite"):
                ours = quad_vec(lambda x: 1.0 / (1.0 + x**2), 0.0, np.inf, norm="max", epsabs=1e-6, quadrature=quadrature)
                theirs = scipy_quad_vec(lambda x: 1.0 / (1.0 + x**2), 0.0, np.inf, norm="max", epsabs=1e-6, quadrature=quadrature)
                np.testing.assert_allclose(ours[0], theirs[0], atol=1e-10, rtol=1e-10)
                np.testing.assert_allclose(ours[1], theirs[1], atol=1e-10, rtol=1e-10)


class CubatureCompatTests(unittest.TestCase):
    def test_rule_aliases_and_basic_interface_cases(self) -> None:
        powers = np.arange(5, dtype=np.float64)
        exact = _basic_nd_exact(powers)

        for rule in ("gauss-kronrod", "gk21", "gk15", "genz-malik"):
            with self.subTest(rule=rule):
                result = cubature(_basic_nd_integrand, [0.0, 0.0], [2.0, 2.0], rule=rule, args=(powers,))
                np.testing.assert_allclose(result.estimate, exact, rtol=1e-8, atol=0.0)

        one_d = cubature(_basic_1d_integrand, [0.0], [2.0], args=(powers,))
        np.testing.assert_allclose(one_d.estimate, _basic_1d_exact(powers), rtol=1e-8, atol=0.0)

        zero_width = cubature(_basic_1d_integrand, [0.0], [0.0], args=(powers,))
        np.testing.assert_allclose(zero_width.estimate, np.zeros_like(powers), rtol=0.0, atol=0.0)

        reversed_limits = cubature(_basic_1d_integrand, [2.0], [0.0], args=(powers,))
        np.testing.assert_allclose(reversed_limits.estimate, -_basic_1d_exact(powers), rtol=1e-8, atol=0.0)

    def test_infinite_limits_and_break_points(self) -> None:
        def gaussian(points: np.ndarray, alphas: np.ndarray) -> np.ndarray:
            return np.exp(-np.sum((alphas[None, :] * points) ** 2, axis=-1))

        result = cubature(
            gaussian,
            [-np.inf, -np.inf],
            [np.inf, np.inf],
            rule="genz-malik",
            rtol=1e-5,
            atol=1e-6,
            args=(np.array([2.0, 3.0]),),
            points=[np.array([0.0, 0.0])],
        )
        exact = np.pi / 6.0
        self.assertEqual(result.status, "converged")
        np.testing.assert_allclose(result.estimate, exact, rtol=1e-5, atol=1e-5)

    @unittest.skipUnless(scipy_cubature is not None, "SciPy is not installed")
    def test_matches_scipy_cubature_on_ported_cases(self) -> None:
        def oscillatory(points: np.ndarray) -> np.ndarray:
            r = np.repeat(0.5, 4)
            alphas = np.repeat(0.1, 12).reshape(4, 3)
            reshaped = points.reshape(points.shape[0], 1, points.shape[1])
            return np.cos(2.0 * np.pi * r + np.sum(alphas[None, :, :] * reshaped, axis=-1))

        ours = cubature(oscillatory, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], rule="genz-malik", rtol=1e-8, atol=0.0)
        theirs = scipy_cubature(oscillatory, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0], rule="genz-malik", rtol=1e-8, atol=0.0)

        np.testing.assert_allclose(ours.estimate, theirs.estimate, atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(ours.error, theirs.error, atol=1e-10, rtol=1e-10)
        self.assertEqual(ours.status, theirs.status)

