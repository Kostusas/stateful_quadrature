"""Quadrature rules for the lean stateful integrator."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class NestedRule:
    """Fixed embedded rule on the reference hypercube ``[-1, 1]^ndim``."""

    name: str
    ndim: int
    reference_nodes: np.ndarray
    high_weights: np.ndarray
    low_weights: np.ndarray

    @property
    def n_nodes(self) -> int:
        return int(self.reference_nodes.shape[0])


def resolve_rule(name: str, ndim: int, dtype: np.dtype | type = np.float64) -> NestedRule:
    """Return a built-in rule supported by ``StatefulIntegrator``."""

    if name == "gk21":
        if ndim != 1:
            raise ValueError("gk21 is only supported for 1D integrals")
        return _gauss_kronrod_21(dtype)
    if name == "genz_malik":
        if ndim < 2:
            raise ValueError("genz_malik is only supported for ndim >= 2")
        return _genz_malik(ndim, dtype)
    raise ValueError(f"unknown rule {name!r}")


def map_rule(rule: NestedRule, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map a reference rule from ``[-1, 1]^n`` to a rectangular region ``[a, b]``."""

    a = np.asarray(a, dtype=rule.reference_nodes.dtype)
    b = np.asarray(b, dtype=rule.reference_nodes.dtype)
    half_widths = 0.5 * (b - a)
    centers = 0.5 * (a + b)
    nodes = centers[None, :] + half_widths[None, :] * rule.reference_nodes
    weight_scale = np.prod(half_widths, dtype=rule.reference_nodes.dtype)
    return (
        np.ascontiguousarray(nodes),
        np.ascontiguousarray(rule.high_weights * weight_scale),
        np.ascontiguousarray(rule.low_weights * weight_scale),
    )


def _gauss_kronrod_21(dtype: np.dtype | type) -> NestedRule:
    reference_nodes = np.asarray(
        [
            0.9956571630258081,
            0.9739065285171717,
            0.9301574913557082,
            0.8650633666889845,
            0.7808177265864169,
            0.6794095682990244,
            0.5627571346686047,
            0.4333953941292472,
            0.2943928627014602,
            0.14887433898163122,
            0.0,
            -0.14887433898163122,
            -0.2943928627014602,
            -0.4333953941292472,
            -0.5627571346686047,
            -0.6794095682990244,
            -0.7808177265864169,
            -0.8650633666889845,
            -0.9301574913557082,
            -0.9739065285171717,
            -0.9956571630258081,
        ],
        dtype=dtype,
    )[:, None]
    high_weights = np.asarray(
        [
            0.011694638867371874,
            0.03255816230796473,
            0.054755896574352,
            0.07503967481091996,
            0.09312545458369761,
            0.10938715880229764,
            0.12349197626206585,
            0.13470921731147333,
            0.14277593857706008,
            0.1477391049013385,
            0.1494455540029169,
            0.1477391049013385,
            0.14277593857706008,
            0.13470921731147333,
            0.12349197626206585,
            0.10938715880229764,
            0.09312545458369761,
            0.07503967481091996,
            0.054755896574352,
            0.03255816230796473,
            0.011694638867371874,
        ],
        dtype=dtype,
    )
    low_weights = np.zeros_like(high_weights)
    low_weights[[1, 3, 5, 7, 9, 11, 13, 15, 17, 19]] = np.asarray(
        [
            0.06667134430868814,
            0.1494513491505806,
            0.21908636251598204,
            0.26926671930999635,
            0.29552422471475287,
            0.29552422471475287,
            0.26926671930999635,
            0.21908636251598204,
            0.1494513491505806,
            0.06667134430868814,
        ],
        dtype=dtype,
    )
    return NestedRule("gk21", 1, np.ascontiguousarray(reference_nodes), high_weights, low_weights)


def _genz_malik(ndim: int, dtype: np.dtype | type) -> NestedRule:
    l_2 = math.sqrt(9.0 / 70.0)
    l_3 = math.sqrt(9.0 / 10.0)
    l_4 = math.sqrt(9.0 / 10.0)
    l_5 = math.sqrt(9.0 / 19.0)

    base_nodes = list(
        itertools.chain(
            [(0.0,) * ndim],
            _distinct_permutations((l_2,) + (0.0,) * (ndim - 1)),
            _distinct_permutations((-l_2,) + (0.0,) * (ndim - 1)),
            _distinct_permutations((l_3,) + (0.0,) * (ndim - 1)),
            _distinct_permutations((-l_3,) + (0.0,) * (ndim - 1)),
            _distinct_permutations((l_4, l_4) + (0.0,) * (ndim - 2)),
            _distinct_permutations((l_4, -l_4) + (0.0,) * (ndim - 2)),
            _distinct_permutations((-l_4, -l_4) + (0.0,) * (ndim - 2)),
            itertools.product((l_5, -l_5), repeat=ndim),
        )
    )
    reference_nodes = np.asarray(base_nodes, dtype=dtype)

    w_1 = (2.0**ndim) * (12824.0 - 9120.0 * ndim + 400.0 * ndim**2) / 19683.0
    w_2 = (2.0**ndim) * 980.0 / 6561.0
    w_3 = (2.0**ndim) * (1820.0 - 400.0 * ndim) / 19683.0
    w_4 = (2.0**ndim) * 200.0 / 19683.0
    w_5 = 6859.0 / 19683.0

    high_weights = np.concatenate(
        [
            np.full(1, w_1, dtype=dtype),
            np.full(2 * ndim, w_2, dtype=dtype),
            np.full(2 * ndim, w_3, dtype=dtype),
            np.full(2 * (ndim - 1) * ndim, w_4, dtype=dtype),
            np.full(2**ndim, w_5, dtype=dtype),
        ]
    )

    low_w_1 = (2.0**ndim) * (729.0 - 950.0 * ndim + 50.0 * ndim**2) / 729.0
    low_w_2 = (2.0**ndim) * 245.0 / 486.0
    low_w_3 = (2.0**ndim) * (265.0 - 100.0 * ndim) / 1458.0
    low_w_4 = (2.0**ndim) * 25.0 / 729.0

    low_weights = np.concatenate(
        [
            np.full(1, low_w_1, dtype=dtype),
            np.full(2 * ndim, low_w_2, dtype=dtype),
            np.full(2 * ndim, low_w_3, dtype=dtype),
            np.full(2 * (ndim - 1) * ndim, low_w_4, dtype=dtype),
            np.zeros(2**ndim, dtype=dtype),
        ]
    )

    return NestedRule(
        "genz_malik",
        ndim,
        np.ascontiguousarray(reference_nodes),
        np.ascontiguousarray(high_weights),
        np.ascontiguousarray(low_weights),
    )


def _distinct_permutations(values: tuple[float, ...]):
    """Yield distinct permutations in lexicographic order."""

    items = sorted(values)
    size = len(items)
    used = [False] * size
    current = [0.0] * size

    def visit(depth: int):
        if depth == size:
            yield tuple(current)
            return

        previous = None
        for index, value in enumerate(items):
            if used[index] or value == previous:
                continue
            used[index] = True
            current[depth] = value
            yield from visit(depth + 1)
            used[index] = False
            previous = value

    yield from visit(0)
