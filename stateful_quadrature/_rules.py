"""Nested quadrature and cubature rules used by the stateful integrator."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NestedRule:
    """Fixed nested rule on the reference hypercube ``[-1, 1]^ndim``."""

    name: str
    ndim: int
    reference_nodes: np.ndarray
    high_weights: np.ndarray
    low_weights: np.ndarray


def resolve_rule(name: str, ndim: int, dtype: np.dtype | type = np.float64) -> NestedRule:
    """Return a built-in nested rule."""

    if name == "gk15":
        base = _gauss_kronrod_15(dtype)
        return base if ndim == 1 else _product_rule(base, ndim)
    if name == "gk21":
        base = _gauss_kronrod_21(dtype)
        return base if ndim == 1 else _product_rule(base, ndim)
    if name == "trapezoid":
        if ndim != 1:
            raise ValueError("trapezoid is only supported for 1D integrals")
        return _trapezoid(dtype)
    if name == "genz_malik":
        if ndim < 2:
            raise ValueError("genz_malik is only supported for ndim >= 2")
        return _genz_malik(ndim, dtype)
    raise ValueError(f"unknown rule {name!r}")


def map_rule(rule: NestedRule, a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map a reference rule from ``[-1, 1]^n`` to a rectangular region ``[a, b]``."""

    a = np.asarray(a, dtype=rule.reference_nodes.dtype)
    b = np.asarray(b, dtype=rule.reference_nodes.dtype)
    lengths = b - a
    nodes = (rule.reference_nodes + 1.0) * (lengths * 0.5) + a
    weight_scale = np.prod(lengths, dtype=rule.reference_nodes.dtype) / (2.0 ** rule.ndim)
    return nodes, rule.high_weights * weight_scale, rule.low_weights * weight_scale


def _gauss_kronrod_15(dtype: np.dtype | type) -> NestedRule:
    x = np.asarray(
        [
            0.9914553711208126,
            0.9491079123427585,
            0.8648644233597691,
            0.7415311855993945,
            0.5860872354676911,
            0.4058451513773972,
            0.20778495500789847,
            0.0,
            -0.20778495500789847,
            -0.4058451513773972,
            -0.5860872354676911,
            -0.7415311855993945,
            -0.8648644233597691,
            -0.9491079123427585,
            -0.9914553711208126,
        ],
        dtype=dtype,
    )[:, None]
    high = np.asarray(
        [
            0.022935322010529225,
            0.06309209262997856,
            0.10479001032225018,
            0.14065325971552592,
            0.1690047266392679,
            0.19035057806478542,
            0.20443294007529889,
            0.20948214108472782,
            0.20443294007529889,
            0.19035057806478542,
            0.1690047266392679,
            0.14065325971552592,
            0.10479001032225018,
            0.06309209262997856,
            0.022935322010529225,
        ],
        dtype=dtype,
    )
    low = np.zeros_like(high)
    low[[1, 3, 5, 7, 9, 11, 13]] = np.asarray(
        [
            0.1294849661688697,
            0.27970539148927664,
            0.3818300505051189,
            0.4179591836734694,
            0.3818300505051189,
            0.27970539148927664,
            0.1294849661688697,
        ],
        dtype=dtype,
    )
    return NestedRule("gk15", 1, x, high, low)


def _gauss_kronrod_21(dtype: np.dtype | type) -> NestedRule:
    x = np.asarray(
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
    high = np.asarray(
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
    low = np.zeros_like(high)
    low[[1, 3, 5, 7, 9, 11, 13, 15, 17, 19]] = np.asarray(
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
    return NestedRule("gk21", 1, x, high, low)


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

    high = np.concatenate(
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

    low = np.concatenate(
        [
            np.full(1, low_w_1, dtype=dtype),
            np.full(2 * ndim, low_w_2, dtype=dtype),
            np.full(2 * ndim, low_w_3, dtype=dtype),
            np.full(2 * (ndim - 1) * ndim, low_w_4, dtype=dtype),
            np.zeros(2**ndim, dtype=dtype),
        ]
    )

    return NestedRule("genz_malik", ndim, reference_nodes, high, low)


def _product_rule(base_rule: NestedRule, ndim: int) -> NestedRule:
    index_grid = np.asarray(list(itertools.product(range(base_rule.reference_nodes.shape[0]), repeat=ndim)))
    reference_nodes = np.stack(
        [base_rule.reference_nodes[index_grid[:, axis], 0] for axis in range(ndim)],
        axis=-1,
    ).astype(base_rule.reference_nodes.dtype, copy=False)
    high = np.prod(base_rule.high_weights[index_grid], axis=1, dtype=base_rule.high_weights.dtype)
    low = np.prod(base_rule.low_weights[index_grid], axis=1, dtype=base_rule.low_weights.dtype)
    return NestedRule(base_rule.name, ndim, reference_nodes, high, low)


def _trapezoid(dtype: np.dtype | type) -> NestedRule:
    x = np.asarray([1.0, 0.0, -1.0], dtype=dtype)[:, None]
    high = np.asarray([0.5, 1.0, 0.5], dtype=dtype)
    low = np.asarray([1.0, 0.0, 1.0], dtype=dtype)
    return NestedRule("trapezoid", 1, x, high, low)


def _distinct_permutations(values: tuple[float, ...]):
    """Yield distinct permutations in lexicographic order."""

    items = sorted(values)
    size = len(items)

    while True:
        yield tuple(items)

        for i in range(size - 2, -1, -1):
            if items[i] < items[i + 1]:
                break
        else:
            return

        for j in range(size - 1, i, -1):
            if items[i] < items[j]:
                break

        items[i], items[j] = items[j], items[i]
        items[i + 1 :] = items[: i - size : -1]
