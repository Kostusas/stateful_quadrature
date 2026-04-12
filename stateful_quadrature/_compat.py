"""SciPy-style compatibility wrappers and benchmark targets."""

from __future__ import annotations

import collections
import copy
import functools
import heapq
import itertools
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from ._rules import NestedRule, map_rule, resolve_rule


class LRUDict(collections.OrderedDict):
    def __init__(self, max_size: int) -> None:
        super().__init__()
        self._max_size = max_size

    def __setitem__(self, key: tuple[float, float], value: Any) -> None:
        existing = key in self
        super().__setitem__(key, value)
        if existing:
            self.move_to_end(key)
        elif self._max_size >= 0 and len(self) > self._max_size:
            self.popitem(last=False)


class _Bunch:
    def __init__(self, **kwargs: Any) -> None:
        self._keys = tuple(kwargs)
        self.__dict__.update(kwargs)

    def __repr__(self) -> str:
        pairs = ", ".join(f"{key}={self.__dict__[key]!r}" for key in self._keys)
        return f"_Bunch({pairs})"


class _MapWrapper:
    def __init__(self, workers: int | Callable[..., Any]) -> None:
        self._workers = workers
        self._executor: ThreadPoolExecutor | None = None
        self._map: Callable[..., Any] | None = None

    def __enter__(self) -> Callable[..., Any]:
        workers = self._workers
        if callable(workers):
            self._map = workers
            return self._map

        if workers in (None, 1):
            self._map = map
            return self._map

        if not isinstance(workers, int):
            raise TypeError("workers must be an integer or a map-like callable")

        max_workers = (os.cpu_count() or 1) if workers == -1 else workers
        if max_workers <= 0:
            raise ValueError("workers must be positive or -1")

        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        self._map = self._executor.map
        return self._map

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)


class SemiInfiniteFunc:
    """Argument transform from ``(start, +/-inf)`` to ``(0, 1)``."""

    def __init__(self, func: Callable[[float], Any], start: float, infty: float) -> None:
        self._func = func
        self._start = float(start)
        self._sgn = -1.0 if infty < 0 else 1.0
        self._tmin = sys.float_info.min**0.5

    def get_t(self, x: float) -> float:
        z = self._sgn * (x - self._start) + 1.0
        if z == 0.0:
            return math.inf
        return 1.0 / z

    def __call__(self, t: float) -> Any:
        if t < self._tmin:
            return 0.0
        x = self._start + self._sgn * (1.0 - t) / t
        f_val = self._func(x)
        return self._sgn * (f_val / t) / t


class DoubleInfiniteFunc:
    """Argument transform from ``(-inf, inf)`` to ``(-1, 1)``."""

    def __init__(self, func: Callable[[float], Any]) -> None:
        self._func = func
        self._tmin = sys.float_info.min**0.5

    def get_t(self, x: float) -> float:
        sign = -1.0 if x < 0 else 1.0
        return sign / (abs(x) + 1.0)

    def __call__(self, t: float) -> Any:
        if abs(t) < self._tmin:
            return 0.0
        x = (1.0 - abs(t)) / t
        f_val = self._func(x)
        return (f_val / t) / t


def _max_norm(x: Any) -> float:
    return float(np.amax(np.abs(x)))


def _quadrature_trapezoid(x1: float, x2: float, f: Callable[[float], Any], norm_func: Callable[[Any], float]):
    x3 = 0.5 * (x1 + x2)
    f1 = f(x1)
    f2 = f(x2)
    f3 = f(x3)

    s2 = 0.25 * (x2 - x1) * (f1 + 2 * f3 + f2)
    round_err = (
        0.25
        * abs(x2 - x1)
        * (float(norm_func(f1)) + 2.0 * float(norm_func(f3)) + float(norm_func(f2)))
        * 2e-16
    )

    s1 = 0.5 * (x2 - x1) * (f1 + f2)
    err = (1.0 / 3.0) * float(norm_func(s1 - s2))
    return s2, err, round_err


_quadrature_trapezoid.cache_size = 9
_quadrature_trapezoid.num_eval = 3


def _quadrature_gk(
    a: float,
    b: float,
    f: Callable[[float], Any],
    norm_func: Callable[[Any], float],
    nodes: tuple[float, ...],
    gauss_weights: tuple[float, ...],
    kronrod_weights: tuple[float, ...],
):
    values = [0.0] * len(nodes)
    center = 0.5 * (a + b)
    half_width = 0.5 * (b - a)

    s_k = 0.0
    s_k_abs = 0.0
    for idx, node in enumerate(nodes):
        value = f(center + half_width * node)
        values[idx] = value
        weight = kronrod_weights[idx]
        s_k += weight * value
        s_k_abs += weight * abs(value)

    s_g = 0.0
    for idx, weight in enumerate(gauss_weights):
        s_g += weight * values[2 * idx + 1]

    s_k_dabs = 0.0
    y0 = s_k / 2.0
    for idx, weight in enumerate(kronrod_weights):
        s_k_dabs += weight * abs(values[idx] - y0)

    err = float(norm_func((s_k - s_g) * half_width))
    dabs = float(norm_func(s_k_dabs * half_width))
    if dabs != 0.0 and err != 0.0:
        err = dabs * min(1.0, (200.0 * err / dabs) ** 1.5)

    eps = sys.float_info.epsilon
    round_err = float(norm_func(50.0 * eps * half_width * s_k_abs))
    if round_err > sys.float_info.min:
        err = max(err, round_err)

    return half_width * s_k, err, round_err


def _quadrature_gk21(x1: float, x2: float, f: Callable[[float], Any], norm_func: Callable[[Any], float]):
    nodes = (
        0.995657163025808080735527280689003,
        0.973906528517171720077964012084452,
        0.930157491355708226001207180059508,
        0.865063366688984510732096688423493,
        0.780817726586416897063717578345042,
        0.679409568299024406234327365114874,
        0.562757134668604683339000099272694,
        0.433395394129247190799265943165784,
        0.294392862701460198131126603103866,
        0.148874338981631210884826001129720,
        0.0,
        -0.148874338981631210884826001129720,
        -0.294392862701460198131126603103866,
        -0.433395394129247190799265943165784,
        -0.562757134668604683339000099272694,
        -0.679409568299024406234327365114874,
        -0.780817726586416897063717578345042,
        -0.865063366688984510732096688423493,
        -0.930157491355708226001207180059508,
        -0.973906528517171720077964012084452,
        -0.995657163025808080735527280689003,
    )
    gauss_weights = (
        0.066671344308688137593568809893332,
        0.149451349150580593145776339657697,
        0.219086362515982043995534934228163,
        0.269266719309996355091226921569469,
        0.295524224714752870173892994651338,
        0.295524224714752870173892994651338,
        0.269266719309996355091226921569469,
        0.219086362515982043995534934228163,
        0.149451349150580593145776339657697,
        0.066671344308688137593568809893332,
    )
    kronrod_weights = (
        0.011694638867371874278064396062192,
        0.032558162307964727478818972459390,
        0.054755896574351996031381300244580,
        0.075039674810919952767043140916190,
        0.093125454583697605535065465083366,
        0.109387158802297641899210590325805,
        0.123491976262065851077958109831074,
        0.134709217311473325928054001771707,
        0.142775938577060080797094273138717,
        0.147739104901338491374841515972068,
        0.149445554002916905664936468389821,
        0.147739104901338491374841515972068,
        0.142775938577060080797094273138717,
        0.134709217311473325928054001771707,
        0.123491976262065851077958109831074,
        0.109387158802297641899210590325805,
        0.093125454583697605535065465083366,
        0.075039674810919952767043140916190,
        0.054755896574351996031381300244580,
        0.032558162307964727478818972459390,
        0.011694638867371874278064396062192,
    )
    return _quadrature_gk(x1, x2, f, norm_func, nodes, gauss_weights, kronrod_weights)


_quadrature_gk21.num_eval = 21


def _quadrature_gk15(x1: float, x2: float, f: Callable[[float], Any], norm_func: Callable[[Any], float]):
    nodes = (
        0.991455371120812639206854697526329,
        0.949107912342758524526189684047851,
        0.864864423359769072789712788640926,
        0.741531185599394439863864773280788,
        0.586087235467691130294144838258730,
        0.405845151377397166906606412076961,
        0.207784955007898467600689403773245,
        0.0,
        -0.207784955007898467600689403773245,
        -0.405845151377397166906606412076961,
        -0.586087235467691130294144838258730,
        -0.741531185599394439863864773280788,
        -0.864864423359769072789712788640926,
        -0.949107912342758524526189684047851,
        -0.991455371120812639206854697526329,
    )
    gauss_weights = (
        0.129484966168869693270611432679082,
        0.279705391489276667901467771423780,
        0.381830050505118944950369775488975,
        0.417959183673469387755102040816327,
        0.381830050505118944950369775488975,
        0.279705391489276667901467771423780,
        0.129484966168869693270611432679082,
    )
    kronrod_weights = (
        0.022935322010529224963732008058970,
        0.063092092629978553290700663189204,
        0.104790010322250183839876322541518,
        0.140653259715525918745189590510238,
        0.169004726639267902826583426598550,
        0.190350578064785409913256402421014,
        0.204432940075298892414161999234649,
        0.209482141084727828012999174891714,
        0.204432940075298892414161999234649,
        0.190350578064785409913256402421014,
        0.169004726639267902826583426598550,
        0.140653259715525918745189590510238,
        0.104790010322250183839876322541518,
        0.063092092629978553290700663189204,
        0.022935322010529224963732008058970,
    )
    return _quadrature_gk(x1, x2, f, norm_func, nodes, gauss_weights, kronrod_weights)


_quadrature_gk15.num_eval = 15


def _subdivide_interval(args: tuple[tuple[float, float, float, Any], Callable[[float], Any], Callable[[Any], float], Any]):
    interval, f, norm_func, quadrature = args
    old_err, a, b, old_int = interval
    midpoint = 0.5 * (a + b)

    if getattr(quadrature, "cache_size", 0) > 0:
        f = functools.lru_cache(quadrature.cache_size)(f)

    left_int, left_err, left_round = quadrature(a, midpoint, f, norm_func)
    neval = quadrature.num_eval
    right_int, right_err, right_round = quadrature(midpoint, b, f, norm_func)
    neval += quadrature.num_eval

    if old_int is None:
        old_int, _, _ = quadrature(a, b, f, norm_func)
        neval += quadrature.num_eval

    if getattr(quadrature, "cache_size", 0) > 0:
        neval = f.cache_info().misses

    delta_integral = left_int + right_int - old_int
    delta_error = left_err + right_err - old_err
    delta_round = left_round + right_round

    subintervals = ((a, midpoint, left_int, left_err), (midpoint, b, right_int, right_err))
    return delta_integral, delta_error, delta_round, subintervals, neval


def quad_vec(
    f: Callable[..., Any],
    a: float,
    b: float,
    epsabs: float = 1e-200,
    epsrel: float = 1e-8,
    norm: str | Callable[[Any], float] | None = "2",
    cache_size: float = 100e6,
    limit: int = 10_000,
    workers: int | Callable[..., Any] = 1,
    points: tuple[float, ...] | list[float] | None = None,
    quadrature: str | None = None,
    full_output: bool = False,
    *,
    args: tuple[Any, ...] = (),
):
    a = float(a)
    b = float(b)

    if args:
        if not isinstance(args, tuple):
            args = (args,)

        func = f

        def f(x: float) -> Any:
            return func(x, *args)

    if np.isfinite(a) and np.isinf(b):
        transformed = SemiInfiniteFunc(f, start=a, infty=b)
        mapped_points = None if points is None else tuple(transformed.get_t(float(point)) for point in points)
        return quad_vec(
            transformed,
            0.0,
            1.0,
            epsabs=epsabs,
            epsrel=epsrel,
            norm=norm,
            cache_size=cache_size,
            limit=limit,
            workers=workers,
            points=mapped_points,
            quadrature="gk15" if quadrature is None else quadrature,
            full_output=full_output,
        )

    if np.isfinite(b) and np.isinf(a):
        transformed = SemiInfiniteFunc(f, start=b, infty=a)
        mapped_points = None if points is None else tuple(transformed.get_t(float(point)) for point in points)
        res = quad_vec(
            transformed,
            0.0,
            1.0,
            epsabs=epsabs,
            epsrel=epsrel,
            norm=norm,
            cache_size=cache_size,
            limit=limit,
            workers=workers,
            points=mapped_points,
            quadrature="gk15" if quadrature is None else quadrature,
            full_output=full_output,
        )
        return (-res[0],) + res[1:]

    if np.isinf(a) and np.isinf(b):
        sign = -1.0 if b < a else 1.0
        transformed = DoubleInfiniteFunc(f)
        mapped_points = (0.0,) if points is None else (0.0,) + tuple(transformed.get_t(float(point)) for point in points)
        if a != b:
            res = quad_vec(
                transformed,
                -1.0,
                1.0,
                epsabs=epsabs,
                epsrel=epsrel,
                norm=norm,
                cache_size=cache_size,
                limit=limit,
                workers=workers,
                points=mapped_points,
                quadrature="gk15" if quadrature is None else quadrature,
                full_output=full_output,
            )
        else:
            res = (0.0, 0.0, _Bunch(
                neval=0,
                success=True,
                status=0,
                message="Target precision reached.",
                intervals=np.empty((0, 2), dtype=float),
                integrals=np.empty((0,), dtype=float),
                errors=np.empty((0,), dtype=float),
            )) if full_output else (0.0, 0.0)
        return (res[0] * sign,) + res[1:]

    if not (np.isfinite(a) and np.isfinite(b)):
        raise ValueError(f"invalid integration bounds a={a}, b={b}")

    norm_funcs = {None: _max_norm, "max": _max_norm, "2": np.linalg.norm}
    norm_func = norm if callable(norm) else norm_funcs[norm]

    try:
        quadrature_fn = {
            None: _quadrature_gk21,
            "gk21": _quadrature_gk21,
            "gk15": _quadrature_gk15,
            "trapezoid": _quadrature_trapezoid,
        }[quadrature]
    except KeyError as exc:
        raise ValueError(f"unknown quadrature {quadrature!r}") from exc

    if points is None:
        initial_intervals = [(a, b)]
    else:
        previous = a
        initial_intervals: list[tuple[float, float]] = []
        for point in sorted(float(p) for p in points):
            if not (a < point < b) or point == previous:
                continue
            initial_intervals.append((previous, point))
            previous = point
        initial_intervals.append((previous, b))

    global_integral = None
    global_error = None
    rounding_error = None
    interval_cache = None
    intervals: list[tuple[float, float, float]] = []
    neval = 0

    for x1, x2 in initial_intervals:
        integral_piece, err_piece, round_piece = quadrature_fn(x1, x2, f, norm_func)
        neval += quadrature_fn.num_eval

        if global_integral is None:
            if isinstance(integral_piece, (float, complex)) and norm_func in (_max_norm, np.linalg.norm):
                norm_func = abs
            global_integral = integral_piece
            global_error = float(err_piece)
            rounding_error = float(round_piece)
            cache_count = int(cache_size // max(sys.getsizeof(integral_piece), 1))
            interval_cache = LRUDict(cache_count)
        else:
            global_integral += integral_piece
            global_error += float(err_piece)
            rounding_error += float(round_piece)

        interval_cache[(x1, x2)] = copy.copy(integral_piece)
        intervals.append((-float(err_piece), x1, x2))

    heapq.heapify(intervals)

    CONVERGED = 0
    NOT_CONVERGED = 1
    ROUNDING_ERROR = 2
    NOT_A_NUMBER = 3

    status_message = {
        CONVERGED: "Target precision reached.",
        NOT_CONVERGED: "Target precision not reached.",
        ROUNDING_ERROR: "Target precision could not be reached due to rounding error.",
        NOT_A_NUMBER: "Non-finite values encountered.",
    }

    parallel_count = 128
    min_intervals = 2
    ier = NOT_CONVERGED

    with _MapWrapper(workers) as map_func:
        while intervals and len(intervals) < limit:
            tolerance = max(epsabs, epsrel * float(norm_func(global_integral)))
            to_process = []
            err_sum = 0.0

            for index in range(parallel_count):
                if not intervals:
                    break
                if index > 0 and err_sum > global_error - tolerance / 8.0:
                    break

                neg_old_err, left, right = heapq.heappop(intervals)
                old_integral = interval_cache.pop((left, right), None)
                to_process.append(((-neg_old_err, left, right, old_integral), f, norm_func, quadrature_fn))
                err_sum += -neg_old_err

            for delta_integral, delta_error, delta_round, subintervals, delta_neval in map_func(_subdivide_interval, to_process):
                neval += delta_neval
                global_integral += delta_integral
                global_error += float(delta_error)
                rounding_error += float(delta_round)
                for x1, x2, sub_integral, sub_error in subintervals:
                    interval_cache[(x1, x2)] = sub_integral
                    heapq.heappush(intervals, (-float(sub_error), x1, x2))

            if len(intervals) >= min_intervals:
                tolerance = max(epsabs, epsrel * float(norm_func(global_integral)))
                if global_error < tolerance / 8.0:
                    ier = CONVERGED
                    break
                if global_error < rounding_error:
                    ier = ROUNDING_ERROR
                    break

            if not (np.isfinite(global_error) and np.isfinite(rounding_error)):
                ier = NOT_A_NUMBER
                break

    result = global_integral
    error = global_error + rounding_error

    if full_output:
        result_arr = np.asarray(result)
        dummy = np.full(result_arr.shape, np.nan, dtype=result_arr.dtype)
        integrals = np.array([interval_cache.get((entry[1], entry[2]), dummy) for entry in intervals], dtype=result_arr.dtype)
        errors = np.array([-entry[0] for entry in intervals], dtype=float)
        interval_bounds = np.array([[entry[1], entry[2]] for entry in intervals], dtype=float)
        info = _Bunch(
            neval=neval,
            success=(ier == CONVERGED),
            status=ier,
            message=status_message[ier],
            intervals=interval_bounds,
            integrals=integrals,
            errors=errors,
        )
        return result, error, info

    return result, error


@dataclass(slots=True)
class CubatureRegion:
    estimate: Any
    error: Any
    a: np.ndarray
    b: np.ndarray

    @property
    def priority(self) -> float:
        return float(np.max(np.abs(np.asarray(self.error))))


@dataclass(slots=True)
class CubatureResult:
    estimate: Any
    error: Any
    status: str
    regions: list[CubatureRegion]
    subdivisions: int
    atol: float
    rtol: float


class _InfiniteLimitsTransform:
    def __init__(self, func: Callable[..., np.ndarray], a: np.ndarray, b: np.ndarray) -> None:
        self._func = func
        self._orig_a = np.array(a, copy=True)
        self._orig_b = np.array(b, copy=True)

        self._double_inf_pos = np.isneginf(a) & np.isposinf(b)
        start_inf_mask = np.isfinite(a) & np.isposinf(b)
        inf_end_mask = np.isneginf(a) & np.isfinite(b)
        self._semi_inf_pos = start_inf_mask | inf_end_mask

        self._orig_a[inf_end_mask] = -b[inf_end_mask]
        self._orig_b[inf_end_mask] = -a[inf_end_mask]
        self._num_inf = int(np.count_nonzero(self._double_inf_pos | self._semi_inf_pos))

    @property
    def transformed_limits(self) -> tuple[np.ndarray, np.ndarray]:
        a = np.array(self._orig_a, copy=True)
        b = np.array(self._orig_b, copy=True)
        a[self._double_inf_pos] = -1.0
        b[self._double_inf_pos] = 1.0
        a[self._semi_inf_pos] = 0.0
        b[self._semi_inf_pos] = 1.0
        return a, b

    @property
    def points(self) -> list[np.ndarray]:
        if self._num_inf == 0:
            return []
        return [np.zeros(self._orig_a.shape, dtype=self._orig_a.dtype)]

    def inv(self, x: np.ndarray) -> np.ndarray:
        points = np.array(x, copy=True)
        npoints = points.shape[0]
        double_mask = np.tile(self._double_inf_pos[None, :], (npoints, 1))
        semi_mask = np.tile(self._semi_inf_pos[None, :], (npoints, 1))

        double_values = points[double_mask]
        zero_mask = double_values == 0.0
        mapped_double = np.empty_like(double_values)
        mapped_double[zero_mask] = math.inf
        mapped_double[~zero_mask] = 1.0 / (double_values[~zero_mask] + np.sign(double_values[~zero_mask]))
        points[double_mask] = mapped_double

        if np.any(semi_mask):
            start = np.tile(self._orig_a[self._semi_inf_pos], npoints)
            points[semi_mask] = 1.0 / (points[semi_mask] - start + 1.0)

        return points

    def __call__(self, t: np.ndarray, *args: Any) -> np.ndarray:
        x = np.array(t, copy=True)
        npoints = t.shape[0]
        double_mask = np.tile(self._double_inf_pos[None, :], (npoints, 1))
        semi_mask = np.tile(self._semi_inf_pos[None, :], (npoints, 1))

        x[double_mask] = (1.0 - np.abs(t[double_mask])) / t[double_mask]
        if np.any(semi_mask):
            start = np.tile(self._orig_a[self._semi_inf_pos], npoints)
            x[semi_mask] = start + (1.0 - t[semi_mask]) / t[semi_mask]

        if self._num_inf == 0:
            return self._func(x, *args)

        jacobian = 1.0 / np.prod(np.reshape(t[semi_mask | double_mask] ** 2, (-1, self._num_inf)), axis=-1)
        values = np.asarray(self._func(x, *args))
        reshape = (-1,) + (1,) * (values.ndim - 1)
        return values * jacobian.reshape(reshape)


def _is_strictly_in_region(a: np.ndarray, b: np.ndarray, point: np.ndarray) -> bool:
    if np.all(point == a) or np.all(point == b):
        return False
    return bool(np.all(a <= point) and np.all(point <= b))


def _split_subregion(a: np.ndarray, b: np.ndarray, split_at: np.ndarray | None = None) -> list[tuple[np.ndarray, np.ndarray]]:
    midpoint = 0.5 * (a + b) if split_at is None else split_at
    children = []
    for side in itertools.product((0, 1), repeat=a.size):
        child_a = np.array(a, copy=True)
        child_b = np.array(b, copy=True)
        for axis, upper_half in enumerate(side):
            if upper_half:
                child_a[axis] = midpoint[axis]
            else:
                child_b[axis] = midpoint[axis]
        children.append((child_a, child_b))
    return children


def _split_region_at_points(
    a: np.ndarray,
    b: np.ndarray,
    points: list[np.ndarray],
) -> list[tuple[np.ndarray, np.ndarray]]:
    regions = [(a, b)]
    for point in points:
        if np.any(np.isinf(point)):
            continue
        new_regions: list[tuple[np.ndarray, np.ndarray]] = []
        for left, right in regions:
            if _is_strictly_in_region(left, right, point):
                for child_left, child_right in _split_subregion(left, right, point):
                    if np.any(child_left == child_right):
                        continue
                    new_regions.append((child_left, child_right))
            else:
                new_regions.append((left, right))
        regions = new_regions
    return regions


def _gauss_kronrod_region_error(
    region_a: np.ndarray,
    region_b: np.ndarray,
    high_weights: np.ndarray,
    region_values: np.ndarray,
    high_estimate: np.ndarray | np.generic,
    low_estimate: np.ndarray | np.generic,
) -> np.ndarray | np.generic:
    half_width = 0.5 * float(region_b[0] - region_a[0])
    err = np.abs(high_estimate - low_estimate)
    if half_width == 0.0:
        return err

    reshape = (high_weights.shape[0],) + (1,) * (region_values.ndim - 1)
    weighted_high = high_weights.reshape(reshape)
    center_value = high_estimate / (2.0 * half_width)
    dabs = np.sum(weighted_high * np.abs(region_values - center_value), axis=0)
    ratio = np.divide(200.0 * err, dabs, out=np.zeros_like(np.asarray(err, dtype=np.result_type(err, np.float64))), where=dabs != 0)
    err = np.where((dabs != 0) & (err != 0), dabs * np.minimum(1.0, ratio**1.5), err)

    abs_integrand = np.sum(weighted_high * np.abs(region_values), axis=0)
    round_err = np.abs(50.0 * np.finfo(high_weights.dtype).eps * abs_integrand)
    err = np.where(round_err > np.finfo(high_weights.dtype).tiny, np.maximum(err, round_err), err)
    return err


def _evaluate_rule(
    func: Callable[..., np.ndarray],
    left: np.ndarray,
    right: np.ndarray,
    rule: NestedRule,
    args: tuple[Any, ...],
) -> tuple[np.ndarray | np.generic, np.ndarray | np.generic]:
    nodes, high_weights, low_weights = map_rule(rule, left, right)
    values = np.asarray(func(nodes, *args))
    high_estimate = np.tensordot(high_weights, values, axes=(0, 0))
    low_estimate = np.tensordot(low_weights, values, axes=(0, 0))
    if rule.ndim == 1 and rule.name in {"gk15", "gk21"}:
        error = _gauss_kronrod_region_error(left, right, high_weights, values, high_estimate, low_estimate)
    else:
        error = np.abs(high_estimate - low_estimate)
    return high_estimate, error


def _resolve_cubature_rule(rule: str, ndim: int, dtype: np.dtype) -> NestedRule:
    aliases = {
        "gauss-kronrod": "gk21",
        "genz-malik": "genz_malik",
        "gk21": "gk21",
        "gk15": "gk15",
        "genz_malik": "genz_malik",
    }
    try:
        return resolve_rule(aliases[rule], ndim, dtype)
    except KeyError as exc:
        raise ValueError(f"unknown rule {rule!r}") from exc


def cubature(
    f: Callable[..., np.ndarray],
    a: np.ndarray | list[float],
    b: np.ndarray | list[float],
    *,
    rule: str = "gk21",
    rtol: float = 1e-8,
    atol: float = 0.0,
    max_subdivisions: int | None = 10_000,
    args: tuple[Any, ...] = (),
    workers: int | Callable[..., Any] = 1,
    points: list[np.ndarray | list[float]] | None = None,
) -> CubatureResult:
    if not isinstance(args, tuple):
        args = (args,)

    a_arr = np.asarray(a)
    b_arr = np.asarray(b)
    if a_arr.size == 0 or b_arr.size == 0:
        raise ValueError("`a` and `b` must be nonempty")
    if a_arr.ndim != 1 or b_arr.ndim != 1:
        raise ValueError("`a` and `b` must be 1D arrays")

    result_dtype = np.result_type(a_arr, b_arr, np.float64)
    a_arr = np.asarray(a_arr, dtype=result_dtype)
    b_arr = np.asarray(b_arr, dtype=result_dtype)
    sign = -1 if np.count_nonzero(a_arr > b_arr) % 2 else 1
    a_sorted = np.minimum(a_arr, b_arr)
    b_sorted = np.maximum(a_arr, b_arr)

    if max_subdivisions is None:
        max_subdivisions = sys.maxsize

    point_arrays = [] if points is None else [np.asarray(point, dtype=result_dtype) for point in points]
    if np.any(np.isinf(a_sorted)) or np.any(np.isinf(b_sorted)):
        transformed = _InfiniteLimitsTransform(f, a_sorted, b_sorted)
        a_sorted, b_sorted = transformed.transformed_limits
        point_arrays = [transformed.inv(point.reshape(1, -1)).reshape(-1) for point in point_arrays]
        point_arrays.extend(transformed.points)

        def f(points_batch: np.ndarray, *inner_args: Any) -> np.ndarray:
            return transformed(points_batch, *inner_args)

    rule_impl = _resolve_cubature_rule(rule, a_sorted.size, np.dtype(result_dtype))
    initial_regions = [(a_sorted, b_sorted)] if len(point_arrays) == 0 else _split_region_at_points(a_sorted, b_sorted, point_arrays)

    region_heap: list[tuple[float, int, CubatureRegion]] = []
    estimate_total = None
    error_total = None

    for index, (left, right) in enumerate(initial_regions):
        region_estimate, region_error = _evaluate_rule(f, left, right, rule_impl, args)
        region = CubatureRegion(region_estimate, region_error, left, right)
        if estimate_total is None:
            estimate_total = np.array(region_estimate, copy=True)
            error_total = np.array(region_error, copy=True)
        else:
            estimate_total = estimate_total + region_estimate
            error_total = error_total + region_error
        heapq.heappush(region_heap, (-region.priority, index, region))

    if estimate_total is None or error_total is None:
        raise RuntimeError("cubature could not create an initial region set")

    subdivisions = 0
    next_index = len(region_heap)

    with _MapWrapper(workers) as map_func:
        while np.any(error_total > atol + rtol * np.abs(estimate_total)):
            if subdivisions >= max_subdivisions or not region_heap:
                break

            _, _, region = heapq.heappop(region_heap)
            estimate_total = estimate_total - region.estimate
            error_total = error_total - region.error

            subregions = _split_subregion(region.a, region.b)
            evaluator_args = ((f, left, right, rule_impl, args) for left, right in subregions)

            for left, right, child_estimate, child_error in map_func(_process_subregion, evaluator_args):
                child = CubatureRegion(child_estimate, child_error, left, right)
                estimate_total = estimate_total + child_estimate
                error_total = error_total + child_error
                heapq.heappush(region_heap, (-child.priority, next_index, child))
                next_index += 1

            subdivisions += 1

    status = "converged" if np.all(error_total <= atol + rtol * np.abs(estimate_total)) else "not_converged"
    final_regions = [entry[2] for entry in sorted(region_heap, key=lambda item: item[1])]
    return CubatureResult(
        estimate=sign * estimate_total,
        error=error_total,
        status=status,
        regions=final_regions,
        subdivisions=subdivisions,
        atol=atol,
        rtol=rtol,
    )


def _process_subregion(data: tuple[Callable[..., np.ndarray], np.ndarray, np.ndarray, NestedRule, tuple[Any, ...]]):
    func, left, right, rule, args = data
    estimate, error = _evaluate_rule(func, left, right, rule, args)
    return left, right, estimate, error
