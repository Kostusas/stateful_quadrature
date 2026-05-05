# Stateful Quadrature

`stateful_quadrature` is a focused adaptive quadrature package for repeated finite-domain
integrals of the form

`I(lambda) = integral Psi(x, K(x), lambda) dx`

where `K(x)` is expensive, independent of `lambda`, and worth caching exactly.

The package is intentionally small. Its public surface centers on
`stateful_quadrature.StatefulIntegrator`, which keeps a live adaptive leaf mesh and reuses cached
kernel payloads exactly across repeated calls. It can also prepare mutable per-node payload
objects once and reuse them across later integrations.

The package is inspired by [SciPy's cubature](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.cubature.html) function.

## What It Does

- adaptive finite-domain quadrature on rectangular regions,
- exact per-leaf kernel reuse across repeated integrations,
- optional one-time prepared payload building for mutable node-local state,
- scalar, vector, matrix, complex, and fixed-shape tensor outputs,
- built-in `gk21` for 1D integrals,
- built-in `genz_malik` for `ndim >= 2`.

Each active leaf stores only its bounds, cached payloads at that leaf's rule nodes, and its
current estimate and error. Cached payloads can be raw numeric kernel arrays or prepared per-node
objects. When the parameters change, the mesh is reused as-is: cached payloads stay valid, the
cheap evaluator is recomputed on the current leaves, and refinement only continues if the new
parameter still fails tolerance.

## Current Limits

- finite rectangular domains only,
- no SciPy compatibility wrapper,
- no infinite limits,
- no breakpoint seeding,
- no disk-backed caches or serialization,
- no public custom-rule API.

## Installation

Install from PyPI:

```bash
python -m pip install stateful-quadrature
```

For local development in a clone, prefer Pixi:

```bash
pixi run test
pixi run example
```

If you want an editable Python install in the same clone:

```bash
python -m pip install -e .
```

## Minimal Example

```python
import numpy as np

from stateful_quadrature import StatefulIntegrator


def kernel(points):
    x = points[:, 0]
    return np.stack([np.sin(x), np.cos(x)], axis=-1)


def evaluator(points, payload, params):
    alpha = params["alpha"]
    return payload[:, 0] + alpha * payload[:, 1]


integrator = StatefulIntegrator(
    a=[-1.0],
    b=[1.0],
    kernel=kernel,
    evaluator=evaluator,
    rule="gk21",
)

result = integrator.integrate(alpha=1.5, atol=1e-10, rtol=1e-10)
print(result.status, result.estimate, result.error)
```

The example above integrates `sin(x) + 1.5 cos(x)` over `[-1, 1]`. The returned
`IntegrationResult` also includes `n_kernel_evals`, `n_evaluator_evals`, `n_leaves`,
`n_leaf_nodes`, and `subdivisions`.

An executable copy of this example lives in
[`examples/basic_usage.py`](https://github.com/Kostusas/stateful_quadrature/blob/main/examples/basic_usage.py)
and can be run from a clone with `pixi run example`.

## Prepared Payloads

If raw numeric kernel data should be turned into mutable node-local state once, pass a
`payload_builder`:

```python
import numpy as np

from stateful_quadrature import StatefulIntegrator


def kernel(points):
    return points[:, 0]


def payload_builder(points, raw_payloads):
    return [{"value": float(value), "history": []} for value in raw_payloads]


def evaluator(points, payloads, scale):
    return np.array([scale * payload["value"] for payload in payloads])


integrator = StatefulIntegrator(
    a=[0.0],
    b=[1.0],
    kernel=kernel,
    evaluator=evaluator,
    payload_builder=payload_builder,
)
```

`kernel` still returns numeric arrays. `payload_builder(points, raw_payloads)` runs only when new
rule nodes are created and must return one prepared payload object per input point. The evaluator
then receives a Python `list` of prepared payloads aligned with `points`.

Repeated `integrate(...)` calls reuse the same prepared payload objects on the live mesh, and
`replace_evaluator(...)` shallow-shares those objects into the cloned integrator.

## Reusing The Live Mesh

If a second integral uses the same kernel but a different evaluator, clone the live mesh with:

```python
other = integrator.replace_evaluator(other_evaluator)
```

The new integrator shares the current leaf payload cache but can refine independently from that
point onward. This applies to both numeric payload arrays and prepared payload objects.

`integrate(...)` accepts either a single `params` object or keyword arguments. Keyword arguments
are bundled into a dictionary and passed to the evaluator as the third argument.

## Output Shapes

The evaluator can return any fixed trailing shape. Typical forms are:

- scalar output: `(npoints,)`,
- vector output: `(npoints, m)`,
- matrix output: `(npoints, m, n)`,
- higher-rank tensor output: `(npoints, *shape)`.

The trailing shape must stay fixed for a given integrator instance.

## Development

This repository is Pixi-centric. The default Pixi environment provides the supported local
development workflow.

Run the test suite:

```bash
pixi run test
```

Run the executable example:

```bash
pixi run example
```

Run the optional SciPy benchmark comparison:

```bash
pixi run -e test benchmark-quick
```

The benchmark script is for local comparison work and is not part of the package runtime
dependency set.

## Maintainers

Maintainer workflow notes live in
[`RELEASING.md`](https://github.com/Kostusas/stateful_quadrature/blob/main/RELEASING.md).

For design background on the leaf-only reuse model, see
[`DESIGN.md`](https://github.com/Kostusas/stateful_quadrature/blob/main/DESIGN.md).
