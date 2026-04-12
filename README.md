# Stateful Quadrature

`stateful_quadrature` is a NumPy-first adaptive quadrature package for parameter sweeps of

$$
I(\lambda) = \int_\Omega \Psi(x, K(x), \lambda)\, dx
$$

where `K(x)` is expensive, independent of `lambda`, and worth caching exactly.

The package keeps:

- the adaptive leaf partition,
- the exact quadrature nodes that were already visited,
- the cached kernel payloads at those nodes.

For each new parameter value, it recomputes the cheap dynamic evaluator on the current leaves and
refines only if the requested tolerance is not yet met.

Alongside the stateful API, the package also exposes SciPy-style compatibility wrappers:

- `stateful_quadrature.quad_vec`
- `stateful_quadrature.cubature`

## Install and test

Use Pixi for environments:

```bash
pixi run test
pixi run -e test test
```

The default environment contains only NumPy. The `test` environment adds SciPy for comparison
tests and benchmarks.

## Benchmarks

Run the SciPy-inspired comparison harness with:

```bash
pixi run -e test python benchmarks/compare_scipy.py --mode quick
```

`--mode full` adds heavier oscillatory cubature cases. The benchmark suite mirrors the same
problem families SciPy uses in its ASV integration benchmarks (`quad_vec` oscillatory,
`cubature` sphere, and oscillatory cubature), and also includes a stateful parameter-sweep case
to measure exact node reuse against repeated SciPy solves.

## Minimal example

```python
import numpy as np

from stateful_quadrature import StatefulIntegrator


def kernel(points):
    x = points[:, 0]
    return np.stack([np.sin(x), np.cos(x)], axis=-1)


def evaluator(points, payload, alpha):
    return payload[:, 0] + alpha * payload[:, 1]


integrator = StatefulIntegrator(
    a=[-1.0],
    b=[1.0],
    kernel=kernel,
    evaluator=evaluator,
    rule="gk21",
)

result = integrator.integrate(alpha=1.5, atol=1e-10, rtol=1e-10)
print(result.estimate, result.error, result.n_cached_nodes)
```
