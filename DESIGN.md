# Adaptive Quadrature Reuse

## Problem

We want to evaluate a family of parameter-dependent integrals

$$
I(\lambda) = \int_\Omega \Psi(x, K(x), \lambda)\, dx
$$

where:

- $K(x)$ is expensive and does not depend on $\lambda$,
- $\Psi$ is cheap once the kernel payload is known,
- each requested $\lambda$ still needs its own adaptive error control.

## Leaf-Only State

The package uses a fixed embedded rule and an adaptive leaf mesh.
The mutable state is only the current live leaf set.

Each active leaf stores:

- its bounds,
- cached kernel payloads at that leaf's rule nodes,
- its current estimate,
- its current error indicator.

The quadrature rule itself is global and immutable:

- reference nodes on $[-1, 1]^n$,
- high-order weights,
- low-order weights.

These are never copied into the mutable state.

## Reuse Contract

For a new $\lambda$:

1. keep the current live leaf set,
2. remap the rule nodes for each leaf from its bounds,
3. reuse the cached kernel payloads stored on that leaf,
4. recompute the cheap evaluator on the live leaf nodes,
5. recompute the leaf estimates and errors,
6. refine only if the requested tolerance is still not met.

This gives exact reuse of the expensive kernel data without global point hashing, cross-leaf node
deduplication, or historical tree storage.

## Refinement

When refinement is required:

1. pop the worst leaf from the error heap,
2. split it into children,
3. evaluate the expensive kernel only at the children's rule nodes,
4. compute the child estimates and errors,
5. update the global totals,
6. discard the parent leaf completely.

For Gauss-Kronrod and Genz-Malik style subdivision, parent nodes are not retained after splitting.

## Summary

The package is intentionally solving one problem:

> adaptive finite-domain cubature with exact reuse of per-leaf kernel payloads across repeated
> integrations.

The design is leaf-only because that is the smallest state that still preserves the reuse that
matters for the target workload.
