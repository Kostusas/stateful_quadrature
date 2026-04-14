"""Leaf-only stateful adaptive cubature."""

from ._integrator import IntegrationResult, StatefulIntegrator
from ._version import __version__

__all__ = [
    "IntegrationResult",
    "StatefulIntegrator",
    "__version__",
]
