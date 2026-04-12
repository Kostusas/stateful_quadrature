"""Stateful adaptive quadrature with exact node reuse."""

from ._compat import CubatureResult, cubature, quad_vec
from ._integrator import IntegrationResult, StatefulIntegrator

__all__ = [
    "CubatureResult",
    "IntegrationResult",
    "StatefulIntegrator",
    "cubature",
    "quad_vec",
]
