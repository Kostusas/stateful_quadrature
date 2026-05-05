"""Leaf-only stateful adaptive cubature."""

from typing import TYPE_CHECKING

from ._version import __version__

if TYPE_CHECKING:
    from ._integrator import IntegrationResult, StatefulIntegrator

__all__ = [
    "IntegrationResult",
    "StatefulIntegrator",
    "__version__",
]


def __getattr__(name):
    if name in {"IntegrationResult", "StatefulIntegrator"}:
        from ._integrator import IntegrationResult, StatefulIntegrator

        return {
            "IntegrationResult": IntegrationResult,
            "StatefulIntegrator": StatefulIntegrator,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
