import numpy as np

from stateful_quadrature import StatefulIntegrator


def kernel(points):
    x = points[:, 0]
    return np.stack([np.sin(x), np.cos(x)], axis=-1)


def evaluator(points, payload, params):
    alpha = params["alpha"]
    return payload[:, 0] + alpha * payload[:, 1]


if __name__ == "__main__":
    integrator = StatefulIntegrator(
        a=[-1.0],
        b=[1.0],
        kernel=kernel,
        evaluator=evaluator,
        rule="gk21",
    )

    result = integrator.integrate(alpha=1.5, atol=1e-10, rtol=1e-10)
    print(result.status, result.estimate, result.error)
