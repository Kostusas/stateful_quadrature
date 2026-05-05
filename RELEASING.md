# Maintainer Workflow

This repository is PyPI-publish and Pixi centric.

## Local Workflow

1. Start from a clean working tree.
2. Run the main checks through Pixi:

   ```bash
   pixi run test
   pixi run example
   ```

3. If you want the SciPy comparison benchmark, run:

   ```bash
   pixi run -e test benchmark-quick
   ```

## Versioning And GitHub Releases

1. Update `stateful_quadrature/_version.py`, `pyproject.toml`, and any relevant docs when the
   version changes.
2. Keep the README install path pointed at:

   ```bash
   python -m pip install stateful-quadrature
   ```

3. Push the release commit and matching version tag:

   ```bash
   git tag v0.2.0
   git push origin main --tags
   ```

4. Create and publish a GitHub Release for the same tag. The repository workflow at
   `.github/workflows/publish.yml` only runs when the release is published.
5. Approve the protected GitHub Actions environment named `pypi` when the publish job reaches
   its deployment gate.
6. After approval, GitHub Actions builds the sdist and wheel, validates that the release tag
   matches both version files, and publishes the distributions to PyPI through Trusted
   Publishing.

## External Setup

1. In GitHub repository settings, create an environment named `pypi`.
2. Add required reviewer protection to that environment.
3. Restrict deployments to release tags if you use environment deployment policies.
4. In PyPI account publishing settings, create a pending publisher with:

   ```text
   PyPI project name: stateful-quadrature
   Owner: Kostusas
   Repository name: stateful_quadrature
   Workflow name: publish.yml
   Environment name: pypi
   ```

5. Do not store a long-lived PyPI API token in GitHub Secrets for this flow.

## Notes

- The benchmark script under `benchmarks/` is optional developer tooling and is not part of the
  package runtime dependency set.
- For a new PyPI project, use a pending publisher on PyPI so the first successful release can
  create the project automatically.
