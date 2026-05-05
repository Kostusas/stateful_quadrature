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
   `.github/workflows/publish.yml` only runs for published releases.
5. Approve the protected GitHub Actions environment named `pypi`.

## External Setup

1. In GitHub repository settings, create an environment named `pypi`.
2. Add required reviewer protection to that environment.
3. In PyPI account publishing settings, create a pending publisher with:

   ```text
   PyPI project name: stateful-quadrature
   Owner: Kostusas
   Repository name: stateful_quadrature
   Workflow name: publish.yml
   Environment name: pypi
   ```

4. Do not store a long-lived PyPI API token in GitHub Secrets for this flow.
