# Maintainer Workflow

This repository is currently GitHub-source-install and Pixi centric.

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
   python -m pip install git+https://github.com/Kostusas/stateful_quadrature.git
   ```

3. After the final commit, create a tag for the version you want users to install from:

   ```bash
   git tag v0.1.0
   git push origin main --tags
   ```

4. If you publish a GitHub release, attach notes there. A package index release can be added
   later when the project is ready for that distribution path.

## Notes

- The benchmark script under `benchmarks/` is optional developer tooling and is not part of the
  package runtime dependency set.
- The packaging metadata is kept valid so `pip install git+...` works cleanly from the GitHub
  repository, even though package-index publication is intentionally deferred.
