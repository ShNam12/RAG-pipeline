# Repository Guidelines

## Project Structure and Module Organization

This is a small Python package using the `src` layout.
Package code lives in `src/rag1/`; `src/rag1/__init__.py` currently defines the `rag1` command's `main()` function.
Project metadata, the command entry point, and the build backend are declared in `pyproject.toml`.
Place new package modules under `src/rag1/` and add focused tests under `tests/` as behavior grows.
There are no assets or test files yet.

## Build, Test, and Development Commands

- `uv sync` creates the local environment and installs the package using the Python 3.12 version selected by `.python-version`.
- `uv run rag1` runs the command line entry point; it currently prints a greeting.
- `uv build` creates source and wheel distributions with `uv_build`.
- `uv run python -m unittest discover -s tests` runs tests once a `tests/` directory is added.

The project currently has no runtime dependencies or configured test, lint, or formatting tools.
Add tool configuration to `pyproject.toml` when it becomes necessary, and document the corresponding command here.

## Coding Style and Naming Conventions

Use four spaces for indentation and follow standard Python naming: `snake_case` for modules and functions, `PascalCase` for classes, and `UPPER_CASE` for constants.
Keep type annotations on public functions, as the existing `main() -> None` does.
Keep changes close to the module that owns the behavior.

## Testing Guidelines

There is no test framework or coverage threshold configured yet.
For new behavior, add focused standard-library `unittest` cases named `tests/test_*.py` and cover important success and failure paths.
Run the test command above after adding tests.

## Commits and Pull Requests

This repository has no commit history, so no established commit-message convention exists.
Use a short, imperative subject that describes the change, and keep commits focused.
In pull requests, summarize the behavior, list the verification performed, and link a related issue when one exists.
Include screenshots only for changes that affect a visible interface.
