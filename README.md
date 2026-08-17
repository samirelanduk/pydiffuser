# pydiffuser

[![CI](https://img.shields.io/github/actions/workflow/status/samirelanduk/pydiffuser/ci.yml?branch=master&logo=github&label=CI)](https://github.com/samirelanduk/pydiffuser/actions/workflows/ci.yml)
[![Python versions](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-blue?logo=python&logoColor=white)](https://github.com/samirelanduk/pydiffuser/blob/master/pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Checked with pyright](https://img.shields.io/badge/pyright-checked-2a6db2)](https://github.com/microsoft/pyright)

A python library for generating media with diffusion.

## Development

The project uses [uv](https://docs.astral.sh/uv/). With it installed:

```bash
git clone git@github.com:samirelanduk/pydiffuser.git
cd pydiffuser
uv sync
pre-commit install
```

`uv sync` creates the virtualenv in `.venv` and installs the project with its dev
dependencies from the lockfile. `pre-commit install` sets up the ruff check and
format hooks, which CI also enforces.

Then, to run the checks:

```bash
uv run python -m unittest discove
uv run pyright
uv run pre-commit run --all-files
```

