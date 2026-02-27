[![CI](https://github.com/yifen9/te-net-lib/actions/workflows/ci.yaml/badge.svg)](https://github.com/yifen9/te-net-lib/actions/workflows/ci.yaml) [![Docs](https://github.com/yifen9/te-net-lib/actions/workflows/docs.yaml/badge.svg)](https://github.com/yifen9/te-net-lib/actions/workflows/docs.yaml) [![Image Dev](https://github.com/yifen9/te-net-lib/actions/workflows/image-dev.yaml/badge.svg)](https://github.com/yifen9/te-net-lib/actions/workflows/image-dev.yaml) [![Release](https://github.com/yifen9/te-net-lib/actions/workflows/release.yaml/badge.svg)](https://github.com/yifen9/te-net-lib/actions/workflows/release.yaml)

# te-net-lib

## Installation

### Install from GitHub

```bash
python -m pip install "git+https://github.com/yifen9/te-net-lib.git@main
```

### Install from a GitHub Release artifact

1. Download the wheel (`.whl`) from the GitHub Release page.
2. Install it locally:

```bash
python -m pip install ./te_net_lib-*.whl
```

### Development install

```bash
uv venv
uv sync
uv pip install -e .
```

## Examples

Quickstart scripts live in `examples/` and are designed to be minimal, runnable, and suitable for documentation snippets.

- OLS TE + fixed-density graph + PR/F1:

```bash
uv run python examples/quickstart_ols.py
```

- Lasso TE + fixed-density graph + PR/F1:

```bash
uv run python examples/quickstart_lasso.py
```

- Raw vs factor-neutral pipeline (OLS) + NIO:

```bash
uv run python examples/quickstart_pipeline_raw_neutral.py
```

- Lasso TE path (alpha grid) + graph metrics:

```bash
uv run python examples/quickstart_lasso_path.py
```

## Development

This repository uses `just` to drive common tasks.

- Run tests:

```bash
just test
```

- Formatting:

```bash
just fmt
```

- Run examples:

```bash
just ex-qs
```

- Benchmarks:

```bash
just bench
```

- Build local documentation (pdoc):

```bash
just docs-build
```

- CI entry point (used by GitHub Actions):

```bash
just ci
```

## License

te-net-lib is released under the MIT License. See `LICENSE` for details.
