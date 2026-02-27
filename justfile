set shell := ["bash", "-lc"]

default:
    just --list

init:
    just venv && \
    just sync

venv:
    test -d .venv || uv venv

sync:
    uv sync --all-packages

sync-lock:
    uv sync --locked --all-packages

up:
    uv lock --upgrade

add PKG:
    uv add {{PKG}}

add-dev PKG:
    uv add --dev {{PKG}}

rm PKG:
    uv remove {{PKG}}

rm-dev PKG:
    uv remove --dev {{PKG}}

fmt:
    just sync && \
    uv run ruff format .

fmt-check:
    uv run ruff format --check .

lint:
    uv run ruff check . --fix

lint-check:
    uv run ruff check .

test:
    uv run pytest

ci:
    just venv && \
    just sync-lock && \
    just fmt-check && \
    just lint-check && \
    just test

docs-build:
	uv run pdoc --math -o site te_net_lib

docs-serve:
    uv run pdoc --math -p 8080 te_net_lib

ex-qs:
    just ex-qs-lp && \
    just ex-qs-l && \
    just ex-qs-o && \
    just ex-qs-prn

ex-qs-lp:
    uv run python examples/quickstart_lasso_path.py

ex-qs-l:
    uv run python examples/quickstart_lasso.py

ex-qs-o:
    uv run python examples/quickstart_ols.py

ex-qs-prn:
    uv run python examples/quickstart_pipeline_raw_neutral.py

BENCH_REPEATS := "3"
BENCH_WARMUP := "1"
BENCH_SEED := "0"

bench:
    just bench-gs && \
    just bench-lp && \
    just bench-lt && \
    just bench-ot && \
    just bench-pf

bench-gs:
    uv run python -m bench.bench_graph_select --repeats {{BENCH_REPEATS}} --warmup {{BENCH_WARMUP}} --seed {{BENCH_SEED}}

bench-lp:
    uv run python -m bench.bench_lasso_path --repeats {{BENCH_REPEATS}} --warmup {{BENCH_WARMUP}} --seed {{BENCH_SEED}}

bench-lt:
    uv run python -m bench.bench_lasso_te --repeats {{BENCH_REPEATS}} --warmup {{BENCH_WARMUP}} --seed {{BENCH_SEED}}

bench-ot:
	uv run python -m bench.bench_ols_te --repeats {{BENCH_REPEATS}} --warmup {{BENCH_WARMUP}} --seed {{BENCH_SEED}}
	
bench-pf:
	uv run python -m bench.bench_preprocess_fn --repeats {{BENCH_REPEATS}} --warmup {{BENCH_WARMUP}} --seed {{BENCH_SEED}}
