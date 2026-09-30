.PHONY: setup test lint serve worker demo bench eval-sample push

setup:
	python3 -m venv .venv
	.venv/bin/pip install -q -r requirements.txt
	@echo "copy .env.example to .env and adjust"

test:
	.venv/bin/python -m pytest -q

lint:
	.venv/bin/ruff check app tests bench
	.venv/bin/ruff format --check app tests bench

serve:
	.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8001

worker:
	.venv/bin/python -m app.workers.runner

demo:
	.venv/bin/python demo/run_demo.py

bench:
	.venv/bin/python bench/run_bench.py --runs 8

eval-sample:
	.venv/bin/python bench/make_eval_samples.py

push:
	python3 ~/workspace/portfolio/ref/gh_push.py . main
