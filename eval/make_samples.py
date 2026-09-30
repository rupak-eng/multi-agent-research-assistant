"""Generate eval/sample_outputs.jsonl from actual research runs.

Each line: {"input", "output", "contexts", "expected", "metadata"}.
Sibling Project 3 consumes these as eval samples. All outputs are real
system outputs (stub or real provider), never hand-written.

Usage:
  python eval/make_samples.py --provider stub --out eval/sample_outputs.jsonl
  python eval/make_samples.py --provider real --out eval/sample_outputs_real.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import redis

from app.config import Settings
from app.graph.builder import build_graph
from app.graph.nodes import make_deps
from app.providers.factory import build_llm, build_search
from app.services.orchestrator import build_initial_state, drive_run
from app.state.schemas import RunBudgets
from app.storage.checkpoint import RedisCheckpointer
from app.storage.registry import RedisRunRegistry, new_run_id
from bench.run_bench import TOPICS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["stub", "real"], default="stub")
    ap.add_argument("--out", default="eval/sample_outputs.jsonl")
    ap.add_argument("--topics", type=int, default=5)
    ap.add_argument("--redis-db", type=int, default=15)
    args = ap.parse_args()

    if args.provider == "stub":
        settings = Settings(llm_provider="stub", search_provider="stub",
                            redis_url=f"redis://localhost:6379/{args.redis_db}")
    else:
        settings = Settings(llm_provider="groq", search_provider="tavily",
                            groq_model="openai/gpt-oss-20b",
                            redis_url=f"redis://localhost:6379/{args.redis_db}")

    client = redis.Redis.from_url(settings.redis_url, decode_responses=False)
    client.ping()
    registry = RedisRunRegistry(client)
    deps = make_deps(settings, registry, llm=build_llm(settings),
                     search=build_search(settings))
    graph = build_graph(deps, RedisCheckpointer(client))
    budgets = RunBudgets(max_subquestions=3, max_searches=9)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w") as f:
        for topic in TOPICS[:args.topics]:
            run_id = new_run_id()
            initial = build_initial_state(run_id, topic, budgets)
            registry.create_run(run_id, topic, json.dumps(initial))
            try:
                final = drive_run(graph, registry, run_id, initial)
            except Exception as e:  # noqa: BLE001 - record honestly, skip
                print(f"SKIP {topic[:50]}: {e}")
                continue
            if final.get("status") != "COMPLETED":
                print(f"SKIP {topic[:50]}: status={final.get('status')}")
                continue
            for r in final.get("researcher_results") or []:
                contexts = [
                    {"text": fnd.get("snippet", ""),
                     "source": fnd.get("url", ""),
                     "chunk_id": fnd.get("id", "")}
                    for fnd in r.get("findings", [])
                ]
                sample = {
                    "input": r.get("subquestion_id", "") + ": " + topic,
                    "output": r.get("summary", ""),
                    "contexts": contexts,
                    "expected": ("a concise, citation-backed summary answering "
                                 "the sub-question"),
                    "metadata": {
                        "run_id": run_id,
                        "provider": args.provider,
                        "model": getattr(settings, "groq_model", "stub-llm-v1")
                        if args.provider == "real" else "stub-llm-v1",
                        "search": settings.search_provider,
                        "status": r.get("status"),
                        "generated_utc": datetime.now(timezone.utc).isoformat(),
                    },
                }
                f.write(json.dumps(sample) + "\n")
                n += 1
    print(f"wrote {n} samples to {out}")


if __name__ == "__main__":
    main()
