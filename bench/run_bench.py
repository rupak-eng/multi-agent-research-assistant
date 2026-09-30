"""Benchmark harness: fixed-topic end-to-end research runs.

Labels (exact):
  Deterministic/Stub            -> stub LLM + stub search (no network, no cost)
  Real Provider: Groq <model> + Tavily -> Groq chat completions + Tavily search

Records per run: latency, input/output tokens, searches, routing decisions,
citation-validation outcome, status. Summary: p50/p95 latency, token totals,
searches/run, routing distribution, citation pass rate, cost (published
Groq pricing; Tavily usage-based, reported as searches).

Usage:
  python bench/run_bench.py --provider stub   # deterministic, offline
  python bench/run_bench.py --provider real   # Groq gpt-oss-20b + Tavily
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import Settings
from app.graph.builder import build_graph
from app.graph.nodes import make_deps
from app.providers.factory import build_llm, build_search
from app.services.orchestrator import build_initial_state, drive_run
from app.state.schemas import RunBudgets
from app.storage.checkpoint import RedisCheckpointer
from app.storage.registry import RedisRunRegistry, new_run_id

# Groq published pricing (USD per 1M tokens), openai/gpt-oss-20b.
# Sources: console.groq.com/docs/models pricing page (via public mirrors,
# 2026-09-30). Tavily is usage-credit based; reported as searches/run.
GROQ_PRICE_IN_PER_1M = 0.075
GROQ_PRICE_OUT_PER_1M = 0.30

TOPICS = [
    "What are vector databases and how do they power RAG pipelines?",
    "How does LangGraph handle state and checkpointing for AI agents?",
    "What are the trade-offs of hybrid search versus pure vector search?",
    "How do AI agents use tools safely in production systems?",
    "What is retrieval-augmented generation and how is it evaluated?",
]


def percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * (p / 100.0)
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def run_once(graph, registry, topic: str, budgets: RunBudgets) -> dict:
    run_id = new_run_id()
    initial = build_initial_state(run_id, topic, budgets)
    registry.create_run(run_id, topic, json.dumps(initial))
    t0 = time.monotonic()
    final = drive_run(graph, registry, run_id, initial)
    latency = time.monotonic() - t0
    usage = final.get("usage") or {}
    trace = final.get("trace") or []
    routing = {}
    for step in trace:
        rd = (step.get("routing_decision") or {})
        nxt = rd.get("next")
        if nxt:
            routing[nxt] = routing.get(nxt, 0) + 1
    validate_steps = [s for s in trace if s["node"] == "validate"]
    citation_pass = (final.get("status") == "COMPLETED"
                     and bool(validate_steps)
                     and validate_steps[-1].get("error") is None)
    return {
        "run_id": run_id,
        "topic": topic,
        "status": final.get("status"),
        "latency_sec": round(latency, 2),
        "tokens_in": usage.get("input_tokens", 0),
        "tokens_out": usage.get("output_tokens", 0),
        "searches": usage.get("searches", 0),
        "findings": len(final.get("findings") or []),
        "routing_decisions": routing,
        "trace_steps": len(trace),
        "citation_pass": citation_pass,
        "error": final.get("error"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["stub", "real"], default="stub")
    ap.add_argument("--topics", type=int, default=5,
                    help="number of fixed topics (max 5)")
    ap.add_argument("--redis-db", type=int, default=15)
    ap.add_argument("--delay", type=float, default=0,
                    help="seconds to wait between topics (rate-limit pacing)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.provider == "stub":
        label = "Deterministic/Stub"
        settings = Settings(llm_provider="stub", search_provider="stub",
                            redis_url=f"redis://localhost:6379/{args.redis_db}")
    else:
        model = "openai/gpt-oss-20b"
        label = f"Real Provider: Groq {model} + Tavily"
        settings = Settings(llm_provider="groq", search_provider="tavily",
                            groq_model=model,
                            redis_url=f"redis://localhost:6379/{args.redis_db}")

    import redis
    client = redis.Redis.from_url(settings.redis_url, decode_responses=False)
    client.ping()
    registry = RedisRunRegistry(client)
    deps = make_deps(settings, registry, llm=build_llm(settings),
                     search=build_search(settings))
    graph = build_graph(deps, RedisCheckpointer(client))

    topics = TOPICS[:args.topics]
    budgets = RunBudgets(max_subquestions=3, max_searches=9)
    started = datetime.now(timezone.utc)

    runs = []
    for i, topic in enumerate(topics, 1):
        if i > 1 and args.delay > 0:
            print(f"    pacing: sleeping {args.delay}s...", flush=True)
            time.sleep(args.delay)
        print(f"[{i}/{len(topics)}] {topic[:60]}...", flush=True)
        try:
            runs.append(run_once(graph, registry, topic, budgets))
        except Exception as e:  # noqa: BLE001 - record provider failures honestly
            runs.append({"run_id": None, "topic": topic, "status": "ERROR",
                         "latency_sec": 0.0, "tokens_in": 0, "tokens_out": 0,
                         "searches": 0, "findings": 0, "routing_decisions": {},
                         "trace_steps": 0, "citation_pass": False,
                         "error": str(e)[:300]})
        print(f"    -> {runs[-1]['status']} "
              f"{runs[-1]['latency_sec']}s "
              f"tok={runs[-1]['tokens_in']}/{runs[-1]['tokens_out']} "
              f"searches={runs[-1]['searches']} "
              f"cite={'pass' if runs[-1]['citation_pass'] else 'FAIL'}",
              flush=True)

    ok = [r for r in runs if r["status"] == "COMPLETED"]
    lat = [r["latency_sec"] for r in ok]
    tin = sum(r["tokens_in"] for r in ok)
    tout = sum(r["tokens_out"] for r in ok)
    routing_total: dict[str, int] = {}
    for r in ok:
        for k, v in r["routing_decisions"].items():
            routing_total[k] = routing_total.get(k, 0) + v

    if args.provider == "real":
        cost = tin / 1e6 * GROQ_PRICE_IN_PER_1M + tout / 1e6 * GROQ_PRICE_OUT_PER_1M
        cost_note = (f"Groq {settings.groq_model} published pricing "
                     f"(${GROQ_PRICE_IN_PER_1M}/1M in, ${GROQ_PRICE_OUT_PER_1M}/1M out); "
                     f"Tavily usage-credit cost not included")
    else:
        cost, cost_note = 0.0, "stub providers: no real cost"

    result = {
        "label": label,
        "provider": {"llm": settings.llm_provider,
                     "llm_model": getattr(settings, "groq_model", None)
                     if args.provider == "real" else "stub-llm-v1",
                     "search": settings.search_provider},
        "dataset": {"topics": topics, "size": len(topics)},
        "budgets": budgets.model_dump(),
        "started_utc": started.isoformat(),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "runs": runs,
        "summary": {
            "completed": len(ok),
            "failed": len(runs) - len(ok),
            "latency_sec_p50": round(percentile(lat, 50), 2),
            "latency_sec_p95": round(percentile(lat, 95), 2),
            "latency_sec_mean": round(statistics.mean(lat), 2) if lat else 0.0,
            "tokens_in_total": tin,
            "tokens_out_total": tout,
            "tokens_in_mean": round(tin / len(ok), 1) if ok else 0.0,
            "tokens_out_mean": round(tout / len(ok), 1) if ok else 0.0,
            "searches_total": sum(r["searches"] for r in ok),
            "searches_per_run_mean": (round(sum(r["searches"] for r in ok) / len(ok), 2)
                                      if ok else 0.0),
            "routing_decisions": routing_total,
            "citation_pass_rate": (round(sum(r["citation_pass"] for r in ok) / len(ok), 3)
                                   if ok else 0.0),
            "estimated_cost_usd": round(cost, 6),
            "cost_note": cost_note,
        },
    }

    out_path = (Path(args.out) if args.out else
                Path(__file__).parent / "results" /
                f"bench_{args.provider}_{started.strftime('%Y%m%dT%H%M%SZ')}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nwrote {out_path}")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
