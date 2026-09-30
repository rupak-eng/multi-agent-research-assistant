"""Deterministic stub search provider for tests, benchmarks and offline demos.

Returns canned results keyed by topic keywords found in the query. Unknown
queries get a fixed generic set (deterministic). URLs live under
stub-search.local so synthetic data is never mistaken for the real web.
"""

from __future__ import annotations

import re

from .base import SearchHit, SearchProvider

_ARTICLES: dict[str, list[dict[str, str]]] = {
    "vector database": [
        {"slug": "vector-db-survey-2026", "title": "Vector database landscape 2026: benchmarks and trade-offs",
         "snippet": "We benchmarked 8 vector databases on 10M-vector workloads. HNSW recall at 0.99 required 2.1x the memory of IVF-PQ. Write throughput varied 40x between systems."},
        {"slug": "hnsw-tuning-guide", "title": "HNSW parameter tuning: efConstruction vs query latency",
         "snippet": "Increasing efConstruction from 100 to 400 improved recall@10 from 0.94 to 0.985 while raising index build time 3.4x and memory 1.6x on SIFT1M."},
        {"slug": "hybrid-search-pgvector", "title": "Hybrid search with pgvector: BM25 + dense fusion",
         "snippet": "Combining BM25 lexical scores with dense embeddings via reciprocal rank fusion lifted NDCG@10 by 12 points over dense-only on BEIR-style enterprise corpora."},
        {"slug": "vector-db-costs", "title": "The real cost of managed vector databases",
         "snippet": "At 50M vectors, managed offerings ranged $380-$2,400/month. Self-hosted on a single 64GB node handled the same workload for ~$180/month in compute."},
    ],
    "langgraph": [
        {"slug": "supervisor-pattern", "title": "Supervisor agent pattern in LangGraph: routing and review",
         "snippet": "A supervisor node inspects worker outputs against the plan and routes to the next worker. Deterministic routing functions beat LLM-based routers on reliability in production traces."},
        {"slug": "langgraph-checkpointing", "title": "LangGraph checkpointing: resumable agent executions",
         "snippet": "Checkpoints persist channel state after every node. Resuming with a thread_id replays from the last completed node instead of restarting the run."},
        {"slug": "langgraph-vs-autogen", "title": "LangGraph vs AutoGen for production orchestration",
         "snippet": "LangGraph's explicit state graph suits auditable pipelines; AutoGen's conversational model fits exploratory tasks. Teams report fewer runaway loops with explicit graphs."},
    ],
    "rag evaluation": [
        {"slug": "ragas-metrics", "title": "RAGAS: reference-free metrics for RAG pipelines",
         "snippet": "RAGAS measures faithfulness, answer relevancy and context precision without human labels, using LLM judges. Faithfulness correlates 0.78 with human ratings on WikiEval."},
        {"slug": "deepeval-llm-judge", "title": "DeepEval: unit-testing LLMs with G-Eval style judges",
         "snippet": "DeepEval wraps LLM-as-judge scoring in pytest-style test cases with configurable thresholds, enabling regression detection in CI pipelines."},
        {"slug": "rag-eval-pitfalls", "title": "Where RAG evaluation goes wrong",
         "snippet": "Common failure: evaluating retrieval and generation jointly hides which stage regressed. Measure retrieval (recall@k) and generation (faithfulness) separately."},
    ],
    "knowledge graph": [
        {"slug": "graphrag-microsoft", "title": "GraphRAG: knowledge graphs meet retrieval",
         "snippet": "Microsoft's GraphRAG builds entity/relationship graphs from documents and uses community summaries for global questions, beating vector-only RAG on multi-hop queries."},
        {"slug": "neo4j-vector-search", "title": "Neo4j vector search: graphs plus embeddings",
         "snippet": "Neo4j 5.x supports vector indexes alongside Cypher traversals, enabling hybrid queries that filter by graph structure then rank by embedding similarity."},
        {"slug": "entity-resolution-rag", "title": "Entity resolution in Graph RAG pipelines",
         "snippet": "Merging duplicate entities ('NYC' vs 'New York City') before graph construction cut disconnected components by 61% in a SEC-filings corpus study."},
    ],
    "llm cost": [
        {"slug": "llm-cost-optimization", "title": "Cutting LLM inference spend: caching, routing, batching",
         "snippet": "Semantic caching cut repeat-query spend 34%. Routing simple queries to small models saved another 41%. Combined with batching, teams report 60-70% cost reduction."},
        {"slug": "token-budgeting", "title": "Per-run token budgets for agent systems",
         "snippet": "Enforcing hard token caps per run stopped runaway agent loops. Median agent run cost dropped from $0.42 to $0.09 after budgets were introduced."},
    ],
    "prompt injection": [
        {"slug": "indirect-injection", "title": "Indirect prompt injection via retrieved documents",
         "snippet": "Attackers plant instructions in web pages that RAG systems retrieve. In tests, 23% of agents executed injected instructions when no output validation was present."},
        {"slug": "prompt-injection-defenses", "title": "Defenses: delimiters, classifiers, and least privilege",
         "snippet": "Layered defenses — input delimiters, an injection classifier on retrieved text, and tool-level least privilege — blocked 97% of test attacks in combination."},
    ],
}

_GENERIC: list[dict[str, str]] = [
    {"slug": "overview-methods", "title": "Methods overview: current approaches",
     "snippet": "Recent surveys compare the main approaches on accuracy, latency and cost. Hybrid methods that combine complementary techniques lead on complex queries."},
    {"slug": "benchmarks-2026", "title": "2026 benchmark roundup",
     "snippet": "Standard benchmarks show a 15-25 point spread between naive baselines and tuned systems. Evaluation methodology matters more than model choice at the top end."},
    {"slug": "production-lessons", "title": "Lessons from production deployments",
     "snippet": "Teams report that monitoring, budgets and graceful degradation matter more than peak benchmark scores. Most outages came from unhandled edge cases, not model quality."},
]


def _topic_for(query: str) -> str | None:
    q = query.lower()
    for topic in _ARTICLES:
        if all(w in q for w in topic.split()):
            return topic
    for topic in _ARTICLES:
        if any(w in q for w in topic.split() if len(w) > 3):
            return topic
    return None


class StubSearchProvider(SearchProvider):
    """Deterministic canned search. Also counts calls (for idempotency tests)."""

    name = "stub"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def search(self, query: str, max_results: int = 5) -> list[SearchHit]:
        self.calls.append(query)
        topic = _topic_for(query)
        articles = _ARTICLES[topic] if topic else _GENERIC
        hits = []
        for a in articles[:max_results]:
            url = f"https://stub-search.local/{a['slug']}"
            hits.append(SearchHit(url=url, title=a["title"], snippet=a["snippet"]))
        return hits

    def reset(self) -> None:
        self.calls.clear()


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40]
