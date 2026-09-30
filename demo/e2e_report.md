# Research report: How do vector databases power RAG pipelines?

This report synthesizes 9 findings from 3 sources.

## Finding group 1 (sq1)

We benchmarked 8 vector databases on 10M-vector workloads [1] Increasing efConstruction from 100 to 400 improved recall@10 from 0.94 to 0.985 while raising index build time 3.4x and memory 1.6x on SIFT1M. [2] Combining BM25 lexical scores with dense embeddings via reciprocal rank fusion lifted NDCG@10 by 12 points over dense-only on BEIR-style enterprise corpora. [3]

## Finding group 2 (sq2)

We benchmarked 8 vector databases on 10M-vector workloads [4] Increasing efConstruction from 100 to 400 improved recall@10 from 0.94 to 0.985 while raising index build time 3.4x and memory 1.6x on SIFT1M. [5] Combining BM25 lexical scores with dense embeddings via reciprocal rank fusion lifted NDCG@10 by 12 points over dense-only on BEIR-style enterprise corpora. [6]

## Finding group 3 (sq3)

We benchmarked 8 vector databases on 10M-vector workloads [7] Increasing efConstruction from 100 to 400 improved recall@10 from 0.94 to 0.985 while raising index build time 3.4x and memory 1.6x on SIFT1M. [8] Combining BM25 lexical scores with dense embeddings via reciprocal rank fusion lifted NDCG@10 by 12 points over dense-only on BEIR-style enterprise corpora. [9]

## Sources

[1] Vector database landscape 2026: benchmarks and trade-offs — https://stub-search.local/vector-db-survey-2026
[2] HNSW parameter tuning: efConstruction vs query latency — https://stub-search.local/hnsw-tuning-guide
[3] Hybrid search with pgvector: BM25 + dense fusion — https://stub-search.local/hybrid-search-pgvector
[4] Vector database landscape 2026: benchmarks and trade-offs — https://stub-search.local/vector-db-survey-2026
[5] HNSW parameter tuning: efConstruction vs query latency — https://stub-search.local/hnsw-tuning-guide
[6] Hybrid search with pgvector: BM25 + dense fusion — https://stub-search.local/hybrid-search-pgvector
[7] Vector database landscape 2026: benchmarks and trade-offs — https://stub-search.local/vector-db-survey-2026
[8] HNSW parameter tuning: efConstruction vs query latency — https://stub-search.local/hnsw-tuning-guide
[9] Hybrid search with pgvector: BM25 + dense fusion — https://stub-search.local/hybrid-search-pgvector
