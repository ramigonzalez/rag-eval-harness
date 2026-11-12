# scifact retrieval results

5,183 documents, 100 labeled queries, mean 1.15 relevant documents per query, mean document length 214.6 words.

## Chunking

| Chunker | Chunks | Avg words | Chunks per doc | Chunk time (s) | Index time (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| whole | 5,183 | 214.6 | 1.00 | 0.0 | 0.6 |
| fixed | 15,596 | 88.0 | 3.01 | 0.1 | 0.9 |
| recursive | 15,727 | 78.1 | 3.03 | 0.3 | 0.9 |
| semantic | 9,879 | 112.6 | 1.91 | 0.7 | 0.9 |

Index time is shared by every retriever on that chunker. It includes embedding when a dense index is built.

## Retrieval

| Chunker | Retriever | Recall@10 | MRR@10 | nDCG@10 | Context precision@10 | Online latency (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| whole | bm25 | 0.790 | 0.677 | 0.696 | 0.671 | 7.4 |
| whole | rerank | 0.820 | 0.657 | 0.687 | 0.645 | 520.3 |
| fixed | bm25 | 0.775 | 0.639 | 0.665 | 0.575 | 17.6 |
| fixed | rerank | 0.840 | 0.675 | 0.708 | 0.621 | 242.7 |
| recursive | bm25 | 0.785 | 0.635 | 0.666 | 0.578 | 16.9 |
| recursive | rerank | 0.850 | 0.671 | 0.708 | 0.618 | 223.4 |
| semantic | bm25 | 0.807 | 0.650 | 0.683 | 0.625 | 11.7 |
| semantic | rerank | 0.807 | 0.666 | 0.696 | 0.631 | 376.4 |

Online latency is the mean of up to 3 single-query calls on this machine, including query embedding. It is not the batched throughput of the full eval.

Metrics are macro-averaged over labeled queries. Recall, MRR, and nDCG are computed after mapping chunks back to documents. Context precision scores the chunks in the order they were retrieved.

## Generation

Model `openai/gpt-4o-mini`, provider `openrouter` judged 100 queries.
Retrieval grid size: 100 queries. Generation sample: 100 queries, context_k=5.
Token counts from the provider: 1260305 input, 46274 output, 1122 calls, 376 cache hits. Estimated cost $0.2168 at $0.15/$0.60 per million.

An exact answer `I do not know.` is an abstention: no claims, and it is left out of faithfulness, hallucination rate, and answer relevancy. Refusal rate is abstentions divided by the generation sample. n is the number of answered queries that have a score. Hallucination rate is 1 minus faithfulness on those claims.

| Chunker | Retriever | Refusal rate | n | Faithfulness | Hallucination rate | Answer relevancy | Errors |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| whole | bm25 | 0.540 | 46 | 0.953 | 0.047 | 0.791 | 0 |
| whole | rerank | 0.560 | 44 | 0.909 | 0.091 | 0.798 | 0 |
| fixed | rerank | 0.500 | 49 | 0.918 | 0.082 | 0.796 | 1 |
| recursive | rerank | 0.460 | 54 | 0.957 | 0.043 | 0.797 | 0 |
| semantic | rerank | 0.440 | 56 | 0.950 | 0.050 | 0.787 | 0 |

Means that scored every row, including abstentions, are in `all_rows` on each pair in metrics.json.

154/250 refusals had a gold-relevant document in the 5 contexts (whole/bm25 35/54, whole/rerank 33/56, fixed/rerank 29/50, recursive/rerank 29/46, semantic/rerank 28/44). Each context is matched to a document by its alphanumeric tokens, then checked against the SciFact test qrels.

![nDCG@10 by chunker and retriever](ndcg_at_10.png)
