# Stage-wise ablation

Dev: train, 80 queries (of 809 available), seed 7.
Test: test, 300 queries (of 300 available).

Each stage is chosen by dev nDCG@10. Delta is the change in test nDCG@10 from the previous row. Local stages have an estimated cost of $0.

| Stage | Chosen | Dev nDCG@10 | Test nDCG@10 | Delta test nDCG@10 | Test latency (ms) | Est. USD |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| baseline | whole/none/bm25/none | 0.597 | 0.664 |  | 6.7 | 0.00 |
| chunking | whole | 0.597 | 0.664 | 0.000 | 6.7 | 0.00 |
| query_transform | none | 0.597 | 0.664 | 0.000 | 7.4 | 0.00 |
| retrieval | rrf | 0.637 | 0.685 | 0.021 | 11.9 | 0.00 |
| rerank | cross_encoder | 0.637 | 0.693 | 0.008 | 654.2 | 0.00 |

Final pipeline chosen on dev: chunking=whole, query_transform=none, retrieval=rrf, rerank=cross_encoder.
