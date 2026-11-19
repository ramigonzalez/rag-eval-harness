# Generation, faithfulness judge v2

Same 500 answers as `results/generation-100/answers.jsonl` (100 SciFact queries x 5 stacks, context_k=5). Only faithfulness was re-scored, with judge v2 (`openai/gpt-4o-mini`, temperature 0): claims keep the answer's yes/no verdict and negations, then each claim is checked against the context as supported, contradicted, or not_found. Answer relevancy is carried over from the v1 run.

Answered-only: an exact `I do not know.` is an abstention and is excluded from faithfulness, hallucination rate, and relevancy.

| Chunker | Retriever | Refusal rate | Answered | Faithfulness v2 | Hallucination rate v2 | Answer relevancy | Faithfulness v1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| whole | bm25 | 0.54 | 46 | 0.851 | 0.149 | 0.791 | 0.953 |
| whole | rerank | 0.56 | 44 | 0.920 | 0.080 | 0.798 | 0.909 |
| fixed | rerank | 0.50 | 50 | 0.877 | 0.123 | 0.796 | 0.918 |
| recursive | rerank | 0.46 | 54 | 0.867 | 0.133 | 0.797 | 0.957 |
| semantic | rerank | 0.44 | 56 | 0.792 | 0.208 | 0.787 | 0.950 |

Re-scoring cost: 398 LLM calls, 380403 input and 19616 output tokens, about $0.0688 at $0.15/$0.60 per million.
