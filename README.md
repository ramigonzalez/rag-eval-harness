# RAG Lab

A small, reproducible benchmark of chunking and retrieval strategies for retrieval-augmented generation. It scores a public dataset with local open-source models, writes a table and a chart, and keeps a regression gate in CI.

Ramiro Gonzalez.

## The question

On a corpus of short scientific abstracts, which chunking policy and which retriever put the labeled evidence highest in the list, and what does that cost in latency?

The grid is four chunkers (whole document, fixed windows, recursive splitting, semantic splitting) by four retrievers (BM25, dense, hybrid, cross-encoder rerank). HyDE and RAG-Fusion are implemented and stay out of the grid until an API key is set.

## Metrics

All of these are macro-averages over the 300 labeled test queries.

- **Recall@10.** Share of the labeled relevant documents that appear in the top 10. Chunks are collapsed to their source document first, keeping the best rank. SciFact has 1.13 relevant documents per query, so this is close to a hit rate.
- **MRR@10.** One divided by the rank of the first relevant document. Rank 1 scores 1, rank 2 scores 0.5, and a miss outside the top 10 scores 0.
- **nDCG@10.** Graded ranking quality with a log discount, divided by the perfect ranking. SciFact labels are binary, so a relevant document at rank 1 is worth more than the same document at rank 5.
- **Context precision@10.** The RAGAS formula, scored from the gold labels rather than an LLM. A chunk counts as relevant when its source document is labeled relevant. An irrelevant chunk placed above a relevant one lowers the score. Irrelevant chunks below the last relevant one do not. This is the shape of the context window you would hand to a generator.
- **Faithfulness.** Share of answer claims that the retrieved context supports, judged by an LLM.
- **Answer relevancy.** Mean cosine similarity, clipped at 0, between the user question and questions an LLM says the answer is addressing. The embeddings are local. The LLM only proposes the questions.
- **Hallucination rate.** Share of answer claims the context does not support. On the same claims it is 1 minus faithfulness.

Faithfulness, answer relevancy, and hallucination rate did not run. **run with API key**.

## How to run

From the repository root, with Python 3.10 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
raglab run
```

That command reads `configs/scifact.yaml`, downloads the SciFact zip (about 8 MB, checksum pinned), downloads two small ONNX models on first use, and writes `results/scifact/`:

- `metrics.csv` and `metrics.json`
- `results.md`
- `ndcg_at_10.png`
- `examples.json` (a few real retrieved snippets)

`pytest` runs the unit tests without downloading models. The smoke gate uses the real MiniLM models on a 10-document fixture:

```bash
raglab run --config configs/smoke.yaml --check configs/smoke_thresholds.json
```

The scored SciFact command took **26 minutes 43 seconds** on this machine (4 vCPUs, Python 3.12.3, fastembed 0.9.1, numpy 2.5.4). Whole-document and fixed-size embeddings were loaded from `data/cache` in 0.8s and 1.1s because an earlier attempt on the same corpus had written them. Recursive indexing (244.5s) and semantic chunking (393.0s, then 165.0s to embed the chunks) were cold. A machine with an empty cache also embeds the whole-document and fixed indexes.

To add the LLM strategies and the generation metrics:

```bash
export OPENAI_API_KEY=...
# optional: RAGBENCH_API_BASE (default https://api.openai.com/v1)
# optional: RAGBENCH_LLM_MODEL (default gpt-4o-mini)
raglab run --generation on
```

`RAGBENCH_API_KEY` is accepted as an alias of `OPENAI_API_KEY`. With no key, the command skips generation, HyDE, and RAG-Fusion and says so in the report. It does not invent those numbers.

## Results

SciFact test split, full corpus of 5,183 abstracts, all 300 labeled queries. Mean document length 214.6 words. Embeddings are `sentence-transformers/all-MiniLM-L6-v2`. The reranker is `Xenova/ms-marco-MiniLM-L-6-v2`. Both run locally through ONNX. No API key was used.

| Chunker | Retriever | Recall@10 | MRR@10 | nDCG@10 | Context precision@10 | Online latency (ms) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| whole | bm25 | 0.785 | 0.633 | 0.664 | 0.628 | 6.9 |
| whole | dense | 0.783 | 0.605 | 0.645 | 0.599 | 17.4 |
| whole | hybrid | 0.815 | 0.647 | 0.685 | 0.644 | 16.2 |
| whole | rerank | 0.819 | 0.664 | 0.693 | 0.656 | 993.9 |
| fixed | bm25 | 0.767 | 0.599 | 0.634 | 0.539 | 17.1 |
| fixed | dense | 0.819 | 0.611 | 0.658 | 0.562 | 21.5 |
| fixed | hybrid | 0.831 | 0.652 | 0.693 | 0.587 | 36.8 |
| fixed | rerank | 0.834 | 0.675 | 0.708 | 0.612 | 379.3 |
| recursive | bm25 | 0.755 | 0.586 | 0.623 | 0.529 | 15.5 |
| recursive | dense | 0.817 | 0.616 | 0.660 | 0.552 | 20.9 |
| recursive | hybrid | 0.808 | 0.644 | 0.682 | 0.577 | 36.7 |
| recursive | rerank | 0.838 | 0.666 | 0.703 | 0.607 | 360.6 |
| semantic | bm25 | 0.780 | 0.612 | 0.649 | 0.583 | 10.1 |
| semantic | dense | 0.812 | 0.592 | 0.644 | 0.553 | 8.1 |
| semantic | hybrid | 0.812 | 0.645 | 0.682 | 0.607 | 18.5 |
| semantic | rerank | 0.819 | 0.663 | 0.694 | 0.620 | 717.5 |

![SciFact nDCG@10 by chunker and retriever](results/scifact/ndcg_at_10.png)

Online latency is the mean of 10 single-query calls, including query embedding. It is not the batched throughput of the eval. Every hit list contained at least 18 distinct documents, so recall@10 was not cut off by duplicate chunks of the same abstract. Full records, including @5 and the config, are in `results/scifact/metrics.json`.

| Chunker | Chunks | Avg words | Chunks per doc | Chunk time (s) | Index time (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| whole | 5,183 | 214.6 | 1.00 | 0.0 | 0.8 |
| fixed | 15,596 | 88.0 | 3.01 | 0.1 | 1.1 |
| recursive | 15,727 | 78.1 | 3.03 | 0.7 | 244.5 |
| semantic | 9,879 | 112.6 | 1.91 | 393.0 | 165.0 |

Index time is shared by every retriever on that chunker. The 0.8s and 1.1s figures are cache loads. The 244.5s, 393.0s, and 165.0s figures were computed during this run.

## What the results suggest

On this dataset, hybrid retrieval is the consistent step up. On every chunker, hybrid nDCG@10 is higher than both BM25 and dense alone. Whole-document hybrid is 0.685, against 0.664 for BM25 and 0.645 for dense. Fixed hybrid is 0.693.

The cross-encoder adds a smaller gain on top of that. The best nDCG@10 in the grid is fixed + rerank at 0.708. Fixed hybrid, which feeds that reranker, is 0.693, and it does the job in 36.8 ms instead of 379.3 ms. Whole-document rerank reaches 0.693 as well, at 993.9 ms, because the cross-encoder is reading the entire abstract.

Chunking is not free, and it does not help every retriever. BM25 is best on the whole abstract (nDCG@10 0.664) and worse on fixed windows (0.634) and recursive splits (0.623). Dense search moves the other way: recursive dense is 0.660 and whole-document dense is 0.645. The highest recall@10 is recursive + rerank at 0.838. The highest context precision@10 is whole-document rerank at 0.656. Splitting the abstract can raise document-level nDCG while making the context window noisier, because the label applies to the whole abstract and a window full of chunks has more chances to put an irrelevant piece above a relevant one.

Semantic chunking does not win on this corpus. Abstracts are single-topic and about 215 words, so the splitter keeps 1.91 chunks per document and lands next to the whole-document rows (semantic rerank nDCG@10 is 0.694). Deciding those boundaries took 393.0 seconds of sentence embedding.

## How I'd use this in production

I would not copy the winning cell into a product. I would copy the comparison.

For a corpus shaped like SciFact, I would ship hybrid search first: BM25 plus a small local embedding, fused with reciprocal rank fusion (k = 60), over either the whole abstract or a 100-word fixed window. On these numbers that is the knee of the curve. Fixed hybrid is within a short step of the best nDCG@10, and it stays under 40 ms here. I would add the MiniLM cross-encoder on the top 20 hybrid hits only if the extra nDCG, 0.708 against 0.693 on the fixed index, was worth a few hundred milliseconds. I would not rerank whole abstracts at a measured 993.9 ms per query unless the collection was tiny.

I would not semantic-chunk this kind of document. The build cost is real and the ranking did not move past a dumb window. I would turn semantic chunking back on for long, multi-topic documents, and I would re-run this grid rather than assume the SciFact order still holds. BM25 in particular wanted the whole abstract here; on a 20-page PDF I would expect the opposite.

I would keep generation metrics, HyDE, and RAG-Fusion behind the same key check this repo uses. They are extra model calls on the request path. I would not turn them on until a labeled run showed a gain over hybrid large enough to pay for the latency and the bill. This run does not have that evidence.

I would put a tiny deterministic gate in CI, the way `configs/smoke_thresholds.json` locks BM25 MRR@10 at 1.0 on a fixture where each query contains a token that appears in only one document. That fails if indexing or tokenization breaks, and it does not need a GPU or an API key. I would not gate a neural metric at its exact measured value. The dense floor in that file sits below the measured score for that reason.

The retriever registry is the slot for a community RAG framework later. Wrapping LlamaIndex, LangChain, or Haystack is another `Retriever`, not a rewrite. I would run it through this same grid before trusting it on a customer corpus.

## Dataset

SciFact, as packaged by BEIR. A claim is the query. The corpus is paper abstracts. A relevance label means the abstract is evidence for or against the claim. The test split has 300 labeled queries and 5,183 documents. Labels are binary. The official files are downloaded from the BEIR release and checked against a pinned SHA-256. They are not committed.

I used SciFact instead of FiQA or MS MARCO because those corpora are much larger and a full chunker-by-retriever grid would not finish on CPU in one sitting. Abstracts are still long enough that a 100-word window splits them: the fixed chunker produced 3.01 chunks per document. The whole-document row is there so a chunker has to beat "index the abstract as it was written."

## Strategies

Chunkers, registered in `raglab.chunking`:

- **whole.** One chunk per abstract. The baseline.
- **fixed.** 100-word windows, 25-word overlap. Cuts wherever the window ends.
- **recursive.** Same budget, but splits on paragraphs and sentences first, and only then on words.
- **semantic.** Embeds neighboring sentences and cuts where the cosine distance is above the 95th percentile of distances in that document. Groups longer than 220 words fall back to the recursive splitter. Very short groups are merged.

Retrievers, registered in `raglab.retrieval`:

- **bm25.** Okapi BM25, k1 = 1.5, b = 0.75. Each query term contributes once. Tokens are lowercase words and numbers.
- **dense.** Cosine similarity on L2-normalized MiniLM embeddings.
- **hybrid.** Reciprocal rank fusion of the BM25 and dense lists. A document's score is the sum of `1 / (60 + rank)` across the two lists. Ties break on chunk id.
- **rerank.** The hybrid list, with the top 20 chunks rescored by the MiniLM cross-encoder. The tail keeps hybrid order. Scores from this row are logits, not cosines.
- **hyde.** Asks an LLM for a short hypothetical answer, embeds that passage, and dense-searches with it. Needs an API key.
- **rag_fusion.** Asks an LLM for three extra queries, dense-searches each one plus the original, and fuses them with the same RRF. Needs an API key.

## Adding a strategy

A community framework adapter is another retriever. Put the class in a module, register it, and list that module under `plugins` in the config.

```python
from raglab.retrieval import register_retriever
from raglab.retrieval.strategies import Retriever

@register_retriever
class FrameworkRetriever(Retriever):
    name = "framework"
    needs_embeddings = True

    def retrieve(self, index, query, k, services):
        ...
```

```yaml
plugins:
  - my_package.framework_retriever
retrievers: [bm25, dense, hybrid, framework]
```

Chunkers register the same way, with `register_chunker` and a `from_config` classmethod.

## CI

`.github/workflows/ci.yml` installs the package, runs `ruff` and `pytest`, then runs the smoke eval. The gate fails if BM25 MRR@10 on the fixture drops below 1.0, or if fixed-window dense MRR@10 drops below 0.8. The BM25 bar is exact because the fixture is deterministic. The dense bar has room under the measured score so a small ONNX change does not fail the build.

## Layout

```
configs/scifact.yaml          # the published grid
configs/smoke.yaml            # CI fixture run
configs/smoke_thresholds.json
src/raglab/                   # chunkers, retrievers, metrics, CLI
data/fixtures/smoke.json      # original tiny corpus for the gate
results/scifact/              # the measured grid
tests/
```

`raglab run --help` lists the overrides for query count, chunkers, retrievers, and the generation switch.
