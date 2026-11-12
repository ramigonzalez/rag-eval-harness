# Metrics

This page defines every number the result tables report. A hiring manager can read the short list and stop. A learner can read one metric at a time: what it measures, the formula, how this repository computes it, why a production system cares, a toy example, how to read the range, and what the number does not say.

The toy examples are made up. They are not SciFact scores. When a real number appears, the path of the result file is next to it.

Published tables show the cutoff k = 10. The scorer also writes @5 (`recall@5`, and the same for the other retrieval metrics) because `k_values` is `[5, 10]`. Generation uses `context_k` = 5 chunks.

## If you only have a minute

| Name | Column | What it means | Better |
| --- | --- | --- | --- |
| Recall@10 | `recall@10` | Of the documents labeled relevant, how many showed up anywhere in the first 10? 1.0 = all of them did. | Higher |
| MRR@10 | `mrr@10` | How close to the top is the first relevant document? 1.0 = it is rank 1. 0 = it is not in the first 10. | Higher |
| nDCG@10 | `ndcg@10` | Are the relevant documents near the top of the first 10 results? 1.0 = perfect order. | Higher |
| Context precision@10 | `context_precision@10` | Are the chunks the model would read the labeled ones, and are those chunks near the front? 1.0 = relevant chunks packed at the top. | Higher |
| Online latency | `online_latency_ms` | Once the index is built, how many milliseconds does one question take? | Lower |
| Chunk time | `chunk_seconds` | How long did it take to split the corpus into chunks, paid once before any question? | Lower |
| Index time | `index_seconds` | How long did it take to build the search index for that chunking, including embeddings when a dense index is built? | Lower |
| Batched eval time | `eval_seconds` | How long did scoring every query in one batch take? This is not the per-question response time. | Lower |
| Ablation test latency | `test_latency_ms` | On the ablation, how many milliseconds per query did the full test split take when run as a batch? | Lower |
| Estimated USD | `estimated_usd` | What is the estimated model bill? 0 means this stage did not count a paid model call. | Lower |
| Refusal rate | `refusal_rate` | How often was the answer exactly "I do not know."? 0 = it always tried. 1 = it abstained every time. | Depends |
| Faithfulness | `faithfulness` | Of the claims in the answer, what fraction did the judge say the retrieved context supports? 1.0 = every scored claim supported. | Higher |
| Hallucination rate | `hallucination_rate` | Of those same claims, what fraction did the judge mark unsupported? 0 = none. This is 1 minus faithfulness, not a second judge. | Lower |
| Answer relevancy | `answer_relevancy` | Does the answer address the question the user asked? 1.0 = it stays on that question. | Higher |
| Delta test nDCG@10 | `delta_test_ndcg@10` | After a stage was chosen on dev, how much did test nDCG@10 move versus the previous stage? Positive means the test score rose. | Higher |

Retrieval metrics are a macro-average: each labeled query counts once, then we take the mean. A query with one relevant abstract counts the same as a query with two.

## Recall@10

**What it measures.** Of the documents the dataset marks relevant, how many show up in the top 10. It does not care whether the hit is rank 1 or rank 10, and it ignores extra documents that were not labeled relevant.

**Formula.** Let \(R_q\) be the set of documents with a positive label for query \(q\). Let \(D_q(k)\) be the first \(k\) document ids after chunks are collapsed (see below).

\[
\mathrm{Recall@}k(q) = \frac{|R_q \cap D_q(k)|}{|R_q|}
\]

The table value is the unweighted mean of that score over labeled queries.

This repository computes it in `recall_at_k` and `score_rankings` in `src/raglab/metrics.py`.

- The retriever returns up to `hit_k` chunks. The recorded configs have `hit_k` 60 (`results/scifact/metrics.json`, `results/generation-100/metrics.json`). `score_rankings` then cuts at k = 10.
- `dedupe_documents` walks that chunk list from best to worst and keeps the first time each `doc_id` appears. Later chunks of the same document do not take another slot. Rank 1 means "the best chunk of this document was the first new document."
- A document counts as relevant when its qrel grade is greater than 0. Unlabeled documents are treated as not relevant.
- Queries with no positive label are skipped. They are not entered as zero. `score_rankings` counts them in `skipped_queries`.
- The mean is macro: one number per query, then the average. It is not micro-averaged over documents.
- Recall does not use the LLM, so abstentions and judge errors do not apply.

**Why it matters.** If the labeled evidence never enters the top 10, the generator cannot ground the answer in it. In production this is the first question: did we fetch the document that contains the fact?

**Toy example.** Query with relevant documents \(\{A, B, C\}\). After collapsing chunks, the top 10 document ids are \(A, X, B, \ldots\). Then \(|R \cap D(10)| = 2\) and \(\mathrm{Recall@10} = 2/3\). This is not a result from this repo.

**Range.** 0 to 1. Higher is better. 1 means every labeled document is inside the top 10. 0 means none of them are.

**Limitations.** SciFact labels are sparse. The 300-query file has mean 1.13 relevant documents per query (`results/scifact/results.md`). The 100-query generation sample has mean 1.15 (`results/generation-100/results.md`). Missing the one labeled abstract makes recall 0 for that query, even if other abstracts are on the same topic but unlabeled. A high recall also says nothing about rank order inside the top 10, and nothing about whether the answer stayed faithful to the text.

## MRR@10

**What it measures.** How far down the collapsed document list you have to go to find the first labeled document. A hit at rank 1 scores 1. A hit at rank 10 scores 0.1. No hit in the top 10 scores 0.

**Formula.** Let \(\mathrm{rank}_q\) be the 1-based position of the first document in \(D_q(k)\) that is in \(R_q\). If there is no such document,

\[
\mathrm{MRR@}k(q) = 0
\]

otherwise

\[
\mathrm{MRR@}k(q) = \frac{1}{\mathrm{rank}_q}.
\]

The table value is the mean over labeled queries. MRR is "mean reciprocal rank."

Computed in `mrr_at_k` and `score_rankings` in `src/raglab/metrics.py`, with the same document collapsing, the same k = 10 cut, and the same skip rule as recall.

This is MRR cut at k, not the uncut reciprocal rank. A labeled document at rank 11 or later scores 0. The parity test checks that case against `pytrec_eval` after the run is cut at 10 (`tests/test_parity.py`).

**Why it matters.** Two systems can have the same recall@10 while one puts the evidence at rank 1 and the other at rank 10. The generator usually sees only the first few chunks (`context_k` = 5 in the generation run). MRR tells you whether the first useful document is actually near the top.

**Toy example.** The first labeled document is at position 4 in the collapsed list. \(\mathrm{MRR@10} = 1/4\). If that document were at position 11, \(\mathrm{MRR@10} = 0\). This is not a result from this repo.

**Range.** 0 to 1. Higher is better.

**Limitations.** Only the first hit matters. A second relevant document does not raise MRR. On SciFact most queries have about one relevant document (1.13 and 1.15 in the files cited above), so that limit is mild here and stricter on a dataset with many relevant documents. MRR is not a latency metric and not a faithfulness metric.

## nDCG@10

**What it measures.** Ranking quality in the top k. A relevant document near the top adds more than the same document near rank 10. The score is divided by the ideal ranking, so the best possible order scores 1.

**Formula.** For a ranked list of grades \(rel_1, \ldots, rel_k\),

\[
\mathrm{DCG@}k = \sum_{i=1}^{k} \frac{2^{rel_i} - 1}{\log_2(i+1)}.
\]

\(\mathrm{IDCG@}k\) is the same sum on the best possible grades: the positive labels for that query, sorted from high to low, cut at k. Then

\[
\mathrm{nDCG@}k = \frac{\mathrm{DCG@}k}{\mathrm{IDCG@}k}.
\]

Computed in `dcg` and `ndcg_at_k` in `src/raglab/metrics.py`. The code uses a 0-based index, and `log2(index + 2)` is \(\log_2(i+1)\) for rank \(i\).

How this repo applies it:

- Grades come from the collapsed document list, not from raw chunks. The grade of a document is its qrel value. A document that is not in the qrels contributes grade 0.
- Only grades greater than 0 are kept when the ideal list is built. `score_rankings` also drops non-positive grades before calling `ndcg_at_k`.
- The gain is \(2^{rel} - 1\), the usual TREC form, not \(rel\) itself and not \(\log(1+rel)\). SciFact labels are binary, so a positive grade is 1 and the gain is \(2^1 - 1 = 1\). A zero grade adds nothing. With binary labels this matches binary DCG. The code would still give a higher grade more gain if a qrel used grades above 1.
- If the ideal DCG is 0, `ndcg_at_k` raises. `score_rankings` avoids that by skipping queries with no positive label.
- The table is the macro-average over the queries that were scored.
- `tests/test_parity.py` compares this nDCG@10 to `pytrec_eval`'s `ndcg_cut.10` within \(10^{-8}\).

**Why it matters.** nDCG is the single number the ablation uses to choose a pipeline (`src/raglab/ablation.py`, dev `ndcg@10`). It punishes a system that retrieves the right document but buries it. In production, the chunks at the front of the prompt matter more than the ones at the end.

**Toy example.** Binary grades in order \((1, 0, 1)\), and k = 3. Two relevant documents exist, so the ideal order is \((1, 1, 0)\).

\[
\mathrm{DCG} = \frac{1}{\log_2 2} + 0 + \frac{1}{\log_2 4} = 1 + 0.5 = 1.5
\]

\[
\mathrm{IDCG} = 1 + \frac{1}{\log_2 3}
\]

\[
\mathrm{nDCG@3} = \frac{1.5}{1 + 1/\log_2 3}
\]

This is not a result from this repo. The published tables use k = 10.

**Range.** 0 to 1. Higher is better. 1 means the top of the list is an ideal order of the labeled documents.

**Limitations.** With sparse binary labels, nDCG@10 and MRR@10 often move together, because there is usually one relevant document and the gain is 1. Unlabeled but useful abstracts count as grade 0, so a sensible ranking can score poorly. nDCG does not measure whether the generated answer used the text.

## Context precision@10

**What it measures.** Looking at the chunks in the order the retriever returned them, are the ones whose source document is labeled relevant packed at the top? This is the list a generator would see. It is not collapsed to documents.

**Formula.** Let \(v_i = 1\) when chunk \(i\)'s `doc_id` is in \(R_q\), and 0 otherwise. Let

\[
\mathrm{Precision@}i = \frac{1}{i}\sum_{j=1}^{i} v_j.
\]

Then, as in RAGAS,

\[
\mathrm{ContextPrecision@}k = \frac{\sum_{i=1}^{k} (\mathrm{Precision@}i) \, v_i}{\sum_{i=1}^{k} v_i}.
\]

If no chunk in the top k comes from a labeled document, the code returns 0 (the denominator would be 0).

Computed in `context_precision_at_k` in `src/raglab/metrics.py`. The docstring there states the same rule: an irrelevant chunk below the last relevant chunk does not change the score; an irrelevant chunk above a relevant one does.

Differences from recall, MRR, and nDCG:

- The input is the chunk list, not `dedupe_documents`. Two chunks from the same relevant document are two relevant verdicts.
- The cutoff is still k = 10, but it is 10 chunks, not 10 unique documents.
- There is no LLM verdict. "Relevant" means the gold document id, the same qrels as recall.

The table value is the macro-average over labeled queries. Queries with no positive label are skipped, same as the other retrieval metrics.

**Why it matters.** The model reads chunks, not a deduplicated document list. A relevant document buried under irrelevant chunks wastes context window and invites the model to follow the wrong passage. Context precision is the retrieval number that most closely matches that prompt.

**Toy example.** Verdicts for k = 3 are \((1, 0, 1)\).

- At i = 1, \(v_1 = 1\), precision = 1, contribution 1.
- At i = 2, \(v_2 = 0\), no contribution.
- At i = 3, \(v_3 = 1\), precision = \(2/3\), contribution \(2/3\).

Numerator \(1 + 2/3 = 5/3\). Denominator 2. Context precision = \((5/3)/2 = 5/6\).

Plain precision would have been \(2/3\). The RAGAS form is higher here because it averages precision only at the relevant positions. This is not a result from this repo.

**Range.** 0 to 1. Higher is better.

**Limitations.** A chunk is relevant whenever its document is relevant, even if that chunk is the wrong sentence of a long document. On SciFact the documents are short abstracts (mean 214.6 words in `results/scifact/results.md`), so this is a smaller issue than it would be on full papers. Duplicate chunks of one relevant document can inflate the score relative to document-level recall. The metric still uses sparse qrels: a good unlabeled chunk counts as irrelevant.

## Online latency

**What it measures.** How long one query takes when the index is already in memory. It is a response-time number, not the time to score the whole dataset.

**Formula.** For a sample of \(n\) queries,

\[
\mathrm{online\_latency\_ms} = \frac{1000}{n} \sum_{j=1}^{n} t_j
\]

where \(t_j\) is the wall time of one `retriever.retrieve` call.

Computed in `_online_latency` in `src/raglab/runner.py`. The column in CSV and JSON is `online_latency_ms`. The markdown header is "Online latency (ms)".

How the published runs do it:

- The call uses `hit_k`, not k = 10. Recorded runs set `hit_k` to 60 (`results/scifact/metrics.json`, `results/generation-100/metrics.json`). The timer covers building the candidate list the eval uses, not a separate top-10 cut.
- The sample is `_evenly`: evenly spaced queries, not a random draw. The result files state the sample size. `results/scifact/results.md` says up to 10 calls. `results/generation/results.md` says up to 5. `results/generation-100/results.md` says up to 3.
- The clock includes query embedding when that retriever embeds the query (dense, hybrid, rerank). BM25 does not embed the query. The result files describe the column as including query embedding.
- Index construction is not inside this timer. Batched scoring of every query is `eval_seconds` in `metrics.csv`, and the result files say online latency is not that throughput.
- The ablation column `test_latency_ms` is a different measurement. See "Indexing and offline cost".

**Why it matters.** A ranking gain that adds hundreds of milliseconds per question may be the wrong trade for an interactive product. Online latency is the number to put next to nDCG when choosing a retriever.

**Toy example.** Three calls take 10 ms, 20 ms, and 30 ms. The mean is 20 ms. This is not a result from this repo.

**Range.** Positive milliseconds. Lower is better. There is no fixed ceiling.

**Limitations.** The sample is small (3, 5, or 10 calls in the files above), so the number is noisy. It is one machine, one process, warm indexes. Do not compare milliseconds across the 300-query run and the generation runs: the README records them on different CPU counts. HyDE and RAG-Fusion latency in `results/generation/results.md` is measured after the prompt cache is warm and does not include the chat call.

## Indexing and offline cost

**What it measures.** Work done before any question is asked, plus the time to score a whole split in a batch, plus the dollar estimate when a run calls a model.

**What is reported.**

| What | Where | What the code times |
| --- | --- | --- |
| Chunk time | Markdown "Chunk time (s)"; JSON `chunkers[].chunk_seconds` | `chunker.chunk_corpus` in `run_benchmark` (`src/raglab/runner.py`). |
| Index time | Markdown "Index time (s)"; JSON `chunkers[].index_seconds` | `build_search_index` for that chunker. Shared by every retriever on that chunker. Includes embedding when a dense index is built. |
| Batched eval | CSV `eval_seconds` | Wall time of `retrieve_all` over the full query list at `hit_k`. Not divided per query in the table. |
| Ablation test latency | `test_latency_ms` | In `_score` in `src/raglab/ablation.py`: wall time of `run_pipeline` on the split, divided by the number of queries, times 1000. This is the whole test set, not the 10-call online sample. |
| Estimated USD | Ablation column `estimated_usd`; generation usage `estimated_usd` | Ablation `_score` sets `estimated_usd` to 0.0 for these local stages. Generation cost is token counts times the configured list price (`src/raglab/cost.py`), not a measured invoice. |

`results/ablation/results.md` shows estimated cost 0.00 on every row. `results/generation-100/results.md` reports the generation token estimate at the list rates in that file.

**Formula.** There is no quality formula. Times are wall-clock seconds or milliseconds. The ablation per-query figure is

\[
\mathrm{test\_latency\_ms} = 1000 \times \frac{T_{\text{pipeline}}}{N_{\text{queries}}}.
\]

**Why it matters.** Semantic chunking and a cross-encoder can be cheap per query after the index exists, or expensive to build. A production system pays index time when the corpus changes and pays online latency on every question. Mixing those up leads to the wrong capacity plan.

**Toy example.** Chunking takes 2 seconds and indexing takes 10 seconds, once. Then each of 100 batched queries adds a share of a 5-second `retrieve_all`, which is 50 ms per query in the batch. A separate single-query timer might still read 80 ms because it does not batch. These figures are not from this repo.

**Range.** Times are non-negative. Lower is better for a fixed corpus and a fixed quality target. Dollars here are an estimate, and zero means "this stage did not call a paid model," not "the machines were free."

**Limitations.** Chunk time and index time depend on whether embeddings were already cached. Semantic chunk time is 393.0 seconds in `results/scifact/results.md` and 0.7 seconds in `results/generation-100/results.md` on the same chunker name. The second run is not a faster algorithm; the file is a different machine and a warm cache. Do not treat index seconds as a property of the chunker alone.

The ablation latency and the retrieval table's online latency answer different questions. `results/ablation/results.md` uses the full test split (300 queries). `results/scifact/results.md` uses up to 10 single calls.

## Refusal rate

**What it measures.** How often the generator answers with the exact sentence the prompt tells it to use when the context is not enough: `I do not know.`

**Formula.**

\[
\mathrm{refusal\_rate} = \frac{N_{\text{abstain}}}{N_{\text{sample}}}
\]

\(N_{\text{sample}}\) is `num_queries` on the generation pair (the judged sample, 100 in the generation-100 run). \(N_{\text{abstain}}\) is the number of rows whose answer strips to exactly `I do not know.`

Computed in `is_abstention` and `summarize_answer_records` in `src/raglab/generation.py`. The column is `refusal_rate`. The markdown table prints it to three decimals. `num_refusals` is the numerator.

An abstention is not a judge error. `num_errors` counts exceptions (the generation-100 fixed + rerank row has one timeout). Abstentions are also not scored for faithfulness, hallucination rate, or answer relevancy.

**Why it matters.** A high refusal rate means the system often declines. That can be the safe behavior when the context has no evidence. It can also mean the prompt is stricter than the product wants, so users see "I do not know." even though the right abstract was retrieved. Refusal rate is how you tell those apart from a model that answers and is simply unfaithful.

**Toy example.** 100 answers, 40 of them exactly `I do not know.` Refusal rate = 0.40. The other 60 are the pool for faithfulness. This is not a result from this repo. The measured counts are in `results/generation-100/results.md`.

**Range.** 0 to 1. Neither end is automatically better. 0 means the model always attempted an answer. 1 means it abstained on every query.

**Limitations.** Only the exact string counts. "The context does not say" is not an abstention in this code. The rate uses the sample size, not "rows that finished." If a run stopped early and left queries unwritten, those missing rows are not refusals, so the rate can look lower than the share among completed rows. The published generation-100 file has one row per query per stack, so that caveat does not change that file. Whether a refusal was justified is a separate check: `results/generation-100/results.md` reports how many refusals still had a gold document in the 5 contexts.

## Faithfulness

**What it measures.** After the model writes an answer from the retrieved chunks, an LLM judge splits that answer into claims and marks each claim supported or not. Faithfulness is the fraction of claims marked supported. It is the RAGAS-style check: did the answer stay inside the context?

**Formula.** For an answered query with claims \(c_1, \ldots, c_m\) and verdicts \(s_j \in \{0, 1\}\),

\[
\mathrm{Faith}(q) = \frac{1}{m}\sum_{j=1}^{m} s_j.
\]

The reported faithfulness is the macro-average of \(\mathrm{Faith}(q)\) over answered queries that have a score (`num_scored`, the column `n` in the generation table).

Computed in `run_generation` (the claim loop) and `summarize_answer_records` in `src/raglab/generation.py`. The judge prompt is `CLAIMS_PROMPT`. The answer is produced with `ANSWER_PROMPT` at temperature 0 (`LLMClient.complete` defaults to 0). Context is `context_k` = 5 chunks.

Handling of special rows:

- Exact abstention (`I do not know.`): the judge does not build claims. The row is excluded even if an older file stored a faithfulness value on that answer.
- No claims, and the row is not an abstention and not an error: excluded from the mean, counted in `num_empty_claims`.
- Error with no stored score: excluded from the mean, counted in `num_errors`. In `results/generation-100/metrics.json`, fixed + rerank has `num_answered` 50, `num_errors` 1, and `num_scored` 49. The error row has no faithfulness value, so it is not in the mean. The same `n` of 49 is in `results/generation-100/results.md`.
- If a score was stored and then a later step threw, the stored score is still included and `num_errors` still increments. The published timeout has no score, so it is not in the mean.
- `all_rows.faithfulness` in `results/generation-100/metrics.json` is the mean of every stored score, including abstentions that the older judge had scored. The headline field `faithfulness` does not include those.

The verdict is whatever the judge model returns (`supported` true or false). It is not a human label and not a string match against the context.

**Why it matters.** A retriever can be excellent and the answer can still introduce facts that were not in the chunks. Faithfulness is the check that the words shown to the user are supported by the text the system retrieved. In production this is the grounding metric.

**Toy example.** The answer yields three claims. The judge marks two supported. \(\mathrm{Faith} = 2/3\). A second query abstains with `I do not know.` and is not averaged in. The reported faithfulness is \(2/3\), not "2 supported out of 3 claims plus a refusal." This is not a result from this repo.

**Range.** 0 to 1. Higher is better. 1 means every scored claim was marked supported.

**Limitations.** The judge has not been checked against human labels. RAGAS agreement has not been run (`scripts/ragas_agreement.py`). The judge can mark a claim supported when a person would not, or the reverse. Abstentions are out of the average, so a system can look highly faithful while refusing half the questions. Read `refusal_rate` beside faithfulness. Sparse qrels do not enter this formula; a claim can be faithful to a retrieved chunk that is the wrong document.

## Hallucination rate

**What it measures.** The share of those same claims that the judge marked unsupported. This repository defines it as one minus faithfulness on the same claims. It is not a separate hallucination detector.

**Formula.** For each scored answered query,

\[
\mathrm{Hall}(q) = 1 - \mathrm{Faith}(q).
\]

The table value is the macro-average of \(\mathrm{Hall}(q)\). When every scored query has both fields set, that average equals \(1\) minus the average faithfulness. `run_generation` sets both from the same claim list.

Same exclusions as faithfulness: abstentions out, empty claim lists out, errors without a stored score out. `all_rows.hallucination_rate` keeps the older mean that included stored scores on abstentions.

**Why it matters.** Product copy often asks "how often do we make things up?" This column is the unsupported-claim rate under this judge. It is the right chart to put next to faithfulness, and the wrong chart to treat as an independent audit.

**Toy example.** Faithfulness \(2/3\) on the only scored query means hallucination rate \(1/3\). This is not a result from this repo.

**Range.** 0 to 1. Lower is better.

**Limitations.** Because it is \(1 - \mathrm{Faith}(q)\) averaged, it cannot disagree with faithfulness. A second model, a human, or a contradiction check could. Unsupported claims include both invented facts and claims the judge failed to match to the context. Abstentions are not counted as hallucinations in the headline number. The judge is not validated against human labels yet.

## Answer relevancy

**What it measures.** Whether the answer is about the user's question. The judge proposes questions that the answer could be addressing. Local embeddings compare those questions to the original question. The score is the mean clipped cosine.

**Formula.** Let \(q\) be the user question and \(q_1, \ldots, q_m\) the proposed questions. \(e(\cdot)\) is the local embedder, L2-normalized.

\[
\mathrm{AnswerRelevancy}(q) = \frac{1}{m} \sum_{j=1}^{m} \max\left(0, \cos(e(q), e(q_j))\right).
\]

The reported value is the macro-average over answered queries that produced at least one proposed question (`num_relevancy`). Abstentions do not call the question prompt and do not enter the mean.

Computed in `_answer_relevancy` and `summarize_answer_records` in `src/raglab/generation.py`. The prompt is `QUESTIONS_PROMPT`. Embeddings are local; the chat model is not used for the cosine. Negative cosines are clipped to 0, so the score stays in \([0, 1]\).

This is related to RAGAS answer relevancy and not a copy of it. RAGAS has used an extra penalty when the answer is non-committal. This code does not apply that penalty. Non-committal answers are handled only by the exact-string abstention rule. Any other hedging answer is scored like a normal answer.

**Why it matters.** An answer can be faithful to the context and still fail the user, because it explains a different fact. Answer relevancy is the "did we address the question?" number. Faithfulness will not catch that.

**Toy example.** Two proposed questions have cosines 0.8 and \(-0.2\). The clipped values are 0.8 and 0. The score is \(0.4\). This is not a result from this repo.

**Range.** 0 to 1. Higher is better. 1 means every proposed question points the same way as the user question in the embedding space.

**Limitations.** The score depends on the proposed questions. A vague answer can make the judge propose vague questions and still look relevant, or the reverse. Cosine on a small local embedding model is a rough topical match, not a proof the answer contains the asked fact. Abstentions are excluded, so this number does not describe the queries the system refused. The metric is not validated against human labels, and RAGAS agreement has not been run.

## Parity checks

`tests/test_parity.py` checks that our scores agree with two outside implementations, and it records the places they do not.

**pytrec_eval.** For a fixed toy ranking, mean nDCG@10, Recall@10, and MRR@10 match `pytrec_eval` (`ndcg_cut.10`, `recall.10`, `recip_rank`) within absolute tolerance \(10^{-8}\). MRR@10 is compared to reciprocal rank on a run that has already been cut at 10. A relevant document past rank 10 scores 0 on both sides. This is a check on the metric code, not a SciFact result.

**rank_bm25.** BM25 scores are not required to be identical.

- Our idf, in `src/raglab/retrieval/bm25.py`, is \(\log\left(1 + \frac{N - \mathrm{df} + 0.5}{\mathrm{df} + 0.5}\right)\). `rank_bm25` uses \(\log\frac{N - \mathrm{df} + 0.5}{\mathrm{df} + 0.5}\) with an epsilon floor. The \(1 +\) form stays positive when a term is in every document.
- We add each query term once. `rank_bm25` adds it again for each repeat in the query. The test checks both of those differences on purpose.

The BM25 saturation formula itself (tf, \(k_1\), \(b\), document length) is the usual Okapi form. The 300-query run records \(k_1 = 1.5\) and \(b = 0.75\) in `results/scifact/metrics.json` under `config.bm25`.

Parity does not cover context precision, the LLM-judge metrics, or latency.

## Dev and test in the ablation

`raglab ablate` (`src/raglab/ablation.py`) walks stages in order: chunking, query transform, retrieval, rerank. At each stage it tries the options in the config, scores each one on the dev split, and keeps the option with the highest dev nDCG@10. A tie keeps the earlier option, because the comparison is strict `>`. It then freezes that choice and scores the new pipeline on the test split. `delta_test_ndcg@10` is that test score minus the previous stage's test score. It is not dev minus test.

`results/ablation/results.md` records the split used for the published table: dev is the SciFact train split, 80 queries out of 809 available, seed 7; test is the official test split, all 300 queries. Local stages have estimated cost 0.00 in that file.

**Why choosing on the test set overfits.** The test number is the one you will quote as "how this does on unseen queries." If you also pick the winner by looking at those same queries, you have searched several options for the one that got lucky on that sample. The quoted gain is biased upward. A new sample of queries will, on average, look worse than the option you selected. Holding the choice to a dev split, then touching the test split only to score the frozen pipeline, is what keeps the test number from being part of the search.

That protection is only as good as the dev split. Here dev is 80 training queries, not a slice of the test set. It can still prefer a pipeline that fits the training claims and moves less on the test set. The delta column is there so you can see the test movement after the dev choice, not instead of a dev choice.

The ablation's `test_latency_ms` is the batched pipeline time on all test queries, not `online_latency_ms`. Compare those columns only after reading "Online latency" and "Indexing and offline cost" above.

## Where this code differs from the textbook

These are the differences that change how to read the tables.

- Recall, MRR, and nDCG are computed after collapsing chunks to documents, keeping the best rank. A textbook formula that assumes the ranked list is already documents is the same math only after that collapse.
- Context precision stays on chunks, and it uses the RAGAS weighted sum, not \(\text{relevant chunks}/k\). Duplicate chunks of one relevant document count more than once.
- nDCG uses exponential gain \(2^{rel}-1\). With SciFact's binary labels the gain is 0 or 1, so it matches binary DCG. The implementation does not force grades to be binary if a future qrel uses richer grades.
- Queries with no positive label are omitted from the retrieval averages. They are not zeros.
- MRR is cut at k. A hit past k is 0.
- Online latency times `retrieve` at `hit_k` (60 in the recorded configs cited above), on a handful of queries. It is not `eval_seconds` and not the ablation's `test_latency_ms`.
- BM25 idf and repeated query terms differ from `rank_bm25`, as the parity test states.
- Hallucination rate is \(1 - \mathrm{faithfulness}\) on the same claims, not another model.
- Faithfulness, hallucination rate, and answer relevancy exclude the exact abstention `I do not know.` Answer relevancy does not add a separate non-committal penalty on top of that.
- Judge metrics are not human labels. Nothing in the result files is a completed RAGAS agreement number.
