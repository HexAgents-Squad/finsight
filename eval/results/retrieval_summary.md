# Retrieval results (WP5)

Recall@k and MRR of three retrievers (BM25, dense, hybrid RRF) over three chunking variants and three filter
settings, plus the W1-05 trials on C3 (period filter, statements first, cross-encoder reranking), on the 73
gold-v1 questions that have gold sources. Produced by `python eval/run_retrieval.py --rerank`; the tables between
the GENERATED markers are rewritten on every full run, the text around them is kept. Design and choices: ADR-013
in `docs/decisions.md`; embedding model ADR-021; page furniture removed from the corpus ADR-023 (W1-04);
retrieval default ADR-017 (W1-05).

## Setup

- **Corpus:** `data/sections/sections.jsonl` (1,260 Item documents from 120 filings, 48.25M chars). Since W1-04
  it holds no "Table of Contents" links, bare page numbers, running footers or running headers (16,817 lines,
  0.69% of the characters; `eval/results/furniture_removal.md`).
- **Chunking** (`ingest/03_chunk.py`); every chunk starts with a context header such as
  `[AAPL 10-K FY2024 · Item 7 · MD&A]` and has a deterministic ID `<doc_id>_<variant>_<n>`:

  | Variant | Rule | Chunks |
  |---|---|---:|
  | C1 | fixed windows of 800 chars, 120 overlap | 71,407 |
  | C2 | section-aware: paragraphs packed to ~1,200 chars, 150 overlap, never across Items | 57,056 |
  | C3 | C2 for prose; each table kept whole with its caption (split with repeated header rows only above 4,000 chars) | 58,649 (11,736 tables) |

- **BM25:** `rank_bm25` Okapi on lower-cased words and numbers (`1,234.5` → `1234.5`), header included.
  Index build 20-34 s per variant.
- **Dense:** Chroma (cosine, HNSW), 384-dimension embeddings from `bge-small-en-v1.5` on the course's embedding
  endpoint via `core.llm.embed()` (ADR-021; it replaced `all-MiniLM-L6-v2`, which ADR-013 had picked for its local
  speed). Each embedded text (context header plus chunk) is cut to 512 tokens, the model's limit: that touched
  no C1 chunk, 368 C2 chunks (0.6%) and 2,054 C3 chunks (3.5%, C3's whole tables), counted in the build; BM25
  still sees every chunk whole. Full builds, 16 texts per request:

  | Variant | Chunks | Dense build | Chunks/s |
  |---|---:|---:|---:|
  | C1 | 71,407 | 31.1 min | 38.3 |
  | C2 | 57,056 | 32.0 min | 29.8 |
  | C3 | 58,649 | 31.0 min | 31.5 |

  A rebuild deletes and recreates the collection and checks that the vector count equals the chunk count, so
  re-indexing never duplicates vectors; loading an index checks that its vector IDs equal the chunk IDs and that
  it was built with the configured model. The collection records the model without its backend
  (`bge-small-en-v1.5`), so a query embedded locally with `BAAI/bge-small-en-v1.5` (the same vectors) is accepted.
- **Hybrid:** reciprocal rank fusion of the top 50 from each retriever, score Σ 1/(60 + rank).
- **Scoring** (`eval/retrieval_eval.py`): a retrieved chunk is a hit when its (ticker, fiscal year, fiscal
  period, Item) equals a gold source (ADR-009). Follow-ups are searched with the previous question prepended.
- **Filters:** `none` searches all 120 filings. `question` restricts each search to the companies and fiscal
  years read from the question by `retrieval/entities.py` (names, aliases and capitalised tickers from
  `config/companies.yaml`, four-digit years and "FY24", "latest"; a follow-up keeps what it does not name from
  the turn before): what the agent can do. `company+year` takes them from the gold item's `companies` and
  `fiscal_years` (correct by construction, an upper bound). A question with several companies or years gets one
  search per (company, year), and the results are interleaved.
- **Trials (C3, hybrid; W1-05):** `+period` also filters on the fiscal period the question names (FY when it
  names no quarter; "Q2", "second quarter", "six months" or "first half" = Q2, Q4 = the 10-K), which also picks
  the form; a period the corpus does not index for that year is dropped. `+statement` (the Item 8 preference,
  `retrieval/statement.py`): for a question that asks for a figure (a metric word and no "why", "drove",
  "describe" ...), a second search restricted to the statement Item (Item 8, or Part I Item 1 of a 10-Q) is
  interleaved first. `hybrid+rerank` reorders the hybrid top 30 with `cross-encoder/ms-marco-MiniLM-L-6-v2`
  (`retrieval/rerank.py`, revision pinned, run locally on the CPU), each passage given as
  `[TICKER FORM FYyyyy · Item]` plus its text.
- **figure@5** (added in W1-05, after the first run showed the Item-level score and the passages disagreeing):
  of the 59 questions whose gold answer states a figure as the gold set writes it ("391,035"), how many have it
  verbatim in one of the first 5 chunks (any one of its figures for a multi-figure answer). recall@k asks for the
  right Item; figure@5 asks for the chunk that answers.
- **Paired tests:** the pairs in the third table were fixed before the first W1-05 run (`PAIRS` in the script).
  For each, the questions better and worse in b than in a, with a two-sided sign test on those that differ.
- `s/query` is wall-clock per question on a laptop CPU and moves by up to ~50% between runs with machine load.
  Query embeddings are cached for the run, so only the first search of a question pays the endpoint request
  (the unfiltered dense rows show ~0.00, the filtered ones Chroma's filtered search); a statement trial
  searches twice per filter, and the
  reranker scores 30 passages per search.
- **Reproducibility:** BM25, the endpoint's vectors (identical bytes for the same text) and every filtered search
  reproduce exactly; the `company+year` run files are byte-identical to the W1-04 ones. Unfiltered dense top-50
  lists can differ by a few near-equal chunks between processes (Chroma's HNSW search is approximate; measured
  with fixed query vectors). Against the W1-04 run files this changed the order of near-equal chunks for one
  question each in C1 and C2 hybrid `none` (no score moved in this run); in another W1-05 run it moved C2 hybrid
  `none` by 3 questions (recall@10 0.459 instead of 0.452, MRR 0.210 instead of 0.207).

<!-- GENERATED:START (eval/run_retrieval.py) -->
| Variant | Retriever | Filters | recall@1 | recall@5 | recall@10 | MRR | figure@5 | s/query |
|---|---|---|---:|---:|---:|---:|---:|---:|
| C1 | bm25 | none | 0.103 | 0.322 | 0.445 | 0.240 | 9/59 | 0.66 |
| C1 | bm25 | question | 0.342 | 0.692 | 0.822 | 0.539 | 17/59 | 0.03 |
| C1 | bm25 | company+year | 0.342 | 0.692 | 0.822 | 0.539 | 17/59 | 0.03 |
| C1 | dense | none | 0.123 | 0.295 | 0.500 | 0.235 | 16/59 | 0.06 |
| C1 | dense | question | 0.281 | 0.671 | 0.829 | 0.488 | 29/59 | 0.38 |
| C1 | dense | company+year | 0.281 | 0.671 | 0.829 | 0.488 | 29/59 | 0.33 |
| C1 | hybrid | none | 0.144 | 0.342 | 0.493 | 0.268 | 17/59 | 0.70 |
| C1 | hybrid | question | 0.295 | 0.685 | 0.842 | 0.519 | 24/59 | 0.34 |
| C1 | hybrid | company+year | 0.295 | 0.685 | 0.842 | 0.519 | 24/59 | 0.34 |
| C2 | bm25 | none | 0.116 | 0.274 | 0.452 | 0.223 | 9/59 | 0.63 |
| C2 | bm25 | question | 0.295 | 0.664 | 0.842 | 0.513 | 21/59 | 0.03 |
| C2 | bm25 | company+year | 0.295 | 0.664 | 0.842 | 0.513 | 21/59 | 0.03 |
| C2 | dense | none | 0.089 | 0.288 | 0.514 | 0.219 | 15/59 | 0.00 |
| C2 | dense | question | 0.240 | 0.658 | 0.808 | 0.456 | 27/59 | 0.32 |
| C2 | dense | company+year | 0.240 | 0.658 | 0.808 | 0.456 | 27/59 | 0.30 |
| C2 | hybrid | none | 0.103 | 0.308 | 0.452 | 0.207 | 17/59 | 0.55 |
| C2 | hybrid | question | 0.301 | 0.719 | 0.829 | 0.529 | 24/59 | 0.31 |
| C2 | hybrid | company+year | 0.301 | 0.719 | 0.829 | 0.529 | 24/59 | 0.29 |
| C3 | bm25 | none | 0.137 | 0.301 | 0.418 | 0.227 | 3/59 | 0.49 |
| C3 | bm25 | question | 0.308 | 0.685 | 0.842 | 0.527 | 9/59 | 0.02 |
| C3 | bm25 | company+year | 0.308 | 0.685 | 0.842 | 0.527 | 9/59 | 0.02 |
| C3 | dense | none | 0.110 | 0.425 | 0.589 | 0.264 | 19/59 | 0.00 |
| C3 | dense | question | 0.322 | 0.740 | 0.836 | 0.546 | 30/59 | 0.24 |
| C3 | dense | company+year | 0.322 | 0.740 | 0.836 | 0.546 | 30/59 | 0.29 |
| C3 | hybrid | none | 0.103 | 0.336 | 0.548 | 0.230 | 12/59 | 0.53 |
| C3 | hybrid | question | 0.301 | 0.774 | 0.870 | 0.550 | 23/59 | 0.25 |
| C3 | hybrid | company+year | 0.301 | 0.774 | 0.870 | 0.550 | 23/59 | 0.25 |

Trials on C3 (period filter, statement Item first, cross-encoder reranking):

| Retriever | Filters | recall@1 | recall@5 | recall@10 | MRR | figure@5 | s/query |
|---|---|---:|---:|---:|---:|---:|---:|
| hybrid | question | 0.301 | 0.774 | 0.870 | 0.550 | 23/59 | 0.25 |
| hybrid | company+year | 0.301 | 0.774 | 0.870 | 0.550 | 23/59 | 0.25 |
| hybrid | question+period | 0.377 | 0.890 | 0.959 | 0.633 | 25/59 | 0.55 |
| hybrid | question+period+statement | 0.808 | 0.986 | 1.000 | 0.975 | 31/59 | 0.97 |
| hybrid | company+year+period | 0.377 | 0.890 | 0.959 | 0.633 | 25/59 | 0.51 |
| hybrid | company+year+period+statement | 0.808 | 0.986 | 1.000 | 0.975 | 31/59 | 0.95 |
| hybrid+rerank | question | 0.281 | 0.726 | 0.836 | 0.493 | 29/59 | 6.39 |
| hybrid+rerank | company+year | 0.281 | 0.726 | 0.836 | 0.493 | 29/59 | 5.87 |
| hybrid+rerank | question+period | 0.397 | 0.856 | 0.932 | 0.609 | 37/59 | 6.00 |
| hybrid+rerank | question+period+statement | 0.822 | 1.000 | 1.000 | 0.986 | 42/59 | 9.46 |
| hybrid+rerank | company+year+period | 0.397 | 0.856 | 0.932 | 0.609 | 37/59 | 4.98 |
| hybrid+rerank | company+year+period+statement | 0.822 | 1.000 | 1.000 | 0.986 | 42/59 | 9.93 |

Paired on C3: questions better / worse with b than with a, by recall@5 (all questions with gold sources) and by the answer figure in the first 5 chunks (questions whose gold answer states one), with a two-sided sign test on the questions that differ:

| a | b | recall@5 better / worse / same | p | figure better / worse | p |
|---|---|---|---:|---|---:|
| hybrid, question | hybrid, company+year | 0 / 0 / 73 | 1.000 | 0 / 0 | 1.000 |
| hybrid, question | hybrid, question+period | 11 / 1 / 61 | 0.006 | 2 / 0 | 0.500 |
| hybrid, question+period | hybrid, question+period+statement | 11 / 0 / 62 | 0.001 | 8 / 2 | 0.109 |
| hybrid, question | hybrid+rerank, question | 6 / 12 / 55 | 0.238 | 14 / 8 | 0.286 |
| hybrid, question+period | hybrid+rerank, question+period | 4 / 8 / 61 | 0.388 | 17 / 5 | 0.017 |
| hybrid, question+period+statement | hybrid+rerank, question+period+statement | 1 / 0 / 72 | 1.000 | 14 / 3 | 0.013 |
| hybrid, company+year | hybrid, company+year+period+statement | 21 / 0 / 52 | <0.001 | 10 / 2 | 0.039 |

By question category, best setting (C3 hybrid+rerank, filters question+period+statement):

| Category | n | recall@1 | recall@5 | recall@10 | MRR |
|---|---:|---:|---:|---:|---:|
| fact | 15 | 1.000 | 1.000 | 1.000 | 1.000 |
| ratio | 15 | 1.000 | 1.000 | 1.000 | 1.000 |
| period_comparison | 12 | 0.500 | 1.000 | 1.000 | 1.000 |
| multi_company | 10 | 0.500 | 1.000 | 1.000 | 1.000 |
| qualitative | 12 | 0.833 | 1.000 | 1.000 | 0.917 |
| multi_turn | 6 | 1.000 | 1.000 | 1.000 | 1.000 |
| refuse | 3 | 1.000 | 1.000 | 1.000 | 1.000 |

By question category, best setting with question-derived filters and no reranker (C3 hybrid, filters question+period+statement):

| Category | n | recall@1 | recall@5 | recall@10 | MRR |
|---|---:|---:|---:|---:|---:|
| fact | 15 | 1.000 | 1.000 | 1.000 | 1.000 |
| ratio | 15 | 1.000 | 1.000 | 1.000 | 1.000 |
| period_comparison | 12 | 0.500 | 1.000 | 1.000 | 1.000 |
| multi_company | 10 | 0.500 | 1.000 | 1.000 | 1.000 |
| qualitative | 12 | 0.750 | 0.917 | 1.000 | 0.847 |
| multi_turn | 6 | 1.000 | 1.000 | 1.000 | 1.000 |
| refuse | 3 | 1.000 | 1.000 | 1.000 | 1.000 |

Question-derived filters against the items' own (retrieval/entities.py; exact matches, items with gold sources; a period counts as FY when the question names none):

| Set | n | companies | fiscal years | periods | all three |
|---|---:|---:|---:|---:|---:|
| development set | 14 | 14 | 14 | 14 | 14 |
| gold-v1 | 73 | 73 | 73 | 73 | 73 |

73 gold questions with gold sources (gold-v1); items without sources (out-of-scope refusals) are skipped.
<!-- GENERATED:END -->

## Interpretation

**Filters are the biggest lever.** Restricting to the right company and year about doubles recall@5 for every
retriever and variant (BM25 C1 0.322 → 0.692; dense C3 0.425 → 0.740) and lifts MRR from ~0.2 to ~0.5. Without
filters the same wording appears in 120 filings (every 10-K has "total revenue" and a risk-factor section), so
the right filing is often outranked by look-alikes from other companies and years. The agent should therefore
extract company and year from the question and filter before ranking (WP6).

**With filters, hybrid C3 is the best setting:** recall@5 0.774, recall@10 0.870, MRR 0.550, the highest of the
grid's filtered rows on all three (recall@1 is led by C1 BM25, 0.342, against 0.301). On C3, dense with bge-small is now the
stronger single retriever (recall@5 0.740 vs BM25 0.685); on C1 and C2 the two are level (0.671 vs 0.692,
0.658 vs 0.664). They still find different things: at rank 5 on C3, 9 questions are hit only by BM25 and 10 only
by dense (union 67 of 73), and hybrid hits 62: 60 of those 67 plus 2 that neither finds alone in its top 5.
Hybrid helps C2 (0.719, above both its parts) and C3, and is level on C1 (0.685 vs BM25 0.692).

**C3 (table-aware) helps where it should.** Whole tables with their caption keep a statement's title, units,
dates and rows together: C3 has the best filtered recall@5 for dense (0.740) and hybrid (0.774). For BM25 the
three variants are within noise by recall@5 (0.664-0.692), but BM25 on C3 rarely ranks the chunk holding the
answer figure first (figure@5 9/59 with filters, against 17 on C1 and 21 on C2), whereas dense C3 has the most
(30/59). With
bge-small's 512-token limit C2's larger chunks are almost never cut (0.6%, against about 23% at MiniLM's 256).

**Without filters dense is the best retriever on C3** (recall@5 0.425 and recall@10 0.589, the best unfiltered
rows; C1 hybrid has the best unfiltered MRR, 0.268), but no unfiltered setting is good enough to answer from; see below for why
unfiltered dense lost ground in W1-04.

**What W1-04 changed.** Two things changed since the previous table: the corpus lost its page furniture and the
dense model moved from MiniLM to bge-small (ADR-021). They are separated here question by question: the counts
are questions with a gold source in the top 5 in one run and not in the other (better / worse), with a
two-sided sign test on those; differences of a few questions are noise.
- *Furniture, BM25 (same retriever, only the text changed):* every row moved within noise, -0.048 to +0.055 in
  recall@5; the most lopsided are C1 unfiltered (8 questions better, 3 worse, p = 0.23) and C2 unfiltered
  (2 / 7, p = 0.18), in opposite directions. Removing furniture did not cost term matching anything, and gained
  nothing measurable on 73 questions either.
- *Furniture, dense (bge-small on C3, against the W1-16 run on the old corpus, `eval/results/embedding_comparison`):*
  with filters level, recall@5 0.788 → 0.740 (3 better, 6 worse, p = 0.51), and hybrid with filters 0.760 →
  0.774 (3 / 1). Without filters dense lost: 0.555 → 0.425 (3 better, 14 worse, p = 0.013). The running headers
  and footers were naming the company in the text ("Bank of America Corporation and Subsidiaries", "Apple Inc. |
  2024 Form 10-K | 23"): 22.8% of C3 chunks contained their company's name (the full name in
  `config/companies.yaml` without "Inc." or "Corporation", or a multi-word alias) before W1-04 and 12.8% after,
  while the chunk header gives only the ticker. In 6 of the 14 losses an asked-about company left the top 5
  altogether ("Bank of America" → Citigroup, "Coca-Cola" → NVIDIA, Walmart and AbbVie, "Apple" → NVIDIA,
  Pfizer, Amazon and Microsoft, and one company of each of three multi-company questions); in 4 the gold filing
  is still there but another of its Items outranks the gold one, and in 4 another filing of the same company
  takes the place, the year and form cue ("2024 Form 10-K") being gone. With filters company and year are
  fixed, so nothing is lost there. The fix is to name the company in every chunk's header instead of in some
  chunks by accident of page breaks (a follow-up: it changes the chunk text, so all three variants are
  re-embedded, ~95 min).
- *Net, against the previous committed table (MiniLM on the old corpus):* C3 hybrid with filters recall@5 0.747
  → 0.774 (6 better, 3 worse, p = 0.51), C3 dense with filters 0.630 → 0.740 (16 / 8, p = 0.15).

**W1-05: question-derived filters equal the gold ones.** Read from the question, companies, years and periods
match the gold items exactly on all 73 gold questions with sources and on 14 of 14 development questions, so the
`question` and `company+year` rows are identical in every variant (0 questions differ). One gold question was
read wrongly in the first run (A15, "first half of fiscal 2025" took no quarter); "first half" was added with
"six months" as the standard name of the Q2 year to date, with a test. The gold set is not a held-out test of
the reader after that fix, and real users will phrase things the rules do not know; a missed entity fails open
(no filter is a wider search, never a wrong one), while a wrongly read company or year narrows the search to the
wrong filing, which the planner can see in the filters it passes.

**The period filter helps.** Also filtering on the fiscal period (C3 hybrid): recall@5 0.774 -> 0.890 (11
better, 1 worse, p = 0.006), recall@10 0.959. It removes most of the misses W1-04 left where a 10-K question
got that year's 10-Qs.

**Statements first fixes the wrong-Item misses; read its recall with care.** With the statement Item interleaved
first for figure questions: recall@5 0.890 -> 0.986 (11 better, 0 worse, p = 0.001), recall@1 0.377 -> 0.808,
MRR 0.633 -> 0.975. Most of that is by construction: the gold source of a figure question is its statement
Item (spec §8), and the trial puts a chunk of that Item first. The passage-level check is smaller: figure@5 25
-> 31 of 59 (8 better, 2 worse, p = 0.11, not significant alone); against company+year without either trial,
period + statements gives 21 / 0 by recall@5 and 10 / 2 by figure (p = 0.039). In all 28 questions where this
setting misses the figure, the figure is in a chunk of the right Item, just not in the first five chunks (Item
8 holds every statement and the notes; figure@10 is 45 of 59). The only recall@5 miss left is E05 (what drove
Bank of America's net interest income), a qualitative question the statement trial does not touch.

**The reranker finds the answering chunk, not the Item.** By recall@5 the cross-encoder looks neutral or worse
(question filters 0.774 -> 0.726, 6 better / 12 worse, p = 0.24), but by figure@5 it is better in every pair
and significantly so with the trials: with the period filter 25 -> 37 (17 / 5, p = 0.017), with period +
statements 31 -> 42 (14 / 3, p = 0.013), where recall@5 reaches 1.000 (E05 found) and MRR 0.986. It reads the passage against
the question, so it moves the chunk with the asked-for line up, often an MD&A chunk that repeats it, which costs
Item-level recall. The price is time: ~5-6 s per question with one search, ~9-10 s with statements first
(two reranked searches per filter), against ~0.3-1.0 s without, on a laptop CPU, while the spec allows 60 s for
the whole question (§9) and an agent question makes several searches.

**Default (ADR-017).** `search_filings` keeps C3 hybrid and puts the statements first for a figure question
whose filters name a fiscal period; the planner passes the company, year and period the question names
(`retrieval/entities.py` reads them). That is the best setting without the reranker (recall@5 0.986, figure@5
31/59, ~1 s per question). Reranking is opt-in (`RETRIEVAL_RERANK=true`): it is the best setting found (figure@5
42/59) but adds ~4-6 s per search, so it waits until the agent's end-to-end latency is measured (W2); turning
it on is one setting, with no index change.

**Where W1-04 missed** (C3 hybrid, company+year, rank 5; before the W1-05 trials): 25 gold sources were missed
(27 before W1-04). 22 of them were in the right filing but the wrong Item, 21 being statement questions (gold
Item 8, or Part I Item 1 for a 10-Q) where the MD&A chunks that repeat the same figures ranked higher; the other
3 asked about a 10-K and got that fiscal year's 10-Qs. The W1-05 period filter and statements-first trial
address exactly these two patterns (above).

**Reading the numbers.** 73 questions: one question moves recall by 0.014, so differences below ~0.04 (three
questions) between rows are within noise. For a question with two gold sources recall@1 is at most 0.5, so
recall@1 understates period-comparison and multi-company retrieval. The refuse row holds the three "not
reported" traps (JPM gross profit, AMZN total liabilities, UNH R&D), whose gold source is the statement that
shows the line is absent.

**Next steps.** The planner (WP6) passes the question's company, year and period to `search_filings` (with
`retrieval/entities.py`); measure end-to-end latency and turn reranking on if it fits the 60 s budget; rank the
right statement table within Item 8 (the remaining figure misses), or answer statement metrics from the XBRL
facts tool (WP3); name the company in the chunk header (above); raise unfiltered dense search's `ef_search` (or
search exactly) so unfiltered rows reproduce exactly.
