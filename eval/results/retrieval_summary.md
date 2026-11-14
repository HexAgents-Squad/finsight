# Retrieval results (WP5)

Recall@k and MRR of three retrievers (BM25, dense, hybrid RRF) over three chunking variants, with and without
company + year filters, on the 73 gold-v1 questions that have gold sources. Produced by
`python eval/run_retrieval.py`; the tables between the GENERATED markers are rewritten on every run, the text
around them is kept. Design and choices: ADR-013 in `docs/decisions.md`; embedding model ADR-021; page
furniture removed from the corpus ADR-023 (W1-04).

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
  speed). Each text is cut to 512 tokens, the model's limit: that touches 0.0% of C1, 0.4% of C2 and 3.3% of C3
  chunks (C3's whole tables; every chunk counted, ADR-023; `eval/results/chunk_token_lengths.md` has the same count
  on the corpus before W1-04), and BM25 still sees every chunk whole. Full builds, 16 texts per request:

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
- **Filters:** `none` searches all 120 filings. `company+year` restricts each search to the companies and fiscal
  years the question names, taken from the gold item's `companies` and `fiscal_years` (correct by construction,
  so an upper bound: the agent will have to extract them from the question, WP6); a question with several
  companies or years gets one search per (company, year), and the results are interleaved. Form and quarter are
  not filtered, so a 10-K question can still retrieve that year's 10-Qs, and vice versa.
- `s/query` is wall-clock and includes embedding the query through the endpoint (one request per search, so a
  filtered question with several companies or years pays several).

<!-- GENERATED:START (eval/run_retrieval.py) -->
| Variant | Retriever | Filters | recall@1 | recall@5 | recall@10 | MRR | s/query |
|---|---|---|---:|---:|---:|---:|---:|
| C1 | bm25 | none | 0.103 | 0.322 | 0.445 | 0.240 | 0.60 |
| C1 | bm25 | company+year | 0.342 | 0.692 | 0.822 | 0.539 | 0.03 |
| C1 | dense | none | 0.123 | 0.295 | 0.500 | 0.235 | 0.06 |
| C1 | dense | company+year | 0.281 | 0.671 | 0.829 | 0.488 | 0.36 |
| C1 | hybrid | none | 0.144 | 0.342 | 0.493 | 0.268 | 0.65 |
| C1 | hybrid | company+year | 0.295 | 0.685 | 0.842 | 0.519 | 0.34 |
| C2 | bm25 | none | 0.116 | 0.274 | 0.452 | 0.223 | 0.51 |
| C2 | bm25 | company+year | 0.295 | 0.664 | 0.842 | 0.513 | 0.02 |
| C2 | dense | none | 0.089 | 0.288 | 0.514 | 0.219 | 0.05 |
| C2 | dense | company+year | 0.240 | 0.658 | 0.808 | 0.456 | 0.30 |
| C2 | hybrid | none | 0.103 | 0.308 | 0.452 | 0.207 | 0.55 |
| C2 | hybrid | company+year | 0.301 | 0.719 | 0.829 | 0.529 | 0.31 |
| C3 | bm25 | none | 0.137 | 0.301 | 0.418 | 0.227 | 0.54 |
| C3 | bm25 | company+year | 0.308 | 0.685 | 0.842 | 0.527 | 0.02 |
| C3 | dense | none | 0.110 | 0.425 | 0.589 | 0.264 | 0.05 |
| C3 | dense | company+year | 0.322 | 0.740 | 0.836 | 0.546 | 0.31 |
| C3 | hybrid | none | 0.103 | 0.336 | 0.548 | 0.230 | 0.56 |
| C3 | hybrid | company+year | 0.301 | 0.774 | 0.870 | 0.550 | 0.30 |

By question category, best setting (C3 hybrid, filters company+year):

| Category | n | recall@1 | recall@5 | recall@10 | MRR |
|---|---:|---:|---:|---:|---:|
| fact | 15 | 0.267 | 0.867 | 0.933 | 0.503 |
| ratio | 15 | 0.333 | 0.867 | 0.867 | 0.484 |
| period_comparison | 12 | 0.250 | 0.583 | 0.750 | 0.626 |
| multi_company | 10 | 0.100 | 0.650 | 0.750 | 0.508 |
| qualitative | 12 | 0.583 | 0.833 | 0.917 | 0.679 |
| multi_turn | 6 | 0.000 | 0.667 | 1.000 | 0.381 |
| refuse | 3 | 0.667 | 1.000 | 1.000 | 0.778 |

73 gold questions with gold sources (gold-v1); items without sources (out-of-scope refusals) are skipped.
<!-- GENERATED:END -->

## Interpretation

**Filters are the biggest lever.** Restricting to the right company and year about doubles recall@5 for every
retriever and variant (BM25 C1 0.322 → 0.692; dense C3 0.425 → 0.740) and lifts MRR from ~0.2 to ~0.5. Without
filters the same wording appears in 120 filings (every 10-K has "total revenue" and a risk-factor section), so
the right filing is often outranked by look-alikes from other companies and years. The agent should therefore
extract company and year from the question and filter before ranking (WP6).

**With filters, hybrid C3 is the best setting:** recall@5 0.774, recall@10 0.870, MRR 0.550, the highest of the
18 rows on all three (recall@1 is led by C1 BM25, 0.342, against 0.301). On C3, dense with bge-small is now the
stronger single retriever (recall@5 0.740 vs BM25 0.685); on C1 and C2 the two are level (0.671 vs 0.692,
0.658 vs 0.664). They still find different things: at rank 5 on C3, 9 questions are hit only by BM25 and 10 only
by dense (union 67 of 73), and hybrid hits 62: 60 of those 67 plus 2 that neither finds alone in its top 5.
Hybrid helps C2 (0.719, above both its parts) and C3, and is level on C1 (0.685 vs BM25 0.692).

**C3 (table-aware) helps where it should.** Whole tables with their caption keep a statement's title, units,
dates and rows together: C3 has the best filtered recall@5 for dense (0.740) and hybrid (0.774), and its fact
and ratio questions reach recall@5 0.867. For BM25 the three variants are within noise (0.664-0.692). With
bge-small's 512-token limit C2's larger chunks are no longer cut (0.4% vs 23% under MiniLM's 256).

**Without filters dense is the best retriever on C3** (recall@5 0.425 and recall@10 0.589, the best unfiltered
rows; C1 hybrid has the best unfiltered MRR, 0.268), but no unfiltered setting is good enough to answer from; see below for why
unfiltered dense lost ground in W1-04.

**What W1-04 changed.** Two things changed since the previous table: the corpus lost its page furniture and the
dense model moved from MiniLM to bge-small (ADR-021). They are separated here question by question (recall@5,
two-sided sign test on the questions that differ; differences of a few questions are noise).
- *Furniture, BM25 (same retriever, only the text changed):* every row moved within noise, -0.048 to +0.055 in
  recall@5; the most lopsided are C1 unfiltered (8 questions better, 3 worse, p = 0.23) and C2 unfiltered
  (2 / 7, p = 0.18), in opposite directions. Removing furniture
  did not cost term matching anything, and gained nothing measurable on 73 questions either.
- *Furniture, dense (bge-small on C3, against the W1-16 run on the old corpus, `eval/results/embedding_comparison`):*
  with filters level, 0.788 → 0.740 (3 better, 6 worse, p = 0.51), and hybrid with filters 0.760 → 0.774 (3 / 1).
  Without filters dense lost: 0.555 → 0.425 (3 better, 14 worse, p = 0.013). The running headers and footers
  were naming the company in the text ("Bank of America Corporation and Subsidiaries", "Apple Inc. | 2024 Form
  10-K | 23"): 22.8% of C3 chunks contained their company's name before W1-04 and 12.8% after, while the chunk
  header gives only the ticker. In 3 of the 14 losses the asked-about company left the top 5 altogether ("Bank
  of America" → Citigroup, "Coca-Cola" → NVIDIA and Walmart, "Apple" → NVIDIA and Pfizer); in the others the
  company stayed but the year and form cue ("2024 Form 10-K") went, and another year's filing or a 10-Q took
  the place. With filters company and year are fixed, so nothing is lost there. The fix is to name the
  company in every chunk's header instead of in a fifth of them by accident of page breaks (a follow-up: it
  changes the chunk text, so all three variants are re-embedded, ~95 min).
- *Net, against the previous committed table (MiniLM on the old corpus):* C3 hybrid with filters 0.747 →
  0.774 (6 better, 3 worse, p = 0.51), C3 dense with filters 0.630 → 0.740 (16 / 8, p = 0.15).

**Where it still misses** (C3 hybrid, filtered, rank 5): 25 gold sources are missed (27 before). 22 of them are
in the right filing but the wrong Item, 21 being statement questions (gold Item 8, or Part I Item 1 for a 10-Q)
where the MD&A chunks that repeat the same figures are ranked higher. For the answer these chunks often hold the
right number, but the citation rule (spec §8) asks for the statement, so they count as misses. The other 3 (9
before) had no chunk of the gold filing in the top 5; all three ask about a 10-K and get that fiscal year's
10-Qs instead (GOOGL in a period comparison, NVIDIA in a multi-company question, NVIDIA in a qualitative one).
Form and quarter are not filtered. Period comparisons (recall@5 0.583) and multi-company questions (0.650) are
lowest because each needs two or more sources inside five results.

**Reading the numbers.** 73 questions: one question moves recall by 0.014, so differences below ~0.04 (three
questions) between rows are within noise. For a question with two gold sources recall@1 is at most 0.5, so
recall@1 understates period-comparison and multi-company retrieval. The refuse row holds the three "not
reported" traps (JPM gross profit, AMZN total liabilities, UNH R&D), whose gold source is the statement that
shows the line is absent.

**Next steps.** Use C3 + hybrid with company+year filters as the default retriever for WP6 (W1-05 measures it
with question-derived filters); extract the filters from the question; add form and quarter filters (a question
about an annual figure should search 10-Ks, a quarterly one its quarter's 10-Q); prefer Item 8 chunks for
statement metrics or use the XBRL facts tool (WP3) for them; name the company in the chunk header (above); a
reranker could address the remaining ranking losses.
