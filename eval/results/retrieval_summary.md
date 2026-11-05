# Retrieval results (WP5)

Recall@k and MRR of three retrievers (BM25, dense, hybrid RRF) over three chunking variants, with and without
company + year filters, on the 73 gold-v1 questions that have gold sources. Produced by
`python eval/run_retrieval.py`; the tables between the GENERATED markers are rewritten on every run, the text
around them is kept. Design and choices: ADR-013 in `docs/decisions.md`.

## Setup

- **Corpus:** `data/sections/sections.jsonl` (1,260 Item documents from 120 filings, 48.6M chars).
- **Chunking** (`ingest/03_chunk.py`); every chunk starts with a context header such as
  `[AAPL 10-K FY2024 · Item 7 · MD&A]` and has a deterministic ID `<doc_id>_<variant>_<n>`:

  | Variant | Rule | Chunks |
  |---|---|---:|
  | C1 | fixed windows of 800 chars, 120 overlap | 71,886 |
  | C2 | section-aware: paragraphs packed to ~1,200 chars, 150 overlap, never across Items | 57,455 |
  | C3 | C2 for prose; each table kept whole with its caption (split with repeated header rows only above 4,000 chars) | 59,668 (11,744 tables) |

- **BM25:** `rank_bm25` Okapi on lower-cased words and numbers (`1,234.5` → `1234.5`), header included.
  Index build 16-22 s per variant.
- **Dense:** Chroma (cosine, HNSW), embeddings from `sentence-transformers/all-MiniLM-L6-v2` (384 dims) via
  `core.llm.embed()`. Speed was measured first on a sample of real chunks: Ollama
  `nomic-embed-text` 2.7 chunks/s, `BAAI/bge-small-en-v1.5` 6.9/s, MiniLM 23/s, i.e. MiniLM is 8.5x faster
  than nomic and 3.3x faster than bge-small (nomic would need ~6 h per variant). Full builds on a 12-core CPU,
  no GPU:

  | Variant | Chunks | Dense build | Chunks/s |
  |---|---:|---:|---:|
  | C1 | 71,886 | 83.1 min | 14.4 |
  | C2 | 57,455 | 61.5 min | 15.6 |
  | C3 | 59,668 | 55.3 min | 18.0 |

  A rebuild deletes and recreates the collection and checks that the vector count equals the chunk count, so
  re-indexing never duplicates vectors; loading an index checks that its vector IDs equal the chunk IDs. MiniLM reads at most 256 word pieces: the rest of a longer chunk is not
  embedded (C1 11%, C2 31%, C3 16% of chunks, measured on 3,000 per variant; BM25 still sees the whole chunk).
- **Hybrid:** reciprocal rank fusion of the top 50 from each retriever, score Σ 1/(60 + rank).
- **Scoring** (`eval/retrieval_eval.py`): a retrieved chunk is a hit when its (ticker, fiscal year, fiscal
  period, Item) equals a gold source (ADR-009). Follow-ups are searched with the previous question prepended.
- **Filters:** `none` searches all 120 filings. `company+year` restricts each search to the companies and fiscal
  years the question names, taken from the gold item's `companies` and `fiscal_years` (correct by construction,
  so an upper bound: the agent will have to extract them from the question, WP6); a question with several
  companies or years gets one search per (company, year), and the results are interleaved. Form and quarter are
  not filtered, so a 10-K question can still retrieve that year's 10-Qs, and vice versa.
- `s/query` is wall-clock on the same CPU and includes query embedding; the first dense run (C1, no filters)
  also includes loading the model.

<!-- GENERATED:START (eval/run_retrieval.py) -->
| Variant | Retriever | Filters | recall@1 | recall@5 | recall@10 | MRR | s/query |
|---|---|---|---:|---:|---:|---:|---:|
| C1 | bm25 | none | 0.103 | 0.267 | 0.397 | 0.196 | 0.54 |
| C1 | bm25 | company+year | 0.274 | 0.705 | 0.822 | 0.495 | 0.02 |
| C1 | dense | none | 0.055 | 0.356 | 0.527 | 0.196 | 0.52 |
| C1 | dense | company+year | 0.253 | 0.658 | 0.781 | 0.448 | 0.37 |
| C1 | hybrid | none | 0.123 | 0.370 | 0.521 | 0.273 | 0.65 |
| C1 | hybrid | company+year | 0.281 | 0.664 | 0.877 | 0.497 | 0.37 |
| C2 | bm25 | none | 0.123 | 0.322 | 0.479 | 0.238 | 0.49 |
| C2 | bm25 | company+year | 0.288 | 0.705 | 0.829 | 0.517 | 0.02 |
| C2 | dense | none | 0.082 | 0.301 | 0.562 | 0.221 | 0.05 |
| C2 | dense | company+year | 0.281 | 0.623 | 0.856 | 0.480 | 0.37 |
| C2 | hybrid | none | 0.116 | 0.322 | 0.541 | 0.246 | 0.59 |
| C2 | hybrid | company+year | 0.212 | 0.692 | 0.877 | 0.468 | 0.40 |
| C3 | bm25 | none | 0.144 | 0.308 | 0.425 | 0.226 | 0.48 |
| C3 | bm25 | company+year | 0.295 | 0.678 | 0.829 | 0.509 | 0.02 |
| C3 | dense | none | 0.137 | 0.349 | 0.486 | 0.252 | 0.03 |
| C3 | dense | company+year | 0.315 | 0.630 | 0.822 | 0.487 | 0.37 |
| C3 | hybrid | none | 0.158 | 0.301 | 0.541 | 0.259 | 0.57 |
| C3 | hybrid | company+year | 0.329 | 0.747 | 0.904 | 0.539 | 0.40 |

By question category, best setting (C3 hybrid, filters company+year):

| Category | n | recall@1 | recall@5 | recall@10 | MRR |
|---|---:|---:|---:|---:|---:|
| fact | 15 | 0.400 | 0.867 | 1.000 | 0.547 |
| ratio | 15 | 0.267 | 0.800 | 0.867 | 0.454 |
| period_comparison | 12 | 0.208 | 0.625 | 0.750 | 0.572 |
| multi_company | 10 | 0.150 | 0.600 | 0.800 | 0.495 |
| qualitative | 12 | 0.667 | 0.833 | 1.000 | 0.730 |
| multi_turn | 6 | 0.167 | 0.500 | 1.000 | 0.366 |
| refuse | 3 | 0.333 | 1.000 | 1.000 | 0.511 |

73 gold questions with gold sources (gold-v1); items without sources (out-of-scope refusals) are skipped.
<!-- GENERATED:END -->

## Interpretation

**Filters are the biggest lever.** Restricting to the right company and year about doubles recall@5 for every
retriever and variant (BM25 C1 0.267 → 0.705; dense C3 0.349 → 0.630) and lifts MRR from ~0.2 to ~0.5. Without
filters the same wording appears in 120 filings (every 10-K has "total revenue" and a risk-factor section), so
the right filing is often outranked by look-alikes from other companies and years. The agent should therefore
extract company and year from the question and filter before ranking (WP6).

**With filters, hybrid C3 is the best setting:** recall@5 0.747, recall@10 0.904, MRR 0.539, the highest of the
18 rows on every metric. With filters BM25 is the strongest single retriever (recall@5 0.678-0.705) and dense
trails it (0.623-0.658): questions name exact line items ("total liabilities", "diluted earnings per share"),
which is what term matching rewards. The two find different things: at rank 5, 13 questions are hit only by
BM25 and 9 only by dense on C3 (union 63 of 73), and RRF keeps most of both (59). Hybrid does not help every
variant: on C1 and C2 its recall@5 (0.664, 0.692) is below BM25's, and C2 hybrid has the lowest filtered
recall@1 (0.212), because RRF rewards chunks both lists rank moderately well over one list's clear first place.

**C3 (table-aware) helps where it should.** Whole tables with their caption keep a statement's title, units,
dates and rows together, which raises the top of the ranking for statement questions: C3 has the best filtered
recall@1 for BM25 (0.295), dense (0.315) and hybrid (0.329), and its fact questions reach recall@10 1.000. C2's
larger prose chunks help unfiltered BM25 (recall@5 0.322 vs 0.267 for C1), but C2 loses most from MiniLM's
256-token limit (31% of chunks truncated).

**Without filters dense beats BM25 on recall@10** (0.486-0.562 vs 0.397-0.479), and hybrid C1 has the best
unfiltered recall@5 (0.370) and MRR (0.273): meaning-based matching copes better with paraphrase when nothing
narrows the search, but no unfiltered setting is good enough to answer from.

**Where it still misses** (C3 hybrid, filtered, rank 5): 27 gold sources are missed. 18 of them are in the right
filing but the wrong Item, 17 being statement questions (gold Item 8) where the MD&A (Item 7) chunks that repeat
the same figures are ranked higher. For the answer these chunks often hold the right number, but the citation
rule (spec §8) asks for the statement, so they count as misses. The other 9 had no chunk of the gold filing in
the top 5. 8 of them ask about a 10-K: in 5 the same fiscal year's 10-Qs take its place (GS twice, JNJ, NVDA, UNH:
quarterly MD&A discusses the same topics), in 3 period comparisons the other year's 10-K does. The ninth is a
10-Q question (MSFT Q2) that retrieves Q1 and the 10-K. Form and quarter are not filtered. Period comparisons (recall@5 0.625) and multi-company questions (0.600)
are lowest because each needs two or more sources inside five results.

**Reading the numbers.** 73 questions: one question moves recall by 0.014, so differences below ~0.04 (three
questions) between rows are within noise. For a question with two gold sources recall@1 is at most 0.5, so
recall@1 understates period-comparison and multi-company retrieval. The refuse row holds the three "not
reported" traps (JPM gross profit, AMZN total liabilities, UNH R&D), whose gold source is the statement that
shows the line is absent.

**Next steps.** Use C3 + hybrid with company+year filters as the default retriever for WP6; extract the filters
from the question; add form and quarter filters (a question about an annual figure should search 10-Ks,
a quarterly one its quarter's 10-Q); prefer Item 8 chunks for statement metrics or
use the XBRL facts tool (WP3) for them; a reranker or a 512-token embedding model could address the remaining
ranking and truncation losses, at the cost of indexing time.
