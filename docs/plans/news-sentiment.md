# Plan — Ticker-level sentiment on the news feed (Phase 2)

_Roadmap item (§5, Phase 2): "**Ticker-level sentiment on the news feed**; technical
indicators on the chart." The technical-indicators half shipped in PR #10. This change
delivers the **sentiment** half._

> The technical-indicators plan deferred sentiment, arguing it "depends on FinBERT
> scoring (an explicit Phase-3 deliverable) or a keyed external sentiment API
> (Marketaux/Finnhub), neither of which is self-contained at this point." This plan
> resolves that objection the same way semantic search did: ship a **self-contained,
> dependency-free baseline now**, with the heavyweight model as a drop-in upgrade later.
> A finance-aware lexicon analyzer needs no model download and no API key, exactly like
> the `HashingEmbedder` baseline that precedes `sentence-transformers`. **FinBERT becomes
> the Phase-3 drop-in upgrade** behind the same `SentimentAnalyzer` interface.

## Goal

Score each ingested news headline for financial sentiment, **persist** the score on
`news_articles` (so Phase 3 can aggregate/lag it as an ML feature), expose it over the
JSON API, and surface it on the dashboard news feed — matching the existing
ingest → enrich → store → API → UI pattern that semantic search already established.

## Approach (verified against current practice)

Financial text needs a **finance-specific** sentiment source: general-purpose lexicons
misread finance terms ("liability", "cost", "tax" are neutral in finance, not negative).
The standard interpretable baseline is the **Loughran–McDonald (LM)** financial word
lists; the SOTA is **FinBERT**. Research (2026) ranks FinBERT > LM on accuracy, but LM
remains the right *dependency-free, interpretable* baseline — and the emerging best
practice is a hybrid LM+BERT pipeline, which our pluggable interface keeps open.

We mirror `embeddings.py` precisely:

- **`SentimentAnalyzer` Protocol** — `name: str`, `score(texts: list[str]) -> list[float]`
  returning a polarity in `[-1, 1]` per text.
- **`LexiconSentiment` (default, dependency-free)** — an LM-inspired curated set of
  positive/negative financial terms. Per headline: tokenize, count polarity hits with
  light **negation handling** (a negator flips the next polarity word), score =
  `(pos − neg) / (pos + neg)`, `0.0` when no polarity words occur (neutral). Pure Python,
  deterministic, no model, no key.
- **`get_default_analyzer()`** — returns the baseline; override at call sites.
- **`classify(score)`** — maps a score to `positive` / `neutral` / `negative` using a
  small neutral band (`|score| <= 0.05`), or `unknown` for `None` (not-yet-scored).
- A `FinBertSentiment` upgrade is documented (Phase 3) behind the same interface — not
  built here, to avoid an unused heavyweight class (kept lean for `/simplify`).

## Storage

- Add `sentiment: Mapped[float | None]` to `NewsArticle` (nullable; `None` = not scored
  yet, exactly like `embedding`). Persisting it (vs. computing on read) lets Phase 3
  aggregate per-asset/day and lets the feed sort/filter without recompute.
- **No migration machinery** — consistent with the repo convention ("no migrations —
  Alembic later"); `init_db`'s `create_all` covers fresh DBs and all (SQLite) tests, just
  as it did when the `embedding` column was introduced.

## Enrichment (`search.py`, mirrors `embed_pending`)

- `score_sentiment_pending(session, analyzer=None, limit=500)` — score articles whose
  `sentiment IS NULL`, commit, return the count. Same shape and placement as
  `embed_pending` (the de-facto news-enrichment module).
- **`scheduler.py`** GDELT job calls it right after `embed_pending`, so the nightly/30-min
  poll keeps sentiment current alongside embeddings. Per-job errors stay isolated.

## API (`api/app.py`)

- `_article_record` gains `sentiment` (the rounded float, JSON-safe via `_num`) and
  `sentiment_label` (`classify(...)`). Flows automatically to `/api/news/recent` and
  `/api/news/search` (both keyword and semantic).

## Frontend (`api/static/index.html`)

- Each news item gets a colored sentiment dot/label in `.news-meta`
  (green = positive, red = negative, muted = neutral; hidden when `unknown`).
- A compact aggregate readout in the panel header (share of positive vs. negative across
  the currently shown feed) so the panel summarizes mood at a glance.
- No new dependencies; styling reuses the existing CSS variables.

## Tests (`tests/test_sentiment.py` + extend `tests/test_api.py`)

- Analyzer: positive/negative/neutral headlines score with the expected sign; empty /
  no-polarity text → `0.0`; negation flips polarity ("not a strong quarter" ≤ 0); scores
  stay within `[-1, 1]`; `score()` length matches input.
- `classify`: band thresholds and the `None → "unknown"` case.
- `score_sentiment_pending`: scores only NULL rows, is idempotent (a second call scores
  0), and respects `limit`.
- API: `/api/news/recent` and `/api/news/search` records include `sentiment` and
  `sentiment_label`; a clearly positive/negative seeded headline gets the right label.

## Out of scope

True per-ticker entity attribution (GDELT headlines aren't ticker-tagged), FinBERT
scoring and per-asset/day aggregation (Phase 3), configurable lexicons/thresholds, and
scoring article bodies (GDELT ArtList provides only titles).

Sources: [Loughran–McDonald + BERT for financial sentiment (ScienceDirect, 2025)](https://www.sciencedirect.com/science/article/pii/S1877050925015807) ·
[Evaluation of Sentiment Analysis in Finance: From Lexicons to Transformers](https://www.academia.edu/53978564/Evaluation_of_Sentiment_Analysis_in_Finance_From_Lexicons_to_Transformers) ·
[Large language models in finance: what is financial sentiment? (arXiv 2503.03612)](https://arxiv.org/pdf/2503.03612)
