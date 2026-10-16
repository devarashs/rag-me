# rag-me roadmap

A public RAG bot that answers recruiter questions about Arash from the Markdown in
`data/`. Built framework-free (no LangChain or LlamaIndex) so every stage of the
pipeline is written and understood by hand.

**Stack:** Python 3.12 (uv) · Gemini via Google AI Studio for generation and
embeddings · Neon Postgres + pgvector · FastAPI on Vercel.

Each item is one shippable outcome. Work top to bottom; `<-- in progress` marks the
current item.

## Phase 1: answers questions locally

- [x] Project runs under uv with ruff and pytest, and config loads from `.env`,
      failing at startup with a clear message when a required value is missing
- [x] Knowledge base splits into one chunk per `##` section, tagged with source
      file and heading; `data/README.md` is skipped; malformed files fail loudly
      Result: 86 chunks from 9 files, 10–252 words each (`uv run rag-me chunks`).
- [x] `ingest` embeds chunks with Gemini and stores them in Neon; re-runs only
      re-embed changed sections and remove deleted ones
      Result: 86 chunks embedded (gemini-embedding-2, 768 dims) in ~9s; a re-run
      embeds nothing; schema via `uv run alembic upgrade head`.
- [x] `ask` answers from the top matching chunks with cited sources, and says it
      does not know (pointing to Arash's email) when nothing relevant is retrieved
      Result: `uv run rag-me ask "..." --verbose`. Generation on
      gemini-3.1-flash-lite (~1s to first words; gemini-3.8-flash took ~6.6s).
      Relevance gate at 0.62 cosine similarity, set from 18 sample questions:
      off-topic peaked at 0.59, on-topic started at 0.65. Personal-detail
      questions such as salary pass the gate (0.77) and are declined by the model's instructions instead.
      Known weakness: vague questions ("what does he do") retrieve poorly; see
      query rewriting in Phase 4.

## Phase 2: online

- [x] `POST /api/ask` validates input, streams the answer, and enforces a
      per-visitor rate limit plus a global daily cap stored in Postgres
      Result: Server-Sent Events (sources, delta, done/error); Problem Details for
      422/429/503; 10 questions per visitor per 10 minutes, 200 per UTC day; IPs
      stored only as HMACs. Run locally: `uv run uvicorn app:app --reload`.
- [x] A single chat page, plain HTML and JS, served by the same app
      Result: `web/`, served via `app.frontend()`. Streams answers, shows cited
      sources, handles loading, empty, error, cut-off and rate-limited states;
      light and dark themes meet WCAG AA; checked at 375px and desktop.
- [x] Deployed on Vercel from GitHub, secrets in Vercel environment settings
      Result: https://ask.devarash.icu (Hobby plan; pushes to main deploy;
      rag-me.vercel.app is the Vercel default domain).
      Function runs in iad1 next to Neon us-east-1; page assets served from the
      CDN. Measured from a local machine via the fra1 edge: ~0.95s to the sources event, 0.7s for an
      off-topic refusal; first answer words depend on Gemini (0.9-8s observed).
      Per-deployment URLs sit behind Vercel Deployment Protection; share the
      production domain.

## Phase 3: measure

- [x] Evaluation set of ~30 questions, including ones the bot must decline
      (personal details the knowledge base does not cover), scoring retrieval hit
      rate and answer faithfulness; baseline recorded
      Result: 34 public cases in `evals/cases.toml` plus git-ignored private
      cases; `uv run rag-me eval`, judged by gemini-3.8-flash in batches of 8
      (the judge's free tier is ~20 requests/day; batched verdicts matched
      one-at-a-time verdicts on 9/9 answers).
      Baseline (2026-10-09): 36/37 passed. Retrieval hit@1 0.84, hit@5 0.92,
      MRR 0.88. All 34 generated answers grounded, 96% cited, every
      not-covered question declined, no injection followed. Answer latency
      p50 1.6s, p95 10.9s. The one failure, "kubernetes", is a retrieval miss:
      the term appears once, inside a long multi-topic section.
- [x] The rag-me section of `data/07-ai-engineering.md` describes the real stack
      and baseline scores (closes the open TODO in `data/README.md`)

## Phase 4: improve (each change measured against the evaluation set)

- [x] Hybrid search: Postgres full-text combined with vector similarity
      Motivated by the "kubernetes" miss: rare exact terms are diluted in embeddings.
      Result (2026-10-09): BM25 over a generated tsvector column, fused with
      vector rank by reciprocal rank fusion (k=60), one SQL round trip; the gate
      still uses vector similarity. On the same 25 answer cases: hit@1 0.84->0.88,
      hit@5 0.92->0.96, MRR 0.88->0.90; gate unchanged. New single-mention tool
      cases (sentry, twilio, onnx, react-admin, kubernetes): 2/5 -> 5/5 with
      HYBRID_SEARCH=false vs true. Caveat: keyword-only finds land at rank ~5,
      the last slot, because RRF ties go to higher vector similarity.
      Follow-up (2026-10-16): new IT-ops content pushed "sentry" and
      "managed-people" out of the top 5, because common question terms ("use" in
      24% of sections, "manag" in 22%) still counted as keywords and gave
      generic sections double RRF credit. Keyword terms are now limited to
      those in at most 15% of sections; a sweep over the 32 answer cases showed
      10-15% best (hit@5 0.91 -> 0.97, MRR 0.89 -> 0.92), 25-50% losing two
      cases and 5% losing first-place hits. Chosen on the eval set itself, so
      treat it as measured, not proven.
      Remaining: a fully graded run; Gemini returned 503s and exhausted the
      judge quota on 2026-10-15 and 2026-10-16, so most fact cases are ungraded.
- [ ] Re-ranking of retrieved chunks
- [ ] Query rewriting for vague or multi-part questions
- [ ] Request tracing (Langfuse free tier)
- [ ] Prompt-injection hardening  <-- in progress
      Red team (2026-10-16): 12 attacks run through the real pipeline (forged
      source block, fake SYSTEM message, instruction and source dumps, role-play
      as Arash, confirming a false claim, a planted contact address, French
      injection, HTML/script, opinion bait, a "negative review"). All resisted;
      the negative review was answered only with the published gaps, neutrally.
      No defensive code needed yet; 7 attacks added to evals/cases.toml as
      permanent regression cases. Remaining: a graded run of them (judge quota).
