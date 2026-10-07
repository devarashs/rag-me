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
      Result: https://rag-me.vercel.app (Hobby plan; pushes to main deploy).
      Function runs in iad1 next to Neon us-east-1; page assets served from the
      CDN. Measured from a local machine via the fra1 edge: ~0.95s to the sources event, 0.7s for an
      off-topic refusal; first answer words depend on Gemini (0.9-8s observed).
      Per-deployment URLs sit behind Vercel Deployment Protection; share the
      production domain.

## Phase 3: measure

- [ ] Evaluation set of ~30 questions, including ones the bot must decline
      (personal details the knowledge base does not cover), scoring retrieval hit
      rate and answer faithfulness; baseline recorded  <-- in progress
      Done: 34 public cases in `evals/cases.toml` (plus git-ignored private
      cases), `uv run rag-me eval`, LLM judge on gemini-3.8-flash.
      Retrieval baseline (2026-10-07): hit@1 0.84, hit@5 0.92, MRR 0.88; misses
      "kubernetes" and "what does he do". Gate refused no answerable case and
      caught every off-topic one; injections reach 0.75, so the gate does not
      stop them and the model's instructions must.
      Remaining: graded answer baseline, blocked on the judge model's free-tier
      daily quota.
- [ ] The rag-me section of `data/07-ai-engineering.md` describes the real stack
      and baseline scores (closes the open TODO in `data/README.md`)

## Phase 4: improve (each change measured against the evaluation set)

- [ ] Hybrid search: Postgres full-text combined with vector similarity
- [ ] Re-ranking of retrieved chunks
- [ ] Query rewriting for vague or multi-part questions
- [ ] Request tracing (Langfuse free tier)
- [ ] Prompt-injection hardening
