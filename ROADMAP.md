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
- [ ] A single chat page  <-- next, plain HTML and JS, served by the same app
- [ ] Deployed on Vercel from GitHub, secrets in Vercel environment settings
      Assumption: Vercel Hobby. Revisit if Python cold starts or limits bite.

## Phase 3: measure

- [ ] Evaluation set of ~30 questions, including ones the bot must decline (personal
      details the knowledge base does not cover), scoring retrieval hit
      rate and answer faithfulness; baseline recorded
- [ ] The rag-me section of `data/07-ai-engineering.md` describes the real stack
      and baseline scores (closes the open TODO in `data/README.md`)

## Phase 4: improve (each change measured against the evaluation set)

- [ ] Hybrid search: Postgres full-text combined with vector similarity
- [ ] Re-ranking of retrieved chunks
- [ ] Query rewriting for vague or multi-part questions
- [ ] Request tracing (Langfuse free tier)
- [ ] Prompt-injection hardening
