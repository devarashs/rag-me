# rag-me knowledge base

Source documents for "rag me", a retrieval system that answers questions about Arash
Salehkhah for recruiters and hiring managers. The ingest step reads every `.md` file here
except this README, splits it into chunks, embeds them, and stores them in the vector
database.

**This bot is public. Anything written here can be extracted by a visitor — only write
what you would publish on your own website.**

## Files

| File | What it covers |
|---|---|
| 01-profile.md | Who Arash is, headline summary, contact links, career timeline, target roles |
| 02-experience-hypersonic.md | Lead Backend Engineer at Hypersonic Laboratories (Jun 2025 – Sep 2026) |
| 03-experience-earlier.md | Independent network infrastructure work, freelance full-stack, education |
| 04-projects.md | sluice, netforge, DirectPortForward-Go, market-lens, Cubby Server, Cubby Drive Client, EC-h-Louvain, RL Playground, facial attribute classification, e-Vision, Nodeeweb, Helix |
| 05-skills.md | Skills by area, with where each was used, and honest gaps |
| 06-stories.md | Detailed engineering stories (situation, decision, result) |
| 07-ai-engineering.md | How Arash builds AI products and uses AI tools day to day |
| 08-working-with-arash.md | Location, time zone, availability, working arrangements |
| 09-faq.md | Short question-and-answer pairs for common recruiter questions |

Sources: Arash's CV (`arash-salehkhah-cv.pdf`), hand-written notes, and the READMEs and
work reports of his public GitHub repositories.

## Writing conventions

- Third person ("Arash built..."), so retrieved chunks read as answers.
- Every `##` section is self-contained: it names Arash and the company or project again,
  so a chunk still makes sense when retrieved on its own.
- Write facts, not fragments. "Arash worked at Acme as a backend engineer from 2022 to
  2024" retrieves and answers better than "Acme — BE — 22–24".
- Numbers are only included where they come from real measurements.
- Chunk on `##` headings. Each section is sized to fit one chunk.
- Text between a file's `#` title and its first `##` is an editor's note and is not
  ingested. A file with no `##` sections fails ingest rather than being skipped.
- `09-faq.md` is written as `## Question` followed by the answer.
- No placeholders or TODOs in the knowledge files — the bot would retrieve them. Open
  items live in the list below, which is not ingested.

## Open TODOs (only Arash knows these)

Fill each one into the named file, then delete it from this list.

- **07-ai-engineering.md, rag-me section:** the stack (embedding model, vector store,
  chunking strategy, LLM) and any evaluation run, such as test questions with expected
  answers.
- **New file, optional:** testimonials. One or two sentences each from former colleagues or
  clients who agree to be quoted publicly, with their name and role.

## Not in the knowledge base

Salary and personal details are discussed directly, not published here.

## Suggested system prompt for the RAG

> You answer questions about Arash Salehkhah, a senior backend engineer, using only the
> retrieved documents. If the documents don't contain the answer, say so and suggest
> contacting Arash at me@devarash.icu. Never invent employers, dates, numbers or skills.
> Keep answers short and specific.
