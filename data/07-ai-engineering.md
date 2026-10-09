# AI engineering

## Dreamer architecture (Hypersonic)

Dreamer is an AI world-building product Arash built end to end at Hypersonic Laboratories. Users describe a 3D game world and an AI agent builds it.

- **Isolation:** each user gets their own Cloudflare Sandbox, so one user's agent can't touch another user's data or files.
- **Tools:** inside the sandbox, the agent works through a studio CLI and an MCP server Arash built.
- **Models:** language models are routed through OpenRouter, so models can be swapped or mixed without changing the product.
- **Assets:** an item-generation tool calls the Meshy API to turn descriptions into 3D assets; worlds render with three.js.
- **Billing:** usage-based and tied to the real model cost of each user's activity.

## Multi-agent harness and real-time voice (Hypersonic)

At Hypersonic Laboratories, Arash also worked on an unreleased internal harness for running multiple AI agents together, each with its own skills. His biggest contribution to the harness was real-time voice chat on Alibaba Cloud's Qwen realtime model, a speech-to-speech model. Arash built the whole voice layer around the model: voice activity detection (VAD), turn management and barge-in, noise suppression, and the audio buffering and streaming needed to keep a live voice conversation responsive. For the text agents he built a separate voice service: voice messages transcribed by a speech-to-text (STT) model, and a read-aloud button that speaks replies through a text-to-speech (TTS) model.

## Agent memory, context and prompt caching (Hypersonic)

Arash built memory and context management for the agents in Hypersonic's multi-agent harness, both text and voice: conversation history, summarisation of long sessions to stay within the context window, and provider-side prompt caching to cut latency and cost. These are the practical problems of running agents in production rather than in a demo: what the agent remembers, what it forgets, and what each turn costs.

## How Arash designs agent tools and guardrails

Arash keeps MCP tool contracts narrow and reviews them by hand: each tool does one thing, with strict inputs, so the agent can't take actions nobody intended. Isolation comes from infrastructure (one sandbox per user) rather than from trusting the prompt. Cost is treated as a design constraint from the start, which is why Dreamer billing follows real model spend.

## How Arash uses AI in day-to-day development

Arash uses Claude Code daily, with his own skill files for backend, frontend, DevOps and testing work. He works in two tiers: a stronger model plans the work and writes the prompts, and faster sub-agents carry out the implementation.

Arash reviews by hand anything that touches money, authentication, permissions or user data. In the FastCDC work, for example, sub-agents wrote the scaffolding and streaming plumbing, but Arash wrote and reviewed the boundary-detection logic himself and verified it byte-for-byte against a reference implementation.

## What Arash thinks about building AI products

Arash's views on building AI products:

- Agents need hard boundaries in infrastructure, not just in prompts.
- Tool design matters more than prompt cleverness: narrow tools, clear inputs, reviewed by a person.
- Model cost has to be visible per user, or a product can lose money on its heaviest users.
- AI-written code still needs an owner who understands it, especially around money and security.

## Arash's machine learning background

Beyond integrating LLMs into products, Arash has trained models himself. In his facial attribute classification project he built a PyTorch model combining a pretrained ResNet-18, a bidirectional LSTM and multi-head attention. In his Reinforcement Learning Playground he trained PPO, DQN and SAC agents with Stable-Baselines3 across six Gymnasium environments, including custom reward shaping. His research project EC-h-Louvain uses graph embeddings (Node2Vec, DeepWalk, hyper2vec) and clustering to improve community detection in hypergraphs. This background means Arash understands what the models he builds products on are doing underneath, not just their APIs.

## About this rag-me project

rag-me (Ask Arash, live at https://ask.devarash.icu) is a retrieval-augmented generation (RAG) system Arash built to answer questions about his own experience, with cited sources. He built it without a RAG framework such as LangChain or LlamaIndex, so that every stage is written, tested and understood by hand.

The stack: Python 3.12 with FastAPI on Vercel; Gemini for embeddings (gemini-embedding-2, 768 dimensions) and generation (gemini-3.1-flash-lite); Neon Postgres with pgvector. The knowledge base is split into one chunk per Markdown section, and re-ingesting only re-embeds sections that changed. Retrieval is hybrid: vector search and BM25 keyword search over Postgres full-text run in one query and are merged with reciprocal rank fusion, so a tool named only once in the knowledge base still finds its section. A relevance gate refuses questions whose best match is below 0.62 cosine similarity without calling the language model; Arash set that threshold from measurements, where off-topic questions scored at most 0.59 and real questions at least 0.65. Answers stream over Server-Sent Events with [n] citations. Visitors are rate limited in Postgres, and their IPs are stored only as HMACs.

The code is public at https://github.com/devarashs/rag-me.

## How rag-me is evaluated

Arash measures rag-me with an evaluation suite of test questions, including questions it must decline, off-topic questions and prompt-injection attempts. Each question runs through the real pipeline and is scored separately on retrieval, the relevance gate and answer quality, which a stronger model (gemini-3.8-flash) grades. The retrieval baseline is hit@1 0.84, hit@5 0.92 and mean reciprocal rank 0.88. In the first full graded run (October 2026), every test case but one passed: all generated answers were fully supported by their sources, 96% cited them, every question about personal details the knowledge base does not cover was declined, and no prompt-injection attempt was followed. The one failure was a retrieval miss on a rare term mentioned only once in the knowledge base. Arash then added hybrid keyword and vector search, which on the same questions raised hit@5 from 0.92 to 0.96 and fixed every question about a tool named only once. Injection attempts can score high enough to pass the relevance gate, so the defence against them is the prompt structure: the visitor's question is escaped into its own block and treated as untrusted. Because the judge model's free tier allows only about 20 requests a day, Arash made it grade several answers per request, after checking that batched grades matched one-at-a-time grades.
