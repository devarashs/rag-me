# AI engineering

## Dreamer architecture (Hypersonic)

Dreamer is an AI world-building product Arash built end to end at Hypersonic Laboratories. Users describe a 3D game world and an AI agent builds it.

- **Isolation:** each user gets their own Cloudflare Sandbox, so one user's agent can't touch another user's data or files.
- **Tools:** inside the sandbox, the agent works through a studio CLI and an MCP server Arash built.
- **Models:** language models are routed through OpenRouter, so models can be swapped or mixed without changing the product.
- **Assets:** an item-generation tool calls the Meshy API to turn descriptions into 3D assets; worlds render with three.js.
- **Billing:** usage-based and tied to the real model cost of each user's activity.

## Multi-agent harness and real-time voice (Hypersonic)

At Hypersonic Laboratories, Arash also worked on an unreleased internal harness for running multiple AI agents together, each with its own skills. His biggest contribution to the harness was real-time voice chat on Alibaba Cloud's Qwen models. Arash built the system around the model, including the buffering and streaming needed to keep a live voice conversation responsive.

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

rag-me is a retrieval system Arash built to answer questions about his own experience, as a working example of retrieval-augmented generation (RAG).
