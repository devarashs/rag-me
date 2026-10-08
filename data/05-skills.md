# Skills

Each skill lists where Arash actually used it, so answers can point to evidence.

## Programming languages Arash uses

- **TypeScript / JavaScript (Node.js):** Arash's primary language for about six years. All Hypersonic backend services, freelance work.
- **Go:** sluice, netforge, DirectPortForward-Go and the independent network infrastructure tooling. netforge is written on the standard library alone, with no third-party dependencies. Arash has also used the Gin web framework in personal experiments and projects, not in production.
- **Python:** working level. market-lens (asyncio, real-time ingestion), and machine learning and research work: EC-h-Louvain, the Reinforcement Learning Playground and facial attribute classification.
- **SQL:** daily use with PostgreSQL, including recursive CTEs and query optimisation.
- **Kotlin:** the native Android SSH/SFTP module in Cubby Drive Client.

## Arash's backend frameworks and patterns

- **NestJS:** Hypersonic REST API and economy microservice.
- **Express.js, Fastify:** Fastify in Cubby Server (REST API for auth, files and folders); Express.js in the e-Vision server and freelance work.
- **REST, RPC, microservices:** Hypersonic platform (REST API plus an economy microservice over RPC).
- **WebSockets:** a separate Socket.IO server for real-time features in the e-Vision e-commerce platform.
- **Idempotency, webhooks, retries:** Stripe economy service at Hypersonic.
- **Permission systems:** bitmask permissions with owner-defined groups in Hypersonic's Creator Hub.
- **Package and dependency resolution:** Hypersonic package registry.
- **Content-defined chunking (FastCDC) and delta patching:** Hypersonic game launcher.

## Arash's databases and storage experience

- **PostgreSQL:** main database at Hypersonic (registry, payments, permissions); Cubby Server.
- **Redis:** caching and coordination at Hypersonic.
- **MySQL, MongoDB:** freelance projects; MongoDB in Nodeeweb and the e-Vision e-commerce platform, including multi-document transactions for payment approval.
- **DynamoDB:** the key-value game inventory store for Helix game servers at Hypersonic, with partitioning and usage tracking.
- **ORMs and ODMs:** TypeORM (used heavily across the Hypersonic backend), Prisma (Cubby Server), Mongoose (e-Vision server).
- **Query optimisation and indexing:** Hypersonic registry queries, N+1 elimination with batch and eager loading.
- **Content-addressed storage:** Hypersonic launcher uploads.

## Arash's payments experience

- **Stripe:** idempotent real-money payments and webhook de-duplication at Hypersonic.
- **Escrow and payout maturation:** Hypersonic creator marketplace.
- **Usage-based billing:** Dreamer (billed on real model cost) and game-server provisioning at Hypersonic.
- **Cryptocurrency payments:** automated billing through Telegram bots in the independent network infrastructure work.

## Arash's cloud and infrastructure experience

- **AWS:** part of the Hypersonic platform stack; S3 storage in Cubby Server and the e-Vision server. DynamoDB for Helix game inventory at Hypersonic; EC2 and Lightsail servers and CloudFront in the multi-provider fleet of the independent network infrastructure work; light use of IAM at Hypersonic.
- **Cloudflare:** part of the Hypersonic platform stack, including Sandbox for Dreamer; R2 support in Cubby Server. Services used: Workers, Pages, Sandbox, R2, D1, KV, CDN.
- **Railway:** Hypersonic deployment.
- **Microsoft Azure:** obtaining the code-signing certificate for Hypersonic's Electron launcher, and hosting a few of Helix's game servers (most ran on OVH Public Cloud).
- **OVH Public Cloud:** automated game-server provisioning through the OVH APIs at Hypersonic.
- **OVH, Hetzner, DigitalOcean:** alongside AWS EC2 and Lightsail, the providers behind the 20+ server fleet in the independent network infrastructure work.
- **Docker, Linux, Nginx, Caddy, Bash:** service packaging and server operations, including a self-managed fleet of 20+ servers.
- **CI/CD and environments:** Hypersonic platform, where Arash built a pre-deploy test pipeline and separate dev, staging and production environments; cross-platform release builds with GitHub Actions for open-source Go tools; fully automated semantic-version releases from Conventional Commits in netforge.
- **Observability:** Prometheus metrics (sluice), watchdog across 10+ services with alerting (Hypersonic), traffic monitoring with anomaly detection (network infrastructure work).

## Arash's authentication and security experience

- **OAuth 2.0 and OpenID Connect:** used heavily in the Hypersonic platform's authentication.
- **JWT authentication:** Cubby Server and the e-Vision server.
- **API keys and rate limiting:** API-key middleware and rate limiting in the e-Vision server.
- **Encryption and secure tunnel design:** sluice (TLS with certificate pinning), encrypted tunnels and secure gateways in the network infrastructure work.
- **Applied cryptography:** netforge (AES-256-GCM authenticated encryption, PBKDF2 key derivation implemented by hand, RSA-OAEP, X.509 certificate authorities and leaf certificates with RSA, ECDSA and Ed25519 keys).
- **SSH and SFTP:** Cubby Drive Client (password and key authentication, trust-on-first-use host keys, credentials in the platform keychain).
- **Low-level networking:** netforge (hand-built IPv4 and TCP headers, raw sockets, port scanning, DNS diagnostics).
- **DDoS mitigation and security hardening:** independent network infrastructure work.

## Arash's AI engineering experience

- **LLM integration and model routing:** OpenRouter in Dreamer; Alibaba Cloud Qwen realtime for voice in Hypersonic's multi-agent harness; Gemini in rag-me.
- **Multi-agent systems:** worked on Hypersonic's unreleased internal harness for multiple AI agents that each have their own skills.
- **MCP servers and tool design:** Dreamer's per-user MCP server.
- **Sandboxed agent execution:** Cloudflare Sandbox per user in Dreamer.
- **Image and 3D generation APIs:** Meshy in Dreamer.
- **Usage-based AI billing:** Dreamer.
- **Real-time voice AI:** real-time voice chat in Hypersonic's multi-agent harness on Alibaba Cloud's Qwen realtime model, with voice activity detection (VAD), turn management and barge-in, noise suppression, and audio buffering and streaming built by Arash.
- **Speech-to-text and text-to-speech:** voice messages to the text agents through an STT model, and read-aloud replies through a TTS model, at Hypersonic.
- **Agent memory and context management:** conversation history, session summarisation and provider-side prompt caching for the text and voice agents at Hypersonic.
- **RAG and LLM evaluation:** rag-me, a framework-free RAG system on pgvector with a 34-case evaluation suite (retrieval hit@k and MRR, relevance-gate checks, LLM-as-judge answer grading).
- **AI-assisted development:** Claude Code daily with custom skill files.

## Arash's machine learning experience

- **Deep learning with PyTorch:** facial attribute classification (pretrained ResNet-18, bidirectional LSTM, multi-head attention, transfer learning on CelebA).
- **Reinforcement learning:** Reinforcement Learning Playground (Stable-Baselines3, Gymnasium; PPO, DQN and SAC; custom reward shaping and environment wrappers; TensorBoard monitoring).
- **Graph and hypergraph learning:** EC-h-Louvain research project (Node2Vec, DeepWalk and hyper2vec embeddings, k-means clustering, Louvain and h-Louvain community detection, modularity evaluation).
- **Data tooling:** Hugging Face `datasets` with stratified splits.

## Arash's front-end and desktop experience

- **React, Next.js:** Creator Hub, market-lens UI, Nodeeweb, the e-Vision admin panel and page builder, freelance work.
- **React Native:** Cubby Drive Client (bare workflow, Zustand, a custom Kotlin native module); the Melk Pro real-estate app, built as a freelancer and published on Android app stores.
- **Tailwind CSS, shadcn/ui, Radix UI:** the e-Vision page builder.
- **Electron:** Hypersonic game launcher.
- **three.js:** Dreamer world rendering.

## Arash's testing experience

- **Jest, Vitest:** Hypersonic services, more than 2,500 tests covering over 70% of the backend.
- **Uptime monitoring:** Better Stack at Hypersonic, where weekly uptime rose from as low as 88% to 100% in most weeks after the test pipeline.
- **Load and stress testing:** netforge's `stress` tools for HTTP, TCP, UDP, ICMP and SYN load against infrastructure you own.
- **Race-enabled tests:** Go tests run with the race detector in netforge's CI.
- **API fuzz testing:** the Hypersonic API.
- **Automated end-to-end testing:** part of the Hypersonic test suite, which ran in each environment before promotion to production.
- **Reference-based verification:** FastCDC chunk boundaries checked byte-for-byte against a reference implementation.

## Gaps Arash is honest about

- Only light Azure use (code signing and a few hosted game servers); no deeper Azure platform work such as landing zones, Entra ID or Front Door. No Kubernetes at scale or Terraform-heavy platform work.
- No dedicated security engineering (penetration testing, red teaming).
- Python is a working skill, not Arash's primary language.
- GraphQL only in small personal projects, never in production. Arash is confident he could pick it up for a production API, since it sits on the same backend fundamentals he uses daily.
- Little Java, .NET, Scala or Elixir.
