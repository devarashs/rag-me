# Experience: Lead Backend Engineer, Hypersonic Laboratories

## Role overview (Hypersonic Laboratories)

From June 2025 to September 2026, Arash was Lead Backend Engineer at Hypersonic Laboratories, working remotely as a contractor for the UK-based company. Hypersonic was building Helix, a creator-driven gaming platform with real-money transactions, user-made packages and worlds, and a desktop launcher. Arash led the backend: API design, the economy service behind every transaction, the creator platform, the data model, the reliability tooling that kept the platform up, and the AI features the studio shipped. The main stack was NestJS and TypeScript, PostgreSQL with TypeORM, Redis, OAuth 2.0 and OpenID Connect for authentication, AWS, Cloudflare and Railway, with React/Next.js and Electron on the client side.

## Team and leadership (Hypersonic Laboratories)

At Hypersonic Laboratories, Arash started as one of two backend engineers and later became the only backend engineer, owning the entire backend himself. He also guided the two frontend engineers, helping them with their work, distributing frontend tasks between them and reviewing their code. He held regular 1:1s with them and gave them ongoing feedback on their work and development. Arash worked in sprints at Hypersonic: he took part in sprint planning, broke projects into milestones with estimates, and reported progress to the studio's leadership. Hypersonic started with more than 25 engineers, but after several rounds of layoffs only three remained, Arash among them. Arash was let go in the final round of layoffs, in September 2026.

## Testing pipeline and environments (Hypersonic Laboratories)

When Arash joined Hypersonic Laboratories, the backend had no testing pipeline, and other teams, including the Unreal Engine and art teams, were hit by constant backend regressions. The first thing Arash did, which was also how he learned the backend in depth, was build an automated testing pipeline that ran before every deploy. He separated the platform into dev, staging and production environments, and changes ran their tests in each environment before being promoted to the next. He added more than 2,500 tests, from unit tests to end-to-end tests, across all of Hypersonic's projects. This ended the constant daily regressions the other teams had been dealing with. Once Arash's tests covered more than 70% of the backend, Hypersonic's uptime reached 100% in most weeks according to Better Stack's weekly uptime reports, up from weeks as low as 88% before.

## Economy service and real-money payments (Hypersonic)

At Hypersonic Laboratories, Arash built the platform's NestJS REST API and a separate economy microservice, also NestJS and reached over RPC, that handled real money through Stripe. The economy service was idempotent end to end: an idempotency key was enforced with a unique database constraint, the result of each operation was stored, and incoming Stripe webhooks were de-duplicated on the provider's event ID. As a result, retried or duplicate webhooks could never charge or credit a user twice.

## Creator marketplace payment flow (Hypersonic)

At Hypersonic Laboratories, Arash designed the payment flow for the creator marketplace. Orders were signed by the server so clients couldn't tamper with prices, funds were held in escrow and released only on confirmed delivery, and creator withdrawals only became available after a maturation period, which protected the platform against refunds and chargebacks.

## Package registry (Hypersonic)

At Hypersonic Laboratories, Arash owned the package registry for user-made game content. It supported private packages, dependency permission checks, dependency validation at publish time, and multipart uploads of 5 GB and larger. He added a guard using a recursive CTE in PostgreSQL that prevented a package version from being made private again while public packages or servers still depended on it, anywhere in the dependency tree. He also optimised the registry's queries.

## Collectible trading (Hypersonic)

At Hypersonic Laboratories, Arash built a collectible trading system: listings, offers that could be accepted or rejected, and owned collectibles usable across servers, worlds and player profiles.

## Game inventory storage on DynamoDB (Hypersonic)

At Hypersonic Laboratories, the game servers for Helix needed a key-value storage system for player game inventory. Arash designed and built it on AWS DynamoDB, with proper partitioning of the data and tracking of each server's storage usage.

## Dreamer: AI world-building product (Hypersonic)

At Hypersonic Laboratories, Arash built Dreamer end to end, an AI product where users create playable 3D game worlds by describing them. Each user got an isolated Cloudflare Sandbox containing the studio's CLI and an MCP server. Language models were routed through OpenRouter, worlds were rendered with three.js, and an item-generation tool called the Meshy API to produce 3D assets. Billing was usage-based and tied to real model cost, so heavy users paid for what they consumed.

## Electron game launcher and delta patching (Hypersonic)

At Hypersonic Laboratories, Arash built the studio's desktop launcher in Electron: auto-updates, switching between environments, a verify-and-repair feature, content-addressed uploads, and an automated publishing CLI for creators. The launcher was code-signed with a certificate obtained through Microsoft Azure. He replaced fixed-size chunking with FastCDC content-defined chunking, which brought a 70 GiB Unreal editor update down to about 70 MB, where the comparable Steam update was around 50 GiB.

## Creator Hub and game-server provisioning (Hypersonic)

At Hypersonic Laboratories, Arash built Creator Hub, a Next.js web app for creators, and its backend. Arash built the automated provisioning system behind Creator Hub's game servers: it created servers through the APIs of an OVH Public Cloud project, and each server was billed through the economy service. OVH hosted most of the game servers; a few also ran on Microsoft Azure. Creator Hub also let creators view logs, restart servers and install packages from the browser. He designed a bitmask-based server permission system with owner-defined groups, and added reviews for packages and worlds.

## Reliability layer (Hypersonic)

At Hypersonic Laboratories, Arash built the platform's reliability layer: a watchdog over more than 10 services with health checks, auto-restart and alerting; a connection-pool manager that recovered stale connections; and self-updating game servers. He also eliminated N+1 query patterns that had been slowing down key endpoints, using batch and eager loading. Arash also fuzz-tested the Hypersonic API, sending malformed and unexpected inputs to its endpoints to find failures before users did.

## Multi-agent harness and real-time voice chat (Hypersonic)

At Hypersonic Laboratories, Arash worked on an unreleased internal harness the studio was building that lets multiple AI agents work together, each with its own set of skills. Arash built several features in the harness. The biggest was real-time voice chat on Alibaba Cloud's Qwen realtime model, a speech-to-speech model that needs no separate speech-to-text or text-to-speech step. The model itself was only called through Alibaba Cloud's API; Arash built everything around it: voice activity detection (VAD), turn management and barge-in (letting the user interrupt the agent mid-reply), noise suppression, and the audio buffering and streaming that keep a live voice conversation responsive.

Arash also added voice to the harness's text agents as a separate service. Users could send a voice message, which was transcribed by a speech-to-text (STT) model and passed to the text model, and could press a read-aloud button to hear any reply, generated by a text-to-speech (TTS) model.

## Agent memory and context management (Hypersonic)

At Hypersonic Laboratories, Arash built the memory and context management for the AI agents in the studio's multi-agent harness, for both the text agents and the real-time voice agent. This covered storing conversation history, summarising long sessions so they fit the model's context window, and managing provider-side prompt caching to reduce latency and model cost.
