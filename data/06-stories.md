# Engineering stories

Detailed stories in situation, decision, result form. These answer questions like "tell me about a hard problem Arash solved".

## Story: FastCDC delta patching for the game launcher (Hypersonic)

**Situation.** Hypersonic's game launcher split game files into fixed-size chunks so updates only downloaded changed chunks. On a real ~30 GB Unreal Engine 5 build, this gave under 10% deduplication, because inserting even one byte shifts every later chunk boundary, so almost every chunk looked new.

**Decision.** Arash measured the problem, wrote it up as an architecture decision record (ADR-0002), and chose content-defined chunking. He read the FastCDC paper and a reference implementation, then implemented FastCDC in TypeScript. He used AI sub-agents for scaffolding and streaming plumbing but personally wrote and reviewed all the boundary-detection logic.

**Verification.** Arash confirmed his chunk boundaries were byte-identical to the reference implementation on the same inputs.

**Result.** Game updates on a 60 GiB title dropped to roughly 5–7 GiB per version. A 70 GiB Unreal editor update came down to about 70 MB, where the comparable Steam update was around 50 GiB.

**Trade-offs.** Arash considered three approaches. Fixed-size chunking was the simplest, but on game builds a single changed timestamp can shift every following byte, so a build with no real changes could still produce many gigabytes of "changed" chunks. Rabin fingerprinting, the classic content-defined chunking method, gives slightly better deduplication than FastCDC, but FastCDC's results are almost the same and it computes boundaries much faster. Arash deliberately traded that small amount of deduplication for faster chunking and faster iteration, because FastCDC's computation cost and speed were reasonable for the launcher's publish and update flow.

## Story: idempotent real-money payments (Hypersonic)

**Situation.** The Hypersonic economy service moved real money through Stripe. Payment providers retry webhooks, and networks deliver duplicates, so a naive handler could charge or credit a user twice.

**Decision.** Arash made the economy service idempotent end to end. Each operation carried an idempotency key enforced with a unique database constraint, the result was stored so a repeat request returned the same outcome, and incoming webhooks were de-duplicated on Stripe's event ID.

**Result.** Retried or duplicate webhooks could never double-charge or double-credit a Hypersonic user.

## Story: preventing broken dependencies in the package registry (Hypersonic)

**Situation.** Creators on Hypersonic's platform could make packages private. If a public package depended on one that became private, anything downstream would break.

**Decision.** Arash wrote a guard using a recursive CTE in PostgreSQL that walks the whole dependency tree and blocks the change if any public package still depends on the one being made private.

**Result.** Creators couldn't accidentally break other people's packages, and the check ran in a single database query.

## Story: weekly backend freezes during playtests (Hypersonic)

**Situation.** At Hypersonic Laboratories, the backend froze every week at the same time: during the studio's wide playtests of the game.

**Investigation.** Arash investigated many possible causes and found that the backend was holding on to database connections instead of releasing them, so under playtest load it ran out of connections and stalled. He then ran a full audit of the backend and found N+1 query patterns that had been written long before. They had gone unnoticed while the package registry was small and playtests were light, but surfaced as the registry grew and playtests got larger.

**Fix.** Arash added proper connection-pool management, including recovery of stale connections, and removed the N+1 queries using batch and eager loading.

**Result.** The weekly freezes during wide playtests were fixed.

## Story: ending daily regressions with a testing pipeline (Hypersonic)

**Situation.** When Arash joined Hypersonic Laboratories, the backend had no testing pipeline. Other teams, such as the Unreal Engine team and the art team, were hit by backend regressions every day, and each regression caused more issues downstream.

**Decision.** Arash's first project was an automated testing pipeline that ran before any deploy. He split the platform into dev, staging and production environments and made changes pass their tests in each environment before being promoted to the next. He wrote more than 2,500 tests, from unit to end-to-end, across all of Hypersonic's projects. Building the pipeline was also how Arash came to understand the whole backend.

**Result.** The constant daily regressions that other teams had been dealing with stopped. Once Arash's tests covered more than 70% of the backend, Hypersonic's uptime reached 100% in most weeks according to Better Stack's weekly uptime reports, up from weeks as low as 88% before.
