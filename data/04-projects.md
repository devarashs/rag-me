# Projects

## sluice (Go, open source)

sluice is Arash's open-source TCP tunnelling tool written in Go. It provides multiplexed reverse tunnels over TLS with certificate pinning, so many logical connections share one secured connection, and it exposes Prometheus metrics for monitoring throughput and connection health. sluice shows Arash's work on networking, concurrency in Go, and observability.

Repository: https://github.com/devarashs/sluice

## netforge (Go, open source)

netforge is Arash's open-source network and security toolkit: a single Go binary with zero third-party dependencies, built entirely on the Go standard library. It has four tool groups: `stress` for authorized load and resilience testing of infrastructure you own (HTTP, TCP, UDP, ICMP and raw SYN), `cert` for generating and inspecting X.509 certificates (self-signed CAs and leaf certificates with RSA, ECDSA or Ed25519 keys), `crypto` for hashing, secure random, AES-256-GCM passphrase encryption and RSA-OAEP, and `net` for read-only diagnostics such as port scanning, latency checks, uptime monitoring and DNS lookups.

Arash built netforge to understand network tooling at a low level rather than through a library. He assembled IPv4 and TCP headers byte by byte with their one's-complement checksums and sent raw SYN segments through a raw socket, implemented PBKDF2 key derivation on top of HMAC himself, and wrote a small command-tree CLI framework. The load-testing commands refuse to run without an explicit authorization flag and never spoof source addresses.

netforge's releases are fully automated: every push to main runs CI (gofmt, go vet, race-enabled tests, build), and a green run computes the next semantic version from Conventional Commit messages and publishes cross-compiled binaries for Linux, macOS and Windows on amd64 and arm64, with checksums.

Repository: https://github.com/devarashs/netforge

## DirectPortForward-Go (Go, open source)

DirectPortForward-Go is Arash's open-source, high-performance TCP port forwarder written in Go. It handles thousands of concurrent connections and has metrics, connection limiting, graceful shutdown and configurable timeouts. An encrypted variant of DirectPortForward-Go adds tunnel-level encryption. Cross-platform binaries are built with GitHub Actions.

Repository: https://github.com/devarashs/DirectPortForward-Go

## market-lens (Python and React, open source)

market-lens is Arash's open-source real-time crypto market microstructure tool. It ingests live order-book data from 9 crypto exchanges using Python asyncio, normalises the different exchange formats into one consistent model, and shows the results in a React front end. Keeping 9 live feeds healthy means handling disconnects, rate limits and format changes from each exchange. market-lens shows Arash's work on real-time data pipelines, unreliable third-party APIs, and Python.

Repository: https://github.com/devarashs/market-lens

## Cubby Server (TypeScript, open source)

Cubby Server is an open-source, self-hosted, privacy-first cloud storage server that Arash built under ZeroNode Labs. It is designed to run on a cheap VPS next to other self-hosted apps and store files in any S3-compatible backend, such as AWS S3, Cloudflare R2 or MinIO. Cubby Server provides a REST API for user authentication (JWT) and file and folder management, uploads and downloads, and ships with Docker Compose plus interactive setup scripts for Windows and Linux that generate secure credentials and initialise the database. It is built with TypeScript, Fastify, PostgreSQL, Prisma and Docker. Cubby Server is in active development; end-to-end encryption is the main goal on its roadmap.

Repository: https://github.com/ZeroNode-Labs/cubby-server

## Cubby Drive Client (React Native and Kotlin, open source)

Cubby Drive Client is an open-source, Android-first mobile file manager for Linux servers over SSH/SFTP, which Arash built under ZeroNode Labs. It lets users browse, upload, download and edit files on a remote server and stream media directly from it. Features include saved connection profiles with password or SSH key authentication, credentials stored in the platform keychain, queue-based transfers with retry, pause/resume and checkpointed downloads, automatic reconnection when a session drops mid-transfer, and a remote text editor.

Cubby Drive Client is built with React Native (bare workflow) and TypeScript, using Zustand for state, on top of a native Android module Arash wrote in Kotlin with the SSHJ library. Media playback works through a local HTTP range-request proxy in native code, and host keys use a trust-on-first-use model.

Repository: https://github.com/ZeroNode-Labs/cubby-drive-client

## EC-h-Louvain: hypergraph community detection (Python, research)

EC-h-Louvain is Arash's open-source research project on community detection in hypergraphs. It is a hybrid pipeline: it embeds the hypergraph into a vector space (with Node2Vec, DeepWalk and hyper2vec), clusters the embeddings with k-means to get initial communities, and uses those clusters to initialise the h-Louvain algorithm instead of starting from a random partition. The pipeline is evaluated against standalone EC-Louvain, h-Louvain and plain Louvain on hypergraph modularity and convergence time, using the Coauth-DBLP co-authorship dataset. The comparison is reproducible from the code in the repository; the final results write-up is not yet published.

Repository: https://github.com/devarashs/EC-h-Louvain

## Reinforcement Learning Playground (Python, open source)

The Reinforcement Learning Playground is Arash's open-source collection of reinforcement learning experiments built with Stable-Baselines3, Gymnasium and PyTorch. Arash trained agents on six environments, choosing the algorithm to fit each one: PPO for CartPole and the grid-based MiniHack Room, DQN for MountainCar, Acrobot and LunarLander, and SAC for the continuous-action MountainCarContinuous. He wrote custom environment wrappers, including reward shaping that makes the sparse-reward MountainCarContinuous task learnable, and wall-penalty and goal-reward wrappers for MiniHack. Training uses separate evaluation environments, automatic best-model checkpointing and TensorBoard monitoring, and each environment has a written work report.

Repository: https://github.com/devarashs/implementation-for-stable-baselines3-gym

## Facial attribute classification (Python, open source)

Arash's facial attribute classification project is a PyTorch deep learning model for predicting the "Male" attribute on the CelebA face dataset. It uses a hybrid architecture: a pretrained ResNet-18 extracts spatial features (transfer learning from ImageNet), a bidirectional LSTM models the relationships between those features as a sequence, and a multi-head attention layer weights the most relevant image regions. The data pipeline uses the Hugging Face `datasets` library with stratified train and validation splits. The training script is deliberately sized for limited hardware such as Google Colab, so the project's value is the architecture and implementation rather than headline accuracy.

Repository: https://github.com/devarashs/facial_attribute_classification

## e-Vision e-commerce platform (Node.js, React, Next.js, MongoDB; 2024, archived)

e-Vision is an e-commerce platform Arash built end to end as ChillHubCore, his personal project, mostly in 2024. The goal was a service that let users create their own online shops, with built-in CMS and CRM. Arash started it in October 2023 with a Node.js server template and built e-Vision itself from February to June 2024, alongside the early months of his independent network infrastructure work. As that work took over he stopped developing e-Vision, and later made it public and archived it. It has four parts, all written by Arash: an Express and MongoDB server, a React admin panel, a Next.js page builder, and a documentation repository.

The e-Vision server covers users, products, orders, promotions, transactions, tickets, blogs and CMS pages, with Zod input validation, JWT authentication, API-key middleware, rate limiting, image uploads to S3 and Cloudinary with sharp processing, a Telegram bot integration, and Jest tests. Payment approval runs inside a MongoDB multi-document transaction, so an order and its product stock are updated together or not at all. A separate Socket.IO server handles real-time features. The server grew out of Arash's own Node.js, Express and MongoDB server template (NEM-Server-Template).

The e-Vision admin panel is a React app (Mantine, Redux Toolkit, React Query, TipTap rich-text editing) for running the store: products, orders, promotions, CMS content, customer tickets and role-based permissions. The e-Vision page builder is a Next.js app with Tailwind CSS, Radix UI and shadcn-style components.

Repositories: https://github.com/ChillHubCore/e-Vision-Server, https://github.com/ChillHubCore/e-Vision-admin-panel, https://github.com/ChillHubCore/e-vision-page-builder

## Nodeeweb Framework (Node.js, React, MongoDB)

Nodeeweb is a no-code/low-code MERN framework for rapid app development, made by the company Idehweb. Arash was part of the Idehweb team that built Nodeeweb, working as a freelancer, and worked on its visual drag-and-drop page builder and its CMS and CRM modules. The Nodeeweb framework powers more than 6 production websites, including Gameboss (https://gameboss.shop), Titipluse (https://titipluse.com) and Gomrok24 (https://gomrok24.com).

Websites: https://nodeeweb.com, https://idehweb.com

## Helix Game Platform (Hypersonic Laboratories)

Helix is the production gaming platform whose backend Arash led at Hypersonic Laboratories. The Helix backend includes a REST API, an economy microservice over RPC with real-money Stripe transactions, the Dreamer AI world-building feature, the package registry's dependency model, the Electron launcher with content-addressed delta updates, and the CI/CD and watchdog systems underneath it. Stack: NestJS, Stripe, microservices, Railway and Cloudflare.

Website: https://helixgame.com
