# ECA Local-First Roadmap

**Status:** Canonical planning and tracking document, August 2026

`architecture.md` records what is implemented. This document is the only active plan for future work. The viable work from the original brain and memory plans has been consolidated here.

## Scope

This is a single-operator system by design. Multi-tenancy, per-user isolation, and rate limiting are not goals, and their absence should not be tracked as debt. If external access is ever added, it will be bring-your-own-key with explicit model selection, which is the reason provider credentials and model names are configuration rather than constants. Provider neutrality is a hard requirement; Gemini being the currently configured cloud provider is an operational fact, not an architectural commitment.

## Tracker Conventions

- **Implemented:** Code exists and is wired into a runtime path.
- **Validated:** Implemented behavior has acceptance tests or measured evidence.
- **Planned:** Approved work that has not yet been implemented.
- **Deferred:** Deliberately outside the current path; revisit only after its prerequisite evidence exists.

Do not mark a capability validated merely because its service class exists. Learning, retrieval, and routing claims require a fixture suite and measured baseline.

## Current Progress

**The system ran end to end for the first time on August 1, 2026.** Before this date nothing had ever executed: `.env` held placeholder credentials and no `chroma_db` existed. Every prior status claim was therefore about code that had never run.

**Verified by running it:**

- A full `/chat` cognitive cycle completes. Fully local: 103s. Local agents with Gemini synthesis: 44.8s first turn, 66.8s with memory context.
- Memory closes the loop. A second turn correctly recalled a name and detail stored by the first, retrieved from ChromaDB with local embeddings.
- Local-only operation is real: with all roles on `ollama` the cycle makes no cloud call.
- Warm local inference is ~1.1s per agent call; the 17.9s first call is model load into VRAM.
- `format: json` gives reliable structured output from `gemma4:e4b` (4/4 valid on a smoke test), so agents no longer depend on parsing prose.
- Collections are stamped with their embedding identity on creation: `cognitive_cycles` and `conversation_summaries` carry `ollama/embeddinggemma:latest@768d`.

**Bugs found only by running it, now fixed:**

- `TokenCounter` called the Gemini API for *every* token count, with 3 retries and exponential backoff. A cloud round-trip sat in the hot path of every memory operation, and it silently broke the local-only claim. Now a local estimator.
- Chroma's `DefaultEmbeddingFunction()` downloaded an 80 MB ONNX model at startup for a model never used, since vectors are always supplied explicitly. It blocked first boot on a network download.
- `generate_error_analysis` is synchronous on both `ConflictMonitor` and `MetaCognitiveMonitor` but was awaited, raising on every cycle.
- `gemma4:e4b` is a reasoning model. It consumed the whole output budget on thinking tokens and returned an empty response with `done_reason=length`, silently failing every summary update. Fixed with `OLLAMA_THINKING=false`.
- Ollama's default context window truncated long prompts; `OLLAMA_NUM_CTX` is now always sent.
- `EMBEDDING_MODEL_NAME` was `models/embedding-001`, which the API does not offer. Real identifiers were confirmed with `python -m src.tools.list_models`.

All four recurring error classes are now at zero across a two-turn conversation.

**Provider work:**

- Generation, embedding, moderation, and synthesis are independently selectable. `SYNTHESIS_PROVIDER` allows local agents with a cloud synthesiser without making the cloud a dependency of the cycle.
- `GEMINI_API_KEY` is optional; a fully local configuration starts without it.
- `LOCAL_ONLY_MODE=true` fails startup if any role is remote.
- `CompositeProvider.is_local` accounts for the safety provider, so cloud moderation cannot hide in an otherwise-local setup.
- Embedding identity is enforced by provider and model, never dimension, because Gemini `embedding-001` and `embeddinggemma` are both 768-dimensional.
- Tooling: `src.tools.list_models`, `src.tools.reembed`, `src.tools.eval_retrieval`.
- The canonical retrieval fixture now contains 25 synthetic records and 50 reviewed queries with category, temporal, and expected-fact labels. `--seeded` creates an ephemeral identity-stamped collection, batches record and query embeddings, and never reads or changes personal memory. Two local runs with `embeddinggemma:latest` measured recall, MRR, and NDCG of `1.000` at k=1 and k=5; the collection was absent from the persistent database afterward. The earlier 12-query personal-database result (`0.958`/`0.958`/`0.952` at k=5) remains a useful smoke result only.
- Both primary collections already carry `ollama/embeddinggemma:latest@768d`. Migration dry-runs correctly refuse them because rebuilding into the same vector identity would be meaningless; no legacy Gemini primary collection is present for a like-for-like comparison.
- The first research-governance slice is implemented. Discovery and the meta-cognitive SEARCH_FIRST path use a deterministic local `EscalationPolicy` and `ResearchService`; LLM-suggested queries cannot self-authorize, normal/disabled/local-only paths never invoke a provider, and every decision has structured audit metadata. Context is hard-limited to `question_only`. The live adapter now exists but remains disabled by default.
- The cognitive research-drive slice is implemented in shadow mode. Waking metacognition, reflection, and dream-like consolidation feed a bounded evidence accumulator; high-value unresolved questions enter a durable, de-duplicated local SQLite queue. Atomic claiming supports a future waking reviewer, while policy approval alone cannot cross the independent cognitive provider gate.
- The grounded research round trip is implemented. Waking review can resolve, defer, require user approval, or invoke Gemini Google Search grounding after both gates authorize it. Strict packet validation and citation-aware Cognitive Brain synthesis fail closed. A guarded live smoke returned 2 claims/2 sources after 2 searches in `5567.7ms`; runtime defaults remain disabled and shadowed.
- The waking review and calibration API is implemented. Authenticated routes list and inspect inquiries, approve with fresh revalidation, dismiss, retry failed research, inspect the immutable ledger, label real shadow assessments, submit feedback only against persisted verified packet sources, and retrieve calibration/source-quality summaries. The SQLite ledger rejects updates/deletes and chains event hashes. This creates the measurement surface; it is not yet longitudinal calibration evidence.
- The React operator console is implemented. It exposes the real inquiry queue, packet/source inspection, approve/dismiss/retry decisions with reasons, editable source-quality outcomes, independent calibration labels and strata, the decision ledger, and system telemetry in a responsive full-screen interface.
- Research activation is controllable from that interface. Provider access, active cognitive control, and automatic non-explicit research are separate interlocked stages. Automatic activation requires the exact typed confirmation, every transition is appended to the ledger and restored after restart, and emergency stop returns the system to provider-disabled, shadow, explicit-approval-required posture. A configured Gemini key uses the preselected research model, while provider access remains disabled by default.
- Frontend security/tooling is repaired. Create React App and its vulnerable webpack development graph were removed in favor of Vite `8.2.0`; direct Axios/PostCSS/UUID dependencies were updated, TypeScript checking is part of every production build, development and preview bind only to localhost, and a server-side same-origin proxy injects `API_KEY` without embedding it in the normal browser bundle. `npm audit` now reports `0` vulnerabilities, down from `55` (`29` high and `2` critical).
- The salience foundation is implemented and remains disabled/shadowed by default. It records a complete explainable alternative ranking after waking retrieval and for sleep replay, preserves baseline order/selection, and exposes only bounded Working Memory hints when explicitly made active. The previously unwired emotional-salience encoder now persists affective tags on completed cycles. No pruning path exists.
- Reliable sleep/consolidation is implemented and disabled by default. A single lifecycle owner admits bounded work only after idle, cooldown, candidate, and local-provider gates; waking activity and shutdown cancel inference. Episodic and semantic outputs persist with stable IDs and full source/provider/embedding provenance, Cognitive Brain retrieves semantic knowledge, completed stages resume safely, and an append-only hash-chained ledger records every run and job outcome.
- The event-driven observability plane is implemented. Cognitive, memory, research, salience, sleep, and autonomous-work events use a versioned process stream with monotonic cursors, filtered replay, heartbeats, explicit gaps, and per-viewer non-blocking queues. Research/governor events appear only after authoritative ledger commits and carry references rather than sensitive bodies. The System UI resumes transient disconnects, detects process-stream resets, and displays domain health plus ordered activity.
- The local-first visual sensory path is implemented. Ollama capability declarations gate a dedicated local visual role; strict JPEG/PNG byte, MIME, dimension, and pixel bounds run before provider contact. Raw images are replaced after one local observation call by typed, provenance-marked, confidence-bounded `VisualEvidence`. General agents, memory, and telemetry never receive pixels, and OCR/image text remains explicitly untrusted data. `gemma4:e4b` declared `vision` and completed a live synthetic-image smoke.
- The equivalent local-first auditory path is implemented. Ollama's declared `audio` capability gates a dedicated local role; the browser emits canonical 16 kHz mono 16-bit PCM WAV, and backend base64/RIFF/MIME/byte/duration/format plus deterministic signal-quality checks run before inference. Near-silence short-circuits locally. Raw samples are replaced by typed `AudioEvidence`; transcripts remain untrusted data and never become instruction text. The current local model completed a conservative synthetic-tone negative-speech smoke.

**Test baseline:** `218 passed, 3 skipped` on August 2, 2026. The backend suite includes visual and auditory validation/capability/locality/instruction-boundary tests, deterministic silence/quality handling, raw-media removal, typed-evidence orchestration, authenticated telemetry WebSocket rejection/acceptance, typed domain mapping, cursor replay/gaps, post-commit ledger projection, governed-work controls, governed-sleep lifecycle/recovery, provenance, and a real ephemeral-Chroma semantic round trip. The frontend passes TypeScript checking and a Vite `8.2.0` production build with Node `v24.18.1`; `npm audit` reports zero findings. Live local `gemma4:e4b` visual and synthetic-tone auditory smokes validated capability, transport, JSON parsing, quality capping, and conservative non-speech behavior, but not real-world perceptual accuracy. The skips are three root-level async tests not handled by an async test plugin. The 20-case synthetic research-drive fixture measures action accuracy `1.000`, escalation precision `1.000`, and escalation recall `1.000`; representative real-cycle calibration observations have not yet accumulated.

**Latency is now measured, not guessed.** `StageTimer` wraps every orchestrator stage and emits a `CYCLE_TIMING` line plus `metadata["stage_timings_ms"]`; `OLLAMA_CALL` and `OLLAMA_EMBED` lines give per-request server, queue, prompt-eval, and eval time. Three cycles on August 1, 2026 overturned the previous assumption:

- Orchestration costs **under 200ms per cycle**. Routing, both conflict checks, contextual encoding, theory-of-mind validation, and RL updates are collectively free. The "~40s of orchestration overhead" theory was wrong.
- The cost is **local token generation at ~26 tokens/second**. Prompt evaluation is 0.4-0.8s and model load a constant ~0.8s per call; everything else is tokens coming out.
- The earlier "warm agent call ~1.1s" figure was a trivial smoke-test prompt. Real agent calls generate 80-600 tokens and cost 4-25s each.
- In the August 1 baseline, three stages owned the whole cycle: `stage1_agents`/`stage2_agents` (20-57s, serialised by `max_interactive=1`), `meta_cognitive` (20-25s for a single unbounded call producing 483-605 tokens), and `memory_upsert` (16-22s, including a ~12s summarisation call). The August 2 slice bounded the first and deferred the second.
- Gemini synthesis is 3.0-3.2s and consistently the cheapest stage in the cycle.
- Per-agent `AGENT_METRIC` durations were previously always ~0ms because the timer started after `asyncio.gather` had already returned. Now measured correctly.

**First latency-reduction slice completed on August 2, 2026:**

1. Meta-cognitive uncertainty output is capped at 64 tokens and hard-truncated to 40 words by default.
2. Per-turn summary generation and STM-flush summarisation are submitted to the durable autonomous-work governor off the response path, with a per-user lock around summary writes.
3. The synchronous fallback remains for isolated/test construction. The production queue is in-process and non-durable; shutdown cancels unfinished jobs.

Two matched post-change cycles are now preserved. Meta-cognition measured `0.53-0.54s` versus the prior `20-25s` slow path; `memory_upsert` measured `3.92-4.24s` versus `16-22s`, while its deferred summaries completed `4.09-5.05s` after the response path. Total cycles were `65.74s` and `49.12s`; Stage 1 and Stage 2 generation still consumed `85-89%`, so agent output volume remains the next latency lever. Do not raise `max_interactive` without a separate VRAM/contention measurement.

**Memory defect found and repaired during the live measurement:** The collections use Chroma's default squared-L2 metric, but application code converted distances below `1.0` as if they were cosine distance and distances above `1.0` with a different reciprocal formula. A closer result at `0.865` therefore scored `0.135`, while a worse result at `1.088` scored `0.479`. The first matched run gave Cognitive Brain zero memories and an incorrect clarification. Metric-aware monotonic conversion now aligns normalized L2 scores with STM cosine similarity; the second run supplied three LTM memories and correctly recalled both Tom and Leeds.

**Current evidence slice:** Complete. Policy approval cannot bypass the cognitive gate; shadow, disabled, private, unapproved offline, local-only, malformed, uncited, timed-out, or provider-failed paths contribute no research content. Approved waking research produces bounded question-only packets. Cognitive Brain receives only verified claims and deterministic source links. The real adapter completed one guarded connectivity/grounding smoke. Automatic non-explicit authorization remains off by default and is available only as a deliberate, confirmed operator action; calibration does not self-activate it.

## Direction

Run the normal cognitive cycle locally through Ollama, using a capable local multimodal model for chat, image understanding, audio understanding where supported, agent analysis, and final synthesis. Reserve cloud inference for explicit research escalation: the Discovery/Curiosity path may call a configurable cloud model when local knowledge is insufficient or current external information is required.

This is a good fit for ECA. It makes the persistent memory and cognitive orchestration inexpensive to run, keeps routine conversations private, and makes every cloud call intentional and inspectable.

The implementation supports the local routine path, capability-gated local visual and auditory paths, and a guarded grounded-research path. Roles are independently selectable, Ollama JSON mode is requested for structured calls, and shared JSON extraction/repair is used by agents and Cognitive Brain. Research authorization, question-only release, grounded Gemini search, packet validation, citation-aware synthesis, authenticated review APIs, the responsive operator console, ledger-restored runtime controls, and a durable calibration ledger are implemented. Representative real-cycle sensory labels and calibration remain operator validation work.

## Target Architecture

```
User input
    |
    v
Local multimodal provider (Ollama)
    |- routine chat and final synthesis
    |- Stage 1 and Stage 2 agents
    |- local image/audio analysis when the selected model supports it
    |- local safety/classification where feasible
    |
    v
Memory and orchestration (local)
    |- ChromaDB and local embeddings
    |- working memory, conflict checks, learning, metrics
    |
    v
Research escalation policy
    |- no escalation: use local answer
    |- web/current-information need: cloud research provider
    |- return sources and compact findings to local synthesis
```

Cloud research is a capability, not the default model. The local final synthesizer should receive only a bounded research packet: question, answer, source URLs, timestamps, confidence, and any uncertainty.

## Principles

- **Local by default:** Ordinary conversation, memory access, agent reasoning, and final response generation do not require a cloud API.
- **Explicit escalation:** A policy makes cloud calls based on need for current information, task difficulty, local failure, or user request; it records why the call happened.
- **Capability-driven routing:** Audio, image, embeddings, tool use, JSON mode, and moderation are model/provider capabilities, not assumptions tied to a model name.
- **Provider-neutral configuration:** Ollama, Gemini, and later providers are selected by configuration rather than direct imports in agents.
- **Scheduled local inference:** A local model is a constrained shared resource. Interactive work, background work, and model residency require explicit admission control rather than nominal agent parallelism.
- **Bounded context:** Continue limiting raw transcript and agent output sizes before sending anything to a provider.
- **Preserve local state:** ChromaDB, logs, metrics, and user memory remain local. Cloud requests contain the minimum necessary context.

## Phase 0: Establish a Runnable Baseline

**Goal:** Make current behavior observable before changing model infrastructure.

1. Replace placeholder values in `.env` with real local credentials.
2. Repair test collection:
   - Fix the syntax error in `tests/test_llm_integration_service.py`.
   - Remove or rewrite the stale MongoDB test setup in `tests/test_memory_service.py` for ChromaDB.
3. Add a minimal startup smoke test using fake/local providers so tests do not require real API keys.
4. Add one end-to-end mocked cognitive-cycle test covering Stage 1, Stage 2, final synthesis, and memory persistence.
5. Record baseline metrics for local hardware: cold start, latency per agent, peak RAM/VRAM, requests per minute, and failure rate.

**Exit criteria:** `pytest` collects and passes with no real cloud credentials; a documented baseline exists.

## Phase 1: Introduce an LLM Provider Boundary

**Goal:** Decouple ECA services from Gemini.

1. Define provider protocols/interfaces:
   - `ChatProvider.generate(...)`
   - `MultimodalProvider.generate(..., image, audio)`
   - `EmbeddingProvider.embed(text)`
   - `SafetyProvider.assess(content)`
   - `ResearchProvider.research(query, context)`
2. Replace direct `LLMIntegrationService` usage with a provider facade injected through application startup.
3. Keep `GeminiProvider` as the first adapter so existing cloud behavior stays available during migration.
4. Add provider capabilities such as `supports_images`, `supports_audio`, `supports_embeddings`, `supports_structured_output`, `supports_tools`, and `is_local`.
5. Use an explicit structured-output parser/repair boundary. Local models may not consistently return the JSON currently expected by agents.
6. Keep retry, concurrency, timeout, context-size, and audit logging at the provider facade rather than inside a specific provider.
7. Introduce shared `ProviderRequest` and `ProviderResult` envelopes so every provider path carries purpose, required capabilities, structured-output schema, privacy classification, context budget, timeout, provider/model, usage, latency, finish reason, parse/repair status, and capability evidence.

**Exit criteria:** The app can run unchanged with a configured Gemini adapter, while no agent imports a Gemini SDK or hardcodes a Gemini model name.

**Status:** Met for the current text path. Provider interfaces, adapters, capabilities, request/result envelopes, startup injection, scheduling, Ollama JSON mode, and shared JSON extraction/repair are implemented. Parse/repair status is not yet propagated consistently through `ProviderResult`, so that observability detail remains follow-up work rather than a blocker to local agent routing.

## Phase 2: Add Ollama as the Local Default

**Goal:** Run text reasoning locally before enabling multimodal paths.

1. Implemented: `OllamaProvider` uses the Ollama HTTP API for bounded local text generation; its scheduler is initialized at application startup.
2. Implemented: local defaults are selectable per role.
   - `LLM_PROVIDER=ollama`
   - `OLLAMA_BASE_URL=http://localhost:11434`
   - `OLLAMA_CHAT_MODEL=<verified-local-model>`
   - `OLLAMA_NUM_CTX` is sent on every call and each request carries an output limit; workload-specific budget tuning remains ongoing
3. Implemented and verified: routine agent calls can use `gemma4:e4b`; embeddings use the separately selected `embeddinggemma:latest` adapter. Ollama JSON mode plus shared repair handles structured agent output.
4. Implemented: health checks for Ollama availability and installed model presence.
5. Implemented for normal requests: provider failures surface rather than silently falling back to cloud. A separate explicit research-escalation policy remains Phase 5 work.
6. Partially implemented: `ModelExecutionScheduler` bounds interactive and background concurrency and exposes active counts. Per-cycle call/token budgets, cancellation, compact-cycle degradation, residency management, and richer queue/VRAM telemetry remain planned.
7. Measured: two post-change full cycles verify the first output/critical-path reduction. Agent generation owns `85-89%` of current latency. Concurrency remains unchanged pending a separate VRAM/contention experiment.

**Exit criteria:** A text-only `/chat` cycle completes locally with no Gemini API calls; local inference remains bounded under concurrent work; and the selected provider/model plus scheduling decision are visible in cycle metadata and logs.

## Phase 3: Separate Embeddings and Memory Retrieval

**Goal:** Remove the remaining cloud dependency from persistent memory.

1. Implemented: `OllamaEmbeddingProvider` is a dedicated local embedding adapter, selectable via `EMBEDDING_PROVIDER=ollama`.
2. Implemented: embedding provider/model/dimension/identity-version are stored in ChromaDB collection metadata, and a provider+model comparison fails closed on mismatch. Dimension is not used as the compatibility test because Gemini `embedding-001` and `embeddinggemma` share 768 dimensions.
3. Implemented: `src.tools.reembed` rebuilds into a new identity-stamped collection, leaves the source intact, skips ids already present when resuming, supports batching, and refuses a same-identity migration.
4. Partially validated: a reviewed 12-query local smoke fixture records recall@5 `0.958`, MRR `0.958`, and NDCG@5 `0.952`. There is no retained Gemini collection to compare, and uploading the private local corpus to create one requires an explicit privacy decision.
5. Implemented and runtime-verified: `MemoryService`, summaries, and retrieval use the selected local embedding provider. The initial collections were created directly in the local vector space rather than migrated from Gemini.
6. Implemented: `OllamaEmbeddingProvider.embed_batch()` sends document lists to `/api/embed`; the migration tool uses it per batch.
7. Extend identity stamping to the remaining embedding-backed collections (self-model, autobiographical, emotional profiles) once the migration path is proven on the two primary collections.

**Exit criteria:** Memory ingestion and retrieval work with no cloud embedding call, and legacy vector collections remain recoverable.

## Phase 4: Enable Multimodal Capability Safely

**Goal:** Use the selected Ollama model for image/audio only after verifying actual support.

1. Implemented for vision: verify the exact Ollama model tag through `/api/show`, never by name. `gemma4:e4b` currently reports `completion, vision, audio, tools, thinking`; image transport completed a live synthetic smoke. Audio declaration still needs a real bounded fixture test before activation.
2. Implemented for vision: startup capability probing and `/health/deep` expose the visual role and limits.
3. Implemented: `VisualInputProcessor` is wired into startup/orchestration only through a declared local image capability.
4. Introduce `TranscriptionProvider` as a separate capability and pipeline stage:
   - If a local model accepts audio, adapt it behind `TranscriptionProvider`.
   - Otherwise use a local speech-to-text engine, then pass its transcript to the normal local chat provider.
5. Implemented for images: preserve verified MIME, bytes, dimensions, hash, provider/model/locality, quality, timestamp, and provenance; enforce limits before provider contact. Audio duration/bounds/provenance remain open.
6. Implemented for images: deterministic tests cover success, unverified/remote capability, MIME mismatch, oversize, instruction boundaries, raw-media removal, and orchestration. Equivalent audio fixtures remain open.

**Exit criteria:** Image and audio paths are either locally functional with tested models or cleanly marked unavailable; no hidden Gemini fallback exists.

## Phase 5: Replace Web Browsing With Cloud Research Escalation

**Goal:** Make Discovery the only cloud-capable node by policy.

**Current status:** The provider-neutral safety boundary, cognitive research drive, persistent inquiry queue, waking revalidation, grounded Gemini provider, strict packet validation, citation-aware local synthesis, authenticated review APIs, responsive operator console, ledger-restored runtime controls, and append-only calibration ledger are implemented. Legacy `WebBrowsingService` is unreachable. The complete path is tested with fakes and one guarded live grounding smoke. Runtime defaults remain disabled/shadowed/approval-required; representative real-cycle labeling and broader operational/source-quality evaluation remain pending.

1. Complete: replace `WebBrowsingService` with a `ResearchService` protocol, disabled provider, and `GeminiGroundedResearchProvider` using current Google Search grounding.
   - Optional future direct-search adapter when source-level control is needed.
2. Cognitive controller implemented in shadow; real-cycle observation, manual labeling, and summary metrics are now durable, while collection and threshold calibration remain pending. The deterministic `EscalationPolicy` supplies interpretable signals to Discovery and metacognition:
   - current or time-sensitive information
   - explicit user request to research/search the web
   - local confidence below a configured threshold
   - named fact that is absent from local memory
   `CognitiveResearchDrive` accumulates these with cloud cost/privacy inhibition, hysteresis, and refractory cooldown. It emits the action ladder: routine local -> deeper local thought -> clarification/uncertainty -> queue -> cloud authorization. Shadow mode records the recommendation while forcing the effective action to local routine.
3. Boundary complete: require a local policy decision before provider contact and record decision/request IDs, reasons, disposition, provider, model, timestamp, estimated query size, context size/policy, provider latency, source/claim counts, and cost when supplied. Review, decision, packet, and feedback events are persisted in the calibration ledger; broader provider operational metrics still belong to Phase 6.
4. First privacy boundary complete: only the question is permitted; no transcript, summary, agent output, or raw memory crosses the provider contract. A future compact/redacted-summary mode requires its own release policy and tests.
5. Complete: structured packets contain answer text, annotated claim spans, source IDs/URLs, search queries, confidence, caveats, grounding status, context size, latency, and optional cost. Cognitive Brain accepts verified packets only, treats them as untrusted evidence, requests inline `[R#]` citations, and deterministically appends missing source links.
6. Complete and disabled by default: provider/model, timeout, confidence threshold, query bounds, drive shadow mode, and inquiry approval policy are explicit. A configured Gemini key and preselected research model make the capability available to the operator without activating it. Authenticated runtime controls separately enable the provider, activate the controller, and—after typed confirmation—allow automatic non-explicit research; local-only mode refuses provider activation.

**Exit criteria:** A normal local chat does not call the cloud; a research-required query produces an auditable cloud research packet and a locally synthesized response with sources.

## Phase 6: Local Safety, Observability, and Operations

**Goal:** Operate the hybrid architecture predictably.

1. Separate content safety from the chat provider. Start with simple local policy checks and make any cloud moderation an explicit opt-in escalation.
2. Add provider-level metrics: selected provider/model, local/cloud ratio, latency, failure class, context size, escalation reason, and estimated cloud usage.
3. Complete: replace the snapshot-only WebSocket with authenticated typed subscriptions, bounded replay, explicit backpressure/replay gaps, reconnect semantics, and a live six-domain operator surface.
4. Add a local-only mode that rejects all cloud escalation, plus a research-enabled mode that requires an explicit configuration flag.
5. Add CI checks for tests, type checking, frontend build, `.env.example` completeness, and provider contract tests.

**Exit criteria:** Local-only mode is enforceable and tested; hybrid activity is observable; background services have explicit lifecycle management.

## Phase 7: Memory Reliability and Evaluation

**Goal:** Make the existing STM, summary, LTM, and consolidation design reliable with local models and measurable retrieval quality.

**Current status:** Core memory and governed sleep paths are implemented with deterministic reliability coverage and a reproducible direct-ranking baseline. The sleep scheduler remains disabled by default pending real-cycle validation and tuning. Application-level expected-fact evaluation, STM cleanup/snapshot policy, summary fault-injection/recovery work, and measured sleep-cycle retrieval benefit remain incomplete.

1. Make all memory budgets model-neutral. Derive STM/context reserves from active provider capabilities instead of Gemini-specific limits.
2. Complete for direct vector ranking: the version 2 fixture seeds 25 synthetic records and evaluates 50 reviewed queries in an ephemeral collection. It includes time-sensitive cases and expected-fact labels without depending on mutable personal memory.
3. Local measurement complete: the canonical seeded corpus measured recall, MRR, and NDCG of `1.000` at k=1 and k=5 with `embeddinggemma:latest`; the earlier personal-database smoke run measured `0.958`, `0.958`, and `0.952` at k=5. The evaluator reports partial recall and ranking failures. Define an acceptable delta before any cross-provider comparison; the synthetic corpus is safe to compare, but no retained Gemini collection exists.
4. Partially complete: episodic and semantic memories retain source cycle IDs, consolidation job, generation provider/model, timestamps, and embedding identity/version. Summary provenance still needs the same contract.
5. Add per-user locking and fault-injection tests for summary-before-flush, failed upserts, and interrupted recovery.
6. Add a deliberate STM cleanup and snapshot policy: retention bounds, recovery age limits, periodic snapshots, and no silent data deletion.
7. Complete: the scheduler is behind `SLEEP_CYCLE_ENABLED`, has one lifecycle owner, idle/cooldown/local-provider gates, task de-duplication, waking cancellation, and cancellation-safe shutdown coverage. Disabled mode creates no task.
8. Partially complete: sleep completion/failure/cancellation metrics exist. Retrieval latency, hit source (STM/summary/LTM/semantic), flush reason, token counts before/after, and summary failures remain.
9. Complete: `MemoryService.get_user_cycles()` exists and every consolidation lookup is scoped by both `user_id` and `cycle_id`; missing sources fail the job instead of producing a false success.
10. Complete: episodic and semantic outputs use explicit active-provider embeddings in identity-stamped v2 Chroma collections, retain provenance, survive a real ephemeral-Chroma round trip, and semantic knowledge is supplied to Cognitive Brain synthesis.
11. Test summary identity extraction on diverse inputs and treat regex matching as a fallback heuristic, not a durable identity system.
12. Implemented: a persistent offline `InquiryCandidate` queue accepts waking, reflection, and dream/consolidation output. Candidates retain question, hypothesis, source-cycle/pattern provenance, full drive assessment, priority, expected information gain, status, and expiry. Active duplicates merge, failed duplicates re-queue on new evidence, and waking review can resolve, defer, await approval, research, or record retryable failure. Consolidation has no provider route. Authenticated list/inspect/approve/dismiss/retry APIs, immutable history, and the React operator view are complete. Explicit approval remains the default; automatic scheduling can be deliberately enabled only through the interlocked runtime control plane.

**Exit criteria:** Local memory retrieval has a documented quality baseline; summary/flush/recovery paths are covered by integration tests; consolidation runs only when explicitly enabled and leaves an audit record.

## Phase 8: Validate Learning and Metacognition

**Goal:** Turn the existing RL, procedural-learning, Theory-of-Mind, and meta-cognitive services from plausible mechanisms into measured capabilities.

**Current status:** Implemented with provisional signals; not validated as learning systems. The existing single-user setup is appropriate for initial experiments, but conclusions require repeatable evaluation.

1. Define a small evaluation corpus for factual uncertainty, ambiguous requests, emotional support, technical explanation, and research-required queries.
2. Measure meta-cognitive calibration: answer accuracy, confidence calibration, correct abstentions, unnecessary abstentions, and research-escalation precision/recall.
3. Define outcome labels before changing rewards. Start with explicit thumbs up/down or review fixtures; treat sentiment and conversation length as weak proxy signals.
4. Add deterministic replay tests for RL updates, Q-value persistence, strategy selection, and habit-formation thresholds.
5. Add procedural-learning tests that attribute a failure to a documented skill category and verify a sequence recommendation changes only with sufficient evidence.
6. Verify Theory-of-Mind prediction validation is written and measured per cycle; track prediction coverage and accuracy separately from response quality.
7. Keep global learning disabled until user isolation, consent, and evaluation methodology exist. Local single-user learning remains the default experiment.

**Exit criteria:** Each learning claim has a metric, a baseline, and a repeatable evaluation; no dashboard or architecture claim implies demonstrated improvement without that evidence.

## Phase 9: Attention and Salience

**Goal:** Complete the focus-control loop without coupling it to a particular model provider.

### 9.1 Attention Controller General Availability

**Current status:** Implemented, default disabled/shadow mode.

1. Build synthetic fixtures for topic shifts, emotional shifts, urgency spikes, and stable conversations.
2. Define drift labels and evaluate precision, recall, and latency overhead. Tune thresholds from results rather than adopting an unverified percentage target.
3. Run a bounded shadow period, compare proposed routing with baseline routing, and inspect false suppressions before enabling control.
4. Record whether routing changes improve latency, response quality, or downstream reward. Do not feed attention outcomes into learning services until that attribution is validated.

**Exit criteria:** Active routing is opt-in, regression-tested, and supported by measured improvement over shadow-mode baseline.

### 9.2 Salience Network

**Current status:** Advisory foundation implemented; default disabled/shadowed, with no pruning implementation.

1. Complete: advisory-only ranking after MemoryAgent retrieval combines query relevance, recency, emotional salience, novelty, goal alignment, and must-keep flags.
2. Complete: the versioned assessment returns top-$k$, scores, factor contributions, reasons, the entire recommended order, and the untouched baseline order.
3. Complete for the waking foundation: full assessments are audited in cycle metadata/metrics; active advisory mode emits concise Working Memory hints while CognitiveBrain excludes the verbose object. Shadow mode emits no hint, and neither mode prunes.
4. Complete for the sleep hook: consolidation jobs retain a replay advisory alongside the unchanged baseline selection, and the reliable opt-in sleep coordinator now consumes that selection without enabling salience pruning.
5. Pending: compare advisory rankings with baseline retrieval through the application-level memory fixture before considering any pruning design.
6. Pending: add user-facing outcome measures for focus and conciseness, not just reduced context size.

**Exit criteria:** Salience pruning demonstrably retains required memories and improves focus without degrading retrieval quality.

## Phase 10: Govern Autonomous Work

**Goal:** Make reflection, discovery, self-assessment, curiosity, proactive engagement, and consolidation safe background capabilities rather than incidental tasks.

**Current status:** Core implementation complete. `AutonomousWorkGovernor` now acts as the prefrontal/basal-ganglia-inspired execution gate for sleep, reflection, discovery, curiosity, self-assessment, proactive engagement, summary updates, and STM flushes. `BackgroundTaskQueue` remains only as a compatibility adapter, and `DecisionEngine` is a signal producer rather than an execution authority. Real-cycle trigger quality and operating thresholds still need calibration.

1. Complete: every signal source maps to one of eight explicit task types with an operator-visible policy envelope; sleep retains its idle signal coordinator but not a separate execution authority.
2. Complete: the shared durable contract records user/task identity, trigger reason, signal snapshot, payload, priority, cooldown policy, de-duplication key, provider policy, attempt budget, status, timing, result, and failure/rejection reason.
3. Complete: global and per-user concurrency, hourly quotas, cooldowns, timeouts, bounded retries, active duplicate merging, foreground cancellation, operator cancel/retry, shutdown cancellation, and restart recovery prevent unbounded or falsely-running work.
4. Complete: all autonomous categories require the active cognitive provider to be local. Autonomous discovery has no cloud route; it can only propose an offline inquiry to the independently gated waking research system.
5. Complete for the control plane: operational task state and runtime toggles persist in SQLite; a trigger-protected append-only hash chain records every decision/outcome. Authenticated APIs and the responsive Autonomy workspace expose master/category toggles, limits, status, cancel/retry, and integrity. Deterministic tests cover rejection, duplicate events, retry, waking cancellation, persistence, provider enforcement, and SQL immutability.
6. Complete for safe posture: proactive messaging is independently governed and disabled by default. Pending: accumulate and review real reception data before changing its cooldown/rate envelope.

**Exit criteria:** Autonomous work is bounded, explainable, observable, and respects local-only/provider privacy policy.

## Phase 11: Predictive Cognition Experiments

**Goal:** Explore anticipation only after memory, learning, and attention have reliable measurements.

**Status:** Planned research work, not a product commitment.

1. Start with a small, interpretable next-intent or topic-transition baseline rather than a learned world model.
2. Track prediction accuracy, calibration, and whether prefetching context reduces latency without increasing irrelevant retrieval.
3. Use prediction errors as evaluation data first; do not let predictions change user-facing behavior automatically.
4. Consider episodic future simulation and affective forecasting only after simple prediction demonstrates value and safety.

**Exit criteria:** Prediction has measurable value beyond the existing Theory-of-Mind heuristics; otherwise retain it as an experiment.

## Explicitly Deferred

- Simulated “interoception” or claims that the system feels tired. Use resource metrics and backpressure instead.
- Specialized hippocampal pattern-separation work beyond improving retrieval evaluation and metadata. Revisit only if measured retrieval failures justify it.
- Reputation modeling as a separate social subsystem. Existing emotional memory and Theory of Mind should be evaluated first.
- Deep-RL, sequence models, and autonomous cloud activity. These are not justified until the simpler measurable systems above are validated.

## Configuration Shape

Implemented today:

```dotenv
# Provider roles, resolved independently. Each accepts 'gemini' or 'ollama'.
LLM_PROVIDER=gemini
EMBEDDING_PROVIDER=gemini
MODERATION_PROVIDER=gemini
SYNTHESIS_PROVIDER=
LOCAL_ONLY_MODE=false

OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_CHAT_MODEL=gemma4:e4b
OLLAMA_EMBEDDING_MODEL=embeddinggemma:latest
VISUAL_INPUT_ENABLED=true
VISUAL_PROVIDER=ollama
OLLAMA_VISION_MODEL=
VISUAL_MAX_IMAGE_BYTES=8388608
VISUAL_MAX_IMAGE_PIXELS=24000000
VISUAL_MAX_OUTPUT_TOKENS=900
AUDIO_INPUT_ENABLED=true
AUDIO_PROVIDER=ollama
OLLAMA_AUDIO_MODEL=
AUDIO_MAX_BYTES=4194304
AUDIO_MAX_DURATION_SECONDS=60
AUDIO_MAX_OUTPUT_TOKENS=800
OLLAMA_MAX_INTERACTIVE_REQUESTS=1
OLLAMA_MAX_BACKGROUND_REQUESTS=1
OLLAMA_NUM_CTX=16384
OLLAMA_THINKING=false

META_COGNITIVE_MAX_OUTPUT_TOKENS=64
META_COGNITIVE_MAX_RESPONSE_WORDS=40

# Fails startup rather than mixing vector spaces.
EMBEDDING_IDENTITY_ENFORCED=true
```

Implemented fail-closed Phase 5 boundary:

```dotenv
# Explicit research escalation
RESEARCH_ENABLED=false
RESEARCH_PROVIDER=disabled
RESEARCH_MODEL=models/gemini-3.5-flash-lite
RESEARCH_LOW_CONFIDENCE_THRESHOLD=0.55
RESEARCH_MAX_QUERIES=3
RESEARCH_MAX_QUERY_CHARS=500
RESEARCH_TIMEOUT_SECONDS=30

# Cognitive effort controller and durable offline inquiry queue
COGNITIVE_RESEARCH_DRIVE_ENABLED=false
COGNITIVE_RESEARCH_DRIVE_SHADOW_MODE=true
COGNITIVE_RESEARCH_DEEPEN_THRESHOLD=0.28
COGNITIVE_RESEARCH_UNCERTAINTY_THRESHOLD=0.48
COGNITIVE_RESEARCH_INQUIRY_THRESHOLD=0.64
COGNITIVE_RESEARCH_EXTERNAL_THRESHOLD=0.78
COGNITIVE_RESEARCH_COOLDOWN_MINUTES=30
COGNITIVE_RESEARCH_HYSTERESIS_MINUTES=15
INQUIRY_QUEUE_ENABLED=true
INQUIRY_DB_PATH=./chroma_db/inquiry_queue.sqlite3
INQUIRY_TTL_DAYS=14
INQUIRY_REQUIRE_USER_APPROVAL=true
```

Implemented governed sleep boundary:

```dotenv
# Disabled mode creates no scheduler task. Automatic runs require a fully local provider stack by default.
SLEEP_CYCLE_ENABLED=false
SLEEP_IDLE_MINUTES=30
SLEEP_CHECK_INTERVAL_SECONDS=60
SLEEP_COOLDOWN_MINUTES=360
SLEEP_MAX_CYCLES=20
SLEEP_REQUIRE_LOCAL_PROVIDER=true
SLEEP_LEDGER_DB_PATH=./chroma_db/sleep_cycle.sqlite3
```

The current provider contract always sends `question_only`. A compact/redacted context mode and its maximum size are not configurable until a context-release policy is implemented.

## Open Decisions

Resolved:

1. Ollama `0.32.5`, `gemma4:e4b` for generation, `embeddinggemma:latest` (768d) for embeddings.
2. `gemma4:e4b` is text-only in the installed Ollama release; its embedding endpoint returns `501`.
3. The embedding model is selected separately from the chat model, per role.
4. The initial cloud-context boundary is `question_only`: no transcript, summary, agent output, or memory enters a research request.

Still open:

5. The evidence thresholds for recommending automatic non-explicit activation. The operator can enable it deliberately in the UI, but calibration never self-activates it.
6. Whether to add an opt-in compact/redacted-summary context mode later; the default remains question-only.
7. The current Google model identifier, confirmed from the provider console/API rather than a marketing label.
8. The acceptable retrieval-quality delta and privacy-safe source corpus for any future Gemini-versus-local comparison. No legacy Gemini primary collection exists today.

## Current Execution Order

1. Complete: Phase 0 test repair and runtime baseline.
2. Complete for the current text path: Phase 1 provider boundary, shared structured-output repair, and Phase 2 local adapter/routing.
3. Complete: cycle latency instrumentation. Per-stage and per-request timing identifies local token generation as the bottleneck.
4. Complete and measured: cap meta-cognitive output and move summary/STM-flush summarisation off the response path. The formerly dominant stages now take about `0.53s` and `4.1s`; agent generation owns `85-89%` of the response path.
5. Complete as a smoke slice: verify both primary collections are already in the active local vector space, author a 12-query reviewed fixture, and record the first local retrieval baseline.
6. Complete: repair the application-level L2 distance conversion exposed by the matched live query; verify three LTM hits and correct Tom/Leeds recall.
7. Complete: replace the mutable personal-database fixture with a reproducible, synthetic, identity-stamped ephemeral collection; measure 50 reviewed queries without persistent database pollution.
8. Complete: introduce the disabled-by-default `ResearchService`, local `EscalationPolicy`, structured contracts, Discovery/meta-cognitive integration, and negative provider-invocation tests.
9. Complete in shadow: implement the brain-inspired `CognitiveResearchDrive` and persistent offline `InquiryCandidate` queue. Multi-signal accumulation, local-first effort allocation, inhibition, hysteresis/cooldowns, atomic dream-to-waking handoff, and the invariant that consolidation cannot invoke a provider are covered by deterministic tests.
10. Complete and live-smoked: waking inquiry revalidation and the grounded Gemini round trip—claim/reassess/resolve/approve, default approval for offline inquiries, strict response/source validation, timeout/failure behavior, and Cognitive Brain packet consumption with citations. Provider execution remains disabled by default.
11. Complete: add the waking inquiry review and calibration API: authenticated list/inspect/approve/dismiss/retry operations, a database-enforced append-only hash chain, persisted decisions/packets, verified-source feedback, real-cycle shadow observations and labels, paginated history, and calibration summaries. Approval remains consent rather than an authorization bypass, and calibration never self-activates automatic non-explicit research.
12. Complete: add the responsive React operator console and authenticated runtime control plane. Queue review, source feedback, labels, calibration strata, ledger inspection, staged provider/controller/automation toggles, typed confirmation, restart restoration, and emergency stop are live. Defaults remain unchanged.
13. Complete: implement the advisory SalienceNetwork foundation. Waking retrieval and sleep-candidate discovery retain baseline order/selection and a versioned explainable alternative ranking; shadow mode cannot influence synthesis, active mode supplies only compact hints, and pruning is absent.
14. Complete: reliable sleep/consolidation now has repaired user-scoped retrieval, persistent provenance-rich semantic output, Cognitive Brain retrieval, a single lifecycle owner, disabled-by-default enable flag, idle/cooldown/local-provider gates, stage-aware retry, waking cancellation, cancellation-safe shutdown, outcome metrics, and a tamper-evident audit record for every run/job.
15. Complete: the unified autonomous-work governor now gives reflection, discovery, self-assessment, curiosity, proactive engagement, summary work, STM flushes, and sleep one bounded task contract, policy gate, cancellation model, de-duplication key, retry budget, durable task state, immutable audit surface, authenticated API, and operator UI.
16. Complete: the event-driven observability plane publishes typed bounded telemetry for cognitive cycles, memory, research, salience, sleep, and governed autonomous work. The authenticated WebSocket now supports process stream identity, monotonic cursors, filtered replay, snapshots, heartbeats, explicit replay/backpressure gaps, and non-blocking per-viewer queues. The operator System workspace reconnects with bounded backoff and renders live domain health plus an ordered activity feed. Domain databases and immutable ledgers remain authoritative.
17. Complete: the local-first visual sensory path verifies Ollama's declared `vision` capability, rejects malformed/mismatched/oversized JPEG and PNG inputs before inference, performs one scheduled local image observation, caps confidence by observable input quality, and replaces pixels with typed provenance-marked untrusted evidence. Perception, Cognitive Brain, autobiographical memory, health, telemetry, and the frontend MIME contract are wired without any visual cloud fallback. A live `gemma4:e4b` smoke validated transport and parsing.
18. Complete: the local-first auditory sensory path verifies Ollama's declared `audio` capability, accepts only bounded canonical PCM WAV, performs deterministic signal checks and conservative local observation, and replaces samples with typed provenance-marked untrusted `AudioEvidence`. Transcripts cannot alter instruction text; Perception, synthesis, autobiographical memory, health, telemetry, and browser upload/microphone flows preserve the boundary without cloud fallback.
19. **Next code slice:** implement multisensory temporal binding and reliability-aware attention: one typed `SensoryEpisode` aligns text, vision, and audition for the same turn, preserves each modality's provenance and uncertainty, detects cross-modal agreement/contradiction, and produces advisory salience cues without letting generative fusion rewrite primary evidence. The research shadow-calibration study, salience baseline comparison, autonomous-trigger calibration, representative visual/audio accuracy fixtures, and real sleep-cycle tuning remain operator validation work in parallel; none may self-enable control.

The next implementation slice is multisensory integration, analogous to thalamic timing plus association-cortex binding. Keep `VisualEvidence` and `AudioEvidence` immutable as primary observations, attach explicit temporal/co-occurrence relationships, arbitrate conflicts using measured modality quality rather than model confidence alone, and initially expose fusion only as typed advisory evidence. In parallel, collect human-labelled image, clean speech, silence, tone, noise, and adversarial spoken-instruction fixtures; successful transport smokes are not perceptual-accuracy benchmarks.
