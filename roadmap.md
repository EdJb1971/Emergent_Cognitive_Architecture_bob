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

**Completed on August 1, 2026:**

- Test collection and the current backend suite are repaired: `64 passed, 3 skipped` using the repository virtual environment.
- The Gemini-specific service now sits behind a minimal provider contract and `GeminiProvider` compatibility adapter.
- `OllamaProbe` checks `/api/tags` and `/health/deep` reports local server/model availability.
- Ollama `0.32.5` is running locally and `gemma4:e4b` is installed. The model passed a GPU text smoke test.
- `ModelExecutionScheduler`, `ProviderRequest`, `ProviderResult`, and a text-only `OllamaProvider` are implemented and covered by provider-contract tests. Scheduler limits are visible through `/health/deep`.
- `gemma4:e4b` returns `501 Not Implemented` from Ollama's embedding endpoint. It is therefore a local generation model only; a separate local embedding model is required before full local routing.
- `embeddinggemma:latest` is installed and verified through `OllamaEmbeddingProvider`. Its runtime vector dimension is `768`.
- **Provider selection is now configuration-driven.** `_build_active_provider()` resolves generation, embedding, and moderation independently from `LLM_PROVIDER`, `EMBEDDING_PROVIDER`, and `MODERATION_PROVIDER`, composing them through `CompositeProvider` only when they differ. Unknown values fail startup.
- **`GEMINI_API_KEY` is optional.** It is required only when a Gemini-backed role is selected, so a fully local configuration can start without a cloud credential. Gemini remains the configured default because it is the only cloud key currently held; nothing in the architecture depends on it.
- **`CompositeProvider.is_local` accounts for the safety provider**, so a cloud moderation call can no longer hide inside an otherwise-local configuration. This is the flag `LOCAL_ONLY_MODE` will key off in Phase 6.
- **Embedding identity is enforced at the collection boundary.** `EmbeddingProvider.verify()` resolves the runtime dimension at startup; embedding-backed collections are stamped with provider/model/dimension; `apply_embedding_identity()` compares provider and model and raises `EmbeddingIdentityMismatch` on conflict. Untagged non-empty collections are treated as legacy `gemini/models/embedding-001`.
- The guard deliberately does **not** compare dimensions. Gemini `embedding-001` and `embeddinggemma` are both 768-dimensional, so a dimension check would have accepted two incompatible vector spaces and degraded retrieval silently rather than failing.
- `/health/deep` now reports the active provider per role, `is_local`, and the resolved embedding identity.

**Known consequence:** setting `EMBEDDING_PROVIDER=ollama` against the existing database now fails at startup, by design. This is the correct behavior and stays until the migration command in Phase 3 exists.

**Next action:** Build the versioned re-embedding command (Phase 3 step 3) and the retrieval fixture set (Phase 7 step 2), then compare local against Gemini retrieval before switching any collection.

## Direction

Run the normal cognitive cycle locally through Ollama, using a capable local multimodal model for chat, image understanding, audio understanding where supported, agent analysis, and final synthesis. Reserve cloud inference for explicit research escalation: the Discovery/Curiosity path may call a configurable cloud model when local knowledge is insufficient or current external information is required.

This is a good fit for ECA. It makes the persistent memory and cognitive orchestration inexpensive to run, keeps routine conversations private, and makes every cloud call intentional and inspectable.

The existing implementation does not fully support this split yet. The provider boundary exists and roles are independently selectable, but `LLMIntegrationService` is still the only implementation behind agents' structured-output expectations, and local generation cannot be enabled until JSON parsing is made resilient.

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

**Status:** Met. Items 1, 3, 4, 6, and 7 are implemented. Item 2 is implemented at startup wiring; item 5 (structured-output parse/repair boundary) is still outstanding and is a prerequisite for routing agents to a local model.

## Phase 2: Add Ollama as the Local Default

**Goal:** Run text reasoning locally before enabling multimodal paths.

1. Implemented: `OllamaProvider` uses the Ollama HTTP API for bounded local text generation; its scheduler is initialized at application startup.
2. Implemented: local defaults are selectable per role.
   - `LLM_PROVIDER=ollama`
   - `OLLAMA_BASE_URL=http://localhost:11434`
   - `OLLAMA_CHAT_MODEL=<verified-local-model>`
   - model-specific context and output limits remain outstanding
3. Do not route routine calls yet: `gemma4:e4b` is verified for local text generation but does not implement Ollama embeddings, and agents still assume reliable JSON. Enable local generation only after the structured-output repair boundary (Phase 1 item 5) exists.
4. Implemented: health checks for Ollama availability and installed model presence.
5. Add a provider fallback policy: fail clearly for normal local requests; never silently send private context to a cloud provider.
6. Add a `ModelExecutionScheduler` (or `InferenceBudgetManager`) before enabling local routine routing. It owns bounded concurrency, interactive-over-background priority, per-cycle call/token budgets, cancellation, compact-cycle degradation, model-residency state, and queue/VRAM telemetry.
7. Benchmark the full cycle and tune agent activation, token budgets, concurrency, and context trimming for the available hardware.

**Exit criteria:** A text-only `/chat` cycle completes locally with no Gemini API calls; local inference remains bounded under concurrent work; and the selected provider/model plus scheduling decision are visible in cycle metadata and logs.

## Phase 3: Separate Embeddings and Memory Retrieval

**Goal:** Remove the remaining cloud dependency from persistent memory.

1. Implemented: `OllamaEmbeddingProvider` is a dedicated local embedding adapter, selectable via `EMBEDDING_PROVIDER=ollama`.
2. Implemented: embedding provider/model/dimension/identity-version are stored in ChromaDB collection metadata, and a provider+model comparison fails closed on mismatch. Dimension is not used as the compatibility test because Gemini `embedding-001` and `embeddinggemma` share 768 dimensions.
3. Create a migration command that rebuilds embeddings into a new collection rather than mixing vector spaces in the current collection. **This is the active blocker.** It must re-embed `cognitive_cycles` and `conversation_summaries` into new identity-stamped collections, leave the originals intact, and be resumable.
4. Compare retrieval quality against the existing Gemini embeddings with a small curated memory-query fixture set.
5. Switch `MemoryService`, summaries, and retrieval to the local embedding provider only after quality and dimensional compatibility are verified.
6. Add batched embedding to `OllamaEmbeddingProvider`. `/api/embed` accepts a list, and re-embedding a full collection one string at a time will be needlessly slow.
7. Extend identity stamping to the remaining embedding-backed collections (self-model, autobiographical, emotional profiles) once the migration path is proven on the two primary collections.

**Exit criteria:** Memory ingestion and retrieval work with no cloud embedding call, and legacy vector collections remain recoverable.

## Phase 4: Enable Multimodal Capability Safely

**Goal:** Use the selected Ollama model for image/audio only after verifying actual support.

1. Verify the exact Ollama model tag and its declared capabilities. Do not assume `gemma-4-E4B-it` supports image or audio solely from its name; confirm what Ollama reports and test it with known fixtures.
2. Add model-capability probes at startup and expose their results through `/health/deep`.
3. Wire `VisualInputProcessor` into the cognitive cycle only when the active provider supports images.
4. Introduce `TranscriptionProvider` as a separate capability and pipeline stage:
   - If a local model accepts audio, adapt it behind `TranscriptionProvider`.
   - Otherwise use a local speech-to-text engine, then pass its transcript to the normal local chat provider.
5. Preserve MIME type, file size, duration, and provenance metadata. Enforce local limits before base64 content enters a prompt.
6. Add image and audio fixture tests for success, unsupported capability, invalid MIME type, oversized input, and provider failure.

**Exit criteria:** Image and audio paths are either locally functional with tested models or cleanly marked unavailable; no hidden Gemini fallback exists.

## Phase 5: Replace Web Browsing With Cloud Research Escalation

**Goal:** Make Discovery the only cloud-capable node by policy.

1. Replace `WebBrowsingService` with a `ResearchService` that has pluggable strategies:
   - `CloudResearchProvider` for a cloud LLM with grounded search/retrieval capability.
   - Optional future direct-search adapter when source-level control is needed.
2. Add an `EscalationPolicy` used by `DiscoveryAgent` and `MetaCognitiveMonitor`. Initial reasons:
   - current or time-sensitive information
   - explicit user request to research/search the web
   - local confidence below a configured threshold
   - named fact that is absent from local memory
3. Require a local policy decision before cloud contact and record: request ID, reason, provider, model, timestamp, token/context size, source list, and cost estimate if available.
4. Minimize cloud context. Send the question and a compact, redacted local summary, not full transcripts or raw memory by default.
5. Return a structured research packet with claims, sources, publication/access dates, confidence, and caveats. Have the local Cognitive Brain synthesize the user-facing answer.
6. Make the cloud provider configurable, for example `RESEARCH_PROVIDER=gemini`, `RESEARCH_MODEL=<verified-model-name>`, so Gemini Flash can be used now without making it a permanent architectural dependency.

**Exit criteria:** A normal local chat does not call the cloud; a research-required query produces an auditable cloud research packet and a locally synthesized response with sources.

## Phase 6: Local Safety, Observability, and Operations

**Goal:** Operate the hybrid architecture predictably.

1. Separate content safety from the chat provider. Start with simple local policy checks and make any cloud moderation an explicit opt-in escalation.
2. Add provider-level metrics: selected provider/model, local/cloud ratio, latency, failure class, context size, escalation reason, and estimated cloud usage.
3. Extend dashboard streaming only after the current snapshot-only WebSocket is replaced with a real subscription/broadcast mechanism.
4. Add a local-only mode that rejects all cloud escalation, plus a research-enabled mode that requires an explicit configuration flag.
5. Add CI checks for tests, type checking, frontend build, `.env.example` completeness, and provider contract tests.

**Exit criteria:** Local-only mode is enforceable and tested; hybrid activity is observable; background services have explicit lifecycle management.

## Phase 7: Memory Reliability and Evaluation

**Goal:** Make the existing STM, summary, LTM, and consolidation design reliable with local models and measurable retrieval quality.

**Current status:** Implemented in parts, not yet validated end-to-end. STM, summary, ChromaDB storage, and the consolidation service exist. Periodic consolidation scheduling, cleanup policy, recovery verification, and evaluation evidence are incomplete.

1. Make all memory budgets model-neutral. Derive STM/context reserves from active provider capabilities instead of Gemini-specific limits.
2. Create a versioned memory evaluation fixture: at least 50 queries, known relevant cycle IDs, time-sensitive cases, and expected summary facts.
3. Measure local embedding retrieval with top-$k$ recall, MRR, and NDCG before switching any collection. Define an acceptable quality delta before migration.
4. Preserve provenance: summaries and semantic memories must retain source cycle IDs, generation provider/model, timestamps, and embedding version.
5. Add per-user locking and fault-injection tests for summary-before-flush, failed upserts, and interrupted recovery.
6. Add a deliberate STM cleanup and snapshot policy: retention bounds, recovery age limits, periodic snapshots, and no silent data deletion.
7. After repairing and testing the consolidation contract, start its scheduler only behind an enable flag, with one lifecycle owner, task de-duplication, cooldowns, and a shutdown test.
8. Record memory metrics: retrieval latency, hit source (STM/summary/LTM), flush reason, token counts before/after, summary failures, and consolidation outcome.
9. Repair and test the consolidation service contract before scheduling it: implement or replace its missing `MemoryService.get_user_cycles()` dependency and pass both `user_id` and `cycle_id` to `get_cycle_by_id()`.
10. Verify the episodic-to-semantic extraction path has a persistent destination and a retrieval path before presenting semantic memories as available to CognitiveBrain.
11. Test summary identity extraction on diverse inputs and treat regex matching as a fallback heuristic, not a durable identity system.

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

**Current status:** Planned; no implementation exists.

1. Add an advisory-only memory ranking service after retrieval, using query relevance, recency, emotional salience, novelty, and must-keep flags.
2. Return top-$k$ candidates with scores and reasons; preserve the full candidate list for audit and evaluation.
3. Pass concise memory and response-priority hints to Working Memory and Cognitive Brain rather than raw large retrieval sets.
4. Compare advisory rankings with baseline retrieval on the memory fixture set before allowing pruning.
5. Add user-facing outcome measures for focus and conciseness, not just reduced context size.

**Exit criteria:** Salience pruning demonstrably retains required memories and improves focus without degrading retrieval quality.

## Phase 10: Govern Autonomous Work

**Goal:** Make reflection, discovery, self-assessment, curiosity, proactive engagement, and consolidation safe background capabilities rather than incidental tasks.

**Current status:** `DecisionEngine` and `BackgroundTaskQueue` exist, but autonomous behavior needs explicit scheduling, de-duplication, auditing, and evaluation.

1. Document every signal source and trigger policy: reflection, discovery, self-assessment, curiosity, STM pressure, summary updates, and consolidation candidates.
2. Define a task contract with user ID, trigger reason, input metrics snapshot, cooldown, de-duplication key, provider policy, and completion result.
3. Add per-task rate limits, cancellation, retries, and idempotency; no autonomous task may silently create an unbounded loop.
4. Apply the research escalation policy to autonomous discovery. A background task may not contact the cloud in local-only mode.
5. Add audit views and integration tests for trigger thresholds, cooldowns, task failure, shutdown, and duplicate events.
6. Keep proactive messages opt-in and measure negative reactions before expanding their frequency.

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

OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_CHAT_MODEL=gemma4:e4b
OLLAMA_EMBEDDING_MODEL=embeddinggemma:latest
OLLAMA_MAX_INTERACTIVE_REQUESTS=1
OLLAMA_MAX_BACKGROUND_REQUESTS=1

# Fails startup rather than mixing vector spaces.
EMBEDDING_IDENTITY_ENFORCED=true
```

Proposed, not yet implemented. Add with Phase 5 and Phase 6:

```dotenv
# Explicit research escalation
RESEARCH_ENABLED=true
RESEARCH_PROVIDER=gemini
RESEARCH_MODEL=<verified-cloud-model>
RESEARCH_ALLOW_LOCAL_CONTEXT=false

# Privacy and operating modes
LOCAL_ONLY_MODE=false
MAX_CLOUD_CONTEXT_CHARS=12000
```

## Open Decisions

Resolved:

1. Ollama `0.32.5`, `gemma4:e4b` for generation, `embeddinggemma:latest` (768d) for embeddings.
2. `gemma4:e4b` is text-only in the installed Ollama release; its embedding endpoint returns `501`.
3. The embedding model is selected separately from the chat model, per role.

Still open:

4. Whether cloud research is allowed automatically by policy or only after user confirmation.
5. The privacy boundary for cloud escalation: no local context, compact summary only, or user-approved selected memories.
6. The current Google model identifier, confirmed from the provider console/API rather than a marketing label.
7. The acceptable retrieval quality delta for migrating from Gemini to local embeddings, defined before the comparison is run.

## Current Execution Order

1. Complete: Phase 0 test repair and runtime baseline.
2. Complete: Phase 1 provider boundary and Phase 2 local adapter/scheduler, except the structured-output repair boundary.
3. **Active:** Phase 3 migration command plus the Phase 7 retrieval fixture set, then measure local against Gemini retrieval.
4. Then the structured-output parse/repair boundary, which unblocks routing agents to the local model.
5. Then Phase 5 and Phase 6: controlled cloud research escalation, local-only enforcement, and hybrid observability.
6. Then Phase 8 through Phase 10: validate learning, attention, salience, and autonomous task mechanisms before adding predictive cognition.

The next slice is the re-embedding migration command and the retrieval fixture set. Until retrieval is measured, the identity guard is what keeps the existing memory intact.