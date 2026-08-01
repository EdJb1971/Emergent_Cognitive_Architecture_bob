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
- The first research-governance slice is implemented. Discovery and the meta-cognitive SEARCH_FIRST path now use a deterministic local `EscalationPolicy` and `ResearchService`; LLM-suggested queries cannot self-authorize, normal/disabled/local-only paths never invoke a provider, and every decision has structured audit metadata. Context is hard-limited to `question_only`. No live research adapter is present or enabled.

**Test baseline:** `116 passed, 3 skipped` on August 2, 2026. The skips are three root-level async tests that are not handled by an async test plugin.

**Latency is now measured, not guessed.** `StageTimer` wraps every orchestrator stage and emits a `CYCLE_TIMING` line plus `metadata["stage_timings_ms"]`; `OLLAMA_CALL` and `OLLAMA_EMBED` lines give per-request server, queue, prompt-eval, and eval time. Three cycles on August 1, 2026 overturned the previous assumption:

- Orchestration costs **under 200ms per cycle**. Routing, both conflict checks, contextual encoding, theory-of-mind validation, and RL updates are collectively free. The "~40s of orchestration overhead" theory was wrong.
- The cost is **local token generation at ~26 tokens/second**. Prompt evaluation is 0.4-0.8s and model load a constant ~0.8s per call; everything else is tokens coming out.
- The earlier "warm agent call ~1.1s" figure was a trivial smoke-test prompt. Real agent calls generate 80-600 tokens and cost 4-25s each.
- In the August 1 baseline, three stages owned the whole cycle: `stage1_agents`/`stage2_agents` (20-57s, serialised by `max_interactive=1`), `meta_cognitive` (20-25s for a single unbounded call producing 483-605 tokens), and `memory_upsert` (16-22s, including a ~12s summarisation call). The August 2 slice bounded the first and deferred the second.
- Gemini synthesis is 3.0-3.2s and consistently the cheapest stage in the cycle.
- Per-agent `AGENT_METRIC` durations were previously always ~0ms because the timer started after `asyncio.gather` had already returned. Now measured correctly.

**First latency-reduction slice completed on August 2, 2026:**

1. Meta-cognitive uncertainty output is capped at 64 tokens and hard-truncated to 40 words by default.
2. Per-turn summary generation and STM-flush summarisation are queued off the response path when `BackgroundTaskQueue` is wired, with a per-user lock around summary writes.
3. The synchronous fallback remains for isolated/test construction. The production queue is in-process and non-durable; shutdown cancels unfinished jobs.

Two matched post-change cycles are now preserved. Meta-cognition measured `0.53-0.54s` versus the prior `20-25s` slow path; `memory_upsert` measured `3.92-4.24s` versus `16-22s`, while its deferred summaries completed `4.09-5.05s` after the response path. Total cycles were `65.74s` and `49.12s`; Stage 1 and Stage 2 generation still consumed `85-89%`, so agent output volume remains the next latency lever. Do not raise `max_interactive` without a separate VRAM/contention measurement.

**Memory defect found and repaired during the live measurement:** The collections use Chroma's default squared-L2 metric, but application code converted distances below `1.0` as if they were cosine distance and distances above `1.0` with a different reciprocal formula. A closer result at `0.865` therefore scored `0.135`, while a worse result at `1.088` scored `0.479`. The first matched run gave Cognitive Brain zero memories and an incorrect clarification. Metric-aware monotonic conversion now aligns normalized L2 scores with STM cosine similarity; the second run supplied three LTM memories and correctly recalled both Tom and Leeds.

**Current evidence slice:** Complete. The research boundary is fail-closed and tested: an LLM suggestion alone cannot trigger research, explicit requests remain blocked when disabled, and local-only mode overrides an enabled/available test provider. An enabled test provider produces bounded, de-duplicated, question-only structured packets.

## Direction

Run the normal cognitive cycle locally through Ollama, using a capable local multimodal model for chat, image understanding, audio understanding where supported, agent analysis, and final synthesis. Reserve cloud inference for explicit research escalation: the Discovery/Curiosity path may call a configurable cloud model when local knowledge is insufficient or current external information is required.

This is a good fit for ECA. It makes the persistent memory and cognitive orchestration inexpensive to run, keeps routine conversations private, and makes every cloud call intentional and inspectable.

The implementation supports the local routine path today: roles are independently selectable, Ollama JSON mode is requested for structured calls, and shared JSON extraction/repair is used by the agents and Cognitive Brain. Research authorization and a question-only privacy boundary are now implemented, but the target split is still incomplete because no live grounded research adapter, research-packet synthesis path, multimodal capability routing, or hybrid observability exists.

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

**Current status:** The provider-neutral safety boundary is implemented and disabled by default. Legacy `WebBrowsingService` is no longer instantiated or reachable. Deterministic reflex signals, fail-closed dispositions, question-only context, structured packets, audit logs, and Discovery/meta-cognitive integration are tested. The intended brain-inspired research drive, offline inquiry queue, real grounded provider, and packet-to-synthesis path remain unimplemented.

1. Boundary complete; provider pending: replace `WebBrowsingService` with a `ResearchService` protocol and disabled provider. Next add:
   - `CloudResearchProvider` for a cloud LLM with grounded search/retrieval capability.
   - Optional future direct-search adapter when source-level control is needed.
2. First-line gate complete; cognitive controller pending: the deterministic `EscalationPolicy` supplies interpretable signals to Discovery and the meta-cognitive SEARCH_FIRST path:
   - current or time-sensitive information
   - explicit user request to research/search the web
   - local confidence below a configured threshold
   - named fact that is absent from local memory
   Build a bounded evidence accumulator above these gates using calibrated uncertainty, inter-agent/memory conflict, novelty or prediction error, temporal volatility, task stakes, persistence after local attempts, expected information gain, cloud cost/privacy, and refractory cooldown. It must support the action ladder: deeper local thought -> clarification/uncertainty -> approval if required -> cloud research.
3. Boundary complete: require a local policy decision before provider contact and record decision/request IDs, reasons, disposition, provider, model, timestamp, estimated query size, context size/policy, provider latency, source/claim counts, and cost when supplied. Persistent metrics still belong to Phase 6.
4. First privacy boundary complete: only the question is permitted; no transcript, summary, agent output, or raw memory crosses the provider contract. A future compact/redacted-summary mode requires its own release policy and tests.
5. Contract complete; synthesis pending: structured packets contain claims, source IDs, URLs, publication/access dates, confidence, caveats, status, context size, and optional cost. Cognitive Brain does not consume them yet.
6. Configuration boundary complete; adapter pending: `RESEARCH_ENABLED`, provider/model, confidence threshold, and query bounds are explicit. Only `RESEARCH_PROVIDER=disabled` is accepted until a real adapter is implemented and verified.

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

**Current status:** Implemented in parts, with a reproducible direct-ranking baseline but no end-to-end reliability validation. STM, background summary updates, ChromaDB storage, and the consolidation service exist. The diverse synthetic corpus is complete; application-level expected-fact evaluation, periodic consolidation scheduling, cleanup policy, and recovery verification remain incomplete.

1. Make all memory budgets model-neutral. Derive STM/context reserves from active provider capabilities instead of Gemini-specific limits.
2. Complete for direct vector ranking: the version 2 fixture seeds 25 synthetic records and evaluates 50 reviewed queries in an ephemeral collection. It includes time-sensitive cases and expected-fact labels without depending on mutable personal memory.
3. Local measurement complete: the canonical seeded corpus measured recall, MRR, and NDCG of `1.000` at k=1 and k=5 with `embeddinggemma:latest`; the earlier personal-database smoke run measured `0.958`, `0.958`, and `0.952` at k=5. The evaluator reports partial recall and ranking failures. Define an acceptable delta before any cross-provider comparison; the synthetic corpus is safe to compare, but no retained Gemini collection exists.
4. Preserve provenance: summaries and semantic memories must retain source cycle IDs, generation provider/model, timestamps, and embedding version.
5. Add per-user locking and fault-injection tests for summary-before-flush, failed upserts, and interrupted recovery.
6. Add a deliberate STM cleanup and snapshot policy: retention bounds, recovery age limits, periodic snapshots, and no silent data deletion.
7. After repairing and testing the consolidation contract, start its scheduler only behind an enable flag, with one lifecycle owner, task de-duplication, cooldowns, and a shutdown test.
8. Record memory metrics: retrieval latency, hit source (STM/summary/LTM), flush reason, token counts before/after, summary failures, and consolidation outcome.
9. Repair and test the consolidation service contract before scheduling it: implement or replace its missing `MemoryService.get_user_cycles()` dependency and pass both `user_id` and `cycle_id` to `get_cycle_by_id()`.
10. Verify the episodic-to-semantic extraction path has a persistent destination and a retrieval path before presenting semantic memories as available to CognitiveBrain.
11. Test summary identity extraction on diverse inputs and treat regex matching as a fallback heuristic, not a durable identity system.
12. Add a persistent offline `InquiryCandidate` queue for dream/consolidation output. Candidates retain question, hypothesis, source-cycle provenance, uncertainty, novelty/prediction error, salience, expected information gain, status, and expiry. Consolidation may propose candidates but never execute research; waking cognition must revalidate and authorize them.

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
SYNTHESIS_PROVIDER=
LOCAL_ONLY_MODE=false

OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_CHAT_MODEL=gemma4:e4b
OLLAMA_EMBEDDING_MODEL=embeddinggemma:latest
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
RESEARCH_MODEL=
RESEARCH_LOW_CONFIDENCE_THRESHOLD=0.55
RESEARCH_MAX_QUERIES=3
RESEARCH_MAX_QUERY_CHARS=500
```

The current provider contract always sends `question_only`. A compact/redacted context mode and its maximum size are not configurable until a context-release policy is implemented.

## Open Decisions

Resolved:

1. Ollama `0.32.5`, `gemma4:e4b` for generation, `embeddinggemma:latest` (768d) for embeddings.
2. `gemma4:e4b` is text-only in the installed Ollama release; its embedding endpoint returns `501`.
3. The embedding model is selected separately from the chat model, per role.
4. The initial cloud-context boundary is `question_only`: no transcript, summary, agent output, or memory enters a research request.

Still open:

5. Whether cloud research is allowed automatically by policy or only after user confirmation.
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
9. **Next:** implement the brain-inspired `CognitiveResearchDrive` and persistent offline `InquiryCandidate` queue. Validate multi-signal accumulation, local-first effort allocation, hysteresis/cooldowns, dream-to-waking handoff, and the invariant that consolidation cannot invoke a provider.
10. Then implement one grounded `CloudResearchProvider` behind that controller, plus explicit context release, packet validation, timeout/failure behavior, and Cognitive Brain packet consumption. Keep it disabled by default until its model identifier and source behavior are verified.
11. Then Phase 6 observability, followed by Phase 8 through Phase 10 validation of learning, attention, salience, and autonomous task mechanisms before predictive cognition.

The next slice is the cognitive trigger layer, before any real cloud adapter: build a bounded research-drive accumulator and an offline inquiry queue shared by waking metacognition and sleep-like consolidation. Dream output may propose research, but only waking governance may approve it. A live provider follows only after those invariants are measured and tested.
