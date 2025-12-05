## Memory Architecture Deep Dive

This document explains how the memory system works end-to-end: what is stored, where, when agents access it, how recall works (STM and LTM), thresholds and token budgets, and how to retrieve an in-order running log of the conversation.

### TL;DR
- Short-Term Memory (STM): In-memory per-user cache of recent CognitiveCycles, token-budgeted, ordered newest-first. Used for fast recall with per-field embeddings (user_input and final_response).
- Long-Term Memory (LTM): ChromaDB persistent store with complete CognitiveCycles and discovered patterns, plus a Conversation Summary collection. Survives restarts and provides vector search.
- Summaries: Conversation-level summary per user, updated on every cycle and stored in Chroma. Used to provide concise context to the Cognitive Brain and agents.
- Recall: STM first (cosine similarity using precomputed per-cycle vectors), then LTM (Chroma query distance converted to [0..1] score). Default min relevance: 0.5.
- In-order log: Every cycle is persisted to LTM with a timestamp and session_id. Use `MemoryService.list_cycles` to retrieve an ordered transcript.
 - Immediate Transcript: Per-user rolling verbatim buffer (~50k tokens) of the latest conversation turns (user input + AI final response). Ephemeral, in-memory only, and injected into the CognitiveBrain prompt before summaries and retrieved memories.


## Data Model (key types)

All types below live under `src/models`.

- CognitiveCycle (src/models/core_models.py)
  - cycle_id: UUID
  - user_id: UUID
  - session_id: UUID
  - timestamp: datetime (UTC)
  - user_input: str
  - user_input_embedding: Optional[List[float]]
  - agent_outputs: List[AgentOutput]
  - final_response: Optional[str]
  - final_response_embedding: Optional[List[float]]
  - response_metadata: Optional[ResponseMetadata]
  - outcome_signals: Optional[OutcomeSignals]
  - metadata: Dict[str, Any]
  - reflection_status: str
  - discovery_status: str
  - score: Optional[float] (query relevance)

- MemoryQueryRequest (src/models/core_models.py)
  - user_id: UUID
  - query_text: str
  - query_embedding: Optional[List[float]]
  - limit: int (default 5)
  - min_relevance_score: float (default 0.5)
  - metadata_filters: Optional[Dict[str, Any]]

- ShortTermMemory (src/models/memory_models.py)
  - user_id: UUID
  - recent_cycles: List[CognitiveCycle] (newest first)
  - token_count, token_budget, token_reserve
  - add_cycle(), flush_cycles(), get_recent_cycles()

- ConversationSummary (src/models/memory_models.py)
  - summary_id: UUID
  - user_id: UUID
  - key_topics: List[str]
  - latest_topic: Optional[str]
  - entities: List[str]  (stored as list; normalized to a set in-memory on read to avoid duplicates)
  - context_points: List[str]
  - conversation_state: str
  - identity: Optional[{ user_name?: str, ai_name?: str, location?: str }]
  - preferences: Optional[List[str]]
  - embedding: Optional[List[float]]
  - last_updated, update_count

- DiscoveredPattern (src/models/core_models.py)
  - pattern_id, user_id, timestamp, pattern_type, description, embedding, metadata


## Storage Layers

### 0) Immediate Conversation Transcript (verbatim)
- Location: In-memory (`MemoryService._transcripts[user_id]` -> `RollingTranscript`).
- Contents: Alternating verbatim turns from the current session: user inputs and the AI's final responses.
- Budget: Token-based (~50k tokens by default) with automatic pruning of oldest utterances when over budget.
- Purpose: Gives the LLM a large, literal in-session context window beyond summaries and retrieved memories.
- Persistence: None. This is ephemeral and cleared on restart or process restart.

### 1) Short-Term Memory (STM)
- Location: In-memory (`MemoryService._stm_cache[user_id]` -> `ShortTermMemory`).
- Order: Newest-first list `recent_cycles`.
- Budget: Token-based. New cycles added via `ShortTermMemory.add_cycle()` increase `token_count`. When over budget, MemoryService emits `stm_pressure` and `flush_to_ltm` later consolidates/flushes.
- Embeddings: For each new cycle, we precompute two embeddings:
  - `user_input_embedding` for `user_input`
  - `final_response_embedding` for `final_response`
  This enables fast vector STM recall without re-embedding.

### 2) Long-Term Memory (LTM)
- Backend: ChromaDB PersistentClient at `settings.CHROMA_DB_PATH`.
- Collections:
  - `cognitive_cycles` (settings.CHROMA_COLLECTION_CYCLES)
    - Upserts store: vector embedding (of a compact text view), `documents` (the same text), and rich `json_data` (the full `CognitiveCycle` via `.model_dump_json()`), plus metadata like `user_id`, `timestamp`.
  - `discovered_patterns` (settings.CHROMA_COLLECTION_PATTERNS)
    - Stores pattern embeddings/documents/metadata.
  - `conversation_summaries` (SummaryManager internal)
    - Stores one summary per user (with `embedding` and `json_data`).
- Telemetry: Chroma anonymized telemetry disabled in MemoryService for reliability.

### 3) Summaries
- Manager: `SummaryManager`
- Update cadence: Called on every `upsert_cycle()`
- LLM analysis: Extracts topics/entities/context updates and identity hints (user_name, ai_name, location). Uses structured JSON extraction with robust fallbacks (regex identity extraction and mining recent cycles) and persists the result.
- Consolidation: When STM flushes, a consolidated STM text is embedded and stored for traceability.


## Write Path (what gets memorized and when)

Entry point: `MemoryService.upsert_cycle(cycle)`
1) Precompute per-field embeddings (if available):
   - `user_input_embedding` and `final_response_embedding` via `LLMIntegrationService.generate_embedding`.
2) Update Immediate Transcript: Append verbatim `user_input` and, after generation, the `final_response` to the per-user `RollingTranscript` (auto-pruned to budget).
3) Add to STM: `add_cycle(cycle)` updates token counts and returns whether a flush is needed.
4) Update Summary: `SummaryManager.update_summary(user_id, cycle)`
   - Generates analysis, updates summary fields, re-embeds, persists to Chroma.
5) Persist cycle to LTM: `_store_cycle(cycle)` upserts into `cognitive_cycles` with a unified embedding text and full JSON payload.
6) Flush (if needed): `flush_to_ltm(user_id, cycles)`
   - Summarizes flushed cycles, stores consolidated STM record, removes cycles from STM, emits `flush_completed` signal.

What’s retained per cycle (LTM):
- Full `CognitiveCycle` JSON (user input, agent outputs, final response, metadata, response metadata, outcome signals, timestamps, session id, statuses).
- A compact `document` text for embedding/search.

What’s retained in STM:
- Full `CognitiveCycle` objects in `recent_cycles`, ordered newest first.
- Precomputed `user_input_embedding` and `final_response_embedding` for fast STM recall.


## Read Path (how recall works)

Entry point: `MemoryService.query_memory(MemoryQueryRequest)`
1) Embed query once: `query_embedding` from `LLMIntegrationService.generate_embedding`.
2) STM recall (first):
   - For each recent cycle in STM, compute cosine similarity with both `user_input_embedding` and `final_response_embedding` and take the best.
   - Keep cycles with `best >= min_relevance_score` (default 0.5).
   - Sort by score and clip to `limit`.
3) LTM vector search (Chroma):
   - Query `cognitive_cycles` with `query_embeddings=[query_embedding]` and `where={'user_id': str(user_id), ...}`.
   - Convert returned distances to [0..1] scores via `_distance_to_score` and filter with `min_relevance_score`.
4) Merge results: `STM + LTM`, sort by score desc, and return top `limit`.
5) Telemetry: Update `MemoryAccessStats` and emit `query_metrics` signals when the DecisionEngine is wired.

Immediate verbatim context:
- `MemoryService.get_immediate_transcript(user_id, max_tokens)` returns the most recent turns (user + AI) as a single string, respecting the transcript budget. CognitiveBrain places this before summaries and retrieved memories in the prompt.

Thresholds & budgets:
- `MemoryQueryRequest.min_relevance_score` default: 0.5 (tunable).
- STM token budget: default 25k tokens (tunable via settings/env).


## Agents and Access Timing

The `OrchestrationService` runs agents in two stages per cycle and provides `user_id` so they can access memory when needed.

- Stage 1 (Foundational):
  - PerceptionAgent, EmotionalAgent, MemoryAgent
  - MemoryAgent explicitly queries memory (STM+LTM) and logs hits.

- Stage 2 (Analytical & Creative):
  - PlanningAgent, CreativeAgent, CriticAgent, DiscoveryAgent
  - These agents may incorporate memory context provided by the orchestration or query patterns/summary as needed.

- CognitiveBrain (final response synthesis):
  - Calls `_get_memory_context()`:
    - Fetches Immediate Transcript: `MemoryService.get_immediate_transcript(user_id, max_tokens)`
    - `SummaryManager.get_or_create_summary(user_id)`
    - `MemoryService.query_memory` with the user’s latest input
  - The prompt includes the Immediate Transcript first, followed by the current summary and relevant memories. This ordering maximizes continuity while keeping high-level context.

- Background engines:
  - DecisionEngine: consumes memory-related signals (stm_pressure, summary_updated, flush_completed, query_metrics) to drive autonomous behaviors.
  - SelfReflection & Discovery Engine: may analyze recent cycles and store `DiscoveredPattern` in LTM.


## In-order Running Log (Transcript)

Yes, the system maintains an in-order log of the conversation in LTM. Each `CognitiveCycle` includes a timestamp and session_id and is persisted on each turn.

How to retrieve:
- Use `MemoryService.list_cycles(request: CycleListRequest)` to fetch all cycles for a user, optionally filter by session/timeframe, and it returns them sorted by timestamp (newest-first) with pagination.
- Use `get_recent_cycles_for_reflection(user_id, limit)` to fetch the most recent pending cycles for reflection, also sorted by time.

STM is always inherently ordered (newest-first) via `ShortTermMemory.recent_cycles`.

Immediate Transcript (in-session):
- For the current process, `MemoryService.get_immediate_transcript(user_id, max_tokens)` returns a rolling, in-order slice of the latest turns. It is not persisted and resets on restart.


## Embeddings and Similarity

- Embedding model: `settings.EMBEDDING_MODEL_NAME` (default: `models/embedding-001`).
- Dimensions: currently 768 (from provider, may vary by model version).
- STM: cosine similarity on per-field vectors (user_input and final_response); best is taken.
- LTM: Chroma distance converted to similarity score in [0..1] via `_distance_to_score`.


## Signals & Autonomy

Memory emits signals to the DecisionEngine when configured:
- `stm_pressure` when token budget nears/exceeds threshold
- `summary_updated` after each summary update
- `flush_completed` after STM flush to LTM
- `query_metrics` after queries (stm_hits, ltm_hits, avg_relevance)

These can trigger background tasks (reflection, discovery, self-assessment, curiosity) depending on policy.


## Configuration

Key settings (src/core/config.py):
- `CHROMA_DB_PATH` (default `./chroma_db`)
- Collections: `CHROMA_COLLECTION_CYCLES`, `CHROMA_COLLECTION_PATTERNS`
- LLM models: `LLM_MODEL_NAME`, `EMBEDDING_MODEL_NAME`, `LLM_MODEL_FOR_RESPONSE_GENERATION`, `LLM_MODEL_FOR_MODERATION`
- Token budgets: `STM_TOKEN_BUDGET`, `TOKEN_RESERVE_RATIO`

Additional:
- `IMMEDIATE_TOKEN_BUDGET` for the Immediate Transcript (default ~50k tokens, adjustable).

Chroma client is shared across services (e.g., `SummaryManager` uses the same client) with anonymized telemetry disabled for reliability.


## Retention & What’s Stored

- STM: Last N cycles (bounded by token budget). Each entry is a full `CognitiveCycle` plus per-field embeddings.
- Immediate Transcript: Last M tokens of verbatim user/AI turns for the current session (ephemeral, in-memory only).
- LTM (cognitive_cycles): Full `CognitiveCycle` JSON, a compact embedding text, and metadata like user_id and timestamp.
- Summary (conversation_summaries): One per user, persisted with embedding and JSON data. Entities are stored as lists and normalized to sets on read.
- Patterns (discovered_patterns): High-level insights with embeddings and metadata.

Nothing is silently dropped; old STM cycles are flushed to LTM when over budget and summaries are updated to retain high-level context.


## Known Limitations & Notes

- If the LLM sometimes returns non-JSON, CognitiveBrain now gracefully degrades by using the raw text and synthesizing minimal metadata (so conversation doesn’t break).
- The default STM recall threshold (0.5) is chosen to favor recall in conversational settings; tune as needed.
- STM is per-process memory; it’s not shared across instances. LTM is the cross-session source of truth.
- Chroma telemetry may log non-fatal PostHog errors internally; anonymized telemetry is disabled on the client we create.
- Immediate Transcript is per-process and ephemeral; it clears on restart and is not a substitute for LTM persistence.


## Future Enhancements (Ideas)

- Field-aware LTM recall: weigh user_input vs final_response differently.
- Richer chunking: break long user inputs or responses into multiple embedding spans.
- Dedupe/merge: automatically merge near-duplicate memories.
- Timeline views: render a first-class transcript UI grouped by sessions and enriched with summary deltas.
- Per-user memory preferences: allow user to configure what to remember and for how long.
- Identity quick cache: small, durable cache of `user_name`, `ai_name`, and `location` for instant prompt injection alongside the Summary.


## FAQ

Q: Do we have an in-order running log of the conversation?
- A: Yes. Every turn is a `CognitiveCycle` persisted to LTM with timestamps and session IDs. Use `MemoryService.list_cycles` to retrieve an ordered transcript for any user (optionally filtered by session/time range).

Q: Which agents see memory?
- A: MemoryAgent always queries memory; CognitiveBrain consults the current summary and top relevant memories; other agents receive `user_id` and can access STM/LTM through the MemoryService as needed. Their analyses are recorded in the `agent_outputs` of each cycle and persist to LTM.

Q: What exactly is retained in LTM?
- A: The full `CognitiveCycle` JSON (inputs, analyses, outputs, metadata, signals), plus an embedding document for vector search. Summaries and patterns are also stored.
