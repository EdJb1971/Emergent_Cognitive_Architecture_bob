# Memory Architecture Enhancement Plan

## Current State
- Long-term vector memory implemented using ChromaDB
- User interactions stored as `CognitiveCycle` objects
- Memory isolation per user
- Vector search for semantic relevance
- Basic metadata filtering

## Proposed Memory Architecture

### 1. Short-Term Memory (STM)
- **Purpose**: Hold recent interactions and context for immediate access
- **Implementation**:
  - In-memory cache using a fixed-size circular buffer per user
  - Store last N interactions (configurable, e.g., N=10)
  - Include full `CognitiveCycle` objects
  - Fast access without vector search overhead

### 2. Summary Memory
- **Purpose**: Maintain condensed representation of conversation context
- **Implementation**:
  - Dynamic summary updated after each interaction
  - Store in ChromaDB with special metadata tag
  - Include:
    - Key topics discussed
    - Important context
    - User preferences/patterns
    - Current conversation state
  - Update strategy: 
    - Incremental updates for efficiency
    - Periodic full regeneration for coherence

### 3. Long-Term Memory (LTM) - Enhanced
- **Purpose**: Persistent storage of all interactions and derived knowledge
- Current Implementation (to keep):
  - ChromaDB vector storage
  - Semantic search capabilities
  - User isolation
- Enhancements:
  - Hierarchical memory organization
  - Improved metadata tagging
  - Pattern storage from self-reflection
  - Memory consolidation from STM

## Implementation Plan

### 1. Code Structure Updates

#### Memory Models (`src/models/core_models.py`)
```python
class ShortTermMemory(BaseModel):
    user_id: UUID
    recent_cycles: List[CognitiveCycle]
    max_size: int = 10

class ConversationSummary(BaseModel):
    user_id: UUID
    summary_text: str
    key_topics: List[str]
    user_preferences: Dict[str, Any]
    last_updated: datetime
    conversation_state: str
```

#### Memory Service Enhancement (`src/services/memory_service.py`)
- Add STM management methods
- Implement summary generation and updates
- Enhance memory consolidation logic

### 2. Database Schema Updates

#### ChromaDB Collections
1. `cycles_collection` (existing)
   - Add new metadata fields for improved organization
   - Add summary linkage

2. `summaries_collection` (new)
   - Store conversation summaries
   - Enable quick context retrieval

### 3. Access Patterns

#### Memory Agent
- Query priority:
  1. Short-term memory (immediate context)
  2. Summary memory (conversation context)
  3. Long-term memory (historical context)

#### Other Agents
- Read access to summaries for context
- Indirect access to LTM via Memory Agent

## Implementation Phases

### Phase 1: Short-Term Memory [IN PROGRESS]
1. ✅ Implement `ShortTermMemory` class
   - Created in `src/models/memory_models.py`
   - Includes circular buffer implementation
   - Added MemoryAccessStats for monitoring
   - Key decision: Max size configurable (default=10)

2. ✅ Add STM management to `MemoryService`
   - Enhanced MemoryService with STM cache
   - Implemented unified query system
   - Added performance monitoring
   - Key decision: STM checked before LTM for better performance

3. ✅ Update Memory Agent to utilize STM
   - Enhanced `process_input` method with STM awareness
   - Added intelligent confidence scoring based on STM hits
   - Implemented memory analysis with recency boost
   - Added comprehensive logging for monitoring
   
   Key Improvements:
   - Dynamic confidence calculation based on STM hits
   - Recency boost for STM matches
   - Enhanced memory analysis with combined context
   - Better debugging through detailed logging

4. ✅ Add tests for STM functionality
   - Created `test_memory_service_stm.py` for memory service testing
   - Created `test_memory_agent_enhanced.py` for agent testing
   - Coverage for initialization, cycle management
   - Tests for unified memory queries
   - Includes error handling verification
   - Added confidence calculation testing
   - Added STM integration testing

Key Learnings:
- STM implementation provides significant performance boost for recent queries
- Unified query system ensures consistent results across memory types
- Access statistics will help optimize memory management
- Error handling crucial for maintaining system stability

Next Steps:
1. Update Memory Agent implementation
2. Add memory statistics endpoints
3. Implement memory cleanup strategies
4. Add configuration options for STM size

### Phase 2: Summary Memory [IN PROGRESS]
1. ✅ Implement summary models and logic
   - Created `ConversationSummary` model
   - Implemented incremental updates
   - Added embedding support for semantic search
   - Key decision: Summary includes topics, entities, preferences

2. ✅ Create Summary Management Service
   - Implemented `SummaryManager` class
   - Added ChromaDB integration for persistence
   - Created LLM-based analysis system
   - Key decision: Real-time updates with LLM analysis

3. ✅ Add Comprehensive Testing
   - Created `test_summary_manager.py`
   - Coverage for creation, updates, storage
   - Tests for multi-user scenarios
   - Includes error handling verification

4. 🔄 Integration Steps [IN PROGRESS]
   - ✅ Enhanced Memory Agent
     * Added summary context to memory analysis
     * Improved context relevance with summary data
     * Combined memory and summary insights
     * Key decision: Enhanced context structure with types

   - ✅ Memory Service Integration
     * Added SummaryManager initialization
     * Connected summary storage system
     * Enhanced memory queries with summary context
     * Key decision: 20% relevance boost for summary-referenced memories

   - ⏳ Remaining Tasks
     * Implement cleanup strategies
     * Add monitoring endpoints
     * Add summary decay mechanism
     * Create summary analytics API

Key Learnings:
- Summary context significantly improves memory relevance
- Typed context items enable better downstream processing
- Combined STM + Summary + LTM provides rich context
- Memory boost for summary-referenced items improves recall

Next Steps:
1. Implement memory cleanup and decay
2. Add summary compression for long conversations
3. Create monitoring dashboard
4. Add summary analytics endpoints

Key Learnings:
- LLM-based analysis provides rich context extraction
- Incremental updates maintain summary coherence
- User isolation critical for multi-user support
- Vector search enables contextual summary retrieval
- Robust error handling needed for summary integration
- Memory cleanup essential for long-running systems

Integration Testing Added:
- Memory service initialization
- Summary-enhanced querying
- Error handling and recovery
- Statistics tracking
- Memory cleanup scenarios

Next Steps:
1. Implement Memory Cleanup [NEXT]
   - Add TTL for STM entries
   - Implement periodic cleanup
   - Add configuration options
   - Monitor memory usage

2. Add Monitoring System
   - Memory usage statistics
   - Query performance metrics
   - Summary update frequency
   - Error rate tracking

3. Optimize Performance
   - Cache frequently accessed summaries
   - Batch summary updates
   - Implement summary compression
   - Add query result caching

4. Create Analytics API
   - Memory usage patterns
   - Topic evolution tracking
   - User interaction insights
   - System performance metrics

### Phase 3: Enhanced Long-Term Memory
1. Implement hierarchical organization
2. Enhance metadata system
3. Add memory consolidation logic
4. Optimize query patterns

### Phase 4: Integration
### Phase 4: Integration (fully integrate STM, Summary, LTM)

Goal: Ensure STM, Summary Memory, and LTM work as a coherent system where short-term context is token-budgeted, summaries are generated before long-term upserts, and the system gracefully shifts information from fast transient storage into durable, searchable knowledge.

Core design decision (token-limited STM):
- STM should be limited by tokens rather than time. That is, maintain a per-user token budget for immediate context (configurable, e.g. 1,500 tokens by default) so that when the total context tokens approach the budget we summarize and consolidate older STM content.
- Reason: Token-limited STM directly maps to LLM prompt constraints and forces explicit trade-offs between immediate context and model input capacity. It also mirrors human short-term memory capacity and makes it easier to leave headroom for system prompts, tool outputs, and other context.

Model-specific guidance (Gemini / large-token LLMs):
- If you use Gemini (250k token input limit) you can safely set a much larger STM_TOKEN_BUDGET (for example 25k–50k tokens) to keep more conversational context in working memory. Choose a budget that leaves sufficient reserved tokens for system prompts, tool outputs, and expected response length.
- Implement an adaptive reserve: keep a token_reserve (e.g., 10–20% of model limit) so the system never exhausts the model input window when composing prompts.
- Add robust retry/backoff behavior for LLM calls: treat 429 / rate-limit responses as transient and implement exponential backoff with jitter and a capped number of retries. On persistent 429s, fall back to smaller summarization windows (temporarily reduce STM flush size) and surface a metric/alert.
- Cost and latency: larger STM budgets increase LLM cost and latency for summarization operations. Consider tiered summarization (compressive summarization + light-touch incremental updates) to reduce LLM calls while preserving context.

Suggested flow (per new cognitive cycle):
1. New cycle appended to in-memory STM (ShortTermMemory.add_cycle).
2. Update STM token usage (token_count_current += cycle_token_count).
3. If token_count_current <= STM_TOKEN_BUDGET: continue (no forced summarization).
4. If token_count_current > STM_TOKEN_BUDGET (or token_reserve threshold for LLM input):
   a. Trigger summary generation for oldest portion of STM (SummaryManager.summarize_stm or incremental summarization). Summary must run before flushing to LTM.
   b. Create/append the generated summary to the `summaries_collection` with proper metadata (user_id, topics, timestamp).
   c. Upsert a consolidated STM document into LTM (single per-user upserted record such as `stm_consolidated_{user_id}`) that contains the condensed recent context (or its embedding) to support retrieval without pushing the full token set into the prompt.
   d. Remove/flushed cycles from in-memory STM (e.g., drop the oldest cycles that were summarized), recompute token_count_current.

Notes on ordering: summary happens before upsert to LTM. The summary is the source-of-truth for what gets persisted from STM into LTM. The upsert should contain the summary text and/or its embedding and a pointer to the original cycles stored elsewhere if needed for provenance.

Data model and storage suggestions:
- Keep an in-memory `ShortTermMemory` per user that tracks cycles and token usage, and exposes `peek`, `add_cycle`, `get_token_count`, and `flush_oldest(n_tokens)` methods.
- Add a per-user single upserted LTM document id (e.g. `stm_consolidated:{user_id}`) stored in `cycles_collection` or a separate `stm_collection` to hold the condensed/embedded STM snapshot. Always update (upsert) this record rather than appending to avoid unbounded growth.
- Summaries stored in `summaries_collection` should be independent documents with their own embeddings and metadata for search and provenance.

Technical contract (minimal):
- Inputs: CognitiveCycle objects (with text fields used for embedding), per-user STM token budget and reserve.
- Outputs: Updated in-memory STM, zero-or-more summary documents (persisted), updated `stm_consolidated` upsert entry in LTM, and metrics/audit log entries describing why a flush happened.
- Error modes: If summary generation fails, keep STM in-memory and enter a degraded mode (retry with backoff, log audit). If upsert fails, keep the summary in a retry queue and continue operating.

STM persistence & recovery:
- On graceful shutdown: serialize STM state to disk (JSON/pickle) with metadata (user_id, timestamp, token counts).
  - Store in configurable location (e.g., `{data_dir}/stm_snapshots/{user_id}_stm.json`).
  - Include: cognitive cycles, token counts, last summary timestamp, and other state needed for full recovery.
- On startup: attempt to restore STM from disk snapshot if available and valid.
  - Validate snapshot age (reject if too old, configurable max age).
  - Verify token counts and cycle integrity.
  - If snapshot corrupt/invalid: fall back to empty STM, log warning.
- Recovery flow:
  1. Load STM snapshot from disk.
  2. Validate cycles and recompute token counts.
  3. If token budget exceeded: trigger immediate summarization.
  4. Update `stm_consolidated` with current state.
  5. Resume normal operation.
- Periodic snapshots (optional):
  - Save STM state periodically (e.g., every N minutes or after M cycles).
  - Keep last N snapshots for rollback if needed.
  - Useful for crash recovery or debugging.

Edge cases & operational concerns:
- Token estimation: Use the LLMIntegrationService tokenizer (or tiktoken/compatible util) to get reliable token counts. If not available, fall back to conservative character-based estimates.
- Concurrency: Use a per-user lock (async lock) around STM mutations and flushes to avoid races when multiple agents/processes operate on the same user's STM.
- Partial flushes: Prefer incremental summarization (flush oldest cycles until token usage under budget) rather than dropping everything at once.
- Provenance: Keep mapping from summary documents to the original cycle ids (or a compact provenance list) to enable audits and potential replay.
- Hysteresis/backoff: Avoid constant summarization by using a reserve threshold (e.g., trigger when token usage > 90% of budget, aim to reduce to 50–70%).
- Safety: Rate-limit summarization and upsert operations to avoid overwhelming LLM or ChromaDB.

Testing and validation:
- Unit tests for token accounting and STM.add_cycle behavior (including threshold crossing and flush decisions).
- Integration tests that simulate sequences of cycles, confirm summary creation, confirm `stm_consolidated` upsert semantics, and confirm retrieval behavior uses the summary as expected.
- Fault injection tests where summary or upsert fails and the system retries or degrades gracefully.

Implementation checklist (phase tasks):
1. Add token-counting util and expose via LLMIntegrationService or a small helper (e.g., `tokenizer.count_tokens(text)`).
2. Update `ShortTermMemory` to be token-aware and to expose flush helpers and token metrics.
3. Update `MemoryService` to:
   - Maintain per-user STM token budgets in config.
   - Call SummaryManager to summarize when STM exceeds budget.
   - Upsert `stm_consolidated:{user_id}` into LTM with the generated summary (embedding + metadata).
   - Use per-user async locking for STM flush path.
4. Ensure `SummaryManager` exposes `summarize_stm(user_id, cycles_to_summarize)` that returns summary text + key topics and optionally an embedding.
5. Add metrics and audit logs for each flush and upsert (why triggered, token counts before/after, latency, success/fail).
6. Add unit and integration tests.

Why this is sensible:
- Token-limited STM directly aligns memory management with LLM input constraints. It makes prompt-construction predictable and helps prevent accidental context truncation at call time.
- Summarize-before-upsert preserves semantic fidelity: the summary captures the salient information that should be carried forward into persistent memory.
- Upserting a single consolidated STM document prevents LTM growth from frequent small writes and maintains a single fast-access record representing recent context.

Follow-ups (after Phase 4 is implemented):
- Tune token budgets per-user based on observed usage and model constraints.
- Explore compressive summarization strategies (e.g., abstractive compression with topic preservation).
- Add tooling to inspect `stm_consolidated` records, summarize history, and tune policies.

## Current Implementation Progress

### ✅ Memory Foundation (Nov 2025)
1. Token-aware STM
   - Implemented `TokenCounter` utility with Gemini support (250k context)
   - Added fallback counting and caching for reliability
   - Built-in retry logic for API failures
   - Unit tests for all token counting scenarios

2. Enhanced `ShortTermMemory`
   - Token budget enforcement (configurable, 25k–50k for Gemini)
   - Async-safe operations with proper locking
   - Automatic summary triggering on budget exceeded
   - Token-aware cycle management
   - Persistence/recovery system
     * Save/load snapshots (JSON/pickle)
     * Age validation and corruption handling
     * Per-user snapshot management
   - Comprehensive test coverage

### ✅ Summary Integration (Nov 2025)
1. SummaryManager Implementation
   - Added `SummaryManager.summarize_stm()` method
   - Implemented ChromaDB upsert for summaries and consolidated STM
   - Created summary-before-flush workflow
   - Added comprehensive integration tests
   - Implemented proper error handling and logging
   - Added embedding generation and storage
   - Created semantic search capabilities

### ✅ STM/Summary/LTM Integration (Nov 2025)
1. MemoryService and SummaryManager integration complete
   - STM token budget enforcement and automatic flush workflow implemented
   - SummaryManager.summarize_stm generates summaries and upserts consolidated STM before LTM flush
   - Cycles flushed from STM are persisted to LTM after summarization
   - Per-user async locks ensure concurrency safety
   - Metrics and logging for flush events added
   - Comprehensive integration tests for STM flush, summary generation, and LTM upsert

2. Core Model Integration
   - CognitiveBrain uses integrated memory context (STM, summary, LTM) for response generation
   - Outcome signals adjusted based on memory performance
   - Error handling and safety checks validated

3. ⏳ Agent Integration [IN PROGRESS]
   - Update Memory Agent to leverage enhanced flush workflow
   - Test cross-agent interactions and memory context usage
   - Add performance monitoring

### ✅ Agent Integration Progress (Nov 2025)
- MemoryAgent fully integrated with STM/Summary/LTM context and confidence logic
- CriticAgent now retrieves STM, summary, and LTM context for critical analysis
- CriticAgent prompt enhanced with memory context and agent outputs
- All errors and metrics logged for agent-memory interactions
- CreativeAgent now retrieves STM, summary, and LTM context for creative analysis
- CreativeAgent prompt enhanced with memory context and agent outputs
- All errors and metrics logged for agent-memory interactions
- PerceptionAgent, PlanningAgent, EmotionalAgent, and DiscoveryAgent now fully integrated with STM/Summary/LTM context
- All agent prompts enhanced with memory context and agent outputs
- Unified memory access pattern established for all agents
- Next: Finalize cross-agent testing and monitoring

### ⏳ Next Steps
1. Finalize Agent Integration
   - Ensure all agents use STM/summary/LTM context
   - Add agent-level metrics and monitoring
   - Complete cross-agent test coverage

2. Enhance Analytics and Monitoring
   - Expand metrics for flush, summarization, and upsert events
   - Create dashboards for memory usage and performance
   - Add alerting for anomalies and failures

3. Autonomous Triggering
   - Implement Decision Engine for automatic reflection/discovery/self-assessment
   - Wire up signal collection and safe trigger policies
   - Add audit logging and monitoring for autonomous events

## Benefits
1. Faster access to recent context
2. Better conversation coherence
3. Reduced vector search overhead
4. Improved scalability
5. More natural memory hierarchy

## Technical Considerations
1. Memory Management
   - Token-aware budget enforcement
   - Automatic cleanup triggers
   - Persistence and recovery strategies
   - Summary compression techniques

2. Performance Optimization
   - Concurrent access patterns with proper locking
   - Smart caching of frequently accessed summaries
   - Efficient memory context retrieval
   - Vector database query optimization

3. Integration Patterns
   - Memory-aware response generation
   - Context prioritization (STM → Summary → LTM)
   - Memory performance monitoring
   - Safety and moderation checks

4. Error Handling
   - Graceful degradation on service failures
   - Robust retry mechanisms
   - Comprehensive logging and monitoring
   - Data consistency preservation

5. Scalability Concerns
   - Per-user memory isolation
   - Resource usage monitoring
   - Rate limiting and throttling
   - Background task management

## Next Steps
1. Complete Memory Agent Integration
   - Update Memory Agent to leverage enhanced memory context
   - Implement cross-agent communication patterns
   - Add comprehensive agent testing

2. Enhance Monitoring System
   - Create memory performance dashboard
   - Add detailed memory access metrics
   - Implement alerting for anomalies
   - Track memory-enhanced response quality

3. Optimize Performance
   - Implement smart caching strategies
   - Add batch processing for summaries
   - Optimize vector search patterns
   - Fine-tune token budgets

4. Add Analytics Capabilities
   - Create memory usage analytics
   - Track context enhancement metrics
   - Monitor user satisfaction signals
   - Analyze memory performance impact

5. Plan Production Rollout
   - Create gradual deployment strategy
   - Set up A/B testing framework
   - Define success metrics
   - Prepare rollback procedures

## Autonomous Triggering — Implementation Plan

### Overview
Autonomous triggering enables the system to automatically initiate reflection, discovery, self-assessment, and curiosity exploration based on real-time signals and configurable policies. This ensures adaptive, self-improving behavior and robust context management.

### Key Components
1. **Decision Engine**
   - Monitors signals from MemoryService, SummaryManager, agent outcomes, and system metrics.
   - Evaluates rules and policies to decide when to trigger autonomous workflows.
   - Supports configurable global and per-user policies (reflection interval, discovery thresholds, cooldowns).

2. **Signal Emitters & Metrics**
   - Expose memory access stats (STM hits, misses, relevance, flushes).
   - Track agent outcome signals (confidence, satisfaction, engagement, failure rates).
   - Monitor summary coverage (topics, update frequency, entropy).
   - Push metrics to Decision Engine and monitoring dashboards.

3. **Trigger Policies**
   - Reflection: Trigger after N cycles (configurable, e.g., N=10).
   - Discovery: Trigger when outcome signals are low or repeated unknown intents.
   - Self-assessment: Trigger periodically (e.g., every 24h or after K interactions).
   - Curiosity: Trigger when knowledge gaps or summary coverage drops below threshold.
   - Rate-limiting and cooldown windows to prevent over-triggering.

4. **Task Scheduling & Execution**
   - Integrate with BackgroundTaskQueue and OrchestrationService for safe, asynchronous execution.
   - De-duplicate tasks and enforce cooldowns.
   - Audit log all autonomous triggers (reason, metrics snapshot, user_id, triggered task id).

5. **Observability & Safety**
   - Export metrics and audit logs for all triggers.
   - Alert on unusual trigger rates or failures.
   - Unit and integration tests for all trigger rules and degraded modes.

### Implementation Steps
1. ✅ Design and implement the Decision Engine module:
   - Completed: a rule-based `DecisionEngine` implemented in `src/services/decision_engine.py`
   - Supports configurable policies, per-user cooldowns, and an `ingest_signals(user_id, signals)` API
   - Enhanced evaluators to use memory performance metrics and query patterns
   - Added configurable thresholds for all trigger conditions

2. ✅ Integrate signal emitters in MemoryService:
   - Added four key signal types:
     * `stm_pressure`: Memory pressure and budget tracking
     * `summary_updated`: Summary generation and updates
     * `flush_completed`: Memory consolidation events
     * `query_metrics`: Performance and relevance metrics
   - Enhanced with comprehensive metrics collection
   - Added proper error handling and retry logic

3. ✅ Implement trigger policies:
   - Enhanced reflection triggers:
     * Cycle-based triggers (default: 10 cycles)
     * Memory flush triggers for consolidation review
     * Performance degradation triggers
   - Enhanced discovery triggers:
     * Knowledge gap detection from query metrics
     * Low relevance score patterns
     * High miss rate patterns
   - Enhanced self-assessment:
     * System performance monitoring
     * Error rate tracking
     * Manual override support
   - Enhanced curiosity triggers:
     * Topic entropy analysis
     * Novel query pattern detection
     * Coverage gap identification

4. ✅ Wire up task scheduling and execution:
   - Enhanced BackgroundTaskQueue with proper task routing:
     * Added OrchestrationService integration
     * Implemented task-specific handlers
     * Added comprehensive error handling
   - Implemented four autonomous task types:
     * `autonomous:reflection` → Cycle reflection (10 cycles)
     * `autonomous:discovery` → Knowledge gap exploration
     * `autonomous:self_assess` → Deep reflection (20 cycles)
     * `autonomous:curiosity` → Exploratory discovery
   - Added proper task deduplication and cooldowns
   - Enhanced error handling and retry logic

5. ⏳ Add observability and safety features:
   - Next steps:
     * Add comprehensive integration tests
     * Implement audit logging system
     * Add Prometheus metrics export
     * Create monitoring dashboards
   - Planned test coverage:
     * Signal emission and propagation
     * Decision Engine evaluation accuracy
     * Task routing and execution
     * Error handling and recovery
     * End-to-end autonomous flows

### Short-Term Roadmap
- [x] Decision Engine module (rule-based, config-driven)
- [x] Signal emitters (initial: STM pressure, summary updates)
- [x] DecisionEngine unit tests
- [x] Basic task scheduling wiring (BackgroundTaskQueue.enqueue shim + startup wiring)
- [x] Add additional signal emitters (`flush_completed`, `query_metrics`, `stm_hits`, `avg_relevance`)
- [x] Enhance Decision Engine evaluators to use new signals
- [x] Route enqueued `autonomous:*` tasks to OrchestrationService handlers:
  * `autonomous:reflection` -> trigger_reflection (10 cycles)
  * `autonomous:discovery` -> trigger_discovery with knowledge gaps
  * `autonomous:self_assess` -> deeper reflection (20 cycles)
  * `autonomous:curiosity` -> discovery with exploration context
- [ ] Add integration tests for the complete autonomous cycle:
  1. Signal emission
  2. Decision Engine evaluation
  3. Task enqueuing
  4. Handler execution
  5. Outcome verification
- [ ] Add audit logging and metrics export for autonomous events

---

## Next Steps (Cleaned Up)
1. Finalize Autonomous Triggering
   - Implement Decision Engine and signal emitters
   - Integrate with BackgroundTaskQueue and OrchestrationService
   - Add audit logging, metrics, and monitoring
   - Complete unit and integration tests for triggers
2. Monitor and Tune System
   - Track autonomous trigger rates and system performance
   - Tune policies and thresholds based on observed behavior
   - Expand analytics and dashboards as needed
3. Prepare for Production Rollout
   - Validate end-to-end workflows
   - Document autonomous triggering logic and configuration
   - Plan gradual deployment and rollback procedures