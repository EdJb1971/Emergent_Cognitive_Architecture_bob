import logging
import json
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Union
from uuid import UUID
from datetime import datetime
import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.utils import embedding_functions

from src.utils.token_counter import TokenCounter

from src.core.config import settings
from src.core.exceptions import ConfigurationError, APIException
from src.models.core_models import CognitiveCycle, MemoryQueryRequest, DiscoveredPattern, CycleListRequest
from src.models.memory_models import ConversationSummary, ShortTermMemory, MemoryAccessStats
from src.services.llm_integration_service import LLMIntegrationService
from src.services.summary_manager import SummaryManager
from src.services.metrics_service import MetricsService, MetricType
from src.agents.utils import UUIDEncoder

logger = logging.getLogger(__name__)

class RollingTranscript:
    """A rolling, token-budgeted transcript of recent conversation turns."""
    def __init__(self, token_counter: "TokenCounter", max_tokens: int = 50000):
        from collections import deque
        self._deque = deque()  # items: {role, text, tokens, cycle_id, timestamp}
        self.token_counter = token_counter
        self.max_tokens = max_tokens
        self.total_tokens = 0
        self.pruned_utterances = 0

    def _estimate_tokens(self, text: str) -> int:
        try:
            # Prefer project token counter if available
            return int(self.token_counter.count_tokens(text))
        except Exception:
            # Fallback rough heuristic: 1 token ≈ 4 chars
            return max(1, len(text) // 4)

    def add(self, role: str, text: str, cycle_id: str, timestamp: datetime):
        if not text:
            return
        tokens = self._estimate_tokens(text)
        entry = {
            "role": role,
            "text": text,
            "tokens": tokens,
            "cycle_id": str(cycle_id),
            "timestamp": timestamp.isoformat(),
        }
        self._deque.append(entry)
        self.total_tokens += tokens
        # Prune from the left until under budget
        while self.total_tokens > self.max_tokens and self._deque:
            removed = self._deque.popleft()
            self.total_tokens -= int(removed.get("tokens", 0))
            self.pruned_utterances += 1

    def get_context(self, max_tokens: int) -> str:
        """Return the most recent transcript slice up to max_tokens, oldest-to-newest order."""
        if not self._deque:
            return ""
        acc = []
        used = 0
        # iterate from right (newest) backwards, then reverse to chronological
        for entry in reversed(self._deque):
            t = int(entry.get("tokens", 0))
            if used + t > max_tokens:
                break
            prefix = "User:" if entry["role"] == "user" else "Assistant:"
            acc.append(f"{prefix} {entry['text']}")
            used += t
        acc.reverse()
        return "\n".join(acc)

class MemoryService:
    """
    Manages all interactions with a local ChromaDB vector database.
    Responsible for generating embeddings, storing complete cognitive cycles with rich metadata,
    and providing various querying capabilities while enforcing strict user isolation.
    """
    def __init__(self, llm_service: LLMIntegrationService, metrics_service: Optional[MetricsService] = None):
        self.llm_service = llm_service
        self.metrics_service = metrics_service
        self.client: Optional[chromadb.Client] = None
        self.cycles_collection: Optional[chromadb.Collection] = None
        self.patterns_collection: Optional[chromadb.Collection] = None
        
        # Memory components
        self._stm_cache: Dict[UUID, ShortTermMemory] = {}
        self._access_stats: Dict[UUID, MemoryAccessStats] = {}
        self.summary_manager = SummaryManager(llm_service)
        self.token_counter = TokenCounter()
        # Per-user locks for STM operations
        # Optional DecisionEngine wiring (set after app startup to avoid import cycles)
        self.decision_engine = None
        
        # Per-user locks for STM operations
        self._stm_locks: Dict[UUID, asyncio.Lock] = {}
        
        # Token budget configuration
        # Use getattr to avoid AttributeError when settings don't define these (e.g., in tests)
        self.STM_TOKEN_BUDGET = getattr(settings, 'STM_TOKEN_BUDGET', 25000)
        self.TOKEN_RESERVE_RATIO = getattr(settings, 'TOKEN_RESERVE_RATIO', 0.2)
        # Immediate transcript token budget (large window leveraged by Gemini)
        self.IMMEDIATE_TOKEN_BUDGET = getattr(settings, 'IMMEDIATE_TOKEN_BUDGET', 50000)
        # Rolling transcript buffer per user
        self._transcripts: Dict[UUID, RollingTranscript] = {}
        
        logger.info("Enhanced MemoryService initialized with STM and Summary support.")

    async def connect(self):
        """
        Establishes connections to ChromaDB for both LTM and summary storage.
        """
        try:
            # Initialize LTM storage
            try:
                # Disable Chroma telemetry to avoid posthog capture signature mismatches/noise
                self.client = chromadb.PersistentClient(
                    path=settings.CHROMA_DB_PATH,
                    settings=ChromaSettings(
                        anonymized_telemetry=False,
                        allow_reset=True,
                        is_persistent=True
                    )
                )
                self.cycles_collection = self.client.get_or_create_collection(
                    name=settings.CHROMA_COLLECTION_CYCLES,
                    embedding_function=embedding_functions.DefaultEmbeddingFunction()
                )
                self.patterns_collection = self.client.get_or_create_collection(
                    name=settings.CHROMA_COLLECTION_PATTERNS
                )
                logger.info("Successfully connected to ChromaDB for LTM storage.")
            except chromadb.errors.ChromaDBError as e:
                logger.error(f"Failed to initialize ChromaDB: {e}", exc_info=True)
                raise ConfigurationError(detail=f"Failed to initialize long-term memory storage: {e}", status_code=503)
            except Exception as e:
                logger.error(f"Unexpected error initializing ChromaDB: {e}", exc_info=True)
                raise APIException(detail=f"Unexpected error initializing long-term memory storage: {e}", status_code=503)
            
            try:
                # Initialize summary storage, reusing the same Chroma client
                await self.summary_manager.connect(client=self.client)
                logger.info("Successfully connected summary storage.")
            except Exception as e:
                logger.error(f"Failed to initialize summary storage: {e}", exc_info=True)
                # Continue without summary storage - it can be initialized later
        except Exception as e:
            logger.critical(f"Memory system initialization failed: {e}", exc_info=True)
            raise ConfigurationError(detail=f"Failed to initialize memory system: {e}")

    async def close(self):
        """
        Closes the ChromaDB connection.
        """
        # ChromaDB PersistentClient doesn't require explicit close
        logger.info("ChromaDB connection managed automatically.")
        pass

    async def get_stm(self, user_id: Union[UUID, str]) -> Optional[ShortTermMemory]:
        """Return the ShortTermMemory object for a user if present.

        If STM isn't initialized in-memory yet, attempt to lazily load a snapshot
        from disk (if available) and cache it.

        Args:
            user_id: UUID or string UUID

        Returns:
            Optional[ShortTermMemory]: The STM instance or None if unavailable.
        """
        try:
            # Normalize user_id
            if isinstance(user_id, str):
                user_id = UUID(user_id)

            # Fast path: in-memory STM cache
            stm = self._stm_cache.get(user_id)
            if stm:
                return stm

            # Attempt to load snapshot for this user lazily
            loaded = ShortTermMemory.load_snapshot(user_id)
            if loaded:
                self._stm_cache[user_id] = loaded
                return loaded
            return None
        except Exception as e:
            logger.debug(f"get_stm failed for user {user_id}: {e}")
            return None

    async def upsert_cycle(self, cognitive_cycle: CognitiveCycle) -> bool:
        """
        Stores a cognitive cycle in STM, LTM, and updates the conversation summary.
        """
        if not cognitive_cycle.cycle_id or not cognitive_cycle.user_id:
            raise APIException(detail="cycle_id and user_id are required to store a cycle.", status_code=400)

        try:
            # Precompute per-field embeddings for improved STM recall
            try:
                if getattr(cognitive_cycle, 'user_input', None) and not getattr(cognitive_cycle, 'user_input_embedding', None):
                    logger.info(f"Computing user_input_embedding for cycle {cognitive_cycle.cycle_id}")
                    cognitive_cycle.user_input_embedding = await self.llm_service.generate_embedding(
                        text=cognitive_cycle.user_input,
                        model_name=settings.EMBEDDING_MODEL_NAME
                    )
                    logger.info(f"Generated user_input_embedding with {len(cognitive_cycle.user_input_embedding)} dimensions")
                if getattr(cognitive_cycle, 'final_response', None) and not getattr(cognitive_cycle, 'final_response_embedding', None):
                    logger.info(f"Computing final_response_embedding for cycle {cognitive_cycle.cycle_id}")
                    cognitive_cycle.final_response_embedding = await self.llm_service.generate_embedding(
                        text=cognitive_cycle.final_response,
                        model_name=settings.EMBEDDING_MODEL_NAME
                    )
                    logger.info(f"Generated final_response_embedding with {len(cognitive_cycle.final_response_embedding)} dimensions")
            except Exception as e:
                # Non-fatal: continue without per-field embeddings
                logger.warning(f"Failed to generate per-field embeddings for cycle {cognitive_cycle.cycle_id}: {e}")
                pass

            # 1) Add to STM and determine if flush is needed
            should_flush, cycles_to_flush = await self.add_cycle(cognitive_cycle)
            logger.debug(f"Added cycle to STM for user {cognitive_cycle.user_id} (should_flush={should_flush})")

            # 2) Update conversation summary
            try:
                await self.summary_manager.update_summary(cognitive_cycle.user_id, cognitive_cycle)
                logger.info(f"Updated conversation summary for user {cognitive_cycle.user_id}")
                if self.decision_engine:
                    signals = {
                        "event": "summary_updated",
                        "user_id": str(cognitive_cycle.user_id),
                        "cycle_id": str(cognitive_cycle.cycle_id),
                        "timestamp": cognitive_cycle.timestamp.isoformat(),
                    }
                    await self._emit_signals(cognitive_cycle.user_id, signals)
            except Exception as e:
                logger.error(f"Summary update failed for user {cognitive_cycle.user_id}: {e}", exc_info=True)

            # 3) Store the new cycle in LTM
            await self._store_cycle(cognitive_cycle)

            # 3.5) Append to immediate transcript buffer (verbatim rolling window)
            try:
                rt = self._transcripts.get(cognitive_cycle.user_id)
                if not rt:
                    rt = RollingTranscript(self.token_counter, max_tokens=self.IMMEDIATE_TOKEN_BUDGET)
                    self._transcripts[cognitive_cycle.user_id] = rt
                # Add user and assistant turns
                if getattr(cognitive_cycle, 'user_input', None):
                    rt.add("user", cognitive_cycle.user_input, str(cognitive_cycle.cycle_id), cognitive_cycle.timestamp)
                if getattr(cognitive_cycle, 'final_response', None):
                    rt.add("assistant", cognitive_cycle.final_response, str(cognitive_cycle.cycle_id), cognitive_cycle.timestamp)
                logger.info(
                    f"Immediate transcript updated for user {cognitive_cycle.user_id}: "
                    f"~{rt.total_tokens} tokens, pruned_utterances={rt.pruned_utterances}"
                )
            except Exception as e:
                logger.warning(f"Failed updating immediate transcript: {e}")

            # 4) If STM over budget, summarize and flush old cycles to LTM
            if should_flush and cycles_to_flush:
                await self.flush_to_ltm(cognitive_cycle.user_id, cycles_to_flush)
            return True
        except Exception as e:
                logger.error(f"Unexpected error storing cycle for user {cognitive_cycle.user_id}: {e}", exc_info=True)

    def _get_stm_lock(self, user_id: UUID) -> asyncio.Lock:
        """Get or create a lock for STM operations for a user."""
        if user_id not in self._stm_locks:
            self._stm_locks[user_id] = asyncio.Lock()
        return self._stm_locks[user_id]

    async def add_cycle(self, cognitive_cycle: CognitiveCycle) -> Tuple[bool, Optional[List[CognitiveCycle]]]:
        """
        Add a cognitive cycle to STM and determine if summarization/flush is needed.
        Returns: (should_flush, cycles_to_flush)
        """
        user_id = cognitive_cycle.user_id
        
        # Get user's STM lock
        async with self._get_stm_lock(user_id):
            # Initialize or get STM for user
            if user_id not in self._stm_cache:
                self._stm_cache[user_id] = ShortTermMemory(
                    user_id=user_id,
                    token_budget=self.STM_TOKEN_BUDGET
                )
            
            stm = self._stm_cache[user_id]
            
            logger.info(f"Adding cycle {cognitive_cycle.cycle_id} to STM for user {user_id}. Current STM size: {len(stm.recent_cycles)}")

            # Delegate token-budget logic to STM; it returns whether a summary/flush is needed
            needs_summary, cycles_to_summarize = await stm.add_cycle(cognitive_cycle)

            if needs_summary:
                # Emit a signal indicating STM pressure and suggested cycles to flush
                try:
                    if self.decision_engine:
                        signals = {
                            "event": "stm_pressure",
                            "user_id": str(user_id),
                            "cycles_to_flush": len(cycles_to_summarize) if cycles_to_summarize else 0,
                            "current_token_count": getattr(stm, 'token_count', 0),
                        }
                        await self._emit_signals(user_id, signals)
                except Exception:
                    logger.debug("DecisionEngine STM pressure signal failed or not configured.")

                return True, cycles_to_summarize
            else:
                return False, None

    def set_decision_engine(self, decision_engine: object):
        """Attach a DecisionEngine instance to receive memory signals.

        Kept loosely typed to avoid circular import issues at module import time.
        """
        self.decision_engine = decision_engine

    async def _emit_signals(self, user_id: UUID, signals: Dict[str, Any]):
        """Internal helper to forward signals to the DecisionEngine if configured."""
        try:
            if self.decision_engine:
                # decision_engine.ingest_signals expects user_id as str
                await self.decision_engine.ingest_signals(str(user_id), signals)
        except Exception as e:
            logger.exception(f"Failed to emit signals to DecisionEngine for user {user_id}: {e}")

    async def flush_to_ltm(self, user_id: UUID, cycles: List[CognitiveCycle]):
        """
        Flush cycles from STM to LTM, generating a summary first.
        """
        try:
            # Generate summary before flushing
            summary, consolidation = await self.summary_manager.summarize_stm(
                user_id=user_id,
                cycles=cycles
            )
            
            # Store cycles in LTM
            for cycle in cycles:
                await self._store_cycle(cycle)
            # Remove flushed cycles from STM
            if user_id in self._stm_cache:
                try:
                    await self._stm_cache[user_id].flush_cycles(cycles)
                except Exception:
                    logger.debug("Failed to flush cycles from STM; continuing.")
            
            logger.info(f"Successfully flushed {len(cycles)} cycles to LTM for user {user_id}")
            
            # Emit flush_completed signal with key metrics
            try:
                if self.decision_engine:
                    signals = {
                        "event": "flush_completed",
                        "user_id": str(user_id),
                        "cycles_flushed": len(cycles),
                        "summary_length": len(consolidation),
                        "topics": summary.key_topics if hasattr(summary, 'key_topics') else [],
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    await self._emit_signals(user_id, signals)
            except Exception:
                logger.debug("DecisionEngine flush_completed signal failed or not configured.")
            
        except Exception as e:
            logger.error(f"Error during STM flush for user {user_id}: {e}", exc_info=True)
            raise APIException(detail=f"Failed to flush STM to LTM: {e}", status_code=500)
            
    def _distance_to_score(self, distance: float) -> float:
        """
        Convert ChromaDB distance to a similarity-like score in [0,1] where higher is better.
        Behavior depends on the metric used by ChromaDB. This implements:
        - if distance in [0,1]: score = 1 - distance (common for normalized distances)
        - otherwise: fallback to 1 / (1 + distance) to produce a bounded [0,1] value.
        """
        try:
            if 0.0 <= distance <= 1.0:
                return 1.0 - distance
            # fallback normalization
            return 1.0 / (1.0 + distance)
        except Exception:
            return 0.0

    async def query_memory(self, query_request: MemoryQueryRequest) -> List[CognitiveCycle]:
        """
        Queries memory using vector search for semantic relevance and optional metadata filters.
        Checks STM first, then queries LTM.
        """
        if not self.cycles_collection:
            raise APIException(detail="MemoryService not connected to ChromaDB.", status_code=503)

        try:
            # Compute query embedding once (used for both STM and LTM)
            query_embedding = query_request.query_embedding or await self.llm_service.generate_embedding(
                text=query_request.query_text,
                model_name=settings.EMBEDDING_MODEL_NAME
            )

            # Check STM first (embedding-based similarity when available)
            stm_results: List[CognitiveCycle] = []
            stm = self._stm_cache.get(query_request.user_id)
            if stm:
                cycles = stm.get_recent_cycles()
                logger.info(f"STM query: checking {len(cycles)} cycles for user {query_request.user_id}")
                for c in cycles:
                    try:
                        u_emb = getattr(c, 'user_input_embedding', None)
                        r_emb = getattr(c, 'final_response_embedding', None)
                        logger.info(f"STM cycle {c.cycle_id}: u_emb={'present' if u_emb else 'None'} (len={len(u_emb) if u_emb else 0}), r_emb={'present' if r_emb else 'None'} (len={len(r_emb) if r_emb else 0})")
                        best = 0.0
                        if u_emb:
                            sim_u = self._cosine_similarity(query_embedding, u_emb)
                            best = max(best, sim_u)
                            logger.info(f"  user_input similarity: {sim_u:.4f}")
                        if r_emb:
                            sim_r = self._cosine_similarity(query_embedding, r_emb)
                            best = max(best, sim_r)
                            logger.info(f"  final_response similarity: {sim_r:.4f}")
                        logger.info(f"  best similarity: {best:.4f}, threshold: {query_request.min_relevance_score}")
                        if best >= query_request.min_relevance_score:
                            c.score = best
                            stm_results.append(c)
                    except Exception:
                        continue
                stm_results.sort(key=lambda x: getattr(x, 'score', 0.0), reverse=True)
                stm_results = stm_results[:query_request.limit]

            # Always enforce user_id - do not allow callers to override it via metadata_filters
            where_clause = {'user_id': str(query_request.user_id)}
            if query_request.metadata_filters:
                if 'user_id' in query_request.metadata_filters:
                    raise APIException(detail="metadata_filters may not include user_id.", status_code=400)
                where_clause.update(query_request.metadata_filters)

            results = self.cycles_collection.query(
                query_embeddings=[query_embedding],
                n_results=query_request.limit,
                where=where_clause
            )

            # (STM results already computed above)

            # Get LTM results
            ltm_cycles = []
            # ChromaDB .query returns nested lists per query; results['metadatas'][0] is the list for our single query
            if results and results.get('metadatas'):
                metadatas_for_query = results['metadatas'][0]
                distances_for_query = results.get('distances', [[]])[0]
                for i, metadata in enumerate(metadatas_for_query):
                    distance = distances_for_query[i] if i < len(distances_for_query) else float('inf')
                    score = self._distance_to_score(distance)
                    if score >= query_request.min_relevance_score:
                        cycle_data = json.loads(metadata['json_data'])
                        cycle_data['score'] = score
                        ltm_cycles.append(CognitiveCycle(**cycle_data))

            # Combine and sort results
            all_cycles = stm_results + ltm_cycles
            all_cycles.sort(key=lambda x: x.score if hasattr(x, 'score') else 0.0, reverse=True)
            
            # Update access stats and emit metrics
            stm_hit_count = len(stm_results)
            ltm_hit_count = len(ltm_cycles)
            avg_relevance = sum(c.score if hasattr(c, 'score') else 0.0 for c in all_cycles) / len(all_cycles) if all_cycles else 0.0
            
            stats = self._access_stats.setdefault(query_request.user_id, MemoryAccessStats())
            stats.record_query(
                stm_hits=stm_hit_count,
                ltm_hits=ltm_hit_count,
                avg_relevance=avg_relevance
            )
            
            # Emit memory query metrics
            try:
                if self.decision_engine:
                    signals = {
                        "event": "query_metrics",
                        "user_id": str(query_request.user_id),
                        "stm_hits": stm_hit_count,
                        "ltm_hits": ltm_hit_count,
                        "avg_relevance": avg_relevance,
                        "total_results": len(all_cycles),
                        "query_text_prefix": (query_request.query_text or "")[:50],
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    await self._emit_signals(query_request.user_id, signals)
            except Exception:
                logger.debug("DecisionEngine query_metrics signal failed or not configured.")
            
            # Record memory access metrics
            if self.metrics_service:
                # Calculate overall hit rate and record for each tier
                total_queries = stm_hit_count + ltm_hit_count
                if total_queries > 0:
                    # Record STM tier metrics
                    if stm_hit_count > 0:
                        await self.metrics_service.record_metric(
                            MetricType.MEMORY_ACCESS,
                            {
                                "tier_accessed": "stm",
                                "hit_rate": stm_hit_count / total_queries,
                                "retrieval_time_ms": 50.0,  # Estimated STM retrieval time
                                "access_count": stm_hit_count
                            },
                            user_id=str(query_request.user_id)
                        )
                    
                    # Record LTM tier metrics
                    if ltm_hit_count > 0:
                        await self.metrics_service.record_metric(
                            MetricType.MEMORY_ACCESS,
                            {
                                "tier_accessed": "ltm",
                                "hit_rate": ltm_hit_count / total_queries,
                                "retrieval_time_ms": 200.0,  # Estimated LTM retrieval time
                                "access_count": ltm_hit_count
                            },
                            user_id=str(query_request.user_id)
                        )
            
            logger.info(f"AUDIT: Queried memory for user {query_request.user_id}, retrieved {len(all_cycles)} cycles ({stm_hit_count} STM, {ltm_hit_count} LTM) with query '{(query_request.query_text or '')[:50]}...'.")
            return all_cycles[:query_request.limit]
        except APIException:
            raise
        except chromadb.errors.ChromaDBError as e:
            logger.error(f"ChromaDB error during query_memory for user {query_request.user_id}: {e}", exc_info=True)
            raise APIException(detail=f"Database error during memory query: {e}", status_code=503)
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error during query_memory for user {query_request.user_id}: {e}", exc_info=True)
            raise APIException(detail=f"Error processing memory data: {e}", status_code=500)
        except Exception as e:
            logger.error(f"Unexpected error during query_memory for user {query_request.user_id}: {e}", exc_info=True)
            raise APIException(detail=f"An unexpected error occurred while querying memory: {e}", status_code=500)

    async def get_access_stats(self, user_id: UUID) -> MemoryAccessStats:
        """Return memory access stats for a user, creating defaults if missing."""
        return self._access_stats.setdefault(user_id, MemoryAccessStats())

    def get_immediate_transcript(self, user_id: UUID, max_tokens: Optional[int] = None) -> str:
        """
        Return a recent verbatim transcript slice for this user up to max_tokens.
        Falls back to IMMEDIATE_TOKEN_BUDGET when not provided.
        """
        rt = self._transcripts.get(user_id)
        if not rt:
            return ""
        budget = max_tokens or self.IMMEDIATE_TOKEN_BUDGET
        try:
            text = rt.get_context(budget)
            logger.info(
                f"Immediate transcript retrieval for user {user_id}: returned ~{self.token_counter.count_tokens(text)} tokens"
            )
            return text
        except Exception as e:
            logger.warning(f"Failed to build immediate transcript context: {e}")
            return ""

    async def _store_cycle(self, cycle: CognitiveCycle) -> None:
        """Store a single cognitive cycle in LTM (ChromaDB) with embedding and metadata."""
        if not self.cycles_collection:
            logger.error(f"Unable to store cycle in LTM - ChromaDB not connected for user {cycle.user_id}")
            return
        try:
            # Prepare embedding text and embedding
            embed_text = self._get_cycle_embedding_text(cycle)
            embedding = await self.llm_service.generate_embedding(
                text=embed_text,
                model_name=settings.EMBEDDING_MODEL_NAME
            )

            # Serialize cycle
            cycle_json = cycle.model_dump_json()

            # Upsert into ChromaDB
            self.cycles_collection.upsert(
                ids=[str(cycle.cycle_id)],
                embeddings=[embedding],
                documents=[embed_text],
                metadatas=[{
                    'user_id': str(cycle.user_id),
                    'timestamp': cycle.timestamp.isoformat(),
                    'json_data': cycle_json,
                    'reflection_status': getattr(cycle, 'reflection_status', 'pending')
                }]
            )
            logger.info(f"AUDIT: Stored cycle {cycle.cycle_id} in LTM for user {cycle.user_id}")
            
            # Record memory storage metrics
            if self.metrics_service:
                await self.metrics_service.record_metric(
                    MetricType.MEMORY_ACCESS,
                    {
                        "operation": "store",
                        "cycle_id": str(cycle.cycle_id),
                        "embedding_text_length": len(embed_text),
                        "agent_outputs_count": len(cycle.agent_outputs),
                        "has_final_response": cycle.final_response is not None,
                        "reflection_status": getattr(cycle, 'reflection_status', 'pending')
                    },
                    cycle_id=str(cycle.cycle_id),
                    user_id=str(cycle.user_id)
                )
            
        except Exception as e:
            logger.error(f"Error storing cycle {cycle.cycle_id} for user {cycle.user_id}: {e}", exc_info=True)

    def _get_cycle_embedding_text(self, cycle: CognitiveCycle) -> str:
        """Compose a text representation of the cycle for embedding/search."""
        parts = [
            f"User: {getattr(cycle, 'user_input', '')}",
            f"Response: {getattr(cycle, 'final_response', '')}",
        ]
        # Optionally include response metadata strategies/tone if present
        try:
            if getattr(cycle, 'response_metadata', None):
                parts.append(f"Type: {cycle.response_metadata.response_type}")
                parts.append(f"Tone: {cycle.response_metadata.tone}")
                if cycle.response_metadata.strategies:
                    parts.append(f"Strategies: {', '.join(cycle.response_metadata.strategies)}")
        except Exception:
            pass
        return "\n".join([p for p in parts if p])

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Compute cosine similarity between two vectors, guarding against zeros and mismatched dims."""
        try:
            if not a or not b or len(a) != len(b):
                return 0.0
            dot = 0.0
            na = 0.0
            nb = 0.0
            for x, y in zip(a, b):
                dot += x * y
                na += x * x
                nb += y * y
            if na <= 0.0 or nb <= 0.0:
                return 0.0
            return dot / ((na ** 0.5) * (nb ** 0.5))
        except Exception:
            return 0.0

    async def get_recent_cycles_for_reflection(self, user_id: UUID, limit: int) -> List[CognitiveCycle]:
        """
        Retrieves the most recent cognitive cycles for a specific user, suitable for reflection.
        """
        if not self.cycles_collection:
            raise APIException(detail="MemoryService not connected to ChromaDB.", status_code=503)
        
        try:
            results = self.cycles_collection.get(
                where={'user_id': str(user_id), 'reflection_status': 'pending'},
                limit=limit,
                include=["metadatas"]
            )
            
            recent_cycles = []
            # .get returns a flat list under 'metadatas'
            if results and results.get('metadatas'):
                for metadata in results['metadatas']:
                    try:
                        cycle_data = json.loads(metadata['json_data'])
                        recent_cycles.append(CognitiveCycle(**cycle_data))
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse cycle data for user {user_id}: {e}", exc_info=True)
                        continue
                    except Exception as e:
                        logger.error(f"Error processing cycle data for user {user_id}: {e}", exc_info=True)
                        continue
            
            # ChromaDB doesn't support sorting by timestamp directly in the query, so we sort in Python
            # Be defensive: ensure timestamp exists and is comparable
            recent_cycles = [c for c in recent_cycles if getattr(c, "timestamp", None) is not None]
            recent_cycles.sort(key=lambda c: c.timestamp, reverse=True)

            logger.info(f"AUDIT: Retrieved {len(recent_cycles)} cycles for reflection for user {user_id}")
            return recent_cycles

        except chromadb.errors.ChromaDBError as e:
            logger.error(f"ChromaDB error retrieving cycles for user {user_id}: {e}", exc_info=True)
            raise APIException(detail=f"Database error retrieving memory cycles: {e}", status_code=503)
        except Exception as e:
            logger.error(f"Unexpected error retrieving cycles for user {user_id}: {e}", exc_info=True)
            raise APIException(detail=f"An unexpected error occurred retrieving memory cycles: {e}", status_code=500)

            logger.info(f"AUDIT: Retrieved {len(recent_cycles)} recent pending cycles for reflection for user {user_id}.")
            return recent_cycles
        except Exception as e:
            logger.error(f"Unexpected error during get_recent_cycles_for_reflection for user {user_id}: {e}", exc_info=True)
            raise APIException(detail=f"An unexpected error occurred while retrieving recent cycles: {e}", status_code=500)

    async def get_cycle_by_id(self, user_id: UUID, cycle_id: UUID) -> Optional[CognitiveCycle]:
        """
        Retrieves a single cognitive cycle by its ID.
        """
        if not self.cycles_collection:
            raise APIException(detail="MemoryService not connected to ChromaDB.", status_code=503)

        try:
            result = self.cycles_collection.get(
                ids=[str(cycle_id)],
                where={'user_id': str(user_id)}
            )
            if result and result.get('metadatas'):
                # result['metadatas'] is a list of metadata dicts for requested ids (order preserved)
                metadata = result['metadatas'][0]
                cycle_data = json.loads(metadata['json_data'])
                logger.info(f"AUDIT: Retrieved cognitive cycle {cycle_id} for user {user_id}.")
                return CognitiveCycle(**cycle_data)
            logger.info(f"AUDIT: Cognitive cycle {cycle_id} not found for user {user_id}.")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during get_cycle_by_id for user {user_id}, cycle {cycle_id}: {e}", exc_info=True)
            raise APIException(detail=f"An unexpected error occurred while retrieving cycle by ID: {e}", status_code=500)

    async def update_cycle_metadata(self, user_id: UUID, cycle_id: UUID, metadata_to_update: Dict[str, Any]) -> bool:
        """
        Updates specific metadata fields of a cognitive cycle.
        """
        if not self.cycles_collection:
            raise APIException(detail="MemoryService not connected to ChromaDB.", status_code=503)

        try:
            cycle = await self.get_cycle_by_id(user_id, cycle_id)
            if not cycle:
                return False

            for key, value in metadata_to_update.items():
                # Prevent changing user_id or cycle_id via metadata updates
                if key in ("user_id", "cycle_id"):
                    continue
                setattr(cycle, key, value)

            await self.upsert_cycle(cycle)
            logger.info(f"AUDIT: Updated metadata for cycle {cycle_id} for user {user_id}.")
            return True
        except Exception as e:
            logger.error(f"Unexpected error during update_cycle_metadata for user {user_id}, cycle {cycle_id}: {e}", exc_info=True)
            raise APIException(detail=f"An unexpected error occurred while updating cycle metadata: {e}", status_code=500)

    async def list_cycles(self, request: CycleListRequest) -> Tuple[List[CognitiveCycle], int]:
        """
        Retrieves a list of cognitive cycles for a specific user with optional filtering and pagination.
        """
        if not self.cycles_collection:
            raise APIException(detail="MemoryService not connected to ChromaDB.", status_code=503)

        try:
            where_clause = {'user_id': str(request.user_id)}
            if request.session_id:
                where_clause['session_id'] = str(request.session_id)
            # ChromaDB doesn't support date ranges or complex filters in the same way as MongoDB.
            # This implementation will fetch all cycles for the user and filter in memory.
            
            results = self.cycles_collection.get(
                where=where_clause,
                include=["metadatas"]
            )

            cycles = []
            if results and results.get('metadatas'):
                for metadata in results['metadatas']:
                    cycle_data = json.loads(metadata['json_data'])
                    cycles.append(CognitiveCycle(**cycle_data))

            # In-memory filtering
            if request.start_date:
                cycles = [c for c in cycles if c.timestamp >= request.start_date]
            if request.end_date:
                cycles = [c for c in cycles if c.timestamp <= request.end_date]
            if request.response_type:
                cycles = [c for c in cycles if c.response_metadata and c.response_metadata.response_type == request.response_type]
            if request.min_confidence is not None:
                cycles = [c for c in cycles if any(ao.confidence >= request.min_confidence for ao in c.agent_outputs)]

            total_cycles = len(cycles)
            cycles.sort(key=lambda c: c.timestamp, reverse=True)
            paginated_cycles = cycles[request.skip : request.skip + request.limit]

            logger.info(f"AUDIT: Listed {len(paginated_cycles)} of {total_cycles} cognitive cycles for user {request.user_id}.")
            return paginated_cycles, total_cycles
        except Exception as e:
            logger.error(f"Unexpected error during list_cycles for user {request.user_id}: {e}", exc_info=True)
            raise APIException(detail=f"An unexpected error occurred while listing cycles: {e}", status_code=500)

    async def upsert_pattern(self, pattern: DiscoveredPattern) -> bool:
        """
        Stores a discovered pattern in ChromaDB.
        """
        if not self.patterns_collection:
            raise APIException(detail="MemoryService not connected to ChromaDB.", status_code=503)

        try:
            # Ensure we have an embedding; compute if missing
            embedding = getattr(pattern, 'embedding', None) or []
            if not embedding:
                try:
                    embedding = await self.llm_service.generate_embedding(
                        text=pattern.description,
                        model_name=settings.EMBEDDING_MODEL_NAME
                    )
                    pattern.embedding = embedding
                except Exception:
                    embedding = []

            # Build metadata and document text
            pattern_data = pattern.model_dump()
            metadata = {
                k: (str(v) if isinstance(v, (UUID, datetime)) else v)
                for k, v in pattern_data.items()
                if v is not None and not isinstance(v, (list, dict))
            }
            metadata['user_id'] = str(pattern.user_id)
            metadata['json_data'] = json.dumps(pattern_data, cls=UUIDEncoder)

            document_text = pattern.description or json.dumps({
                'pattern_type': pattern.pattern_type,
                'description': pattern.description,
            })

            # Chroma requires at least one of embeddings/documents/images/uris. Since the
            # patterns_collection has no embedding function, we must pass embeddings explicitly.
            if not embedding:
                # As a last resort, create a small constant embedding vector to satisfy API
                # (still useful since we also store the document for reference)
                embedding = [0.0] * 10

            self.patterns_collection.upsert(
                ids=[str(pattern.pattern_id)],
                embeddings=[embedding],
                documents=[document_text],
                metadatas=[metadata],
            )
            logger.info(f"AUDIT: Upserted discovered pattern {pattern.pattern_id} for user {pattern.user_id}.")
            return True
        except Exception as e:
            logger.error(f"Unexpected error during upsert_pattern for user {pattern.user_id}: {e}", exc_info=True)
            raise APIException(detail=f"An unexpected error occurred while storing pattern: {e}", status_code=500)

    async def delete_user_data(self, user_id: UUID) -> bool:
        """
        Deletes all data associated with a specific user.
        """
        if not self.cycles_collection or not self.patterns_collection:
            raise APIException(detail="MemoryService not connected to ChromaDB.", status_code=503)

        try:
            self.cycles_collection.delete(where={'user_id': str(user_id)})
            self.patterns_collection.delete(where={'user_id': str(user_id)})
            logger.info(f"AUDIT: Deleted all data for user {user_id}.")
            return True
        except Exception as e:
            logger.error(f"Unexpected error during delete_user_data for user {user_id}: {e}", exc_info=True)
            raise APIException(detail=f"An unexpected error occurred while deleting user data: {e}", status_code=500)

    async def get_patterns_for_user(self, user_id: UUID) -> List[DiscoveredPattern]:
        """
        Retrieves meta-learnings and discovered patterns for a specific user.
        """
        if not self.patterns_collection:
            raise APIException(detail="MemoryService not connected to ChromaDB.", status_code=503)
        
        try:
            results = self.patterns_collection.get(
                where={'user_id': str(user_id)},
                include=["metadatas"]
            )
            patterns = []
            if results and results.get('metadatas'):
                for metadata in results['metadatas']:
                    pattern_data = json.loads(metadata['json_data'])
                    patterns.append(DiscoveredPattern(**pattern_data))
            return patterns
        except Exception as e:
            logger.error(f"Unexpected error during get_patterns_for_user for user {user_id}: {e}", exc_info=True)
            raise APIException(detail=f"An unexpected error occurred while retrieving patterns: {e}", status_code=500)
