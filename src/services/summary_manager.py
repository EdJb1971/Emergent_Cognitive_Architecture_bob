import logging
import json
import ast
from typing import List, Optional, Dict, Any, Tuple
from uuid import UUID
from datetime import datetime, timedelta

import chromadb
from chromadb.utils import embedding_functions

from src.core.config import settings
from src.core.exceptions import ConfigurationError, APIException
from src.models.core_models import CognitiveCycle
from src.models.memory_models import ConversationSummary
from src.services.llm_integration_service import LLMIntegrationService

logger = logging.getLogger(__name__)

class SummaryManager:
    """
    Manages conversation summaries, providing dynamic context across interactions.
    Works alongside MemoryService to maintain high-level conversation context.
    """
    
    def __init__(self, llm_service: LLMIntegrationService):
        self.llm_service = llm_service
        self.client: Optional[chromadb.Client] = None
        self.summaries_collection: Optional[chromadb.Collection] = None
        self.cycles_collection: Optional[chromadb.Collection] = None
        self._active_summaries: Dict[UUID, ConversationSummary] = {}
        logger.info("SummaryManager initialized.")

    async def connect(self, client: Optional[chromadb.Client] = None):
        """
        Initialize ChromaDB connection for summary storage.
        
        Args:
            client: Optional existing ChromaDB client to reuse. If None, creates a new one.
        """
        try:
            if client:
                self.client = client
                logger.info("SummaryManager reusing existing ChromaDB client.")
            else:
                self.client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
                logger.info("SummaryManager created new ChromaDB client.")
            
            self.summaries_collection = self.client.get_or_create_collection(
                name="conversation_summaries",
                embedding_function=embedding_functions.DefaultEmbeddingFunction()
            )
            # Also reference the cycles collection for identity mining; reuse existing without redefining embedding function
            try:
                self.cycles_collection = self.client.get_or_create_collection(
                    name=settings.CHROMA_COLLECTION_CYCLES
                )
            except Exception as e:
                logger.warning(f"Unable to initialize cycles collection in SummaryManager: {e}")
            logger.info("Successfully connected to ChromaDB for summary storage.")
        except Exception as e:
            logger.error(f"Failed to connect to ChromaDB for summaries: {e}")
            raise ConfigurationError(detail=f"Summary storage initialization failed: {e}")

    async def get_or_create_summary(self, user_id: UUID) -> ConversationSummary:
        """Get existing or create new conversation summary for a user."""
        if user_id in self._active_summaries:
            return self._active_summaries[user_id]

        # Try to load from storage
        if self.summaries_collection:
            # Use .get with a where filter to retrieve existing summary without requiring query_* params
            results = self.summaries_collection.get(
                where={"user_id": str(user_id)},
                limit=1,
                include=["metadatas"]
            )
            if results and results.get('metadatas'):
                metadata = results['metadatas'][0]
                if metadata and 'json_data' in metadata:
                    summary_data = json.loads(metadata['json_data'])
                    # Normalize entities back to a set (handles legacy string-serialized values)
                    if 'entities' in summary_data:
                        entities_val = summary_data['entities']
                        if isinstance(entities_val, list):
                            summary_data['entities'] = set(entities_val)
                        elif isinstance(entities_val, str):
                            # Handle legacy cases like "set()" or "{'a', 'b'}" or "['a','b']"
                            normalized: set = set()
                            if entities_val.strip() == "set()":
                                normalized = set()
                            else:
                                try:
                                    parsed = ast.literal_eval(entities_val)
                                    if isinstance(parsed, (list, set, tuple)):
                                        normalized = set(parsed)
                                except Exception:
                                    normalized = set()
                            summary_data['entities'] = normalized
                    summary = ConversationSummary(**summary_data)
                    self._active_summaries[user_id] = summary
                    # Optionally persist normalized structure back to storage (best-effort)
                    try:
                        await self._store_summary(summary)
                    except Exception:
                        pass
                    return summary

        # Create new if not found
        summary = ConversationSummary(user_id=user_id)
        self._active_summaries[user_id] = summary
        return summary

    async def update_summary(self, user_id: UUID, cycle: CognitiveCycle) -> ConversationSummary:
        """
        Update conversation summary based on new interaction.
        Uses LLM to extract key information and update summary intelligently.
        """
        summary = await self.get_or_create_summary(user_id)
        
        try:
            # STEP 1: Check if we need to mine past cycles for missing identity info
            # (Only if we don't have user name or location in summary yet)
            if not any("name:" in str(ctx).lower() for ctx in summary.context_points):
                await self._mine_past_cycles_for_identity(user_id, summary)
            
            # STEP 2: Analyze current cycle
            # Generate analysis prompt with JSON structure
            analysis_prompt = (
                f"Analyze this conversation turn and extract key information:\n\n"
                f"User Input: {cycle.user_input}\n"
                f"System Response: {cycle.final_response}\n\n"
                f"Current Summary Context:\n"
                f"Topics: {', '.join(summary.key_topics) if summary.key_topics else 'None'}\n"
                f"Latest Topic: {summary.latest_topic or 'None'}\n"
                f"Key Entities: {', '.join(summary.entities) if summary.entities else 'None'}\n"
                f"Current State: {summary.conversation_state}\n\n"
                f"Extract and return as JSON:\n"
                f"{{\n"
                f'  "user_name": "Name",  // CRITICAL: Extract user\'s name if mentioned (e.g., "my name is X", "I\'m X", "call me X")\n'
                f'  "ai_name": "Name",  // Extract what user calls the AI (e.g., "Bob", "your name is X")\n'
                f'  "location": "Place",  // Extract user\'s location if mentioned (e.g., "I live in X", "I\'m in X")\n'
                f'  "new_topics": ["topic1", "topic2"],  // New topics discussed in this turn\n'
                f'  "entities": ["entity1", "entity2"],  // Other important entities (NOT names/locations already captured)\n'
                f'  "context_points": ["point1", "point2"],  // Key facts or context to remember\n'
                f'  "preferences": {{"key": "value"}},  // User preferences/patterns revealed\n'
                f'  "conversation_state": "ongoing/initial/deep_discussion/problem_solving"  // Current state\n'
                f"}}\n\n"
                f"CRITICAL: Pay special attention to user name, AI name, and location - these are the most important identity markers.\n"
                f"Look for patterns like: 'my name is', 'I\\'m', 'call me', 'named you', 'I live in', 'I\\'m in', 'based in'."
            )

            # Get LLM analysis
            analysis_str = await self.llm_service.generate_text(
                prompt=analysis_prompt,
                model_name=settings.LLM_MODEL_NAME,
                temperature=0.3,  # Lower temperature for more structured output
                max_output_tokens=500
            )
            
            logger.info(f"Summary analysis for user {user_id}: {analysis_str[:200]}...")

            # Parse JSON response
            from src.agents.utils import extract_json_from_response
            analysis = extract_json_from_response(analysis_str)
            
            # FALLBACK: Manual pattern matching for critical identity markers
            # (in case LLM misses them)
            analysis = self._enhance_with_manual_extraction(analysis, cycle)
            
            # Apply updates from parsed JSON
            self._apply_json_analysis_updates(summary, analysis, cycle)
            
            # Update summary metadata
            summary.referenced_memories.append(cycle.cycle_id)
            summary.mark_updated()
            
            # Generate new embedding
            summary_text = self._generate_summary_text(summary)
            summary.embedding = await self.llm_service.generate_embedding(
                text=summary_text,
                model_name=settings.EMBEDDING_MODEL_NAME
            )

            # Store updated summary
            await self._store_summary(summary)
            
            logger.info(
                f"Summary updated for user {user_id}: "
                f"topics={summary.key_topics}, entities={summary.entities}, state={summary.conversation_state}"
            )
            
            return summary

        except Exception as e:
            logger.error(f"Failed to update summary for user {user_id}: {e}", exc_info=True)
            # Don't raise - allow conversation to continue even if summary fails
            return summary

    async def _mine_past_cycles_for_identity(self, user_id: UUID, summary: ConversationSummary):
        """
        Mine past cycles for identity information (name, location) if summary doesn't have it yet.
        This is a one-time backfill operation per summary.
        """
        try:
            # Get recent cycles from LTM (not using semantic search, just recent chronological)
            # Import here to avoid circular dependency
            import re
            
            # Query recent cycles (could use memory_service.list_cycles if available)
            # For now, just check if we have client access
            if not self.client:
                return
            
            # Get last 20 cycles for this user
            if not self.cycles_collection:
                try:
                    self.cycles_collection = self.client.get_or_create_collection(
                        name=settings.CHROMA_COLLECTION_CYCLES
                    )
                except Exception:
                    return

            results = self.cycles_collection.get(
                where={"user_id": str(user_id)},
                limit=20,
                include=["metadatas"]
            )
            
            if not results or not results.get('metadatas'):
                return
            
            logger.info(f"Mining {len(results['metadatas'])} past cycles for identity information for user {user_id}")
            
            # Check each cycle for identity markers
            for metadata in results['metadatas']:
                try:
                    cycle_data = json.loads(metadata['json_data'])
                    user_input = cycle_data.get('user_input', '').lower()
                    
                    # Look for name patterns
                    name_match = re.search(r"my name is (\w+)|i'm (\w+)|im (\w+)|call me (\w+)", user_input)
                    if name_match:
                        name = next((g.capitalize() for g in name_match.groups() if g), None)
                        if name:
                            summary.add_entity(name)
                            summary.add_context(f"User's name: {name}")
                            logger.info(f"Mined user name from past cycle: {name}")
                    
                    # Look for location patterns
                    location_match = re.search(r"i live in (\w+)|im in (\w+)|i'm in (\w+)|based in (\w+)|from (\w+)", user_input)
                    if location_match:
                        location = next((g.capitalize() for g in location_match.groups() if g), None)
                        if location:
                            summary.add_entity(location)
                            summary.add_context(f"User's location: {location}")
                            logger.info(f"Mined location from past cycle: {location}")
                    
                    # Look for AI name patterns
                    ai_name_match = re.search(r"named you (\w+)|call you (\w+)|your name is (\w+)", user_input)
                    if ai_name_match:
                        ai_name = next((g.capitalize() for g in ai_name_match.groups() if g), None)
                        if ai_name:
                            summary.add_entity(ai_name)
                            summary.add_context(f"AI's name: {ai_name}")
                            logger.info(f"Mined AI name from past cycle: {ai_name}")
                
                except Exception as e:
                    logger.debug(f"Error mining cycle for identity: {e}")
                    continue
            
            # Store updated summary if we found anything
            if summary.entities:
                await self._store_summary(summary)
                
        except Exception as e:
            logger.warning(f"Error mining past cycles for identity: {e}", exc_info=True)

    def _enhance_with_manual_extraction(self, analysis: dict, cycle: CognitiveCycle) -> dict:
        """
        Fallback manual extraction of critical identity markers.
        Uses regex patterns to catch what LLM might miss.
        """
        import re
        
        original_text = cycle.user_input or ""
        user_input_lower = original_text.lower()

        STOPWORDS = {"and", "but", "so", "then", "also", "ok", "okay", "yeah", "yep", "no", "yes", "the", "this", "that"}

        def _clean_token(tok: str) -> Optional[str]:
            t = (tok or "").strip().strip(".,!?;:()[]{}\"'")
            if not t or len(t) < 3:
                return None
            if t.lower() in STOPWORDS:
                return None
            if not any(c.isalpha() for c in t):
                return None
            return t.title()
        
        # Extract user name patterns
        if not analysis.get("user_name") or analysis.get("user_name") in ["Name", "name", "None"]:
            name_patterns = [
                r"\bmy name is\s+([A-Za-z][A-Za-z\-']{1,})",
                r"\bi['’]?m\s+([A-Za-z][A-Za-z\-']{1,})",
                r"\bim\s+([A-Za-z][A-Za-z\-']{1,})",
                r"\bcall me\s+([A-Za-z][A-Za-z\-']{1,})",
                r"\bthis is\s+([A-Za-z][A-Za-z\-']{1,})",
            ]
            for pattern in name_patterns:
                match = re.search(pattern, original_text, flags=re.IGNORECASE)
                if match:
                    raw = match.group(1)
                    name = _clean_token(raw)
                    if name and name.lower() not in {"working", "trying", "building", "still"}:
                        analysis["user_name"] = name
                        logger.info(f"Manual extraction found user name: {name}")
                        break
        
        # Extract AI name patterns
        if not analysis.get("ai_name") or analysis.get("ai_name") in ["Name", "name", "None"]:
            ai_name_patterns = [
                r"\byour name is\s+([A-Za-z][A-Za-z\-']{1,})",
                r"\bnamed you\s+([A-Za-z][A-Za-z\-']{1,})",
                r"\bcall you\s+([A-Za-z][A-Za-z\-']{1,})",
                r"\bcalling you\s+([A-Za-z][A-Za-z\-']{1,})",
                r"^\s*([A-Za-z][A-Za-z\-']{1,})[,\s]+(?:im|i['’]?m)\b",  # "Bob, I'm ..."
            ]
            for pattern in ai_name_patterns:
                match = re.search(pattern, original_text, flags=re.IGNORECASE)
                if match:
                    ai_raw = match.group(1)
                    ai_name = _clean_token(ai_raw)
                    if ai_name:
                        analysis["ai_name"] = ai_name
                        logger.info(f"Manual extraction found AI name: {ai_name}")
                        break
        
        # Extract location patterns
        if not analysis.get("location") or analysis.get("location") in ["Place", "place", "None"]:
            # Prefer properly formatted multi-word locations, capturing from the original text
            location_patterns = [
                r"\b(?:i live in|i['’]?m in|im in|based in|from|at)\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)",
                r"\b(?:i live in|i['’]?m in|im in|based in|from|at)\s+([A-Za-z][A-Za-z\-']{2,})",
            ]
            for pattern in location_patterns:
                match = re.search(pattern, original_text)
                if not match:
                    match = re.search(pattern, original_text, flags=re.IGNORECASE)
                if match:
                    loc_raw = match.group(1)
                    location = _clean_token(loc_raw)
                    if location:
                        analysis["location"] = location
                        logger.info(f"Manual extraction found location: {location}")
                        break
        
        return analysis

    def _apply_json_analysis_updates(self, summary: ConversationSummary, analysis: dict, cycle: CognitiveCycle):
        """Apply JSON analysis updates to the summary."""
        try:
            # PRIORITY 1: Extract identity markers (user name, AI name, location)
            user_name = analysis.get("user_name")
            if user_name and isinstance(user_name, str) and user_name.lower() not in ["name", "none", ""]:
                summary.add_entity(user_name)
                # Also add to context
                summary.add_context(f"User's name: {user_name}")
                logger.info(f"Extracted user name: {user_name}")
            
            ai_name = analysis.get("ai_name")
            if ai_name and isinstance(ai_name, str) and ai_name.lower() not in ["name", "none", ""]:
                summary.add_entity(ai_name)
                summary.add_context(f"AI's name: {ai_name}")
                logger.info(f"Extracted AI name: {ai_name}")
            
            location = analysis.get("location")
            if location and isinstance(location, str) and location.lower() not in ["place", "none", ""]:
                summary.add_entity(location)
                summary.add_context(f"User's location: {location}")
                logger.info(f"Extracted location: {location}")
            
            # Add new topics
            new_topics = analysis.get("new_topics", [])
            for topic in new_topics:
                if topic and isinstance(topic, str):
                    summary.add_topic(topic)
            
            # Add OTHER entities (not already captured as identity markers)
            entities = analysis.get("entities", [])
            for entity in entities:
                if entity and isinstance(entity, str):
                    summary.add_entity(entity)
            
            # Add context points
            context_points = analysis.get("context_points", [])
            for point in context_points:
                if point and isinstance(point, str):
                    summary.add_context(point)
            
            # Update preferences
            preferences = analysis.get("preferences", {})
            if isinstance(preferences, dict):
                for key, value in preferences.items():
                    if key and value:
                        summary.update_preference(str(key), str(value))
            
            # Update conversation state
            new_state = analysis.get("conversation_state")
            if new_state and isinstance(new_state, str):
                summary.conversation_state = new_state
            
            # Update latest topic based on current turn
            if new_topics:
                summary.latest_topic = new_topics[-1]
            
        except Exception as e:
            logger.warning(f"Error applying JSON analysis updates: {e}", exc_info=True)

    def _apply_analysis_updates(self, summary: ConversationSummary, analysis: str):
        """
        DEPRECATED: Legacy text-based parsing. Kept for backwards compatibility.
        Use _apply_json_analysis_updates instead.
        """
        # This is a simplified version - in practice, you'd want more robust parsing
        try:
            sections = analysis.split('\n')
            for section in sections:
                if section.startswith("Topic:"):
                    summary.add_topic(section[6:].strip())
                elif section.startswith("Entity:"):
                    summary.add_entity(section[7:].strip())
                elif section.startswith("Context:"):
                    summary.add_context(section[8:].strip())
                elif section.startswith("Preference:"):
                    pref = section[11:].strip().split(':')
                    if len(pref) == 2:
                        summary.update_preference(pref[0].strip(), pref[1].strip())
                elif section.startswith("State:"):
                    summary.conversation_state = section[6:].strip()

        except Exception as e:
            logger.warning(f"Error parsing analysis updates: {e}")

    def _generate_summary_text(self, summary: ConversationSummary) -> str:
        """Generate text representation of summary for embedding."""
        return (
            f"Topics: {', '.join(summary.key_topics)}\n"
            f"Current Topic: {summary.latest_topic}\n"
            f"Entities: {', '.join(summary.entities)}\n"
            f"Context: {'. '.join(summary.context_points)}\n"
            f"State: {summary.conversation_state}"
        )

    async def _store_summary(self, summary: ConversationSummary):
        """Store summary in ChromaDB."""
        if not self.summaries_collection:
            raise APIException(detail="Summary storage not initialized.", status_code=503)

        try:
            # Convert to dict and handle sets for JSON serialization
            summary_dict = summary.model_dump()
            # Convert entities set to list for JSON serialization
            if 'entities' in summary_dict and isinstance(summary_dict['entities'], set):
                summary_dict['entities'] = list(summary_dict['entities'])
            
            self.summaries_collection.upsert(
                ids=[str(summary.summary_id)],
                embeddings=[summary.embedding] if summary.embedding else None,
                metadatas=[{
                    "user_id": str(summary.user_id),
                    "last_updated": summary.last_updated.isoformat(),
                    "json_data": json.dumps(summary_dict, default=str)
                }]
            )
        except Exception as e:
            logger.error(f"Failed to store summary: {e}")
            raise APIException(detail=f"Failed to store summary: {e}", status_code=500)

    async def summarize_stm(self, user_id: UUID, cycles: List[CognitiveCycle]) -> Tuple[ConversationSummary, str]:
        """
        Generate a summary of STM cycles and update the consolidated STM record.
        
        Args:
            user_id: The user's UUID
            cycles: List of cycles to summarize (oldest first)
            
        Returns:
            Tuple[ConversationSummary, str]: Updated conversation summary and consolidated text
        """
        if not cycles:
            raise ValueError("No cycles provided for summarization")

        try:
            # Get current summary
            summary = await self.get_or_create_summary(user_id)
            
            # Prepare cycles for summarization
            cycles_text = "\n\n".join([
                f"Time: {c.timestamp.isoformat()}\n"
                f"Input: {getattr(c, 'user_input', '') or ''}\n"
                f"Output: {getattr(c, 'final_response', '') or ''}\n"
                f"Context: {getattr(c, 'metadata', {}).get('context', '') if isinstance(getattr(c, 'metadata', {}), dict) else ''}"
                for c in cycles
            ])
            
            # Generate consolidation prompt
            consolidation_prompt = (
                f"Review these recent conversation cycles and create:\n"
                f"1. A concise summary of key points\n"
                f"2. Important topics discussed\n"
                f"3. User preferences or patterns noticed\n"
                f"4. Action items or follow-ups needed\n\n"
                f"Current Summary Context:\n"
                f"Topics: {', '.join(summary.key_topics)}\n"
                f"Latest Topic: {summary.latest_topic}\n"
                f"State: {summary.conversation_state}\n\n"
                f"Recent Interactions:\n{cycles_text}"
            )

            # Get LLM analysis
            consolidation = await self.llm_service.generate_completion(
                prompt=consolidation_prompt,
                model_name=settings.LLM_MODEL_NAME,
                temperature=0.3  # Lower temperature for more focused summary
            )

            # Update conversation summary
            self._apply_analysis_updates(summary, consolidation)
            for cycle in cycles:
                # Use cycle_id consistently with CognitiveCycle model
                summary.referenced_memories.append(getattr(cycle, 'cycle_id', getattr(cycle, 'id', None)))
            summary.mark_updated()
            
            # Generate embedding for the consolidated STM record
            stm_text = self._generate_stm_text(summary, consolidation)
            stm_embedding = await self.llm_service.generate_embedding(
                text=stm_text,
                model_name=settings.EMBEDDING_MODEL_NAME
            )
            
            # Store consolidated STM record
            await self._store_stm_consolidated(user_id, stm_text, stm_embedding, cycles)
            
            # Update and store summary
            summary.embedding = await self.llm_service.generate_embedding(
                text=self._generate_summary_text(summary),
                model_name=settings.EMBEDDING_MODEL_NAME
            )
            await self._store_summary(summary)
            
            return summary, consolidation

        except Exception as e:
            logger.error(f"Failed to summarize STM for user {user_id}: {e}", exc_info=True)
            raise APIException(detail=f"Failed to summarize STM: {e}", status_code=500)

    def _generate_stm_text(self, summary: ConversationSummary, consolidation: str) -> str:
        """Generate text for the consolidated STM record."""
        return (
            f"Recent Conversation Summary:\n{consolidation}\n\n"
            f"Active Topics: {', '.join(summary.key_topics)}\n"
            f"Current Topic: {summary.latest_topic}\n"
            f"Context Points:\n" + "\n".join([f"- {point}" for point in summary.context_points]) + "\n"
            f"Current State: {summary.conversation_state}"
        )

    async def _store_stm_consolidated(self, user_id: UUID, stm_text: str, 
                                    embedding: List[float], cycles: List[CognitiveCycle]):
        """Store the consolidated STM record in ChromaDB."""
        if not self.summaries_collection:
            raise APIException(detail="Summary storage not initialized.", status_code=503)

        try:
            # Use a consistent ID for the consolidated STM record
            stm_id = f"stm_consolidated:{user_id}"
            
            self.summaries_collection.upsert(
                ids=[stm_id],
                embeddings=[embedding],
                documents=[stm_text],
                metadatas=[{
                    "user_id": str(user_id),
                    "type": "stm_consolidated",
                    "last_updated": datetime.utcnow().isoformat(),
                    "cycle_ids": [str(c.id) for c in cycles],
                    "cycle_count": len(cycles)
                }]
            )
            logger.info(f"Updated consolidated STM record for user {user_id}")
            
        except Exception as e:
            logger.error(f"Failed to store consolidated STM: {e}", exc_info=True)
            raise APIException(detail=f"Failed to store consolidated STM: {e}", status_code=500)

    async def get_relevant_summaries(self, 
                                   query_text: str, 
                                   user_id: UUID,
                                   limit: int = 1) -> List[ConversationSummary]:
        """
        Retrieve relevant conversation summaries based on semantic search.
        """
        if not self.summaries_collection:
            raise APIException(detail="Summary storage not initialized.", status_code=503)

        try:
            query_embedding = await self.llm_service.generate_embedding(
                text=query_text,
                model_name=settings.EMBEDDING_MODEL_NAME
            )

            results = self.summaries_collection.query(
                query_embeddings=[query_embedding],
                where={"user_id": str(user_id)},
                n_results=limit
            )

            summaries = []
            if results and results.get('metadatas'):
                for metadata in results['metadatas']:
                    summary_data = json.loads(metadata['json_data'])
                    # Normalize entities to a set (handles list or legacy string)
                    if 'entities' in summary_data:
                        entities_val = summary_data['entities']
                        if isinstance(entities_val, list):
                            summary_data['entities'] = set(entities_val)
                        elif isinstance(entities_val, str):
                            normalized: set = set()
                            if entities_val.strip() == "set()":
                                normalized = set()
                            else:
                                try:
                                    parsed = ast.literal_eval(entities_val)
                                    if isinstance(parsed, (list, set, tuple)):
                                        normalized = set(parsed)
                                except Exception:
                                    normalized = set()
                            summary_data['entities'] = normalized
                    summaries.append(ConversationSummary(**summary_data))

            return summaries

        except Exception as e:
            logger.error(f"Failed to retrieve summaries: {e}")
            raise APIException(detail=f"Failed to retrieve summaries: {e}", status_code=500)