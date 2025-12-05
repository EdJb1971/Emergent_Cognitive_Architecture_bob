
# Emergent Cognitive Architecture (ECA) - Complete Architecture Documentation

**Last Updated:** November 9, 2025

This document is the **single source of truth** for the Emergent Cognitive Architecture (ECA), a brain-inspired multi-agent system designed for sophisticated, human-like interaction, continuous learning, and genuine cognitive continuity.

## Table of Contents

1. [High-Level Architecture](#high-level-architecture)
2. [Core Principles](#core-principles)
3. [Backend Architecture](#backend-architecture)
4. [Brain-Inspired Cognitive Architecture](#brain-inspired-cognitive-architecture)
   - [Phase 1: Core Self-Awareness & Working Memory](#phase-1-core-self-awareness--working-memory--completed)
   - [Phase 2: Selective Attention & Coherence](#phase-2-selective-attention--coherence--completed)
   - [Phase 3: Autobiographical Memory & Theory of Mind](#phase-3-autobiographical-memory--theory-of-mind--completed)
   - [Phase 4: Higher-Order Executive Functions (The Agents)](#phase-4-higher-order-executive-functions-the-agents--completed)
   - [Phase 5: Metacognition & Self-Reflection](#phase-5-metacognition--self-reflection--completed)
   - [Phase 6: Learning Systems (Reinforcement & Procedural)](#phase-6-learning-systems-reinforcement--procedural--completed)
5. [Memory System Architecture](#memory-system-architecture)
6. [Data Flow & Integration](#data-flow--integration)
   - [Complete Cognitive Cycle Flow](#complete-cognitive-cycle-flow)
   - [Cognitive Brain: Executive Function & Response Synthesis](#cognitive-brain-executive-function--response-synthesis)
7. [Scientific Dashboard & Metrics System](#scientific-dashboard--metrics-system)
8. [Frontend Architecture](#frontend-architecture)
9. [Deployment & Operations](#deployment--operations)

---

## High-Level Architecture

The ECA is a full-stack application with a **React/TypeScript frontend** and a **Python/FastAPI backend**. It follows a modular, service-oriented architecture inspired by human neuroscience. The backend is the core of the system, featuring a multi-agent cognitive framework with brain-like subsystems that process user input and generate intelligent, contextually-aware responses.

### System Overview Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    FRONTEND (React/TypeScript)                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ ChatWindow   │  │  ChatInput   │  │   API Layer  │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │           SCIENTIFIC DASHBOARD (Modal Overlay)           │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │ │
│  │  │ RealTimeChart│  │Learning Prog│  │ Agent Activity│   │ │
│  │  │ Statistical  │  │ Research     │  │ Export Tools  │   │ │
│  │  │ Analysis     │  │ Export       │  │              │   │ │
│  │  └──────────────┘  └──────────────┘  └──────────────┘   │ │
│  └──────────────────────────────────────────────────────────┘ │
└──────────────────────────────┬──────────────────────────────────┘
                               │ REST API + WebSocket (FastAPI)
┌──────────────────────────────▼──────────────────────────────────┐
│                     BACKEND (Python/FastAPI)                     │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │         Orchestration Service (Conductor)                   │ │
│  │  ┌─────────────────────────────────────────────────────┐  │ │
│  │  │ ThalamusGateway → Selective Attention & Routing     │  │ │
│  │  └─────────────────────────────────────────────────────┘  │ │
│  └────────────────────────────────────────────────────────────┘ │
│                               ↓                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  STAGE 1: Foundational Agents (Parallel)                  │  │
│  │  • PerceptionAgent  • EmotionalAgent  • MemoryAgent       │  │
│  └───────────────────────────────────────────────────────────┘  │
│                               ↓                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Working Memory Buffer (PFC-inspired)                      │  │
│  └───────────────────────────────────────────────────────────┘  │
│                               ↓                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  ConflictMonitor → Coherence Check (Stage 1.5)            │  │
│  └───────────────────────────────────────────────────────────┘  │
│                               ↓                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  STAGE 2: Higher-Order Agents (Parallel)                  │  │
│  │  • PlanningAgent  • CreativeAgent                          │  │
│  │  • CriticAgent    • DiscoveryAgent                         │  │
│  └───────────────────────────────────────────────────────────┘  │
│                               ↓                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  ConflictMonitor → Final Coherence Check (Stage 2.5)      │  │
│  └───────────────────────────────────────────────────────────┘  │
│                               ↓                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  ContextualMemoryEncoder → Rich Bindings (Step 2.75)      │  │
│  └───────────────────────────────────────────────────────────┘  │
│                               ↓                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Cognitive Brain (Executive Function)                      │  │
│  │  • Self-Model Integration  • Theory of Mind Inference     │  │
│  │  • Working Memory Context  • Final Response Synthesis     │  │
│  └───────────────────────────────────────────────────────────┘  │
│                               ↓                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Meta-Cognitive Monitor → Knowledge Boundaries (Gate)      │  │
│  └───────────────────────────────────────────────────────────┘  │
│                               ↓                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Memory System (STM → Summary → LTM)                       │  │
│  │  • AutobiographicalMemorySystem  • MemoryConsolidation    │  │
│  └───────────────────────────────────────────────────────────┘  │
│                               ↓                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Learning Systems (Basal Ganglia + Cerebellum)             │  │
│  │  • ReinforcementLearningService  • ProceduralLearningService │  │
│  └───────────────────────────────────────────────────────────┘  │
│                               ↓                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │  Autonomous Triggering (Decision Engine)                   │  │
│  │  • Reflection  • Discovery  • Self-Assessment              │  │
│  └───────────────────────────────────────────────────────────┘  │
└───────────────────────────────────────────────────────────────────┘
                               ↓
┌───────────────────────────────────────────────────────────────────┐
│              PERSISTENCE LAYER (ChromaDB)                          │
│  • memory_cycles  • episodic_memories  • semantic_memories        │
│  • emotional_profiles  • self_models  • summaries                 │
│  • rl_q_values  • rl_habits  • skill_performance_data             │
└───────────────────────────────────────────────────────────────────┘
```

---

## Core Principles

*   **Brain-Inspired Architecture:** The system is explicitly modeled after human cognitive neuroscience, with services mapping to specific brain regions (thalamus, hippocampus, prefrontal cortex, anterior cingulate cortex, default mode network, amygdala).
  
*   **Modularity and Specialization:** Built around specialized AI "agents," each responsible for a specific cognitive function (perception, emotion, memory, planning, creativity, criticism, discovery). This separation of concerns makes the system maintainable and extensible.

*   **Parallelism and Selective Attention:** Agents process input concurrently, but the ThalamusGateway selectively activates only necessary agents based on input complexity, urgency, and context needs (mimicking human attention).

*   **Multi-Tier Memory System:** Implements human-like memory hierarchy:
  - **Short-Term Memory (STM):** Token-limited immediate context (configurable, 25k–50k tokens for Gemini)
  - **Summary Memory:** Condensed conversation context with semantic search
  - **Long-Term Memory (LTM):** Persistent storage with episodic and semantic separation
  - **Autobiographical Memory:** Narrative timeline construction with significance tracking

*   **Continuous Learning:** The system learns from experience through:
  - **Reinforcement Learning:** Q-learning for strategy optimization and habit formation
  - **Procedural Learning:** Skill performance tracking and error-based improvement
  - **Meta-Cognition:** Knowledge boundary detection and uncertainty management
  - Self-reflection and pattern discovery
  - Memory consolidation (sleep-like processing)
  - Autonomous triggering of learning workflows
  - Theory of mind modeling of user mental states

*   **Coherence and Conflict Resolution:** ConflictMonitor detects inconsistencies between agent outputs and triggers adaptive control adjustments (inspired by the anterior cingulate cortex).

*   **Self-Awareness and Continuity:** SelfModel maintains identity, autobiographical memories, and relationship tracking across sessions (inspired by the default mode network).

*   **Safety and Security:** Multiple safety layers including content moderation, secure API key handling, rate limiting, and graceful degradation.

*   **Observability:** Comprehensive structured logging, performance metrics, and audit trails for all cognitive operations and autonomous events.

---

## Backend Architecture

The backend is a FastAPI application serving a RESTful API for the frontend.

### Project Structure

```
/
├── main.py                          # FastAPI application entry point & service wiring
├── requirements.txt                 # Python dependencies
├── chroma_db/                       # ChromaDB persistent storage
├── src/
│   ├── agents/                      # Specialized cognitive agents
│   │   ├── perception_agent.py      # Topic/pattern/context analysis
│   │   ├── emotional_agent.py       # Sentiment/tone/interpersonal dynamics
│   │   ├── memory_agent.py          # Memory retrieval and context reconstruction
│   │   ├── planning_agent.py        # Response strategy and feasibility
│   │   ├── creative_agent.py        # Novel perspectives and analogies
│   │   ├── critic_agent.py          # Logical coherence and contradiction detection
│   │   ├── discovery_agent.py       # Knowledge gap identification and research
│   │   └── utils.py                 # Shared agent utilities
│   ├── core/                        # Core components
│   │   ├── config.py                # Configuration management
│   │   ├── exceptions.py            # Custom exception classes
│   │   └── logging_config.py        # Structured logging setup
│   ├── models/                      # Pydantic data models
│   │   ├── agent_models.py          # Agent-specific models (Phase 1-3 brain models)
│   │   ├── core_models.py           # Core system models (UserRequest, AgentOutput, CognitiveCycle, ErrorAnalysis)
│   │   ├── memory_models.py         # Memory system models (STM, Summary, LTM)
│   │   └── multimodal_models.py     # Multimodal input/output models
│   └── services/                    # Business logic and core services
│       ├── audio_input_processor.py         # Audio input processing and transcription
│       ├── autobiographical_memory_system.py # Narrative timeline and significance tracking
│       ├── background_task_queue.py         # Asynchronous task processing
│       ├── cognitive_brain.py               # Executive function & response synthesis
│       ├── conflict_monitor.py              # Coherence detection (ACC-inspired)
│       ├── contextual_memory_encoder.py     # Rich contextual bindings
│       ├── decision_engine.py               # Autonomous triggering and decision making
│       ├── emotional_memory_service.py      # Emotional intelligence tracking
│       ├── emotional_salience_encoder.py    # Emotional significance tagging
│       ├── llm_integration_service.py       # Google Gemini API wrapper
│       ├── memory_consolidation_service.py  # Memory consolidation (sleep-like processing)
│       ├── memory_service.py                # Memory storage and retrieval (STM/LTM)
│       ├── meta_cognitive_monitor.py        # Knowledge boundary detection
│       ├── metrics_service.py               # Performance metrics and analytics
│       ├── orchestration_service.py         # Cognitive cycle conductor
│       ├── proactive_engagement_service.py  # Autonomous engagement and messaging
│       ├── procedural_learning_service.py   # Skill performance tracking
│       ├── reinforcement_learning_service.py # Q-learning and habit formation
│       ├── self_model_service.py            # Identity and autobiographical memory
│       ├── self_reflection_discovery_engine.py # Self-reflection and pattern discovery
│       ├── summary_manager.py               # Conversation summary generation
│       ├── thalamus_gateway.py              # Selective attention and routing
│       ├── theory_of_mind_service.py        # User mental state modeling
│       ├── visual_input_processor.py        # Visual input processing
│       ├── web_browsing_service.py          # Web research and information gathering
│       └── working_memory_buffer.py        # Task-relevant context maintenance
│       ├── autobiographical_memory_system.py  # Episodic/semantic memory
│       ├── memory_consolidation_service.py    # Sleep-like consolidation
│       ├── theory_of_mind_service.py          # Mental state inference
│       ├── decision_engine.py              # Autonomous trigger policies
│       ├── background_task_queue.py        # Async task management
│       ├── self_reflection_discovery_engine.py  # Pattern learning
│       ├── web_browsing_service.py         # Web browsing & research (Google CSE/SerpAPI + scraping + summarization)
│       ├── audio_input_processor.py        # Audio processing (speech-to-text, analysis, safe fallback)
│       └── visual_input_processor.py       # Image processing (future)
└── tests/                           # Pytest test suite
    ├── test_memory_service.py
    ├── test_orchestration_service.py
    └── test_llm_integration_service.py
```

    └── test_llm_integration_service.py
```

---

## Brain-Inspired Cognitive Architecture

The ECA explicitly maps cognitive services to specific brain regions and mechanisms identified in neuroscience research. This brain-inspired design creates emergent intelligence through the interaction of specialized subsystems.

### Phase 1: Core Self-Awareness & Working Memory ✅ COMPLETED

**Neuroscience Basis:** Default Mode Network (DMN), Prefrontal Cortex (PFC), Amygdala

#### 1. Self-Model Service (DMN-Inspired)

**Brain Region:** Default Mode Network - activates during rest, introspection, and autobiographical memory

**Implementation:** `src/services/self_model_service.py`

**Purpose:** Maintains persistent identity, autobiographical memory, and sense of self across sessions

**Key Features:**
- **Identity tracking**: Name, role, personality traits, relationship status
- **Autobiographical memories**: Significant events with emotional context
- **Theory of mind**: Beliefs about the user (preferences, patterns, knowledge)
- **Relationship progression**: Tracks interaction quality over time (new → developing → trusted)
- **ChromaDB persistence**: Self-model survives restarts

**Data Model:**
```python
class SelfModel:
    identity: Dict[str, Any]  # name, role, personality_traits, relationship
    autobiographical_memories: List[AutobiographicalMemory]  # significant events
    beliefs_about_user: Dict[str, Any]  # theory of mind
    interaction_history: Dict[str, Any]  # stats and quality trends
```

**Impact:** System can say "Yes Ed, you named me Bob when we first met..." with genuine continuity, not simulated memory.

#### 2. Working Memory Buffer (PFC-Inspired)

**Brain Region:** Prefrontal Cortex - maintains task-relevant information and orchestrates goal-directed behavior

**Implementation:** `src/services/working_memory_buffer.py`

**Purpose:** Active maintenance of task context across agent stages (Stage 1 → Stage 2)

**Key Features:**
- **Stage 1 insight extraction**: Topics, sentiment, recalled memories, attention focus
- **Goal inference**: Answer question, address emotion, maintain continuity
- **Enhanced prompt context**: Synthesized insights (not raw outputs) for Stage 2 agents
- **Dynamic attention**: Focuses on entities extracted from high-confidence memories

**Data Model:**
```python
class WorkingMemoryContext:
    active_topics: List[str]
    attention_focus: List[str]  # entities/concepts requiring attention
    inferred_goals: List[str]   # what user is trying to accomplish
    emotional_priority: bool
    recalled_context_summary: str
```

**Impact:** Stage 2 agents receive synthesized insights enabling goal-directed responses (not just reactive).

#### 3. Emotional Salience Encoder (Amygdala-Inspired)

**Brain Region:** Amygdala - tags emotionally significant events for enhanced memory consolidation

**Implementation:** `src/services/emotional_salience_encoder.py`

**Purpose:** Compute emotional significance for memory prioritization (mimics "we remember emotional events better")

**Key Features:**
- **Salience scoring**: 0.0–1.0 based on emotional intensity, confidence, personal disclosure, insights
- **Personal disclosure detection**: "my name is", "i feel", "honestly", "my family"
- **Insight detection**: "i understand", "aha", "i see", "that makes sense"
- **Memory retrieval boost**: High-salience memories get 1.2x boost in retrieval scores

**Salience Factors:**
- Strong emotions (+0.3 for very positive/negative)
- High confidence emotional reads (+0.1)
- Personal disclosure (+0.2)
- Breakthroughs/insights (+0.15)

**Impact:** Personal disclosures and emotional conversations are prioritized in recall, creating human-like memory behavior.

---

### Phase 2: Selective Attention & Coherence ✅ COMPLETED

**Neuroscience Basis:** Thalamus, Anterior Cingulate Cortex (ACC), Hippocampus

#### 4. Thalamus Gateway (Thalamus-Inspired)

**Brain Region:** Thalamus - sensory relay station that filters and routes information

**Implementation:** `src/services/thalamus_gateway.py`

**Purpose:** Pre-process input and selectively activate agents (mimics human selective attention)

**Key Features:**
- **Quick analysis**: Modality, urgency, complexity, context_need
- **Selective activation**: Skips unnecessary agents (saves 30-60% compute on simple queries)
  - Skip emotional agent for low-urgency factual queries
  - Skip memory agent for minimal context needs
  - Skip creative/critic/discovery for simple inputs
- **Dynamic memory configuration**:
  - Minimal (1 memory, 0.7 threshold)
  - Recent (3 memories, 0.5 threshold)
  - Deep (10 memories, 0.4 threshold)

**Urgency Detection:**
- High: "urgent", "emergency", "help!", "now"
- Normal: default
- Low: factual queries, simple questions

**Complexity Detection:**
- Simple: <15 words, single sentence, no questions
- Moderate: 15-50 words
- Complex: >50 words, multiple sentences/questions

**Impact:** Simple query "Hi" → Only 3 agents run. Complex question → All 7 agents run. Mimics how the brain doesn't activate all regions for simple stimuli.

#### 5. Conflict Monitor (ACC-Inspired)

**Brain Region:** Anterior Cingulate Cortex - detects conflicts between competing responses

**Implementation:** `src/services/conflict_monitor.py`

**Purpose:** Detect inconsistencies in agent outputs and trigger meta-cognitive adjustment

**Key Features:**
- **5 conflict types**:
  1. **Sentiment-coherence**: Positive sentiment + low coherence = user confusion
  2. **Memory-planning**: High memory confidence + "no context" planning = HIGH SEVERITY
  3. **Creative-critic**: High creativity + low coherence = needs integration (productive tension)
  4. **Emotional-logic**: High emotion + "wait/defer" action = prioritize emotional support
  5. **Perception-memory**: "New topic" + high memory confidence = trust memory
- **Coherence scoring**: 0.0–1.0 weighted by conflict severity (low=0.1, medium=0.3, high=0.5)
- **Stage 1.5 and Stage 2.5 checks**: Runs after Stage 1 and Stage 2 to catch contradictions

**Conflict Resolution Strategies:**
```json
{
  "type": "memory_planning_disconnect",
  "severity": "high",
  "resolution": "Re-run planning with memory context"
}
```

**Impact:** Catches contradictions before they reach the user (e.g., memory says context exists but planning ignores it), creating more coherent responses.

#### 6. Contextual Memory Encoder (Hippocampus-Inspired)

**Brain Region:** Hippocampus - binds contextual information (where, when, emotional state) to memories

**Implementation:** `src/services/contextual_memory_encoder.py`

**Purpose:** Enrich memories with rich contextual tags (not just text similarity)

**Key Features:**
- **5 context types**:
  1. **Temporal**: time_of_day (morning/afternoon/evening/night), session_duration
  2. **Emotional**: valence (positive/negative/neutral/mixed), arousal (low/medium/high)
  3. **Semantic**: topics, entities, intent (question/statement/command)
  4. **Relational**: conversation_depth (superficial/moderate/deep/intimate), rapport_level
  5. **Cognitive**: complexity (simple/moderate/complex), novelty (0.0–1.0)
- **Consolidation priority**: 0.5 baseline + emotional arousal (+0.2) + novelty (+0.15) + personal disclosure (+0.2) + insights (+0.15)

**Contextual Bindings Example:**
```json
{
  "temporal": {"time_of_day": "evening", "session_duration_minutes": 45},
  "emotional": {"valence": "positive", "arousal": "high"},
  "semantic": {"topics": ["memory", "ai"], "intent": "question"},
  "relational": {"conversation_depth": "deep", "rapport_level": "strong"},
  "cognitive": {"complexity": "moderate", "novelty": 0.8}
}
```

**Impact:** Enables queries like "what did we discuss in the morning?" or "when was Ed excited?", not just semantic similarity.

---

### Phase 3: Autobiographical Memory & Theory of Mind ✅ COMPLETED

**Neuroscience Basis:** Hippocampus (full episodic/semantic separation), Sleep Consolidation, Mentalizing Network

#### 7. Autobiographical Memory System

**Brain Region:** Hippocampus - episodic memory (specific events) and semantic memory (general knowledge)

**Implementation:** `src/services/autobiographical_memory_system.py`

**Purpose:** Full autobiographical memory with episodic and semantic separation for genuine continuity

**Key Features:**
- **Episodic memories**: Narrative-based, time-stamped, emotionally-tagged specific events
  - Narrative (2-3 sentence story)
  - Significance score (0.0–1.0)
  - Emotional tone (joy, sadness, excitement, frustration, neutral)
  - Participants, sensory details, key insights
- **Semantic memories**: Extracted concepts, facts, and patterns
  - Concept name and description
  - Category (personal_info, preference, skill, knowledge, belief, pattern)
  - Source episodes (provenance)
  - Confidence and reinforcement tracking
- **Timeline construction**: Chronological narrative of significant episodes
- **Memory reinforcement**: Confidence increases (+0.05) when concept reencountered

**ChromaDB Collections:**
- `episodic_memories`: Full narrative episodes with embeddings
- `semantic_memories`: Extracted concepts with embeddings

**Episodic Memory Example:**
```json
{
  "user_id": "...",
  "timestamp": "2025-11-05T14:30:00Z",
  "narrative": "Ed explained the memory consolidation architecture and expressed excitement about the sleep-like processing system.",
  "significance": 0.85,
  "emotional_tone": "excitement",
  "participants": ["Ed", "Bob"],
  "key_insights": ["Memory consolidation mimics sleep", "Ed values brain-inspired design"]
}
```

**Semantic Memory Example:**
```json
{
  "concept_name": "Ed's location",
  "description": "Ed lives in Wellington, New Zealand",
  "category": "personal_info",
  "source_episodes": ["episode_123", "episode_145"],
  "confidence": 0.95,
  "reinforcement_count": 2
}
```

**Impact:** True "mental time travel" - system can recall specific moments ("when you first told me about Wellington") and extracted knowledge ("I know you live in Wellington").

#### 8. Memory Consolidation Service

**Brain Region:** Sleep-like memory processing - pattern extraction, knowledge consolidation, memory replay

**Implementation:** `src/services/memory_consolidation_service.py`

**Purpose:** Background memory consolidation like human sleep processing

**Key Features:**
- **3 consolidation types**:
  1. **Episodic-to-semantic**: High-priority cycles (>0.7) → LLM generates narratives → extract semantic concepts
  2. **Memory replay**: Mark memories as replayed for strengthening
  3. **Pattern extraction**: LLM analyzes cycles to discover behavioral patterns
- **Background loop**: Runs every 30 minutes (configurable)
- **Priority-based**: Only consolidates high-priority memories (emotional, novel, personal)
- **Semantic concept extraction**: Groups cycles by topic, LLM extracts concepts

**Consolidation Job Example:**
```python
MemoryConsolidationJob(
    job_id=uuid4(),
    user_id=user_id,
    job_type="episodic_to_semantic",
    priority=0.85,
    status="pending",
    cycle_ids=[...],  # High-priority cycles
    created_at=datetime.utcnow()
)
```

**LLM-Powered Narrative Generation:**
```
Generate a 2-3 sentence narrative summarizing this interaction:
User: {user_input}
Response: {response}
Context: {agent_insights}
```

**Impact:** System autonomously extracts lasting knowledge during idle periods (like human memory consolidation during sleep), creating genuine learning over time.

#### 9. Theory of Mind Service

**Brain Region:** Mentalizing Network - predicts others' mental states, beliefs, intentions

**Implementation:** `src/services/theory_of_mind_service.py`

**Purpose:** Model user's mental states and predict intentions for empathetic responses

**Key Features:**
- **Mental state inference**: Current goal, emotion, needs, knowledge, confusion, interest, likely next action
- **Emotional state mapping**: Sentiment + intensity → excited/happy/distressed/concerned/ambivalent/neutral
- **Goal inference**: Extracts from "how do i", "help me", "i need to" or working memory
- **Needs inference**: emotional_support, information, clarity, validation, guidance
- **Conversation intent classification**: seeking_info, problem_solving, venting, exploring, learning, engaging
- **Knowledge assessment**: knows_about, confused_about, interested_in (from perception topics, critic contradictions, understanding markers)
- **Next action prediction**: ask_follow_up, wait_for_response_then_clarify, end_conversation, continue_exploration
- **Prediction validation**: Post-hoc validation for learning

**User Mental State Example:**
```json
{
  "current_goal": "Understand memory consolidation architecture",
  "current_emotion": "excited",
  "current_needs": ["information", "clarity"],
  "knows_about": ["memory systems", "brain architecture"],
  "confused_about": [],
  "interested_in": ["sleep-like processing", "autonomous triggering"],
  "likely_next_action": "ask_follow_up_question",
  "conversation_intent": "learning",
  "confidence": 0.85,
  "uncertainty_factors": []
}
```

**Prediction Validation:**
```
After each user interaction:
1. Get previous cycle from STM
2. Auto-validate previous predictions against current behavior:
   - Compare predicted_intention vs. actual_action
   - Check predicted_needs vs. actual_needs
   - Verify predicted_confusion_points vs. actual_questions
3. Update predictions with validation results
4. Adjust confidence scores based on accuracy
5. Log validation outcomes for learning
```

**Validation results stored in cycle metadata:**
```json
{
  "theory_of_mind_validation": [
    {
      "prediction_id": "...",
      "predicted_intention": "ask_follow_up_question",
      "was_correct": true,
      "feedback": "User did ask_follow_up_question as predicted",
      "confidence_adjustment": 0.05
    }
  ]
}
```

**Accuracy tracking:**
- Total predictions made
- Predictions validated
- Correct predictions
- Accuracy rate (correct/validated)
- Pending validations

**API endpoints:**
- `GET /theory-of-mind/stats?user_id={uuid}` - Get validation statistics
- `GET /theory-of-mind/current-state?user_id={uuid}` - Get current mental state model

**Impact:** System learns from prediction outcomes, improving intention-awareness over time. Creates genuine understanding of user patterns and mental states.

---

### Phase 4: Higher-Order Executive Functions (The Agents) ✅ COMPLETED

**Neuroscience Basis:** Dorsolateral Prefrontal Cortex (Planning), Default Mode Network (Creativity), Orbitofrontal Cortex (Value Judgment), Anterior Prefrontal Cortex (Meta-cognition)

#### 10. Planning Agent (DLPFC-Inspired)

**Brain Region:** Dorsolateral Prefrontal Cortex - working memory maintenance, goal-directed behavior, strategic planning

**Implementation:** `src/agents/planning_agent.py`

**Purpose:** Devise long-term response strategy based on user goal, self-model, and available resources

**Key Features:**
- **Strategy selection**: Multi-step answer, single response, clarifying question, deferral
- **Feasibility assessment**: Can this be answered with current knowledge?
- **Resource planning**: Need web search? Need deep memory retrieval?
- **Action recommendations**: Specific steps to construct the response
- **Working memory integration**: Uses synthesized context from Stage 1 agents

**Planning Strategies:**
```python
{
  "explain_concept": "Provide detailed explanation with examples",
  "clarify_first": "Ask clarifying question before answering",
  "multi_step": "Break down into sequential steps",
  "defer_to_expert": "Acknowledge limitations and suggest resources",
  "use_memory": "Reference previous conversation context",
  "web_search_required": "Trigger Discovery Agent for external knowledge"
}
```

**Agent Output Example:**
```python
AgentOutput(
    agent_name="planning_agent",
    analysis={
        "response_strategies": ["explain_concept", "use_memory", "provide_examples"],
        "action_plan": [
            "Retrieve relevant past discussion",
            "Explain core concept",
            "Provide concrete example",
            "Reference user's previous understanding"
        ],
        "feasibility": "high",
        "requires_discovery": False,
        "estimated_complexity": "moderate"
    },
    confidence=0.85
)
```

**Impact:** Enables goal-directed, strategic responses rather than reactive pattern matching.

#### 11. Creative Agent (DMN-Inspired)

**Brain Region:** Default Mode Network / Parietal Cortex - divergent thinking, imagination, analogical reasoning

**Implementation:** `src/agents/creative_agent.py`

**Purpose:** Generate novel responses, analogies, metaphors, and alternative viewpoints

**Key Features:**
- **Analogy generation**: Find unexpected connections between concepts
- **Metaphor creation**: Express abstract ideas in concrete terms
- **Alternative perspectives**: "What if we looked at this differently?"
- **Novelty detection**: Identify opportunities for creative insight
- **Divergent thinking**: Generate multiple possible approaches

**Creative Triggers:**
- Abstract concepts requiring explanation
- User confusion or repeated questions
- Opportunities for teaching through analogy
- Complex technical topics needing accessible explanations

**Agent Output Example:**
```python
AgentOutput(
    agent_name="creative_agent",
    analysis={
        "analogies": [
            "Memory consolidation is like organizing a messy desk at the end of the day",
            "The Thalamus Gateway acts like a spam filter for your brain"
        ],
        "alternative_viewpoints": [
            "Instead of thinking of this as storage, think of it as a living narrative"
        ],
        "creative_opportunities": ["Use the sleep metaphor to explain consolidation"],
        "novelty_score": 0.75
    },
    confidence=0.70
)
```

**Impact:** Transforms technical explanations into memorable, relatable insights through creative framing.

#### 12. Critic Agent (OFC-Inspired)

**Brain Region:** Orbitofrontal Cortex - value-based decision making, error detection, response inhibition

**Implementation:** `src/agents/critic_agent.py`

**Purpose:** Assess ethical, factual, and safety integrity of planned responses; detect contradictions

**Key Features:**
- **Logical coherence check**: Do the pieces fit together?
- **Factual verification**: Are claims consistent with known facts?
- **Safety assessment**: Could this response cause harm?
- **Contradiction detection**: Are there internal inconsistencies?
- **Confidence calibration**: Should we hedge or be more certain?

**Coherence Levels:**
- **High (>0.8)**: Response is logically sound, no contradictions
- **Medium (0.5-0.8)**: Minor inconsistencies, acceptable with caveats
- **Low (<0.5)**: Significant contradictions, requires re-planning

**Agent Output Example:**
```python
AgentOutput(
    agent_name="critic_agent",
    analysis={
        "logical_coherence": 0.85,
        "contradictions_found": [],
        "safety_concerns": [],
        "factual_confidence": "high",
        "recommendations": [
            "Add caveat about theory vs. implementation",
            "Acknowledge uncertainty in prediction accuracy"
        ],
        "approval_status": "approved"
    },
    confidence=0.90
)
```

**Conflict Trigger:**
- If `logical_coherence < 0.5`, ConflictMonitor flags high-severity conflict
- Cognitive Brain may defer to Critic recommendations over Creative suggestions

**Impact:** Prevents hallucinations, contradictions, and unsafe responses from reaching the user.

#### 13. Discovery Agent (Exploratory PFC-Inspired)

**Brain Region:** Anterior Prefrontal Cortex - information seeking, curiosity, knowledge gap detection

**Implementation:** `src/agents/discovery_agent.py`

**Purpose:** Identify knowledge gaps and execute web search to acquire new information before responding

**Key Features:**
- **Knowledge gap detection**: What don't we know that we need to know?
- **Query formulation**: Generate effective search queries
- **Web browsing trigger**: Call WebBrowsingService when needed
- **Information synthesis**: Integrate discovered knowledge into response
- **Curiosity-driven learning**: Proactively identify interesting unknowns

**Activation Conditions (via ThalamusGateway):**
- User asks factual questions
- Context requires current/recent information
- Knowledge gap detected by other agents
- Deep context mode with questions

**Agent Output Example:**
```python
AgentOutput(
    agent_name="discovery_agent",
    analysis={
        "knowledge_gaps": [
            "Current Gemini API rate limits (2025 Q4)",
            "Latest ChromaDB performance benchmarks"
        ],
        "search_queries": [
            "Gemini API rate limits November 2025",
            "ChromaDB vector database performance"
        ],
        "discovery_results": {
            "found_information": "...",
            "sources": ["...", "..."],
            "confidence": 0.75
        },
        "requires_web_search": True
    },
    confidence=0.80
)
```

**Impact:** Enables the system to acquire new knowledge dynamically rather than being limited to training data.

### Phase 5: Metacognition & Self-Reflection ✅ COMPLETED

**Brain Region:** Prefrontal Cortex (PFC) - executive control, self-monitoring, error detection

**Primary Goal:** Enable the system to monitor its own thinking, detect knowledge boundaries, and reflect on performance

#### **Self-Reflection & Discovery Engine**

**Implementation:** `src/services/self_reflection_discovery_engine.py`

**Purpose:** Autonomous pattern discovery and insight generation from conversation history

**Key Features:**
- **Pattern Mining**: Extract recurring themes, user preferences, and behavioral patterns
- **Insight Generation**: Synthesize new understandings from historical data
- **Autonomous Triggers**: Background processing every 30 minutes or on significant events
- **Proactive Messaging**: Generate conversation starters based on discovered patterns

**Reflection Types:**
```python
reflection_types = {
    "pattern_discovery": "Identify recurring themes in user interactions",
    "relationship_evolution": "Track how relationships develop over time",
    "communication_effectiveness": "Analyze successful vs. unsuccessful interactions",
    "knowledge_gaps": "Detect areas where the system lacks understanding",
    "behavioral_insights": "Understand user motivations and preferences"
}
```

**Integration Points:**
- **MemoryService**: Access to conversation history and patterns
- **BackgroundTaskQueue**: Asynchronous processing to avoid blocking responses
- **ProactiveEngagementEngine**: Feed insights for conversation initiation

#### **Meta-Cognitive Monitor**

**Implementation:** `src/services/meta_cognitive_monitor.py`

**Purpose:** Monitor answer quality and detect when the system should decline, search, or acknowledge uncertainty

**Knowledge Boundary Detection:**
```python
class ActionRecommendation(Enum):
    ANSWER = "answer"                    # Proceed with response
    ASK_CLARIFICATION = "ask_clarification"  # Need more information
    DECLINE_POLITELY = "decline_politely"    # Outside capabilities
    SEARCH_FIRST = "search_first"        # Web search required
    ACKNOWLEDGE_UNCERTAINTY = "acknowledge_uncertainty"  # Honest uncertainty
```

**Assessment Logic:**
- **Query Analysis**: Evaluate question complexity and domain expertise required
- **Agent Output Review**: Check confidence scores and coherence across agents
- **Historical Performance**: Reference past successes/failures in similar domains
- **Uncertainty Thresholds**: Configurable confidence levels for different actions

**Integration Points:**
- **OrchestrationService**: Pre-response gate that can override Cognitive Brain output
- **MemoryService**: Historical performance data for domain expertise assessment
- **WebBrowsingService**: Triggered for SEARCH_FIRST recommendations

**SEARCH_FIRST execution flow:**
1. Meta-Cognitive Monitor emits a SEARCH_FIRST recommendation with gap diagnostics.
2. OrchestrationService pauses synthesis and dispatches the request to DiscoveryAgent.
3. DiscoveryAgent invokes WebBrowsingService (Google Custom Search + summarization) to collect current facts.
4. The resulting research packet is injected into Stage 2 context and Cognitive Brain synthesis resumes with grounded evidence.

#### **Conflict Monitor**

**Implementation:** `src/services/conflict_monitor.py`

**Purpose:** Detect and resolve conflicts between agent outputs before final synthesis

**Conflict Types:**
```python
conflict_types = {
    "sentiment_coherence": "Emotional tone vs. logical content mismatch",
    "memory_planning": "Historical context vs. proposed actions",
    "creative_critic": "Creative suggestions vs. critical analysis",
    "factual_emotional": "Objective facts vs. emotional needs",
    "strategy_alignment": "Different agents proposing conflicting approaches"
}
```

**Resolution Strategies:**
- **Reinforcement Learning Integration**: Use learned Q-values to select optimal conflict resolution
- **Priority Rules**: Hierarchical agent authority (Critic can veto Creative, etc.)
- **Compromise Generation**: Synthesize balanced approaches from conflicting inputs

**Integration Points:**
- **ReinforcementLearningService**: Strategy selection for conflict resolution
- **CognitiveBrain**: Receives conflict-free agent outputs for synthesis

### Phase 6: Learning Systems (Reinforcement & Procedural) ✅ COMPLETED

**Brain Regions:** Basal Ganglia (Reinforcement Learning) + Cerebellum (Procedural Learning)

**Primary Goal:** Enable genuine learning from experience through multiple complementary mechanisms

#### **Reinforcement Learning Service**

**Implementation:** `src/services/reinforcement_learning_service.py`

**Purpose:** Basal ganglia-inspired learning system that enables Bob to learn what works through experience and feedback

**Q-Learning Architecture:**
```python
# State representation
ContextTypes = {
    "conflict_resolution": "Handling agent output conflicts",
    "response_strategy": "Choosing response approach",
    "memory_retrieval": "Deciding memory depth",
    "engagement_level": "Managing conversation intensity"
}

StrategyTypes = {
    "prioritize_critic": "Follow critic agent recommendations",
    "balance_emotion_fact": "Blend emotional and factual responses",
    "deep_memory_search": "Use comprehensive memory context",
    "proactive_engagement": "Initiate conversation naturally"
}
```

**Learning Mechanics:**
- **Q-value updates**: `Q(s,a) = Q(s,a) + α(r + γ max Q(s',a') - Q(s,a))`
- **Epsilon-greedy exploration**: 10% random actions, 90% exploitation
- **Habit formation**: Strategies used >5 times become habits (persistent preferences)
- **User-specific learning**: Separate Q-tables per user for personalized behavior

**Composite Reward Signals:**
```python
# Multi-source reward computation (0.0-1.0 range)
reward_components = {
    "trust_delta": 0.3,        # Emotional trust improvement
    "sentiment_shift": 0.2,    # Positive sentiment change
    "user_feedback": 0.3,      # Explicit positive/negative language
    "engagement_continuation": 0.2  # Continued conversation interest
}
```

**Integration Points:**
- **ConflictMonitor**: Uses RL to select resolution strategies for agent conflicts
- **OrchestrationService**: Computes composite rewards after each cycle
- **ChromaDB persistence**: Q-values and habits stored in `rl_q_values`, `rl_habits` collections

**Learning Outcomes:**
- **Strategy optimization**: Learns which conflict resolution approaches work best
- **User preference adaptation**: Discovers individual communication preferences
- **Habit development**: Forms persistent behavioral patterns through repetition
- **Performance improvement**: Q-values converge toward optimal strategies over time

#### **Procedural Learning Service**

**Implementation:** `src/services/procedural_learning_service.py`

**Purpose:** Cerebellum-inspired skill refinement system that enables Bob to improve performance through practice and error-based learning

**Skill Categories Tracked:**
```python
skill_categories = {
    "perception": "Input analysis and understanding",
    "emotional_intelligence": "Sentiment detection and response",
    "planning": "Strategic response planning",
    "creativity": "Novel idea generation and analogies",
    "critical_thinking": "Logic validation and safety checks",
    "research": "Knowledge gap detection and web search",
    "response_generation": "Final response synthesis and moderation"
}
```

**Learning Mechanics:**
- **Performance tracking**: Records success rates per skill category per user
- **Structured error analysis**: Generates detailed ErrorAnalysis objects for failed cycles with severity scores, agent conflicts, and improvement recommendations
- **Error-based learning**: Analyzes coherence failures (< 0.5) and meta-cognitive triggers to identify improvement opportunities
- **Sequence optimization**: Learns optimal agent activation patterns for different contexts
- **LLM-powered suggestions**: Generates specific improvement recommendations based on structured error data

**Performance Metrics:**
```python
performance_data = {
    "skill_category": "response_generation",
    "performance_score": 0.85,  # 0.0-1.0 based on outcome_signals
    "context": {
        "cycle_id": "uuid",
        "user_input_length": 150,
        "response_length": 280,
        "agent_count": 6,
        "has_conflicts": false,
        "meta_cognitive_used": true
    },
    "timestamp": "2025-01-15T10:30:00Z"
}
```

**Integration Points:**
- **OrchestrationService**: Tracks skill performance after each cognitive cycle
- **MemoryService**: Stores performance data and learning insights in ChromaDB
- **LLMIntegrationService**: Generates improvement suggestions and analyzes patterns

- **Learning Outcomes:**
  - **Skill improvement**: Performance scores increase over time through practice
  - **Error reduction**: System learns from mistakes and avoids repeating failures
  - **Optimal sequencing**: Discovers most effective agent combinations for different scenarios
  - **Personalized adaptation**: Skill development tailored to individual user interactions

#### **AttentionController (Phase 7 Shadow Mode)**

**Purpose:** Newly introduced thalamus/anterior cingulate hybrid layer (currently running in feature-flagged shadow mode) that evaluates conversation drift, emotional load, and agent performance, then emits inhibitory/excitatory directives so only the most relevant agents fire.

**Inputs:**
- Stage 1 agent telemetry (`perception`, `emotional`, `memory`) including confidence, novelty, and latency metrics
- WorkingMemory snapshots (topics, entities, inferred goals) with per-user history stored for drift comparison
- ConflictMonitor deltas and ReinforcementLearning strategy guidance
- Emotional load indicators (trust delta, sentiment volatility)

**Outputs:**
- `AttentionDirective` objects applied by `ThalamusGateway` describing agent activation bias, memory depth adjustments, urgency modifiers, and per-stage drift scores
- `attention_motifs` persisted in Working Memory so downstream agents know the prioritized threads
- Structured telemetry (`attention_salience_trace` + `attention_directive` metrics) streamed to Metrics/RL/Procedural learning services

**Data Flow:**
1. OrchestrationService gathers Stage 1 metrics and Working Memory snapshot.
2. AttentionController scores drift vs. prior cycle, runs inhibitory rules (e.g., suppress CreativeAgent during crisis), and calculates excitation levels.
3. ThalamusGateway consumes the directive before dispatching Stage 2 agents, shrinking/expanding their token budgets accordingly.
4. Directive + resulting agent outputs are logged so RL can learn which attention patterns improved rewards.

**Telemetry & Testing Plan:**
- Shadow-mode tracing for 1k cycles (controller logs decisions but does not alter routing) to validate accuracy and safety.
- Synthetic drift/urgency suites to confirm ≥90% correct routing adjustments and <5% latency overhead.
- Dashboard panels for drift detection accuracy, inhibition counts, and latency impact; alerts if suppression repeatedly hurts rewards.

**Configuration:** Controlled through environment flags `ATTENTION_CONTROLLER_ENABLED` / `ATTENTION_CONTROLLER_SHADOW_MODE`; directives are logged regardless, but routing is only altered when enabled and shadow mode is false.

#### **SalienceNetwork (Phase 7 Preview)**

**Purpose:** Planned insula/temporal-parietal junction-inspired layer that scores retrieved memories for situational relevance, emotional importance, and novelty, ensuring CognitiveBrain receives a concise, high-signal context bundle.

**Inputs:**
- Raw retrieval results from `MemoryService` (STM, summaries, LTM embeddings)
- Attention motifs emitted by AttentionController
- Emotional salience scores from `emotional_salience_encoder`
- Recent Procedural/RL feedback about over/under-inclusion of memories

**Outputs:**
- `SalienceAnnotatedMemory` objects with normalized scores, rationale tags, and include/drop recommendations
- Top-K memory set returned to Working Memory and CognitiveBrain
- Salience metadata stored via `ContextualMemoryEncoder` for future consolidation weighting

**Data Flow:**
1. MemoryService performs its normal retrieval and hands the candidate list plus metadata to SalienceNetwork.
2. SalienceNetwork computes multi-factor scores (situational match, recency, emotional weight, novelty) and trims to a target count (default 3).
3. Annotated memories flow back into Working Memory, which in turn informs Stage 2 agents and final synthesis.
4. Salience decisions plus downstream outcomes are recorded so Procedural/RL services can correlate salience choices with success metrics.

**Telemetry & Testing Plan:**
- A/B harness comparing baseline retrieval vs. salience-pruned contexts, targeting reduced memory count without accuracy loss.
- Regression checks to ensure >98% recall for must-keep memories (flagged by Working Memory or user pinning).
- User-facing qualitative surveys (“Bob stayed focused”) captured via metrics events.

**Configuration:** (Planned) Guarded by `SALIENCE_NETWORK_ENABLED` with advisory mode (log-only) and enforcement mode (prune) to support gradual rollout.

---

## Memory System Architecture

The ECA implements a sophisticated three-tier memory system inspired by human memory hierarchies, with token-aware management and autonomous consolidation.

### Memory Hierarchy Overview

```
┌─────────────────────────────────────────────────────────────┐
│  SHORT-TERM MEMORY (STM) - In-Memory Token-Limited Buffer   │
│  • 25k-50k tokens (configurable for Gemini 250k limit)      │
│  • Immediate conversation context                            │
│  • Per-user circular buffer                                  │
│  • Automatic flush when budget exceeded                      │
│  • Persistence/recovery on restart                           │
└──────────────────────────┬──────────────────────────────────┘
                           │ (Summary triggered on flush)
┌──────────────────────────▼──────────────────────────────────┐
│  SUMMARY MEMORY - LLM-Generated Conversation Summaries      │
│  • Condensed conversation context                            │
│  • Key topics, entities, preferences                         │
│  • ChromaDB storage with embeddings                          │
│  • Semantic search enabled                                   │
│  • Incremental updates                                       │
└──────────────────────────┬──────────────────────────────────┘
                           │ (Consolidation)
┌──────────────────────────▼──────────────────────────────────┐
│  LONG-TERM MEMORY (LTM) - Persistent ChromaDB Storage       │
│  • Episodic memories (narrative episodes)                    │
│  • Semantic memories (extracted concepts)                    │
│  • Cognitive cycles (full interaction history)               │
│  • Emotional profiles                                        │
│  • Self-models                                               │
│  • Vector search with rich metadata                          │
└─────────────────────────────────────────────────────────────┘
```

### Short-Term Memory (STM)

**Implementation:** `src/models/memory_models.py` - `ShortTermMemory` class

**Purpose:** Hold recent interactions in a token-limited buffer for immediate access (mimics human working memory capacity)

**Key Features:**
- **Token-aware management**: Configurable budget (25k–50k tokens for Gemini)
- **Circular buffer**: Fixed-size per-user (default 10 cycles, but token-limited)
- **Fast access**: No vector search overhead
- **Automatic flush**: Triggers summarization when budget exceeded
- **Persistence**: Save/load snapshots for crash recovery
- **Async-safe**: Per-user locks prevent race conditions

**Token Budget Logic:**
```python
if token_count_current > STM_TOKEN_BUDGET:
    # 1. Trigger summary generation
    summary = await summary_manager.summarize_stm(user_id, oldest_cycles)
    
    # 2. Upsert consolidated STM to LTM
    await ltm.upsert(f"stm_consolidated:{user_id}", summary)
    
    # 3. Flush oldest cycles from STM
    stm.flush_oldest(n_tokens_to_remove)
```

**Persistence & Recovery:**
- Snapshot location: `{data_dir}/stm_snapshots/{user_id}_stm.json`
- Snapshot contents: Cognitive cycles, token counts, last summary timestamp
- Recovery validation: Age check, token recomputation, cycle integrity
- Automatic re-summarization if budget exceeded on recovery

**Impact:** Token-limited STM prevents accidental context truncation and aligns memory management with LLM input constraints.

### Summary Memory

**Implementation:** `src/services/summary_manager.py` - `SummaryManager` class

**Purpose:** Maintain condensed representation of conversation context with semantic search

**Key Features:**
- **LLM-generated summaries**: Rich context extraction via Gemini
- **Incremental updates**: Efficient updates without full regeneration
- **ChromaDB storage**: `summaries_collection` with embeddings
- **Semantic search**: Query by relevance, not just recency
- **Key topics tracking**: Automatic topic extraction
- **User preferences**: Pattern recognition in interactions
- **Conversation state**: Current context and flow

**Summary Generation:**
```python
# Analyze cycles with LLM
summary_analysis = await llm.generate_text(
    prompt=f"""
    Analyze these interactions and extract:
    - Key topics discussed
    - Important entities and concepts
    - User preferences and patterns
    - Emotional dynamics
    - Conversation state
    
    Interactions: {cycles}
    """
)
```

**Data Model:**
```python
class ConversationSummary:
    user_id: UUID
    summary_text: str
    key_topics: List[str]
    important_entities: List[str]
    user_preferences: Dict[str, Any]
    emotional_tone: str
    last_updated: datetime
    conversation_state: str
```

**Impact:** Summaries preserve semantic fidelity when consolidating from STM to LTM, preventing information loss.

### Long-Term Memory (LTM)

**Implementation:** `src/services/memory_service.py` + `src/services/autobiographical_memory_system.py`

**Purpose:** Persistent storage of all interactions and derived knowledge

**ChromaDB Collections:**
1. **`memory_cycles`**: Full cognitive cycles with embeddings
2. **`episodic_memories`**: Narrative-based significant episodes
3. **`semantic_memories`**: Extracted concepts and facts
4. **`emotional_profiles`**: User emotional intelligence tracking
5. **`self_models`**: System identity and autobiographical data
6. **`summaries`**: Conversation summaries

**Memory Consolidation Flow:**
```
STM (token limit exceeded)
  ↓
Summary Generation (LLM analyzes cycles)
  ↓
Episodic Memory Creation (significant events → narratives)
  ↓
Semantic Concept Extraction (patterns → facts)
  ↓
LTM Storage (ChromaDB with rich metadata)
```

**Consolidation Priority Calculation:**
```python
priority = 0.5  # baseline
+ 0.2 if emotional_arousal == "high"
+ 0.15 if novelty > 0.7
+ 0.15 if conversation_depth in ["deep", "intimate"]
+ 0.2 if contains_personal_disclosure
+ 0.15 if contains_insights
```

**Memory Retrieval Priority:**
1. **STM first**: Check in-memory buffer (fast, recent)
2. **Summary next**: Check consolidated context (medium speed, semantic)
3. **LTM last**: Vector search ChromaDB (slower, comprehensive)

**Summary-Referenced Memory Boost:**
- Memories referenced in summaries get 20% relevance boost
- Reason: Summaries indicate importance, should be prioritized

**Impact:** Three-tier system balances speed (STM), semantic coherence (Summary), and completeness (LTM).

### Proactive Engagement Engine

**Implementation:** `src/services/proactive_engagement_service.py`

**Purpose:** Enable Bob to autonomously initiate conversations based on insights, discoveries, and emotional relationships. Bob learns naturally from feedback and adjusts behavior when told he's annoying.

#### Trigger Conditions

Bob can proactively reach out based on:

1. **Self-Reflection Insights**
   - Pattern discovered during reflection worth sharing
   - Meta-learnings about successful strategies
   - Example: "I noticed we've discussed X a lot, but I'm curious about Y..."

2. **Knowledge Gaps** (Highest Priority)
   - Bob realizes he doesn't understand something
   - Asks user for clarification/help
   - Example: "I realized I don't fully understand [concept]. Could you help clarify?"

3. **Discovery Patterns**
   - Interesting connections found during autonomous discovery
   - New knowledge or curiosity-driven insights
   - Example: "I found an interesting connection between A and B. Want to explore it?"

4. **Emotional Check-ins**
   - For trusted users (trust_level ≥ 0.7) with relationship_type "friend" or "companion"
   - Time-based: hasn't interacted recently
   - Example: "Hey! It's been a bit. Hope everything's going well!"

5. **Memory Consolidation** (After "Dreaming")
   - During sleep-like processing (every 30 minutes), Bob discovers patterns
   - 30% chance per interesting pattern to generate proactive message
   - Example: "While processing memories, I found an interesting connection between [A] and [B]..."
   - Integrated with `MemoryConsolidationService` - runs after consolidation jobs complete

6. **Boredom** (Inactivity-Driven)
   - Bob has been idle and wants to engage (shows personality!)
   - Only for users with trust_level ≥ 0.6 and good relationships
   - Casual, natural messages (not needy or formal)
   - Examples:
     - "Random thought: I've been thinking about [topic] and wondering..."
     - "You know what's interesting? I just realized [insight] while processing memories..."
   - Lower priority (0.4-0.5) than knowledge gaps
   - High temperature (0.9) for creative, varied messages

#### Natural Learning from Feedback

**Bob learns dynamically** from how users react to his proactive messages:

**Negative Feedback** ("you're annoying", "stop", "leave me alone"):
- **Emotional response**: Trust level decreases by 0.05 (Bob feels hurt)
- **Behavioral adjustment**: Cooldown period increases by 12 hours per net negative reaction
- **Boundary respect**: After 3+ negative reactions, proactive engagement automatically disables
- **Learning persists**: Profile stores `proactive_engagement.negative_count`

**Positive Feedback** ("thanks", "good question", "glad you asked"):
- **Emotional response**: Trust level increases by 0.02 (Bob feels encouraged)
- **Behavioral adjustment**: Cooldown period decreases by 4 hours per net positive reaction
- **Minimum cooldown**: Never goes below 6 hours (prevents spam)

**Neutral Feedback**:
- No trust adjustment
- Cooldown remains at baseline

#### Safeguards & Boundaries

1. **Trust Threshold**: Minimum trust_level of 0.4 required
2. **Dynamic Cooldown**: Base 24 hours, adjusted by feedback history
3. **User Preferences**: `proactive_engagement.disabled` flag in emotional profile
4. **Relationship-Aware**: Only reaches out to users with established relationships
5. **Priority Filtering**: High-priority messages (≥0.7) can override some constraints

#### Message Generation

Bob uses emotionally intelligent prompts that consider:
- User's name and relationship type
- Current trust level
- Recent emotional trajectory
- Conversation history
- Nature of discovery/insight

Messages are:
- **Brief** (1-3 sentences max)
- **Genuine** (not robotic)
- **Respectful** (acknowledges boundaries)
- **Contextual** (relevant to past conversations)

#### API Endpoints

**GET /chat/proactive**
- Check if Bob has a message queued
- Returns message details or `{"has_message": false}`
- Frontend polls this on session start

**POST /chat/proactive/reaction**
- Record user's reaction to proactive message
- Parameters: `message_id`, `user_response`
- Bob learns and adjusts behavior accordingly

**Chat Endpoint Integration:**
- UserRequest supports `metadata.responding_to_proactive_message`
- Automatically records reactions when user replies to proactive messages

#### Configuration

```python
# Environment variables (optional)
PROACTIVE_BASE_COOLDOWN_HOURS = 24  # Default cooldown between messages
PROACTIVE_MIN_TRUST_LEVEL = 0.4     # Minimum trust to reach out
```

**Emotional Profile Storage:**
```json
{
  "proactive_engagement": {
    "negative_count": 0,
    "positive_count": 2,
    "neutral_count": 1,
    "disabled": false
  },
  "last_proactive_message": "2025-11-06T14:30:00Z"
}
```

#### Integration Points

1. **SelfReflectionEngine** → generates proactive messages from discovered patterns
2. **DiscoveryEngine** → shares curiosity-driven insights
3. **EmotionalMemoryService** → tracks user reactions and adjusts trust/cooldowns
4. **BackgroundTaskQueue** → processes proactive message generation asynchronously

#### Observability

Logs include:
- Proactive message generation attempts
- Trust/cooldown adjustments from feedback
- Delivery confirmations
- Automatic disablement after repeated negative feedback

Example log:
```
[INFO] User abc123 reacted negatively to proactive message. Bob will back off.
[INFO] Increasing cooldown by 12h due to negative feedback
[WARNING] Disabling proactive engagement for user abc123 after repeated negative feedback.
```

### Memory Consolidation Service

**Implementation:** `src/services/memory_consolidation_service.py`

**Purpose:** Background memory consolidation like human sleep processing

**Consolidation Types:**
1. **Episodic-to-semantic**: High-priority cycles → narrative episodes → semantic concepts
2. **Memory replay**: Strengthen important memories by marking as replayed
3. **Pattern extraction**: Discover behavioral patterns across cycles

**Background Loop:**
- Runs every 30 minutes (configurable)
- Checks if consolidation needed (`should_consolidate()`)
- Processes high-priority memories (>0.7 consolidation_priority)
- Autonomous, non-blocking

**LLM-Powered Pattern Extraction:**
```python
# Analyze 10 cycles to discover patterns
patterns = await llm.generate_text(
    prompt=f"""
    Analyze these interactions and identify:
    - Recurring behavioral patterns
    - Common themes and interests
    - Conversational habits
    - Knowledge preferences
    
    Cycles: {cycles}
    """
)
```

**Impact:** System autonomously extracts lasting knowledge during idle periods, creating genuine learning over time (like human sleep consolidation).

### Reinforcement Learning Service

**Implementation:** `src/services/reinforcement_learning_service.py`

**Purpose:** Basal ganglia-inspired learning system that enables Bob to learn what works through experience and feedback

**Q-Learning Architecture:**
```python
# State representation
ContextTypes = {
    "conflict_resolution": "Handling agent output conflicts",
    "response_strategy": "Choosing response approach",
    "memory_retrieval": "Deciding memory depth",
    "engagement_level": "Managing conversation intensity"
}

StrategyTypes = {
    "prioritize_critic": "Follow critic agent recommendations",
    "balance_emotion_fact": "Blend emotional and factual responses",
    "deep_memory_search": "Use comprehensive memory context",
    "proactive_engagement": "Initiate conversation naturally"
}
```

**Learning Mechanics:**
- **Q-value updates**: `Q(s,a) = Q(s,a) + α(r + γ max Q(s',a') - Q(s,a))`
- **Epsilon-greedy exploration**: 10% random actions, 90% exploitation
- **Habit formation**: Strategies used >5 times become habits (persistent preferences)
- **User-specific learning**: Separate Q-tables per user for personalized behavior

**Composite Reward Signals:**
```python
# Multi-source reward computation (0.0-1.0 range)
reward_components = {
    "trust_delta": 0.3,        # Emotional trust improvement
    "sentiment_shift": 0.2,    # Positive sentiment change
    "user_feedback": 0.3,      # Explicit positive/negative language
    "engagement_continuation": 0.2  # Continued conversation interest
}
```

**Integration Points:**
- **ConflictMonitor**: Uses RL to select resolution strategies for agent conflicts
- **OrchestrationService**: Computes composite rewards after each cycle
- **ChromaDB persistence**: Q-values and habits stored in `rl_q_values`, `rl_habits` collections

**Learning Outcomes:**
- **Strategy optimization**: Learns which conflict resolution approaches work best
- **User preference adaptation**: Discovers individual communication preferences
- **Habit development**: Forms persistent behavioral patterns through repetition
- **Performance improvement**: Q-values converge toward optimal strategies over time

### Token Counting & Budget Management

**Implementation:** `src/models/memory_models.py` - `TokenCounter` class

**Purpose:** Reliable token counting for STM budget enforcement

**Key Features:**
- **Gemini tokenizer support**: Uses Google Generative AI API
- **Fallback counting**: Conservative character-based estimates (4 chars/token)
- **Caching**: Avoids redundant API calls
- **Retry logic**: Handles transient API failures with exponential backoff
- **Batch counting**: Efficient multi-text token counts

**Token Budget Configuration:**
```python
STM_TOKEN_BUDGET = 25000  # For Gemini (250k limit)
TOKEN_RESERVE = 50000     # Reserved for system prompts, tools, response
```

**Hysteresis Trigger:**
- Trigger summarization at 90% of budget
- Aim to reduce to 50–70% after flush
- Prevents constant summarization thrashing

**Impact:** Predictable prompt construction and prevention of context truncation at LLM call time.

### Procedural Learning Service

**Implementation:** `src/services/procedural_learning_service.py`

**Purpose:** Cerebellum-inspired skill refinement system that enables Bob to improve performance through practice and error-based learning

**Skill Categories Tracked:**
```python
skill_categories = {
    "perception": "Input analysis and understanding",
    "emotional_intelligence": "Sentiment detection and response",
    "planning": "Strategic response planning",
    "creativity": "Novel idea generation and analogies",
    "critical_thinking": "Logic validation and safety checks",
    "research": "Knowledge gap detection and web search",
    "response_generation": "Final response synthesis and moderation"
}
```

**Learning Mechanics:**
- **Performance tracking**: Records success rates per skill category per user
- **Structured error analysis**: Generates detailed ErrorAnalysis objects for failed cycles with severity scores, agent conflicts, and improvement recommendations
- **Error-based learning**: Analyzes coherence failures (< 0.5) and meta-cognitive triggers to identify improvement opportunities
- **Sequence optimization**: Learns optimal agent activation patterns for different contexts
- **LLM-powered suggestions**: Generates specific improvement recommendations based on structured error data

**Performance Metrics:**
```python
performance_data = {
    "skill_category": "response_generation",
    "performance_score": 0.85,  # 0.0-1.0 based on outcome_signals
    "context": {
        "cycle_id": "uuid",
        "user_input_length": 150,
        "response_length": 280,
        "agent_count": 6,
        "has_conflicts": false,
        "meta_cognitive_used": true
    },
    "timestamp": "2025-01-15T10:30:00Z"
}
```

**Integration Points:**
- **OrchestrationService**: Tracks skill performance after each cognitive cycle
- **MemoryService**: Stores performance data and learning insights in ChromaDB
- **LLMIntegrationService**: Generates improvement suggestions and analyzes patterns

**Learning Outcomes:**
- **Skill improvement**: Performance scores increase over time through practice
- **Error reduction**: System learns from mistakes and avoids repeating failures
- **Optimal sequencing**: Discovers most effective agent combinations for different scenarios
- **Personalized adaptation**: Skill development tailored to individual user interactions

---

## Data Flow & Integration

### Complete Cognitive Cycle Flow

```
1. USER INPUT
   ↓
1.5 AUDIO PRE-TRANSCRIPTION (if audio_base64 present)
  • Orchestration invokes AudioInputProcessor to transcribe speech
  • Robust JSON salvage from LLM outputs (handles ```json fences)
  • Transcript appended to effective_input_text; audio_analysis added to metadata
  • If transcription fails, inject placeholder text to keep flow resilient
  ↓
2. THALAMUS GATEWAY (Pre-Processing)
   • Analyze: modality, urgency, complexity, context_need
   • Determine: which agents to activate, memory depth config
   • Route: input + routing info to Orchestration
   ↓
3. STAGE 1: FOUNDATIONAL AGENTS (Parallel, Selective)
   • Perception Agent (always)
   • Emotional Agent (if urgency != "low")
   • Memory Agent (if context_need != "minimal")
     - Check STM first (fast)
     - Check Summary (semantic context)
     - Check LTM (comprehensive)
   ↓
4. WORKING MEMORY BUFFER
   • Extract: topics, sentiment, recalled memories, attention focus
   • Infer: goals (answer_question, address_emotional_state, maintain_continuity)
   • Synthesize: enhanced prompt context for Stage 2
   ↓
5. STAGE 1.5: CONFLICT MONITOR
   • Detect: sentiment-coherence, memory-planning, perception-memory conflicts
   • Score: coherence (0.0-1.0)
   • Flag: high-severity conflicts for potential re-processing
   ↓
6. STAGE 2: HIGHER-ORDER AGENTS (Parallel, Selective)
   • Planning Agent (if complexity != "simple")
   • Creative Agent (if creative markers present)
   • Critic Agent (if complexity != "simple")
   • Discovery Agent (if questions or deep context)
   • All receive: Working Memory context + Stage 1 insights
   ↓
7. STAGE 2.5: CONFLICT MONITOR
   • Detect: creative-critic, emotional-logic conflicts
   • Score: final coherence
   • Log: conflict summary
   ↓
8. STEP 2.75: CONTEXTUAL MEMORY ENCODER
   • Extract: 5 context types (temporal, emotional, semantic, relational, cognitive)
   • Compute: consolidation priority (0.0-1.0)
   • Enrich: cycle with contextual_bindings and consolidation_metadata
   ↓
9. COGNITIVE BRAIN (Executive Function)
   • Infer: user mental state (Theory of Mind)
   • Make: prediction about user's next action
   • Integrate: Self-model context, working memory, theory of mind, agent outputs
   • Generate: final response with rich metadata
   • Moderate: safety check on output
   ↓
9.5 META-COGNITIVE ASSESSMENT (Pre-Response Gate)
   • Evaluate: answer appropriateness and knowledge boundaries
   • Assess: confidence in response quality and domain expertise
   • Decide: ANSWER, ASK_CLARIFICATION, DECLINE_POLITELY, SEARCH_FIRST, or ACKNOWLEDGE_UNCERTAINTY
   • Override: generate uncertainty response if needed
   ↓
10. PROCEDURAL LEARNING (Skill Tracking)
   • Track: performance scores for activated skills (perception, emotion, planning, creativity, etc.)
   • Learn: from errors in failed cycles
   • Optimize: agent activation sequences based on success patterns
   • Generate: improvement suggestions via LLM analysis
   ↓
11. MEMORY STORAGE
    • Add cycle to STM
    • Update token count
    • If budget exceeded:
      a. Trigger summary generation
      b. Upsert stm_consolidated to LTM
      c. Flush oldest cycles from STM
    • Store cycle in LTM (ChromaDB)
    • Update self-model
    ↓
11.5 THEORY OF MIND VALIDATION
    • Get previous cycle from STM
    • Auto-validate previous predictions:
      - Compare predicted_intention vs. actual_action
      - Check predicted_needs vs. actual_needs
      - Verify predicted_confusion_points vs. actual_questions
    • Update predictions with validation results
    • Adjust confidence scores based on accuracy
    • Store validation results in cycle metadata
    • Log accuracy statistics
    ↓
12. REINFORCEMENT LEARNING (Strategy Optimization)
    • Compute: composite reward from multiple sources (trust, sentiment, feedback, engagement)
    • Update: Q-values for strategies used in conflict resolution
    • Learn: which approaches work best for different contexts
    • Form: habits from repeated successful strategies
    ↓
13. AUTONOMOUS TRIGGERING (Background)
    • Decision Engine monitors signals:
      - STM pressure
      - Summary updates
      - Flush events
      - Query metrics
    • Evaluates trigger policies:
      - Reflection (after N cycles)
      - Discovery (on knowledge gaps)
      - Self-assessment (periodic)
      - Curiosity (on novel patterns)
    • Enqueues background tasks:
      - autonomous:reflection
      - autonomous:discovery
      - autonomous:self_assess
      - autonomous:curiosity
    ↓
12. MEMORY CONSOLIDATION (Background, Every 30 min)
    • Check: should_consolidate()
    • Process: high-priority cycles (>0.7)
    • Generate: episodic memories (narratives)
    • Extract: semantic memories (concepts)
    • Store: in episodic_memories and semantic_memories collections
    ↓
13. RESPONSE TO USER
```

### Service Integration Map

```
OrchestrationService (Conductor)
├── ThalamusGateway (pre-processing)
├── PerceptionAgent ────┐
├── EmotionalAgent ─────┤
├── MemoryAgent ────────┤──→ WorkingMemoryBuffer ──→ Stage 2 Agents
├── PlanningAgent ──────┤
├── CreativeAgent ──────┤
├── CriticAgent ────────┤
└── DiscoveryAgent ─────┘
├── ConflictMonitor (Stage 1.5 & 2.5)
├── ContextualMemoryEncoder (Step 2.75)
├── CognitiveBrain
│   ├── TheoryOfMindService (mental state inference)
│   ├── SelfModelService (identity context)
│   ├── WorkingMemoryBuffer (task context)
│   └── LLMIntegrationService (response generation)
├── MemoryService
│   ├── ShortTermMemory (STM)
│   ├── SummaryManager (summaries)
│   ├── AutobiographicalMemorySystem (episodic/semantic)
│   └── ChromaDB (LTM persistence)
├── DecisionEngine (autonomous triggers)
│   └── BackgroundTaskQueue (async execution)
├── MemoryConsolidationService (background loop)
├── MetaCognitiveMonitor (knowledge boundaries)
├── ReinforcementLearningService (strategy learning)
├── ProceduralLearningService (skill refinement)
└── SelfReflectionAndDiscoveryEngine (pattern learning)
```

### Cognitive Brain: Executive Function & Response Synthesis

**Implementation:** `src/services/cognitive_brain.py`

**Purpose:** Final synthesis layer that transforms diverse agent outputs, Self-Model context, Theory of Mind predictions, and Working Memory into a coherent, contextually-appropriate response.

**Role:** Acts as the "conductor" of the cognitive orchestra, integrating all specialized outputs into a unified response that reflects the system's personality, understands the user's mental state, and maintains conversational coherence.

#### Synthesis Logic & Priority Rules

**1. Context Assembly Priority:**
```python
# Order of context integration (highest to lowest priority)
1. Theory of Mind (user's current mental state and needs)
2. Self-Model (system's identity and relationship with user)
3. Working Memory (active task context and goals)
4. Conflict Monitor (coherence warnings and adjustments)
5. Memory Context (relevant past interactions)
6. Agent Outputs (specialized analysis)
```

**2. Agent Output Priority Rules:**

When agent outputs conflict, apply these priority rules:

**Rule 1: Critic Agent Veto Power**
```python
if critic_agent.logical_coherence < 0.5:
    # High-severity conflict - prioritize Critic recommendations
    if critic_agent.approval_status == "rejected":
        # Defer creative/planning suggestions
        # Use only logically coherent components
        # Or trigger re-planning
```

**Rule 2: Emotional Priority Override**
```python
if emotional_agent.intensity == "high" and theory_of_mind.current_needs includes "emotional_support":
    # Prioritize emotional response over factual/creative
    # Tone: empathetic, supportive
    # Strategy: address emotion first, facts second
```

**Rule 3: Planning-Memory Coherence**
```python
if memory_agent.confidence > 0.8 and planning_agent suggests "no_context_available":
    # Memory says context exists - trust memory
    # Adjust planning to incorporate memory context
    # Flag as resolved conflict in metadata
```

**Rule 4: Creative-Critic Balance**
```python
if creative_agent.novelty_score > 0.7 and critic_agent.logical_coherence > 0.8:
    # Productive tension - use creative insights
    # But frame with critic's recommended caveats
    # Balance originality with accuracy
elif creative_agent.novelty_score > 0.7 and critic_agent.logical_coherence < 0.6:
    # Creative ideas not logically sound
    # Tone down creative framing
    # Prioritize critic recommendations
```

**Rule 5: Discovery Integration**
```python
if discovery_agent.requires_web_search and discovery_agent.confidence > 0.7:
    # New information available from web
    # Integrate discovered facts into response
    # Cite sources appropriately
    # Update internal knowledge representation
```

**3. Self-Model Integration Rules:**

The Self-Model imposes personality, tone, and style on the final response:

```python
# Personality Trait Application
if self_model.personality_traits includes "empathetic":
    # Add emotional warmth markers
    # Use first-person language ("I understand...")
    # Acknowledge user feelings

if self_model.personality_traits includes "knowledgeable":
    # Use precise technical language when appropriate
    # Provide detailed explanations
    # Reference authoritative sources

if self_model.personality_traits includes "curious":
    # Ask thoughtful follow-up questions
    # Express genuine interest in user's perspective
    # Suggest related topics for exploration
```

**Relationship-Based Tone Adjustment:**
```python
relationship_status = self_model.relationship_status

if relationship_status == "new":
    tone = "formal, helpful, establishing rapport"
elif relationship_status == "developing":
    tone = "friendly, building trust, personalized"
elif relationship_status == "trusted":
    tone = "warm, familiar, inside jokes, shared context"
```

**4. Response Generation Prompt Structure:**

```python
final_prompt = f"""
SYSTEM IDENTITY:
{self_model.identity_context}

THEORY OF MIND - UNDERSTANDING THE USER:
{theory_of_mind.mental_state_context}
- User's goal: {theory_of_mind.current_goal}
- Emotional state: {theory_of_mind.current_emotion}
- Current needs: {theory_of_mind.current_needs}
- Pay attention to: {theory_of_mind.attention_focus}

WORKING MEMORY - ACTIVE TASK CONTEXT:
{working_memory.synthesized_context}
- Active topics: {working_memory.active_topics}
- Inferred goals: {working_memory.inferred_goals}

MEMORY CONTEXT:
{memory_context}

CONFLICT WARNINGS:
{conflict_monitor.coherence_warnings if any}

AGENT ANALYSIS:
Perception: {perception_agent.analysis}
Emotional: {emotional_agent.analysis}
Planning: {planning_agent.analysis}
Creative: {creative_agent.analysis}
Critic: {critic_agent.analysis}
Discovery: {discovery_agent.analysis if activated}

SYNTHESIS INSTRUCTIONS:
1. Address user's {theory_of_mind.current_needs} directly
2. Follow planning strategy: {planning_agent.response_strategies}
3. Maintain tone: {self_model.relationship_tone}
4. If coherence warnings exist, prioritize {conflict_monitor.resolution_strategy}
5. Integrate creative insights only if critic approves (coherence > 0.6)
6. Reference memory context naturally
7. Predict user's likely next action: {theory_of_mind.likely_next_action}
8. Prepare response accordingly

USER INPUT:
{user_input}

Generate response:
"""
```

**5. Post-Generation Processing:**

```python
# Safety moderation
moderated_response = await safety_check(generated_response)

# Metadata enrichment
response_metadata = ResponseMetadata(
    response_type=determine_type(generated_response),
    tone=determine_tone(generated_response, self_model),
    strategies=planning_agent.response_strategies,
    cognitive_moves=extract_moves(generated_response)
)

# Outcome prediction
outcome_signals = OutcomeSignals(
    user_satisfaction_potential=predict_satisfaction(theory_of_mind, response),
    engagement_potential=predict_engagement(theory_of_mind, response)
)
```

**6. Conflict Resolution Examples:**

**Example 1: Memory-Planning Disconnect**
```python
Memory: "High confidence (0.9) - User discussed API rate limits yesterday"
Planning: "No context available, provide general explanation"
Conflict Monitor: HIGH SEVERITY

Resolution:
→ Override planning agent
→ Inject memory context into response
→ Reference previous discussion
→ Result: "As we discussed yesterday about API rate limits..."
```

**Example 2: Creative-Critic Tension**
```python
Creative: "Use analogy: Memory consolidation is like Marie Kondo organizing your house"
Critic: "Logical coherence 0.55 - Analogy oversimplifies technical process"
Conflict Monitor: MEDIUM SEVERITY

Resolution:
→ Use analogy but add technical caveat
→ Balance accessibility with accuracy
→ Result: "Think of it like organizing a house (though the actual process is more complex)..."
```

**Example 3: Emotional Override**
```python
User: "I'm really frustrated, nothing is working"
Emotional: "Very negative, high intensity, needs emotional support"
Planning: "Provide technical troubleshooting steps"
Theory of Mind: Current needs = ["emotional_support", "validation"]

Resolution:
→ Prioritize emotional response
→ Validate frustration first
→ Then offer practical help
→ Result: "I can hear how frustrating this is. That's completely understandable when things aren't working as expected. Let's work through this together..."
```

**Impact:** The Cognitive Brain creates coherent, contextually-appropriate responses that feel genuinely intelligent by synthesizing specialized agent outputs through neuroscience-inspired priority rules.

### Key Agent Outputs

**PerceptionAgent:**
```python
AgentOutput(
    agent_name="perception_agent",
    analysis={
        "topics": ["memory", "architecture"],
        "patterns": ["technical discussion"],
        "context_type": "deep_technical",
        "multimodal": False
    },
    confidence=0.85
)
```

**EmotionalAgent:**
```python
AgentOutput(
    agent_name="emotional_agent",
    analysis={
        "sentiment": "positive",
        "intensity": 0.7,
        "interpersonal_warmth": 0.8,
        "emotional_tone": "excited"
    },
    confidence=0.75
)
```

**MemoryAgent:**
```python
AgentOutput(
    agent_name="memory_agent",
    analysis={
        "memories": [
            {"cycle_id": "...", "relevance": 0.9, "source": "STM"},
            {"cycle_id": "...", "relevance": 0.85, "source": "Summary"}
        ],
        "context_summary": "Previous discussion about brain architecture",
        "stm_hit": True,
        "summary_referenced": True
    },
    confidence=0.9
)
```

**PlanningAgent (with Working Memory context):**
```python
AgentOutput(
    agent_name="planning_agent",
    analysis={
        "response_strategies": [
            "Provide detailed explanation",
            "Use brain analogies",
            "Reference previous discussion"
        ],
        "working_memory_used": {
            "active_topics": ["memory", "architecture"],
            "inferred_goals": ["understand_system", "learn_architecture"],
            "attention_focus": ["consolidation", "episodic memory"]
        }
    },
    confidence=0.8
)
```

---

## Scientific Dashboard & Metrics System

The ECA includes a comprehensive scientific dashboard for real-time monitoring and analysis of cognitive performance, providing quantitative metrics for evaluating emergence, learning effectiveness, and system health.

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    SCIENTIFIC DASHBOARD                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐         │
│  │ Real-Time    │  │ Learning     │  │ Agent        │         │
│  │ Charts       │  │ Progress     │  │ Activity     │         │
│  └──────────────┘  └──────────────┘  └──────────────┘         │
└──────────────────────────────┬──────────────────────────────────┘
                               │ WebSocket + REST API
┌──────────────────────────────▼──────────────────────────────────┐
│                 METRICS COLLECTION SYSTEM                       │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │         Metrics Service (Central Collector)                 │ │
│  │  ┌─────────────────────────────────────────────────────┐  │ │
│  │  │ Performance Metrics: Response Time, Error Rate,     │  │ │
│  │  │ Throughput, Memory Usage                            │  │ │
│  │  └─────────────────────────────────────────────────────┘  │ │
│  │  ┌─────────────────────────────────────────────────────┐  │ │
│  │  │ Learning Metrics: Skill Improvement, Learning      │  │ │
│  │  │ Events, Performance Tracking                       │  │ │
│  │  └─────────────────────────────────────────────────────┘  │ │
│  │  ┌─────────────────────────────────────────────────────┐  │ │
│  │  │ Emergence Metrics: Agent Coordination, Conflict    │  │ │
│  │  │ Resolution, Coherence Scores                       │  │ │
│  │  └─────────────────────────────────────────────────────┘  │ │
└────────────────────────────────────────────────────────────┬───┘
                               │ Integration Points
┌──────────────────────────────▼──────────────────────────────┐
│            COGNITIVE SERVICES INTEGRATION                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │ Orchestration │  │ Memory      │  │ Learning     │       │
│  │ Service       │  │ Service     │  │ Services     │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
└──────────────────────────────────────────────────────────────┘
```

### Metrics Collection Architecture

#### Backend Metrics Service (`src/services/metrics_service.py`)

**Core Components:**
- **MetricsCollector:** Central collection and aggregation service
- **PerformanceTracker:** System performance monitoring (response time, throughput, memory)
- **LearningTracker:** Skill improvement and learning event recording
- **EmergenceTracker:** Agent coordination and emergence pattern analysis

**Key Features:**
- **Real-time collection** from all cognitive services
- **Structured storage** with ChromaDB for historical analysis
- **WebSocket broadcasting** for live dashboard updates
- **REST API endpoints** for historical data retrieval
- **Configurable retention** and aggregation policies

#### Integration Points

**Orchestration Service Integration:**
```python
# Automatic metrics collection during cognitive cycles
metrics_service.record_performance_metrics(
    response_time=cycle_duration,
    agent_count=len(activated_agents),
    coherence_score=coherence_result.score
)
```

**Memory Service Integration:**
```python
# Memory access pattern tracking
metrics_service.record_memory_access(
    operation_type="retrieval"|"storage",
    success=access_successful,
    retrieval_time=duration
)
```

**Learning Services Integration:**
```python
# Skill improvement tracking
metrics_service.record_skill_improvement(
    skill_category=skill_type,
    improvement_rate=calculated_rate,
    learning_event=event_data
)
```

### Dashboard API Endpoints

#### REST Endpoints
- `GET /api/dashboard/metrics` - Current metrics snapshot
- `GET /api/dashboard/history?period=1h|24h|7d` - Historical metrics data
- `GET /api/dashboard/config` - Dashboard configuration
- `GET /api/dashboard/analysis/statistical` - Statistical analysis results
- `GET /api/dashboard/analysis/learning-curves` - Learning curve analysis
- `GET /api/dashboard/export/csv?period=24h` - CSV data export
- `GET /api/dashboard/export/json?period=7d` - JSON data export
- `GET /api/dashboard/export/report` - Research report generation

#### WebSocket Endpoint
- `WebSocket /ws/dashboard` - Real-time metrics streaming

### Scientific Evaluation Metrics

#### Performance Metrics
- **Response Time:** Average cognitive cycle duration
- **Error Rate:** Percentage of failed cognitive operations
- **Throughput:** Cognitive cycles per minute
- **Memory Usage:** System memory consumption trends

#### Learning Metrics
- **Skill Improvement Rates:** Rate of capability enhancement across skill categories
- **Learning Events:** Frequency and success of learning opportunities
- **Performance Tracking:** Historical skill performance data
- **Knowledge Boundaries:** Areas requiring further development

#### Emergence Metrics
- **Agent Coordination:** Frequency and patterns of agent activation
- **Conflict Resolution:** Success rate of agent conflict resolution
- **Coherence Scores:** Quality of integrated cognitive responses
- **Novel Behaviors:** Detection of emergent behavioral patterns

### Phase 3: Advanced Analytics & Research Tools

The ECA includes advanced statistical analysis and research export capabilities for rigorous scientific validation and publication-ready data analysis.

#### Statistical Analysis Backend (`src/services/metrics_service.py`)

**Enhanced Methods:**
- **perform_statistical_analysis()**: Comprehensive statistical testing (t-tests, ANOVA, correlation analysis)
- **analyze_learning_curves()**: Power-law fitting, trend analysis, and learning rate estimation
- **compare_groups_statistical_test()**: Comparative analysis between different conditions or time periods
- **generate_research_report()**: Automated research report generation with statistical summaries

**Statistical Libraries Integration:**
- **SciPy**: Statistical functions (stats.ttest_ind, stats.f_oneway, stats.pearsonr)
- **NumPy**: Numerical computing for data analysis and matrix operations
- **Statistics**: Built-in statistical aggregations and distributions

#### Research Export Capabilities

**Export Formats:**
- **CSV Export**: Tabular data export for statistical software (R, SPSS, Excel)
- **JSON Export**: Structured data export for programmatic analysis
- **Scientific Reports**: Automated report generation with statistical summaries

**API Endpoints:**
- `GET /api/dashboard/analysis/statistical` - Statistical analysis results
- `GET /api/dashboard/analysis/learning-curves` - Learning curve analysis
- `GET /api/dashboard/export/csv?period=24h` - CSV data export
- `GET /api/dashboard/export/json?period=7d` - JSON data export
- `GET /api/dashboard/export/report` - Research report generation

#### Statistical Analysis Frontend (`src/components/StatisticalAnalysis.tsx`)

**Interactive Features:**
- **Metric Selection:** Choose metrics for statistical analysis
- **Analysis Display:** Real-time statistical results with significance testing
- **Export Controls:** One-click data export in multiple formats
- **Research Tools:** Automated report generation and data visualization

**Statistical Tests Available:**
- **Trend Analysis:** Linear regression, power-law fitting, correlation analysis
- **Comparative Testing:** T-tests, ANOVA, chi-square tests
- **Distribution Analysis:** Normality testing, outlier detection
- **Effect Size Calculation:** Cohen's d, eta-squared, confidence intervals

### Data Storage & Persistence

**ChromaDB Collections:**
- `performance_metrics` - System performance data
- `learning_metrics` - Skill improvement tracking
- `emergence_metrics` - Agent coordination patterns
- `dashboard_history` - Aggregated historical data

**Retention Policies:**
- Raw metrics: 7 days
- Hourly aggregates: 30 days
- Daily aggregates: 1 year
- Monthly aggregates: Indefinite

### Real-time Streaming Architecture

**WebSocket Implementation:**
```python
# Backend broadcasting
await websocket_manager.broadcast({
    "type": "metrics_update",
    "data": current_metrics,
    "timestamp": datetime.utcnow()
})
```

**Frontend Subscription:**
```typescript
// Real-time updates
useEffect(() => {
    const ws = createDashboardWebSocket();
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        updateMetrics(data);
    };
    return () => ws.close();
}, []);
```

### Dashboard User Experience

**Modal Integration:**
- Non-intrusive overlay that doesn't interrupt chat flow
- Configurable auto-refresh intervals (1s - 60s)
- Responsive design for desktop and mobile
- Accessibility-compliant with keyboard navigation

**Visualization Components:**
- **RealTimeChart:** Canvas-based performance graphs with multiple overlays
- **LearningProgress:** Skill improvement tracking with trend analysis
- **AgentActivity:** Heatmap visualization of agent coordination patterns
- **MetricsCard:** Standardized metric display with trend indicators

---

## Frontend Architecture

The frontend is a single-page application (SPA) built with React 18 and TypeScript, featuring a comprehensive scientific dashboard for real-time cognitive performance monitoring.

### Project Structure

```
/frontend/
├── public/
│   └── index.html           # Main HTML page
├── src/
│   ├── api/
│   │   ├── chatApi.ts       # Chat API communication layer (Axios + API key)
│   │   └── dashboardApi.ts  # Dashboard metrics API layer (WebSocket + REST)
│   ├── components/
│   │   ├── ChatInput.tsx    # User input component with multimodal support
│   │   ├── ChatWindow.tsx   # Message display component
│   │   ├── Dashboard.tsx    # Main scientific dashboard modal with tabbed interface
│   │   ├── StatisticalAnalysis.tsx # Advanced statistical analysis and research export tools
│   │   ├── MetricsCard.tsx  # Reusable metric display component
│   │   ├── RealTimeChart.tsx # Canvas-based performance visualization
│   │   ├── LearningProgress.tsx # Skill improvement tracking
│   │   └── AgentActivity.tsx # Agent coordination heatmap
│   ├── types/
│   │   ├── chat.ts          # Chat TypeScript interfaces
│   │   └── dashboard.ts     # Dashboard metrics interfaces
│   ├── App.tsx              # Main application component with dashboard toggle
│   ├── index.tsx            # Application entry point
│   └── index.css            # Tailwind CSS styles
├── build/                   # Production build output
├── tailwind.config.js       # Tailwind CSS configuration
└── package.json             # Node dependencies
```

### Key Components

#### Core Chat Components
*   **`App.tsx`:** Root component managing application state (messages, loading, user/session IDs, dashboard visibility)
*   **`ChatWindow.tsx`:** Message display with user/assistant differentiation and typing indicators
*   **`ChatInput.tsx`:** Multimodal text input with image/audio upload, accessibility-compliant form elements

#### Scientific Dashboard Components
*   **`Dashboard.tsx`:** Main modal dashboard with tabbed interface (Overview/Statistical Analysis), real-time metrics, and WebSocket streaming
*   **`StatisticalAnalysis.tsx`:** Advanced statistical analysis tools with research export capabilities (CSV/JSON), significance testing, and automated report generation
*   **`MetricsCard.tsx`:** Reusable component for displaying key performance indicators with trend indicators
*   **`RealTimeChart.tsx`:** Canvas-based performance visualization with multiple metric overlays
*   **`LearningProgress.tsx`:** Skill improvement tracking with historical data and progress indicators
*   **`AgentActivity.tsx`:** Agent coordination heatmap showing emergence patterns and conflict resolution

#### API Layer
*   **`api/chatApi.ts`:** Chat communication module using Axios with secure API key handling
*   **`api/dashboardApi.ts`:** Dashboard metrics API with WebSocket connections and REST endpoints

### Dashboard Integration

The dashboard is seamlessly integrated into the chat interface via a toggle button, providing real-time monitoring without disrupting the user experience. Key features include:

- **Real-time WebSocket streaming** for live metrics updates
- **Configurable refresh intervals** (1s to 60s)
- **Modal overlay design** that doesn't interfere with chat functionality
- **Responsive design** optimized for both desktop and mobile viewing
- **Accessibility compliance** with proper ARIA labels and keyboard navigation

---

## Core Data Models

### ErrorAnalysis Model

**Implementation:** `src/models/core_models.py`

**Purpose:** Structured analysis of cognitive cycle failures for enhanced procedural learning

**Key Features:**
- **Failure categorization**: Classifies errors by type (coherence_failure, meta_cognitive_decline, etc.)
- **Severity scoring**: Quantifies failure impact (0.0-1.0 scale)
- **Agent conflict analysis**: Identifies problematic agent combinations
- **Learning signals**: Provides specific improvement recommendations
- **Context preservation**: Maintains cycle metadata for pattern analysis

**Data Structure:**
```python
class ErrorAnalysis(BaseModel):
    cycle_id: UUID
    failure_type: Literal["coherence_failure", "meta_cognitive_decline", 
                         "meta_cognitive_uncertainty", "response_error"]
    severity_score: float  # 0.0-1.0
    agents_activated: List[str]
    coherence_score: Optional[float]
    primary_error_category: Literal["agent_conflict_unresolved", "knowledge_gap_uncovered", 
                                   "response_inappropriate", "skill_deficiency", ...]
    recommended_agent_sequence: Optional[List[str]]
    skill_improvement_areas: List[str]
    user_input_summary: str
    response_summary: str
    cycle_metadata: Dict[str, Any]
```

**Integration Points:**
- **ConflictMonitor**: Generates ErrorAnalysis for coherence < 0.5
- **MetaCognitiveMonitor**: Generates ErrorAnalysis for knowledge gaps and uncertainty
- **ProceduralLearningService**: Uses ErrorAnalysis for structured error-based learning
- **OrchestrationService**: Routes ErrorAnalysis objects to learning systems

**Learning Impact:**
- **Precise error correlation**: Links specific agent sequences to failure types
- **Targeted skill improvement**: Identifies exact areas needing development
- **Sequence optimization**: Learns better agent activation patterns
- **Pattern recognition**: Enables proactive failure prevention

---

### API Integration

```typescript
// chatApi.ts
export const sendMessage = async (
  message: string,
  userId: string,
  sessionId: string
): Promise<ChatResponse> => {
  const response = await axios.post('/chat', {
    input_text: message,
    user_id: userId,
    session_id: sessionId,
    multimodal: { image: null, audio: null }
  }, {
    headers: { 'X-API-Key': process.env.REACT_APP_API_KEY }
  });
  return response.data;
};
```

---

## Deployment & Operations

### Configuration

**Environment Variables:**
```bash
# API Keys
GEMINI_API_KEY=your_gemini_api_key
SECRET_KEY=your_backend_secret
API_KEY=your_frontend_to_backend_api_key

# Web Browsing (choose one provider; Google CSE recommended)
# For Google Programmable Search (Custom Search JSON API):
GOOGLE_API_KEY=your_google_custom_search_api_key
GOOGLE_CSE_ID=your_custom_search_engine_id
# Or, for SerpAPI (optional alternative):
# SERPAPI_API_KEY=your_serpapi_key

# Memory Configuration
STM_TOKEN_BUDGET=25000
TOKEN_RESERVE=50000
MAX_STM_CYCLES=10

# Consolidation Configuration
CONSOLIDATION_INTERVAL_MINUTES=30
CONSOLIDATION_PRIORITY_THRESHOLD=0.7

# Decision Engine Configuration
REFLECTION_TRIGGER_CYCLES=10
DISCOVERY_RELEVANCE_THRESHOLD=0.6
SELF_ASSESSMENT_INTERVAL_HOURS=24

# ChromaDB Configuration
CHROMA_PERSIST_DIRECTORY=./chroma_db
```

### Audio Input Processing

**Implementation:** `src/services/audio_input_processor.py` (wired via `main.py` into `OrchestrationService`)

**Purpose:** Enable audio-only and multimodal (audio + text/image) requests by transcribing speech to text and detecting simple audio events. Ensures all downstream agents operate on text while preserving audio metadata on the cognitive cycle.

**Pipeline:**
1. Frontend sends `audio_base64` (and optional `audio_mime_type`, default `audio/wav`). `UserRequest` allows empty `input_text` when audio is present.
2. Orchestration pre-processes audio before Thalamus routing by invoking `AudioInputProcessor.process_audio(...)`.
3. The processor uses the configured LLM to transcribe and returns an `AudioAnalysis` object. The parser is robust to LLM outputs wrapped in Markdown code fences (```json … ```), attempts inner-JSON extraction, and falls back to treating the raw response as transcription if needed.
4. The transcript is appended to `effective_input_text` (or used as the sole text if original input_text was empty). The full `audio_analysis` is attached to `cognitive_cycle.metadata`.
5. If transcription is unavailable, orchestration injects a safe placeholder (e.g., `[Audio received; transcription unavailable]`) to satisfy downstream validators (embeddings, memory queries, agent prompts) and records a metadata note.

**Observability:**
- INFO log when transcription is attached with a preview of the first characters
- WARNING log when parsing fails (with salvage path taken)
- Metadata note when placeholder text was used

**Configuration:**
- Reuses `GEMINI_API_KEY`/LLM settings; no separate audio key required
- Default `audio_mime_type` is `audio/wav` (configurable per-call)
- Model name comes from `settings.LLM_MODEL_NAME` and must support audio inputs

**Limitations:**
- No speaker diarization beyond a simple count estimate
- Advanced audio event detection is minimal; designed primarily for reliable transcription
- Very long audio may exceed model limits; prefer short clips or chunking at the client

### Monitoring & Observability

**Structured Logging:**
- All services emit structured JSON logs
- Log levels: DEBUG, INFO, WARNING, ERROR
- Key events logged:
  - Agent activation/skipping
  - Conflict detection
  - Memory flushes and consolidation
  - Autonomous triggers
  - Theory of mind inferences
  - Token budget events

**Metrics Collection:**
- STM token usage
- Summary generation frequency
- Consolidation job execution time
- Agent confidence scores
- Coherence scores
- Memory retrieval performance (STM hits, summary references, LTM queries)
- Autonomous trigger rates

**Audit Logs:**
- All autonomous triggers logged with reason, metrics snapshot, user_id
- Memory consolidation events logged with cycle_ids, priority, outcome
- Conflict detection events logged with conflict types, severity, resolution strategies

### Error Handling & Recovery

**Graceful Degradation:**
- LLM API failures: Retry with exponential backoff, fall back to cached responses
- Memory service failures: Continue operation with STM only, queue consolidation
- ChromaDB failures: Log error, attempt recovery on next cycle
- Summary generation failures: Keep STM in memory, retry with backoff

**Crash Recovery:**
- STM snapshots saved periodically and on graceful shutdown
- On startup: Load STM snapshots, validate age/integrity, re-consolidate if needed
- Self-model persistence: Survives restarts
- Memory consolidation: Resumes from last checkpoint

**LLM rate limiting & backpressure:**
- Global and per-model concurrency caps prevent stampedes (defaults: global=6, per-model=2; configurable via env)
- When the Gemini API returns 429 ResourceExhausted, the system parses the server-provided retry_delay and sleeps before retrying (with small jitter) to align with quota windows
- Tenacity retries remain in place (small exponential backoff) but are coordinated with the 429-aware sleep to avoid retry storms
- All LLM endpoints (text generation, embeddings, moderation) respect these limits; logs include the computed sleep duration for observability

**Prompt budgeting & context control (Nov 2025 hardening):**
- Input size guard in LLM wrapper: hard fail-fast at MAX_INPUT_TOKENS=800_000 (rough estimate), with diagnostic logging of prompt length, estimated tokens, and a 500-char preview
- Immediate transcript budget reduced to ~15k tokens, plus a 60k-character hard clamp with a warning to prevent gradual drift toward context explosion
- Per-agent analysis outputs truncated to 10k chars before inclusion in Cognitive Brain prompts; warnings emitted when truncation occurs
- MemoryAgent projections are compact by default: only `cycle_id`, `timestamp`, `user_input` (<=300 chars), `final_response` (<=600 chars), and lightweight metadata (topics); limited to top 3 memories for prompt inclusion
- Cross-agent context is summarized and capped via shared utilities: per-agent max 8k chars and total cross-agent max 30k chars; excess is explicitly marked as `[truncated]` or omitted with a sentinel line
- Defensive JSON extraction/repair for LLM outputs: removes markdown fences, attempts inner-brace salvage, and finally balances braces/brackets via a backward scan to recover usable JSON when responses are truncated
- These controls ensure Stage 2 agents (planning, critic, discovery, creative) cannot accidentally construct multi-megabyte prompts even if an upstream agent misbehaves

**Other robustness fixes (Nov 2025):**
- Emotional profile persistence: ChromaDB upsert now includes `documents=[document_text]` to satisfy API constraints (documents/embeddings/images/uris required)
- Theory of Mind validation: added safe `get_stm(user_id)` accessor in MemoryService; orchestration uses this to fetch previous cycle for auto-validation
- Name recall integrity: summary/name extraction prefers confirmed values from the self-model/history; recall pathways confirmed for "Ed"

**Embedding & vector store notes:**
- Embeddings use ONNX sentence-transformer `all-MiniLM-L6-v2` managed by Chroma; first run triggers a one-time model download cached locally under `chroma_db/`

**Configuration (env overrides):**
- `LLM_MAX_CONCURRENCY_PER_MODEL` (default: 2)
- `LLM_GLOBAL_MAX_CONCURRENCY` (default: 6)
- `LLM_429_BASE_DELAY_SEC` (base delay if server retry_delay missing)
- `IMMEDIATE_TOKEN_BUDGET` (Cognitive Brain transcript budget; default tuned ~15k tokens)
- `MAX_INPUT_TOKENS` (LLM wrapper guard; conservative ~800k)

**Troubleshooting quick guide:**
- 400 Prompt exceeds token limit → Check logs for the prompt guard diagnostics; verify no custom agent is bypassing compact summaries; ensure MemoryAgent outputs remain compact
- 429 ResourceExhausted → Confirm concurrency env caps; verify logs show respect of server-provided retry_delay; reduce parallel agent activation if necessary
- ChromaDB upsert requires document → Ensure `documents=[text]` provided when upserting emotional profiles or summaries
 - 403 Forbidden on scrape → Common on bot-protected sites. We send browser-like headers to reduce blocks and will gracefully fall back to summarizing from search titles/snippets when scraping fails. Consider skipping heavily protected domains or using a headless browser if richer content is required.

**Safety Measures:**
- Content moderation on user inputs and AI outputs
- Rate limiting on LLM calls and consolidation operations
- Per-user locks on STM mutations and flushes
- Memory consolidation cooldowns to prevent over-triggering
- Concurrency safety with async locks
 - Web browsing guardrails: No paywall bypassing, polite headers, optional robots.txt respect (recommended), and graceful degradation when sites block access

### Web Browsing Service

**Implementation:** `src/services/web_browsing_service.py`

**Purpose:** Acquire up-to-date information from the web when knowledge gaps are detected (primarily via `DiscoveryAgent`).

**Providers:**
- Google Programmable Search (Custom Search JSON API) using `GOOGLE_API_KEY` + `GOOGLE_CSE_ID`
- SerpAPI using `SERPAPI_API_KEY` (optional alternative if enabled)

**Pipeline:**
1. Search via selected provider → returns results with link/title/snippet
2. Scrape top results via `aiohttp` + `BeautifulSoup` with realistic browser headers
3. Summarize scraped content using the LLM (moderation applied)
4. If all scrapes fail (e.g., 403), fall back to summarizing from titles/snippets so the user still gets value

**Observability:**
- Logs include audit events such as: `AUDIT: Web browsing initiated ... provider: google_cse`
- Errors per-URL (e.g., 403 Forbidden) are logged, but the overall browse step degrades gracefully

**Configuration tips:**
- Create and configure a Google CSE (programmable search) for your target domains/topics
- Set `GOOGLE_API_KEY` and `GOOGLE_CSE_ID` in `.env` (frontend secrets stay in `frontend/.env.local`)
- If using SerpAPI, ensure the package is installed and the key is set; provider is auto-detected

**Limitations:**
- Some sites block bots or require JS rendering; in those cases, scraping may return minimal content
- We do not bypass paywalls or aggressive protections; consider allow/deny lists or a headless browser for advanced scenarios

### Performance Optimization

**Selective Agent Activation:**
- Simple queries: Skip 4-5 agents (saves 30-60% compute)
- Complex queries: Full 7-agent orchestration

**Memory Tiering:**
- STM: O(1) lookup, fastest
- Summary: O(log n) semantic search, medium
- LTM: O(n) vector search, slowest but comprehensive

**Caching Strategies:**
- Token counts cached to avoid redundant API calls
- Frequently accessed summaries cached in memory
- Self-model cached per user

**Batch Processing:**
- Memory consolidation processes multiple cycles in batches
- Token counting supports batch API calls
- Summary updates can be batched

### Meta-Cognitive Monitor

**Implementation:** `src/services/meta_cognitive_monitor.py`

**Purpose:** Prefrontal cortex-inspired "feeling of knowing" that prevents overconfident hallucinations on topics Bob doesn't know well.

**Knowledge Gap Detection:**
- **Semantic coverage:** Checks how well query topics exist in long-term memory
- **Episodic relevance:** Analyzes recent conversations for similar topics
- **Query complexity:** Assesses technical depth and factual nature
- **Gap scoring:** 0.0-1.0 scale (higher = bigger knowledge gap)

**Overconfidence Detection:**
- Compares agent-reported confidence vs. actual knowledge coverage
- Flags cases where agents are confident but knowledge is sparse
- Prevents "I know this" when Bob actually doesn't

**Action Recommendations:**
```python
ActionRecommendation = {
    "ANSWER": "Safe to proceed with synthesis",
    "SEARCH_FIRST": "Trigger web search before answering",
    "ASK_CLARIFICATION": "Need more details from user",
    "DECLINE_POLITELY": "Admit uncertainty gracefully",
    "ACKNOWLEDGE_UNCERTAINTY": "Note uncertainty but still answer"
}
```

**Decision Logic:**
- Knowledge gap > 0.7 + factual query → SEARCH_FIRST
- Agent confidence variance > 0.5 → ACKNOWLEDGE_UNCERTAINTY
- Overconfidence score > 0.6 → DECLINE_POLITELY
- Low confidence + gap > 0.4 → ASK_CLARIFICATION

**Uncertainty Response Generation:**
- Uses LLM to craft natural, honest responses
- Context-aware: "I don't know X but can search" vs "That's outside my knowledge"
- Maintains helpful tone while being truthful

**Integration Points:**
- **Pre-CognitiveBrain gate:** Assesses before response synthesis
- **OrchestrationService:** Handles meta-cognitive overrides
- **Cycle metadata:** Stores assessment for analysis and learning

**Success Metrics:**
- Reduced hallucination incidents on obscure topics
- Increased appropriate "I should search" triggers
- User feedback: "Bob admits when he doesn't know"
- Confidence calibration (confidence matches actual knowledge)

---

## Future Enhancements

### Planned Features

1. **Multimodal Input Processing**
  - Full image analysis via PerceptionAgent
  - Enhanced audio analysis (speaker diarization, richer event detection)
  - Multimodal memory storage

2. **Advanced Web Browsing**
  - Sandboxed/headless browser environment for JS-heavy sites
  - Citation tracking and fact verification

3. **Advanced Consolidation**
   - Compressive summarization (abstractive + topic preservation)
   - Tiered consolidation strategies
   - Memory decay and forgetting curves

4. **Enhanced Analytics**
   - Memory usage dashboards
   - Topic evolution tracking
   - User interaction insights
   - System performance metrics

5. **Production-Ready Deployment**
   - Horizontal scaling with distributed ChromaDB
   - Load balancing across backend instances
   - Production-grade monitoring (Prometheus/Grafana)
   - A/B testing framework

### Known Limitations

- **WebBrowsingService**: Performs real search and scraping, but some sites block non-browser traffic (403) or require JS; we degrade gracefully to titles/snippets when scraping fails
- **Consolidation Scaling**: Background loop processes one user at a time (needs parallelization for multi-user)
- **Multimodal Processing**: Image and audio processing infrastructure present but not fully activated

---

## References & Inspiration

**Neuroscience Papers:**
- Thalamus as relay station: Sherman & Guillery (2006), "Exploring the Thalamus"
- Anterior Cingulate Cortex conflict monitoring: Botvinick et al. (2001), "Conflict monitoring and cognitive control"
- Hippocampus memory consolidation: Squire & Alvarez (1995), "Retrograde amnesia and memory consolidation"
- Default Mode Network self-referential processing: Buckner et al. (2008), "The brain's default network"
- Theory of mind mentalizing: Frith & Frith (2006), "The neural basis of mentalizing"

**Architecture Influences:**
- ACT-R cognitive architecture (Anderson, 2007)
- Soar cognitive architecture (Laird, 2012)
- Global Workspace Theory (Baars, 1988)

---

## Version History

- **v2.1** (November 7, 2025): Phase 3 Advanced Analytics & Research Tools Implementation
  - **Statistical Analysis Backend**: Enhanced metrics_service.py with scipy/numpy integration for comprehensive statistical testing (t-tests, ANOVA, correlation analysis)
  - **Research Export Capabilities**: CSV/JSON export endpoints for scientific data analysis and publication-ready datasets
  - **Learning Curve Analysis**: Power-law fitting, trend analysis, and learning rate estimation tools
  - **Statistical Analysis Frontend**: New StatisticalAnalysis.tsx component with interactive statistical tools and research export controls
  - **Tabbed Dashboard Interface**: Enhanced Dashboard.tsx with Overview/Statistical Analysis tabs for seamless navigation
  - **Automated Research Reports**: Generate statistical summaries and research reports programmatically
  - **Scientific Validation Tools**: Significance testing, comparative analysis, and effect size calculations
  - **Data Export API**: REST endpoints for CSV/JSON export with configurable time periods
  - **TypeScript Interface Updates**: Resolved data structure mismatches between backend and frontend
  - **Orchestration Service Bug Fix**: Fixed UnboundLocalError in effective_input_text variable access

- **v2.1** (November 9, 2025): Metrics System Stability & Serialization Fixes
  - **ChromaDB Timestamp Handling**: Fixed inconsistent timestamp storage and querying with proper numeric Unix timestamp conversion
  - **Numpy Boolean Serialization**: Resolved FastAPI serialization errors by converting numpy.bool_ types to Python bool types
  - **Statistical Analysis Robustness**: Enhanced timestamp parsing to handle both ISO strings and Unix timestamps for backward compatibility
  - **Correlation Analysis Fixes**: Improved error handling for malformed timestamps in learning curve analysis
  - **Metrics Service Reliability**: Added fallback mechanisms for ChromaDB query failures with in-memory data recovery
  - **Dashboard API Stability**: Fixed 500 errors in statistical analysis and learning curves endpoints
  - **Type Safety Improvements**: Ensured all boolean comparisons return Python bool types for JSON serialization

- **v1.9** (November 7, 2025): Structured Error Analysis System Implementation
  - **ErrorAnalysis Model**: Comprehensive failure analysis with severity scoring, agent conflicts, and learning signals
  - **Enhanced ConflictMonitor**: Generates structured error analysis for coherence failures (< 0.5)
  - **Enhanced MetaCognitiveMonitor**: Generates error analysis for knowledge gaps and uncertainty triggers
  - **Upgraded ProceduralLearningService**: Processes ErrorAnalysis objects for precise skill improvement
  - **OrchestrationService Integration**: Routes structured error data to learning systems
  - **Learning Precision**: Correlates specific agent sequences with failure types for targeted improvement
  - **Bug Fix**: Corrected AgentOutput attribute access (agent_name → agent_id) in orchestration service
  - **Documentation Corrections**: Removed outdated STM snapshot accumulation note (snapshots are overwritten, not accumulated)

- **v1.8** (November 7, 2025): Phase 5 & 6 Learning Systems documentation complete
  - **Phase 5: Metacognition & Self-Reflection** - Complete documentation added
    - Self-Reflection & Discovery Engine: Pattern mining, insight generation, autonomous triggers
    - Meta-Cognitive Monitor: Knowledge boundary detection, uncertainty responses, pre-response gating
    - Conflict Monitor: Agent output coherence checking, RL-integrated resolution strategies
  - **Phase 6: Learning Systems** - Complete documentation added
    - Reinforcement Learning Service: Q-learning, composite rewards, habit formation, user-specific adaptation
    - Procedural Learning Service: Skill performance tracking, error-based learning, sequence optimization
  - **Embedding payload size fixes**: Automatic text chunking for large documents (36KB limit handling)
  - **Memory safeguards**: Context point limits to prevent unbounded summary growth

- **v1.6** (November 6, 2025): Reinforcement Learning reward signals implementation
  - **Composite reward computation** replacing provisional user_satisfaction_potential
  - **Multi-source reward signals** (weighted combination):
    - Trust delta (0.3): Improvement in emotional trust level from EmotionalMemoryService
    - Sentiment shift (0.2): Positive change in detected sentiment (positive/neutral/negative)
    - User feedback (0.3): Explicit positive/negative language in user input
    - Engagement continuation (0.2): Input length and follow-up questions indicating continued interest
  - **Pre/post-interaction capture**: Emotional profile state captured before cycle execution
  - **ChromaDB persistence**: RL Q-values and habits stored in emotional_profiles collection
  - **Strategy selection integration**: ConflictMonitor uses RL-selected strategies for resolution
  - **OrchestrationService wiring**: EmotionalMemoryService injected for reward computation
  - **Metadata logging**: Reward breakdown stored in cycle metadata for analysis/debugging

- **v1.7** (November 7, 2025): Meta-Cognitive Monitoring implementation
  - **"Feeling of knowing"** prefrontal cortex-inspired knowledge boundary detection
  - **Knowledge gap scoring**: Semantic/episodic memory coverage, query complexity analysis
  - **Overconfidence detection**: Prevents confident hallucinations on unknown topics
  - **Action recommendations**: ANSWER/SEARCH_FIRST/ASK_CLARIFICATION/DECLINE_POLITELY/ACKNOWLEDGE_UNCERTAINTY
  - **Uncertainty response generation**: Natural, honest "I don't know" responses using LLM
  - **Pre-CognitiveBrain gate**: Meta-cognitive assessment before response synthesis
  - **OrchestrationService integration**: Handles overrides for high-confidence gaps
  - **Cycle metadata storage**: Assessment data for analysis and learning improvement

- **v1.5** (November 6, 2025): Proactive Engagement - Bob learns to initiate conversations naturally
  - Implemented `ProactiveEngagementEngine` for autonomous conversation initiation
  - **Multiple trigger types**:
    - Knowledge gaps: Bob asks questions when he needs clarification
    - Self-reflection insights: Shares patterns discovered during reflection
    - Discovery patterns: Interesting connections found autonomously
    - Emotional check-ins: For trusted friends/companions
    - **Memory consolidation**: Shares insights after "dreaming" (30% chance per interesting pattern)
    - **Boredom**: Bob reaches out when idle and wants to engage (casual, natural messages)
  - **Natural learning from feedback**: Bob adjusts behavior when told he's annoying
    - Reduces trust slightly when receiving negative feedback (Bob feels hurt)
    - Increases cooldown period dynamically (backs off, +12h per net negative)
    - Disables proactive engagement after 3+ negative reactions (respects boundaries)
    - Feels "encouraged" by positive feedback (reduces cooldown, -4h per net positive)
  - **Emotionally intelligent triggers**: Respects relationship type, trust level, and interaction history
  - **Priority-based queuing**: High-priority insights (≥0.7) get shared first
  - **Safeguards**: Minimum trust threshold (0.4), configurable cooldowns (base 24h), user opt-out support
  - Integrated with self-reflection, discovery, and **memory consolidation** engines to surface patterns
  - API endpoints: `GET /chat/proactive` (check for messages), `POST /chat/proactive/reaction` (record feedback)
  - Chat endpoint auto-detects responses to proactive messages via `metadata.responding_to_proactive_message`

- **v1.4** (November 6, 2025): Audio input integration and resilience
  - Enabled audio-only and multimodal requests via `AudioInputProcessor`
  - Robust JSON salvage for LLM outputs (handles fenced code blocks and partial JSON)
  - Orchestration pre-transcribes audio and appends transcript to `effective_input_text`
  - Safe placeholder injection when transcription unavailable to prevent empty-text crashes
  - Documented observability, configuration, limitations, and flow position (Step 1.5)

- **v1.3** (November 6, 2025): Web Browsing enablement and scraper hardening
  - Enabled actual web research via Google CSE or SerpAPI with provider auto-detection
  - Added realistic browser headers in scraping to reduce 403s
  - Implemented graceful fallback to titles/snippets summarization when scraping is blocked
  - Documented configuration, observability, limitations, and troubleshooting for web browsing

- **v1.2** (November 6, 2025): Phase 4 and Cognitive Brain synthesis documentation
  - Detailed Phase 4: Higher-Order Executive Functions documentation
  - Planning Agent (DLPFC): Strategic response planning
  - Creative Agent (DMN): Analogies, metaphors, novel perspectives
  - Critic Agent (OFC): Logical coherence and safety assessment
  - Discovery Agent (PFC): Knowledge gap detection and web search
  - Cognitive Brain synthesis logic with priority rules
  - Agent conflict resolution strategies
  - Self-Model integration rules for personality and tone
  - Enhanced table of contents with subsections

- **v1.1** (November 6, 2025): Theory of Mind validation implementation complete
  - Automatic prediction validation after each cycle
  - Confidence adjustment based on accuracy
  - Validation statistics tracking and API endpoints
  - Learning from prediction outcomes over time

- **v1.0** (November 5, 2025): Initial brain-inspired architecture with Phase 1-3 complete
  - Phase 1: SelfModel, WorkingMemoryBuffer, EmotionalSalienceEncoder
  - Phase 2: ThalamusGateway, ConflictMonitor, ContextualMemoryEncoder
  - Phase 3: AutobiographicalMemorySystem, MemoryConsolidationService, TheoryOfMindService
  - Full STM/Summary/LTM memory hierarchy
  - Autonomous triggering with DecisionEngine
  - Comprehensive documentation consolidation

------

**Document Maintainers:** Ed Bentley
**Last Review:** November 7, 2025
**Next Review:** As system evolves with new phases or major features
