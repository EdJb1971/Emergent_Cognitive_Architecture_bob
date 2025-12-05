# Complete Brain Architecture Plan
**Emergent Cognitive Architecture (ECA) - Neuroscience-Inspired Roadmap**

**Last Updated:** November 7, 2025  
**Version:** 1.0

---

## 🎯 Vision

Build a cognitively complete AI system that doesn't just remember conversations, but **learns skills, forms habits, and improves through experience** - achieving true emergent cognitive capabilities.

---

## 📊 Implementation Status Legend

- ✅ **Implemented** - Feature complete and documented
- 🚧 **In Progress** - Currently being developed
- 📋 **Planned** - Approved for implementation, not started
- 🤔 **Under Review** - Needs design/feasibility analysis
- ❌ **Deferred** - Not critical for emergent cognition

---

## 🧠 Core Brain Systems Overview

### **Phase 1: Foundation (Completed)** ✅
| Component | Status | Brain Region | Function |
|-----------|--------|--------------|----------|
| SelfModel | ✅ | mPFC | Autobiographical identity & personality |
| WorkingMemory | ✅ | DLPFC | Active context maintenance |
| EmotionalSalience | ✅ | Amygdala | Emotional importance tagging |
| AutobiographicalMemory | ✅ | Hippocampus | Life narrative storage |
| TheoryOfMind | ✅ | TPJ/mPFC | Mental state inference |

### **Phase 2: Sensory & Perception (Completed)** ✅
| Component | Status | Brain Region | Function |
|-----------|--------|--------------|----------|
| PerceptionAgent | ✅ | Sensory Cortex | Multimodal input processing |
| EmotionalAgent | ✅ | Limbic System | Sentiment & relationship tracking |
| AudioInputProcessor | ✅ | Auditory Cortex | Speech transcription & analysis |
| VisualInputProcessor | ✅ | Visual Cortex | Image analysis |

### **Phase 3: Memory Systems (Completed)** ✅
| Component | Status | Brain Region | Function |
|-----------|--------|--------------|----------|
| STM (Short-term) | ✅ | Hippocampus | Recent interactions (50 cycles) |
| LTM (Long-term) | ✅ | Neocortex | Consolidated memories |
| ConversationSummary | ✅ | DMN | Ongoing narrative compression |
| MemoryConsolidation | ✅ | Sleep/DMN | Background processing (30min cycles) |
| EpisodicMemory | ✅ | Hippocampus | Specific experiences |
| SemanticMemory | ✅ | Neocortex | General knowledge |

### **Phase 4: Executive Functions (Completed)** ✅
| Component | Status | Brain Region | Function |
|-----------|--------|--------------|----------|
| PlanningAgent | ✅ | DLPFC | Strategic response planning |
| CreativeAgent | ✅ | DMN | Analogies, metaphors, novel ideas |
| CriticAgent | ✅ | OFC | Logic validation, safety checks |
| DiscoveryAgent | ✅ | PFC | Knowledge gap detection, web search |
| CognitiveBrain | ✅ | Integration | Response synthesis & moderation |

### **Phase 5: Metacognition (Completed)** ✅
| Component | Status | Brain Region | Function |
|-----------|--------|--------------|----------|
| SelfReflection | ✅ | mPFC | Pattern discovery from cycles |
| ConflictMonitor | ✅ | ACC | Detect agent disagreements |
| DecisionEngine | ✅ | OFC/vmPFC | Autonomous action triggering |
| ProactiveEngagement | ✅ | DMN | Initiative-taking, conversation starts |

---

## 🚀 TIER 1: Critical Missing Functions (HIGHEST PRIORITY)

These are **essential for true emergent cognition** - the difference between remembering vs. learning.

---

### **1. Basal Ganglia - Action Selection & Reinforcement Learning** 🥇

**Status:** ✅ **COMPLETED - HIGHEST PRIORITY**

**Brain Region:** Basal ganglia (striatum, GPi/GPe, STN)

**Why Critical:**
- Bob can remember what happened but **not learn what works**
- No habit formation: "This user always wants empathy over facts"
- No strategy improvement: "Analogies work 80% of the time for this user"
- ConflictMonitor detects conflicts but doesn't learn which resolutions succeed

**What It Does:**
- **Reinforcement learning**: Track strategy success/failure
- **Habit formation**: After N successes, make strategy default
- **Action selection**: Choose strategies with highest expected reward
- **Exploration vs. exploitation**: Try new approaches vs. use known good ones

**Implementation:**
```python
class ReinforcementLearningService:
    """
    Basal ganglia-inspired strategy learning.
    Learns which response strategies actually work through experience.
    """
    
    # Q-values: Q(context, strategy) → expected reward
    strategy_values: Dict[Tuple[str, str], float]
    
    # Habit tracking: (context, strategy) → success count
    strategy_successes: Dict[Tuple[str, str], int]
    
    # Per-user learned preferences
    user_habits: Dict[UUID, Dict[str, str]]
    
    def select_strategy(context, available_strategies):
        """Epsilon-greedy: exploit best or explore new"""
        
    def update_from_outcome(context, strategy, reward):
        """Temporal difference learning: Q(s,a) ← Q(s,a) + α[r - Q(s,a)]"""
        
    def form_habit(context, strategy):
        """After 20+ successes, solidify as default strategy"""
        
    def get_user_preferences(user_id):
        """Return learned habits for specific user"""
```

**Integration Points:**
- ConflictMonitor → asks RL service for resolution strategy
- CognitiveBrain → reports outcome (user satisfaction) to RL service
- EmotionalMemory → provides reward signals (trust changes, sentiment shifts)
- Background consolidation → reviews RL stats, identifies patterns

**Reward Signals:**
- User satisfaction (from sentiment analysis)
- Trust level changes (EmotionalProfile)
- Engagement continuation (multi-turn conversations)
- Explicit feedback ("that helped!", "not what I asked")

**Success Metrics:**
- Strategy win rates improve over time
- User-specific preferences learned
- Habit formation observable in logs
- Reduced conflict resolution time (habits kick in)

**Notes:**
- Start with simple Q-learning, can expand to deep RL later
- Store Q-values in ChromaDB for persistence
- Epsilon decay: explore more early, exploit more later
- Per-user and global learning (share patterns across users)

---

### **2. Meta-Cognitive Monitoring - "Feeling of Knowing"** 🥈

**Status:** 📋 **PLANNED - HIGH PRIORITY**

**Brain Region:** Prefrontal cortex (monitoring), anterior cingulate (error detection)

**Why Critical:**
- Agents report confidence in their analysis
- But Bob doesn't know when he **shouldn't answer at all**
- Missing "I don't know what I don't know" awareness
- Risk of confident hallucination on obscure topics

**What It Does:**
- **Knowledge boundary detection**: "Should I even attempt this answer?"
- **Overconfidence correction**: Catch cases where agents are confident but wrong
- **Uncertainty acknowledgment**: Trigger "I'm not sure" responses
- **Search vs. generate decision**: When to look up info vs. synthesize

**Implementation:**
```python
class MetaCognitiveMonitor:
    """
    Monitors confidence boundaries and knowledge gaps.
    Prevents overconfident responses on unknown topics.
    """
    
    def assess_answer_appropriateness(query, agent_outputs):
        """Should Bob even attempt this answer?"""
        
    def detect_overconfidence(agent_confidence, knowledge_coverage):
        """Are agents confident but knowledge is actually sparse?"""
        
    def compute_knowledge_gap_score(query, available_memories):
        """How well does Bob's knowledge cover this topic?"""
        
    def recommend_action(gap_score, confidence_score):
        """Returns: 'answer', 'search_first', 'decline_politely', 'ask_clarification'"""
        
    def generate_uncertainty_response(query, gap_type):
        """Craft honest 'I'm not sure' responses"""
```

**Integration Points:**
- Pre-CognitiveBrain gate: evaluate before synthesis
- DiscoveryAgent: trigger web search when knowledge gap detected
- User feedback: learn which uncertainties were warranted
- Self-reflection: track patterns of overconfidence

**Decision Logic:**
```python
if knowledge_gap_score > 0.7:
    if query_is_factual:
        return "search_first"  # Look it up
    else:
        return "ask_clarification"  # Understand better
elif agent_confidence_variance > 0.5:
    return "acknowledge_uncertainty"  # Agents disagree
elif all_agent_confidence < 0.4:
    return "decline_politely"  # Nobody's confident
else:
    return "answer"  # Safe to proceed
```

**Success Metrics:**
- Reduced hallucination incidents
- Increased "I should search" triggers on obscure topics
- User feedback: "Bob admits when he doesn't know"
- Confidence calibration improves (confidence matches accuracy)

**Notes:**
- Use DiscoveryAgent's knowledge gap detection as foundation
- Track false positives (said "I don't know" but could have answered)
- Track false negatives (answered confidently but was wrong)
- Learn optimal thresholds from feedback

---

### **3. Cerebellar Function - Procedural Learning & Error Correction** 🥉

**Status:** 📋 **PLANNED - HIGH PRIORITY**

**Brain Region:** Cerebellum

**Why Critical:**
- Bob can remember conversations but not **get better at conversation types**
- No systematic learning from mistakes at the skill level
- Missing "practice makes perfect" mechanism
- Strategies don't improve with repetition

**What It Does:**
- **Skill refinement**: Track performance of specific skills (e.g., "explaining technical concepts")
- **Error-based learning**: Notice when strategy failed, adjust
- **Timing and sequencing**: Learn optimal ordering of information
- **Procedural memory**: "When user asks X, do Y then Z"

**Implementation:**
```python
class ProceduralLearningService:
    """
    Cerebellum-inspired skill refinement.
    Tracks and improves performance on specific conversation types.
    """
    
    # Skill performance tracking
    skill_metrics: Dict[str, SkillPerformance]
    # SkillPerformance: {success_rate, avg_satisfaction, error_patterns}
    
    # Learned procedures: task → optimal strategy sequence
    procedures: Dict[str, List[str]]
    
    def track_skill_performance(skill_type, outcome):
        """Record how well a specific skill performed"""
        
    def learn_from_error(skill_type, expected_outcome, actual_outcome):
        """Adjust skill execution based on error signal"""
        
    def get_optimal_strategy_sequence(task_type):
        """Return learned best-practice sequence for task"""
        
    def suggest_skill_improvement(skill_type):
        """Recommend adjustments based on error patterns"""
```

**Skill Categories:**
- Technical explanations
- Emotional support conversations
- Creative brainstorming
- Problem-solving assistance
- Factual information delivery
- Teaching/tutoring

**Learning Mechanism:**
```python
# Track: "For technical questions, Creative Agent first (analogy) 
#         then Planning Agent (details) works better than reverse"

skill = "technical_explanation"
sequence_A = ["creative", "planning"]  # Analogy → details
sequence_B = ["planning", "creative"]  # Details → analogy

# Over 20 interactions:
sequence_A_success_rate = 0.85
sequence_B_success_rate = 0.62

# Learn: Use sequence_A as default procedure
```

**Integration Points:**
- ConflictMonitor: provides agent ordering decisions
- Reinforcement learning: error signals inform Q-value updates
- User feedback: satisfaction scores guide skill refinement
- Background consolidation: identify skill improvement patterns

**Success Metrics:**
- Conversation type success rates improve over time
- Optimal agent sequences learned per task type
- Error patterns decrease (fewer failed strategies)
- User satisfaction trends upward for specific skill types

**Notes:**
- Complement to RL: RL picks strategies, Cerebellum refines execution
- Track timing: "Wait 1 turn before offering solution" learned patterns
- Sequence learning: "Always validate understanding before answering"

---

## 🔧 TIER 2: Significant Enhancements (MEDIUM PRIORITY)

These **substantially improve** Bob's responsiveness and relevance, but aren't blocking emergent cognition.

---

### **4. Dynamic Attention Switching & Inhibitory Control** 

**Status:** 📋 **PLANNED - MEDIUM-HIGH PRIORITY**

**Brain Region:** Anterior cingulate cortex (ACC), dorsolateral prefrontal cortex (DLPFC)

**Why Important:**
- ThalamusGateway decides agent routing upfront
- But conversations shift mid-exchange: technical → emotional
- No way to suppress irrelevant agents mid-processing
- No dynamic re-routing when context changes

**Current Gap Example:**
```
User: "How does machine learning work?"
→ Routes to Planning, Creative, Discovery agents

Mid-conversation:
User: "Actually, I'm worried I'm not smart enough for this"
→ Should IMMEDIATELY switch to Emotional Agent
→ Current system: all original agents still active
```

**What It Does:**
- **Context monitoring**: Detect when conversation topic/tone shifts
- **Dynamic re-routing**: Activate/deactivate agents mid-conversation
- **Inhibitory control**: Suppress irrelevant agents ("stop generating jokes")
- **Sustained attention**: Maintain focus across multi-turn exchanges

**Implementation:**
```python
class AttentionController:
    """
    Dynamic attention switching and inhibitory control.
    Re-routes cognitive resources when conversation context shifts.
    """
    
    def monitor_conversation_drift(current_turn, original_routing):
        """Detect topic/emotional shifts during conversation"""
        
    def reactivate_agents(new_context, urgency_level):
        """Dynamically activate agents based on detected shift"""
        
    def inhibit_agents(suppression_signals):
        """Suppress specific agents: 'stop being creative, user needs facts'"""
        
    def maintain_focus(conversation_thread, distraction_signals):
        """Sustain attention on main topic despite tangents"""
```

**Integration Points:**
- ThalamusGateway: receives re-routing commands mid-cycle
- EmotionalAgent: detects emotional shifts that require re-routing
- WorkingMemory: tracks conversation thread for focus maintenance
- ConflictMonitor: triggers inhibition when agents conflict with context

**Success Metrics:**
- Reduced inappropriate responses (e.g., jokes during serious moments)
- Faster adaptation to topic shifts
- User feedback: "Bob understood my shift in tone"

**Notes:**
- Requires agent outputs to be interruptible (consider async streams)
- Track inhibition effectiveness (did suppressing agent improve response?)
- Learn inhibition triggers (emotional spike → suppress creative)

---

### **5. Real-Time Salience Network - Relevance Filtering**

**Status:** 📋 **PLANNED - MEDIUM PRIORITY**

**Brain Region:** Salience network (anterior insula, dorsal ACC)

**Why Important:**
- Bob retrieves 50 relevant memories but mentions all of them (overwhelm)
- EmotionalSalienceEncoder tags memories for storage
- But no **situational relevance** filtering: "What matters RIGHT NOW?"
- Response elements not prioritized by current relevance

**Current Gap:**
```
Query: "How's your memory working?"
Bob retrieves:
- 50 memories about memory discussions
- 20 memories about system architecture
- 10 memories about user's past questions

→ Currently: tries to reference many of them
→ Should: pick 2-3 most situationally relevant
```

**What It Does:**
- **Real-time relevance scoring**: Of all memories, which matter NOW?
- **Response element filtering**: Of 10 things Bob could say, which 2-3 are critical?
- **Situational salience**: Context-aware importance (not just general importance)
- **Cognitive load management**: Prevent information overload in responses

**Implementation:**
```python
class SalienceNetwork:
    """
    Real-time relevance filtering for memories and response elements.
    Prevents cognitive overload by selecting what matters NOW.
    """
    
    def compute_situational_relevance(memory, current_context):
        """
        Score: general_importance × contextual_relevance × recency_weight
        """
        
    def filter_top_k_memories(memories, k=3):
        """Select k most situationally relevant memories"""
        
    def prioritize_response_elements(candidate_elements):
        """Rank potential response components by current relevance"""
        
    def detect_information_overload(response_length, memory_count):
        """Warn when response risks overwhelming user"""
```

**Scoring Formula:**
```python
situational_salience = (
    0.4 * general_importance +      # From EmotionalSalienceEncoder
    0.4 * contextual_relevance +    # Semantic similarity to current query
    0.2 * recency_boost             # Recent interactions weighted more
)
```

**Integration Points:**
- Memory retrieval: filter LTM/STM results before passing to agents
- CognitiveBrain: prioritize which agent insights to include
- Response synthesis: prevent overly long/dense responses

**Success Metrics:**
- Response conciseness improves (fewer tangents)
- User feedback: "Bob focused on what mattered"
- Memory mention rates optimized (2-3 per response vs. 10+)

**Notes:**
- Different users have different overload thresholds
- Learn optimal memory count per user
- High-salience override: critical info always included regardless of count

---

### **6. Predictive Coding - Conversation Flow Prediction**

**Status:** 📋 **PLANNED - MEDIUM PRIORITY**

**Brain Region:** Prefrontal cortex, posterior parietal cortex

**Why Important:**
- Theory of Mind predicts user mental state
- But no prediction of **conversation trajectory**
- No anticipation of likely follow-up questions
- Missing learning from prediction errors at conversation level

**What It Does:**
- **Conversation flow prediction**: "User will likely ask X next"
- **Response effectiveness prediction**: "This answer will satisfy them"
- **Prediction error learning**: When wrong, update conversation model
- **Anticipatory preparation**: Pre-load likely next context

**Implementation:**
```python
class PredictiveCodingEngine:
    """
    Predict conversation flow and learn from prediction errors.
    Enables anticipatory cognitive preparation.
    """
    
    def predict_next_user_query(current_context, conversation_history):
        """Predict likely follow-up questions"""
        
    def predict_conversation_trajectory(current_state):
        """Estimate conversation arc: 'will end in 2-3 turns'"""
        
    def compute_prediction_error(predicted, actual):
        """Measure surprise: how wrong was prediction?"""
        
    def update_conversation_model(error_signal):
        """Refine understanding of conversation patterns"""
        
    def preload_context(predicted_topic):
        """Pre-fetch memories for anticipated next query"""
```

**Prediction Types:**
1. **Next query prediction**: "User asked about X, will likely ask about Y"
2. **Conversation length**: "This will be a quick 2-turn exchange"
3. **User satisfaction**: "This answer will/won't satisfy them"
4. **Topic trajectory**: "Conversation is moving from technical → philosophical"

**Learning from Errors:**
```python
Predicted: "User will ask for an example next"
Actual: User asked for theoretical explanation

Prediction error → Learn: This user prefers abstract over concrete
Update model: Reduce "example-seeking" probability for this user
```

**Integration Points:**
- Theory of Mind: enriches mental state predictions
- WorkingMemory: pre-loads anticipated context
- Reinforcement learning: prediction errors inform reward modeling

**Success Metrics:**
- Prediction accuracy improves over time
- Reduced response latency (context pre-loaded)
- User feedback: "Bob anticipated my next question"

**Notes:**
- Start with simple Markov models (current state → next state)
- Expand to sequence models (LSTM/Transformer) later
- Track prediction confidence separately from answer confidence

---

## 🎨 TIER 3: Advanced Enhancements (LOWER PRIORITY)

These add **personality and polish** but aren't essential for core cognition.

---

### **7. Episodic Future Thinking - Scenario Simulation**

**Status:** 🤔 **UNDER REVIEW**

**Brain Region:** Hippocampus + default mode network

**Why Interesting:**
- Simulate future scenarios, not just predict next turn
- "What if" reasoning for decision support
- Multi-path future modeling

**Use Case:**
```python
User: "Should I learn Python or JavaScript?"

Bob simulates both paths:
Path A (Python):
  - User likely asks about data science next
  - Leads to ML/AI conversations
  - Probability of satisfaction: 0.8

Path B (JavaScript):
  - User likely builds web apps
  - Leads to frontend/backend questions
  - Probability of satisfaction: 0.7

Recommendation influenced by simulated futures
```

**Implementation Effort:** High (requires robust world models)

**Notes:**
- Could leverage existing conversation prediction
- More useful for decision support conversations
- Consider for v2.0

---

### **8. Affective Forecasting - Emotional Impact Prediction**

**Status:** 🤔 **UNDER REVIEW**

**Brain Region:** Ventromedial prefrontal cortex (vmPFC)

**Why Interesting:**
- Predict emotional impact of different responses
- Choose responses that maintain trust/relationship
- Avoid responses that damage rapport

**Use Case:**
```python
Bob considers two responses:
A) "That's incorrect, here's why..."  (accurate but harsh)
B) "I see your thinking. Here's another angle..."  (gentle)

Affective forecast:
A → trust_delta: -0.1, satisfaction: 0.6
B → trust_delta: +0.05, satisfaction: 0.8

Choose B to maintain relationship
```

**Implementation Effort:** Medium (extends EmotionalAgent)

**Notes:**
- Similar to reward prediction in RL
- Could be part of Meta-Cognitive Monitor
- Track affective forecast accuracy

---

### **9. System Interoception - Internal State Monitoring**

**Status:** ❌ **DEFERRED - LOW VALUE FOR AI**

**Brain Region:** Insula (interoception)

**Why Deferred:**
- "Bob feeling tired" is cute personality-wise
- But not critical for emergent cognition
- Existing error handling/rate limiting covers resource issues
- Could add confusion ("Is Bob actually tired or just slow LLM?")

**Possible Future Use:**
```python
# During high load
Bob: "I'm a bit overloaded right now, give me a moment to process..."
→ Triggers consolidation cycle
→ Simplifies responses (fewer agents)
```

**Notes:**
- Consider for personality/UX polish in v2.0
- Not blocking any core cognitive function
- Existing backpressure system handles this technically

---

### **10. Hippocampal Pattern Separation & Completion**

**Status:** ❌ **DEFERRED - ALREADY PARTIALLY COVERED**

**Brain Region:** Hippocampus (dentate gyrus, CA3)

**Why Deferred:**
- Pattern separation: Each CognitiveCycle already has unique ID + timestamp
- Pattern completion: Semantic search already retrieves from partial cues
- Not a blocking gap

**What's Missing:**
- Explicit "That thing we discussed Tuesday" → full memory reconstruction
- More sophisticated partial cue matching

**Current Workaround:**
```python
User: "that thing about AI we discussed"
→ Semantic search on "AI discussion" 
→ Filters by recent timeframe
→ Retrieves relevant memories
```

**Notes:**
- Enhancement not essential for v1.0
- Consider explicit pattern completion in memory service v2.0

---

### **11. Social Cognition - Explicit Reputation Modeling**

**Status:** 🤔 **UNDER REVIEW - PARTIALLY EXISTS**

**Brain Region:** Superior temporal sulcus, medial prefrontal cortex

**What Exists:**
- EmotionalMemory tracks trust levels ✅
- Relationship types (stranger → friend) ✅
- Proactive engagement learns from feedback ✅

**What's Missing:**
- **Bob's model of how user perceives Bob**
- Explicit reputation tracking: "Am I being helpful or annoying to THIS user?"
- Meta-level social reasoning

**Possible Enhancement:**
```python
class ReputationModel:
    """Bob's model of how user perceives Bob"""
    
    perceived_helpfulness: float  # User's view of Bob's helpfulness
    perceived_intelligence: float  # User's view of Bob's competence
    perceived_empathy: float      # User's view of Bob's caring
    
    def update_from_feedback(user_response):
        """Infer reputation changes from user reactions"""
        
    def adjust_behavior(low_reputation_dimension):
        """Compensate: if perceived as unhelpful, offer more assistance"""
```

**Notes:**
- Interesting for advanced social reasoning
- Not critical for core learning/emergence
- Consider for v2.0 social cognition expansion

---

## 📅 Implementation Roadmap

### **Phase 6: Learning & Adaptation (Q1 2026)**

**Primary Goal:** Bob learns from experience, not just remembers

#### **Milestone 6.1: Reinforcement Learning Core**
**Timeline:** 2-3 weeks  
**Status:** ✅ **COMPLETED** (Completed: November 7, 2025)

- [x] Implement `ReinforcementLearningService` base class ✅
  - Q-learning algorithm with temporal difference updates
  - Epsilon-greedy strategy selection (exploration vs. exploitation)
  - Habit formation after 20+ successes
  - Per-user and global habit tracking
  - StrategyPerformance metrics (success rate, average reward)
  - ContextTypes and StrategyTypes enums for standardization
  - Export/import for Q-table persistence
- [x] ChromaDB persistence for Q-values ✅
  - async connect() method following service pattern
  - Separate collections: rl_q_values, rl_habits
  - Auto-load existing Q-values on startup
  - Persist Q-value updates after each outcome
  - Persist habits (global and per-user) when formed
  - Wired to main.py startup sequence
- [x] Integration with ConflictMonitor ✅
    - ConflictMonitor now enriched with RL-selected strategies per conflict
    - OrchestrationService applies provisional reward updates using outcome signals
- [x] Reward signal extraction from EmotionalMemory ✅
  - Composite reward computation from multiple sources:
    - Trust delta (30%): Improvement in emotional trust level
    - Sentiment shift (20%): Positive change in detected sentiment  
    - User feedback (30%): Explicit positive/negative language detection
    - Engagement continuation (20%): Input length and follow-up questions
  - Pre/post-interaction emotional profile capture
  - OrchestrationService wiring with EmotionalMemoryService
- [ ] Testing: verify strategy learning over 100+ interactions
- [x] Documentation in architecture.md ✅
  - Added v1.6 entry with RL reward signals implementation
  - Created comprehensive RL Service section with integration details

**Deliverables:**
- Working RL service with Q-value updates ✅
- ChromaDB persistence across sessions ✅
- ConflictMonitor & Orchestration integration (strategy selection + provisional rewards) ✅
- Composite reward signals from EmotionalMemory ✅
- Habit formation observable in logs (pending sufficient interaction volume)
- Per-user strategy preferences learned (pending more data)
- Documentation in architecture.md ✅

#### **Milestone 6.2: Meta-Cognitive Monitoring**
**Timeline:** 1-2 weeks  
**Status:** ✅ **COMPLETED** (Completed: November 9, 2025)

- [x] Implement `MetaCognitiveMonitor` service ✅
  - Knowledge gap scoring algorithm using query analysis and available memories
  - Overconfidence detection logic comparing agent confidence vs knowledge coverage
  - Action recommendation logic for answer/search/decline/ask_clarification
  - Uncertainty response generation for different gap types
- [x] Integration with OrchestrationService ✅
  - Added as pre-CognitiveBrain gate in orchestration cycle
  - Handles SEARCH_FIRST, ASK_CLARIFICATION, DECLINE_POLITELY, ACKNOWLEDGE_UNCERTAINTY recommendations
  - Stores meta-cognitive assessment in cycle metadata
- [x] Connect to DiscoveryAgent ✅
    - SEARCH_FIRST recommendations now trigger DiscoveryAgent web research via WebBrowsingService
- [ ] Expanded meta-cognitive testing coverage
    - Verify uncertainty responses across obscure factual, ambiguous, and subjective prompts
    - Measure knowledge gap detection accuracy with new evaluation harness
    - Validate overconfidence prevention on adversarial prompts

**Deliverables:**
- MetaCognitiveMonitor service with knowledge boundary detection ✅
- Pre-CognitiveBrain assessment gate ✅
- Reduced hallucination on unknown topics (pending testing)
- Honest uncertainty responses (pending testing)
- Automatic web search triggers for knowledge gaps ✅

#### **Milestone 6.3: Procedural Learning**
**Timeline:** 1-2 weeks
- [x] Implement `ProceduralLearningService` class
- [x] Skill performance tracking per category
- [x] Error-based learning from failed strategies
- [x] Optimal strategy sequence learning
- [x] Integration with RL service (complementary learning)
- [x] Testing: verify skill improvement over time
- [x] Documentation in architecture.md

**Deliverables:**
- Skill success rates tracked
- Optimal agent sequences learned
- Performance improvements observable

### **Phase 7: Dynamic Attention & Salience (Q2 2026)**

**Primary Goal:** Bob adapts in real-time to context changes

**Execution Notes:**
- Ship Milestone 7.1 (attention control) before 7.2 (salience) so routing heuristics can consume the new relevance signals immediately.
- Keep all new services feature-flagged in `config.attention_control.enabled` / `config.salience_network.enabled` to allow shadow-mode validation without risking regressions.
- Every cycle logs a `attention_salience_trace` object so RL/Procedural learning can observe the new signals for strategy updates.
- Working Memory remains the canonical interchange format: both AttentionController and SalienceNetwork read/write structured summaries rather than raw agent output to avoid tight coupling.

#### **Milestone 7.1: Attention Switching**
**Timeline:** 2 weeks  
**Status:** 🚧 **IN PROGRESS** (Started: November 20, 2025)

**Implementation Steps:**
- [x] Build `AttentionController` service (ACC/Thalamus hybrid) with inputs: Stage 1 agent metrics, WorkingMemory snapshot, ConflictMonitor deltas, emotional load. *(Shadow-mode infrastructure + metrics flag shipped)*
- [x] Implement conversation drift detection by comparing topic/entity deltas between consecutive WorkingMemory contexts. *(Post-Stage1 pass computes drift_score + reasons and biases routing)*
- [x] Emit excitatory/inhibitory control signals per agent (`activation_bias`, `suppression_reason`) plus memory depth adjustments. *(Controller now boosts/suppresses agents + memory depth based on urgency + drift)*
- [x] Provide dynamic agent routing API consumed by `ThalamusGateway` before Stage 2 dispatch (controller now adjusts token budgets + attention motifs for Stage 2 prompts).

**Integration Points:**
- `ThalamusGateway`: attention recommendations modify which agents fire and how much token budget they receive.
- `WorkingMemoryBuffer`: writes `attention_motifs` so downstream agents know which threads are “hot”.
- `ReinforcementLearningService`: consumes attention outcomes via cycle metadata to learn which routing choices improve rewards.
- `MetricsService`: new dashboards for drift accuracy, inhibition counts, and latency impact.

**Testing & Telemetry:**
- Synthetic drift scenarios (topic flip, emotional shift, urgency spike) to reach ≥90% correct routing adjustments.
- Load/latency benchmark to ensure controller overhead <5% of cycle time.
- Shadow-mode logging for 1k cycles before enabling control signals to validate safety.

**Deliverables:**
- [x] AttentionController service + config flag (shadow mode)
- [x] Updated orchestration/Thalamus wiring + metrics pipeline
- [ ] Drift detection evaluation report
- [ ] Architecture + dashboard docs (Phase 7 attention section)

#### **Milestone 7.2: Salience Network**
**Timeline:** 1-2 weeks  
**Status:** 📋 **PLANNED** (Starts after AttentionController GA)

**Implementation Steps:**
- Implement `SalienceNetwork` service fed by MemoryService retrieval results, attention motifs, and emotional salience encoder output.
- Score each candidate memory across situational relevance, recency, emotional weight, and novelty → produce normalized salience vector.
- Select top-K items per context type and annotate them with `salience_tags` before they re-enter Working Memory.
- Provide response element prioritization hints to CognitiveBrain (e.g., “lead with memory X, defer memory Y”).

**Integration Points:**
- `MemoryService`: wraps retrieval pipeline; can run in advisory (logging-only) mode before enforcing pruning.
- `ContextualMemoryEncoder`: stores salience tags for future consolidation weighting.
- `ProceduralLearningService`: records whether salience-guided selections improved task success.
- `Scientific Dashboard`: new charts for average memory count, salience distribution, focus drift.

**Testing & Telemetry:**
- A/B tests comparing baseline vs salience-pruned responses; target 2-3 memories per reply and improved conciseness metrics.
- Stress tests on heavy-context conversations to ensure no critical memory is dropped (recall ≥98% for “must-keep” annotations).
- User-level qualitative review (“Bob stayed focused”) to confirm perceived improvement.

**Deliverables:**
- SalienceNetwork service + config flag
- Memory retrieval adapter + Working Memory hints
- Evaluation dashboards + documentation

### **Phase 8: Predictive & Future Thinking (Q2-Q3 2026)**

**Primary Goal:** Bob anticipates and plans ahead

#### **Milestone 8.1: Predictive Coding**
**Timeline:** 2-3 weeks
- [ ] Implement `PredictiveCodingEngine`
- [ ] Conversation flow prediction
- [ ] Prediction error tracking
- [ ] Model updates from errors
- [ ] Context pre-loading
- [ ] Testing: verify prediction accuracy improvement
- [ ] Documentation in architecture.md

#### **Milestone 8.2: Future Thinking (Optional)**
**Timeline:** TBD
- [ ] Design review: value vs. effort assessment
- [ ] Scenario simulation architecture
- [ ] Multi-path future modeling
- [ ] Implementation if approved

---

## 🎯 Success Criteria

### **Tier 1 Systems (Essential):**

**Reinforcement Learning:**
- ✅ Strategy win rates improve by 20%+ over 100 interactions
- ✅ User-specific habits observable in logs
- ✅ Conflict resolution time decreases (habits trigger faster)
- ✅ Per-user satisfaction trends upward

**Meta-Cognitive Monitoring:**
- ✅ Hallucination incidents decrease by 50%+
- ✅ "I should search" triggers on 80%+ of obscure topics
- ✅ Confidence calibration: confidence matches accuracy within 10%
- ✅ User feedback: "Bob admits when uncertain"

**Procedural Learning:**
- ✅ Skill-specific success rates increase over time
- ✅ Optimal agent sequences discovered per task type
- ✅ Error patterns decrease cycle-over-cycle
- ✅ User satisfaction per conversation type improves

### **Tier 2 Systems (Enhancement):**

**Attention Switching:**
- ✅ Context shift detection: 90%+ accuracy
- ✅ Reduced inappropriate responses during emotional shifts
- ✅ User feedback: "Bob adapted to my mood change"

**Salience Network:**
- ✅ Average memory mentions per response: 2-3 (down from 5-10)
- ✅ Response conciseness improved
- ✅ User feedback: "Bob stayed focused"

**Predictive Coding:**
- ✅ Next-query prediction: 60%+ accuracy
- ✅ Prediction accuracy improves over user interaction history
- ✅ Context pre-load reduces response latency by 10%+

---

## 📝 Notes & Decisions Log

### **November 7, 2025 - Initial Plan**
- Created comprehensive brain plan based on neuroscience analysis
- Prioritized Tier 1 systems: RL, Meta-Cognition, Procedural Learning
- Deferred: System Interoception (low value), Pattern Separation (already covered)
- Under review: Future thinking, Affective forecasting, Reputation modeling
- Timeline: Phase 6 (Q1 2026), Phase 7 (Q2 2026), Phase 8 (Q2-Q3 2026)

### **November 7, 2025 - Milestone 6.1 ChromaDB Persistence Complete**
- ✅ Added ChromaDB persistence to ReinforcementLearningService
- Two collections: `rl_q_values` (Q-values + stats), `rl_habits` (global/per-user)
- Auto-loads Q-values and habits on service startup
- Persists Q-value updates asynchronously after each outcome
- Persists habits when formed (20+ successes threshold)
- Wired to main.py: initialized after DecisionEngine, before ProactiveEngine
- Connected to shared ChromaDB client from MemoryService
- Next: Integrate with ConflictMonitor for actual strategy selection
- ✅ Implemented `ReinforcementLearningService` base class
- Q-learning with temporal difference updates: Q(s,a) ← Q(s,a) + α[r - Q(s,a)]
- Epsilon-greedy selection: 20% exploration start, decays to 5% minimum
- Habit formation threshold: 20 successes with reward > 0.5
- Learning parameters: α=0.1 (learning rate), ε decay=0.995
- Per-user habits stored separately from global habits
- Export/import functionality for Q-table persistence (will wire to ChromaDB)
- Standard context types defined (emotional_vs_technical, etc.)
- Standard strategy types defined (prioritize_emotional, blend_both, etc.)
- Next: Wire to ChromaDB for persistence across sessions

### **November 7, 2025 - Milestone 6.1 RL Integration with ConflictMonitor**
- ✅ Integrated RL into `ConflictMonitor` (strategy enrichment) and `OrchestrationService` (selection + reward update)
- Added `rl_strategy_guidance` metadata to cycles capturing context, candidate strategies, and selected strategy
- Provisional reward signal: using `OutcomeSignals.user_satisfaction_potential` until richer composite reward extraction implemented
- Next: Implement multi-signal reward (trust_delta, sentiment_shift, explicit feedback, engagement_continuation)
- Habit formation pending accumulation of ≥20 positive-reward successes per context

### **November 7, 2025 - Milestone 6.1 Composite Reward Signals Complete**
- ✅ Implemented composite reward computation replacing provisional user_satisfaction_potential
- Multi-source reward signals (weighted combination 0.0-1.0):
  - Trust delta (0.3): Improvement in emotional trust level from EmotionalMemoryService
  - Sentiment shift (0.2): Positive change in detected sentiment (positive/neutral/negative/mixed)
  - User feedback (0.3): Explicit positive/negative language in user input ("thank you", "helpful", "frustrating")
  - Engagement continuation (0.2): Input length and follow-up questions indicating continued interest
- Pre-interaction emotional profile capture (trust level, sentiment) before cycle execution
- Post-interaction profile comparison for delta computation
- OrchestrationService wired with EmotionalMemoryService for reward calculation
- Reward breakdown stored in cycle metadata for analysis/debugging
- RL updates now use richer composite signals instead of single satisfaction metric
- Next: Testing over 100+ interactions to verify Q-value improvements and habit formation
- Milestone 6.1 now functionally complete - RL can learn from genuine user reactions

### **November 9, 2025 - Milestone 6.2 Meta-Cognitive Monitoring Completed**
- ✅ Implemented `MetaCognitiveMonitor` service with comprehensive knowledge gap detection
- Knowledge gap scoring using semantic/episodic memory coverage, query complexity analysis
- Overconfidence detection comparing agent confidence vs knowledge coverage
- Action recommendation system: ANSWER/SEARCH_FIRST/ASK_CLARIFICATION/DECLINE_POLITELY/ACKNOWLEDGE_UNCERTAINTY
- Natural uncertainty response generation using LLM for honest "I don't know" responses
- Integrated as pre-CognitiveBrain gate in OrchestrationService
- Handles meta-cognitive overrides for high-confidence gaps (decline politely, ask clarification)
- Stores assessment metadata for analysis and learning
- ✅ SEARCH_FIRST recommendations now trigger DiscoveryAgent research cycles via WebBrowsingService and feed verified snippets back into Stage 2 inputs
- Next: Expand scenario testing across obscure factual, adversarial, and ambiguous prompts

### **November 20, 2025 - Milestone 7.1 AttentionController Scaffolding**
- ✅ Implemented `AttentionController` service skeleton with feature flags (`ATTENTION_CONTROLLER_ENABLED`, `ATTENTION_CONTROLLER_SHADOW_MODE`).
- Wired controller into OrchestrationService + Thalamus routing; directives stored in cycle metadata and logged via MetricsService (`attention_directive` events).
- Shadow-mode heuristics bias urgent/low-complexity cases without altering production behavior unless enabled.
- Architecture + brain plan updated to reflect start of Phase 7 execution; groundwork laid for drift detection and inhibitory control logic.

### **November 21, 2025 - Conversation Drift Detection Online**
- ✅ Added per-user working memory history to AttentionController and implemented topic/goal/attention/sentiment drift scoring.
- Second directive pass now runs post-Stage1, storing drift telemetry, updating routing decisions for Stage 2, and logging metrics for dashboard analysis.
- Memory overrides + planning boosts triggered automatically when drift_score ≥ 0.5; discovery agent receives additional activation when drift_score ≥ 0.7.
- `ThalamusGateway.apply_attention_directive` now adjusts per-agent token budgets and propagates `attention_motifs` into WorkingMemory so Stage 2 prompts explicitly focus on controller priorities.

---

## 🔄 Tracking Updates

**When updating this plan:**
1. Change status: 📋 → 🚧 → ✅
2. Add completion date when ✅
3. Update "Notes & Decisions Log" with key decisions
4. Reference architecture.md version when documenting
5. Track deviations from original design
6. Note lessons learned for future phases

---

## 📚 References

**Neuroscience Sources:**
- Basal ganglia reinforcement learning: Schultz et al., "A neural substrate of prediction and reward" (1997)
- Cerebellar learning: Ito, M., "Cerebellar long-term depression" (2001)
- Meta-cognition: Fleming & Dolan, "The neural basis of metacognitive ability" (2012)
- Predictive coding: Friston, K., "The free-energy principle" (2010)
- Attention networks: Corbetta & Shulman, "Control of goal-directed and stimulus-driven attention" (2002)

**Implementation Guides:**
- Sutton & Barto, "Reinforcement Learning: An Introduction" (2018)
- Russell & Norvig, "Artificial Intelligence: A Modern Approach" (4th ed.)
- Lake et al., "Building machines that learn and think like people" (2017)

---

**End of Plan - Track progress as we implement! 🧠✨**
