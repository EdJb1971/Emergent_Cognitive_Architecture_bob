# Emergent Cognitive Architecture (ECA)

**A brain that learns, not just remembers.**

Neuroscience-inspired multi-agent platform that forms habits, switches strategies mid-conversation, and knows when to say "I don't know." ECA operationalizes prefrontal, limbic, and thalamic dynamics in software so interactive AI systems can develop genuine cognitive continuity.

![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)
![Python 3.11+](https://img.shields.io/badge/Python-3.11+-green.svg)
![Status: Research](https://img.shields.io/badge/Status-Research-orange.svg)

---

## Why ECA Is Different

| Traditional Chatbots | ECA ("Bob") |
|---------------------|-------------|
| Stateless context window | Persistent memory with consolidation |
| Same response patterns always | Learns what works per user |
| Confident about everything | Knows its knowledge boundaries |
| Fixed attention allocation | Dynamic agent routing based on context |
| No skill improvement | Procedural learning from errors |

### Core Innovations

- **Basal Ganglia–style reinforcement learning**: Strategy Q-values, habit formation, and per-user preferences persist in ChromaDB so the system genuinely improves with experience.
- **Meta-cognitive safety net**: A dedicated monitor estimates knowledge gaps, overconfidence, and appropriate actions (answer vs. search vs. decline) before synthesis.
- **Procedural learning loop**: Cerebellum analog tracks skill categories and learns optimal agent execution sequences, complementing RL-based strategy selection.
- **Dynamic attention controller**: A feature-flagged ACC/Thalamus hybrid detects drift, emits excitatory/inhibitory signals, adjusts Stage 2 token budgets, and propagates attention motifs through Working Memory.
- **Theory of Mind with validation**: Predictions about user mental states are auto-validated against actual behavior, with confidence adjusting based on accuracy.
- **Immutable multisensory binding**: Same-turn text, image, and audio evidence is temporally aligned with conservative agreement/conflict detection and reliability-aware advisory attention; primary observations are never generatively rewritten.
- **Shadow predictive perception**: Prior-only, permanently labelled hypotheses are compared with immutable sensory evidence to produce signed errors and bounded clarification/recapture recommendations without changing observations or behavior.

---

## Key Concepts

| Component | Brain Analog | Function |
|-----------|--------------|----------|
| ReinforcementLearningService | Basal Ganglia | Strategy Q-values, habit formation |
| MetaCognitiveMonitor | Prefrontal Cortex | Knowledge boundaries, overconfidence detection |
| ProceduralLearningService | Cerebellum | Skill tracking, error-based learning |
| AttentionController | ACC/Thalamus | Drift detection, agent inhibition |
| WorkingMemoryBuffer | DLPFC | Active context maintenance |
| TheoryOfMindService | TPJ/mPFC | Mental state inference and prediction |
| AutobiographicalMemory | Hippocampus | Episodic/semantic memory separation |
| EmotionalSalienceEncoder | Amygdala | Emotional importance tagging |
| MultisensoryBindingService | Thalamus/Association Cortex | Same-turn temporal binding and advisory conflict attention |
| PredictivePerceptionService | Predictive Cortex | Prior-labelled hypotheses, immutable errors, shadow clarification |

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      User Input                              │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 1: Perception, Emotion, Memory (Parallel)            │
│  → Populates Working Memory with context + salience tags    │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  AttentionController: Drift detection, routing adjustments  │
│  ThalamusGateway: Token budgets, agent activation           │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  MetaCognitiveMonitor: Answer / Search / Decline decision   │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  Stage 2: Planning, Creative, Critic, Discovery (Parallel)  │
│  → CognitiveBrain synthesizes final response                │
└─────────────────────────┬───────────────────────────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│  Learning: RL rewards, Procedural skill tracking, ToM valid │
│  Memory: STM → Summary → LTM consolidation                  │
└─────────────────────────────────────────────────────────────┘
```

### Repository Structure

```
repo/
├─ src/
│  ├─ agents/              # Stage 1 & Stage 2 agent implementations
│  ├─ services/            # RL, meta-cognition, attention, memory, orchestration
│  ├─ core/                # Config, logging, shared exceptions
│  └─ models/              # Pydantic models for directives, memory, routing
├─ frontend/               # React + Tailwind dashboard (optional)
├─ chroma_db/              # Persistent embeddings + RL tables (git-ignored)
├─ tests/                  # Pytest suites for services and integrations
├─ architecture.md         # Audited implementation and system design
├─ roadmap.md              # Canonical plan and delivery tracker
└─ README.md               # You are here
```

---

## Getting Started

### Prerequisites
- Python 3.11+
- Node.js 20.19+, 22.12+, or 24+ (for the optional Vite dashboard frontend)
- ChromaDB (auto-initialized on first run)

### Backend Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/eca.git
cd eca

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy .env.example to .env and add your keys
# Edit .env with your GEMINI_API_KEY, API_KEY, etc.

# Run the server
uvicorn main:app --reload
```

### Frontend Dashboard (Optional)

```bash
cd frontend
npm ci
npm run dev
```

Vite binds to `127.0.0.1:3000` and proxies `/api`, `/chat`, and `/ws` to the local FastAPI server. The proxy reads `API_KEY` from the repository `.env` and injects it server-side, so normal local development does not compile the key into browser JavaScript. Use `npm run build` for the type-checked production build and `npm run preview` for a localhost-only preview.

Open the control room with **Ctrl+K** and select **Autonomy** to pause/resume all autonomous work, toggle each cognitive category, inspect its operating limits, cancel/retry tasks, and verify the immutable executive ledger. UI changes persist across restarts. Sleep can be started or stopped from this view without restarting the backend.

Select **System** for the live event-driven observability plane. It displays authenticated typed signals for cognition, memory, research, salience, sleep, and governed autonomous work. The browser resumes from its last process cursor after a transient disconnect and reports any replay/backpressure gap explicitly; telemetry is observational, while ChromaDB and the domain ledgers remain authoritative.

Chat image attachments now use the local visual sensory path. JPEG and PNG uploads are checked for base64 integrity, content/MIME agreement, byte size, dimensions, and pixel count before Ollama sees them. Raw pixels are discarded after that one local observation stage; downstream agents and memory receive only typed, provenance-marked, explicitly untrusted evidence. The path has no Gemini fallback and reports itself unavailable when Ollama has not declared `vision` for the configured model.

Chat audio attachments use the equivalent local auditory relay. Uploads must be 16 kHz mono 16-bit PCM WAV; microphone recordings are resampled and encoded to that format in the browser. The backend independently checks base64, RIFF structure, MIME, byte size, duration, channel/rate/depth, and simple signal quality before a capability-verified local Ollama model sees the clip. Raw audio is then removed. Transcripts and sound labels are provenance-marked untrusted evidence and are never appended to the user's instruction text. There is no cloud audio fallback.

Every turn also runs a deterministic predictive-perception assessment in enforced shadow mode. It forms bounded hypotheses from recent same-user context before reading the current episode, labels each as a prior rather than an observation, and records immutable matches, mismatches, or unresolved checks. Any clarification or image/audio recapture is a review-only recommendation: it is not shown or executed and cannot affect the response, routing, research, learning, or primary evidence.

### Configuration & Feature Flags

| Flag | Location | Purpose |
|------|----------|---------|
| `ATTENTION_CONTROLLER_ENABLED` | `.env` | Enable dynamic attention routing |
| `ATTENTION_CONTROLLER_SHADOW_MODE` | `.env` | Log decisions without affecting routing |
| `SALIENCE_NETWORK_ENABLED` | `.env` | Compute explainable post-retrieval memory rankings |
| `SALIENCE_NETWORK_SHADOW_MODE` | `.env` | Record rankings without exposing priority hints to synthesis |
| `SALIENCE_NETWORK_TOP_K` | `.env` | Bound the number of advisory priority hints (default: 3) |
| `SALIENCE_RECENCY_HALF_LIFE_DAYS` | `.env` | Configure temporal decay in salience scoring (default: 30) |
| `SLEEP_CYCLE_ENABLED` | `.env` | Start the single-owner idle consolidation scheduler (default: false) |
| `SLEEP_IDLE_MINUTES` | `.env` | Required inactivity before sleep work may begin (default: 30) |
| `SLEEP_COOLDOWN_MINUTES` | `.env` | Minimum delay between successful sleep jobs (default: 360) |
| `SLEEP_REQUIRE_LOCAL_PROVIDER` | `.env` | Refuse automatic sleep unless the configured provider stack is local |
| `AUTONOMOUS_WORK_MASTER_ENABLED` | `.env` | Master admission gate; UI state persists after the first operator change |
| `AUTONOMOUS_WORK_MAX_CONCURRENT` | `.env` | Global autonomous execution capacity (default: 1) |
| `AUTONOMOUS_{REFLECTION,DISCOVERY,CURIOSITY,SELF_ASSESSMENT,PROACTIVE}_ENABLED` | `.env` | Initial initiative-category posture (all default: false) |
| `AUTONOMOUS_{SUMMARY,STM_FLUSH}_ENABLED` | `.env` | Initial memory-housekeeping posture (both default: true) |
| `AUTONOMOUS_DEFAULT_TIMEOUT_SECONDS` | `.env` | Hard deadline per execution attempt (default: 300) |
| `AUTONOMOUS_DEFAULT_MAX_RETRIES` | `.env` | Bounded automatic retry budget (default: 1) |
| `TELEMETRY_REPLAY_SIZE` | `.env` | Process-local cursor replay window (default: 2000 events) |
| `TELEMETRY_SUBSCRIBER_QUEUE_SIZE` | `.env` | Per-viewer non-blocking delivery buffer (default: 256 events) |
| `VISUAL_INPUT_ENABLED` | `.env` | Enable the local visual sensory relay when model capability is verified (default: true) |
| `VISUAL_PROVIDER` | `.env` | Visual provider; only `ollama` or `disabled` is accepted |
| `OLLAMA_VISION_MODEL` | `.env` | Optional dedicated vision model; empty reuses `OLLAMA_CHAT_MODEL` |
| `VISUAL_MAX_IMAGE_BYTES` | `.env` | Decoded JPEG/PNG size limit (default: 8 MiB) |
| `VISUAL_MAX_IMAGE_PIXELS` | `.env` | Decoded image pixel limit (default: 24 million) |
| `AUDIO_INPUT_ENABLED` | `.env` | Enable the local auditory relay when model capability is verified (default: true) |
| `AUDIO_PROVIDER` | `.env` | Auditory provider; only `ollama` or `disabled` is accepted |
| `OLLAMA_AUDIO_MODEL` | `.env` | Optional dedicated audio model; empty reuses `OLLAMA_CHAT_MODEL` |
| `AUDIO_MAX_BYTES` | `.env` | Decoded PCM WAV limit (default: 4 MiB) |
| `AUDIO_MAX_DURATION_SECONDS` | `.env` | Clip duration limit (default: 60 seconds) |
| `MULTISENSORY_MAX_ALIGNMENT_SKEW_SECONDS` | `.env` | Same-turn text/image/audio binding window (default: 120 seconds) |
| `PREDICTIVE_PERCEPTION_ENABLED` | `.env` | Compute bounded predictive assessments (default: true) |
| `PREDICTIVE_PERCEPTION_SHADOW_MODE` | `.env` | Enforced safety posture; v1 rejects false rather than activating influence |
| `PREDICTIVE_PERCEPTION_MAX_PRIOR_CYCLES` | `.env` | Maximum recent same-user cycles considered (default: 3) |
| `PREDICTIVE_PERCEPTION_MAX_HYPOTHESES` | `.env` | Maximum labelled prior hypotheses per turn (default: 8) |
| `PREDICTIVE_PERCEPTION_MIN_OBSERVATION_RELIABILITY` | `.env` | Reliability floor for calibratable comparisons (default: 0.55) |
| `PREDICTIVE_PERCEPTION_CLARIFICATION_THRESHOLD` | `.env` | Minimum bounded surprise for a material mismatch (default: 0.50) |
| `STM_TOKEN_BUDGET` | `.env` | Short-term memory token limit (default: 25000) |
| `CONSOLIDATION_INTERVAL_MINUTES` | `.env` | Memory consolidation frequency (default: 30) |

See `.env.example` for the complete configuration reference.

---

## Usage

### Basic Chat Interaction

```python
import requests

response = requests.post("http://localhost:8000/chat", json={
    "input_text": "How does your memory system work?",
    "user_id": "user-123",
    "session_id": "session-456"
})

print(response.json()["final_response"])
```

### Cognitive Cycle Flow

1. **Input Processing**: User message enters via `/chat` endpoint
2. **Stage 1 Agents**: Perception, Emotion, Memory agents populate Working Memory
3. **Attention Control**: Drift detection adjusts routing and token budgets
4. **Meta-Cognition Gate**: Decides whether to answer, search, or ask for clarification
5. **Stage 2 Agents**: Planning, Creative, Critic, Discovery collaborate
6. **Synthesis**: CognitiveBrain generates final response with self-model integration
7. **Learning**: RL rewards computed, procedural skills tracked, ToM predictions validated
8. **Memory**: Interaction stored in STM, consolidated to LTM over time

---

## Testing

```bash
# Run all tests
pytest tests -q

# Run specific test suites
pytest tests/test_orchestration_service.py -v
pytest tests/test_memory_service.py -v
pytest tests/test_llm_integration_service.py -v
```

---

## Project Documents

- `architecture.md` is the audited description of implemented behavior and known boundaries.
- `roadmap.md` is the canonical plan and delivery tracker for the local-first ECA.

---

## Known Limitations

- **Cold start**: Bob needs 2-3 interactions to "warm up" after downtime as Working Memory populates
- **Single-user optimization**: RL and habits are per-user; cross-user generalization not yet implemented
- **LLM dependency**: Current code is Gemini-specific; the roadmap migrates routine cognition to local Ollama providers.
- **Consolidation scheduling**: The service exists, but its periodic loop is not wired into application startup.
- **Research retrieval**: Legacy direct browsing is disconnected; external research now has a deterministic, disabled-by-default policy boundary, with a live grounded provider still pending.

---

## Research & Citation

This repository accompanies research exploring how layered cortical-basal ganglia circuits can be approximated in production AI assistants. 

### Citing ECA

If you use ECA in your research, please cite:

```bibtex
@software{bentley2025eca,
  author = {Bentley, Ed},
  title = {Emergent Cognitive Architecture (ECA): A Brain-Inspired Learning System},
  year = {2025},
  url = {https://github.com/yourusername/eca},
  note = {Neuroscience-inspired multi-agent platform with reinforcement learning, 
         meta-cognition, and dynamic attention control}
}
```

### Key Documentation

- `architecture.md` - Audited system design, implementation status, and brain-region mappings
- `roadmap.md` - Local-first architecture migration and ECA delivery plan

---

## Contributing

We welcome contributions from researchers and developers interested in cognitive architectures.

### Areas of Interest

- Alternative learning algorithms (A3C, PPO instead of Q-learning)
- Multi-agent theory of mind extensions
- Advanced consolidation strategies (compressive summarization)
- Cross-user pattern generalization
- Real-world evaluation benchmarks

### Submitting Changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-improvement`)
3. Ensure tests pass (`pytest tests -q`)
4. Submit a pull request with clear description

See `CONTRIBUTING.md` for detailed guidelines.

---

## License

Distributed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.

This means:
- ✅ Free to use, modify, and distribute
- ✅ Academic and research use encouraged
- ⚠️ Network-accessible modifications **must** publish source code
- ⚠️ Derivative works must use the same license

If you need dual licensing for closed/commercial deployments, contact: **ed.j.bentley@gmail.com**

---

## Contact

- **Author**: Ed Bentley
- **Email**: ed.j.bentley@gmail.com
- **Issues**: GitHub Issues for bugs and feature requests

---

*"Not just a chatbot — a cognitive architecture that learns, adapts, and knows its limits."*
