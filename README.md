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
├─ architecture.md         # Comprehensive design documentation
├─ completebrainplan.md    # Neuroscience-aligned roadmap with decision logs
└─ README.md               # You are here
```

---

## Getting Started

### Prerequisites
- Python 3.11+
- Node.js 18+ (for the optional dashboard frontend)
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

# Copy environment template and add your keys
cp .env.example .env
# Edit .env with your GEMINI_API_KEY, etc.

# Run the server
uvicorn main:app --reload
```

### Frontend Dashboard (Optional)

```bash
cd frontend
npm install
npm run dev
```

The frontend consumes the FastAPI backend for live cycle traces, drift telemetry, and learning metrics.

### Configuration & Feature Flags

| Flag | Location | Purpose |
|------|----------|---------|
| `ATTENTION_CONTROLLER_ENABLED` | `.env` | Enable dynamic attention routing |
| `ATTENTION_CONTROLLER_SHADOW_MODE` | `.env` | Log decisions without affecting routing |
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

## Roadmap

| Phase | Status | Focus |
|-------|--------|-------|
| Phase 1-5 | ✅ Complete | Foundation, perception, memory, executive functions, metacognition |
| Phase 6 | ✅ Complete | Reinforcement learning, procedural learning, composite rewards |
| Phase 7 | 🚧 In Progress | Attention controller (drift detection live), salience network |
| Phase 8 | 📋 Planned | Predictive coding, episodic future thinking |

Detailed progress notes, neuroscience references, and decision logs are in `completebrainplan.md`.

---

## Known Limitations

- **Cold start**: Bob needs 2-3 interactions to "warm up" after downtime as Working Memory populates
- **Single-user optimization**: RL and habits are per-user; cross-user generalization not yet implemented
- **LLM dependency**: Cognitive quality bounded by underlying Gemini model capabilities
- **Consolidation latency**: Memory consolidation runs every 30 minutes (not real-time)
- **Web scraping**: Some sites block non-browser traffic; graceful fallback to search snippets

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

- `architecture.md` - Comprehensive technical design with brain-region mappings
- `completebrainplan.md` - Detailed roadmap with neuroscience references and implementation notes

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