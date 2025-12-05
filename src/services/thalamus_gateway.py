"""
Thalamus Gateway Service - Sensory Gating & Routing

Inspired by the brain's thalamus, which acts as a relay station for sensory information,
this service pre-processes user input to determine:
- Which agents need activation
- What memory depth is required
- How to route context appropriately

This enables selective attention and efficient resource usage.
"""

from src.models.agent_models import QuickAnalysis, InputRouting, AttentionDirective
from typing import Dict, Any, List, Optional
import re


class ThalamusGateway:
    """
    Pre-processes input and determines which agents need activation and what context they receive.
    Acts as a sensory relay station, filtering and routing information to appropriate cortical areas.
    """
    
    def __init__(self):
        self.question_patterns = [
            r'\?',
            r'^(what|who|where|when|why|how|can|could|would|should|do|does|did|is|are|was|were)',
            r'(tell me|explain|describe|show me|help me understand)'
        ]
        
        self.emotional_markers = [
            'feel', 'emotion', 'worried', 'excited', 'frustrated', 'happy', 'sad',
            'angry', 'anxious', 'scared', 'love', 'hate', 'afraid', 'concerned'
        ]
        
        self.creative_markers = [
            'how could', 'what if', 'imagine', 'create', 'design', 'innovative',
            'new way', 'alternative', 'different approach', 'creative', 'novel'
        ]
        
        self.urgent_markers = [
            'urgent', 'emergency', 'immediately', 'asap', 'critical', 'help!',
            'right now', 'quickly', 'hurry'
        ]
    
    async def route_input(self, user_input: str, user_id: str, has_image: bool = False, has_audio: bool = False) -> InputRouting:
        """
        Analyze input and determine optimal agent activation and memory configuration.
        
        Args:
            user_input: User's text input
            user_id: User identifier for context
            has_image: Whether image data is present
            has_audio: Whether audio data is present
            
        Returns:
            InputRouting with agent activation map and memory configuration
        """
        # Quick preliminary analysis
        quick_analysis = self._analyze_input(user_input, has_image, has_audio)
        
        # Determine which agents to activate
        agent_activation = self._determine_agent_activation(user_input, quick_analysis)
        
        # Determine memory depth needed
        memory_config = self._configure_memory_retrieval(quick_analysis)
        
        agent_token_budget = self._initialize_agent_token_budget(agent_activation, quick_analysis)

        return InputRouting(
            quick_analysis=quick_analysis,
            agent_activation=agent_activation,
            memory_config=memory_config,
            agent_token_budget=agent_token_budget,
            attention_motifs=[]
        )
    
    def _analyze_input(self, user_input: str, has_image: bool, has_audio: bool) -> QuickAnalysis:
        """
        Quick analysis of input characteristics.
        """
        user_input_lower = user_input.lower()
        
        # Detect modality
        modality = "text"
        if has_image and has_audio:
            modality = "multimodal"
        elif has_image:
            modality = "image"
        elif has_audio:
            modality = "audio"
        
        # Assess urgency
        urgency = "normal"
        if any(marker in user_input_lower for marker in self.urgent_markers):
            urgency = "high"
        elif len(user_input.split()) < 5 and not any(pattern for pattern in self.question_patterns if re.search(pattern, user_input_lower, re.IGNORECASE)):
            urgency = "low"  # Short statements are typically low urgency
        
        # Assess complexity
        complexity = "moderate"
        word_count = len(user_input.split())
        sentence_count = len([s for s in user_input.split('.') if s.strip()])
        has_questions = any(re.search(pattern, user_input_lower, re.IGNORECASE) for pattern in self.question_patterns)
        
        if word_count > 50 or sentence_count > 3 or (has_questions and any(marker in user_input_lower for marker in self.creative_markers)):
            complexity = "complex"
        elif word_count < 10 and sentence_count == 1 and not has_questions:
            complexity = "simple"
        
        # Assess context need
        context_need = "recent"
        if any(phrase in user_input_lower for phrase in ['remember', 'last time', 'before', 'earlier', 'we discussed', 'you said']):
            context_need = "deep"
        elif complexity == "simple" and urgency == "low":
            context_need = "minimal"
        
        return QuickAnalysis(
            modality=modality,
            urgency=urgency,
            complexity=complexity,
            context_need=context_need
        )
    
    def _determine_agent_activation(self, user_input: str, quick_analysis: QuickAnalysis) -> Dict[str, bool]:
        """
        Determine which agents should be activated based on input analysis.
        
        Selective activation saves compute and mimics brain's attention mechanism.
        """
        user_input_lower = user_input.lower()
        
        # Perception is always active (primary sensory processing)
        agent_activation = {
            "perception": True
        }
        
        # Emotional agent: Skip for purely factual queries or low urgency
        has_emotional_content = any(marker in user_input_lower for marker in self.emotional_markers)
        agent_activation["emotional"] = (
            quick_analysis.urgency != "low" or 
            has_emotional_content or
            quick_analysis.complexity == "complex"
        )
        
        # Memory agent: Skip for minimal context needs
        agent_activation["memory"] = quick_analysis.context_need != "minimal"
        
        # Planning agent: Activate for moderate/complex tasks
        agent_activation["planning"] = quick_analysis.complexity != "simple"
        
        # Creative agent: Activate for creative/exploratory queries
        has_creative_markers = any(marker in user_input_lower for marker in self.creative_markers)
        is_how_question = re.search(r'\bhow\b', user_input_lower, re.IGNORECASE) is not None
        agent_activation["creative"] = has_creative_markers or (is_how_question and quick_analysis.complexity != "simple")
        
        # Critic agent: Activate for complex reasoning or when multiple agents are active
        active_count = sum(agent_activation.values())
        agent_activation["critic"] = quick_analysis.complexity == "complex" or active_count >= 4
        
        # Discovery agent: Activate for deep context or question-based queries
        has_question = any(re.search(pattern, user_input_lower, re.IGNORECASE) for pattern in self.question_patterns)
        agent_activation["discovery"] = has_question or quick_analysis.context_need == "deep"
        
        return agent_activation
    
    def _configure_memory_retrieval(self, quick_analysis: QuickAnalysis) -> Dict[str, Any]:
        """
        Configure memory retrieval parameters based on context needs.
        
        Mimics how attention and arousal modulate memory recall in the brain.
        """
        memory_configs = {
            "minimal": {
                "limit": 1,
                "min_relevance": 0.7,
                "description": "Retrieve only most relevant memory"
            },
            "recent": {
                "limit": 3,
                "min_relevance": 0.5,
                "description": "Retrieve recent relevant context"
            },
            "deep": {
                "limit": 10,
                "min_relevance": 0.4,
                "description": "Deep memory search for rich context"
            }
        }
        
        return memory_configs[quick_analysis.context_need]

    def apply_attention_directive(
        self,
        input_routing: InputRouting,
        directive: AttentionDirective,
        attention_motifs: Optional[List[str]] = None
    ) -> None:
        """Adjust token budgets and motifs based on attention directive."""
        budgets = dict(input_routing.agent_token_budget)
        for agent, bias in directive.agent_bias.items():
            if agent not in budgets:
                continue
            base = budgets[agent]
            adjustment = int(base * 0.3 * bias)
            budgets[agent] = max(256, min(4096, base + adjustment))
        input_routing.agent_token_budget = budgets

        if attention_motifs:
            trimmed = list(dict.fromkeys(attention_motifs))[:6]
            input_routing.attention_motifs = trimmed

    def _initialize_agent_token_budget(self, agent_activation: Dict[str, bool], quick_analysis: QuickAnalysis) -> Dict[str, int]:
        """Create baseline token budgets per agent based on complexity."""
        base_budget = 1024 if quick_analysis.complexity == "simple" else 1536
        complex_bonus = 512 if quick_analysis.complexity == "complex" else 0
        budgets = {}
        for agent, active in agent_activation.items():
            if not active:
                continue
            bonus = 256 if agent in {"creative", "discovery"} else 0
            budgets[agent] = base_budget + complex_bonus + bonus
        return budgets
