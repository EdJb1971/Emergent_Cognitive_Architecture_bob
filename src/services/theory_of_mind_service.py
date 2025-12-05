"""
Theory of Mind Service - Understanding User Mental States

Inspired by human theory of mind (mentalizing), this service:
- Predicts user intentions and goals
- Models user's current mental/emotional state
- Tracks user's knowledge and confusions
- Enables empathy-driven, intention-aware responses
- Validates predictions to improve accuracy over time

This is what makes the system truly understand the user, not just respond to text.
"""

import logging
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from uuid import uuid4

from src.models.agent_models import UserMentalState, TheoryOfMindPrediction
from src.models.core_models import CognitiveCycle, AgentOutput
from src.services.llm_integration_service import LLMIntegrationService
from src.services.autobiographical_memory_system import AutobiographicalMemorySystem

logger = logging.getLogger(__name__)


class TheoryOfMindService:
    """
    Models user's mental states, intentions, and knowledge.
    Enables the system to respond with true understanding and empathy.
    """
    
    def __init__(
        self,
        llm_service: LLMIntegrationService,
        autobiographical_system: Optional[AutobiographicalMemorySystem] = None
    ):
        self.llm_service = llm_service
        self.autobiographical_system = autobiographical_system
        self.mental_state_cache: Dict[str, UserMentalState] = {}  # user_id -> current state
        self.predictions: Dict[str, List[TheoryOfMindPrediction]] = {}  # user_id -> predictions
        logger.info("TheoryOfMindService initialized.")
    
    async def infer_mental_state(
        self,
        cycle: CognitiveCycle,
        conversation_history: Optional[List[CognitiveCycle]] = None
    ) -> UserMentalState:
        """
        Infer the user's current mental state from the cycle and history.
        
        Args:
            cycle: Current cognitive cycle
            conversation_history: Optional recent conversation history
            
        Returns:
            UserMentalState object
        """
        user_id = str(cycle.user_id)
        
        # Get previous state if available
        previous_state = self.mental_state_cache.get(user_id)
        
        # Extract evidence from current cycle
        emotional_state = self._extract_emotional_state(cycle)
        current_goal = await self._infer_current_goal(cycle, conversation_history)
        current_needs = await self._infer_current_needs(cycle, emotional_state)
        conversation_intent = self._classify_conversation_intent(cycle)
        
        # Extract knowledge state
        knows_about, confused_about, interested_in = await self._assess_knowledge_state(
            cycle, conversation_history
        )
        
        # Predict likely next action
        likely_next_action = await self._predict_next_action(cycle, current_goal)
        
        # Calculate confidence
        confidence = self._calculate_confidence(cycle, conversation_history)
        uncertainty_factors = self._identify_uncertainty_factors(cycle)
        
        mental_state = UserMentalState(
            user_id=user_id,
            timestamp=datetime.utcnow(),
            current_goal=current_goal,
            current_emotion=emotional_state,
            current_needs=current_needs,
            knows_about=knows_about,
            confused_about=confused_about,
            interested_in=interested_in,
            likely_next_action=likely_next_action,
            conversation_intent=conversation_intent,
            confidence=confidence,
            uncertainty_factors=uncertainty_factors
        )
        
        # Cache the state
        self.mental_state_cache[user_id] = mental_state
        
        logger.debug(
            f"Inferred mental state for user {user_id}: "
            f"goal='{current_goal}', emotion={emotional_state}, "
            f"intent={conversation_intent}, confidence={confidence:.2f}"
        )
        
        return mental_state
    
    def _extract_emotional_state(self, cycle: CognitiveCycle) -> str:
        """Extract current emotional state from emotional agent output."""
        for output in cycle.agent_outputs:
            if output.agent_id == "emotional_agent":
                sentiment = output.analysis.get("current_sentiment", "neutral")
                intensity = output.analysis.get("intensity", "low")
                
                # Map to emotional state
                if sentiment in ["very_positive", "positive"] and intensity == "high":
                    return "excited"
                elif sentiment in ["very_positive", "positive"]:
                    return "happy"
                elif sentiment in ["very_negative", "negative"] and intensity == "high":
                    return "distressed"
                elif sentiment in ["very_negative", "negative"]:
                    return "concerned"
                elif sentiment == "mixed":
                    return "ambivalent"
                else:
                    return "neutral"
        
        return "neutral"
    
    async def _infer_current_goal(
        self,
        cycle: CognitiveCycle,
        conversation_history: Optional[List[CognitiveCycle]] = None
    ) -> Optional[str]:
        """Infer what the user is trying to accomplish."""
        user_input = cycle.user_input.lower()
        
        # Check for explicit goal markers
        if any(word in user_input for word in ["how do i", "how can i", "help me", "i need to", "i want to"]):
            # Extract the goal
            try:
                prompt = f"""
The user said: "{cycle.user_input}"

What is their goal? Provide a concise goal statement (max 10 words).
Example: "learn how to configure the system", "fix a bug", "understand a concept"
"""
                
                goal = await self.llm_service.generate_text(
                    prompt=prompt,
                    max_tokens=50,
                    temperature=0.3
                )
                
                return goal.strip()
                
            except Exception as e:
                logger.warning(f"Could not extract goal: {e}")
        
        # Check working memory for inferred goals
        if "working_memory" in cycle.metadata:
            inferred_goals = cycle.metadata["working_memory"].get("inferred_goals", [])
            if inferred_goals:
                return inferred_goals[0]  # Take the first inferred goal
        
        return None
    
    async def _infer_current_needs(
        self,
        cycle: CognitiveCycle,
        emotional_state: str
    ) -> List[str]:
        """Infer what the user needs right now based on context and emotion."""
        needs = []
        
        # Emotional needs
        if emotional_state in ["distressed", "concerned"]:
            needs.append("emotional_support")
        
        # Information needs (check for questions)
        if "?" in cycle.user_input:
            needs.append("information")
        
        # Clarity needs (check critic agent for confusion)
        for output in cycle.agent_outputs:
            if output.agent_id == "critic_agent":
                coherence = output.analysis.get("logical_coherence", "high")
                if coherence == "low":
                    needs.append("clarity")
        
        # Validation needs (check for uncertainty markers)
        if any(word in cycle.user_input.lower() for word in ["right?", "correct?", "is that", "am i"]):
            needs.append("validation")
        
        # Guidance needs
        if any(word in cycle.user_input.lower() for word in ["how do i", "what should i", "help me"]):
            needs.append("guidance")
        
        return needs if needs else ["engagement"]  # Default to engagement
    
    def _classify_conversation_intent(self, cycle: CognitiveCycle) -> str:
        """Classify overall conversation intent."""
        user_input = cycle.user_input.lower()
        
        # Check for patterns
        if "?" in cycle.user_input:
            return "seeking_info"
        elif any(word in user_input for word in ["how do", "how can", "how to"]):
            return "problem_solving"
        elif any(word in user_input for word in ["feel", "worried", "frustrated", "excited"]):
            return "venting"
        elif any(word in user_input for word in ["what if", "imagine", "could we"]):
            return "exploring"
        elif any(word in user_input for word in ["tell me about", "explain", "what is"]):
            return "learning"
        else:
            return "engaging"
    
    async def _assess_knowledge_state(
        self,
        cycle: CognitiveCycle,
        conversation_history: Optional[List[CognitiveCycle]] = None
    ) -> tuple[List[str], List[str], List[str]]:
        """
        Assess what the user knows, is confused about, and is interested in.
        
        Returns:
            (knows_about, confused_about, interested_in) tuple of lists
        """
        knows_about = []
        confused_about = []
        interested_in = []
        
        # Extract from perception agent (topics)
        for output in cycle.agent_outputs:
            if output.agent_id == "perception_agent":
                topics = output.analysis.get("topics", [])
                interested_in.extend(topics)
        
        # Extract confusion from critic agent
        for output in cycle.agent_outputs:
            if output.agent_id == "critic_agent":
                contradictions = output.analysis.get("contradictions_found", [])
                if contradictions:
                    confused_about.extend(contradictions[:2])  # Top 2 confusions
        
        # Check for understanding markers
        user_input_lower = cycle.user_input.lower()
        if any(phrase in user_input_lower for phrase in ["i understand", "i see", "makes sense", "got it"]):
            # Extract what they understand
            perception_output = next((o for o in cycle.agent_outputs if o.agent_id == "perception_agent"), None)
            if perception_output:
                topics = perception_output.analysis.get("topics", [])
                knows_about.extend(topics)
        
        # Check for confusion markers
        if any(phrase in user_input_lower for phrase in ["i don't understand", "confused", "not clear", "what does"]):
            # Extract what they're confused about
            perception_output = next((o for o in cycle.agent_outputs if o.agent_id == "perception_agent"), None)
            if perception_output:
                topics = perception_output.analysis.get("topics", [])
                confused_about.extend(topics)
        
        return (
            list(set(knows_about))[:5],  # Top 5 unique
            list(set(confused_about))[:3],  # Top 3 unique
            list(set(interested_in))[:5]  # Top 5 unique
        )
    
    async def _predict_next_action(
        self,
        cycle: CognitiveCycle,
        current_goal: Optional[str]
    ) -> Optional[str]:
        """Predict what the user will likely do/ask next."""
        if not current_goal:
            return None
        
        # Simple prediction based on goal
        user_input_lower = cycle.user_input.lower()
        
        if "how do i" in user_input_lower and cycle.final_response:
            return "ask_follow_up_question"
        elif "?" in cycle.user_input:
            return "wait_for_response_then_clarify"
        elif any(word in user_input_lower for word in ["thanks", "thank you", "got it"]):
            return "end_conversation_or_new_topic"
        else:
            return "continue_exploration"
    
    def _calculate_confidence(
        self,
        cycle: CognitiveCycle,
        conversation_history: Optional[List[CognitiveCycle]] = None
    ) -> float:
        """Calculate confidence in the mental state model."""
        confidence = 0.5  # Baseline
        
        # More agent outputs = more evidence
        if len(cycle.agent_outputs) >= 5:
            confidence += 0.2
        elif len(cycle.agent_outputs) >= 3:
            confidence += 0.1
        
        # Emotional agent with high confidence
        emotional_output = next((o for o in cycle.agent_outputs if o.agent_id == "emotional_agent"), None)
        if emotional_output and emotional_output.confidence > 0.7:
            confidence += 0.15
        
        # Memory agent with high confidence (we know the user)
        memory_output = next((o for o in cycle.agent_outputs if o.agent_id == "memory_agent"), None)
        if memory_output and memory_output.confidence > 0.7:
            confidence += 0.15
        
        # Conversation history available
        if conversation_history and len(conversation_history) > 3:
            confidence += 0.1
        
        return min(1.0, confidence)
    
    def _identify_uncertainty_factors(self, cycle: CognitiveCycle) -> List[str]:
        """Identify what makes us uncertain about the mental state model."""
        uncertainties = []
        
        # Short input
        if len(cycle.user_input.split()) < 5:
            uncertainties.append("brief_input")
        
        # Low emotional confidence
        emotional_output = next((o for o in cycle.agent_outputs if o.agent_id == "emotional_agent"), None)
        if emotional_output and emotional_output.confidence < 0.5:
            uncertainties.append("unclear_emotional_state")
        
        # Mixed sentiment
        if emotional_output and emotional_output.analysis.get("current_sentiment") == "mixed":
            uncertainties.append("ambiguous_emotion")
        
        # Low memory confidence
        memory_output = next((o for o in cycle.agent_outputs if o.agent_id == "memory_agent"), None)
        if memory_output and memory_output.confidence < 0.5:
            uncertainties.append("limited_history")
        
        # Conflicts detected
        if "final_conflicts" in cycle.metadata:
            conflicts = cycle.metadata["final_conflicts"].get("conflicts", [])
            if conflicts:
                uncertainties.append("conflicting_agent_interpretations")
        
        return uncertainties
    
    async def make_prediction(
        self,
        user_id: str,
        mental_state: UserMentalState,
        prediction_type: str = "intention"
    ) -> TheoryOfMindPrediction:
        """
        Make a specific prediction about the user's mental state or behavior.
        
        Args:
            user_id: User identifier
            mental_state: Current mental state model
            prediction_type: Type of prediction to make
            
        Returns:
            TheoryOfMindPrediction object
        """
        prediction_id = str(uuid4())
        
        # Construct prediction based on mental state
        predicted_intention = mental_state.likely_next_action or "unknown"
        predicted_needs = mental_state.current_needs
        predicted_confusion_points = mental_state.confused_about
        
        # Generate reasoning
        reasoning = (
            f"Based on user's current goal '{mental_state.current_goal}', "
            f"emotional state '{mental_state.current_emotion}', "
            f"and conversation intent '{mental_state.conversation_intent}'"
        )
        
        # Collect evidence
        evidence = [
            f"Current goal: {mental_state.current_goal}",
            f"Emotion: {mental_state.current_emotion}",
            f"Intent: {mental_state.conversation_intent}"
        ]
        
        if mental_state.interested_in:
            evidence.append(f"Interested in: {', '.join(mental_state.interested_in[:3])}")
        
        prediction = TheoryOfMindPrediction(
            prediction_id=prediction_id,
            user_id=user_id,
            predicted_intention=predicted_intention,
            predicted_needs=predicted_needs,
            predicted_confusion_points=predicted_confusion_points,
            confidence=mental_state.confidence,
            reasoning=reasoning,
            evidence=evidence
        )
        
        # Store prediction for validation
        if user_id not in self.predictions:
            self.predictions[user_id] = []
        self.predictions[user_id].append(prediction)
        
        # Keep only recent predictions (last 10)
        self.predictions[user_id] = self.predictions[user_id][-10:]
        
        logger.debug(f"Made prediction {prediction_id} for user {user_id}: {predicted_intention}")
        
        return prediction
    
    def get_mental_state_context(self, user_id: str) -> str:
        """
        Get formatted mental state context for prompt injection.
        
        Args:
            user_id: User identifier
            
        Returns:
            Formatted string for prompt context
        """
        state = self.mental_state_cache.get(user_id)
        if not state:
            return "THEORY OF MIND: No mental state model available yet."
        
        context_parts = ["THEORY OF MIND (Understanding the User):"]
        
        if state.current_goal:
            context_parts.append(f"- User's goal: {state.current_goal}")
        
        context_parts.append(f"- Emotional state: {state.current_emotion}")
        context_parts.append(f"- Current needs: {', '.join(state.current_needs)}")
        context_parts.append(f"- Conversation intent: {state.conversation_intent}")
        
        if state.confused_about:
            context_parts.append(f"- User is confused about: {', '.join(state.confused_about)}")
        
        if state.interested_in:
            context_parts.append(f"- User is interested in: {', '.join(state.interested_in[:3])}")
        
        if state.likely_next_action:
            context_parts.append(f"- Likely next action: {state.likely_next_action}")
        
        context_parts.append(f"- Confidence in this model: {state.confidence:.2f}")
        
        if state.uncertainty_factors:
            context_parts.append(f"- Uncertainties: {', '.join(state.uncertainty_factors)}")
        
        return "\n".join(context_parts)
    
    async def auto_validate_predictions(
        self,
        user_id: str,
        current_cycle: CognitiveCycle,
        previous_cycle: Optional[CognitiveCycle] = None
    ) -> List[Dict[str, Any]]:
        """
        Automatically validate previous predictions against current user behavior.
        
        Args:
            user_id: User identifier
            current_cycle: Current cognitive cycle with actual user behavior
            previous_cycle: Previous cycle (if available)
            
        Returns:
            List of validation results
        """
        if user_id not in self.predictions or not self.predictions[user_id]:
            return []
        
        validation_results = []
        
        # Get most recent unvalidated predictions (last 1-3)
        recent_predictions = [p for p in self.predictions[user_id][-3:] if not p.was_validated]
        
        for pred in recent_predictions:
            validation = await self._validate_single_prediction(pred, current_cycle, previous_cycle)
            if validation:
                validation_results.append(validation)
                
                # Update prediction with validation
                pred.was_validated = True
                pred.validation_result = validation["was_correct"]
                pred.validation_feedback = validation["feedback"]
                
                logger.info(
                    f"Auto-validated prediction {pred.prediction_id}: "
                    f"{'✓ correct' if validation['was_correct'] else '✗ incorrect'} - {validation['feedback']}"
                )
        
        return validation_results
    
    async def _validate_single_prediction(
        self,
        prediction: TheoryOfMindPrediction,
        current_cycle: CognitiveCycle,
        previous_cycle: Optional[CognitiveCycle] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Validate a single prediction against actual behavior.
        
        Returns:
            Validation result dict or None if unable to validate
        """
        validation = {
            "prediction_id": prediction.prediction_id,
            "predicted_intention": prediction.predicted_intention,
            "was_correct": False,
            "feedback": "",
            "confidence_adjustment": 0.0
        }
        
        user_input_lower = current_cycle.user_input.lower()
        
        # Validate predicted next action
        if prediction.predicted_intention:
            actual_action = self._classify_actual_action(current_cycle, previous_cycle)
            
            # Check if prediction matches actual action
            if prediction.predicted_intention == actual_action:
                validation["was_correct"] = True
                validation["feedback"] = f"User did {actual_action} as predicted"
                validation["confidence_adjustment"] = +0.05
            else:
                validation["was_correct"] = False
                validation["feedback"] = f"User did {actual_action}, not {prediction.predicted_intention}"
                validation["confidence_adjustment"] = -0.03
        
        # Validate predicted needs
        if prediction.predicted_needs:
            actual_needs = await self._infer_current_needs(current_cycle, self._extract_emotional_state(current_cycle))
            
            # Check overlap
            needs_overlap = len(set(prediction.predicted_needs) & set(actual_needs))
            if needs_overlap > 0:
                validation["was_correct"] = validation.get("was_correct", True)
                validation["feedback"] += f" | Needs: {needs_overlap}/{len(prediction.predicted_needs)} matched"
                validation["confidence_adjustment"] += 0.02 * needs_overlap
            else:
                validation["feedback"] += f" | Needs: 0/{len(prediction.predicted_needs)} matched"
                validation["confidence_adjustment"] -= 0.02
        
        # Validate predicted confusion points
        if prediction.predicted_confusion_points:
            # Check if user asked about predicted confusion points
            confusion_match = any(
                confusion_point.lower() in user_input_lower 
                for confusion_point in prediction.predicted_confusion_points
            )
            
            if confusion_match:
                validation["was_correct"] = True
                validation["feedback"] += " | User asked about predicted confusion point"
                validation["confidence_adjustment"] += 0.05
        
        return validation
    
    def _classify_actual_action(
        self,
        current_cycle: CognitiveCycle,
        previous_cycle: Optional[CognitiveCycle] = None
    ) -> str:
        """Classify what the user actually did in this cycle."""
        user_input_lower = current_cycle.user_input.lower()
        
        # Follow-up question
        if "?" in current_cycle.user_input and previous_cycle:
            return "ask_follow_up_question"
        
        # Clarification request
        if any(phrase in user_input_lower for phrase in ["what do you mean", "can you explain", "clarify", "i don't understand"]):
            return "wait_for_response_then_clarify"
        
        # Ending conversation
        if any(phrase in user_input_lower for phrase in ["thanks", "thank you", "bye", "goodbye", "got it", "okay thanks"]):
            return "end_conversation_or_new_topic"
        
        # Continuing exploration
        if any(phrase in user_input_lower for phrase in ["tell me more", "what about", "how about", "also"]):
            return "continue_exploration"
        
        # New topic
        if previous_cycle:
            prev_topics = []
            for output in previous_cycle.agent_outputs:
                if output.agent_id == "perception_agent":
                    prev_topics = output.analysis.get("topics", [])
            
            curr_topics = []
            for output in current_cycle.agent_outputs:
                if output.agent_id == "perception_agent":
                    curr_topics = output.analysis.get("topics", [])
            
            # If topics are completely different
            if prev_topics and curr_topics and not any(t in prev_topics for t in curr_topics):
                return "end_conversation_or_new_topic"
        
        return "continue_exploration"
    
    async def validate_prediction(
        self,
        prediction_id: str,
        actual_outcome: str,
        was_correct: bool
    ):
        """
        Manually validate a prediction after the fact to improve future predictions.
        
        Args:
            prediction_id: Prediction identifier
            actual_outcome: What actually happened
            was_correct: Whether the prediction was correct
        """
        # Find the prediction
        for user_predictions in self.predictions.values():
            for pred in user_predictions:
                if pred.prediction_id == prediction_id:
                    pred.was_validated = True
                    pred.validation_result = was_correct
                    pred.validation_feedback = actual_outcome
                    
                    logger.info(
                        f"Manually validated prediction {prediction_id}: "
                        f"{'correct' if was_correct else 'incorrect'} - {actual_outcome}"
                    )
                    return
        
        logger.warning(f"Prediction {prediction_id} not found for validation")
    
    def get_validation_stats(self, user_id: str) -> Dict[str, Any]:
        """
        Get statistics on prediction accuracy for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            Dictionary with validation statistics
        """
        if user_id not in self.predictions:
            return {"total_predictions": 0, "validated": 0, "accuracy": 0.0}
        
        user_predictions = self.predictions[user_id]
        validated = [p for p in user_predictions if p.was_validated]
        correct = [p for p in validated if p.validation_result is True]
        
        stats = {
            "total_predictions": len(user_predictions),
            "validated": len(validated),
            "correct": len(correct),
            "incorrect": len(validated) - len(correct),
            "accuracy": len(correct) / len(validated) if validated else 0.0,
            "pending_validation": len(user_predictions) - len(validated)
        }
        
        return stats
