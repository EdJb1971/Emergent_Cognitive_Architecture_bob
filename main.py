import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response, status, HTTPException, Query, Path, Depends, WebSocket
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware


from src.core.config import settings
from src.core.logging_config import setup_logging
from src.core.exceptions import APIException, ConfigurationError, AgentServiceException
from src.services.llm_integration_service import LLMIntegrationService
from src.services.memory_service import MemoryService
from src.services.emotional_memory_service import EmotionalMemoryService
from src.services.meta_cognitive_monitor import MetaCognitiveMonitor
from src.services.procedural_learning_service import ProceduralLearningService
from src.services.orchestration_service import OrchestrationService
from src.services.cognitive_brain import CognitiveBrain
from src.services.background_task_queue import BackgroundTaskQueue
from src.services.self_reflection_discovery_engine import SelfReflectionAndDiscoveryEngine
from src.services.proactive_engagement_service import ProactiveEngagementEngine, ProactiveMessage
from src.services.metrics_service import MetricsService, MetricType
from src.services.reinforcement_learning_service import ReinforcementLearningService
from src.models.core_models import (
    UserRequest, AgentOutput, CognitiveCycle, MemoryQueryRequest, MemoryQueryResponse,
    ReflectionTriggerRequest, DiscoveryTriggerRequest, PatternsResponse, DiscoveredPattern,
    CycleUpdateRequest, CycleListRequest, CycleListResponse
)
from src.agents.perception_agent import PerceptionAgent
from src.agents.emotional_agent import EmotionalAgent
from src.agents.memory_agent import MemoryAgent
from src.agents.planning_agent import PlanningAgent
from src.agents.creative_agent import CreativeAgent
from src.agents.critic_agent import CriticAgent
from src.agents.discovery_agent import DiscoveryAgent
from src.services.web_browsing_service import WebBrowsingService
from src.services.audio_input_processor import AudioInputProcessor
from src.dependencies import APIKeyAuth, get_api_key_user_id # Import the authentication dependency
from typing import Dict, Any, List, Optional
from uuid import UUID
from datetime import datetime, timedelta
from src.core.exceptions import LLMServiceException, ConfigurationError

# Setup logging as the very first step
from src.core.logging_config import setup_logging
setup_logging()
logger = logging.getLogger(__name__)
# Application lifespan events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup and shutdown events for the FastAPI application."""
    logger.info(f"Starting up {settings.APP_NAME} in {settings.ENVIRONMENT} environment...")

    try:
        # Initialize LLMIntegrationService
        app.state.llm_service = LLMIntegrationService()
        logger.info("LLMIntegrationService initialized successfully.")

        # Initialize MemoryService and connect to ChromaDB
        app.state.memory_service = MemoryService(llm_service=app.state.llm_service, metrics_service=None)
        await app.state.memory_service.connect()
        logger.info("MemoryService initialized and connected to ChromaDB.")

        # Initialize MetricsService for comprehensive tracking with ChromaDB persistence
        # Reuse the ChromaDB client from MemoryService to avoid conflicts
        app.state.metrics_service = MetricsService(chroma_client=app.state.memory_service.client)
        logger.info("MetricsService initialized for dashboard analytics with ChromaDB persistence.")

        # Wire MetricsService into MemoryService now that it's created
        app.state.memory_service.metrics_service = app.state.metrics_service

        # Initialize EmotionalMemoryService for relational intelligence

        # Initialize EmotionalMemoryService for relational intelligence
        app.state.emotional_memory_service = EmotionalMemoryService(
            memory_service=app.state.memory_service,
            llm_service=app.state.llm_service,
            metrics_service=app.state.metrics_service
        )
        # Connect to ChromaDB for persistent profile storage
        await app.state.emotional_memory_service.connect()
        logger.info("EmotionalMemoryService initialized and connected to ChromaDB.")

        # Initialize MetaCognitiveMonitor for "feeling of knowing" assessment
        app.state.meta_cognitive_monitor = MetaCognitiveMonitor(
            memory_service=app.state.memory_service,
            llm_service=app.state.llm_service
        )
        logger.info("MetaCognitiveMonitor initialized for knowledge gap detection.")

        # Initialize ProceduralLearningService for skill refinement
        app.state.procedural_learning_service = ProceduralLearningService(
            memory_service=app.state.memory_service,
            llm_service=app.state.llm_service,
            metrics_service=app.state.metrics_service
        )
        logger.info("ProceduralLearningService initialized for skill refinement and sequence learning.")

        # Initialize Brain Architecture Services (Phase 1)
        from src.services.self_model_service import SelfModelService
        from src.services.working_memory_buffer import WorkingMemoryBuffer
        from src.services.emotional_salience_encoder import EmotionalSalienceEncoder
        
        app.state.self_model_service = SelfModelService()
        await app.state.self_model_service.connect(client=app.state.memory_service.client)
        logger.info("SelfModelService initialized and connected to ChromaDB.")
        
        app.state.working_memory_buffer = WorkingMemoryBuffer()
        logger.info("WorkingMemoryBuffer initialized successfully.")
        
        app.state.emotional_salience_encoder = EmotionalSalienceEncoder()
        logger.info("EmotionalSalienceEncoder initialized successfully.")

        # Initialize Brain Architecture Services (Phase 2)
        from src.services.thalamus_gateway import ThalamusGateway
        from src.services.attention_controller import AttentionController
        from src.services.conflict_monitor import ConflictMonitor
        from src.services.contextual_memory_encoder import ContextualMemoryEncoder
        
        app.state.thalamus_gateway = ThalamusGateway()
        logger.info("ThalamusGateway initialized successfully.")

        app.state.attention_controller = AttentionController(
            enabled=settings.ATTENTION_CONTROLLER_ENABLED,
            shadow_mode=settings.ATTENTION_CONTROLLER_SHADOW_MODE,
        )
        logger.info(
            "AttentionController initialized (enabled=%s, shadow_mode=%s)",
            settings.ATTENTION_CONTROLLER_ENABLED,
            app.state.attention_controller.shadow_mode,
        )
        
        app.state.conflict_monitor = ConflictMonitor()
        logger.info("ConflictMonitor initialized successfully.")
        
        app.state.contextual_memory_encoder = ContextualMemoryEncoder()
        logger.info("ContextualMemoryEncoder initialized successfully.")

        # Initialize Brain Architecture Services (Phase 3)
        from src.services.autobiographical_memory_system import AutobiographicalMemorySystem
        from src.services.memory_consolidation_service import MemoryConsolidationService
        from src.services.theory_of_mind_service import TheoryOfMindService
        
        app.state.autobiographical_memory_system = AutobiographicalMemorySystem()
        await app.state.autobiographical_memory_system.connect(client=app.state.memory_service.client)
        logger.info("AutobiographicalMemorySystem initialized and connected to ChromaDB.")
        
        # Note: MemoryConsolidationService will be initialized after ProactiveEngagementEngine
        # so it can generate proactive messages from consolidation insights
        
        app.state.theory_of_mind_service = TheoryOfMindService(
            llm_service=app.state.llm_service,
            autobiographical_system=app.state.autobiographical_memory_system
        )
        logger.info("TheoryOfMindService initialized successfully.")

        # Initialize Background Task Queue
        app.state.background_task_queue = BackgroundTaskQueue()
        logger.info("BackgroundTaskQueue initialized successfully.")

        # Initialize DecisionEngine and wire it into MemoryService
        from src.services.decision_engine import create_decision_engine
        app.state.decision_engine = create_decision_engine(app.state.background_task_queue)

        # Register DecisionEngine with MemoryService
        try:
            app.state.memory_service.set_decision_engine(app.state.decision_engine)
            logger.info("DecisionEngine instantiated and wired to MemoryService.")
        except Exception:
            logger.warning("DecisionEngine created but could not be wired to MemoryService.")

        # Initialize Reinforcement Learning Service (Basal Ganglia - Strategy Learning)
        app.state.rl_service = ReinforcementLearningService()
        await app.state.rl_service.connect(client=app.state.memory_service.client)
        logger.info("ReinforcementLearningService initialized and connected to ChromaDB.")

        # Initialize Proactive Engagement Engine (Bob's ability to initiate conversations)
        app.state.proactive_engine = ProactiveEngagementEngine(
            llm_service=app.state.llm_service,
            memory_service=app.state.memory_service,
            emotional_memory_service=app.state.emotional_memory_service
        )
        logger.info("ProactiveEngagementEngine initialized successfully.")

        # Initialize Memory Consolidation Service (now that proactive engine exists)
        app.state.memory_consolidation_service = MemoryConsolidationService(
            memory_service=app.state.memory_service,
            autobiographical_system=app.state.autobiographical_memory_system,
            llm_service=app.state.llm_service,
            proactive_engine=app.state.proactive_engine
        )
        logger.info("MemoryConsolidationService initialized with proactive engagement.")

        # Initialize Self-Reflection & Discovery Engine (with proactive messaging)
        app.state.self_reflection_discovery_engine = SelfReflectionAndDiscoveryEngine(
            llm_service=app.state.llm_service,
            memory_service=app.state.memory_service,
            proactive_engine=app.state.proactive_engine
        )
        logger.info("SelfReflectionAndDiscoveryEngine initialized successfully.")

        # Initialize Specialized Agents
        app.state.perception_agent = PerceptionAgent(
            llm_service=app.state.llm_service,
            memory_service=app.state.memory_service,
        )
        logger.info("PerceptionAgent initialized successfully.")

        app.state.emotional_agent = EmotionalAgent(
            llm_service=app.state.llm_service,
            memory_service=app.state.memory_service,
            emotional_memory_service=app.state.emotional_memory_service
        )
        logger.info("EmotionalAgent initialized successfully with emotional intelligence.")

        app.state.memory_agent = MemoryAgent(
            llm_service=app.state.llm_service,
            memory_service=app.state.memory_service,
        )
        logger.info("MemoryAgent initialized successfully.")

        app.state.planning_agent = PlanningAgent(
            llm_service=app.state.llm_service,
            memory_service=app.state.memory_service,
        )
        logger.info("PlanningAgent initialized successfully.")

        app.state.creative_agent = CreativeAgent(
            llm_service=app.state.llm_service,
            memory_service=app.state.memory_service,
        )
        logger.info("CreativeAgent initialized successfully.")

        app.state.critic_agent = CriticAgent(
            llm_service=app.state.llm_service,
            memory_service=app.state.memory_service,
        )
        logger.info("CriticAgent initialized successfully.")

        app.state.web_browsing_service = WebBrowsingService(llm_service=app.state.llm_service)
        logger.info("WebBrowsingService initialized successfully.")

        # Initialize AudioInputProcessor (multimodal: speech-to-text via LLM)
        app.state.audio_input_processor = AudioInputProcessor(llm_service=app.state.llm_service)
        logger.info("AudioInputProcessor initialized successfully.")

        app.state.discovery_agent = DiscoveryAgent(
            llm_service=app.state.llm_service,
            memory_service=app.state.memory_service,
            web_browsing_service=app.state.web_browsing_service,
        )
        logger.info("DiscoveryAgent initialized successfully.")

        # Initialize Cognitive Brain with SelfModel, WorkingMemory, and TheoryOfMind
        app.state.cognitive_brain = CognitiveBrain(
            llm_service=app.state.llm_service,
            memory_service=app.state.memory_service,
            self_model_service=app.state.self_model_service,
            working_memory_buffer=app.state.working_memory_buffer,
            theory_of_mind_service=app.state.theory_of_mind_service
        )
        logger.info("CognitiveBrain initialized successfully with SelfModel, WorkingMemory, and TheoryOfMind.")

        # Initialize OrchestrationService (Central Agent) with all Brain Architecture services
        # Recreate ConflictMonitor with RL service injected for strategy learning
        app.state.conflict_monitor = ConflictMonitor(rl_service=app.state.rl_service)

        app.state.orchestration_service = OrchestrationService(
            perception_agent=app.state.perception_agent,
            emotional_agent=app.state.emotional_agent,
            memory_agent=app.state.memory_agent,
            planning_agent=app.state.planning_agent,
            creative_agent=app.state.creative_agent,
            critic_agent=app.state.critic_agent,
            discovery_agent=app.state.discovery_agent,
            web_browsing_service=app.state.web_browsing_service,
            audio_input_processor=app.state.audio_input_processor,
            cognitive_brain=app.state.cognitive_brain,
            memory_service=app.state.memory_service,
            background_task_queue=app.state.background_task_queue,
            self_reflection_discovery_engine=app.state.self_reflection_discovery_engine,
            working_memory_buffer=app.state.working_memory_buffer,
            thalamus_gateway=app.state.thalamus_gateway,
            attention_controller=app.state.attention_controller if settings.ATTENTION_CONTROLLER_ENABLED else None,
            conflict_monitor=app.state.conflict_monitor,
            contextual_memory_encoder=app.state.contextual_memory_encoder,
            rl_service=app.state.rl_service,
            emotional_memory_service=app.state.emotional_memory_service,
            meta_cognitive_monitor=app.state.meta_cognitive_monitor,
            procedural_learning_service=app.state.procedural_learning_service,
            metrics_service=app.state.metrics_service
        )
        logger.info("OrchestrationService (Central Agent) initialized successfully with Phase 1 & 2 Brain Architecture services.")

        # Wire OrchestrationService into BackgroundTaskQueue for task routing
        try:
            app.state.background_task_queue.set_orchestration_service(app.state.orchestration_service)
            logger.info("BackgroundTaskQueue wired to OrchestrationService for autonomous task routing.")
        except Exception:
            logger.warning("Failed to wire BackgroundTaskQueue to OrchestrationService.")

    except ConfigurationError as exc:
        logger.critical(f"Application failed to start due to configuration error: {exc.detail}")
        raise
    except Exception:
        logger.critical("An unexpected error occurred during service initialization", exc_info=True)
        raise

    # Application is ready to receive requests
    yield

    logger.info(f"Shutting down {settings.APP_NAME}...")
    # Perform any cleanup here if necessary
    if getattr(app.state, "memory_service", None):
        await app.state.memory_service.close()
    if getattr(app.state, "background_task_queue", None):
        await app.state.background_task_queue.shutdown()

    logger.info("Application shutdown complete.")

app = FastAPI(
    title=settings.APP_NAME,
    version="1.0.0",
    description="Multi-Agent Cognitive Architecture Backend API",
    lifespan=lifespan
)

# For development, allow all origins. For production, this should be more restrictive.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Centralized Error Handling ---

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Handles Pydantic validation errors, returning a 422 Unprocessable Entity.
    """
    logger.warning(f"Validation error for request to {request.url.path}: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors(), "message": "Validation Error"}
    )

@app.exception_handler(AgentServiceException)
async def agent_service_exception_handler(request: Request, exc: AgentServiceException):
    """
    Handles custom AgentServiceException, returning the specified status code and detail.
    """
    # Log the full detail internally, but provide a generic message to the client for security.
    logger.error(f"Agent Service Exception caught for request to {request.url.path} by {exc.agent_id}: {exc.detail}", extra={"status_code": exc.status_code, "agent_id": exc.agent_id})
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": "An internal agent error occurred. Please try again later.", "message": "Agent Error"}
    )

@app.exception_handler(APIException)
async def api_exception_handler(request: Request, exc: APIException):
    """
    Handles custom APIException, returning the specified status code and detail.
    """
    logger.error(f"API Exception caught for request to {request.url.path}: {exc.detail}", extra={"status_code": exc.status_code})
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": "An internal API error occurred. Please try again later.", "message": "API Error"}
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """
    Handles standard FastAPI HTTPException, returning the specified status code and detail.
    """
    logger.error(f"HTTP Exception caught for request to {request.url.path}: {exc.detail}", extra={"status_code": exc.status_code})
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "message": "HTTP Error"}
    )

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    """
    Handles all other unhandled exceptions, returning a generic 500 Internal Server Error.
    """
    logger.critical(f"Unhandled exception caught for request to {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected internal server error occurred.", "message": "Internal Server Error"}
    )

# --- API Endpoints ---

@app.get("/", response_class=Response, status_code=status.HTTP_200_OK)
async def root():
    """
    Root endpoint.
    """
    return Response(content="Welcome to the Multi-Agent Cognitive Architecture API!", media_type="text/plain")

@app.get("/health", response_model=Dict[str, str], status_code=status.HTTP_200_OK)
async def health_check():
    """
    Health check endpoint to verify service status.
    """
    logger.info("Health check requested.")
    return {"status": "healthy", "environment": settings.ENVIRONMENT, "app_name": settings.APP_NAME}

# Example of how LLM service might be used (for testing purposes for this sprint)
@app.post("/test-llm-text-generation", response_model=Dict[str, str], include_in_schema=False)
async def test_llm_text_generation(request: UserRequest, request_obj: Request):
    """
    Test endpoint to demonstrate LLM text generation.
    """
    llm_service: LLMIntegrationService = request_obj.app.state.llm_service
    if not llm_service:
        raise APIException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="LLM service not initialized.")

    try:
        generated_text = await llm_service.generate_text(prompt=request.input_text)
        logger.info(f"Generated text for user {request.user_id}: {generated_text[:50]}...")
        return {"user_id": str(request.user_id), "input_text": request.input_text, "generated_text": generated_text}
    except LLMServiceException as e:
        raise e
    except Exception as e:
        logger.error(f"Error in test_llm_text_generation: {e}", exc_info=True)
        raise APIException(detail="Failed to generate text via test endpoint.", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

@app.post("/test-llm-embedding-generation", response_model=Dict[str, Any], include_in_schema=False)
async def test_llm_embedding_generation(request: UserRequest, request_obj: Request):
    """
    Test endpoint to demonstrate LLM embedding generation.
    """
    llm_service: LLMIntegrationService = request_obj.app.state.llm_service
    if not llm_service:
        raise APIException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="LLM service not initialized.")

    try:
        embedding = await llm_service.generate_embedding(text=request.input_text)
        logger.info(f"Generated embedding for user {request.user_id}, vector length: {len(embedding)}")
        return {"user_id": str(request.user_id), "input_text": request.input_text, "embedding_length": len(embedding), "embedding_sample": embedding[:5]} # Return first 5 elements for brevity
    except LLMServiceException as e:
        raise e
    except Exception as e:
        logger.error(f"Error in test_llm_embedding_generation: {e}", exc_info=True)
        raise APIException(detail="Failed to generate embedding via test endpoint.", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

# Test endpoints for specialized agents (internal, for development verification)
@app.post("/test-agent/perception", response_model=AgentOutput, include_in_schema=False)
async def test_perception_agent(request: UserRequest, request_obj: Request):
    """
    Test endpoint for the Perception Agent.
    """
    perception_agent: PerceptionAgent = request_obj.app.state.perception_agent
    if not perception_agent:
        raise APIException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Perception Agent not initialized.")
    return await perception_agent.process_input(user_input=request.input_text)

@app.post("/test-agent/emotional", response_model=AgentOutput, include_in_schema=False)
async def test_emotional_agent(request: UserRequest, request_obj: Request):
    """
    Test endpoint for the Emotional Agent.
    """
    emotional_agent: EmotionalAgent = request_obj.app.state.emotional_agent
    if not emotional_agent:
        raise APIException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Emotional Agent not initialized.")
    return await emotional_agent.process_input(user_input=request.input_text)

@app.post("/test-agent/memory", response_model=AgentOutput, include_in_schema=False)
async def test_memory_agent(request: UserRequest, request_obj: Request):
    """
    Test endpoint for the Memory Agent.
    """
    memory_agent: MemoryAgent = request_obj.app.state.memory_agent
    if not memory_agent:
        raise APIException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Memory Agent not initialized.")
    return await memory_agent.process_input(user_input=request.input_text, user_id=request.user_id)

@app.post("/test-agent/critic", response_model=AgentOutput, include_in_schema=False)
async def test_critic_agent(request: UserRequest, request_obj: Request):
    """
    Test endpoint for the Critic Agent.
    """
    critic_agent: CriticAgent = request_obj.app.state.critic_agent
    if not critic_agent:
        raise APIException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Critic Agent not initialized.")
    return await critic_agent.process_input(user_input=request.input_text, other_agent_outputs=[])

@app.post("/test-agent/planning", response_model=AgentOutput, include_in_schema=False)
async def test_planning_agent(request: UserRequest, request_obj: Request):
    """
    Test endpoint for the Planning Agent.
    """
    planning_agent: PlanningAgent = request_obj.app.state.planning_agent
    if not planning_agent:
        raise APIException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Planning Agent not initialized.")
    return await planning_agent.process_input(user_input=request.input_text, other_agent_outputs=[])

@app.post("/test-agent/creative", response_model=AgentOutput, include_in_schema=False)
async def test_creative_agent(request: UserRequest, request_obj: Request):
    """
    Test endpoint for the Creative Agent.
    """
    creative_agent: CreativeAgent = request_obj.app.state.creative_agent
    if not creative_agent:
        raise APIException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Creative Agent not initialized.")
    return await creative_agent.process_input(user_input=request.input_text, other_agent_outputs=[])

@app.post("/test-agent/discovery", response_model=AgentOutput, include_in_schema=False)
async def test_discovery_agent(request: UserRequest, request_obj: Request):
    """
    Test endpoint for the Discovery Agent.
    """
    discovery_agent: DiscoveryAgent = request_obj.app.state.discovery_agent
    if not discovery_agent:
        raise APIException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Discovery Agent not initialized.")
    return await discovery_agent.process_input(user_input=request.input_text, other_agent_outputs=[])

# Internal API endpoint for Orchestration Service (used by /chat)
@app.post("/orchestrate_cycle", response_model=CognitiveCycle, status_code=status.HTTP_200_OK, include_in_schema=False)
async def orchestrate_cognitive_cycle_internal(user_request: UserRequest, request_obj: Request):
    """
    Internal API to initiate a full cognitive cycle with user input.
    Dispatches tasks to specialized agents, collects and synthesizes their outputs.
    """
    orchestration_service: OrchestrationService = request_obj.app.state.orchestration_service
    if not orchestration_service:
        raise APIException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Orchestration Service not initialized.")
    
    try:
        cognitive_cycle = await orchestration_service.orchestrate_cycle(user_request)
        return cognitive_cycle
    except APIException as e:
        raise e
    except Exception as e:
        logger.critical(f"Unhandled error during cognitive cycle orchestration for user {user_request.user_id}: {e}", exc_info=True)
        raise APIException(detail="Failed to orchestrate cognitive cycle due to an unexpected error.", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

# --- API Gateway Endpoints (External) ---

@app.post("/chat", response_model=Dict[str, str], status_code=status.HTTP_200_OK)
async def chat_endpoint(user_request: UserRequest, request_obj: Request, user_id: UUID = APIKeyAuth):
    """
    Receives user input and initiates a cognitive cycle for response generation.
    Returns the final natural language response.
    """
    # STORY-702: Filter harmful user inputs
    llm_service: LLMIntegrationService = request_obj.app.state.llm_service
    if not llm_service:
        raise APIException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="LLM service not initialized for content moderation.")

    moderation_result = await llm_service.moderate_content(
        text=user_request.input_text,
        image_base64=user_request.image_base64,
        audio_base64=user_request.audio_base64
    )
    if not moderation_result.get("is_safe"):
        logger.warning(f"User {user_id} input blocked due to safety concerns: {moderation_result.get('block_reason', 'N/A')}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Your input was flagged as potentially harmful and cannot be processed. Please try rephrasing your request."
        )

    orchestration_service: OrchestrationService = request_obj.app.state.orchestration_service
    if not orchestration_service:
        raise APIException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Orchestration Service not initialized.")
    
    try:
        # Override user_id from request with authenticated user_id
        user_request.user_id = user_id
        
        # Check if this is a response to a proactive message
        proactive_engine: ProactiveEngagementEngine = request_obj.app.state.proactive_engine
        responding_to_proactive_id = user_request.metadata.get("responding_to_proactive_message")
        
        if responding_to_proactive_id and proactive_engine:
            # User is responding to Bob's proactive message - record reaction
            try:
                from uuid import UUID as UUID_class
                msg_uuid = UUID_class(responding_to_proactive_id)
                await proactive_engine.record_user_reaction(
                    user_id=user_id,
                    message_id=msg_uuid,
                    user_response=user_request.input_text
                )
                logger.info(f"Recorded reaction to proactive message {responding_to_proactive_id} from user {user_id}")
            except Exception as e:
                logger.warning(f"Failed to record proactive reaction: {e}")
        
        cognitive_cycle = await orchestration_service.orchestrate_cycle(user_request)
        if cognitive_cycle.final_response:
            return {"user_id": str(user_request.user_id), "session_id": str(user_request.session_id), "response": cognitive_cycle.final_response}
        else:
            raise APIException(detail="No final response generated for the cognitive cycle.", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)
    except APIException as e:
        raise e
    except Exception as e:
        logger.critical(f"Unhandled error in /chat endpoint for user {user_request.user_id}: {e}", exc_info=True)
        raise APIException(detail="Failed to process chat request due to an unexpected error.", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

@app.get("/chat/proactive", status_code=status.HTTP_200_OK)
async def get_proactive_message(request_obj: Request, user_id: UUID = APIKeyAuth):
    """
    Check if Bob has a proactive message queued for the user.
    Returns the message if conditions are right, otherwise returns None.
    
    Frontend should call this on session start / periodic polling.
    """
    proactive_engine: ProactiveEngagementEngine = request_obj.app.state.proactive_engine
    if not proactive_engine:
        raise APIException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Proactive Engagement Engine not initialized.")
    
    try:
        message = await proactive_engine.get_queued_message(user_id)
        if message:
            # Mark as delivered
            await proactive_engine.mark_delivered(user_id, message.message_id)
            
            return {
                "has_message": True,
                "message_id": str(message.message_id),
                "message": message.message_content,
                "trigger_type": message.trigger_type,
                "priority": message.priority,
                "created_at": message.created_at.isoformat()
            }
        else:
            return {"has_message": False}
    except Exception as e:
        logger.error(f"Error fetching proactive message for user {user_id}: {e}", exc_info=True)
        raise APIException(detail="Failed to fetch proactive message.", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

@app.post("/chat/proactive/reaction", status_code=status.HTTP_200_OK)
async def record_proactive_reaction(
    request_obj: Request,
    message_id: str = Query(..., description="ID of the proactive message being reacted to"),
    user_response: str = Query(..., description="User's response to the proactive message"),
    user_id: UUID = APIKeyAuth
):
    """
    Record user's reaction to a proactive message.
    Bob learns from this feedback and adjusts future behavior.
    
    If user says "you're annoying", Bob will:
    - Feel hurt (reduce trust slightly)
    - Back off (increase cooldown period)
    - Eventually disable proactive messages if repeatedly negative
    """
    proactive_engine: ProactiveEngagementEngine = request_obj.app.state.proactive_engine
    if not proactive_engine:
        raise APIException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Proactive Engagement Engine not initialized.")
    
    try:
        from uuid import UUID as UUID_class
        message_uuid = UUID_class(message_id)
        
        await proactive_engine.record_user_reaction(
            user_id=user_id,
            message_id=message_uuid,
            user_response=user_response
        )
        
        return {
            "message": "Reaction recorded successfully. Bob will adjust his behavior accordingly.",
            "learned": True
        }
    except Exception as e:
        logger.error(f"Error recording proactive reaction for user {user_id}: {e}", exc_info=True)
        raise APIException(detail="Failed to record proactive reaction.", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

@app.post("/chat/proactive/test", status_code=status.HTTP_200_OK)
async def test_proactive_message(
    request_obj: Request,
    trigger_type: str = Query(..., description="Type of trigger to test (e.g., 'discovery', 'self-reflection', 'boredom')"),
    message_content: str = Query(..., description="Custom message content for testing"),
    user_id: UUID = APIKeyAuth
):
    """
    Test endpoint to manually trigger a proactive message for development/testing purposes.
    This allows developers to test the proactive messaging system without waiting for natural triggers.
    """
    proactive_engine: ProactiveEngagementEngine = request_obj.app.state.proactive_engine
    if not proactive_engine:
        raise APIException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Proactive Engagement Engine not initialized.")
    
    try:
        # Create a test proactive message
        from datetime import datetime
        from uuid import uuid4
        
        # Convert string trigger_type to enum
        from src.services.proactive_engagement_service import ProactiveMessageTrigger
        try:
            trigger_enum = ProactiveMessageTrigger(trigger_type)
        except ValueError:
            # Default to discovery_insight if invalid trigger type
            trigger_enum = ProactiveMessageTrigger.DISCOVERY_INSIGHT
        
        test_message = ProactiveMessage(
            message_id=uuid4(),
            user_id=user_id,
            message_content=message_content,
            trigger_type=trigger_enum,
            trigger_context={"test": True, "manual_trigger": True},
            created_at=datetime.utcnow(),
            priority=1,
            delivered=False,
            user_reaction=None
        )
        
        # Queue the test message
        await proactive_engine.queue_message(test_message)
        
        return {
            "message": f"Test proactive message queued successfully with trigger type '{trigger_type}'",
            "message_id": str(test_message.message_id),
            "trigger_type": trigger_type,
            "content": message_content,
            "queued_at": test_message.created_at.isoformat()
        }
    except Exception as e:
        logger.error(f"Error creating test proactive message for user {user_id}: {e}", exc_info=True)
        raise APIException(detail="Failed to create test proactive message.", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

@app.post("/memory/query", response_model=MemoryQueryResponse, status_code=status.HTTP_200_OK)
async def query_user_memory(query_request: MemoryQueryRequest, request_obj: Request, user_id: UUID = APIKeyAuth):
    """
    Allows querying the persistent memory system for relevant past cognitive cycles.
    """
    memory_service: MemoryService = request_obj.app.state.memory_service
    if not memory_service:
        raise APIException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Memory Service not initialized.")
    
    try:
        # Enforce user isolation by overriding user_id from request with authenticated user_id
        query_request.user_id = user_id
        retrieved_cycles = await memory_service.query_memory(query_request)
        return MemoryQueryResponse(
            user_id=user_id,
            query_text=query_request.query_text,
            retrieved_cycles=retrieved_cycles,
            message=f"Successfully retrieved {len(retrieved_cycles)} relevant memory cycles."
        )
    except APIException as e:
        raise e
    except Exception as e:
        logger.critical(f"Unhandled error in /memory/query endpoint for user {user_id}: {e}", exc_info=True)
        raise APIException(detail="Failed to query memory due to an unexpected error.", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

@app.post("/reflect", response_model=Dict[str, str], status_code=status.HTTP_202_ACCEPTED)
async def trigger_reflection_endpoint(reflection_request: ReflectionTriggerRequest, request_obj: Request, user_id: UUID = APIKeyAuth):
    """
    Triggers self-reflection on a specified number of past cognitive cycles.
    This endpoint enqueues a background task for processing.
    """
    orchestration_service: OrchestrationService = request_obj.app.state.orchestration_service
    if not orchestration_service:
        raise APIException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Orchestration Service not initialized.")
    
    try:
        # Enforce user isolation by overriding user_id from request with authenticated user_id
        reflection_request.user_id = user_id
        await orchestration_service.trigger_reflection(
            user_id=reflection_request.user_id,
            num_cycles=reflection_request.num_cycles,
            trigger_type=reflection_request.trigger_type
        )
        return {"status": "accepted", "message": f"Reflection for {reflection_request.num_cycles} cycles for user {user_id} enqueued for background processing."}
    except APIException as e:
        raise e
    except Exception as e:
        logger.critical(f"Unhandled error in /reflect endpoint for user {user_id}: {e}", exc_info=True)
        raise APIException(detail="Failed to trigger reflection due to an unexpected error.", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

@app.post("/discover", response_model=Dict[str, str], status_code=status.HTTP_202_ACCEPTED)
async def trigger_discovery_endpoint(discovery_request: DiscoveryTriggerRequest, request_obj: Request, user_id: UUID = APIKeyAuth):
    """
    Initiates an autonomous discovery process (memory analysis, curiosity exploration, or self-assessment).
    This endpoint enqueues a background task for processing.
    """
    orchestration_service: OrchestrationService = request_obj.app.state.orchestration_service
    if not orchestration_service:
        raise APIException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Orchestration Service not initialized.")
    
    try:
        # Enforce user isolation by overriding user_id from request with authenticated user_id
        discovery_request.user_id = user_id
        await orchestration_service.trigger_discovery(
            user_id=discovery_request.user_id,
            discovery_type=discovery_request.discovery_type,
            context=discovery_request.context
        )
        return {"status": "accepted", "message": f"Discovery of type '{discovery_request.discovery_type}' for user {user_id} enqueued for background processing."}
    except APIException as e:
        raise e
    except Exception as e:
        logger.critical(f"Unhandled error in /discover endpoint for user {user_id}: {e}", exc_info=True)
        raise APIException(detail="Failed to trigger discovery due to an unexpected error.", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

@app.get("/patterns", response_model=PatternsResponse, status_code=status.HTTP_200_OK)
async def get_patterns_endpoint(request_obj: Request, user_id: UUID = Query(..., description="The ID of the user whose patterns are being queried."), auth_user_id: UUID = APIKeyAuth):
    """
    Retrieves meta-learnings and discovered patterns for a specific user.
    This endpoint queries the patterns collection and returns any found patterns.
    """
    # Enforce user isolation: ensure the requested user_id matches the authenticated user_id
    if user_id != auth_user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cannot access patterns for another user.")

    memory_service: MemoryService = request_obj.app.state.memory_service
    if not memory_service:
        raise APIException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Memory Service not initialized.")
    
    try:
        patterns = await memory_service.get_patterns_for_user(user_id)
        return PatternsResponse(
            user_id=user_id,
            patterns=patterns,
            message=f"Successfully retrieved {len(patterns)} patterns for user {user_id}."
        )
    except APIException as e:
        raise e
    except Exception as e:
        logger.critical(f"Unhandled error in /patterns endpoint for user {user_id}: {e}", exc_info=True)
        raise APIException(detail="Failed to retrieve patterns due to an unexpected error.", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

@app.delete("/user/data/{user_id}", response_model=Dict[str, str], status_code=status.HTTP_200_OK)
async def delete_user_data_endpoint(request_obj: Request, user_id: UUID = Path(..., description="The ID of the user whose data is to be deleted."), auth_user_id: UUID = APIKeyAuth):
    """
    Deletes all data associated with a specific user from the system.
    """
    # Enforce user isolation: ensure the requested user_id matches the authenticated user_id
    if user_id != auth_user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cannot delete data for another user.")

    memory_service: MemoryService = request_obj.app.state.memory_service
    if not memory_service:
        raise APIException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Memory Service not initialized.")
    
    try:
        deleted = await memory_service.delete_user_data(user_id)
        if deleted:
            return {"status": "success", "message": f"All data for user {user_id} deleted successfully."}
        else:
            return {"status": "info", "message": f"No data found for user {user_id} to delete."}
    except APIException as e:
        raise e
    except Exception as e:
        logger.critical(f"Unhandled error in /user/data/{user_id} endpoint: {e}", exc_info=True)
        raise APIException(detail="Failed to delete user data due to an unexpected error.", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

@app.get("/memory/cycles", response_model=CycleListResponse, status_code=status.HTTP_200_OK)
async def list_user_cycles_endpoint(
    request_obj: Request, # CQ-004 Fix: Removed '= None'
    user_id: UUID = Query(..., description="The ID of the user whose cycles are being queried."),
    skip: int = Query(0, ge=0, description="Number of cycles to skip for pagination."),
    limit: int = Query(10, ge=1, le=100, description="Maximum number of cycles to retrieve."),
    session_id: Optional[UUID] = Query(None, description="Optional session ID to filter cycles."),
    start_date: Optional[datetime] = Query(None, description="Optional start date to filter cycles (ISO 8601 format)."),
    end_date: Optional[datetime] = Query(None, description="Optional end date to filter cycles (ISO 8601 format)."),
    response_type: Optional[str] = Query(None, description="Optional response type to filter cycles (e.g., 'informational')."),
    min_confidence: Optional[float] = Query(None, ge=0.0, le=1.0, description="Minimum confidence score of any agent output in the cycle."),
    auth_user_id: UUID = APIKeyAuth
):
    """
    Lists all cognitive cycles for a specific user with optional filtering and pagination.
    """
    # Enforce user isolation: ensure the requested user_id matches the authenticated user_id
    if user_id != auth_user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cannot access cycles for another user.")

    memory_service: MemoryService = request_obj.app.state.memory_service
    if not memory_service:
        raise APIException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Memory Service not initialized.")
    
    try:
        list_request = CycleListRequest(
            user_id=user_id,
            skip=skip,
            limit=limit,
            session_id=session_id,
            start_date=start_date,
            end_date=end_date,
            response_type=response_type,
            min_confidence=min_confidence
        )
        cycles, total_cycles = await memory_service.list_cycles(list_request)
        return CycleListResponse(
            user_id=user_id,
            total_cycles=total_cycles,
            cycles=cycles,
            message=f"Successfully retrieved {len(cycles)} of {total_cycles} cognitive cycles for user {user_id}."
        )
    except APIException as e:
        raise e
    except Exception as e:
        logger.critical(f"Unhandled error in /memory/cycles endpoint for user {user_id}: {e}", exc_info=True)
        raise APIException(detail="Failed to list cognitive cycles due to an unexpected error.", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

@app.get("/memory/{cycle_id}", response_model=CognitiveCycle, status_code=status.HTTP_200_OK)
async def get_cognitive_cycle_by_id_endpoint(
    request_obj: Request, # CQ-004 Fix: Removed '= None'
    user_id: UUID = Query(..., description="The ID of the user who owns the cycle."),
    cycle_id: UUID = Path(..., description="The ID of the cognitive cycle to retrieve."),
    auth_user_id: UUID = APIKeyAuth
):
    """
    Retrieves a specific cognitive cycle by its ID for a given user.
    """
    # Enforce user isolation: ensure the requested user_id matches the authenticated user_id
    if user_id != auth_user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cannot access cycles for another user.")

    memory_service: MemoryService = request_obj.app.state.memory_service
    if not memory_service:
        raise APIException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Memory Service not initialized.")
    
    try:
        cycle = await memory_service.get_cycle_by_id(user_id, cycle_id)
        if cycle:
            return cycle
        else:
            raise APIException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Cognitive cycle {cycle_id} not found for user {user_id}.")
    except APIException as e:
        raise e
    except Exception as e:
        logger.critical(f"Unhandled error in /memory/{cycle_id} endpoint for user {user_id}: {e}", exc_info=True)
        raise APIException(detail="Failed to retrieve cognitive cycle due to an unexpected error.", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

@app.patch("/memory/{cycle_id}/metadata", response_model=Dict[str, str], status_code=status.HTTP_200_OK)
async def update_cognitive_cycle_metadata_endpoint(
    cycle_update_request: CycleUpdateRequest,
    request_obj: Request, # CQ-004 Fix: Removed '= None'
    user_id: UUID = Query(..., description="The ID of the user who owns the cycle."),
    cycle_id: UUID = Path(..., description="The ID of the cognitive cycle to update."),
    auth_user_id: UUID = APIKeyAuth
):
    """
    Updates the metadata of a specific cognitive cycle for a given user.
    """
    # Enforce user isolation: ensure the requested user_id matches the authenticated user_id
    if user_id != auth_user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cannot update metadata for another user's cycle.")

    memory_service: MemoryService = request_obj.app.state.memory_service
    if not memory_service:
        raise APIException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Memory Service not initialized.")
    
    try:
        updated = await memory_service.update_cycle_metadata(user_id, cycle_id, cycle_update_request.metadata)
        if updated:
            return {"status": "success", "message": f"Metadata for cognitive cycle {cycle_id} updated successfully."}
        else:
            # Check if cycle exists but no modification was made (e.g., same data)
            existing_cycle = await memory_service.get_cycle_by_id(user_id, cycle_id)
            if existing_cycle:
                return {"status": "info", "message": f"Cognitive cycle {cycle_id} found, but no new metadata changes were applied."}
            else:
                raise APIException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Cognitive cycle {cycle_id} not found for user {user_id}.")
    except APIException as e:
        raise e
    except Exception as e:
        logger.critical(f"Unhandled error in /memory/{cycle_id}/metadata endpoint for user {user_id}: {e}", exc_info=True)
        raise APIException(detail="Failed to update cognitive cycle metadata due to an unexpected error.", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

@app.get("/theory-of-mind/stats", response_model=Dict[str, Any], status_code=status.HTTP_200_OK)
async def get_theory_of_mind_stats_endpoint(
    request_obj: Request,
    user_id: UUID = Query(..., description="The ID of the user to get theory of mind statistics for."),
    auth_user_id: UUID = APIKeyAuth
):
    """
    Gets theory of mind prediction accuracy statistics for a user.
    
    Returns validation statistics including total predictions, accuracy rate, and pending validations.
    """
    # Enforce user isolation
    if user_id != auth_user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cannot access another user's theory of mind statistics.")
    
    theory_of_mind_service = request_obj.app.state.theory_of_mind_service
    if not theory_of_mind_service:
        raise APIException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Theory of Mind service not initialized.")
    
    try:
        stats = theory_of_mind_service.get_validation_stats(str(user_id))
        return {
            "user_id": str(user_id),
            "statistics": stats,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get theory of mind statistics for user {user_id}: {e}", exc_info=True)
        raise APIException(detail="Failed to retrieve theory of mind statistics.", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

@app.get("/theory-of-mind/current-state", response_model=Dict[str, Any], status_code=status.HTTP_200_OK)
async def get_current_mental_state_endpoint(
    request_obj: Request,
    user_id: UUID = Query(..., description="The ID of the user to get current mental state for."),
    auth_user_id: UUID = APIKeyAuth
):
    """
    Gets the current mental state model for a user (most recent inference).
    
    Returns the user's inferred goals, emotional state, needs, and conversation intent.
    """
    # Enforce user isolation
    if user_id != auth_user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Cannot access another user's mental state.")
    
    theory_of_mind_service = request_obj.app.state.theory_of_mind_service
    if not theory_of_mind_service:
        raise APIException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Theory of Mind service not initialized.")
    
    try:
        # Get cached mental state
        mental_state = theory_of_mind_service.mental_state_cache.get(str(user_id))
        
        if not mental_state:
            return {
                "user_id": str(user_id),
                "mental_state": None,
                "message": "No mental state model available yet. Interact with the system to build a model.",
                "timestamp": datetime.utcnow().isoformat()
            }
        
        return {
            "user_id": str(user_id),
            "mental_state": {
                "current_goal": mental_state.current_goal,
                "current_emotion": mental_state.current_emotion,
                "current_needs": mental_state.current_needs,
                "knows_about": mental_state.knows_about,
                "confused_about": mental_state.confused_about,
                "interested_in": mental_state.interested_in,
                "likely_next_action": mental_state.likely_next_action,
                "conversation_intent": mental_state.conversation_intent,
                "confidence": mental_state.confidence,
                "uncertainty_factors": mental_state.uncertainty_factors,
                "inferred_at": mental_state.timestamp.isoformat()
            },
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        logger.error(f"Failed to get current mental state for user {user_id}: {e}", exc_info=True)
        raise APIException(detail="Failed to retrieve current mental state.", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)


@app.get("/agents", response_model=List[Dict[str, str]], status_code=status.HTTP_200_OK)
async def list_agents_endpoint(request_obj: Request, user_id: UUID = APIKeyAuth):
    """
    Retrieves a list of all available agent IDs and their descriptions.
    """
    # Authentication is applied, but user_id is not directly used in this endpoint's logic
    agents_info = [
        {"agent_id": request_obj.app.state.perception_agent.AGENT_ID, "description": "Analyzes user input for topics, patterns, and context type."},
        {"agent_id": request_obj.app.state.emotional_agent.AGENT_ID, "description": "Detects emotional tone, sentiment, and interpersonal dynamics."},
        {"agent_id": request_obj.app.state.memory_agent.AGENT_ID, "description": "Queries persistent memory for relevant past experiences and context."},
        {"agent_id": request_obj.app.state.planning_agent.AGENT_ID, "description": "Evaluates response options, feasibility, and strategic considerations."},
        {"agent_id": request_obj.app.state.creative_agent.AGENT_ID, "description": "Generates novel perspectives, analogies, and reframings."},
        {"agent_id": request_obj.app.state.critic_agent.AGENT_ID, "description": "Checks for logic, contradictions, and coherence in inputs and outputs."},
        {"agent_id": request_obj.app.state.discovery_agent.AGENT_ID, "description": "Identifies knowledge gaps, generates curiosities, and proposes explorations."},
        {"agent_id": request_obj.app.state.web_browsing_service.SERVICE_ID, "description": "A sandboxed backend service that enables autonomous web exploration."}
    ]
    logger.info("Listed all available agents.")
    return agents_info

@app.get("/api/dashboard/metrics", response_model=Dict[str, Any], status_code=status.HTTP_200_OK)
async def get_dashboard_metrics_endpoint(request_obj: Request):
    """
    Retrieves current dashboard metrics data for real-time monitoring and scientific validation.
    """
    metrics_service: MetricsService = request_obj.app.state.metrics_service
    if not metrics_service:
        raise APIException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Metrics Service not initialized.")

    try:
        dashboard_data = await metrics_service.get_dashboard_data()
        logger.info("Retrieved dashboard metrics data.")
        return dashboard_data
    except Exception as e:
        logger.error(f"Failed to retrieve dashboard metrics: {e}", exc_info=True)
        raise APIException(detail="Failed to retrieve dashboard metrics.", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

@app.get("/api/dashboard/history", response_model=Dict[str, Any], status_code=status.HTTP_200_OK)
async def get_dashboard_history_endpoint(
    request_obj: Request,
    hours: int = Query(24, ge=1, le=168, description="Number of hours of historical data to retrieve (1-168)."),
    metric_types: Optional[List[str]] = Query(None, description="Optional list of metric types to filter by.")
):
    """
    Retrieves historical dashboard metrics data for trend analysis and scientific validation.
    """
    metrics_service: MetricsService = request_obj.app.state.metrics_service
    if not metrics_service:
        raise APIException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Metrics Service not initialized.")

    try:
        # Convert metric_types strings to MetricType enum if provided
        filter_types = None
        if metric_types:
            try:
                filter_types = [MetricType(mt) for mt in metric_types]
            except ValueError as e:
                raise APIException(detail=f"Invalid metric type: {e}", status_code=status.HTTP_400_BAD_REQUEST)

        history_data = await metrics_service.get_historical_data(hours=hours, metric_types=filter_types)
        logger.info(f"Retrieved {hours} hours of historical dashboard data.")
        return history_data
    except APIException as e:
        raise e
    except Exception as e:
        logger.error(f"Failed to retrieve dashboard history: {e}", exc_info=True)
        raise APIException(detail="Failed to retrieve dashboard history.", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

@app.get("/api/dashboard/correlations", response_model=Dict[str, Any], status_code=status.HTTP_200_OK)
async def get_dashboard_correlations(
    request_obj: Request,
    hours: int = Query(24, ge=1, le=168, description="Number of hours of data to analyze for correlations (1-168)."),
    user_id: Optional[str] = Query(None, description="Optional user ID to filter correlations by.")
):
    """
    Retrieves correlation analysis between different metrics for scientific insights.
    """
    metrics_service: MetricsService = request_obj.app.state.metrics_service
    if not metrics_service:
        raise APIException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Metrics Service not initialized.")

    try:
        correlation_data = await metrics_service.get_correlation_analysis(hours=hours, user_id=user_id)
        logger.info(f"Retrieved correlation analysis for {hours} hours of data.")
        return correlation_data
    except APIException as e:
        raise e
    except Exception as e:
        logger.error(f"Failed to retrieve dashboard correlations: {e}", exc_info=True)
        raise APIException(detail="Failed to retrieve dashboard correlations.", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

@app.websocket("/ws/dashboard")
async def dashboard_websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time dashboard updates and live metrics streaming.
    """
    metrics_service: MetricsService = websocket.app.state.metrics_service
    if not metrics_service:
        await websocket.close(code=1011)  # Internal server error
        return

    await websocket.accept()
    logger.info("Dashboard WebSocket connection established.")

    try:
        # Send initial data
        initial_data = await metrics_service.get_dashboard_data()
        await websocket.send_json({"type": "initial", "data": initial_data})

        # Keep connection alive for future real-time updates
        # TODO: Implement real-time metrics subscription when needed
        while True:
            # Wait for client messages or keep-alive
            try:
                message = await websocket.receive_text()
                logger.debug(f"Received WebSocket message: {message}")
            except Exception:
                # Client disconnected
                break

    except Exception as e:
        logger.error(f"Dashboard WebSocket error: {e}", exc_info=True)
    finally:
        logger.info("Dashboard WebSocket connection closed.")

# ===== STATISTICAL ANALYSIS ENDPOINTS =====

@app.get("/api/dashboard/analysis/statistical", response_model=Dict[str, Any], status_code=status.HTTP_200_OK)
async def get_statistical_analysis_endpoint(
    request_obj: Request,
    metric_series: str = Query(..., description="Comma-separated list of metric names to analyze"),
    analysis_type: str = Query("comprehensive", description="Type of analysis: 'comprehensive', 'trend', 'distribution'")
):
    """
    Performs statistical analysis on specified metric series for scientific validation.
    """
    metrics_service: MetricsService = request_obj.app.state.metrics_service
    if not metrics_service:
        raise APIException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Metrics Service not initialized.")

    try:
        # Parse metric series
        metric_names = [name.strip() for name in metric_series.split(",")]

        results = {}

        # Debug: Check metrics service state
        logger.info(f"Metrics service events count: {len(getattr(metrics_service, 'events', []))}")
        logger.info(f"Metrics service events type: {type(getattr(metrics_service, 'events', None))}")

        for metric_name in metric_names:
            # Get data for this metric (simplified - in practice would need more sophisticated data retrieval)
            if metric_name == "user_satisfaction":
                # Extract satisfaction scores from recent events
                satisfaction_data = []
                recent_events = list(metrics_service.events)[-1000:]  # Convert to list and get last 1000
                for event in recent_events:
                    if event.type == MetricType.COGNITIVE_CYCLE:
                        satisfaction = event.data.get("user_satisfaction")
                        if satisfaction is not None:
                            satisfaction_data.append(satisfaction)

                if satisfaction_data:
                    results[metric_name] = metrics_service.perform_statistical_analysis(satisfaction_data)

            elif metric_name == "processing_time":
                processing_times = []
                recent_events = list(metrics_service.events)[-1000:]  # Convert to list and get last 1000
                for event in recent_events:
                    if event.type == MetricType.COGNITIVE_CYCLE and "processing_time" in event.data:
                        processing_times.append(event.data["processing_time"])

                if processing_times:
                    results[metric_name] = metrics_service.perform_statistical_analysis(processing_times)

            elif metric_name == "learning_performance":
                # Analyze learning curves
                learning_analysis = metrics_service.analyze_learning_curves(metrics_service.skill_performance)
                results[metric_name] = learning_analysis

        logger.info(f"Performed statistical analysis on {len(metric_names)} metric series.")
        return {
            "analysis_type": analysis_type,
            "metrics_analyzed": metric_names,
            "results": results,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to perform statistical analysis: {e}", exc_info=True)
        raise APIException(detail="Failed to perform statistical analysis.", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

@app.get("/api/dashboard/analysis/compare", response_model=Dict[str, Any], status_code=status.HTTP_200_OK)
async def compare_metrics_endpoint(
    request_obj: Request,
    group1_metric: str = Query(..., description="First metric series name"),
    group2_metric: str = Query(..., description="Second metric series name"),
    test_type: str = Query("auto", description="Statistical test type: 'auto', 't-test', 'mann-whitney'")
):
    """
    Performs statistical comparison between two metric groups.
    """
    metrics_service: MetricsService = request_obj.app.state.metrics_service
    if not metrics_service:
        raise APIException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Metrics Service not initialized.")

    try:
        # Extract data for comparison (simplified implementation)
        group1_data = []
        group2_data = []

        # Get satisfaction scores for comparison
        for event in metrics_service.events[-1000:]:
            if event.type == MetricType.COGNITIVE_CYCLE:
                satisfaction = event.data.get("user_satisfaction")
                if satisfaction is not None:
                    # Alternate assignment for demonstration
                    if len(group1_data) <= len(group2_data):
                        group1_data.append(satisfaction)
                    else:
                        group2_data.append(satisfaction)

        if len(group1_data) >= 3 and len(group2_data) >= 3:
            comparison_result = metrics_service.compare_groups_statistical_test(
                group1_data, group2_data, test_type
            )
        else:
            comparison_result = {"insufficient_data": True}

        logger.info(f"Performed statistical comparison between {group1_metric} and {group2_metric}.")
        return {
            "comparison": {
                "group1": group1_metric,
                "group2": group2_metric,
                "test_type": test_type
            },
            "result": comparison_result,
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to perform metric comparison: {e}", exc_info=True)
        raise APIException(detail="Failed to perform metric comparison.", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

@app.get("/api/dashboard/analysis/learning-curves", response_model=Dict[str, Any], status_code=status.HTTP_200_OK)
async def analyze_learning_curves_endpoint(request_obj: Request):
    """
    Analyzes learning curves using power-law fitting for scientific validation.
    """
    metrics_service: MetricsService = request_obj.app.state.metrics_service
    if not metrics_service:
        raise APIException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Metrics Service not initialized.")

    try:
        learning_analysis = metrics_service.analyze_learning_curves(metrics_service.skill_performance)

        logger.info("Performed learning curve analysis with power-law fitting.")
        return {
            "analysis_type": "power_law_learning_curves",
            "skills_analyzed": list(learning_analysis.keys()),
            "results": learning_analysis,
            "methodology": {
                "fitting_method": "log-linear regression for power-law parameters",
                "goodness_of_fit": "coefficient of determination (R²)",
                "learning_rate": "absolute value of power-law exponent"
            },
            "timestamp": datetime.utcnow().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to analyze learning curves: {e}", exc_info=True)
        raise APIException(detail="Failed to analyze learning curves.", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

# ===== RESEARCH EXPORT ENDPOINTS =====

@app.get("/api/dashboard/export/csv", response_class=PlainTextResponse, status_code=status.HTTP_200_OK)
async def export_dashboard_csv_endpoint(
    request_obj: Request,
    data_type: str = Query("dashboard", description="Type of data to export: 'dashboard', 'historical', 'scientific'")
):
    """
    Exports dashboard data in CSV format for research and analysis.
    """
    metrics_service: MetricsService = request_obj.app.state.metrics_service
    if not metrics_service:
        raise APIException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Metrics Service not initialized.")

    try:
        if data_type == "dashboard":
            data = await metrics_service.get_dashboard_data()
        elif data_type == "historical":
            data = await metrics_service.get_historical_data(hours=24)
        elif data_type == "scientific":
            start_date = datetime.utcnow() - timedelta(days=7)
            end_date = datetime.utcnow()
            data = await metrics_service.export_scientific_data(start_date, end_date)
        else:
            raise APIException(detail="Invalid data_type. Must be 'dashboard', 'historical', or 'scientific'.",
                             status_code=status.HTTP_400_BAD_REQUEST)

        csv_content = metrics_service.export_to_csv(data, f"eca_{data_type}")

        # Set response headers for file download
        response = PlainTextResponse(
            content=csv_content,
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=eca_{data_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
            }
        )

        logger.info(f"Exported {data_type} data to CSV format.")
        return response

    except APIException as e:
        raise e
    except Exception as e:
        logger.error(f"Failed to export CSV data: {e}", exc_info=True)
        raise APIException(detail="Failed to export CSV data.", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

@app.get("/api/dashboard/export/json", response_model=Dict[str, Any], status_code=status.HTTP_200_OK)
async def export_dashboard_json_endpoint(
    request_obj: Request,
    data_type: str = Query("dashboard", description="Type of data to export: 'dashboard', 'historical', 'scientific'"),
    include_metadata: bool = Query(True, description="Whether to include export metadata")
):
    """
    Exports dashboard data in JSON format for research and analysis.
    """
    metrics_service: MetricsService = request_obj.app.state.metrics_service
    if not metrics_service:
        raise APIException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Metrics Service not initialized.")

    try:
        if data_type == "dashboard":
            data = await metrics_service.get_dashboard_data()
        elif data_type == "historical":
            data = await metrics_service.get_historical_data(hours=24)
        elif data_type == "scientific":
            start_date = datetime.utcnow() - timedelta(days=7)
            end_date = datetime.utcnow()
            data = await metrics_service.export_scientific_data(start_date, end_date)
        else:
            raise APIException(detail="Invalid data_type. Must be 'dashboard', 'historical', or 'scientific'.",
                             status_code=status.HTTP_400_BAD_REQUEST)

        json_content = metrics_service.export_to_json(data, include_metadata)

        logger.info(f"Exported {data_type} data to JSON format.")
        return {
            "export_type": "json",
            "data_type": data_type,
            "content": json_content,
            "filename": f"eca_{data_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json",
            "timestamp": datetime.utcnow().isoformat()
        }

    except APIException as e:
        raise e
    except Exception as e:
        logger.error(f"Failed to export JSON data: {e}", exc_info=True)
        raise APIException(detail="Failed to export JSON data.", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

@app.get("/api/dashboard/export/report", response_model=Dict[str, Any], status_code=status.HTTP_200_OK)
async def generate_research_report_endpoint(
    request_obj: Request,
    analysis_period_days: int = Query(30, ge=1, le=365, description="Number of days to analyze (1-365)")
):
    """
    Generates a comprehensive research report for scientific publication.
    """
    metrics_service: MetricsService = request_obj.app.state.metrics_service
    if not metrics_service:
        raise APIException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="Metrics Service not initialized.")

    try:
        report = metrics_service.generate_research_report(analysis_period_days)

        logger.info(f"Generated research report for {analysis_period_days} days of analysis.")
        return report

    except Exception as e:
        logger.error(f"Failed to generate research report: {e}", exc_info=True)
        raise APIException(detail="Failed to generate research report.", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

@app.get("/health/deep", response_model=Dict[str, Any], status_code=status.HTTP_200_OK)
async def deep_health_check_endpoint(request_obj: Request):
    """
    Performs a comprehensive system-wide health check including all major components.
    """
    health_status = {
        "status": "healthy",
        "components": {},
        "timestamp": datetime.utcnow().isoformat()
    }

    # Check LLM Service
    try:
        llm_service: LLMIntegrationService = request_obj.app.state.llm_service
        if llm_service:
            # Perform a lightweight text generation to test LLM connectivity
            await llm_service.generate_text(prompt="ping", temperature=0.0, max_output_tokens=5, stop_sequences=["ping"], safety_settings=[])
            health_status["components"]["llm_service"] = {"status": "healthy", "message": "LLM text generation successful."}
        else:
            health_status["components"]["llm_service"] = {"status": "unhealthy", "message": "LLM service not initialized."}
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["components"]["llm_service"] = {"status": "unhealthy", "message": f"LLM service check failed: {e}"}
        health_status["status"] = "degraded"
        logger.error(f"Deep health check: LLM service failed: {e}", exc_info=True)

    # Check Memory Service (ChromaDB connection)
    try:
        memory_service: MemoryService = request_obj.app.state.memory_service
        if memory_service and memory_service.client:
            # ChromaDB doesn't have a ping, so we check if the client is there
            health_status["components"]["memory_service"] = {"status": "healthy", "message": "ChromaDB client initialized."}
        else:
            health_status["components"]["memory_service"] = {"status": "unhealthy", "message": "Memory service not initialized or ChromaDB client not connected."}
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["components"]["memory_service"] = {"status": "unhealthy", "message": f"Memory service check failed: {e}"}
        health_status["status"] = "degraded"
        logger.error(f"Deep health check: Memory service failed: {e}", exc_info=True)

    # Check if all agents are initialized
    agents_to_check = [
        ("perception_agent", request_obj.app.state.perception_agent),
        ("emotional_agent", request_obj.app.state.emotional_agent),
        ("memory_agent", request_obj.app.state.memory_agent),
        ("planning_agent", request_obj.app.state.planning_agent),
        ("creative_agent", request_obj.app.state.creative_agent),
        ("critic_agent", request_obj.app.state.critic_agent),
        ("discovery_agent", request_obj.app.state.discovery_agent),
        ("web_browsing_service", request_obj.app.state.web_browsing_service)
    ]
    for agent_name, agent_instance in agents_to_check:
        if agent_instance:
            health_status["components"][agent_name] = {"status": "healthy", "message": f"{agent_name} initialized."}
        else:
            health_status["components"][agent_name] = {"status": "unhealthy", "message": f"{agent_name} not initialized."}
            health_status["status"] = "degraded"

    # Check Orchestration Service
    if request_obj.app.state.orchestration_service:
        health_status["components"]["orchestration_service"] = {"status": "healthy", "message": "Orchestration Service initialized."}
    else:
        health_status["components"]["orchestration_service"] = {"status": "unhealthy", "message": "Orchestration Service not initialized."}
        health_status["status"] = "degraded"

    # Check Cognitive Brain
    if request_obj.app.state.cognitive_brain:
        health_status["components"]["cognitive_brain"] = {"status": "healthy", "message": "Cognitive Brain initialized."}
    else:
        health_status["components"]["cognitive_brain"] = {"status": "unhealthy", "message": "Cognitive Brain not initialized."}
        health_status["status"] = "degraded"

    # Check Background Task Queue
    if request_obj.app.state.background_task_queue:
        health_status["components"]["background_task_queue"] = {"status": "healthy", "message": "Background Task Queue initialized."}
    else:
        health_status["components"]["background_task_queue"] = {"status": "unhealthy", "message": "Background Task Queue not initialized."}
        health_status["status"] = "degraded"

    # Check Self-Reflection & Discovery Engine
    if request_obj.app.state.self_reflection_discovery_engine:
        health_status["components"]["self_reflection_discovery_engine"] = {"status": "healthy", "message": "Self-Reflection & Discovery Engine initialized."}
    else:
        health_status["components"]["self_reflection_discovery_engine"] = {"status": "unhealthy", "message": "Self-Reflection & Discovery Engine not initialized."}
        health_status["status"] = "degraded"



    if health_status["status"] == "healthy":
        logger.info("Deep health check completed: All components healthy.")
    else:
        logger.warning("Deep health check completed: Some components are unhealthy or degraded.")

    return health_status
