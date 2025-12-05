import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4, UUID
from datetime import datetime, timedelta
from pymongo.errors import ConnectionFailure

from src.services.memory_service import MemoryService
from src.services.llm_integration_service import LLMIntegrationService
from src.core.exceptions import ConfigurationError, APIException
from src.models.core_models import CognitiveCycle, MemoryQueryRequest, DiscoveredPattern, CycleListRequest, AgentOutput, ResponseMetadata, OutcomeSignals
from src.core.config import settings

@pytest.fixture(autouse=True)
def mock_settings():
    with patch('src.core.config.settings') as mock_config:
        mock_config.MONGO_URI = "mongodb://localhost:27017/"
        mock_config.MONGO_DB_NAME = "test_db"
        mock_config.MONGO_COLLECTION_CYCLES = "test_cycles"
        mock_config.MONGO_COLLECTION_PATTERNS = "test_patterns"
        mock_config.EMBEDDING_MODEL_NAME = "test-embedding-model"
        yield mock_config

@pytest.fixture
def mock_llm_service():
    mock = AsyncMock(spec=LLMIntegrationService)
    mock.generate_embedding.return_value = [0.1] * 1536
    return mock

@pytest.fixture
def memory_service(mock_llm_service):
    return MemoryService(llm_service=mock_llm_service)

@pytest.fixture
def mock_motor_client():
    with patch('motor.motor_asyncio.AsyncIOMotorClient') as mock_client_class:
        mock_client = AsyncMock()
        mock_client.admin.command.return_value = {'ok': 1}
        mock_db = MagicMock()
        mock_cycles_collection = AsyncMock()
        mock_patterns_collection = AsyncMock()
        mock_db.__getitem__.side_effect = lambda key: {
            settings.MONGO_COLLECTION_CYCLES: mock_cycles_collection,
            settings.MONGO_COLLECTION_PATTERNS: mock_patterns_collection
        }[key]
        mock_client.__getitem__.return_value = mock_db
        mock_client_class.return_value = mock_client
        yield mock_client, mock_cycles_collection, mock_patterns_collection

@pytest.mark.asyncio
async def test_memory_service_init_no_mongo_uri():
    with patch('src.core.config.settings.MONGO_URI', ""):
        with pytest.raises(ConfigurationError, match="MongoDB URI and DB Name must be configured"):
            MemoryService(AsyncMock())

@pytest.mark.asyncio
async def test_connect_success(memory_service, mock_motor_client):
    mock_client, mock_cycles_collection, mock_patterns_collection = mock_motor_client
    await memory_service.connect()
    assert memory_service.client is mock_client
    assert memory_service.cycles_collection is mock_cycles_collection
    assert memory_service.patterns_collection is mock_patterns_collection
    mock_client.admin.command.assert_called_once_with('ping')

@pytest.mark.asyncio
async def test_connect_failure(memory_service, mock_motor_client):
    mock_client, _, _ = mock_motor_client
    mock_client.admin.command.side_effect = ConnectionFailure("Connection refused")
    with pytest.raises(ConfigurationError, match="Failed to connect to MongoDB"):
        await memory_service.connect()

@pytest.mark.asyncio
async def test_close_success(memory_service, mock_motor_client):
    mock_client, _, _ = mock_motor_client
    await memory_service.connect()
    await memory_service.close()
    mock_client.close.assert_called_once()

@pytest.mark.asyncio
async def test_upsert_cycle_success(memory_service, mock_llm_service, mock_motor_client):
    _, mock_cycles_collection, _ = mock_motor_client
    await memory_service.connect()

    cycle_id = uuid4()
    user_id = uuid4()
    cognitive_cycle = CognitiveCycle(
        cycle_id=cycle_id,
        user_id=user_id,
        session_id=uuid4(),
        user_input="test input",
        final_response="test response"
    )
    mock_cycles_collection.update_one.return_value = AsyncMock(matched_count=0, modified_count=0, upserted_id=cycle_id)

    result = await memory_service.upsert_cycle(cognitive_cycle)
    assert result is True
    mock_llm_service.generate_embedding.assert_called_with(text="test response", model_name=settings.EMBEDDING_MODEL_NAME)
    assert mock_llm_service.generate_embedding.call_count == 2
    mock_cycles_collection.update_one.assert_called_once()
    args, kwargs = mock_cycles_collection.update_one.call_args
    assert args[0]['_id'] == str(cycle_id)
    assert args[0]['user_id'] == str(user_id)
    assert args[1]['$set']['user_input'] == "test input"

@pytest.mark.asyncio
async def test_upsert_cycle_no_llm_embedding_if_already_present(memory_service, mock_llm_service, mock_motor_client):
    _, mock_cycles_collection, _ = mock_motor_client
    await memory_service.connect()

    cycle_id = uuid4()
    user_id = uuid4()
    cognitive_cycle = CognitiveCycle(
        cycle_id=cycle_id,
        user_id=user_id,
        session_id=uuid4(),
        user_input="test input",
        user_input_embedding=[0.5]*1536,
        final_response="test response",
        final_response_embedding=[0.6]*1536
    )
    mock_cycles_collection.update_one.return_value = AsyncMock(matched_count=0, modified_count=0, upserted_id=cycle_id)

    result = await memory_service.upsert_cycle(cognitive_cycle)
    assert result is True
    mock_llm_service.generate_embedding.assert_not_called()

@pytest.mark.asyncio
async def test_query_memory_success(memory_service, mock_llm_service, mock_motor_client):
    _, mock_cycles_collection, _ = mock_motor_client
    await memory_service.connect()

    user_id = uuid4()
    query_request = MemoryQueryRequest(user_id=user_id, query_text="query text")
    mock_doc = {
        '_id': str(uuid4()),
        'user_id': str(user_id),
        'session_id': str(uuid4()),
        'timestamp': datetime.utcnow().isoformat(),
        'user_input': 'old input',
        'score': 0.8
    }
    mock_cycles_collection.aggregate.return_value = AsyncMock()
    mock_cycles_collection.aggregate.return_value.__aiter__.return_value = [mock_doc]

    retrieved_cycles = await memory_service.query_memory(query_request)
    assert len(retrieved_cycles) == 1
    assert retrieved_cycles[0].user_id == user_id
    mock_llm_service.generate_embedding.assert_called_once_with(text="query text", model_name=settings.EMBEDDING_MODEL_NAME)
    mock_cycles_collection.aggregate.assert_called_once()

@pytest.mark.asyncio
async def test_query_memory_no_results(memory_service, mock_llm_service, mock_motor_client):
    _, mock_cycles_collection, _ = mock_motor_client
    await memory_service.connect()

    user_id = uuid4()
    query_request = MemoryQueryRequest(user_id=user_id, query_text="no match")
    mock_cycles_collection.aggregate.return_value = AsyncMock()
    mock_cycles_collection.aggregate.return_value.__aiter__.return_value = []

    retrieved_cycles = await memory_service.query_memory(query_request)
    assert len(retrieved_cycles) == 0

@pytest.mark.asyncio
async def test_get_recent_cycles_for_reflection_success(memory_service, mock_motor_client):
    _, mock_cycles_collection, _ = mock_motor_client
    await memory_service.connect()

    user_id = uuid4()
    cycle1_id = uuid4()
    cycle2_id = uuid4()
    mock_doc1 = {
        '_id': str(cycle1_id),
        'user_id': str(user_id),
        'session_id': str(uuid4()),
        'timestamp': (datetime.utcnow() - timedelta(minutes=1)).isoformat(),
        'user_input': 'input 1',
        'reflection_status': 'pending'
    }
    mock_doc2 = {
        '_id': str(cycle2_id),
        'user_id': str(user_id),
        'session_id': str(uuid4()),
        'timestamp': datetime.utcnow().isoformat(),
        'user_input': 'input 2',
        'reflection_status': 'pending'
    }
    mock_cursor = AsyncMock()
    mock_cursor.sort.return_value.limit.return_value = mock_cursor
    mock_cursor.__aiter__.return_value = [mock_doc2, mock_doc1]
    mock_cycles_collection.find.return_value = mock_cursor

    recent_cycles = await memory_service.get_recent_cycles_for_reflection(user_id, 2)
    assert len(recent_cycles) == 2
    assert recent_cycles[0].cycle_id == cycle2_id
    assert recent_cycles[1].cycle_id == cycle1_id
    mock_cycles_collection.find.assert_called_once_with({'user_id': str(user_id), 'reflection_status': 'pending'})

@pytest.mark.asyncio
async def test_get_cycle_by_id_found(memory_service, mock_motor_client):
    _, mock_cycles_collection, _ = mock_motor_client
    await memory_service.connect()

    user_id = uuid4()
    cycle_id = uuid4()
    mock_doc = {
        '_id': str(cycle_id),
        'user_id': str(user_id),
        'session_id': str(uuid4()),
        'timestamp': datetime.utcnow().isoformat(),
        'user_input': 'test input'
    }
    mock_cycles_collection.find_one.return_value = mock_doc

    cycle = await memory_service.get_cycle_by_id(user_id, cycle_id)
    assert cycle is not None
    assert cycle.cycle_id == cycle_id
    mock_cycles_collection.find_one.assert_called_once_with({'_id': str(cycle_id), 'user_id': str(user_id)})

@pytest.mark.asyncio
async def test_get_cycle_by_id_not_found(memory_service, mock_motor_client):
    _, mock_cycles_collection, _ = mock_motor_client
    await memory_service.connect()

    user_id = uuid4()
    cycle_id = uuid4()
    mock_cycles_collection.find_one.return_value = None

    cycle = await memory_service.get_cycle_by_id(user_id, cycle_id)
    assert cycle is None

@pytest.mark.asyncio
async def test_update_cycle_metadata_success(memory_service, mock_motor_client):
    _, mock_cycles_collection, _ = mock_motor_client
    await memory_service.connect()

    user_id = uuid4()
    cycle_id = uuid4()
    metadata_to_update = {"new_field": "new_value", "response_metadata.tone": "updated_tone"}
    mock_cycles_collection.update_one.return_value = AsyncMock(matched_count=1, modified_count=1)

    updated = await memory_service.update_cycle_metadata(user_id, cycle_id, metadata_to_update)
    assert updated is True
    mock_cycles_collection.update_one.assert_called_once()
    args, kwargs = mock_cycles_collection.update_one.call_args
    assert args[0]['_id'] == str(cycle_id)
    assert args[0]['user_id'] == str(user_id)
    assert args[1]['$set']['metadata.new_field'] == "new_value"
    assert args[1]['$set']['response_metadata.tone'] == "updated_tone"

@pytest.mark.asyncio
async def test_update_cycle_metadata_no_change(memory_service, mock_motor_client):
    _, mock_cycles_collection, _ = mock_motor_client
    await memory_service.connect()

    user_id = uuid4()
    cycle_id = uuid4()
    metadata_to_update = {"existing_field": "same_value"}
    mock_cycles_collection.update_one.return_value = AsyncMock(matched_count=1, modified_count=0)

    updated = await memory_service.update_cycle_metadata(user_id, cycle_id, metadata_to_update)
    assert updated is False

@pytest.mark.asyncio
async def test_list_cycles_success(memory_service, mock_motor_client):
    _, mock_cycles_collection, _ = mock_motor_client
    await memory_service.connect()

    user_id = uuid4()
    list_request = CycleListRequest(user_id=user_id, limit=1)
    mock_doc = {
        '_id': str(uuid4()),
        'user_id': str(user_id),
        'session_id': str(uuid4()),
        'timestamp': datetime.utcnow().isoformat(),
        'user_input': 'input for list'
    }
    mock_cursor = AsyncMock()
    mock_cursor.sort.return_value.skip.return_value.limit.return_value = mock_cursor
    mock_cursor.__aiter__.return_value = [mock_doc]
    mock_cycles_collection.find.return_value = mock_cursor
    mock_cycles_collection.count_documents.return_value = 1

    cycles, total_cycles = await memory_service.list_cycles(list_request)
    assert len(cycles) == 1
    assert total_cycles == 1
    mock_cycles_collection.find.assert_called_once_with({'user_id': str(user_id)})

@pytest.mark.asyncio
async def test_list_cycles_with_filters(memory_service, mock_motor_client):
    _, mock_cycles_collection, _ = mock_motor_client
    await memory_service.connect()

    user_id = uuid4()
    session_id = uuid4()
    start_date = datetime.utcnow() - timedelta(days=1)
    end_date = datetime.utcnow()
    list_request = CycleListRequest(
        user_id=user_id,
        session_id=session_id,
        start_date=start_date,
        end_date=end_date,
        response_type="informational",
        min_confidence=0.8
    )
    mock_doc = {
        '_id': str(uuid4()),
        'user_id': str(user_id),
        'session_id': str(session_id),
        'timestamp': datetime.utcnow().isoformat(),
        'user_input': 'filtered input',
        'response_metadata': {'response_type': 'informational'},
        'agent_outputs': [{'confidence': 0.9, 'agent_id': 'test', 'analysis': {}, 'priority': 1}]
    }
    mock_cursor = AsyncMock()
    mock_cursor.sort.return_value.skip.return_value.limit.return_value = mock_cursor
    mock_cursor.__aiter__.return_value = [mock_doc]
    mock_cycles_collection.find.return_value = mock_cursor
    mock_cycles_collection.count_documents.return_value = 1

    cycles, total_cycles = await memory_service.list_cycles(list_request)
    assert len(cycles) == 1
    assert total_cycles == 1
    args, kwargs = mock_cycles_collection.find.call_args
    assert args[0]['session_id'] == str(session_id)
    assert args[0]['timestamp']['$gte'] == start_date
    assert args[0]['timestamp']['$lte'] == end_date
    assert args[0]['response_metadata.response_type'] == "informational"
    assert args[0]['agent_outputs.confidence']['$gte'] == 0.8

@pytest.mark.asyncio
async def test_upsert_pattern_success(memory_service, mock_llm_service, mock_motor_client):
    _, _, mock_patterns_collection = mock_motor_client
    await memory_service.connect()

    pattern_id = uuid4()
    user_id = uuid4()
    pattern = DiscoveredPattern(
        pattern_id=pattern_id,
        user_id=user_id,
        pattern_type="meta_learning",
        description="test pattern",
        source_cycle_ids=[uuid4()]
    )
    mock_patterns_collection.update_one.return_value = AsyncMock(upserted_id=pattern_id)

    result = await memory_service.upsert_pattern(pattern)
    assert result is True
    mock_llm_service.generate_embedding.assert_called_once_with(text="test pattern", model_name=settings.EMBEDDING_MODEL_NAME)
    mock_patterns_collection.update_one.assert_called_once()
    args, kwargs = mock_patterns_collection.update_one.call_args
    assert args[0]['_id'] == str(pattern_id)
    assert args[0]['user_id'] == str(user_id)

@pytest.mark.asyncio
async def test_delete_user_data_success(memory_service, mock_motor_client):
    _, mock_cycles_collection, mock_patterns_collection = mock_motor_client
    await memory_service.connect()

    user_id = uuid4()
    mock_cycles_collection.delete_many.return_value = AsyncMock(deleted_count=5)
    mock_patterns_collection.delete_many.return_value = AsyncMock(deleted_count=2)

    deleted = await memory_service.delete_user_data(user_id)
    assert deleted is True
    mock_cycles_collection.delete_many.assert_called_once_with({'user_id': str(user_id)})
    mock_patterns_collection.delete_many.assert_called_once_with({'user_id': str(user_id)})

@pytest.mark.asyncio
async def test_delete_user_data_no_data(memory_service, mock_motor_client):
    _, mock_cycles_collection, mock_patterns_collection = mock_motor_client
    await memory_service.connect()

    user_id = uuid4()
    mock_cycles_collection.delete_many.return_value = AsyncMock(deleted_count=0)
    mock_patterns_collection.delete_many.return_value = AsyncMock(deleted_count=0)

    deleted = await memory_service.delete_user_data(user_id)
    assert deleted is False

@pytest.mark.asyncio
async def test_get_patterns_for_user_success(memory_service, mock_motor_client):
    _, _, mock_patterns_collection = mock_motor_client
    await memory_service.connect()

    user_id = uuid4()
    pattern_id = uuid4()
    mock_doc = {
        '_id': str(pattern_id),
        'pattern_id': str(pattern_id),
        'user_id': str(user_id),
        'timestamp': datetime.utcnow().isoformat(),
        'pattern_type': 'meta_learning',
        'description': 'test pattern description',
        'source_cycle_ids': [str(uuid4())]
    }
    mock_cursor = AsyncMock()
    mock_cursor.__aiter__.return_value = [mock_doc]
    mock_patterns_collection.find.return_value = mock_cursor

    patterns = await memory_service.get_patterns_for_user(user_id)
    assert len(patterns) == 1
    assert patterns[0].pattern_id == pattern_id
    assert patterns[0].user_id == user_id
    mock_patterns_collection.find.assert_called_once_with({'user_id': str(user_id)})

@pytest.mark.asyncio
async def test_get_patterns_for_user_no_patterns(memory_service, mock_motor_client):
    _, _, mock_patterns_collection = mock_motor_client
    await memory_service.connect()

    user_id = uuid4()
    mock_cursor = AsyncMock()
    mock_cursor.__aiter__.return_value = []
    mock_patterns_collection.find.return_value = mock_cursor

    patterns = await memory_service.get_patterns_for_user(user_id)
    assert len(patterns) == 0

@pytest.mark.asyncio
async def test_memory_service_not_connected_raises_exception(memory_service):
    user_id = uuid4()
    cycle_id = uuid4()
    cognitive_cycle = CognitiveCycle(user_id=user_id, session_id=uuid4(), user_input="test")
    query_request = MemoryQueryRequest(user_id=user_id, query_text="test")
    pattern = DiscoveredPattern(user_id=user_id, pattern_type="test", description="test")

    with pytest.raises(APIException, match="MemoryService not connected to MongoDB."):
        await memory_service.upsert_cycle(cognitive_cycle)
    with pytest.raises(APIException, match="MemoryService not connected to MongoDB."):
        await memory_service.query_memory(query_request)
    with pytest.raises(APIException, match="MemoryService not connected to MongoDB."):
        await memory_service.get_recent_cycles_for_reflection(user_id, 1)
    with pytest.raises(APIException, match="MemoryService not connected to MongoDB."):
        await memory_service.get_cycle_by_id(user_id, cycle_id)
    with pytest.raises(APIException, match="MemoryService not connected to MongoDB."):
        await memory_service.update_cycle_metadata(user_id, cycle_id, {"key": "value"})
    with pytest.raises(APIException, match="MemoryService not connected to MongoDB."):
        await memory_service.list_cycles(CycleListRequest(user_id=user_id))
    with pytest.raises(APIException, match="MemoryService not connected to MongoDB."):
        await memory_service.upsert_pattern(pattern)
    with pytest.raises(APIException, match="MemoryService not connected to MongoDB."):
        await memory_service.delete_user_data(user_id)
    with pytest.raises(APIException, match="MemoryService not connected to MongoDB."):
        await memory_service.get_patterns_for_user(user_id)
