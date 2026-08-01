import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.providers.base import EmbeddingModelIdentity
from src.tools.eval_retrieval import (
    FIXTURE_VERSION,
    evaluate_collection,
    load_fixture,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
)

OLLAMA = EmbeddingModelIdentity(provider="ollama", model="embeddinggemma:latest", vector_dimension=768)


def test_recall_counts_unique_relevant_hits_within_k():
    assert recall_at_k(["a", "b", "c"], ["a", "c"], 3) == 1.0
    assert recall_at_k(["a", "b", "c"], ["a", "z"], 3) == 0.5
    assert recall_at_k(["x", "y", "a"], ["a"], 2) == 0.0


def test_recall_is_zero_when_nothing_is_relevant():
    assert recall_at_k(["a"], [], 5) == 0.0


def test_reciprocal_rank_uses_the_first_hit():
    assert reciprocal_rank(["a", "b"], ["a"]) == 1.0
    assert reciprocal_rank(["x", "a"], ["a"]) == 0.5
    assert reciprocal_rank(["x", "y"], ["a"]) == 0.0


def test_ndcg_is_one_for_a_perfect_ranking():
    assert ndcg_at_k(["a", "b", "c"], ["a", "b"], 3) == pytest.approx(1.0)


def test_ndcg_penalises_lower_placement():
    top = ndcg_at_k(["a", "x", "y"], ["a"], 3)
    lower = ndcg_at_k(["x", "y", "a"], ["a"], 3)
    assert 0.0 < lower < top == pytest.approx(1.0)


def test_ndcg_ignores_results_beyond_k():
    assert ndcg_at_k(["x", "y", "a"], ["a"], 2) == 0.0


def _fixture_file(tmp_path: Path, queries, version=FIXTURE_VERSION) -> Path:
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps({"version": version, "queries": queries}), encoding="utf-8")
    return path


def test_load_fixture_rejects_an_unsupported_version(tmp_path):
    path = _fixture_file(tmp_path, [{"query": "q", "relevant_ids": ["a"]}], version=99)
    with pytest.raises(ValueError, match="not supported"):
        load_fixture(path)


def test_load_fixture_rejects_entries_without_relevant_ids(tmp_path):
    path = _fixture_file(tmp_path, [{"query": "q", "relevant_ids": []}])
    with pytest.raises(ValueError, match="no usable entries"):
        load_fixture(path)


def test_load_fixture_reports_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="build-template"):
        load_fixture(tmp_path / "absent.json")


@pytest.mark.asyncio
async def test_evaluation_aggregates_metrics_and_records_misses(monkeypatch):
    collection = MagicMock()
    collection.metadata = OLLAMA.as_collection_metadata()
    collection.query.side_effect = [
        {"ids": [["a", "b", "c"]]},
        {"ids": [["x", "y", "z"]]},
    ]
    client = MagicMock()
    client.get_collection.return_value = collection

    provider = MagicMock()
    provider.embed = AsyncMock(return_value=[0.1] * 768)
    monkeypatch.setattr(
        "src.tools.eval_retrieval.build_embedding_provider_for_identity",
        lambda identity, scheduler: provider,
    )

    result = await evaluate_collection(
        client,
        "cognitive_cycles",
        [
            {"id": "q1", "query": "first", "relevant_ids": ["a"]},
            {"id": "q2", "query": "second", "relevant_ids": ["missing"]},
        ],
        scheduler=MagicMock(),
        k=3,
    )

    assert result.queries == 2
    assert result.recall == pytest.approx(0.5)
    assert result.mrr == pytest.approx(0.5)
    assert result.misses == ["q2"]
    assert [query.query_id for query in result.weak_queries] == ["q2"]
    assert result.weak_queries[0].retrieved == ["x", "y", "z"]
    assert result.weak_queries[0].relevant == ["missing"]
    assert result.identity == "ollama/embeddinggemma:latest@768d"


@pytest.mark.asyncio
async def test_evaluation_records_partial_recall_as_a_weak_query(monkeypatch):
    collection = MagicMock()
    collection.metadata = OLLAMA.as_collection_metadata()
    collection.query.return_value = {"ids": [["a", "x", "y"]]}
    client = MagicMock()
    client.get_collection.return_value = collection

    provider = MagicMock()
    provider.embed = AsyncMock(return_value=[0.1] * 768)
    monkeypatch.setattr(
        "src.tools.eval_retrieval.build_embedding_provider_for_identity",
        lambda identity, scheduler: provider,
    )

    result = await evaluate_collection(
        client,
        "cognitive_cycles",
        [{"id": "partial", "query": "query", "relevant_ids": ["a", "b"]}],
        scheduler=MagicMock(),
        k=3,
    )

    assert result.misses == []
    assert len(result.weak_queries) == 1
    assert result.weak_queries[0].query_id == "partial"
    assert result.weak_queries[0].recall == pytest.approx(0.5)
    assert result.weak_queries[0].mrr == pytest.approx(1.0)


@pytest.mark.asyncio
async def test_evaluation_refuses_a_collection_without_an_identity():
    collection = MagicMock()
    collection.metadata = None
    client = MagicMock()
    client.get_collection.return_value = collection

    with pytest.raises(ValueError, match="no embedding identity"):
        await evaluate_collection(
            client, "cognitive_cycles", [{"query": "q", "relevant_ids": ["a"]}], MagicMock(), k=1
        )
