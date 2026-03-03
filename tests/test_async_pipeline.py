"""
tests/test_async_pipeline.py

Integration tests covering the async equivalents:
- PlannerAgent.aplan()
- WorkerAgent.aexecute()
- EvaluatorAgent.aevaluate()
- AsyncMemoryStore
- async_repair parallel verification
"""

import asyncio
import pytest
import os

from memory.schemas import MemoryItem
from memory.async_store import AsyncMemoryStore
from agents.planner import PlannerAgent
from agents.worker import WorkerAgent
from agents.evaluator import EvaluatorAgent
from memory.async_repair import check_and_repair_async


class MockSyncLLM:
    """Mock LLM that works with asyncio.to_thread sync offloading."""

    def __init__(self, responses):
        self.responses = responses
        self.call_count = 0

    def generate(self, prompt: str) -> str:
        self.call_count += 1
        for key, value in self.responses.items():
            if key in prompt:
                return value
        return "Unknown mock response"


@pytest.fixture
async def async_store(tmp_path):
    db_path = str(tmp_path / "test_async.db")
    store = AsyncMemoryStore(db_path)
    await store.init_schema()
    yield store
    await store.close()


@pytest.fixture
def mock_planner(async_store):
    responses = {
        "Which memory indices": "[]",
        "Plan the steps": "1. Async Step A\n2. Async Step B",
    }
    return PlannerAgent(llm=MockSyncLLM(responses), memory_store=async_store)


@pytest.fixture
def mock_worker(async_store):
    responses = {
        "Which memory indices": "[]",
        "Execute plan.": "Executed A and B asynchronously.",
    }
    return WorkerAgent(llm=MockSyncLLM(responses), memory_store=async_store)


@pytest.fixture
def mock_evaluator(async_store):
    responses = {
        "Which memory indices": "[]",
        "Evaluate result. Explicitly state PASS or FAIL": "PASS - Execution looks good.",
    }
    return EvaluatorAgent(llm=MockSyncLLM(responses), memory_store=async_store)


# ---- Tests ----


@pytest.mark.asyncio
async def test_async_store_crud(async_store):
    """Test AsyncMemoryStore basic operations."""
    mem = MemoryItem(content="test async", memory_type="episodic", source_agent="planner")
    await async_store.add_memory(mem)

    loaded = await async_store.get_memory(str(mem.id))
    assert loaded is not None
    assert loaded.content == "test async"

    await async_store.update_memory(str(mem.id), "usage_count", 5)
    loaded_up = await async_store.get_memory(str(mem.id))
    assert loaded_up.usage_count == 5


@pytest.mark.asyncio
async def test_aplan_returns_output_and_id(mock_planner):
    plan, decision_id = await mock_planner.aplan("Build an async pipeline")
    assert "Async Step A" in plan
    assert decision_id.startswith("planner_")


@pytest.mark.asyncio
async def test_aexecute_returns_output_and_id(mock_worker):
    result, decision_id = await mock_worker.aexecute("1. Async Step A\n2. Async Step B")
    assert "Executed" in result
    assert decision_id.startswith("worker_")


@pytest.mark.asyncio
async def test_aevaluate_returns_output_and_id(mock_evaluator):
    eval_res, decision_id = await mock_evaluator.aevaluate("Executed A and B asynchronously.")
    assert "PASS" in eval_res
    assert decision_id.startswith("evaluator_")


@pytest.mark.asyncio
async def test_async_pipeline_runs_end_to_end(mock_planner, mock_worker, mock_evaluator):
    """Verifies that the whole pipeline flows seamlessly within coroutines."""
    plan, pid = await mock_planner.aplan("Do something")
    assert plan is not None

    result, wid = await mock_worker.aexecute(plan)
    assert result is not None

    eval_res, eid = await mock_evaluator.aevaluate(result)
    assert "PASS" in eval_res


@pytest.mark.asyncio
async def test_async_repair_verifies_in_parallel(async_store):
    """
    Simulates a failed decision and runs the scatter-gather async repair.
    """
    decision_id = "fail_123"

    m1 = MemoryItem(content="Suspect 1", memory_type="episodic", source_agent="planner")
    m1.influenced_decisions.append(decision_id)
    m2 = MemoryItem(content="Suspect 2", memory_type="episodic", source_agent="planner")
    m2.influenced_decisions.append(decision_id)

    await async_store.add_memory(m1)
    await async_store.add_memory(m2)

    responses = {
        "Suspect 1": "INCORRECT",
        "Suspect 2": "CORRECT",
    }
    verifier = EvaluatorAgent(llm=MockSyncLLM(responses), memory_store=async_store)

    await check_and_repair_async(async_store, decision_id, verifier)

    r1 = await async_store.get_memory(str(m1.id))
    assert r1.status == "quarantined"

    r2 = await async_store.get_memory(str(m2.id))
    assert r2.status == "active"

    assert verifier.llm.call_count == 2
