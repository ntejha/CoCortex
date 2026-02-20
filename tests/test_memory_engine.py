"""
Tests for engine/memory_engine.py
Run from project root: python -m pytest tests/test_memory_engine.py -v
"""
import pytest
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.memory_engine import MemoryEngine


@pytest.fixture
def engine(tmp_path):
    db = str(tmp_path / "test_engine.db")
    return MemoryEngine(db_path=db)


SESSION = "test-session-001"


def test_save_and_load_persists_to_sqlite(engine):
    """Core regression: data must survive beyond in-memory dict."""
    records = [{"input": "My name is Alice", "output": "Nice to meet you, Alice!"}]
    engine.save(SESSION, records)

    loaded = engine.load(SESSION)
    assert len(loaded) == 1
    assert loaded[0]["input"] == "My name is Alice"
    assert loaded[0]["output"] == "Nice to meet you, Alice!"


def test_load_empty_session_returns_empty_list(engine):
    result = engine.load("nonexistent-session")
    assert result == []


def test_save_skips_already_persisted_records(engine):
    """Records with memory_id should not be re-saved."""
    records = [{"input": "Hello", "output": "Hi", "memory_id": "existing-id-123"}]
    engine.save(SESSION, records)
    loaded = engine.load(SESSION)
    assert len(loaded) == 0  # skipped because memory_id was set


def test_save_multiple_turns(engine):
    records = [
        {"input": "Turn 1 input", "output": "Turn 1 output"},
        {"input": "Turn 2 input", "output": "Turn 2 output"},
    ]
    engine.save(SESSION, records)
    loaded = engine.load(SESSION)
    assert len(loaded) == 2


def test_save_turn_convenience_method(engine):
    engine.save_turn(SESSION, human="What is Python?", assistant="A programming language.")
    loaded = engine.load(SESSION)
    assert len(loaded) == 1
    assert "Python" in loaded[0]["input"]


def test_load_history_formats_as_human_assistant(engine):
    engine.save_turn(SESSION, "Hello", "Hi there!")
    history = engine.load_history(SESSION)
    assert "Human: Hello" in history
    assert "Assistant: Hi there!" in history


def test_retrieve_by_keyword(engine):
    engine.save_turn(SESSION, "My favorite language is Python", "Great choice!")
    engine.save_turn(SESSION, "I like coffee", "Me too!")

    results = engine.retrieve(SESSION, "Python")
    assert len(results) == 1
    assert "Python" in results[0]["input"]


def test_repair_if_needed_removes_duplicates(engine):
    records = [
        {"input": "same", "output": "thing"},
        {"input": "same", "output": "thing"},
        {"input": "different", "output": "record"},
    ]
    cleaned = engine.repair_if_needed(records)
    assert len(cleaned) == 2


def test_repair_if_needed_removes_empty_records(engine):
    records = [
        {"input": "", "output": ""},
        {"input": "valid", "output": "record"},
    ]
    cleaned = engine.repair_if_needed(records)
    assert len(cleaned) == 1
    assert cleaned[0]["input"] == "valid"


def test_sessions_are_isolated(engine):
    """Data saved under session A should not appear under session B."""
    engine.save_turn("session-A", "Hello from A", "Response A")
    engine.save_turn("session-B", "Hello from B", "Response B")

    a_records = engine.load("session-A")
    b_records = engine.load("session-B")

    assert len(a_records) == 1
    assert len(b_records) == 1
    assert "session-A" not in b_records[0].get("input", "")