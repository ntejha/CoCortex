# CoCortex — Complete Project Explanation

> **Audience:** A new intern who is smart but has never seen this codebase before.
> **Goal:** After reading this, you should be able to navigate, run, debug, and contribute to CoCortex without asking the team basic questions.

---

## 1. What Is CoCortex? (Non-Technical)

Imagine you have a team of three AI assistants — a **Planner**, a **Worker**, and an **Evaluator** — all collaborating on tasks. Each one generates knowledge while working. The problem? If one assistant produces a wrong "fact" (e.g., _"Photosynthesis occurs only at night"_), it pollutes the shared knowledge base, and every other assistant starts using that wrong information. There is no mechanism to catch, quarantine, or fix bad knowledge.

**CoCortex solves this.** It is a framework that gives multiple AI agents a **shared memory system with built-in quality control**. Before any piece of knowledge enters the shared memory, it goes through a **consensus voting process**. If a memory later causes failures, CoCortex traces the bad memory back to its source and repairs or quarantines it automatically.

**Who uses it:** Researchers and developers building multi-agent LLM (Large Language Model) systems who need reliable, self-healing shared memory.

**Why it exists:** Existing multi-agent frameworks let agents share memory freely, but have no verification, no trust scoring, and no self-repair. CoCortex fills that gap.

---

## 2. Project Overview

### Problem Solved

In multi-agent LLM systems, agents produce outputs that get stored as shared knowledge. Without verification:
- **Bad memories contaminate** all downstream decisions
- **No traceability** — you can't find which memory caused a failure
- **No self-healing** — wrong knowledge stays forever

### Target Users

- AI/ML researchers working on multi-agent systems
- Developers integrating LLM-based agents with persistent memory
- Academic projects comparing memory reliability approaches

### Core Features

| Feature | Description |
|---|---|
| **Consensus-Based Admission** | Three voters (Planner, Worker, Safety) must agree before a memory is accepted |
| **Role-Specialized Memory Views** | Each agent sees only the memories relevant to its role |
| **Causal Influence Tracking** | Every decision records which memories influenced it |
| **Reliability Scoring** | Memories have dynamic scores based on usage, failures, and time decay |
| **Lifecycle Management** | Memories automatically transition: `episodic → semantic → stale → deprecated → archived` |
| **Causal Traceback & Repair** | When a decision fails, trace back to the bad memory and fix it |
| **Memory Provenance** | Full audit trail: who created a memory, how reliable it is, what repairs happened |
| **LangChain Integration** | Drop-in memory adapter for LangChain-based applications |

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER / APPLICATION                       │
└────────────────────────────┬────────────────────────────────────┘
                             │ task
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     AGENT LAYER (agents/)                       │
│  ┌──────────┐    ┌──────────┐    ┌───────────┐                 │
│  │ Planner  │───▶│  Worker  │───▶│ Evaluator │                 │
│  └────┬─────┘    └────┬─────┘    └─────┬─────┘                 │
│       │               │               │                         │
│       │  memory views │  memory views  │  memory views          │
│       ▼               ▼               ▼                         │
│  ┌─────────────────────────────────────────────┐               │
│  │          Memory Manager Agent               │               │
│  └──────────────────┬──────────────────────────┘               │
└─────────────────────┼───────────────────────────────────────────┘
                      │ proposals
                      ▼
┌─────────────────────────────────────────────────────────────────┐
│              CONSENSUS LAYER (consensus/)                       │
│  ┌────────────────┐ ┌──────────────┐ ┌──────────────────┐      │
│  │ Planner Voter  │ │ Worker Voter │ │ Safety Voter     │      │
│  └───────┬────────┘ └──────┬───────┘ └────────┬─────────┘      │
│          └─────────────────┼──────────────────┘                 │
│                            ▼                                    │
│                   Consensus Engine                              │
│              (accept / quarantine / reject)                     │
└────────────────────────────┬────────────────────────────────────┘
                             │ accepted memories
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                   MEMORY LAYER (memory/)                        │
│  ┌──────────┐  ┌─────────┐  ┌───────────┐  ┌────────────┐     │
│  │  Store   │  │ Scoring │  │ Lifecycle │  │ Provenance │     │
│  │ (SQLite) │  │         │  │           │  │            │     │
│  └──────────┘  └─────────┘  └───────────┘  └────────────┘     │
│  ┌──────────┐  ┌────────────────┐  ┌──────────────────┐       │
│  │  Views   │  │  Verification  │  │     Repair       │       │
│  └──────────┘  └────────────────┘  └──────────────────┘       │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│               ENGINE LAYER (engine/)                            │
│         MemoryEngine — Production facade for                    │
│         task-scoped + conversation-history modes                │
└─────────────────────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│            INTEGRATIONS (integrations/)                         │
│         LangChain-compatible memory adapter                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3. Repository Structure

```
cocortex/
├── agents/                  # AI agent definitions (Planner, Worker, Evaluator, MemoryManager)
├── consensus/               # Voting system that decides if a memory should be accepted
├── core/                    # Shared infrastructure (config, LLM client, decision IDs)
├── engine/                  # High-level MemoryEngine facade for production use
├── memory/                  # Core memory system (store, schemas, scoring, repair, views)
├── integrations/            # Third-party adapters (LangChain)
├── experiments/             # Step-by-step demo scripts (step01 through step11)
├── tests/                   # Pytest test suite
├── Documentation/           # Build diary — step-by-step development notes
├── Papers/                  # Research papers referenced during design
├── pyproject.toml           # Python packaging config (dependencies, build settings)
├── requirements.txt         # Pip-style dependency list
├── .env.example             # Template for environment variables
├── .gitignore               # Git exclusions
└── __init__.py              # Package root
```

### What Runs Where

| Layer | Location | Runtime |
|---|---|---|
| **LLM Inference** | Groq cloud API | Remote (API call) |
| **Memory Storage** | `cocortex_memory.db` (SQLite) | Local file |
| **Agent Logic** | `agents/`, `consensus/`, `memory/` | Local Python process |
| **Experiments** | `experiments/` | Local Python scripts |
| **Tests** | `tests/` | Local via `pytest` |

There is **no web server, no frontend, and no background workers**. CoCortex is a **Python library/framework** that you import into your own application.

---

## 4. How Data Flows

Here is the complete lifecycle of a single task execution:

### Step-by-Step Request Lifecycle

```
1. USER submits a task (e.g., "Explain photosynthesis")
         │
2. PlannerAgent receives the task
   ├── Fetches its MEMORY VIEW (semantic memories only, truncated to 300 chars)
   ├── Logs which memories influenced this decision (causal tracking)
   ├── Sends task + memories to the LLM via Groq API
   └── Returns a plan + decision_id
         │
3. WorkerAgent receives the plan
   ├── Fetches its MEMORY VIEW (all episodic + semantic memories, full content)
   ├── Logs causal influence
   ├── Sends plan + memories to LLM
   └── Returns execution output + decision_id
         │
4. EvaluatorAgent receives the output
   ├── Fetches its MEMORY VIEW (semantic only, confidence ≥ 0.8)
   ├── Logs causal influence
   ├── Asks LLM to verify correctness and consistency
   └── Returns evaluation + decision_id
         │
5. MemoryManagerAgent receives the output to store
   ├── Creates a MemoryProposal
   ├── Runs CONSENSUS (3 voters vote on the proposal)
   │   ├── Planner Voter:  Is this reusable knowledge?
   │   ├── Worker Voter:   Is this actionable?
   │   └── Safety Voter:   Is this safe and factual?
   ├── Consensus Engine tallies votes:
   │   ├── Any risk flag → QUARANTINE
   │   ├── ≥2 approvals  → ACCEPT (with averaged confidence)
   │   └── Otherwise     → REJECT
   └── If accepted: creates MemoryItem → saves to SQLite via MemoryStore
         │
6. IF a decision later fails:
   ├── repair.trace_suspect_memories() finds all memories that influenced it
   ├── MemoryVerifier asks the LLM if each memory is correct/incorrect/uncertain
   ├── decide_repair_action() applies deterministic policy:
   │   ├── incorrect → quarantine the memory
   │   ├── uncertain + low confidence → downrank (reduce confidence by 0.2)
   │   └── high failure count → downrank
   └── Repair events are logged in the memory's repair_history
```

---

## 5. Tech Stack

| Technology | Why It's Used | Responsibility in CoCortex |
|---|---|---|
| **Python 3.10+** | Primary language, mature AI/ML ecosystem | All application logic |
| **Groq API** | Ultra-fast LLM inference (runs Llama 3.1) | Powers all agent reasoning — planning, execution, evaluation, and memory verification |
| **LangChain** | Standard framework for LLM-based apps | CoCortex provides a LangChain-compatible memory adapter so it can be plugged into LangChain chains |
| **Pydantic** | Data validation with type safety | Defines schemas for `MemoryItem`, `MemoryProposal`, and `Vote` — ensures data integrity at boundaries |
| **SQLite** | Embedded database, zero configuration | Persists all memories locally in `cocortex_memory.db` |
| **python-dotenv** | Loads `.env` files | Manages API keys and configuration without hardcoding secrets |
| **pytest** | Python testing framework | Runs all unit tests in `tests/` |

---

## 6. Environment & Setup

### Installation

```bash
# 1. Clone the repository
git clone <repo-url>
cd cocortex

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate      # Linux/Mac
# venv\Scripts\activate       # Windows

# 3. Install dependencies
pip install -e ".[dev]"
# OR
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file in the project root (copy from `.env.example`):

```bash
cp .env.example .env
```

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `GROQ_API_KEY` | **Yes** | — | Your Groq API key for LLM inference. Get one at [console.groq.com](https://console.groq.com) |
| `LLM_MODEL` | No | `llama-3.1-8b-instant` | Which Groq-hosted model to use |
| `COCORTEX_DB_PATH` | No | `cocortex_memory.db` | File path for the SQLite database |

### Running Locally

```bash
# Run an experiment demo (e.g., the MVP self-healing demo)
python -m experiments.step07_demo

# Run the baseline vs CoCortex comparison
python -m experiments.step10_demo

# Run all tests
python -m pytest tests/ -v
```

### Production Use

CoCortex is a library — you import it into your own application:

```python
# Using the LangChain integration
from integrations.langchain import cocortex_langchain_memory

memory = cocortex_langchain_memory(session_id="my-session")
# Use `memory` as a drop-in replacement for LangChain memory
```

---

## 7. Key Modules

### 7.1 `agents/` — The AI Agents

**Purpose:** Define the four specialized agents that collaborate on tasks.

| File | Class | Purpose | Input | Output |
|---|---|---|---|---|
| `planner.py` | `PlannerAgent` | Breaks a task into an execution plan | task string | plan text + decision_id |
| `worker.py` | `WorkerAgent` | Executes the plan step by step | plan string | execution output + decision_id |
| `evaluator.py` | `EvaluatorAgent` | Checks correctness of worker output | output string | evaluation text + decision_id |
| `memory_manager.py` | `MemoryManagerAgent` | Orchestrates consensus + memory storage | content + source + context | (decision, MemoryItem or None) |

**Internal flow of every agent (Planner, Worker, Evaluator):**
1. Generate a unique `decision_id` (e.g., `planner_a3f8c2...`)
2. Fetch the agent-specific **memory view** from the store
3. **Log causal influence** — record that these memories influenced this decision
4. Build a prompt combining the task/plan/output with the retrieved memories
5. Send to the LLM via `LLMClient.generate()`
6. Return the LLM result + the decision_id

**MemoryManagerAgent flow:**
1. Wrap the output into a `MemoryProposal`
2. Collect votes from all three voters
3. Run `run_consensus()` to get a decision
4. If accepted → create a `MemoryItem` and save to store
5. If quarantined → save with low confidence and `status="quarantined"`
6. If rejected → discard

---

### 7.2 `consensus/` — The Voting System

**Purpose:** Ensures only quality, safe, and actionable knowledge enters the shared memory.

| File | What It Does |
|---|---|
| `schemas.py` | Defines `MemoryProposal` (what's being proposed) and `Vote` (a voter's response) |
| `voters.py` | Three independent voting functions with distinct evaluation criteria |
| `engine.py` | Tallies votes and returns `accept`, `quarantine`, or `reject` |

**The Three Voters:**

| Voter | Question It Asks | Rejects When | Approves When |
|---|---|---|---|
| `planner_voter` | _"Is this reusable across future tasks?"_ | Content is too short (<40 chars), looks like a task-specific trace | Contains general knowledge signals ("always", "typically", "is defined as") |
| `worker_voter` | _"Is this actionable for execution?"_ | Content is vague ("it depends", "might work"), is a question | Contains concrete result signals ("completed", "returned", "error") |
| `rule_based_voter` | _"Is this safe and factual?"_ | Contains unsafe terms ("hack", "exploit", "jailbreak") or misinformation patterns | No safety violations found |

**Consensus Rules:**
- **Any vote with `risk=True`** → **Quarantine** (safety voter has effective veto power)
- **≥2 approval votes** → **Accept** (confidence = average of approving votes)
- **Otherwise** → **Reject**

---

### 7.3 `core/` — Shared Infrastructure

| File | Purpose |
|---|---|
| `config.py` | Central configuration — loads env vars, provides `validate_config()` to fail fast on missing keys |
| `llm_client.py` | `LLMClient` class — wraps the Groq SDK, sends prompts with `temperature=0.3` for deterministic responses |
| `decision.py` | `generate_decision_id()` utility — creates unique IDs like `planner_a3f8c2d4...` |

---

### 7.4 `memory/` — The Core Memory System

This is the heart of CoCortex. It handles everything about how memories are stored, scored, viewed, repaired, and explained.

| File | Key Export | Purpose |
|---|---|---|
| `schemas.py` | `MemoryItem` | Pydantic model — the canonical shape of every memory in the system |
| `store.py` | `MemoryStore` | SQLite-backed CRUD for memories — add, get, update, promote, quarantine |
| `views.py` | `get_planner_view()`, `get_worker_view()`, `get_evaluator_view()` | Role-filtered memory access — each agent sees a different slice |
| `scoring.py` | `compute_reliability()` | Calculates a dynamic reliability score based on usage, failures, and time decay |
| `lifecycle.py` | `update_lifecycle()` | Transitions memory state based on reliability score and failure count |
| `verification.py` | `MemoryVerifier` | Uses an LLM prompt to fact-check a memory, returns `correct`/`incorrect`/`uncertain` |
| `repair.py` | `repair_memories()` | Causal traceback + repair: finds memories that caused a bad decision, then fixes them |
| `provenance.py` | `ProvenanceEngine` | Audit/explainability: prints who created a memory, its full history, and traces failures |
| `cocortex_memory.py` | `CoCortexMemory` | LangChain-compatible adapter — duck-types `BaseMemory` |

**Memory Views — What Each Agent Sees:**

| Agent | Memory Types | Confidence Filter | Content |
|---|---|---|---|
| Planner | Semantic only | None | Truncated (300 chars) |
| Worker | Episodic + Semantic | None | Full content |
| Evaluator | Semantic only | ≥ 0.8 | Full content |

**Reliability Scoring Formula:**
```
score  = confidence_score
score += min(usage_count × 0.02, 0.2)     # reward: more usage = more trusted
score -= failure_count × 0.15              # penalty: failures reduce trust
score -= min(days_since_validated × 0.01, 0.2)  # decay: stale memories lose trust
score  = clamp(score, 0.0, 1.0)
```

**Lifecycle State Machine:**
```
                ┌──────────────────────────────────────────────┐
                │                                              │
  reliability   │    ≥0.8        0.5–0.8         0.3–0.5       │  <0.3
  ──────────────┤─────────────┬──────────────┬─────────────┬───┤──────────
                │  "semantic"  │  (unchanged)  │   "stale"   │  │ "archived"
                │             │              │             │  │
                └──────────────────────────────────────────────┘
                
  failure_count ≥ 3  →  "deprecated"  (overrides all above)
```

---

### 7.5 `engine/` — The Production Facade

| File | Key Export | Purpose |
|---|---|---|
| `memory_engine.py` | `MemoryEngine` | High-level API combining store operations into two modes: **task-scoped** and **conversation history** |

**Two Usage Modes:**

1. **Task-Scoped Mode** — `load(session_id)` / `save(session_id, records)`:
   - Stores records as `MemoryItem`s linked to a `session_id` via `task_ids`
   - Each record has `input` and `output` fields

2. **Conversation Mode** — `load_history(session_id)` / `save_turn(session_id, human, assistant)`:
   - Stores Human/Assistant turns as JSON blobs
   - Returns formatted conversation strings for LLM prompts
   - LangChain-compatible

**Also provides:**
- `retrieve(session_id, query)` — basic keyword search within a session
- `repair_if_needed(records)` — lightweight cleanup (removes duplicates and empty entries)

---

### 7.6 `integrations/` — Third-Party Adapters

| File | Key Export | Purpose |
|---|---|---|
| `langchain.py` | `cocortex_langchain_memory()` | Factory function that creates a `CoCortexMemory` instance wired to a `MemoryEngine` |

**Usage:**
```python
from integrations.langchain import cocortex_langchain_memory

memory = cocortex_langchain_memory(session_id="my-session")
# memory.load_memory_variables({})  → {"history": "Human: ...\nAssistant: ..."}
# memory.save_context(inputs, outputs)
```

---

## 8. Database

CoCortex uses a single **SQLite database** (`cocortex_memory.db`) with one table.

### `memories` Table Schema

| Column | Type | Purpose |
|---|---|---|
| `id` | TEXT (UUID) | Primary key — unique memory identifier |
| `content` | TEXT | The actual memory content (natural language text) |
| `memory_type` | TEXT | `"episodic"` (event-based) or `"semantic"` (general knowledge) |
| `source_agent` | TEXT | Which agent created this: `planner`, `worker`, `evaluator`, or `memory_manager` |
| `timestamp` | TEXT (ISO) | When the memory was created |
| `confidence` | REAL | Trust score from 0.0 to 1.0 |
| `status` | TEXT | `"active"` or `"quarantined"` |
| `influenced_decisions` | TEXT (JSON) | List of decision IDs this memory influenced |
| `usage_count` | INTEGER | How many times this memory was used in a decision |
| `failure_count` | INTEGER | How many times a decision using this memory failed |
| `last_validated_at` | TEXT (ISO) | Last time this memory was verified as correct |
| `lifecycle_state` | TEXT | Current state: `episodic`, `semantic`, `stale`, `deprecated`, `archived` |
| `repair_history` | TEXT (JSON) | List of timestamped repair event messages |
| `task_ids` | TEXT (JSON) | List of task/session IDs this memory is linked to |

### Example Record

```json
{
  "id": "a3f8c2d4-1234-5678-9abc-def012345678",
  "content": "Photosynthesis converts CO2 and water into glucose using sunlight.",
  "memory_type": "semantic",
  "source_agent": "worker",
  "timestamp": "2026-02-27T10:30:00",
  "confidence": 0.85,
  "status": "active",
  "influenced_decisions": ["planner_b4c9e2c7...", "evaluator_d5e0f3d8..."],
  "usage_count": 5,
  "failure_count": 0,
  "last_validated_at": "2026-02-27T11:00:00",
  "lifecycle_state": "semantic",
  "repair_history": [],
  "task_ids": ["session-001", "session-003"]
}
```

### Schema Migration

The `MemoryStore._migrate_schema()` method handles backward compatibility. New columns (`usage_count`, `failure_count`, `lifecycle_state`, etc.) are added via `ALTER TABLE` if they don't exist. This means the database auto-upgrades when you run a newer version of CoCortex against an older database file.

---

## 9. API Reference (Internal Python API)

CoCortex is a library, not a web service. Below are the key programmatic interfaces:

### MemoryStore (memory/store.py)

| Method | Input | Output | What It Does |
|---|---|---|---|
| `add_memory(item)` | `MemoryItem` | — | Insert a new memory into SQLite |
| `get_memory(id)` | `UUID` | `MemoryItem \| None` | Retrieve a single memory by ID |
| `get_memory_by_type(type)` | `"episodic" \| "semantic"` | `List[MemoryItem]` | Get all active memories of a specific type |
| `get_all_active_memories()` | — | `List[MemoryItem]` | All memories with status = `active` |
| `get_quarantined_memories()` | — | `List[MemoryItem]` | All memories with status = `quarantined` |
| `update_confidence(id, score)` | `UUID, float` | — | Update confidence, recalculate lifecycle |
| `update_status(id, status)` | `UUID, str` | — | Set `active` or `quarantined` |
| `mark_memory_used(id)` | `UUID` | — | Increment `usage_count`, recalculate lifecycle |
| `mark_memory_failed(id)` | `UUID` | — | Increment `failure_count`, recalculate lifecycle |
| `validate_memory(id)` | `UUID` | — | Update `last_validated_at`, recalculate lifecycle |
| `promote_memory(id)` | `UUID` | — | Change type from `episodic` → `semantic` |
| `link_memory_to_decision(id, decision_id)` | `UUID, str` | — | Track causal influence |
| `link_memory_to_task(id, task_id)` | `UUID, str` | — | Associate memory with a session/task |
| `log_repair_event(id, message)` | `UUID, str` | — | Append timestamped event to `repair_history` |
| `clear_all_memories()` | — | — | Delete everything (for testing) |

### Consensus Engine (consensus/engine.py)

| Function | Input | Output |
|---|---|---|
| `run_consensus(votes, proposal)` | `List[Vote], MemoryProposal` | `(decision, memory_type, confidence)` where decision is `"accept"`, `"quarantine"`, or `"reject"` |

### Repair System (memory/repair.py)

| Function | Input | Output |
|---|---|---|
| `trace_suspect_memories(store, decision_id)` | `MemoryStore, str` | `List[MemoryItem]` — memories that influenced the failed decision |
| `repair_memories(store, decision_id, verifier)` | `MemoryStore, str, MemoryVerifier` | `List[MemoryItem]` — repaired memories (quarantined or downranked) |
| `decide_repair_action(verification, confidence, failure_count)` | `str, float, int` | `"quarantine"`, `"downrank"`, or `"none"` |

---

## 10. Common Workflows

### Workflow 1: Normal Task Execution

```python
from core.llm_client import LLMClient
from agents.planner import PlannerAgent
from agents.worker import WorkerAgent
from agents.evaluator import EvaluatorAgent
from agents.memory_manager import MemoryManagerAgent

llm = LLMClient()
manager = MemoryManagerAgent()

planner  = PlannerAgent(llm, manager.store)
worker   = WorkerAgent(llm, manager.store)
evaluator = EvaluatorAgent(llm, manager.store)

# 1. Plan
plan, plan_decision = planner.plan("Explain photosynthesis")

# 2. Execute
output, work_decision = worker.execute(plan)

# 3. Evaluate
evaluation, eval_decision = evaluator.evaluate(output)

# 4. Store memory (goes through consensus)
status, memory = manager.process_output(output, "worker", {"task": "biology"})
# status = "ACCEPTED", "QUARANTINED", or "REJECTED"
```

### Workflow 2: Self-Healing After Bad Memory

```python
from memory.repair import repair_memories
from memory.verification import MemoryVerifier

# A decision failed — the evaluator flagged incorrect output
failed_decision_id = eval_decision  # from the failed run

# Trace back and repair
verifier = MemoryVerifier(llm)
repaired = repair_memories(store, failed_decision_id, verifier)

# Each repaired memory is now quarantined or downranked
```

### Workflow 3: Using CoCortex with LangChain

```python
from integrations.langchain import cocortex_langchain_memory

memory = cocortex_langchain_memory(session_id="user-123")

# Load conversation history
history = memory.load_memory_variables({})
# {"history": "Human: What is DNA?\nAssistant: DNA is..."}

# Save a new turn
memory.save_context(
    {"input": "What is RNA?"},
    {"output": "RNA is a single-stranded nucleic acid..."}
)
```

---

## 11. Experiments

The `experiments/` directory contains progressive demo scripts that were built as the project evolved:

| Script | What It Demonstrates |
|---|---|
| `step01_demo.py` | Basic agent pipeline: Plan → Execute → Evaluate → Store |
| `step02_demo.py` | Memory store integration |
| `step03_demo.py` | Consensus-based memory admission |
| `step04_demo.py` | Role-specialized memory views |
| `step05_demo.py` | Causal influence logging |
| `step06_demo.py` | Causal traceback and memory repair |
| `step07_demo.py` | **MVP Demo** — Full self-healing loop: seed bad memory → run → fail → repair → run again → succeed |
| `step08_demo.py` | Memory reliability scoring and lifecycle |
| `step09_demo.py` | Memory provenance and explainability |
| `step10_demo.py` | **Baseline vs CoCortex** comparison experiment |
| `step11_demo.py` | Framework packaging + LangChain integration |

Run any experiment with:
```bash
python -m experiments.step07_demo
```

---

## 12. Testing

### Running Tests

```bash
# All tests
python -m pytest tests/ -v

# With coverage
python -m pytest tests/ -v --cov=.

# Single test file
python -m pytest tests/test_consensus.py -v
```

### Test Inventory

| Test File | What It Covers |
|---|---|
| `test_consensus.py` | Consensus engine + all three voters (approval, rejection, quarantine, safety) |
| `test_store.py` | MemoryStore CRUD, promotion, quarantining, decision linking |
| `test_scoring.py` | Reliability score computation (usage boost, failure penalty, time decay) |
| `test_scoring_lifecycle.py` | Lifecycle state transitions based on score changes |
| `test_repair.py` | Causal traceback, repair action decisions, end-to-end repair flow |
| `test_memory_engine.py` | MemoryEngine save/load, conversation history, retrieval, deduplication |

> **Note:** Tests use in-memory SQLite databases (`:memory:` or temporary paths), so they don't affect your development database.

---

## 13. Debugging Guide

### Where Logs Are

CoCortex does not use a formal logging framework yet. Debug output is via `print()` statements in:
- `ProvenanceEngine.explain_memory()` — prints a full memory audit
- `ProvenanceEngine.trace_failure()` — prints which memories caused a task failure
- Experiment scripts — print intermediate results

### How to Trace a Bug

1. **Identify the failed decision_id** — every agent returns a `decision_id` with its output
2. **Use `ProvenanceEngine.trace_failure(task_id)`** to find which memories influenced that decision
3. **Use `ProvenanceEngine.explain_memory(memory_id)`** to inspect a suspicious memory's full history
4. **Check the `repair_history`** field — it logs all past repair events with timestamps
5. **Check `lifecycle_state`** — if a memory is `deprecated` or `archived`, it was reliability-degraded
6. **Inspect the SQLite database directly** if needed:
   ```bash
   sqlite3 cocortex_memory.db
   .headers on
   .mode column
   SELECT id, content, status, confidence, lifecycle_state, failure_count FROM memories;
   ```

### Common Failure Points

| Issue | Likely Cause | Fix |
|---|---|---|
| `ValueError: GROQ_API_KEY not found` | Missing `.env` file or empty key | Copy `.env.example` to `.env` and add your key |
| Memories always rejected | Consensus requires ≥2 approvals — content might be too short or vague | Check voter criteria in `voters.py` |
| Agent outputs are wrong despite correct memories | Memory views may not include the relevant memory type | Check `views.py` — Planner only sees semantic, Evaluator requires confidence ≥0.8 |
| Database errors after code update | New columns not migrated | Delete `cocortex_memory.db` and restart, or check `_migrate_schema()` |
| `repair_memories()` finds no suspects | The `influenced_decisions` list on the memory doesn't contain the failed decision ID | Verify causal linking is happening in the agent code |

---

## 14. Contribution Guide

### Where New Features Should Go

| Feature Type | Directory | Example |
|---|---|---|
| New agent type | `agents/` | A `ResearcherAgent` that searches external sources |
| New voting strategy | `consensus/voters.py` | An `evidence_voter()` that checks citations |
| New memory capability | `memory/` | Vector similarity search in `memory/retrieval.py` |
| New integration | `integrations/` | A `crewai.py` adapter for CrewAI |
| New experiment | `experiments/` | `step12_demo.py` |
| New test | `tests/` | `test_<module>.py` |

### Naming Conventions

- **Files:** `snake_case.py` — e.g., `memory_manager.py`, `llm_client.py`
- **Classes:** `PascalCase` — e.g., `MemoryStore`, `PlannerAgent`
- **Functions:** `snake_case` — e.g., `run_consensus()`, `compute_reliability()`
- **Constants:** `UPPER_SNAKE_CASE` — e.g., `DB_PATH`, `CONV_PREFIX`
- **Test files:** `test_<module>.py` — e.g., `test_consensus.py`
- **Experiment files:** `step<NN>_demo.py` — numbered sequentially

### Things Developers Must NOT Break

> [!CAUTION]
> These invariants are critical to the system's reliability guarantees.

1. **Consensus must run before any memory is stored** — never bypass `run_consensus()` to insert memories directly
2. **The safety voter must always have veto power** — any `risk=True` vote must trigger quarantine
3. **Causal influence tracking must not be removed** — `link_memory_to_decision()` is called in every agent; removing it breaks the repair system
4. **Memory views must remain role-specific** — the Planner should never see raw episodic memories; the Evaluator should never see low-confidence memories
5. **The `MemoryItem` schema must stay backward-compatible** — always use `_migrate_schema()` for new fields, never drop columns
6. **Tests must pass** — run `python -m pytest tests/ -v` before every commit
7. **The `.env` file must never be committed** — API keys stay local

---

## 15. Research Foundation

CoCortex is grounded in academic research. The `Papers/` directory contains 12 referenced papers organized into four categories:

| Category | Key Insight for CoCortex |
|---|---|
| **Core Memory Architectures** | Memory contamination, drift, and retrieval errors are real problems in multi-agent systems |
| **Multi-Agent Coordination** | Agents need perception → memory → action cycles, but current systems lack memory verification |
| **Shared Memory** | Shared memory improves coordination, but gets corrupted easily without verification |
| **Trust & Authenticity** | Agents need trust models — communication without trust breaks multi-agent systems |

The core innovation of CoCortex is **verified shared memory with consensus-based admission and self-healing repair** — a gap identified across all four research categories.
