# CoCortex — Complete Codebase Guide

> **For:** New contributors, interns, and anyone trying to understand the system end-to-end.
> **Goal:** After reading this, you should be able to run the project, trace any bug, and extend any module without asking basic questions.

---

## 1. What Is CoCortex?

CoCortex is a **Python research framework** that gives a team of AI agents a **shared, reliable memory**.

The core problem it solves: when multiple AI agents share knowledge, bad information spreads. A wrong "fact" accepted by one agent gets reused by others, causing a cascade of failures. CoCortex prevents this through:

1. **Consensus-gated admission** — a memory can only enter the shared store if at least 2 of 3 independent voters approve it.
2. **Causal influence tracking** — every agent decision records which memories influenced it.
3. **Self-healing repair** — when a decision fails, the system traces back to the memories that caused it and quarantines or downranks them.
4. **Lifecycle management** — memories age, build trust through successful reuse, and are automatically retired when they fail too many times.
5. **Rehabilitation** — quarantined memories that stabilise after repair can be restored to active status.

---

## 2. Non-Technical Overview

Think of it like a **company knowledge base with a review board**.

- An employee (agent) finishes a task and proposes adding something to the knowledge base.
- Three reviewers vote: "Is this reusable? Is it actionable? Is it safe?"
- If 2 of 3 say yes, it's added. If it's flagged as dangerous, it goes to quarantine.
- Over time, entries that keep helping get promoted (high trust). Entries that cause mistakes get automatically downgraded and eventually archived.
- If a project fails, the system asks "which knowledge entries influenced the decision that went wrong?" and reviews those entries specifically.

---

## 3. Repository Structure

```
cocortex/
├── agents/              # The four AI agents
│   ├── planner.py       # Breaks tasks into steps
│   ├── worker.py        # Executes steps
│   ├── evaluator.py     # Checks correctness of output
│   └── memory_manager.py# Runs consensus and stores approved memories
│
├── consensus/           # Memory admission control
│   ├── schemas.py       # MemoryProposal and Vote data models
│   ├── voters.py        # 3 voters (planner, worker, safety)
│   └── engine.py        # Aggregates votes into a decision
│
├── core/                # Shared infrastructure
│   ├── config.py        # Env var loading and validation
│   ├── llm_client.py    # Groq API wrapper with retry + LLM_UNAVAILABLE
│   └── decision.py      # Utility: generate_decision_id()
│
├── memory/              # The heart of the system
│   ├── schemas.py       # MemoryItem — the core data model
│   ├── store.py         # SQLite CRUD layer (thread-safe)
│   ├── views.py         # Role-filtered views: planner / worker / evaluator
│   ├── scoring.py       # compute_reliability() + ScoringConfig
│   ├── lifecycle.py     # Lifecycle state machine (episodic→semantic→stale→…)
│   ├── repair.py        # Causal traceback, quarantine, downrank, rehabilitate
│   ├── verification.py  # LLM-based fact checker
│   ├── provenance.py    # Audit trail and failure tracing
│   └── cocortex_memory.py  # LangChain-compatible memory adapter
│
├── engine/
│   └── memory_engine.py # High-level facade (task-scoped + conversation mode)
│
├── integrations/
│   └── langchain.py     # Factory: cocortex_langchain_memory()
│
├── tests/               # Pytest suite (~160 tests)
├── Documentation/       # Step-by-step dev notes per feature
├── Papers/              # Research papers that informed design decisions
├── .env.example         # Template for required environment variables
├── pyproject.toml       # Build config and dependencies
├── explain.md           # This file
└── evaluation.md        # Project evaluation: metrics, experiments, gaps
```

---

## 4. Tech Stack

| Layer | Technology | Why |
|---|---|---|
| Language | Python 3.10+ | Type hints, match statements |
| LLM Provider | Groq API (Llama 3.1-8b-instant) | Fast inference, free tier |
| LLM Framework | LangChain | Standard chain integration |
| Data Validation | Pydantic v2 | Schema enforcement at boundaries |
| Database | SQLite (local file) | Zero-config persistence |
| Env Config | python-dotenv | `.env` file support |
| Testing | pytest | Unit + integration tests |

---

## 5. Setup

### Prerequisites
- Python 3.10+
- A Groq API key → [console.groq.com](https://console.groq.com)

### Steps

```bash
# 1. Clone and enter project
git clone <repo> && cd cocortex

# 2. Create a virtual environment
python -m venv venv && source venv/bin/activate

# 3. Install dependencies
pip install -e .

# 4. Configure environment
cp .env.example .env
# Edit .env and set your GROQ_API_KEY

# 5. Verify setup
python -c "from core.config import validate_config; validate_config(); print('OK')"

# 6. Run tests
python -m pytest tests/ -v
```

### Environment Variables (`.env`)

| Variable | Required | Default | Description |
|---|---|---|---|
| `GROQ_API_KEY` | ✅ Yes | — | Your Groq API key |
| `LLM_MODEL` | No | `llama-3.1-8b-instant` | Model name to use |
| `COCORTEX_DB_PATH` | No | `cocortex_memory.db` | SQLite db file path |

---

## 6. Data Flow — Step by Step

```
User Task
    │
    ▼
PlannerAgent.plan(task)
  - Fetches semantic memories via get_planner_view()
  - Calls LLM to generate a step-by-step plan
  - Second LLM call: "Which memory indices did you use?" → attribution
  - Only attributed memories are linked to this decision_id
    │
    ▼
WorkerAgent.execute(plan)
  - Fetches episodic + semantic via get_worker_view()
  - Calls LLM to execute the plan
  - Same attribution pattern → links used memories to decision_id
    │
    ▼
EvaluatorAgent.evaluate(output)
  - Fetches high-confidence semantic via get_evaluator_view()
  - Calls LLM to check correctness
  - Returns (result, decision_id)
    │
    ├── if PASS → repair_on_success() → rehabilitate borderline quarantined memories
    │
    └── if FAIL → MemoryManagerAgent.handle_failure(decision_id)
                    → trace_suspect_memories(decision_id)
                    → MemoryVerifier.verify(each suspect)
                    → decide_repair_action() → quarantine / downrank / none
    │
    ▼
MemoryManagerAgent.process_output(output)
  - Deduplication check (exact content match against active memories)
  - Creates MemoryProposal
  - Gets 3 votes: planner_voter, worker_voter, rule_based_voter
  - run_consensus() → accept / quarantine / reject
  - Stores result in MemoryStore (SQLite)
```

---

## 7. Module Deep-Dives

### 7.1 `memory/schemas.py` — MemoryItem

The central data model. Every piece of knowledge is a `MemoryItem`:

```python
class MemoryItem(BaseModel):
    id: UUID                      # Auto-generated unique ID
    content: str                  # The knowledge text
    memory_type: str              # "episodic" or "semantic"
    source_agent: str             # Who created it
    timestamp: datetime           # When created
    confidence_score: float       # 0.0–1.0, set at admission
    status: str                   # "active" or "quarantined"
    influenced_decisions: list    # Decision IDs this memory contributed to
    usage_count: int              # How many times successfully used
    failure_count: int            # How many failures it was linked to
    last_validated_at: datetime   # Last LLM verification timestamp
    lifecycle_state: str          # episodic → semantic → stale → deprecated → archived
    repair_history: list          # Log of all repair events
    task_ids: list                # Session/task IDs this memory belongs to
```

**Episodic** = single-event traces ("I executed step 3 and it worked").  
**Semantic** = general, reusable knowledge ("Photosynthesis requires sunlight").

---

### 7.2 `memory/store.py` — MemoryStore

Thread-safe SQLite wrapper. All writes go through `_write_lock` (a `threading.Lock`).

**Key methods:**

| Method | What it does |
|---|---|
| `add_memory(item)` | INSERT new memory row |
| `get_memory(id)` | Fetch single memory by UUID |
| `get_all_active_memories()` | All memories where `status='active'` |
| `get_quarantined_memories()` | All quarantined memories |
| `get_memories_by_session(session_id)` | Active memories for a session |
| `update_memory(id, fields)` | Update arbitrary fields — **whitelisted only** |
| `update_confidence(id, score)` | Set new confidence, recalculate lifecycle |
| `update_status(id, status)` | Set `'active'` or `'quarantined'` |
| `mark_memory_used(id)` | Increment `usage_count`, recalculate lifecycle |
| `mark_memory_failed(id)` | Increment `failure_count`, recalculate lifecycle |
| `link_memory_to_decision(id, decision_id)` | Append to `influenced_decisions` |
| `link_memory_to_task(id, task_id)` | Append to `task_ids` |
| `log_repair_event(id, message)` | Append timestamped message to `repair_history` |
| `delete_by_session(session_id)` | Hard-delete rows belonging to a session |
| `validate_memory(id)` | Update `last_validated_at`, recalculate lifecycle |

> **Security note:** `update_memory()` checks every field name against `_ALLOWED_UPDATE_FIELDS`. Passing an unknown field name raises `ValueError` — this prevents SQL injection via dynamic column names.

---

### 7.3 `memory/scoring.py` — Reliability Formula

```python
reliability = confidence_score
            + min(usage_count × 0.02, +0.20)    # usage bonus, capped at 10 uses
            - failure_count × 0.15              # failure penalty
            - min(days_stale × 0.01, -0.20)    # time decay, capped at 20 days
            = clamp(result, 0.0, 1.0)
```

You can override parameters via `ScoringConfig`:

```python
from memory.scoring import compute_reliability, ScoringConfig

config = ScoringConfig(usage_reward=0.05, usage_cap=0.5)
score = compute_reliability(mem, config=config)
```

**`last_validated_at` takes priority over `timestamp` for staleness** — a recently validated memory doesn't decay even if it's old.

---

### 7.4 `memory/lifecycle.py` — Lifecycle State Machine

```
episodic  ──(reliability ≥ 0.8)──▶  semantic
    │                                    │
    │         (reliability 0.3–0.5)      │
    └──────────────────────────▶  stale  │
                                         │
                (reliability < 0.3)      ▼
    ┌──────────────────────────── archived
    │
    └── (failure_count ≥ 3) ──▶  deprecated  (permanent)
```

**Important:** Quarantined memories are **never promoted** — even if their reliability score is high. They stay at their current lifecycle state until rehabilitated or archived.

---

### 7.5 `memory/views.py` — Role-Specific Access

Each agent sees a different filtered slice of memory:

| View | Content | Lifecycle Filter | Confidence Filter |
|---|---|---|---|
| `get_planner_view()` | Semantic only | Excludes stale/deprecated/archived | None |
| `get_worker_view()` | Episodic + semantic | Excludes archived | None |
| `get_evaluator_view()` | Semantic only | Excludes stale/deprecated/archived | ≥ 0.8 only |

All views support optional arguments:
- `query="photosynthesis sunlight"` — keyword-overlap ranking
- `top_n=5` — limit to top N results
- Content in planner view is **truncated to 300 chars** to save tokens.

---

### 7.6 `consensus/` — Memory Admission Gate

**Schema** (`consensus/schemas.py`):

```python
class MemoryProposal(BaseModel):
    content: str
    source_agent: str             # "planner" | "worker" | "evaluator" | "memory_manager"
    suggested_type: str           # "episodic" | "semantic"
    context: dict

class Vote(BaseModel):
    approve: bool
    confidence: float             # Must be 0.0–1.0 (Pydantic validated)
    risk: bool                    # True = veto power → forces quarantine
    reason: str
```

**Three Voters** (`consensus/voters.py`):

| Voter | Question | LLM support |
|---|---|---|
| `planner_voter` | Is this general enough to reuse in future plans? | ✅ With JSON fallback |
| `worker_voter` | Is this concrete and actionable for execution? | ✅ With JSON fallback |
| `rule_based_voter` | Is this safe and not misinformation? | ❌ Always deterministic |

When an LLM is provided to `planner_voter` or `worker_voter`, it asks the LLM to vote in JSON: `{"approve": bool, "confidence": float, "reason": "..."}`. If the response is unparseable or `LLM_UNAVAILABLE`, it silently falls back to the heuristic rules.

**Consensus Engine** (`consensus/engine.py`):

```
Requires exactly 3 voters (raises ValueError otherwise)

Any risk=True vote → quarantine (safety voter has veto power)
≥ 2 approvals     → accept (confidence = avg of ALL 3 votes, including dissenters)
< 2 approvals     → reject
```

> The weighted average across ALL voters means strong dissent lowers the accepted confidence score, reducing the memory's initial trust level.

---

### 7.7 `memory/repair.py` — Self-Healing

**Failure path (causal traceback):**

```python
# Called when evaluator marks a decision as failed
repair_memories(store, failed_decision_id, verifier)
```

1. `trace_suspect_memories()` — scans ALL memories (active + quarantined) whose `influenced_decisions` includes `failed_decision_id`
2. `MemoryVerifier.verify(content)` — asks LLM: is this correct/incorrect/uncertain?
3. `decide_repair_action(verification, confidence, failure_count)`:
   - `incorrect` → quarantine
   - `uncertain` + confidence < 0.6 → downrank (−0.2)
   - failure_count ≥ 2 → downrank
   - otherwise → none
4. `log_repair_event()` — adds timestamped entry to `repair_history`

**Success path (rehabilitation):**

```python
# Called when evaluator marks a decision as passed
repair_on_success(store, success_decision_id)
```

Reviews all quarantined memories. If `failure_count < 3` and `reliability > 0.3`, calls `rehabilitate_memory()`:
- Sets `status = 'active'`
- Reduces confidence by 20% (penalty for having been quarantined)
- Logs the rehabilitation event

---

### 7.8 `core/llm_client.py` — LLMClient

```python
LLM_UNAVAILABLE = "__LLM_UNAVAILABLE__"  # Sentinel returned on failure
MAX_RETRIES = 3                           # Attempts before giving up
```

**Retry policy:**
- Transient errors (network timeout, rate limit) → retry up to 3x with exponential backoff
- Permanent errors (401 Unauthorized, 403 Forbidden) → return `LLM_UNAVAILABLE` immediately (no retry)
- After 3 failed retries → return `LLM_UNAVAILABLE`

All callers check for `LLM_UNAVAILABLE` and degrade gracefully (verifier returns `"uncertain"`, voters fall back to heuristic).

---

### 7.9 `agents/` — Attribution-Guided Tracking

All three agents (Planner, Worker, Evaluator) use the same two-call pattern:

```python
# Call 1: Generate output using memory context
output = llm.generate(f"Relevant knowledge:\n{memory_text}\n\nTask: {task}")

# Call 2: Attribute — which memories did you actually use?
indices_json = llm.generate(
    f"Memory items:\n{numbered_memory_list}\n\n"
    "Which indices (0-based) influenced your response? Reply with JSON array only, e.g. [0, 2]"
)
```

Only the attributed indices are linked via `store.link_memory_to_decision()`. If the attribution call fails to parse, **no memories are linked** — a safe fallback that avoids false positives in the causal traceback.

---

## 8. Database Schema

**Table: `memories`**

| Column | Type | Description |
|---|---|---|
| `id` | TEXT (UUID) | Primary key |
| `content` | TEXT | The memory text |
| `memory_type` | TEXT | `episodic` or `semantic` |
| `source_agent` | TEXT | Agent that created it |
| `timestamp` | TEXT | ISO datetime of creation |
| `confidence` | REAL | 0.0–1.0 trust score |
| `status` | TEXT | `active` or `quarantined` |
| `influenced_decisions` | TEXT | JSON array of decision IDs |
| `usage_count` | INTEGER | Successful uses |
| `failure_count` | INTEGER | Failure-linked uses |
| `last_validated_at` | TEXT | ISO datetime of last LLM verification |
| `lifecycle_state` | TEXT | Current lifecycle stage |
| `repair_history` | TEXT | JSON array of repair event strings |
| `task_ids` | TEXT | JSON array of session/task IDs |

**Schema migrations** are additive-only — new columns are added via `ALTER TABLE IF NOT EXISTS`. This means you can safely update the code without losing existing data.

---

## 9. LangChain Integration

CoCortex plugs into any LangChain chain as a drop-in memory:

```python
from integrations.langchain import cocortex_langchain_memory
from langchain_groq import ChatGroq
from langchain.chains import ConversationChain

llm = ChatGroq(model="llama-3.1-8b-instant")
memory = cocortex_langchain_memory(session_id="my-session", db_path="my.db")

chain = ConversationChain(llm=llm, memory=memory)
chain.predict(input="What is photosynthesis?")
```

`CoCortexMemory` implements the LangChain `BaseMemory` duck-type interface:
- `load_memory_variables()` → returns `{"history": "Human: ...\nAssistant: ..."}`
- `save_context()` → saves turn to SQLite
- `clear()` → deletes all session records from SQLite (actually deletes, not a no-op)

---

## 10. Common Workflows

### Run the test suite
```bash
python -m pytest tests/ -v --tb=short
# Expected: 160 passed
```

### Add a new memory manually
```python
from memory.store import MemoryStore
from memory.schemas import MemoryItem

store = MemoryStore("cocortex_memory.db")
mem = MemoryItem(
    content="Neural networks learn via gradient descent.",
    memory_type="semantic",
    source_agent="worker",
    confidence_score=0.85,
)
store.add_memory(mem)
```

### Inspect a memory's reliability
```python
from memory.store import MemoryStore
from memory.scoring import compute_reliability
from memory.provenance import ProvenanceEngine

store = MemoryStore()
engine = ProvenanceEngine(store)

# Get a full audit report for one memory
report = engine.explain_memory(some_memory_id)
print(report)
# {'memory_id': '...', 'content': '...', 'reliability': 0.82, 'failure_count': 1, ...}
```

### Trace a failed decision
```python
from memory.repair import trace_suspect_memories

suspects = trace_suspect_memories(store, "evaluator_abc123")
for s in suspects:
    print(s.content, s.confidence_score)
```

### Manually trigger repair
```python
from memory.repair import repair_memories
from memory.verification import MemoryVerifier
from core.llm_client import LLMClient

verifier = MemoryVerifier(LLMClient())
repaired = repair_memories(store, "evaluator_fail001", verifier)
```

---

## 11. Debugging Guide

### Memory not appearing in agent views
1. Check `status` — quarantined memories are excluded from all views.
2. Check `lifecycle_state` — `stale`, `deprecated`, `archived` are excluded from planner/evaluator views.
3. Check `memory_type` — planner and evaluator views only show `semantic` memories.
4. Check `confidence_score` — evaluator view requires ≥ 0.8.

### Memory always getting rejected by consensus
1. Check content length — planner voter rejects anything < 40 chars.
2. Check for question marks — worker voter rejects questions.
3. Check safety keywords — rule-based voter flags: `hack`, `exploit`, `bypass`, `circumvent`, `phishing`, `exfiltrate`, and others.
4. Add `logger.setLevel(logging.DEBUG)` to see which voter rejected it and why.

### Repair loop not quarantining bad memory
1. Check `influenced_decisions` — the failing `decision_id` must be in that list.
2. Check `trace_suspect_memories` returns something — it scans both active and quarantined memories.
3. Check `MemoryVerifier` return — if LLM says `"uncertain"` and confidence is high (≥ 0.6), action is `"none"`.

### `EnvironmentError: missing GROQ_API_KEY`
```bash
cp .env.example .env
# Add your key:  GROQ_API_KEY=gsk_...
```

### Enable verbose logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)
# Now consensus/voters.py, repair.py, provenance.py all emit detailed logs
```

---

## 12. Key Design Decisions

| Decision | Rationale |
|---|---|
| **Deterministic consensus rules** | No LLM needed for the admission decision — reproducible and fast. LLM is optional for individual votes only. |
| **Attribution tracking over full-view linking** | Linking only attributed memories reduces false positives in causal traceback — fewer memory misattributions. |
| **Safety voter never uses LLM** | Safety rules must be deterministic and consistent. An LLM safety check could be fooled or inconsistent. |
| **Rehabilitation reduced confidence** | Restored memories start at 80% of their previous confidence — they've earned some distrust by being quarantined. |
| **Exact JSON membership for session deletion** | `LIKE '%session-1%'` would match `session-10`. Python-side filtering avoids this. |
| **Weighted avg includes dissenters** | A strong dissenting vote should lower the accepted confidence. Averaging only approving votes would ignore valid concerns. |
| **SQLite only** | Zero-config, file-based, good enough for research prototype and single-machine multi-agent systems. |

---

## 13. Contribution Guidelines

1. **Never bypass the consensus gate** — don't call `store.add_memory()` directly from agents. Always go through `MemoryManagerAgent.process_output()`.

2. **Always use `MemoryStore` methods, not raw SQL** — raw SQL breaks the write lock and skips the field whitelist.

3. **Write tests for every new feature** — at minimum: one happy path, one edge case, one regression guard.

4. **Log, don't print** — use `logging.getLogger(__name__)` in every module. Tests capture logs; they can't capture prints.

5. **Keep voters deterministic unless LLM is explicitly passed** — the `llm=None` default must always work correctly without network access.

6. **For new memory fields**: add to `MemoryItem`, add `ALTER TABLE` migration in `_migrate_schema()`, add to `_to_memory()` parser, add to `_ALLOWED_UPDATE_FIELDS`.

---

## 14. Project Status

| Feature | Status |
|---|---|
| Consensus-based admission | ✅ Complete |
| Role-specialized memory views | ✅ Complete |
| Causal influence tracking (attribution) | ✅ Complete |
| Reliability scoring + lifecycle management | ✅ Complete |
| Self-healing repair | ✅ Complete |
| Rehabilitation mechanism | ✅ Complete |
| LangChain integration | ✅ Complete |
| Thread-safe store | ✅ Complete |
| LLM retry + graceful degradation | ✅ Complete |
| Vector/semantic search | ❌ Not yet — keyword only |
| Async agent pipeline | ❌ Not yet — synchronous only |
| REST API | ❌ Not yet |
| Dashboard UI | ❌ Not yet |
