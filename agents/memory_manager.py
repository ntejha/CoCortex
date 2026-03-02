from memory.store import MemoryStore
from memory.schemas import MemoryItem
from consensus.schemas import MemoryProposal
from consensus.voters import planner_voter, worker_voter, rule_based_voter
from consensus.engine import run_consensus

# Signals that suggest a memory is general/reusable (semantic) rather than
# a single-event trace (episodic). Used to auto-classify proposed memories
# so the planner/evaluator views (which only show semantic) are not always empty.
_SEMANTIC_SIGNALS = [
    "always", "never", "typically", "generally", "is defined as",
    "works by", "requires", "consists of", "is used for", "can be",
    "should be", "must be", "is a type of", "are a type of",
    "in general", "by definition", "is known as",
]


def _infer_memory_type(content: str) -> str:
    """
    Infer whether content is 'semantic' (general knowledge) or 'episodic'
    (single-event trace) based on keyword signals.

    Previously everything was hardcoded to 'episodic', which meant the planner
    and evaluator views (semantic-only) always returned empty lists.
    """
    if any(sig in content.lower() for sig in _SEMANTIC_SIGNALS):
        return "semantic"
    return "episodic"


class MemoryManagerAgent:
    def __init__(self, llm=None, store: MemoryStore = None):
        # Accept an injected store so all agents share the same instance.
        # Previously always created MemoryStore() unconditionally, causing
        # split-state when agents were given a custom or in-memory store.
        self.store = store if store is not None else MemoryStore()
        self.llm = llm

    def process_output(self, content: str, source_agent: str, context: dict):
        inferred_type = _infer_memory_type(content)

        proposal = MemoryProposal(
            content=content,
            source_agent=source_agent,
            suggested_type=inferred_type,
            context=context,
        )

        votes = [
            planner_voter(proposal, llm=self.llm),
            worker_voter(proposal, llm=self.llm),
            rule_based_voter(proposal),  # always deterministic
        ]

        decision, mem_type, confidence = run_consensus(votes, proposal)

        if decision == "accept":
            memory = MemoryItem(
                content=content,
                memory_type=mem_type,
                source_agent=source_agent,
                confidence_score=confidence,
            )
            self.store.add_memory(memory)
            return "ACCEPTED", memory

        if decision == "quarantine":
            memory = MemoryItem(
                content=content,
                memory_type=inferred_type,
                source_agent=source_agent,
                confidence_score=0.1,
                status="quarantined",
            )
            self.store.add_memory(memory)
            return "QUARANTINED", memory

        return "REJECTED", None

    def summary(self):
        return {
            "episodic": self.store.get_memory_by_type("episodic"),
            "semantic": self.store.get_memory_by_type("semantic"),
        }