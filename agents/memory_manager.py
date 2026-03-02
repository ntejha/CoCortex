import logging
from memory.store import MemoryStore
from memory.schemas import MemoryItem
from consensus.schemas import MemoryProposal
from consensus.voters import planner_voter, worker_voter, rule_based_voter
from consensus.engine import run_consensus

logger = logging.getLogger(__name__)

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
    """
    if any(sig in content.lower() for sig in _SEMANTIC_SIGNALS):
        return "semantic"
    return "episodic"


def _is_duplicate(content: str, store: MemoryStore) -> bool:
    """
    Return True if an active memory with exactly the same stripped content
    already exists in the store.

    This prevents the same statement being inserted multiple times across
    sequential runs, keeping the store clean without needing vector search.
    """
    needle = content.strip().lower()
    all_active = store.get_all_active_memories()
    return any(m.content.strip().lower() == needle for m in all_active)


class MemoryManagerAgent:
    def __init__(self, llm=None, store: MemoryStore = None):
        # Accept an injected store so all agents share the same instance.
        self.store = store if store is not None else MemoryStore()
        self.llm = llm

    def process_output(self, content: str, source_agent: str, context: dict):
        # --- Deduplication guard ---
        if _is_duplicate(content, self.store):
            logger.debug(
                "Duplicate memory skipped (source=%s): %.80s", source_agent, content
            )
            return "DUPLICATE", None

        inferred_type = _infer_memory_type(content)

        proposal = MemoryProposal(
            content=content,
            source_agent=source_agent,
            suggested_type=inferred_type,
            context=context,
        )

        # Voters are deterministic functions — no llm parameter needed
        votes = [
            planner_voter(proposal),
            worker_voter(proposal),
            rule_based_voter(proposal),
        ]

        decision, mem_type, confidence = run_consensus(votes, proposal)

        logger.info(
            "Consensus decision=%s type=%s confidence=%.2f (source=%s)",
            decision, mem_type, confidence, source_agent,
        )

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