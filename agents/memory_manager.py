from memory.store import MemoryStore
from memory.schemas import MemoryItem
from consensus.schemas import MemoryProposal
from consensus.voters import planner_voter, worker_voter, rule_based_voter
from consensus.engine import run_consensus

class MemoryManagerAgent:
    def __init__(self):
        self.store = MemoryStore()

    def process_output(self, content: str, source_agent: str, context: dict):
        proposal = MemoryProposal(
            content=content,
            source_agent=source_agent,
            suggested_type="episodic",
            context=context
        )

        votes = [
            planner_voter(proposal),
            worker_voter(proposal),
            rule_based_voter(proposal)
        ]

        decision, mem_type, confidence = run_consensus(votes, proposal)

        if decision == "accept":
            memory = MemoryItem(
                content=content,
                memory_type=mem_type,
                source_agent=source_agent,
                confidence_score=confidence
            )
            self.store.add_memory(memory)
            return "ACCEPTED", memory

        if decision == "quarantine":
            memory = MemoryItem(
                content=content,
                memory_type=proposal.suggested_type,
                source_agent=source_agent,
                confidence_score=0.1,
                status="quarantined"
            )
            self.store.add_memory(memory)
            return "QUARANTINED", memory

        return "REJECTED", None

    def summary(self):
        return {
            "episodic": self.store.get_memory_by_type("episodic"),
            "semantic": self.store.get_memory_by_type("semantic")
        }
