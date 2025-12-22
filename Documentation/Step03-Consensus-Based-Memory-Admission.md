# Step 03 : Consensus Based Memory Admission

How this is going to work for now : 
- Agent finishes reasoning / execution
- Memory proposal is created
- Proposal is sent to the voters
    - Planner view
    - worker view
    - rule-based checker
- Consensus decision happens

There are four possible outcomes for now : 
- Accept (weak / new) - stored as episodic memory
- Accept (Strong / validated) - stored directly as semantic memory
- Risky / inconsistent - Stored as quarantined memory
- Useless / redundant - Rejected

Only semantic happens, in two ways : 
- Path 1 : Direct sematic admission (rare)
    - used when :
        - Multiple voters strongly approve
        - High confidence
        - Low risk
        - Clearly reusable
- Path 2 : Promotion later (common, safer)
    - Intially accepted a episodic
    - Reused successfully multiple times
    - Never causes failures

## Things we have to do :

- Memory Proposal (Mandatory)
- Vote Schema (Mandatory)
- Implement Voters(3 only)
- Consensus Engine (Deterministic)
- Memory Manager Integration
- Minimal Demo

## Tech Stack

No need for change we can you the current one itself

## Documentation

This is for my use to trace back : 




