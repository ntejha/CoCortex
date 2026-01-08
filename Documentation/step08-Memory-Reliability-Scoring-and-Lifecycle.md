# Step 08 - Memory Reliability Scoring and Lifecycle

Right now, memory is stored and optionally quarantined but after today. Memory will do the following : 
    - Age
    - gain or lose trust
    - move through lifecycle stages
    - self-demote based on failures

Changes that will occur : 
- Changes to MemoryItem Schema
    - Current : confidence_score and status(active / quarantined)
    - Updated : usage_count, failure_count, last_validated_at and lifecycle_state.
- New feature in memory module
    - It will convert raw metadata into a single trust number.
- New feature in memory module
    - It decides what happens to memory over time.
    - Based on reliability score, failure count and time since last validation. It decides to promote, demote or archive.
- Small Change to memory module
    - Add the new functions in MemoryItem
- Changes in the Exisiting Repair Logic.
- Demo to see it working.

Documentation : 

For me to trace back : 
- New Files
    - `memory/scoring.py` - To calculate the memory_reliability score.
    - `memory/lifecycle.py` - to automate the removal of data
    - `experiments/step08_demo.py`
- Modified Files
    - `memory/schemas.py` - for adding the new columns
    - `memory/store.py` - adding new columns to the db


Output : 

```
ntejha@fedora:~/Major_Project/CoCortex$ python -m experiments.step08_demo
[INFO] Memory store cleared

--- INITIAL STATE ---
Memory ID       : 8b7d058e-35c5-4e48-8ea1-ec18d43036b4
Lifecycle State : episodic
Usage Count     : 0
Failure Count   : 0
Reliability     : 0.9

--- AFTER USAGE (TRUST BUILDING) ---
Memory ID       : 8b7d058e-35c5-4e48-8ea1-ec18d43036b4
Lifecycle State : semantic
Usage Count     : 5
Failure Count   : 0
Reliability     : 1.0

--- AFTER FAILURES (DEMOTION) ---
Memory ID       : 8b7d058e-35c5-4e48-8ea1-ec18d43036b4
Lifecycle State : deprecated
Usage Count     : 5
Failure Count   : 3
Reliability     : 0.55

--- AFTER VALIDATION ---
Memory ID       : 8b7d058e-35c5-4e48-8ea1-ec18d43036b4
Lifecycle State : deprecated
Usage Count     : 5
Failure Count   : 3
Reliability     : 0.55

[INFO] Memory store cleaned

```

