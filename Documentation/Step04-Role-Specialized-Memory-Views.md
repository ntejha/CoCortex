# Step 04 - Role Specialized Memory Views

## Goal

We should have : 
- One shared memory store
- Different read views of the same memory for each agent
- No agent accessing raw memory directly
- Clear seperation between experience, procedure and knowledge

## Tech Stack

- Existing stack

## Deliverables

- Memory View Module
    - Required funtions : 
        - get_planner_view(store)
        - get_worker_view(store)
        - get_evaluator_view(store)
- Planner Memory View 
- Worker Memory View
- Evaluator Memory View
- Agent Integration
- Minimal Demo


## Documentation

For me to trace back the code : 
    - Basically, we have modified the agents codes to make sure the agent can read the exisiting data in the database before doing. `evaluator.py`, `planner.py` and `worker.py` having it owns rules to views the data and how to view the data also.
    - We have added functions on how to view the data for each agent through the functions in `memory/views.py`