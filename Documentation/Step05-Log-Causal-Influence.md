# Step 05 - Log Causal Influence

## Tech Stack

- Existing tech

## What all needs to be done : 

- Generate a Globally Unique Decision ID
- Capture Which Memory Items Were Used
- Log Memory -> Decision Links
- Do This for all three agents
- Create a demo

## Documentation

This is there for me to trace back : 
- Basically, we have a function inside `core/decision.py`. This file has the function to create the decision unique id.
- We have modified the evaluator, planner and worker code to make sure first it searches memory in its view and if something is there it uses the memory id and links it to this decision id. So, this is how the log works.