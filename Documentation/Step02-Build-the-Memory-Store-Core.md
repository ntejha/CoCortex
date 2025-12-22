# Step 02 : Build the Memory Store (Core Infrastructure)

## What we have to complete

We have to create the memory stucture needed for the further steps.

### Checklist

#### **Memory Schema (Pydantic Model)**
- Defines a structured and validated representation for each memory item, including content, provenance, confidence and casual links. This schema ensures consistency across all memory operations.

#### **SQLite Table Initialization**
- Initializes a persistent SQLite table to store all memory items with their associated metadata. This provides reliable long-term storage and supports efficient querying and updates.

#### **Memory Store API (Mandatory Methods)**
- Provides internal Python functions to add, retrieve, update, and link memory items programmatically. These APIs enable controlled interaction between agents and the memory store.

add_memory(memory_item),get_memory_by_type(memory_type) update_memory(memory_id, fields), link_memory_to_decision(memory_id, decision_id), get_memory(memory_id)

#### **Classification Support**
- Implements a mechanism to transition memory items from episodic to semantic form through explicit state updates. This enables validated knowledge to be promoted for long-term use.

promote_memory(memory_id)


#### **Confidence & Status Handling**
- Allows dynamic adjustment of a memory’s confidence score and operational status without deletion. This supports future memory repair, quarantine, and self-healing mechanisms.

update_confidence(memory_id, new_score)
update_status(memory_id, "quarantined")


### **Tech Stack**

- SQLite (Storage Layer)
- Pydantic (Data modeling)
- Utilities :
    - uuid - unique memory ID
    - datetime - Accurate timestamps
    - json - Store influenced decision links

## Documentation

These are notes for me to trace back : 
    - requirements.txt : `pydantic`
    - Pydantic data model in `schemas.py`
    - Functions we have wrote in `store.py` : 
        - _intialize_table(self)
        - add_memory(memory_item)
        - get_memory_by_type(memory_type)
        - update_memory(memory_id, fields) 
        - link_memory_to_decision(memory_id, decision_id)
        - get_memory(memory_id)
        - promote_memory(memory_id)
        - update_confidence(memory_id, new_score)
        - update_status(memory_id, "quarantined")
        - delete_memory
        - clear_all_memories
    - We have also written a test case.



