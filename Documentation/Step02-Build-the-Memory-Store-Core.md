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

#### **Classification Support**
- Implements a mechanism to transition memory items from episodic to semantic form through explicit state updates. This enables validated knowledge to be promoted for long-term use.

#### **Confidence & Status Handling**
- Allows dynamic adjustment of a memory’s confidence score and operational status without deletion. This supports future memory repair, quarantine, and self-healing mechanisms.

### **Tech Stack**

- SQLite (Storage Layer)
- Pydantic (Data modeling)
- Utilities :
    - uuid - unique memory ID
    - datetime - Accurate timestamps
    - json - Store influenced decision links

## Documentation

These are notes for me to trace back : 
