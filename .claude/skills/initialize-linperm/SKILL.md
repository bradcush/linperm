---
name: initialize-linperm
description: Initializes a starting point for developing BiPerm and MulPerm
permutation arguments in Rust. Use when the user asks to "initialize a
linear-time permutations project", "create an initial codebase to develop
linear-time permutation arguments", and "add boilerplate for permutation
argument development in a Rust workspace".
---

# Initialize Permutations

## When to use

This skill should be invoked when starting a project in Rust to build a library
ipmlementing Linear-\*Time Permutation arguments. Before any code has been
written other then a "Hello World", using this skill helps to create a solid
base for development, including necessary helpers, libraries, and more.

## Instructions

Step-by-step guidance for initializing the project.

### Step 1: Research

The original paper is the best resource for documentation on what will be
developed and what might be a necessary dependency for easy development. Start
by reading the paper `2025-ltpc.pdf` saved locally. The same document can be
viewed online via `https://eprint.iacr.org/2025/1850`.

- Read the above resources in full, load them in your memory
- Read references related to initializing this project
- Find other project implementing permutation arguments in Rust
- Research a few of these projects to advise on organization/dependencies
- Decide on which constraint system to use, HyperPlonk is recommended
- Understand that constraint system and how to use it when developing
- Plan project organization and structure based on learnings

#### Planning guidelines

- Don't re-implement libraries we can import easily
- Only use what you consider to be "trusted" libraries
- The starting point for this project should be minimal and simple
- Don't implement the paper, provide what we need to efficiently do so
- If unsure about what direction to take for big decisions, ask me

#### PCS + Fields

- MockPCS: Start w/ storing polynomials in memory
  - Interfaces should be the same for all at callsite
- BN254 curve: Common, pairing-friendly, and right-size

### Step 2: Building

#### Crates

- `permcore`: Local permutation library
- `biperm`: BiPerm specific implemtation
- `mulperm`: MulPerm specific implementation

TBD

### Step 3: Checking

TBD

## Examples

Optional example invocations or expected behavior.
