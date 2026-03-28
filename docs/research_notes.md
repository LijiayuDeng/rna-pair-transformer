# Research Notes

## Current project framing

This is a small demonstration project, not a full research pipeline.

The main objective is to show:

- strong Transformer intuition
- clean RNA-oriented problem formulation
- ability to turn a biological task into a practical deep learning demo

## Chosen v1 task

`miRNA-mRNA site-level interaction prediction`

Why this is a good v1:

- sequence lengths are short and manageable
- the interaction story is easy to visualize
- cross-attention is a natural modeling choice
- training should stay fast on local hardware

## Chosen v1 modeling idea

Use a pairwise Transformer instead of encoding each RNA independently and handing features to a separate classifier.

Core idea:

- self-attention for each sequence
- cross-attention to model pairing between the two RNAs
- pooled representation for binary classification

Optional RNA-aware extension for v1.1:

- add a simple pairing bias based on `A-U`, `C-G`, and `G-U`

## Chosen v1 data direction

Start from a small, already curated benchmark slice rather than building a large dataset from scratch.

Initial target:

- `miRBench / AGO2_CLASH_Hejret2023`

Reasons:

- smaller scope
- cleaner starting point
- better fit for a two-week project

## Questions to resolve soon

- exact processed file schema
- target window length
- best split strategy for the first public demo
- whether to include the pairing bias in the first release or the second
