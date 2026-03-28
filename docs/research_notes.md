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

## Biological intuition

- `miRNA` is a short non-coding RNA, usually around `~22 nt`.
- One of its central functions is to bind a local site on an `mRNA`.
- For target recognition, the most important region is often the `seed region`, roughly positions `2-7` or `2-8` on the miRNA.
- This makes the task naturally suited to pairwise sequence modeling instead of single-sequence classification.

For this demo, we are not trying to predict interactions across the full transcriptome.

We are solving a simpler and cleaner problem:

`given one miRNA and one candidate mRNA target-site window, predict whether they interact`

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

## Dataset snapshot

Balanced miRBench v5 files currently selected:

- `AGO2_CLASH_Hejret2023_train.tsv`
- `AGO2_CLASH_Hejret2023_test.tsv`
- `AGO2_eCLIP_Klimentova2022_test.tsv`

Relevant columns for v1:

- column 1: `gene`
- column 2: `noncodingRNA`
- column 6: `label`

Observed data properties:

- `gene` length is always `50`
- `noncodingRNA` length is roughly `17-26`
- labels are approximately balanced in all three selected files
- sequence alphabet is currently `A/C/G/T`

Important simplification for v1:

- ignore metadata columns such as `feature`, `chr`, `start`, `end`, and `strand`
- convert `T -> U` during preprocessing so the model consistently sees RNA alphabet

## Final v1 task definition

Input:

- one `miRNA` sequence
- one `50 nt` target-site sequence

Output:

- one binary interaction score

Training setup:

- main training source: `Hejret train`
- validation split: stratified `20%` split from `Hejret train`
- in-distribution test set: `Hejret test`
- external test set: `Klimentova test`

Strictness note:

- there are `2` exact overlapping `gene + miRNA + label` pairs between `Hejret train` and `Klimentova test`
- remove those two pairs from the training set during preprocessing so the external test story stays clean

## Preprocessing design

### Sequence normalization

- uppercase all sequences
- replace `T` with `U`

### Vocabulary

Use a minimal character-level vocabulary:

- `PAD`
- `A`
- `C`
- `G`
- `U`
- `N`

Even though the current files only show `A/C/G/T`, keeping `N` in the vocabulary makes the pipeline safer.

### Fixed lengths

- `max_target_len = 50`
- `max_mirna_len = 30`

Rationale:

- `gene` is already fixed at `50`
- miRNA length tops out well below `30`, so this is safe and simple

### Example tensor schema

For each sample, the processed output should be:

- `target_ids`: shape `[50]`
- `target_mask`: shape `[50]`
- `mirna_ids`: shape `[30]`
- `mirna_mask`: shape `[30]`
- `label`: scalar

For a batch, this becomes:

- `target_ids`: `[B, 50]`
- `target_mask`: `[B, 50]`
- `mirna_ids`: `[B, 30]`
- `mirna_mask`: `[B, 30]`
- `label`: `[B]`

## Model implementation plan

### High-level design

Use a lightweight pairwise Transformer:

- one encoder for the miRNA sequence
- one encoder for the target-site sequence
- one cross-attention block for interaction modeling
- one classifier head for binary prediction

This keeps the project clearly Transformer-based while staying small enough for quick iteration.

### Module breakdown

#### 1. Token embedding

- shared embedding layer across both sequences
- embedding size: `d_model = 64`

#### 2. Positional encoding

- learned positional embeddings are fine for v1
- add positional embeddings after token embeddings

#### 3. Sequence encoders

Use two small self-attention encoders:

- `miRNA encoder`: `2` TransformerEncoder layers
- `target encoder`: `2` TransformerEncoder layers

Suggested defaults:

- `d_model = 64`
- `num_heads = 4`
- `dim_feedforward = 128` or `256`
- `dropout = 0.1`

#### 4. Cross-attention block

After both encoders:

- query = encoded miRNA states
- key = encoded target states
- value = encoded target states

This lets the miRNA representation attend directly to the target-site representation.

Use:

- one `MultiheadAttention` layer
- residual connection
- layer normalization
- small feed-forward block

#### 5. Pooling

For v1, use masked mean pooling over the cross-attended miRNA states.

This is simple and stable.

Optional small extension:

- concatenate pooled miRNA states with pooled target states before classification

#### 6. Classifier

Use a small MLP:

- `Linear(d_model, d_model)`
- `GELU`
- `Dropout`
- `Linear(d_model, 1)`

Output:

- one logit

Loss:

- `BCEWithLogitsLoss`

## Attention interpretation plan

The attention map is part of the presentation value of the project.

What we want to inspect:

- whether miRNA positions attend strongly to a compact region of the target site
- whether attention around the miRNA seed region looks sharper on positive examples

Important caveat:

- attention can be used as a qualitative interpretation tool
- it should not be presented as a direct biological mechanism proof

## Training plan

### First training run

Goal of the first run:

- verify the pipeline works end-to-end
- confirm the loss decreases
- confirm the model can beat random guessing
- save one attention example

Suggested first-run defaults:

- batch size: `64`
- epochs: `10-20`
- optimizer: `AdamW`
- learning rate: `1e-3` or `5e-4`
- weight decay: `1e-4`

### Metrics

Primary metrics:

- `AUPRC`
- `ROC-AUC`
- `F1`
- `MCC`

Secondary metric:

- `accuracy`

### Evaluation story for the demo

- train on `Hejret train`
- tune using a validation split from `Hejret train`
- report main test results on `Hejret test`
- report external generalization on `Klimentova test`

## Immediate implementation order

1. Write `scripts/prepare_data.py`
2. Save processed train, val, test, and external-test TSV files
3. Implement tokenization and dataset loading in `src/data.py`
4. Implement the pairwise Transformer in `src/model.py`
5. Implement training loop and metrics in `src/train.py`
6. Implement single-example inference in `src/predict.py`
7. Export PR curve and one attention heatmap for the README

## Deferred ideas

These are explicitly out of scope for the first version:

- pairing bias in cross-attention
- metadata features such as `feature` or genomic coordinates
- k-mer tokenization
- pretrained sequence encoders such as DNABERT
- secondary-structure features
- transcript-level scanning
