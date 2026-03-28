# RNA Pair Transformer Demo

Small, CV-focused demo for `miRNA-mRNA` site-level interaction prediction with a lightweight cross-attention Transformer.

## Why this project

- Show Transformer design skill in an RNA setting.
- Keep the scope small enough to finish in about two weeks.
- Train fast enough for short iteration cycles on a consumer GPU.
- Produce clear visuals for GitHub and CV use.

## Initial scope

- Task: binary classification
- Input: `miRNA sequence` + `candidate mRNA target-site window`
- Output: interaction probability
- Tokenizer: character-level `A/C/G/U/N`
- Model:
  - miRNA encoder
  - mRNA/site encoder
  - one cross-attention block
  - pooling + MLP classifier
- First dataset target: `miRBench / AGO2_CLASH_Hejret2023`
- First deliverables:
  - training curve
  - PR curve
  - attention heatmap
  - simple prediction command

## What we are not doing in v1

- No full transcript scanning
- No graph model
- No DNABERT or large pretrained model
- No large benchmark suite
- No long baseline section

## Project structure

- `assets/` figures for README
- `data/raw/` downloaded raw files
- `data/processed/` cleaned datasets
- `docs/` planning and research notes
- `experiments/` configs, logs, and run notes
- `notebooks/` quick analysis and visualization
- `scripts/` data preparation scripts
- `src/` training and model code

## Next steps

1. Create the conda environment from `environment.yml`.
2. Download the first small dataset slice into `data/raw/`.
3. Define a simple processed TSV/CSV format.
4. Implement the smallest working Transformer.
5. Train one short run and save the first plots.
