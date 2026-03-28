# Two-Week Roadmap

## Goal

Finish a small but polished Transformer demo that is strong enough for GitHub, CV, and project discussion.

## Week 1

### Day 1-2

- Set up environment
- Finalize task framing
- Inspect the chosen dataset
- Decide the processed data schema

### Day 3-4

- Write `prepare_data.py`
- Build train/val/test split
- Sanity-check sequence lengths and class balance

### Day 5-7

- Implement the smallest working model
- Run a short smoke-test training
- Verify loss decreases and attention tensors look sensible

## Week 2

### Day 8-10

- Tune only a few knobs:
  - hidden size
  - dropout
  - target window length
- Save best checkpoint and metrics

### Day 11-12

- Create PR curve
- Create one attention heatmap example
- Write the prediction script

### Day 13-14

- Polish `README.md`
- Export final figures
- Clean the repo for public presentation

## Definition of done

- One command to train
- One command to predict
- One example attention figure
- One concise README with model diagram and results
