# Selective Synaptic Dampening (SSD)

## Base Architecture
- Weight-importance-based unlearning (typically Fisher-style).
- Estimate parameter importance for forget data and dampen selected parameters.

## Inputs
- model weights
- forget dataset (for importance estimation)
- damping policy and thresholds

## Outputs
- dampened checkpoint (`unlearned.pt`)
- summary metadata (importance stats, runtime)

## Main Hyperparameters
- fisher estimation samples / mini-batches
- damping coefficient (`alpha`)
- clipping thresholds
- optional layer-wise scaling

## Notes
- Usually faster than full retraining and often one-shot.
- Implemented as a runnable proxy:
	- dampens selected parameter groups by configurable factor (`alpha`),
	- then runs forget suppression + retain recovery refinement.
- This is an engineering baseline for selective dampening without full Fisher estimation.
