# SISA Unlearning

## Base Architecture
- SISA = Sharded, Isolated, Sliced, and Aggregated training.
- Train multiple shard models, each with ordered slices.
- Unlearning request retrains only affected shard slices, then re-aggregates.

## Inputs
- dataset partition plan (num shards, slices per shard)
- base training config per shard
- unlearning request mapping to shard(s)

## Outputs
- updated shard checkpoints
- aggregated model output / ensemble predictions
- unlearning-time speedup stats

## Main Hyperparameters
- `sisa.shards`
- `sisa.slices_per_shard`
- retrain depth from affected slice onward
- aggregation rule (mean logits/votes)

## Notes
- Strong systems baseline for practical deletion workloads.
- Current repository code has runner/scaffold hooks; full shard training/retraining loop is pending.
