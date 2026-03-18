# SCRUB Unlearning

## Base Architecture
- Teacher-student setup:
  - teacher: original pre-unlearning model,
  - student: trainable unlearning model.
- Objective family:
  - retain set: distill student toward teacher,
  - forget set: maximize disagreement/divergence.

## Inputs
- frozen teacher weights
- student init weights (from original model)
- retain and forget datasets
- distillation temperature and divergence weights

## Outputs
- student checkpoint after SCRUB optimization
- metrics: retain utility, forget effectiveness, runtime

## Main Hyperparameters
- `temperature`
- `lambda_retain_distill`
- `lambda_forget_divergence`
- epochs, batch, optimizer, lr

## Notes
- Implemented as a runnable proxy:
  - forget divergence stage on forget set,
  - retain distillation stage on teacher-generated pseudo labels.
- This approximates SCRUB behavior within Ultralytics training constraints.
