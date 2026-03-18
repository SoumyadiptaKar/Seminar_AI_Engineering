# Gradient Difference (NegGrad+) Unlearning

## Base Architecture
- Typical objective combines:
  - ascent on forget subset gradients,
  - descent on retain subset gradients.
- In this project, implemented as **alternating optimization cycles**:
  - forget suppression step,
  - retain recovery step,
  repeated for configurable cycles.

## Inputs
- `original_weights`
- retain/forget dataset splits (manifest + prepared loaders)
- optimizer settings (`lr`, batch, epochs)

## Outputs
- unlearned checkpoint (`unlearned.pt`)
- run summary JSON with timing and metric card

## Main Hyperparameters
- forget-loss weight (`lambda_forget`)
- retain-loss weight (`lambda_retain`)
- learning rate / epochs / batch size
- optional gradient clipping

## Notes
- Planned implementation path: shared batch loader over retain+forget, then update with
  `grad = lambda_retain * grad_retain - lambda_forget * grad_forget`.
- Current implementation is a practical cycle-based proxy that is fully runnable in this repository.
