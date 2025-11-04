# Numerical Fragility in Transformers (NSoT)

Layer-wise theory and diagnostics to explain, forecast, and mitigate forward-pass numerical instability in Transformers — FP32↔FP16/BF16, lightweight, training-time estimable.

## Features
- **Attention stability factors**: κ_score, κ_softmax, κ(V) — per-layer, interpretable, estimable during training
- **Residual relaxation**: small-gain–style condition that turns multiplicative growth into controlled accumulation
- **LayerNorm indicator ρ_LN**: flags the ε-dominated regime with a first-order forward error bound
- **Unified stability predictor**: per-layer RHS that tracks precision-mismatch across FP32/FP16/BF16
- **Early-warning signals**: κ_softmax peaks anticipate instability for safe scheduling of precision/ε adjustments
- **Minimal mitigation**: conservative **LayerNorm-ε bump** policy; architecture-agnostic and low-overhead

Runs as a single notebook.  
Tested on Python 3.11.
