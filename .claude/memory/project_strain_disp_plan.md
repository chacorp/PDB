---
name: Strain DispNet improvement plan
description: Phase-based plan for diagnosing and fixing lip artifacts in v10 DispNet — variable isolation, architecture options (full strain tensor, principal strain, FiLM, local neighbor strain), loss/hyperparameter tuning strategy
type: project
---

## Strain-conditioned DispNet 개선 실험 계획

Two simultaneous changes from v9→v10 caused lip artifacts:
1. Input: 13dim (source template included) → 7dim (LBS output only)
2. Loss: single final recon → decomposed (smooth LBS + wrinkle)

### Phase 1: Variable Isolation (원인 진단)

| Experiment | Input | Loss | Purpose |
|------------|-------|------|---------|
| Baseline (v9) | 13dim + src template | single final recon | Already trained (200~500 epoch) |
| A | 13dim + src template | decomposed | Is loss decomposition the cause? |
| B | 7dim (no src template) | single final recon | Is input reduction the cause? |
| C (v10) | 7dim | decomposed | Running on other server |

**Why:** Need to isolate which change causes lip artifacts before choosing fix strategy.
**How to apply:** Run experiments A and B next, compare with baseline and C.

### Phase 2: Fix Strategy (based on Phase 1 result)

#### If Loss is the cause:
1. Lambda balancing (recon-def vs recon-lbs vs recon-wrinkle)
2. L1 loss instead of MSE (less outlier sensitivity at lip region)
3. Smooth GT quality (check if smoothing over-smooths lip boundary)
4. Loss annealing (single recon early → decomposed later)

#### If Architecture (Input) is the cause — 4 options:
1. **Full strain tensor** (dim=6 symmetric or 9): directional deformation info
2. **Principal strain** (dim=3): eigenvalues of E, compact + rotation invariant — RECOMMENDED first try
3. **FiLM/AdaIN conditioning**: modulate hidden features by strain instead of concat
4. **Local neighborhood strain**: aggregate 1-ring neighbor strain for spatial patterns

#### If both: 13dim + decomposed loss + lambda tuning

### Phase 3: Hyperparameter Tuning (after direction is set)
- lr: 1e-4, 5e-5, 1e-5
- lambda_recon_lbs / lambda_recon_wrinkle: 0.1~2.0
- L1 vs L2 per loss term
- smooth_n_iter: 8, 16, 32
- batch_size: 8, 16, 32

### Priority: Phase 1 → Phase 2 (principal strain or lambda first) → Phase 3
