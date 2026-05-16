# Govoni Table 1 — Reproduction Results

Source paper: Govoni et al. 2025, arXiv:2504.13659.
Sweep driver: `govoni_sweep.py` → `cable_v2.py`.

Our solver: PhysX TGS, 64 position iterations, Δt = 1/240 s.
Their solver: explicit MSD integration at Δt = 5e-6 s (or 1e-7 s for row 4).

| Row | E (MPa) | i (links) | Paper stable? | Paper t_unstable (s) | Ours stable? | Ours t_unstable (s) | Ours max |ω| (deg/s) | Matches paper? |
|---|---|---|---|---|---|---|---|---|
| 1 | 12.6 | 10 | Stable | — | Stable | — | 5.34e+02 | ✓ matches |
| 2a | 526.0 | 10 | Stable | — | Stable | — | 3.39e+02 | ✓ matches |
| 2b | 1002.0 | 6 | Stable | — | Stable | — | 1.78e+02 | ✓ matches |
| 3 | 1002.6 | 10 | Unstable | 0.40 | Stable | — | 2.57e+02 | ✗ differs |
| 4 | 1002.6 | 10 | Stable | — | Stable | — | 2.57e+02 | ✓ matches |
