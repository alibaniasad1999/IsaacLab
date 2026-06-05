# Cable Method Comparison: Capsule-chain vs Deformable

Common material: PUR robot cable, E = 30 MPa, nu = 0.45.

Test scenarios: hanging-kick + two-robot dual-arm manipulation.

| Criterion | Capsule-chain | Deformable body | Notes |
|---|---|---|---|
| Stability (stayed stable?) | Stable | n/a | Both should stay stable at dt=1/240 s |
| Inextensibility error (mm) | 0.00 mm | n/a | Capsule = 0 by design; deformable stretches |
| Two-robot span error (mm) | n/a | n/a | Lower = stiffer coupling between arms |
| Peak reaction force (proxy) | n/a | n/a | Force transmitted to follower arm |
| Axial elasticity | No (rigid links) | Yes (FEM) | Deformable models stretching physically |
| Self-collision | Approx (capsule contacts) | Native (FEM self-collision) | Deformable handles knots/loops better |
| Compute cost | Low-moderate | High (FEM + remeshing) | Capsule cheaper for many envs (RL) |

**Summary:** the capsule-chain is cheaper and inextensible by construction, making it the better default for large-scale RL training. The deformable body adds physical axial elasticity and native self-collision, which matter for tasks involving knotting, tight wrapping, or accurate force feedback.
