# Why the FEM cable is rigid near the connection, and its collision is rough

Two things you noticed about the **deformable (FEM)** cable:

1. Near the point where it connects (a gripper / a welded anchor) it looks **rigid
   — straight for some length** before it starts to bend.
2. Its **collision is not great** (imprecise, blocky, can clip).

Both are **fundamental to how PhysX FEM works**, not bugs in the tuning. Here's why,
what can be mitigated, and what to use instead. (This is the attachment/collision
companion to `why_fem_cant_be_thin.py`, which covers the thin/jelly/stretch limits.)

---

## 1. Why it's rigid (straight) near the connection

### What PhysX actually does
A PhysX FEM body is a mesh of **tetrahedra**. To attach it to a rigid body (the
gripper, or the welded anchor cube) PhysX uses a `PhysxAutoAttachment`: it finds
**every cable vertex within an overlap distance of the rigid** and **kinematically
ties each of those vertices to the rigid body**.

So the attachment is **not a single pin** — it's a *whole cluster of vertices*
welded to the rigid. That cluster (a "collar" a centimetre or two long) then moves
**as one rigid piece** with the gripper. The cable can only start bending *past*
the collar → it leaves the connection **straight**.

### Why even a perfect attachment stays straight (the deeper reason)
Tying a *region* of the rod to a rigid fixes not just its **position** but its
**orientation**. In beam terms that's a **clamped (built-in) boundary condition**:
the slope at the wall is forced to zero, so the rod must leave **perpendicular to
the clamp** and curve only gradually. A real cable hanging from a hook is a
**pinned** end — it can leave at *any* angle and droops immediately.

```
   CLAMPED (FEM region weld)        PINNED (real cable / capsule / warp)
   ───────┐                         ─┐
          │  <- leaves straight,      \   <- droops right at the
          │     then curves            \     connection
          ╰──                           ╰──
```

FEM gives you a **clamp**; a cable wants a **pin**. That mismatch *is* the rigid
straight section. You can shrink it but not remove it.

### Mitigation (partial)
* **Smaller grab region** → smaller collar. In the two-robot script:
  `CABLE_ATTACH_OVERLAP=0.025` (default 0.04). **Risk:** too small and the
  attachment grabs too few vertices → the cable **detaches and falls** (we hit
  exactly this), or the few held vertices take the whole load and tear/jitter.
* **A softer rod** bends sooner past the collar: `CABLE_FEM_E_EXP=4` (exact EI).
* These reduce the rigid length from ~2–3 cm to ~1 cm; they **cannot** make it a
  true pin.

---

## 2. Why the collision is rough

### What PhysX actually does
The FEM body does **not** collide with its nice smooth visual tube. It collides
with a separate, **simplified, re-meshed collision tetrahedral mesh**
(`add_physx_deformable_body(..., collision_simplification=True)` — the default),
built at a **coarse resolution**. So the collision surface is a **blocky, faceted
approximation** of the cylinder. Consequences:

* contacts are **imprecise** — the cable can sink in or hover by a few mm
  (we tuned `rest_offset`/`contact_offset` to hide this on the bar);
* **thin** obstacles can be **missed** / tunnelled (a soft FEM body squashes into a
  thin bar — see the contact-offset cushion we added);
* **self-collision is OFF by default** (`self_collision=False`) — the cable can
  pass through **itself** when it folds;
* the faceted collision can make the contact look **lumpy**, not smooth.

### Why it's coarse
Collision is resolved per collision-tet. A fine collision mesh = many tets = the
GPU solver gets slow and the cooker (`PxTetMaker`) can **fail** (the same
resolution wall that caps how thin the rod can be). So PhysX *deliberately*
simplifies the collision mesh for speed/robustness — at the cost of accuracy.

### Mitigation (partial)
* `collision_simplification=False` → collide with the full mesh (finer) — but
  **much slower** and may fail to cook on a thin rod.
* Higher `simulation_hexahedral_resolution` → finer collision tets — slower, and
  bounded by the cooking cap (~130 here).
* Enable `self_collision=True` to stop the cable passing through itself — adds cost
  and can jitter.
* Bigger collision offsets smooth the contact but add a standoff gap.

None of these make FEM collision as crisp as an analytic shape.

---

## The root cause (both issues, one source)

FEM is a **volumetric, voxelised** model. Attachment is a **region weld** (→ clamp)
and collision is a **discretised approximate surface** (→ rough). Refining either
hits the **cooking/perf resolution wall**. So on the FEM path you trade attachment
crispness and collision accuracy for the very thing FEM is good at: a **soft body
that squashes/drapes volumetrically**.

| want | FEM (deformable) | capsule (`cable.py`) | warp (`cable_warp.py`) |
|---|---|---|---|
| pin-like connection (droops at the joint) | ✗ region clamp | ✓ ball joint | ✓ single pinned node |
| accurate collision | ✗ simplified tets | ✓ exact capsule shapes | ✓ exact node-vs-primitive |
| volumetric squash on contact | ✓ | ✗ | ✗ |
| thin / no stretch / no jelly | ✗ | ✓ | ✓ |

## Recommendation

For a cable that **bends right at the connection** and **collides cleanly**, use the
**capsule chain** (`cable.py` / `CABLE_METHOD=capsule`) — its ends are real **ball
joints** (pin) and each link is an exact capsule collider — or the **Warp rod**
(`cable_warp.py` / `CABLE_METHOD=warp`), whose end is a single **pinned node** and
which collides node-vs-primitive exactly.

Use the **FEM** cable only when you specifically need a **volumetric soft body**
(a fat rod visibly squashing and conforming on contact). For everything else it is
the wrong tool — the rigid collar and rough collision are the price of the volume
model, the same root as the thinness/jelly/stretch limits in
`why_fem_cant_be_thin.py`.
