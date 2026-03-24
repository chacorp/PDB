---
name: ASM-based LBS redesign plan
description: HierarchicalLBS + DiffusionNetEDD full redesign — Maya export, network architecture, Jacobian→Poisson output, implementation order
type: project
---

## HierarchicalLBS + DiffusionNetEDD 전체 설계

### Problem with Current LBS (NGBC.py)
- W_lbs predicted from scratch (no anatomical prior) → asymmetric movements fail
- No hierarchy → child joint ignores parent motion
- No bind pose (B_inv) → rest→deform transform undefined
- Global transform predicted directly → high learning difficulty
- Result: one-sided eyebrow/eye, puffing, suction all fail

### Current NGBC LBS Architecture (reference)
- `lbs_exp_z_model(deform_in)` → z_exp [B,1,L] (LinearEncoder, out_type='global')
- `lbs_weight_model(source_in)` → W [B,N,J] (LinearEncoder, out_type='vertices', use_pou=True)
- `lbs_pose_model(z_exp)` → T [B,J,9] = 6D rot + 3D translation (LinearEncoder, out_type='global')
- `apply_lbs(source_vert, W, T, C)`: rotate around joint center C, blend with W
- Joint centers: network-predicted OR weighted avg of vertices

---

## HierarchicalLBS Design

### Network reuse
- `lbs_exp_z_model` → keep as-is (expression encoder)
- `lbs_pose_model` → keep as BoneTransformNet: z_exp [B,1,L] → local_rot_6d [B,J,6]
  - Change: predict LOCAL rotation (relative to parent), not global
  - Change: output J*6 only (no translation; translation comes from hierarchy)
- `lbs_weight_model` → repurposed as weight offset: template_in → delta_W [N,J]
- **NEW**: `bind_offset_model`: template_in → delta_t [J,3] (LinearEncoder, out_type='global')
  - Same structure as lbs_weight_model; input=template shape; runs once per identity

### Learnable parameters (per-identity, run once per template)
| Parameter | Shape | Init | Purpose |
|-----------|-------|------|---------|
| `logit_W_base` | [N,J] | log(W_maya+ε) | base skin weights (buffer, not optimized) |
| `delta_W` | [num_ids, N,J] | 0 | per-identity weight offset via bind_offset_model |
| `delta_t` | [num_ids, J,3] | 0 | per-identity bind pose offset via bind_offset_model |

### Fixed buffers (from Maya export)
| Buffer | Shape | Source |
|--------|-------|--------|
| `B_inv` | [J,4,4] | bindPreMatrix from skinCluster |
| `parent_idx` | [J] | hierarchy (-1=root) |
| `bind_pos` | [J,3] | joint world position |

### Forward (standard hierarchical LBS)
```
G_j = T_world_j @ B_inv_j          # skinning matrix [4,4]
v' = Σ_j w_j * (G_j @ [v;1])[:3]   # blend

T_world[root] = T_local[root]
T_world[j]    = T_world[parent[j]] @ T_local[j]

B_inv_id = adjust_B_inv(B_inv, delta_t[id])   # translate-only adjustment
W = softmax(logit_W_base + delta_W[id])
```

### Regularization
- `L_W_reg = ||delta_W||²`  (weight stays close to Maya init)
- `L_t_reg = ||delta_t||²`  (bind pose offset stays small)

### Maya Export (maya_rig/export_rig.py)
Output: `maya_rig/exported/rig_info.json` + `skin_weights.npy`
- joint name, index, parent_index (topological order)
- `bindPreMatrix[i]` from skinCluster → B_inv [J,4,4]
- joint world position [J,3]
- skin weights [V,J] float32
- Coordinate system: Maya Y-up = MF mesh space (no conversion needed; obj imported into Maya)
- **Joint hierarchy from Maya is ground truth** (not asm_paper_bones.json — paper had L/R naming errors)

### Multi-topology weight transfer (BIWI/VOCA from MF)
- Per target vertex v_t: find nearest face in MF mesh
- Project v_t onto that face → barycentric coords (α,β,γ)
- W_target[v_t] = α*W_mf[v0] + β*W_mf[v1] + γ*W_mf[v2]
- Script: `maya_rig/transfer_weights.py`

---

## DiffusionNetEDD Design

### Why Jacobian not Strain
- Strain E = ½(F^T F - I): rotation-invariant but loses area change (puffing/suction)
- Full F: polar decomp F = R @ U → U = stretch tensor, det(F) = area change signal
- det(F) > 1 = puffing, det(F) < 1 = suction, det(F) = 1 = rigid/no deformation
- Rotation R not needed for EDD (head rotation ≠ deformation; R causes false EDD signal)

### Jacobian features: compute_jacobian_features() in mesh_utils.py
- Polar decomp F → R, U per face
- Output: [det(U)-1 (1), upper_tri(U) (6)] = 7 dim per face
- Face → vertex scatter (area-weighted average)

### DiffusionNet Operator Precompute
- `mass, L, evals, evecs, gradX, gradY` via `get_operators(k_eig=128)`
- Per-topology cache: `utils/diffusion_ops/{mf,biwi,voca}_ops.pt`
- Shared DiffusionNet weights across topologies (topology-agnostic, different operators)

### Architecture
```
Input per vertex: [pos(3) + norm(3) + jac_feat(7)] = 13 dim
DiffusionNet(C_in=13, C_out=128, C_width=128, N_block=4, outputs_at='vertices')
FiLM: z_exp → Linear(L,256) → (γ[128], β[128]); x = γ*x + β
Output head: Linear(128 → 9) → per-face Jacobian [B, F, 3, 3]
Poisson solve: L @ delta_v = div(J_edd) → displacement [B, N, 3]
```

### Output: Jacobian → Poisson solve (NOT direct displacement)
- Network predicts per-face delta-Jacobian (deformation gradient of EDD)
- Poisson solve recovers consistent vertex displacement
- Advantages: directional wrinkle consistency, no per-vertex incoherence artifacts
- Precompute Cholesky factorization of Laplacian L per topology → fast at runtime

### Training target
- Phase A (now, LBS not ready): EDD_target = GT - smooth_GT (wrinkle only)
- Phase B (after HierarchicalLBS): EDD_target = GT - LBS_out (wrinkle + puffing + suction)

---

## Implementation Order
1. [ ] Maya export script (maya_rig/export_rig.py) → rig_info.json + skin_weights.npy
2. [ ] Weight transfer script (maya_rig/transfer_weights.py) for BIWI/VOCA
3. [ ] mesh_utils.py: compute_jacobian_features() — polar decomp, face→vertex scatter
4. [ ] scripts/precompute_diffusion_ops.py — per-dataset operator cache
5. [ ] models/hierarchical_lbs.py: HierarchicalLBS class
6. [ ] models/diffusion_edd.py: DiffusionNetEDD class (DiffusionNet + FiLM + Jacobian out + Poisson)
7. [ ] NGBC.py: integrate HierarchicalLBS + DiffusionNetEDD
8. [ ] dataloader: id_idx, operator batch packing
9. [ ] train_CBD.py: L_W_reg, L_t_reg loss terms
10. [ ] Experiment A: DiffusionNetEDD + Jacobian (current LBS) vs old DispNet + strain
11. [ ] Experiment B: HierarchicalLBS + DiffusionNetEDD full integration

### Dataset Vertex Counts
| Dataset | Vertices | Template Path |
|---------|----------|---------------|
| Multiface (mf) | 5,223 | /data/sihun/multiface_align/mf_templates.pkl |
| BIWI | 2,560 | /data/sihun/BIWI_align_deci/templates_align_deci.pkl |
| VOCA | 3,525 | /data/sihun/VOCA-COMA/voca_templates.pkl |
| ICT (face only) | 9,409 | /data/sihun/NFR_data/ict_face_pt/neutral_verts.pt |

- MF template obj: utils/mf/mf_aligned_mean.obj (5223v, 10278f)
