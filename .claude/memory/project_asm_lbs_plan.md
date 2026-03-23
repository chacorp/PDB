---
name: ASM-based LBS redesign plan
description: Plan to replace current prior-free LBS with ASM-style rig prior (Maya bind init + per-identity optimize), dataset vertex counts, current NGBC LBS analysis, EDD puffing solution
type: project
---

## LBS Redesign: ASM-style Rig Prior

### Problem with Current LBS (NGBC.py)
- W_lbs predicted from scratch by `lbs_weight_model` (LinearEncoder) → no anatomical prior
- Joint centers C_bar derived from W via weighted avg → chicken-and-egg problem
- No explicit bind pose B_inv — centers computed dynamically
- num_lbs_joints default=4 (up to 64), no anatomical correspondence
- Bone transforms predicted by `lbs_pose_model` from z_exp (6D rotation + 3D translation)
- Results: eyebrow local movement unstable, lip flip artifacts, no puffing/suction

### Current NGBC LBS Architecture (for reference)
- `lbs_weight_model(source_in)` → W (B,N,J) with POU constraint (sum=1)
- `lbs_exp_z_model(deform_in)` → z_exp latent
- `lbs_pose_model(z_exp)` → T (B,J,9) = 6D rotation + 3D translation
- `apply_lbs(source_vert, W, T, C)`: v_local = v - C, v_rot = R @ v_local, v = v_rot + C, blend with W
- Joint centers: network-predicted OR weighted average of vertices

### Proposed Architecture (ASM-style)
- **Bone hierarchy**: 84 bones from ASM paper Table 8 (NOT Rigify)
  - Flat: root → head → region bones (depth 2-3 max)
  - All bone orientations z-axis aligned (except root)
- **W_init**: Maya auto-skin on 1 template per topology → per-identity ΔW optimize
  - ASM showed: only ONE template needs bone+weight predefine; others optimize
- **B_inv_init**: Maya bind pose → per-identity Δψ optimize (translation only, ASM Eq.14)
- **BoneTransformNet(z_exp) → T(J, 4, 4)**: expression-driven, shared across identities

### Dataset Vertex Counts
| Dataset | Vertices | Template Path |
|---------|----------|---------------|
| Multiface (mf) | 5,223 | /data/sihun/multiface_align/mf_templates.pkl |
| BIWI | 2,560 | /data/sihun/BIWI_align_deci/templates_align_deci.pkl |
| VOCA | 3,525 | /data/sihun/VOCA-COMA/voca_templates.pkl |
| COMA | 3,525 | (shared with VOCA) |
| ICT (full) | 11,248 | /data/sihun/NFR_data/ict_face_pt/neutral_verts.pt |
| ICT (face only) | 9,409 | (same file, different index range) |

- MF template obj: /source/inyup/NeuralFacialAnimation/utils/mf/mf_aligned_mean.obj (5223v, 10278f)
  - Mean-like template of 13 training identities (small diff ~0.004 from exact arithmetic mean, possibly different generation method)
- MF pkl has 13 per-identity neutral templates (5223,3) + shared face indices (10278,3)
- 13 MF training identities: 002757580, 002539136, 6674443, 6795937, 8870559, 2183941, 002643814, 5372021, 7889059, 002914589, 5067077, 002421669, 002645310

### Maya Rig Setup
- Script: `maya_rig/create_face_rig.py` — creates 84 joints from ASM Table 8
- Bone positions: approximate, mapped from Rigify positions via semantic matching
- Coordinate conversion: Blender (X,Y,Z) Z-up → Maya (X,Z,-Y) Y-up
- Bone data also exported to: `maya_rig/asm_paper_bones.json`
- Joint naming in Maya: `jnt_<bone_name>` (dots replaced with underscores)
- **Start with Multiface** (5,223 verts, most used dataset, 13 identities)
- Joint positions: head ≈ mesh 안쪽 (rotation center), tail ≈ surface direction
- One template bind → other identities optimize from same init

### Implementation Order
1. ✅ Maya script created with ASM Table 8 hierarchy
2. **IN PROGRESS**: Position joints on MF template in Maya → Bind Skin → export W_init
3. NGBC.py: replace lbs_weight_model with W_init + per-identity ΔW
4. NGBC.py: add dyn_binding (B_inv_init + Δψ per identity)
5. BoneTransformNet: predict T(J, 4, 4) from z_exp

### EDD Layer (Future)
- Current EDD (smooth_GT → GT) = wrinkles only, cannot handle puffing
- Puffing lives in LBS→smooth_GT gap, trueEDD near-zero for puffing regions (confirmed via vis)
- Solution: EDD target = LBS_out → GT, conditioning = z_exp (not LBS Jacobian)
- Jacobian det(F)-1 visualization confirmed: naiveEDD has strong puffing signal, trueEDD does not
