---
name: ASM unofficial repo uses wrong skeleton
description: ASM-unofficial/data/rig_info.json contains Blender Rigify metarig (85 bones, deep chain), NOT the ASM paper's actual 84-bone flat skeleton (Table 8). Must use paper's Table 8 hierarchy instead.
type: feedback
---

ASM-unofficial repo's rig_info.json is a Blender Rigify face metarig, NOT the paper's skeleton.

**Why:** Paper Table 8 shows 84 bones with flat hierarchy (most parent=head, depth 2), semantic names (nose_tip, lip_corner.L, apple_center.L). Unofficial repo has 85 bones with deep chains (nose→nose.001→nose.002→...), Rigify naming convention. Completely different structure.

**How to apply:** When implementing ASM-style rig, use paper Table 8 for hierarchy and bone names. Do NOT use rig_info.json from unofficial repo. Joint positions from unofficial repo are also for Rigify rig on HIFI3D++ mesh, not directly usable.
