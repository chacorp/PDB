# -*- coding: utf-8 -*-
"""Export neutral meshes for the first 10 ICT test-split identities
(ICT_live_100[0:10] == viser ict_val_000..009) so NFR precompute can be built
for them with the canonical data_prepare.py pipeline (matches the validated set
-> no render jitter). Run data_prepare.py on the produced OBJ dir afterwards.
"""
import os, sys
import numpy as np
import trimesh

REPO = os.environ.get("REPO", "/source/inyup/NeuralFacialAnimation")
sys.path.insert(0, REPO)
os.chdir(REPO)
from utils.remesh_utils import ICT_face_model

OUT = os.path.join(REPO, "_tmp_ict_test10_obj")
os.makedirs(OUT, exist_ok=True)

ict = ICT_face_model(base_dir=REPO)            # scale 0.1, full head (11248 v)
live = np.load(os.path.join(REPO, "data/ICT_live_100/iden_vecs.npy")).astype(np.float32)
print("live100 iden_vecs:", live.shape, "| ict faces:", ict.faces.shape)

for i in range(10):
    neu = ict.apply_coeffs(live[i], exp_coeffs=None)[0].astype(np.float32)  # neutral verts
    m = trimesh.Trimesh(vertices=neu, faces=ict.faces, process=False)
    p = os.path.join(OUT, f"ict_val_{i:03d}.obj")
    m.export(p)
    print(f"  wrote {p}  (V={neu.shape[0]})")
print("DONE export ->", OUT)
