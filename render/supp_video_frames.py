"""Frames (dataloader_CBD indices) per supplementary video and scenario, shared by
supp_video_infer.py (what to infer / render) and supp_video_layout.py (what to compose,
in this order)."""
MF_FREE_FACE = list(range(3773, 4964))  # EXP_free_face in mf_ROM_vertex_data_test.txt
ICT_M00_ALL = list(range(0, 1147))      # 20240325_MySlate_924
ICT_M02_ALL = list(range(0, 183))       # 20240318_MySlate_922
MF_S1 = list(range(3875, 4175))         # user study segment mf_S1 (userstudy_infer.SEGMENTS)
STYLIZED = ["aura", "piers", "face-reference"]
# coma test FaceTalk_170809_00138_TA (first identity block of coma_vertex_data_test.txt):
# cheeks_in -> high_smile -> mouth_extreme -> mouth_open -> mouth_up
COMA_CLIPS = (list(range(205, 363)) + list(range(560, 739)) + list(range(1254, 1447))
              + list(range(1647, 1822)) + list(range(2046, 2241)))

VIDEO_FRAMES = {
    "comparison": {"mf_to_ict": MF_FREE_FACE, "ict_to_mf_m00": ICT_M00_ALL, "ict_to_mf_m02": ICT_M02_ALL},
    "design_study": {"coma_to_biwi": COMA_CLIPS, "mf_to_coma": list(range(4187, 4962))},
    "loss_ablation": {"mf_to_biwi": MF_FREE_FACE},
    "stylized": {f"mf_to_{t}": MF_S1 for t in STYLIZED},
    "ttt": {f"mf_to_{t}": MF_S1 for t in STYLIZED},
}
