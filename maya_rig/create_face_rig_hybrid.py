"""
Maya Python script: Create 64-bone hybrid face rig (JNR base + ASM anim joints).
Run this inside Maya Script Editor (Python tab).

Design rationale:
    JNR (Vesdapunt et al., 2020): 52 joints designed for expression modeling.
        + Jaw_Front, Upper/Lower Eyelid, Top/Bottom Lip, Cheeks
        - Lacks fine-grained lip/eye/nose animation control

    ASM (Yang et al., 2023): 84 joints designed for identity fitting.
        + Dense coverage for face shape modeling
        - Missing key animation joints (jaw open, eyelid)

    Hybrid (Ours): JNR 52 + 12 ASM animation joints = 64 joints
        Selection criteria: only ASM joints that expand animation DoF
        beyond what JNR provides, with clear semantic purpose.

Added ASM joints (14):
    eye_inner_corner_L/R  — squint/crow's feet direction
    eye_outer_corner_L/R  — smile eye crinkle
    lip_upper_side_L/R    — asymmetric sneer
    lip_lower_side_L/R    — asymmetric lip control
    nose_wing_L/R         — nostril flare
    eyebrow_mid_L/R       — eyebrow arch curvature
    eyebrow_inner_L/R     — independent inner brow control (AU4 frown)
"""

import maya.cmds as cmds


# ============================================================
# Hybrid: JNR 52 + ASM Anim 12 = 64 joints
# Coordinate system: Maya Y-up (X=right, Y=up, Z=forward)
# ============================================================

BONE_DATA = [
    # (index, parent_name, bone_name, maya_position [x, y, z])

    # === Root, Neck, Skull (JNR) ===
    (0,  None,                  "Root",                    [0.0,   0.0,    0.0]),
    (1,  "Root",                "Neck",                    [0.0,  -1.8,   -0.3]),
    (2,  "Root",                "Skull_Root",              [0.0,   0.5,    0.0]),
    (3,  "Skull_Root",          "Cranium",                 [0.0,   1.5,    0.0]),
    (4,  "Cranium",             "Crown",                   [0.0,   2.0,    0.0]),
    (5,  "Skull_Root",          "Skull_Center",            [0.0,   0.8,    0.0]),

    # === Brow (JNR + ASM) ===
    (6,  "Skull_Center",        "Brow_Center",             [0.0,   0.9,    0.8]),

    # === Left Eye Region (JNR + ASM additions) ===
    (7,  "Skull_Center",        "Left_Orbital",            [0.45,  0.7,    0.7]),
    (8,  "Left_Orbital",        "Left_Brow_Outer",         [0.7,   0.9,    0.5]),
    (9,  "Left_Orbital",        "Left_Eye_Socket",         [0.45,  0.6,    0.8]),
    (10, "Left_Eye_Socket",     "Left_Upper_Eyelid",       [0.45,  0.7,    0.9]),
    (11, "Left_Eye_Socket",     "Left_Lower_Eyelid",       [0.45,  0.5,    0.9]),
    (12, "Left_Orbital",        "Left_Cheek_Bone",         [0.6,   0.3,    0.7]),
    # ASM additions:
    (13, "Left_Eye_Socket",     "Left_Eye_Inner_Corner",   [0.3,   0.6,    0.85]),
    (14, "Left_Eye_Socket",     "Left_Eye_Outer_Corner",   [0.6,   0.6,    0.78]),

    # === Right Eye Region (JNR + ASM additions) ===
    (15, "Skull_Center",        "Right_Orbital",           [-0.45, 0.7,    0.7]),
    (16, "Right_Orbital",       "Right_Brow_Outer",        [-0.7,  0.9,    0.5]),
    (17, "Right_Orbital",       "Right_Eye_Socket",        [-0.45, 0.6,    0.8]),
    (18, "Right_Eye_Socket",    "Right_Upper_Eyelid",      [-0.45, 0.7,    0.9]),
    (19, "Right_Eye_Socket",    "Right_Lower_Eyelid",      [-0.45, 0.5,    0.9]),
    (20, "Right_Orbital",       "Right_Cheek_Bone",        [-0.6,  0.3,    0.7]),
    # ASM additions:
    (21, "Right_Eye_Socket",    "Right_Eye_Inner_Corner",  [-0.3,  0.6,    0.85]),
    (22, "Right_Eye_Socket",    "Right_Eye_Outer_Corner",  [-0.6,  0.6,    0.78]),

    # === Eyebrow (JNR + ASM mid + inner) ===
    (23, "Left_Orbital",        "Left_Eyebrow_Mid",        [0.55,  0.85,   0.65]),
    (24, "Right_Orbital",       "Right_Eyebrow_Mid",       [-0.55, 0.85,   0.65]),
    (25, "Left_Orbital",        "Left_Eyebrow_Inner",      [0.25,  0.88,   0.75]),
    (26, "Right_Orbital",       "Right_Eyebrow_Inner",     [-0.25, 0.88,   0.75]),

    # === Nose upper (JNR) ===
    (27, "Skull_Center",        "Nose_Root_A",             [0.0,   0.5,    0.9]),
    (28, "Nose_Root_A",         "Nose_Root_B",             [0.0,   0.4,    0.95]),
    (29, "Nose_Root_B",         "Nose_Bridge",             [0.0,   0.2,    1.0]),
    (30, "Nose_Bridge",         "Nose_Ridge_A",            [0.0,   0.0,    1.05]),
    (31, "Nose_Ridge_A",        "Nose_Ridge_B",            [0.0,  -0.1,    1.1]),

    # === Ears (JNR) ===
    (32, "Skull_Root",          "Ears",                    [0.0,   0.3,    0.0]),
    (33, "Ears",                "Left_Ear_Position",       [0.9,   0.3,    0.0]),
    (34, "Left_Ear_Position",   "Left_Ear_Rotate_Scale",   [0.95,  0.3,    0.0]),
    (35, "Ears",                "Right_Ear_Position",      [-0.9,  0.3,    0.0]),
    (36, "Right_Ear_Position",  "Right_Ear_Rotate_Scale",  [-0.95, 0.3,    0.0]),

    # === Lower Face (JNR) ===
    (37, "Skull_Root",          "Lower_Face_Root",         [0.0,  -0.1,    0.5]),
    (38, "Lower_Face_Root",     "Jaw_Root",                [0.0,  -0.3,    0.3]),
    (39, "Lower_Face_Root",     "Maxilla",                 [0.0,  -0.2,    0.7]),

    # === Nose lower (JNR + ASM nose_wing) ===
    (40, "Maxilla",             "Nose_Base_Position",      [0.0,  -0.15,   1.0]),
    (41, "Nose_Base_Position",  "Nose_Base_Rotate",        [0.0,  -0.2,    1.05]),
    (42, "Nose_Base_Rotate",    "Nose_Tip",                [0.0,  -0.25,   1.15]),
    (43, "Nose_Base_Rotate",    "Nose_Nostril",            [0.0,  -0.3,    1.0]),
    # ASM additions:
    (44, "Nose_Base_Rotate",    "Left_Nose_Wing",          [0.15, -0.25,   1.05]),
    (45, "Nose_Base_Rotate",    "Right_Nose_Wing",         [-0.15,-0.25,   1.05]),

    # === Mouth & Jaw (JNR + ASM lip sides) ===
    (46, "Maxilla",             "Mouth_Center_Top",        [0.0,  -0.5,    0.9]),
    (47, "Mouth_Center_Top",    "Mouth_Left_Corner",       [0.35, -0.5,    0.8]),
    (48, "Mouth_Center_Top",    "Mouth_Right_Corner",      [-0.35,-0.5,    0.8]),
    (49, "Mouth_Center_Top",    "Jaw_Front",               [0.0,  -0.6,    0.85]),
    (50, "Jaw_Front",           "Chin",                    [0.0,  -0.8,    0.7]),
    (51, "Jaw_Front",           "Top_Lip",                 [0.0,  -0.5,    0.95]),
    (52, "Mouth_Center_Top",    "Mouth_Center_Bottom",     [0.0,  -0.55,   0.88]),
    (53, "Mouth_Center_Bottom", "Bottom_Lip",              [0.0,  -0.6,    0.92]),
    # ASM additions:
    (54, "Mouth_Center_Top",    "Left_Lip_Upper_Side",     [0.18, -0.48,   0.88]),
    (55, "Mouth_Center_Top",    "Right_Lip_Upper_Side",    [-0.18,-0.48,   0.88]),
    (56, "Mouth_Center_Bottom", "Left_Lip_Lower_Side",     [0.18, -0.57,   0.86]),
    (57, "Mouth_Center_Bottom", "Right_Lip_Lower_Side",    [-0.18,-0.57,   0.86]),

    # === Cheeks (JNR) ===
    (58, "Maxilla",             "Left_Cheek",              [0.55, -0.3,    0.6]),
    (59, "Maxilla",             "Right_Cheek",             [-0.55,-0.3,    0.6]),

    # === Chin sides & Jaw sides (JNR) ===
    (60, "Chin",                "Left_Chin",               [0.25, -0.85,   0.65]),
    (61, "Chin",                "Right_Chin",              [-0.25,-0.85,   0.65]),
    (62, "Jaw_Front",           "Left_Jaw_Root",           [0.5,  -0.6,    0.4]),
    (63, "Jaw_Front",           "Right_Jaw_Root",          [-0.5, -0.6,    0.4]),

    # === Crown outer (JNR) ===
    (64, "Crown",               "Left_Crown_Outer",        [0.5,   2.0,    0.0]),
    (65, "Crown",               "Right_Crown_Outer",       [-0.5,  2.0,    0.0]),
]


def create_face_rig_hybrid(scale=1.0, offset=(0, 0, 0)):
    """
    Create 66-joint hybrid face rig (JNR 52 + ASM Anim 14) in Maya.
    """
    cmds.select(clear=True)
    created = {}

    name_to_data = {}
    for idx, parent, name, pos in BONE_DATA:
        name_to_data[name] = (idx, parent, pos)

    ordered = []
    placed = set()

    def place(name):
        if name in placed:
            return
        _, parent, _ = name_to_data[name]
        if parent and parent not in placed:
            place(parent)
        ordered.append(name)
        placed.add(name)

    for _, _, name, _ in BONE_DATA:
        place(name)

    print("Creating {} joints (Hybrid: JNR 52 + ASM Anim 14)...".format(len(ordered)))

    for name in ordered:
        idx, parent, pos_maya = name_to_data[name]
        pos = (
            pos_maya[0] * scale + offset[0],
            pos_maya[1] * scale + offset[1],
            pos_maya[2] * scale + offset[2],
        )
        if parent and parent in created:
            cmds.select(created[parent])
        else:
            cmds.select(clear=True)
        jnt = cmds.joint(name="jnt_{}".format(name), position=pos)
        created[name] = jnt

    root_jnt = created["Root"]
    cmds.select(root_jnt)
    cmds.joint(root_jnt, edit=True, orientJoint="xyz",
               secondaryAxisOrient="yup", children=True,
               zeroScaleOrient=True)

    print("\nDone! {} joints created.".format(len(created)))
    print("Root joint: {}".format(root_jnt))
    print("\nJoint composition:")
    print("  JNR base:          52 joints (expression modeling)")
    print("  ASM anim additions: 12 joints (fine-grained animation)")
    print("    + eye_inner/outer_corner L/R  (4) — squint, crow's feet")
    print("    + lip_upper/lower_side L/R    (4) — asymmetric lip")
    print("    + nose_wing L/R               (2) — nostril flare")
    print("    + eyebrow_mid L/R             (2) — brow arch")
    print("\nNext steps:")
    print("  1. Select root joint, scale/translate to fit mesh")
    print("  2. Select root joint + mesh shape")
    print("  3. Skin > Bind Skin (Closest Distance)")

    return created


# ============================================================
# Run
# ============================================================
SCALE = 1.0
OFFSET = (0.0, 0.0, 0.0)

joints = create_face_rig_hybrid(scale=SCALE, offset=OFFSET)
