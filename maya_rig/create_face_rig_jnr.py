"""
Maya Python script: Create 52-bone face rig following JNR paper (Vesdapunt et al., 2020).
Run this inside Maya Script Editor (Python tab).

Usage:
    1. Open Maya
    2. File > Import > ict_mean_face.obj (or mf_aligned_mean.obj)
    3. Open Script Editor (Windows > General Editors > Script Editor)
    4. Paste this entire script in Python tab
    5. Execute
    6. Manually adjust root joint to align with mesh
    7. Skin > Bind Skin

Skeleton structure from:
    Vesdapunt et al. 2020, "JNR: Joint-based Neural Rig Representation
    for Compact 3D Face Modeling" - Supplementary Table 1 & Figure 1

Key advantages over ASM (84 bones):
    - Jaw_Front joint for jaw opening
    - Upper/Lower Eyelid joints for eye blink
    - Top_Lip / Bottom_Lip separation
    - Cheek joints
    - More compact (52 vs 84)
"""

import maya.cmds as cmds


# ============================================================
# JNR Supplementary Table 1: 52 joints
# Positions approximate, based on paper Figure 1
# Coordinate system: Maya Y-up (X=right, Y=up, Z=forward)
# ============================================================

BONE_DATA = [
    # (index, parent_name, bone_name, maya_position [x, y, z])

    # === Root, Neck, Skull ===
    (0,  None,                  "Root",                    [0.0,   0.0,    0.0]),
    (1,  "Root",                "Neck",                    [0.0,  -1.8,   -0.3]),
    (2,  "Root",                "Skull_Root",              [0.0,   0.5,    0.0]),
    (3,  "Skull_Root",          "Cranium",                 [0.0,   1.5,    0.0]),
    (4,  "Cranium",             "Crown",                   [0.0,   2.0,    0.0]),
    (5,  "Skull_Root",          "Skull_Center",            [0.0,   0.8,    0.0]),

    # === Brow ===
    (6,  "Skull_Center",        "Brow_Center",             [0.0,   0.9,    0.8]),

    # === Left Eye Region ===
    (7,  "Skull_Center",        "Left_Orbital",            [0.45,  0.7,    0.7]),
    (8,  "Left_Orbital",        "Left_Brow_Outer",         [0.7,   0.9,    0.5]),
    (9,  "Left_Orbital",        "Left_Eye_Socket",         [0.45,  0.6,    0.8]),
    (10, "Left_Eye_Socket",     "Left_Upper_Eyelid",       [0.45,  0.7,    0.9]),
    (11, "Left_Eye_Socket",     "Left_Lower_Eyelid",       [0.45,  0.5,    0.9]),
    (12, "Left_Orbital",        "Left_Cheek_Bone",         [0.6,   0.3,    0.7]),

    # === Right Eye Region ===
    (13, "Skull_Center",        "Right_Orbital",           [-0.45, 0.7,    0.7]),
    (14, "Right_Orbital",       "Right_Brow_Outer",        [-0.7,  0.9,    0.5]),
    (15, "Right_Orbital",       "Right_Eye_Socket",        [-0.45, 0.6,    0.8]),
    (16, "Right_Eye_Socket",    "Right_Upper_Eyelid",      [-0.45, 0.7,    0.9]),
    (17, "Right_Eye_Socket",    "Right_Lower_Eyelid",      [-0.45, 0.5,    0.9]),
    (18, "Right_Orbital",       "Right_Cheek_Bone",        [-0.6,  0.3,    0.7]),

    # === Nose (upper, connected to skull) ===
    (19, "Skull_Center",        "Nose_Root_A",             [0.0,   0.5,    0.9]),
    (20, "Nose_Root_A",         "Nose_Root_B",             [0.0,   0.4,    0.95]),
    (21, "Nose_Root_B",         "Nose_Bridge",             [0.0,   0.2,    1.0]),
    (22, "Nose_Bridge",         "Nose_Ridge_A",            [0.0,   0.0,    1.05]),
    (23, "Nose_Ridge_A",        "Nose_Ridge_B",            [0.0,  -0.1,    1.1]),

    # === Ears ===
    (24, "Skull_Root",          "Ears",                    [0.0,   0.3,    0.0]),
    (25, "Ears",                "Left_Ear_Position",       [0.9,   0.3,    0.0]),
    (26, "Left_Ear_Position",   "Left_Ear_Rotate_Scale",   [0.95,  0.3,    0.0]),
    (27, "Ears",                "Right_Ear_Position",      [-0.9,  0.3,    0.0]),
    (28, "Right_Ear_Position",  "Right_Ear_Rotate_Scale",  [-0.95, 0.3,    0.0]),

    # === Lower Face ===
    (29, "Skull_Root",          "Lower_Face_Root",         [0.0,  -0.1,    0.5]),
    (30, "Lower_Face_Root",     "Jaw_Root",                [0.0,  -0.3,    0.3]),
    (31, "Lower_Face_Root",     "Maxilla",                 [0.0,  -0.2,    0.7]),

    # === Nose (lower, connected to maxilla) ===
    (32, "Maxilla",             "Nose_Base_Position",      [0.0,  -0.15,   1.0]),
    (33, "Nose_Base_Position",  "Nose_Base_Rotate",        [0.0,  -0.2,    1.05]),
    (34, "Nose_Base_Rotate",    "Nose_Tip",                [0.0,  -0.25,   1.15]),
    (35, "Nose_Base_Rotate",    "Nose_Nostril",            [0.0,  -0.3,    1.0]),

    # === Mouth & Jaw ===
    (36, "Maxilla",             "Mouth_Center_Top",        [0.0,  -0.5,    0.9]),
    (37, "Mouth_Center_Top",    "Mouth_Left_Corner",       [0.35, -0.5,    0.8]),
    (38, "Mouth_Center_Top",    "Mouth_Right_Corner",      [-0.35,-0.5,    0.8]),
    (39, "Mouth_Center_Top",    "Jaw_Front",               [0.0,  -0.6,    0.85]),
    (40, "Jaw_Front",           "Chin",                    [0.0,  -0.8,    0.7]),
    (41, "Jaw_Front",           "Top_Lip",                 [0.0,  -0.5,    0.95]),
    (42, "Mouth_Center_Top",    "Mouth_Center_Bottom",     [0.0,  -0.55,   0.88]),
    (43, "Mouth_Center_Bottom", "Bottom_Lip",              [0.0,  -0.6,    0.92]),

    # === Cheeks ===
    (44, "Maxilla",             "Left_Cheek",              [0.55, -0.3,    0.6]),
    (45, "Maxilla",             "Right_Cheek",             [-0.55,-0.3,    0.6]),

    # === Chin sides & Jaw sides ===
    (46, "Chin",                "Left_Chin",               [0.25, -0.85,   0.65]),
    (47, "Chin",                "Right_Chin",              [-0.25,-0.85,   0.65]),
    (48, "Jaw_Front",           "Left_Jaw_Root",           [0.5,  -0.6,    0.4]),
    (49, "Jaw_Front",           "Right_Jaw_Root",          [-0.5, -0.6,    0.4]),

    # === Crown outer ===
    (50, "Crown",               "Left_Crown_Outer",        [0.5,   2.0,    0.0]),
    (51, "Crown",               "Right_Crown_Outer",       [-0.5,  2.0,    0.0]),
]


def create_face_rig_jnr(scale=1.0, offset=(0, 0, 0)):
    """
    Create 52-joint face rig in Maya following JNR paper.

    Args:
        scale: uniform scale to roughly match target mesh size
        offset: (x, y, z) translation after scaling
    """
    cmds.select(clear=True)
    created = {}

    # Build name→data lookup
    name_to_data = {}
    for idx, parent, name, pos in BONE_DATA:
        name_to_data[name] = (idx, parent, pos)

    # Topological sort
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

    print("Creating {} joints (JNR Paper Table 1)...".format(len(ordered)))

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

    # Orient joints
    root_jnt = created["Root"]
    cmds.select(root_jnt)
    cmds.joint(root_jnt, edit=True, orientJoint="xyz",
               secondaryAxisOrient="yup", children=True,
               zeroScaleOrient=True)

    print("\nDone! {} joints created.".format(len(created)))
    print("Root joint: {}".format(root_jnt))
    print("\nKey features (vs ASM 84-bone):")
    print("  + Jaw_Front: jaw opening")
    print("  + Upper/Lower Eyelid: eye blink")
    print("  + Top_Lip / Bottom_Lip: lip separation")
    print("  + Left/Right Cheek")
    print("  - Compact: 52 joints (vs 84)")
    print("\nNext steps:")
    print("  1. Select root joint, scale/translate to fit mesh")
    print("  2. Select root joint + mesh shape")
    print("  3. Skin > Bind Skin (Closest Distance)")
    print("  4. Paint weights or export for training")

    return created


# ============================================================
# Run
# ============================================================
# Adjust SCALE and OFFSET to fit your mesh.
# For ICT mean face (scale=0.1, z-offset=-0.5):
SCALE = 1.0
OFFSET = (0.0, 0.0, 0.0)

joints = create_face_rig_jnr(scale=SCALE, offset=OFFSET)
