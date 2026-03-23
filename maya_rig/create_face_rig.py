"""
Maya Python script: Create 84-bone face rig following ASM paper Table 8.
Run this inside Maya Script Editor (Python tab).

Usage:
    1. Open Maya
    2. File > Import > mf_aligned_mean.obj
    3. Open Script Editor (Windows > General Editors > Script Editor)
    4. Paste this entire script in Python tab
    5. Execute
    6. Manually adjust root joint to align with mesh
    7. Skin > Bind Skin

Skeleton structure from:
    Yang et al. 2023, "ASM: Adaptive Skinning Model for High-Quality 3D Face Modeling"
    Table 8 & Figure 11

Hierarchy: nearly flat (root → head → region bones)
    - 84 bones total
    - Most bones are direct children of 'head' (depth 2)
    - Sub-regions: nose(2), mouth(13), eye.L/R(23/50), eyebrow.L/R(34/61)

Bone positions: approximate, mapped from Rigify bone positions via semantic matching.
    Coordinate conversion: Blender (X,Y,Z) Z-up → Maya (X,Z,-Y) Y-up
"""

import maya.cmds as cmds


# ============================================================
# ASM Paper Table 8: 84 bones
# Positions in Blender space (X,Y,Z), will be converted to Maya
# ============================================================

BONE_DATA = [
    # (index, parent_name, bone_name, blender_position [x,y,z])

    # === Root & Head ===
    (0,  None,           "root",               [0.0, -0.0247, 1.7813]),
    (1,  "root",         "head",               [0.0, -0.0247, 1.8725]),

    # === Nose group (parent: nose, except nose=head) ===
    (2,  "head",         "nose",               [0.000799, -0.16522, 1.89747]),
    (3,  "nose",         "nose_bridge",        [4.36e-05, -0.1785, 1.87042]),
    (4,  "nose",         "nose_tip",           [0.01355, -0.16408, 1.83199]),
    (5,  "nose",         "nose_mid",           [1.21e-09, -0.17272, 1.82989]),
    (6,  "nose",         "nose_wing_L",        [0.0558, -0.14815, 1.84032]),
    (7,  "nose",         "nose_wing_R",        [-0.0558, -0.14815, 1.84032]),
    (8,  "nose",         "nose_bottom_L",      [0.04087, -0.15973, 1.83942]),
    (9,  "nose",         "nose_bottom_R",      [-0.04087, -0.15973, 1.83942]),
    (10, "nose",         "nose_hole_L",        [0.01872, -0.17327, 1.83866]),
    (11, "nose",         "nose_hole_R",        [-0.01872, -0.17327, 1.83866]),
    (12, "nose",         "nose_bridge_upper",  [0.000799, -0.16522, 1.89747]),

    # === Mouth group (parent: mouth, except mouth=head) ===
    (13, "head",         "mouth",              [0.0, -0.16748, 1.79481]),
    (14, "mouth",        "lip_corner_L",       [0.02254, -0.15742, 1.80596]),
    (15, "mouth",        "lip_upper_side_L",   [0.01069, -0.17002, 1.80225]),
    (16, "mouth",        "lip_upper_mid",      [0.0, -0.17167, 1.79861]),
    (17, "mouth",        "lip_lower_mid",      [0.0, -0.16329, 1.79101]),
    (18, "mouth",        "lip_lower_side_L",   [0.01069, -0.16428, 1.79389]),
    (19, "mouth",        "lip_corner_R",       [-0.02254, -0.15742, 1.80596]),
    (20, "mouth",        "lip_upper_side_R",   [-0.01069, -0.17002, 1.80225]),
    (21, "mouth",        "lip_lower_side_R",   [-0.01069, -0.16428, 1.79389]),

    # === Ear ===
    (22, "head",         "ear_L",              [0.07973, -0.01696, 1.84826]),

    # === Eye.L group (parent: eye_L, except eye_L=head) ===
    (23, "head",         "eye_L",              [0.04306, -0.12836, 1.88138]),
    (24, "eye_L",        "eye_inner_upper_L",  [0.02804, -0.1408, 1.88359]),
    (25, "eye_L",        "eye_outer_upper_L",  [0.05578, -0.12236, 1.88424]),
    (26, "eye_L",        "eye_outer_corner_L", [0.06088, -0.10832, 1.87697]),
    (27, "eye_L",        "eye_inner_lower_L",  [0.03691, -0.12388, 1.87593]),
    (28, "eye_L",        "eye_outer_lower_L",  [0.05578, -0.10876, 1.87487]),
    (29, "eye_L",        "eye_inner_corner_L", [0.02804, -0.1408, 1.88359]),
    (30, "eye_L",        "eye_hole_L",         [0.04306, -0.12836, 1.88138]),
    (31, "eye_L",        "eyelid_outer_L",     [0.0478, -0.13546, 1.88826]),
    (32, "eye_L",        "eyelid_middle_L",    [0.0478, -0.13546, 1.88826]),
    (33, "eye_L",        "eyelid_inner_L",     [0.03832, -0.14176, 1.8886]),

    # === Eyebrow.L group ===
    (34, "head",         "eyebrow_L",          [0.0465, -0.13264, 1.90605]),
    (35, "eyebrow_L",    "eyebrow_inner_L",    [0.02807, -0.14618, 1.90476]),
    (36, "eyebrow_L",    "eyebrow_outer_L",    [0.0688, -0.08357, 1.87924]),
    (37, "eyebrow_L",    "eyebrow_mid_L",      [0.05492, -0.12556, 1.90734]),

    # === Forehead ===
    (38, "head",         "forehead",           [0.0, -0.12879, 1.94575]),

    # === Apple (cheek area) L ===
    (39, "head",         "apple_outer_L",      [0.05485, -0.07987, 1.76605]),
    (40, "head",         "apple_lower_L",      [0.02665, -0.12254, 1.77345]),
    (41, "head",         "apple_inner_L",      [0.01774, -0.14035, 1.81424]),
    (42, "head",         "apple_center_L",     [0.04075, -0.1012, 1.78985]),

    # === Eyebrow center, chin, jaw, temple ===
    (43, "head",         "eyebrow_center",     [0.0, -0.14618, 1.90476]),
    (44, "head",         "chin",               [0.0, -0.14034, 1.76765]),
    (45, "head",         "chin_side_L",        [0.03859, -0.11011, 1.7867]),
    (46, "head",         "jaw_L",              [0.07589, -0.08704, 1.89244]),
    (47, "head",         "jaw_corner_L",       [0.07718, -0.0538, 1.80575]),
    (48, "head",         "temple_L",           [0.06088, -0.11672, 1.91419]),

    # === Right side mirrors ===
    (49, "head",         "ear_R",              [-0.07973, -0.01696, 1.84826]),
    (50, "head",         "eye_R",              [-0.04306, -0.12836, 1.88138]),
    (51, "eye_R",        "eye_inner_upper_R",  [-0.02804, -0.1408, 1.88359]),
    (52, "eye_R",        "eye_outer_upper_R",  [-0.05578, -0.12236, 1.88424]),
    (53, "eye_R",        "eye_outer_corner_R", [-0.06088, -0.10832, 1.87697]),
    (54, "eye_R",        "eye_inner_lower_R",  [-0.03691, -0.12388, 1.87593]),
    (55, "eye_R",        "eye_outer_lower_R",  [-0.05578, -0.10876, 1.87487]),
    (56, "eye_R",        "eye_inner_corner_R", [-0.02804, -0.1408, 1.88359]),
    (57, "eye_R",        "eye_hole_R",         [-0.04306, -0.12836, 1.88138]),
    (58, "eye_R",        "eyelid_outer_R",     [-0.0478, -0.13546, 1.88826]),
    (59, "eye_R",        "eyelid_middle_R",    [-0.0478, -0.13546, 1.88826]),
    (60, "eye_R",        "eyelid_inner_R",     [-0.03832, -0.14176, 1.8886]),

    (61, "head",         "eyebrow_R",          [-0.0465, -0.13264, 1.90605]),
    (62, "eyebrow_R",    "eyebrow_inner_R",    [-0.02807, -0.14618, 1.90476]),
    (63, "eyebrow_R",    "eyebrow_outer_R",    [-0.0688, -0.08357, 1.87924]),
    (64, "eyebrow_R",    "eyebrow_mid_R",      [-0.05492, -0.12556, 1.90734]),

    (65, "head",         "apple_outer_R",      [-0.05485, -0.07987, 1.76605]),
    (66, "head",         "apple_lower_R",      [-0.02665, -0.12254, 1.77345]),
    (67, "head",         "apple_inner_R",      [-0.01774, -0.14035, 1.81424]),
    (68, "head",         "apple_center_R",     [-0.04075, -0.1012, 1.78985]),
    (69, "head",         "chin_side_R",        [-0.03859, -0.11011, 1.7867]),
    (70, "head",         "jaw_R",              [-0.07589, -0.08704, 1.89244]),
    (71, "head",         "jaw_corner_R",       [-0.07718, -0.0538, 1.80575]),
    (72, "head",         "temple_R",           [-0.06088, -0.11672, 1.91419]),

    (73, "head",         "cheek_L",            [0.01338, -0.15637, 1.85696]),
    (74, "head",         "cheek_R",            [-0.01338, -0.15637, 1.85696]),
    (75, "head",         "chin_low",           [0.0, -0.15316, 1.78559]),
    (76, "head",         "chin_side_low_L",    [0.01688, -0.14931, 1.7797]),
    (77, "head",         "chin_side_low_R",    [-0.01688, -0.14931, 1.7797]),
    (78, "head",         "eyebrow_center_up",  [0.0, -0.12879, 1.94575]),
    (79, "head",         "forehead_L",         [0.02951, -0.13527, 1.93704]),
    (80, "head",         "forehead_R",         [-0.02951, -0.13527, 1.93704]),

    # === Neck (parent: root) ===
    (81, "root",         "neck_front",         [0.0, -0.02, 1.72]),
    (82, "root",         "neck_side_L",        [0.06, -0.01, 1.72]),
    (83, "root",         "neck_side_R",        [-0.06, -0.01, 1.72]),
]


def blender_to_maya(pos):
    """Blender (X, Y, Z) Z-up → Maya (X, Z, -Y) Y-up"""
    return (pos[0], pos[2], -pos[1])


def create_face_rig(scale=1.0, offset=(0, 0, 0)):
    """
    Create 84-joint face rig in Maya following ASM paper Table 8.

    Args:
        scale: uniform scale to roughly match target mesh size
        offset: (x, y, z) translation after scaling
    """
    cmds.select(clear=True)
    created = {}

    # Build name→data lookup and parent→children order
    name_to_data = {}
    for idx, parent, name, pos in BONE_DATA:
        name_to_data[name] = (idx, parent, pos)

    # Topological sort (parents before children)
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

    print("Creating {} joints (ASM Paper Table 8)...".format(len(ordered)))

    for name in ordered:
        idx, parent, pos_blender = name_to_data[name]

        # Convert Blender → Maya coordinates, then scale + offset
        mx, my, mz = blender_to_maya(pos_blender)
        pos = (
            mx * scale + offset[0],
            my * scale + offset[1],
            mz * scale + offset[2],
        )

        # Select parent joint if exists
        if parent and parent in created:
            cmds.select(created[parent])
        else:
            cmds.select(clear=True)

        # Create joint (Maya doesn't allow dots in names)
        jnt = cmds.joint(name="jnt_{}".format(name), position=pos)
        created[name] = jnt

    # Orient joints
    root_jnt = created["root"]
    cmds.select(root_jnt)
    cmds.joint(root_jnt, edit=True, orientJoint="xyz",
               secondaryAxisOrient="yup", children=True,
               zeroScaleOrient=True)

    print("\nDone! {} joints created.".format(len(created)))
    print("Root joint: {}".format(root_jnt))
    print("\nHierarchy depth distribution:")
    print("  depth 0 (root):     1")
    print("  depth 1 (head/neck): 4")
    print("  depth 2 (regions):  35")
    print("  depth 3 (sub-bones): 44")
    print("\nNext steps:")
    print("  1. Select root joint, scale/translate to fit mesh")
    print("  2. Select root joint + mesh shape")
    print("  3. Skin > Bind Skin (Closest Distance)")
    print("  4. Paint weights or export for training")

    return created


# ============================================================
# Run
# ============================================================
# Rough alignment for mf_aligned_mean.obj:
#   Bone space (Maya): X~[-0.11,0.11], Y~[1.73,1.98], Z~[-0.19,-0.02]
#   MF mesh:           X~[-0.93,0.91], Y~[-1.85,1.58], Z~[-1.48,0.77]
#
# Adjust SCALE and OFFSET after visual inspection.
SCALE = 5.6
OFFSET = (0.0, -10.6, -1.08)

joints = create_face_rig(scale=SCALE, offset=OFFSET)
