"""
Maya Python script: Build Hybrid rig from existing JNR + ASM rigs in scene.

Steps:
    1. Duplicate JNR hierarchy with 'hybrid_' prefix
    2. Find 12 ASM anim joints, unparent them
    3. Reparent under appropriate hybrid joints

Prerequisites:
    - JNR rig in scene with 'jnr_jnt_*' naming (or 'jnt_*')
    - ASM rig in scene with 'asm_jnt_*' naming (or 'jnt_*')

Usage:
    1. Make sure both JNR and ASM rigs exist in scene
    2. Set JNR_PREFIX and ASM_PREFIX below to match your naming
    3. Run in Script Editor
"""

import maya.cmds as cmds

# ============================================================
# CONFIG — adjust these to match your scene naming
# ============================================================
JNR_ROOT = "jnr_jnt_Root"       # JNR root joint name
ASM_PREFIX = "asm_"              # prefix on ASM joints

# 12 ASM joints to add, with their target hybrid parent
# Format: (asm_joint_name, hybrid_parent_name, new_hybrid_name)
ASM_TO_HYBRID = [
    # Eye corners
    ("asm_jnt_eye_inner_corner_L",  "hybrid_jnt_Left_Eye_Socket",     "hybrid_jnt_Left_Eye_Inner_Corner"),
    ("asm_jnt_eye_outer_corner_L",  "hybrid_jnt_Left_Eye_Socket",     "hybrid_jnt_Left_Eye_Outer_Corner"),
    ("asm_jnt_eye_inner_corner_R",  "hybrid_jnt_Right_Eye_Socket",    "hybrid_jnt_Right_Eye_Inner_Corner"),
    ("asm_jnt_eye_outer_corner_R",  "hybrid_jnt_Right_Eye_Socket",    "hybrid_jnt_Right_Eye_Outer_Corner"),

    # Lip sides
    ("asm_jnt_lip_upper_side_L",    "hybrid_jnt_Mouth_Center_Top",    "hybrid_jnt_Left_Lip_Upper_Side"),
    ("asm_jnt_lip_upper_side_R",    "hybrid_jnt_Mouth_Center_Top",    "hybrid_jnt_Right_Lip_Upper_Side"),
    ("asm_jnt_lip_lower_side_L",    "hybrid_jnt_Mouth_Center_Bottom", "hybrid_jnt_Left_Lip_Lower_Side"),
    ("asm_jnt_lip_lower_side_R",    "hybrid_jnt_Mouth_Center_Bottom", "hybrid_jnt_Right_Lip_Lower_Side"),

    # Nose wings
    ("asm_jnt_nose_wing_L",         "hybrid_jnt_Nose_Base_Rotate",    "hybrid_jnt_Left_Nose_Wing"),
    ("asm_jnt_nose_wing_R",         "hybrid_jnt_Nose_Base_Rotate",    "hybrid_jnt_Right_Nose_Wing"),

    # Eyebrow mid
    ("asm_jnt_eyebrow_mid_L",       "hybrid_jnt_Left_Orbital",        "hybrid_jnt_Left_Eyebrow_Mid"),
    ("asm_jnt_eyebrow_mid_R",       "hybrid_jnt_Right_Orbital",       "hybrid_jnt_Right_Eyebrow_Mid"),

    # Eyebrow inner
    ("asm_jnt_eyebrow_inner_L",     "hybrid_jnt_Left_Orbital",        "hybrid_jnt_Left_Eyebrow_Inner"),
    ("asm_jnt_eyebrow_inner_R",     "hybrid_jnt_Right_Orbital",       "hybrid_jnt_Right_Eyebrow_Inner"),
]


def duplicate_hierarchy(root_jnt, prefix="hybrid_"):
    """Duplicate joint hierarchy and add prefix to all joints."""
    # Duplicate
    dup = cmds.duplicate(root_jnt, returnRootsOnly=True, renameChildren=True)[0]

    # Get all joints in duplicated hierarchy
    all_jnts = [dup] + (cmds.listRelatives(dup, allDescendents=True, type='joint', fullPath=True) or [])
    # Rename bottom-up
    all_jnts.sort(key=lambda x: x.count('|'), reverse=True)

    for jnt in all_jnts:
        short = jnt.split("|")[-1]
        # Remove duplicate suffix (e.g., "jnt_Root1" -> "jnt_Root")
        clean = short
        for i in range(10, 0, -1):
            if clean.endswith(str(i)):
                clean = clean[:-len(str(i))]
                break
        # Remove existing prefix if any, then add hybrid_
        if clean.startswith("jnr_"):
            clean = clean[4:]
        new_name = prefix + clean
        cmds.rename(jnt, new_name)

    hybrid_root = prefix + "jnt_Root"
    print("Duplicated JNR hierarchy as hybrid: {}".format(hybrid_root))
    return hybrid_root


def add_asm_joints():
    """Find ASM joints, unparent, rename, and reparent under hybrid."""
    added = 0
    skipped = 0

    for asm_name, hybrid_parent, hybrid_name in ASM_TO_HYBRID:
        # Check ASM joint exists
        if not cmds.objExists(asm_name):
            print("SKIP: ASM joint '{}' not found".format(asm_name))
            skipped += 1
            continue

        # Check hybrid parent exists
        if not cmds.objExists(hybrid_parent):
            print("SKIP: Hybrid parent '{}' not found".format(hybrid_parent))
            skipped += 1
            continue

        # Get world position of ASM joint
        pos = cmds.xform(asm_name, query=True, worldSpace=True, translation=True)

        # Create new joint under hybrid parent
        cmds.select(hybrid_parent)
        new_jnt = cmds.joint(name=hybrid_name, position=pos)

        print("Added: {} -> parent={} (from {})".format(
            hybrid_name, hybrid_parent, asm_name))
        added += 1

    print("\nAdded {} ASM joints, Skipped {}".format(added, skipped))
    return added


# ============================================================
# Run
# ============================================================
print("=" * 60)
print("Building Hybrid Rig (JNR 52 + ASM 12 = 64)")
print("=" * 60)

# Step 1: Duplicate JNR hierarchy
if not cmds.objExists(JNR_ROOT):
    cmds.error("JNR root '{}' not found! Set JNR_ROOT correctly.".format(JNR_ROOT))

hybrid_root = duplicate_hierarchy(JNR_ROOT, prefix="hybrid_")

# Step 2: Add 12 ASM animation joints
add_asm_joints()

# Step 3: Orient all joints
cmds.select(hybrid_root)
cmds.joint(hybrid_root, edit=True, orientJoint="xyz",
           secondaryAxisOrient="yup", children=True,
           zeroScaleOrient=True)

# Count total
all_hybrid = [hybrid_root] + (cmds.listRelatives(hybrid_root, allDescendents=True, type='joint') or [])
print("\n" + "=" * 60)
print("Hybrid rig complete: {} joints".format(len(all_hybrid)))
print("Root: {}".format(hybrid_root))
print("=" * 60)
