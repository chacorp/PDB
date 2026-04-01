"""
Maya Python script: Mirror Left joints to Right joints under selected hierarchy.
Copies Left joint positions and mirrors X axis (negate X).

Usage:
    1. Select the root joint of the hierarchy to mirror within
    2. Make sure all Left_* joints under it are properly positioned
    3. Run this script in Maya Script Editor (Python tab)

Naming convention: "Left_" <-> "Right_", "Left" <-> "Right"
"""

import maya.cmds as cmds


def get_all_children(root_jnt):
    """Get all descendant joints under root_jnt."""
    children = cmds.listRelatives(root_jnt, allDescendents=True, type='joint') or []
    return [root_jnt] + children


# Get selected joint as hierarchy root
sel = cmds.ls(selection=True, type='joint')
if not sel:
    cmds.error("Please select a root joint first!")

root = sel[0]
all_joints = get_all_children(root)
all_names = set(all_joints)

print("Mirroring within hierarchy of: {} ({} joints)".format(root, len(all_joints)))

mirrored = 0
skipped = 0

for jnt in all_joints:
    # Find Left joints and their Right counterparts
    short = jnt.split("|")[-1]  # handle namespaces

    if "Left_" in short:
        right_jnt = short.replace("Left_", "Right_")
    elif "_Left_" in short:
        right_jnt = short.replace("_Left_", "_Right_")
    elif "_L_" in short:
        right_jnt = short.replace("_L_", "_R_")
    elif short.endswith("_L"):
        right_jnt = short[:-2] + "_R"
    else:
        continue  # not a Left joint

    if right_jnt not in all_names:
        if not cmds.objExists(right_jnt):
            print("SKIP: {} -> {} not found".format(short, right_jnt))
            skipped += 1
            continue

    # Get Left joint world position
    pos = cmds.xform(short, query=True, worldSpace=True, translation=True)

    # Mirror: negate X, keep Y and Z
    mirrored_pos = [-pos[0], pos[1], pos[2]]

    # Set Right joint position
    cmds.xform(right_jnt, worldSpace=True, translation=mirrored_pos)
    print("Mirrored: {} ({:.4f}, {:.4f}, {:.4f}) -> {} ({:.4f}, {:.4f}, {:.4f})".format(
        short, pos[0], pos[1], pos[2],
        right_jnt, mirrored_pos[0], mirrored_pos[1], mirrored_pos[2]))
    mirrored += 1

print("\nDone! Mirrored: {}, Skipped: {}".format(mirrored, skipped))
