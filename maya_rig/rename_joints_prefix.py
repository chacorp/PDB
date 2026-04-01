"""
Maya Python script: Add prefix to all joints under selected hierarchy.

Usage:
    1. Select the root joint
    2. Run this script
"""

import maya.cmds as cmds

PREFIX = "hybrid_"

sel = cmds.ls(selection=True, type='joint')
if not sel:
    cmds.error("Please select a root joint first!")

root = sel[0]
children = cmds.listRelatives(root, allDescendents=True, type='joint', fullPath=True) or []
root_full = cmds.ls(root, long=True)[0]
all_joints = [root_full] + children

# Rename bottom-up (deepest children first) to avoid path invalidation
all_joints.sort(key=lambda x: x.count('|'), reverse=True)

renamed = 0
for jnt in all_joints:
    short = jnt.split("|")[-1]
    if not short.startswith(PREFIX):
        new_name = PREFIX + short
        cmds.rename(jnt, new_name)
        print("Renamed: {} -> {}".format(short, new_name))
        renamed += 1

print("\nDone! Renamed {} joints with prefix '{}'".format(renamed, PREFIX))
