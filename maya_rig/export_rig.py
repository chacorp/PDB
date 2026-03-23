"""
Maya Python script: Export rig hierarchy, bind matrices, and skin weights.
Run this inside Maya Script Editor (Python tab) after:
    1. Rig is created (create_face_rig.py)
    2. Joint positions have been manually adjusted
    3. Bind Skin has been applied to the mesh

Output:
    maya_rig/exported/rig_info.json  -- joint hierarchy + bindPreMatrix per joint
    maya_rig/exported/skin_weights.npy  -- [V, J] float32 skin weights

Usage:
    1. Set OUTPUT_DIR below (absolute path)
    2. Set MESH_NAME to your mesh transform node name in Maya
    3. Set ROOT_JOINT to the root joint name (e.g. 'jnt_root')
    4. Paste and execute in Maya Script Editor (Python tab)

Notes:
    - Joints must be exported in topological order (parent before child).
      The script does a depth-first traversal to ensure this.
    - bindPreMatrix is the inverse bind-pose world matrix stored in the
      skinCluster node. This is exactly B_inv needed for standard LBS:
          G_j = T_world_j @ B_inv_j
          v' = sum_j  w_j * (G_j @ [v;1])[:3]
    - Coordinate system: Maya Y-up. Since MF mesh was imported into Maya
      as-is, the exported matrices are in the same space as MF vertex data.
      No coordinate conversion needed when loading in Python training code.
"""

import json
import os
import maya.cmds as cmds

# ============================================================
# USER CONFIG — edit these before running
# ============================================================
OUTPUT_DIR  = '/source/inyup/NeuralFacialAnimation/maya_rig/exported'
MESH_NAME   = 'mf_aligned_mean'   # mesh transform node name in Maya
ROOT_JOINT  = 'jnt_root'          # root joint name
# ============================================================


def get_skin_cluster(mesh_name):
    """Return the skinCluster node attached to mesh_name."""
    history = cmds.listHistory(mesh_name, pruneDagObjects=True) or []
    clusters = cmds.ls(history, type='skinCluster')
    if not clusters:
        raise RuntimeError(
            f"No skinCluster found on '{mesh_name}'. "
            "Apply Skin > Bind Skin first."
        )
    return clusters[0]


def get_joints_topological(root_joint):
    """
    Return all joints in the hierarchy rooted at root_joint,
    in topological (parent-before-child) order via depth-first traversal.
    """
    ordered = []
    stack = [root_joint]
    while stack:
        jnt = stack.pop(0)  # BFS to keep parents before children
        ordered.append(jnt)
        children = cmds.listRelatives(jnt, children=True, type='joint') or []
        stack = children + stack
    return ordered


def mat4_to_list(mat_flat):
    """Convert Maya's flat 16-element matrix to nested 4x4 list (row-major)."""
    return [
        [mat_flat[r*4 + c] for c in range(4)]
        for r in range(4)
    ]


def export_rig_info(skin_cluster, joints, joint_to_idx):
    """
    Build joint info dict with hierarchy and bindPreMatrix.

    bindPreMatrix[i] in a skinCluster = inverse bind-pose world matrix
    of the i-th influence = B_inv_j we need for LBS.
    """
    # Map influence (joint) name to its index in the skinCluster influence list
    influences = cmds.skinCluster(skin_cluster, query=True, influence=True)
    inf_to_sc_idx = {inf: i for i, inf in enumerate(influences)}

    joint_list = []
    for jnt in joints:
        idx = joint_to_idx[jnt]

        # Parent index (-1 for root)
        parents = cmds.listRelatives(jnt, parent=True, type='joint') or []
        parent_name = parents[0] if parents else None
        parent_idx  = joint_to_idx[parent_name] if parent_name else -1

        # Bind-pose world position (translation column of world matrix at bind)
        # We use xform to get the world-space position at current (bind) pose
        world_pos = cmds.xform(jnt, query=True, worldSpace=True, translation=True)

        # bindPreMatrix = B_inv (inverse bind world matrix)
        sc_idx = inf_to_sc_idx.get(jnt)
        if sc_idx is None:
            raise RuntimeError(
                f"Joint '{jnt}' is not an influence of skinCluster '{skin_cluster}'. "
                "Make sure all joints are bound."
            )
        bind_pre_flat = cmds.getAttr(f'{skin_cluster}.bindPreMatrix[{sc_idx}]')
        bind_pre_4x4  = mat4_to_list(bind_pre_flat)

        joint_list.append({
            'index':            idx,
            'name':             jnt,
            'parent_index':     parent_idx,
            'bind_world_pos':   list(world_pos),     # [x, y, z]
            'bind_pre_matrix':  bind_pre_4x4,         # 4x4 B_inv
        })

    return {'joints': joint_list}


def export_skin_weights(skin_cluster, mesh_name, joints, joint_to_idx):
    """
    Export skin weights as [V, J] array.

    Returns a list-of-lists (converted to numpy in the save step).
    Column order matches joint_to_idx.
    """
    num_joints = len(joints)

    # Get vertex count
    vtx_count = cmds.polyEvaluate(mesh_name, vertex=True)

    # influences in skinCluster order
    influences = cmds.skinCluster(skin_cluster, query=True, influence=True)
    inf_to_col = {}
    for inf in influences:
        if inf in joint_to_idx:
            inf_to_col[inf] = joint_to_idx[inf]

    weights = [[0.0] * num_joints for _ in range(vtx_count)]

    print(f"Exporting weights for {vtx_count} vertices x {num_joints} joints ...")
    for vi in range(vtx_count):
        vtx_sel = f'{mesh_name}.vtx[{vi}]'
        for inf, w in zip(
            cmds.skinPercent(skin_cluster, vtx_sel, query=True, transform=None),
            cmds.skinPercent(skin_cluster, vtx_sel, query=True, value=True),
        ):
            col = inf_to_col.get(inf)
            if col is not None:
                weights[vi][col] = w
        if vi % 500 == 0:
            print(f"  ... vertex {vi}/{vtx_count}")

    return weights


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=== NeuralFacialAnimation Rig Export ===")
    print(f"Mesh      : {MESH_NAME}")
    print(f"Root joint: {ROOT_JOINT}")
    print(f"Output dir: {OUTPUT_DIR}")

    # 1. Collect joints in topological order
    joints = get_joints_topological(ROOT_JOINT)
    joint_to_idx = {jnt: i for i, jnt in enumerate(joints)}
    print(f"Found {len(joints)} joints")

    # 2. Get skin cluster
    skin_cluster = get_skin_cluster(MESH_NAME)
    print(f"SkinCluster: {skin_cluster}")

    # 3. Export rig info (hierarchy + B_inv)
    rig_info = export_rig_info(skin_cluster, joints, joint_to_idx)
    rig_path = os.path.join(OUTPUT_DIR, 'rig_info.json')
    with open(rig_path, 'w') as f:
        json.dump(rig_info, f, indent=2)
    print(f"Saved rig_info.json  ({len(joints)} joints)")

    # 4. Export skin weights
    weights = export_skin_weights(skin_cluster, MESH_NAME, joints, joint_to_idx)

    # Save as numpy via eval (Maya has numpy available in newer versions,
    # but we write a plain Python list and convert outside if needed)
    import numpy as np
    import numpy as np
    W = np.array(weights, dtype='float32')   # [V, J]
    w_path = os.path.join(OUTPUT_DIR, 'skin_weights.npy')
    np.save(w_path, W)
    print(f"Saved skin_weights.npy  shape={W.shape}  sum_check={W.sum(axis=1).mean():.4f} (should be ~1.0)")

    print("\n=== Export complete ===")
    print(f"  {rig_path}")
    print(f"  {w_path}")


main()
