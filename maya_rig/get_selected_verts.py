"""
Maya Python script: Print selected vertex IDs and generate reselection command.

Usage:
    1. Select vertices on mesh (lower lip etc.)
    2. Run this script — prints vertex IDs
    3. Copy the RESELECT command at the bottom
    4. After rebinding with different rig, paste to reselect same vertices

Output:
    - Vertex ID list
    - Maya select command for reselection
"""

import maya.cmds as cmds

sel = cmds.ls(selection=True, flatten=True)
if not sel:
    cmds.error("No vertices selected!")

# Extract mesh name and vertex indices
mesh_name = None
vert_ids = []

for s in sel:
    if '.vtx[' in s:
        mesh = s.split('.vtx[')[0]
        vid = int(s.split('.vtx[')[1].replace(']', ''))
        if mesh_name is None:
            mesh_name = mesh
        vert_ids.append(vid)

vert_ids.sort()

print("=" * 60)
print("Mesh: {}".format(mesh_name))
print("Selected vertices: {} total".format(len(vert_ids)))
print("=" * 60)
print("\nVertex IDs:")
print(vert_ids)

# Generate reselect command
vert_strs = ["{}.vtx[{}]".format(mesh_name, v) for v in vert_ids]
select_cmd = 'cmds.select([{}], replace=True)'.format(
    ', '.join(['"{}"'.format(v) for v in vert_strs]))

print("\n" + "=" * 60)
print("RESELECT COMMAND (copy-paste after rebind):")
print("=" * 60)
print(select_cmd)

# Also store as variable for immediate reuse
LOWER_LIP_VERTS = vert_ids
print("\nStored as LOWER_LIP_VERTS = [...]")
print("To reselect later: cmds.select(LOWER_LIP_VERTS_CMD)")
