"""Build maya_rig/hybrid_v3 — v3 helper roster (36 helpers, J=82).
Run from repo root. Copies hybrid (v1 state) -> hybrid_v3, then applies v3."""
import json, os, shutil, glob
import numpy as np

SRC, DST = 'maya_rig/hybrid', 'maya_rig/hybrid_v3'
if os.path.exists(DST): shutil.rmtree(DST)
os.makedirs(DST)
for f in glob.glob(f'{SRC}/*'):
    if os.path.isfile(f): shutil.copy2(f, DST)
print(f'copied {len(os.listdir(DST))} files -> {DST}')

UPPER = ['Chin','Right_Chin','Left_Chin','Nose_Root_A','Nose_Root_B','Nose_Nostril']
LOWER = ['Nose_Bridge','Nose_Ridge_A','Nose_Ridge_B','Ears','Left_Ear_Rotate_Scale','Right_Ear_Rotate_Scale']
BROWS = ['Brow_Outer','Eyebrow_Mid','Eyebrow_Inner']
def full(n): return f'hybrid_jnt_{n}'
def premat(pos):
    return [[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0],[0.0,0.0,1.0,0.0],
            [-pos[0],-pos[1],-pos[2],1.0]]

NEW = []  # (name, parent_name or special)
NEW.append(('Left_Brow_Root',  'Left_Orbital'))
NEW.append(('Right_Brow_Root', 'Right_Orbital'))
for s in ('Left','Right'):
    for k in (1,2): NEW.append((f'helper_extra_{s}_Brow_Root_{k}', f'{s}_Brow_Root'))
for s in ('Left','Right'):
    for k in (1,2): NEW.append((f'helper_extra_{s}_Eye_Socket_{k}', f'{s}_Eye_Socket'))
for s in ('Left','Right'):
    for k in (1,2): NEW.append((f'helper_extra_{s}_Cheek_{k}', f'{s}_Cheek'))
NEW.append(('helper_extra_Left_Nose_1',  'Nose_Base_Rotate'))
NEW.append(('helper_extra_Right_Nose_1', 'Nose_Base_Rotate'))
assert len(NEW) == 16

for topo in ('ict','mf'):
    rp = f'{DST}/rig_info_{topo}.json'
    rig = json.load(open(rp))
    J = rig['joints']
    bak = json.load(open(f'{SRC}/rig_info_{topo}.json.bak_pre_helpers'))
    n2i = {j['name']: i for i, j in enumerate(J)}

    # 1) Nose_Base_Position: restore original hierarchy/bind from pre-v1 backup
    nbp = n2i[full('Nose_Base_Position')]
    for key in ('parent_index','bind_world_pos','bind_pre_matrix'):
        J[nbp][key] = bak['joints'][nbp][key]

    # 2) mouth helpers -> lip-level parents, bind reset to new parent
    for names, pname in ((UPPER,'Mouth_Center_Top'), (LOWER,'Mouth_Center_Bottom')):
        p = n2i[full(pname)]; ppos = J[p]['bind_world_pos']
        for n in names:
            i = n2i[full(n)]
            J[i]['parent_index'] = p
            J[i]['bind_world_pos'] = list(ppos)
            J[i]['bind_pre_matrix'] = premat(ppos)

    # 3) append 16 new joints (bind = parent pos). Brow_Root first so extras resolve.
    for name, pname in NEW:
        p = n2i[full(pname)]
        pos = list(J[p]['bind_world_pos'])
        idx = len(J)
        J.append({'index': idx, 'name': full(name), 'parent_index': p,
                  'bind_world_pos': pos, 'bind_pre_matrix': premat(pos)})
        n2i[full(name)] = idx

    # 4) eyebrow leaves -> Brow_Root (keep their own binds)
    for s in ('Left','Right'):
        br = n2i[full(f'{s}_Brow_Root')]
        for b in BROWS:
            i = n2i[full(f'{s}_{b}')]
            J[i]['parent_index'] = br

    json.dump(rig, open(rp,'w'), indent=1)
    print(f'{topo}: J={len(J)}')

    # 5) skin weights zero-pad
    swp = f'{DST}/skin_weights_{topo}.npy'
    W = np.load(swp)
    W2 = np.concatenate([W, np.zeros((W.shape[0], 16), dtype=W.dtype)], axis=1)
    np.save(swp, W2); print(f'  skin_weights: {W.shape} -> {W2.shape}')

# 6) sigma targets extend (default 0.4 for new helpers)
for f in ('sigma_targets.npy','sigma_targets_tight.npy'):
    p = f'{DST}/{f}'
    if os.path.exists(p):
        s = np.load(p); s2 = np.concatenate([s, np.full(16, 0.4, dtype=s.dtype)])
        np.save(p, s2); print(f'{f}: {len(s)} -> {len(s2)}')

# 7) active_joints_manual.json (v3)
ajp = f'{DST}/active_joints_manual.json'
aj = json.load(open(ajp))
rig = json.load(open(f'{DST}/rig_info_ict.json'))
n2i = {j['name']: i for i, j in enumerate(rig['joints'])}
old_helpers = set(aj['helper_joint_idx'])
old_helpers.discard(n2i[full('Nose_Base_Position')])      # NBP -> base
new_idx = [n2i[full(n)] for n, _ in NEW]
helpers = sorted(old_helpers | set(new_idx))
face = sorted(set(aj['face_joint_idx']) | set(new_idx))    # NBP stays in face (base)
aj['face_joint_idx'] = face
aj['helper_joint_idx'] = helpers
aj['frozen_joint_idx'] = sorted(set(aj['frozen_joint_idx']) - set(helpers))
aj['counts'] = {'active': len(face), 'frozen': len(aj['frozen_joint_idx']), 'helper': len(helpers)}
json.dump(aj, open(ajp,'w'), indent=2)
print(f'active_json: face={len(face)} helper={len(helpers)} frozen={len(aj["frozen_joint_idx"])}')

# 8) helper_joints_v3.json (parents for record + mirror pairs)
pairs = [
 ['Cranium','Crown'], ['Left_Crown_Outer','Right_Crown_Outer'],
 ['Nose_Root_A','Nose_Root_B'], ['Nose_Ridge_A','Nose_Ridge_B'],
 ['Left_Ear_Position','Right_Ear_Position'], ['Left_Jaw_Root','Right_Jaw_Root'],
 ['Left_Chin','Right_Chin'], ['Left_Ear_Rotate_Scale','Right_Ear_Rotate_Scale'],
 ['Left_Brow_Root','Right_Brow_Root'],
 ['helper_extra_Left_Brow_Root_1','helper_extra_Right_Brow_Root_1'],
 ['helper_extra_Left_Brow_Root_2','helper_extra_Right_Brow_Root_2'],
 ['helper_extra_Left_Eye_Socket_1','helper_extra_Right_Eye_Socket_1'],
 ['helper_extra_Left_Eye_Socket_2','helper_extra_Right_Eye_Socket_2'],
 ['helper_extra_Left_Cheek_1','helper_extra_Right_Cheek_1'],
 ['helper_extra_Left_Cheek_2','helper_extra_Right_Cheek_2'],
 ['helper_extra_Left_Nose_1','helper_extra_Right_Nose_1'],
]
hv3 = {
 'version': 4,
 'description': 'v3 (2026-07-08) — 36 helpers / J=82. Leaf-level parenting: mouth helpers as siblings of lip leaves (Mouth_Center_Top/Bottom); NEW L/R_Brow_Root helper above GT-supervised eyebrow leaves + 2 fine/side; +2/side eye (eyelid level); +2/side lower-cheek (L/R_Cheek, nasolabial); +1 pair nose (Nose_Base_Rotate level). Nose_Base_Position restored to base. Over-provision now, prune by influence later.',
 'default_sigma': 0.4, 'default_init_anchor_lambda': 0.1,
 'helpers': [{'joint': full(n), 'parent': p} for n, p in NEW] ,
 'mirror_pairs': [[full(a), full(b)] for a, b in pairs],
}
json.dump(hv3, open(f'{DST}/helper_joints_v3.json','w'), indent=2)
print('helper_joints_v3.json written | mirror pairs:', len(pairs))
print('BUILD OK')
