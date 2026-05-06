"""
dataloader_cross_pair.py — ICT cross-id paired dataset for Track B.

Each __getitem__ samples (id_A, id_B, z_exp) and synthesizes 4 ICT meshes
sharing the same expression:
    - src_neu  = ICT(id_A, zero_exp)
    - src_def  = ICT(id_A, z_exp)
    - tgt_neu  = ICT(id_B, zero_exp)
    - tgt_def  = ICT(id_B, z_exp)        ← cross-retarget GT

Used to supervise:
    pred_tgt_def = model.retarget(src_neu, src_def, tgt_neu, ...)
    L_cross      = MSE(pred_tgt_def, tgt_def)

Disentangles identity (W, bind_pose) from expression (joint transform):
the same z_exp must produce the same animation for any identity.
"""
import os
import sys
import numpy as np
import torch
import torch.utils.data as data
import igl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _sample_z_exp(expression_vecs, p_real=0.5, p_random_full=0.5):
    """50% real ICT-AU expression, 50% synthetic (matching get_ict CBD logic)."""
    if expression_vecs is not None and np.random.random() < p_real:
        idx = np.random.randint(len(expression_vecs))
        return np.asarray(expression_vecs[idx], dtype=np.float32)
    # synthetic
    if np.random.random() < p_random_full:
        return np.random.random(53).astype(np.float32)
    return np.where(np.random.random(53) > 0.9, 1.0, 0.0).astype(np.float32)


class CrossPairICTDataset(data.Dataset):
    """ICT cross-id pair dataset. Topology = ICT only (V=11248).

    Args:
        ict_face_model: utils.remesh_utils.ICT_face_model instance
        iden_vecs:      [N_id, 100] np.float32 identity coeffs (e.g. ict_id_vecs_test.pt)
        expression_vecs: optional [N_exp, 53] real ICT-AU expressions; pass None for synthetic-only
        mode:            'train' | 'eval'
        length:          override number of pairs per epoch (default = N_id * 53)
        seed:            optional fixed RNG seed for eval determinism
    """
    def __init__(self, ict_face_model, iden_vecs, expression_vecs=None,
                 mode='train', length=None, seed=None):
        self.ict_face_model = ict_face_model
        self.iden_vecs      = np.asarray(iden_vecs, dtype=np.float32)
        self.expression_vecs = np.asarray(expression_vecs, dtype=np.float32) \
            if expression_vecs is not None else None
        self.mode = mode
        self.N_id  = self.iden_vecs.shape[0]
        self.N_exp = self.expression_vecs.shape[0] if self.expression_vecs is not None else 53
        self.length = length if length is not None else (self.N_id * 53)
        self._rng = np.random.RandomState(seed) if seed is not None else np.random

        # Cache faces (shared across all ICT meshes)
        self.faces = ict_face_model.faces.astype(np.int32)

    def __len__(self):
        return self.length

    def _sample_pair(self):
        if self.mode == 'train':
            id_a, id_b = self._rng.choice(self.N_id, size=2, replace=False)
            z_exp = _sample_z_exp(self.expression_vecs)
        else:
            # deterministic for eval
            id_a = (np.arange(self.N_id) * 7) % self.N_id      # spread
            id_b = (id_a + 1) % self.N_id
            idx = self._sample_eval_idx
            id_a, id_b = int(id_a[idx % self.N_id]), int(id_b[idx % self.N_id])
            z_exp = self.expression_vecs[idx % self.N_exp] \
                if self.expression_vecs is not None else \
                np.where(np.arange(53) == (idx % 53), 1.5, 0.0).astype(np.float32)
        return id_a, id_b, z_exp

    def _synth(self, id_idx, z_exp):
        """Return (neutral [V,3], deformed [V,3]) np.float32."""
        id_coeff = self.iden_vecs[id_idx]
        zero_exp = np.zeros(53, dtype=np.float32)
        neutral, _, _ = self.ict_face_model.apply_coeffs(id_coeff, zero_exp, return_all=True)
        deformed, _, _ = self.ict_face_model.apply_coeffs(id_coeff, z_exp, return_all=True)
        return neutral[0].astype(np.float32), deformed[0].astype(np.float32)

    def __getitem__(self, index):
        if self.mode != 'train':
            self._sample_eval_idx = index
        id_a, id_b, z_exp = self._sample_pair()

        src_neu, src_def = self._synth(id_a, z_exp)
        tgt_neu, tgt_def = self._synth(id_b, z_exp)

        faces = self.faces
        src_neu_n = igl.per_vertex_normals(src_neu, faces).astype(np.float32)
        src_def_n = igl.per_vertex_normals(src_def, faces).astype(np.float32)
        tgt_neu_n = igl.per_vertex_normals(tgt_neu, faces).astype(np.float32)
        tgt_def_n = igl.per_vertex_normals(tgt_def, faces).astype(np.float32)

        return (
            torch.from_numpy(src_neu),                  # 0
            torch.from_numpy(src_def),                  # 1
            torch.from_numpy(tgt_neu),                  # 2
            torch.from_numpy(tgt_def),                  # 3
            torch.from_numpy(faces).long(),             # 4
            torch.from_numpy(src_neu_n),                # 5
            torch.from_numpy(src_def_n),                # 6
            torch.from_numpy(tgt_neu_n),                # 7
            torch.from_numpy(tgt_def_n),                # 8
            torch.from_numpy(z_exp),                    # 9
            f'ict_{id_a:03d}',                          # 10
            f'ict_{id_b:03d}',                          # 11
        )


class CrossPairBatch:
    """Batch container — analogous to CBDDataBatch but for cross-pair tuples."""
    def __init__(self, data_list):
        if data_list is None:
            return
        t = list(zip(*data_list))
        self.src_template        = torch.stack(t[0], 0)    # [B, V, 3]  src_neu
        self.src_vertices        = torch.stack(t[1], 0)    # [B, V, 3]  src_def
        self.tgt_template        = torch.stack(t[2], 0)    # [B, V, 3]  tgt_neu
        self.tgt_vertices        = torch.stack(t[3], 0)    # [B, V, 3]  tgt_def
        self.faces               = t[4][0]                  # [F, 3]   shared
        self.src_template_normal = torch.stack(t[5], 0)
        self.src_vertices_normal = torch.stack(t[6], 0)
        self.tgt_template_normal = torch.stack(t[7], 0)
        self.tgt_vertices_normal = torch.stack(t[8], 0)
        self.exp_coeff           = torch.stack(t[9], 0)    # [B, 53]
        self.src_id_name         = list(t[10])              # list[str]
        self.tgt_id_name         = list(t[11])

    def to(self, device='cpu'):
        for k in self.__dict__:
            attr = getattr(self, k)
            if isinstance(attr, torch.Tensor):
                setattr(self, k, attr.to(device))
        return self


def cross_pair_collate(batch_list, device='cpu'):
    return CrossPairBatch(batch_list).to(device)


# ── Quick sanity ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    from utils.remesh_utils import ICT_face_model
    ict = ICT_face_model()
    iden_vecs = torch.load('ict_face_pt/ict_id_vecs_test.pt').numpy()
    print(f'iden_vecs: {iden_vecs.shape}')

    ds = CrossPairICTDataset(ict, iden_vecs, expression_vecs=None,
                              mode='train', length=4, seed=42)
    print(f'len: {len(ds)}')
    item = ds[0]
    print(f'item[0] (src_neu): {item[0].shape}, dtype={item[0].dtype}')
    print(f'item[3] (tgt_def): {item[3].shape}')
    print(f'src_id_name={item[10]}, tgt_id_name={item[11]}')

    loader = torch.utils.data.DataLoader(ds, batch_size=2, num_workers=0,
                                          collate_fn=cross_pair_collate)
    batch = next(iter(loader))
    print(f'batch.src_template: {batch.src_template.shape}')
    print(f'batch.tgt_vertices: {batch.tgt_vertices.shape}')
    print(f'batch.exp_coeff:    {batch.exp_coeff.shape}')
    print('OK')
