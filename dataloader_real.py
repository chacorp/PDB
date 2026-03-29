"""
dataloader_real.py — Real npy vertex dataset for EDD training.

Loads raw .npy vertices (not PCA-reconstructed) directly from disk.
Supports MF SEN/ROM datasets with train/val/test splits.

Returns per-sample: (vertices, template, vertices_normal, template_normal, faces)
"""
import os
import sys
import glob
import pickle
import numpy as np
import igl
import torch
from torch.utils import data

__abs_path__ = str(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, __abs_path__)

from utils.remesh_utils import procrustes_LDM


def get_data_splits():
    """Re-use existing split definition."""
    from dataloader_CBD import get_data_splits as _get_splits
    return _get_splits()


class RealNpyDataset(data.Dataset):
    """
    Dataset that loads raw .npy vertex files for EDD training.

    Args:
        data_basedir : base directory (e.g. '/data/sihun')
        datasets     : list of dataset names to include, e.g. ['mf_ROM', 'mf_SEN']
        mode         : 'train', 'val', or 'test'
        toggle       : if True, templates from {basedir} else {basedir}/pca
    """

    def __init__(
        self,
        data_basedir='/data/sihun',
        datasets=None,
        mode='train',
        toggle=True,
    ):
        super().__init__()
        if datasets is None:
            datasets = ['mf_ROM']

        self.mode = mode
        self.data_basedir = data_basedir
        template_basedir = data_basedir if toggle else os.path.join(data_basedir, 'pca')

        _, _, _, mf_data_split, _ = get_data_splits()

        self.file_list = []     # [(npy_path, id_name, dataset_name), ...]
        self.templates = {}     # id_name → np.ndarray [V, 3]
        self.faces_dict = {}    # dataset_key → {'np': ndarray, 't': tensor}
        self.std_dict = {}      # dataset_key → standardization dict

        _, voca_data_split, biwi_data_split, mf_data_split, _ = get_data_splits()

        for ds in datasets:
            if ds in ('mf_SEN', 'mf_ROM'):
                self._load_mf(ds, mf_data_split, data_basedir, template_basedir)
            elif ds == 'biwi':
                self._load_biwi(biwi_data_split, data_basedir, template_basedir)
            elif ds in ('voca', 'coma'):
                self._load_voca_coma(ds, voca_data_split, data_basedir, template_basedir)
            else:
                raise NotImplementedError(f"Dataset '{ds}' not supported yet")

        print(f"[RealNpyDataset] mode={mode}, datasets={datasets}, "
              f"total frames={len(self.file_list)}")

    def _get_faces(self, ds_key, tmpl_dict, std_file):
        """Get or cache faces for a dataset."""
        if ds_key not in self.faces_dict:
            if 'face' in tmpl_dict:
                faces_np = np.array(tmpl_dict['face'], dtype=np.int32)
            else:
                std = np.load(std_file, allow_pickle=True).item()
                faces_np = np.array(std['new_f'], dtype=np.int32)
            self.faces_dict[ds_key] = {
                'np': faces_np,
                't': torch.tensor(faces_np).long(),
            }
        return self.faces_dict[ds_key]

    def _load_mf(self, ds, mf_data_split, data_basedir, template_basedir):
        """Load Multiface SEN or ROM real npy data."""
        sub = 'SEN' if ds == 'mf_SEN' else 'ROM'
        base = os.path.join(data_basedir, 'multiface_align', sub, self.mode)

        # Templates
        pkl_path = os.path.join(template_basedir, 'multiface_align', 'mf_templates.pkl')
        if not os.path.exists(pkl_path):
            pkl_path = os.path.join(data_basedir, 'pca', 'multiface_align', 'mf_templates.pkl')
        with open(pkl_path, 'rb') as f:
            tmpl_dict = pickle.load(f)

        # Faces
        std_file = os.path.join(__abs_path__, 'utils', 'mf', 'standardization.npy')
        self._get_faces('mf', tmpl_dict, std_file)

        # Scan for npy files per identity
        id_list = mf_data_split.get(self.mode, [])
        for id_name in id_list:
            if id_name in tmpl_dict and id_name not in self.templates:
                self.templates[id_name] = tmpl_dict[id_name].astype(np.float32)

            npy_dir = os.path.join(base, 'vertices_npy', id_name)
            if not os.path.isdir(npy_dir):
                continue
            for seq_dir in sorted(os.listdir(npy_dir)):
                seq_path = os.path.join(npy_dir, seq_dir)
                if not os.path.isdir(seq_path):
                    continue
                for npy_file in sorted(glob.glob(os.path.join(seq_path, '*.npy'))):
                    self.file_list.append((npy_file, id_name, 'mf'))

    def _load_biwi(self, biwi_data_split, data_basedir, template_basedir):
        """Load BIWI real npy data."""
        base = os.path.join(data_basedir, 'BIWI_align_deci', self.mode)
        pkl_path = os.path.join(template_basedir, 'BIWI_align_deci', 'templates_align_deci.pkl')
        with open(pkl_path, 'rb') as f:
            tmpl_dict = pickle.load(f)

        std_file = os.path.join(__abs_path__, 'utils', 'biwi', 'standardization.npy')
        self._get_faces('biwi', tmpl_dict, std_file)

        for id_name in biwi_data_split.get(self.mode, []):
            if id_name in tmpl_dict and id_name not in self.templates:
                self.templates[id_name] = tmpl_dict[id_name].astype(np.float32)

            npy_dir = os.path.join(base, 'vertices_npy')
            for npy_file in sorted(glob.glob(os.path.join(npy_dir, f'{id_name}*.npy'))):
                self.file_list.append((npy_file, id_name, 'biwi'))

    def _load_voca_coma(self, ds, voca_data_split, data_basedir, template_basedir):
        """Load VOCA or COMA real npy data."""
        sub = 'VOCASET' if ds == 'voca' else 'COMA'
        base = os.path.join(data_basedir, 'VOCA-COMA', sub, self.mode)
        pkl_path = os.path.join(template_basedir, 'VOCA-COMA', 'voca_templates.pkl')
        with open(pkl_path, 'rb') as f:
            tmpl_dict = pickle.load(f)

        std_file = os.path.join(__abs_path__, 'utils', 'voca', 'standardization.npy')
        self._get_faces('voca', tmpl_dict, std_file)

        for id_name in voca_data_split.get(self.mode, []):
            if id_name in tmpl_dict and id_name not in self.templates:
                self.templates[id_name] = tmpl_dict[id_name].astype(np.float32)

            for seq_dir in sorted(glob.glob(os.path.join(base, id_name, 'vertices_npy', '*'))):
                if not os.path.isdir(seq_dir):
                    continue
                for npy_file in sorted(glob.glob(os.path.join(seq_dir, '*.npy'))):
                    self.file_list.append((npy_file, id_name, 'voca'))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        npy_path, id_name, ds_key = self.file_list[index]

        # Load vertices
        vertices_np = np.load(npy_path, allow_pickle=True).astype(np.float32)

        # Template
        template_np = self.templates[id_name]

        # Faces for this dataset
        faces_info = self.faces_dict[ds_key]
        faces_np = faces_info['np']
        faces_t = faces_info['t']

        # Procrustes alignment (same as EvalDataset)
        R, t, _ = procrustes_LDM(vertices_np, template_np)
        vertices_np = (vertices_np @ R.T + t).astype(np.float32)

        # Normals (igl requires float64 + int64)
        template_normal = igl.per_vertex_normals(
            np.asarray(template_np, dtype=np.float64),
            np.asarray(faces_np, dtype=np.int64),
        ).astype(np.float32)
        vertices_normal = igl.per_vertex_normals(
            np.asarray(vertices_np, dtype=np.float64),
            np.asarray(faces_np, dtype=np.int64),
        ).astype(np.float32)

        # mesh_data label (matches CBDDataBatch_eval expectation)
        _ds_to_label = {'mf': torch.tensor(2), 'biwi': torch.tensor(1),
                        'voca': torch.tensor(0)}
        mesh_label = _ds_to_label.get(ds_key, torch.tensor(-1))

        return (
            torch.tensor(vertices_np),         # [V, 3]
            torch.tensor(template_np),          # [V, 3]
            torch.tensor(vertices_normal),      # [V, 3]
            torch.tensor(template_normal),      # [V, 3]
            faces_t,                            # [F, 3]
            mesh_label,                         # scalar
        )

    def get_data_config(self):
        return f"[RealNpyDataset] mode={self.mode}, frames={len(self.file_list)}\n"


def real_collate_fn(batch):
    """Collate list of (vertices, template, v_normal, t_normal, faces) into batched tensors."""
    from dataloader_CBD import CBDDataBatch_eval
    return CBDDataBatch_eval(batch)


if __name__ == '__main__':
    ds = RealNpyDataset(
        data_basedir='/data/sihun',
        datasets=['mf_ROM'],
        mode='train',
        toggle=True,
    )
    print(f"Train: {len(ds)} frames")
    sample = ds[0]
    print(f"  vertices: {sample[0].shape}, template: {sample[1].shape}, faces: {sample[4].shape}")

    ds_val = RealNpyDataset(
        data_basedir='/data/sihun',
        datasets=['mf_ROM'],
        mode='val',
        toggle=True,
    )
    print(f"Val: {len(ds_val)} frames")
