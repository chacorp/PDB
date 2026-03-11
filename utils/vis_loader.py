"""
CheckpointVisLoader
===================
Loads fixed evaluation frames from raw .npy files (no PCA augmentation) and
runs model forward + visualization at checkpoint save time.

Usage (in train loop):
    vis_loader = CheckpointVisLoader(opts, device)
    ...
    if epoch % opts.save_interval == 0:
        vis_loader.visualize(model_lbs, model_disp, epoch, save_dir)
"""
import os
import glob
import pickle
import numpy as np
import torch
import igl
import yaml

from utils.mesh_utils import calc_norm_torch
from utils.matplotlib_rnd import plot_image_array


# Dataset label ints used by NGBC forward (matches CBDDataset encoding)
_MESH_DATA_INT = {
    'voca':   0,
    'biwi':   1,
    'mf_SEN': 2,
    'mf_ROM': 2,   # ROM is re-coded as 2 (same topology as SEN)
}

# Seg files relative to repo root
_SEG_FILES = {
    'voca':   'utils/voca/flame_seg_24.npy',
    'biwi':   'utils/biwi/biwi_seg_24.npy',
    'mf_SEN': 'utils/mf/mf_seg_24.npy',
    'mf_ROM': 'utils/mf/mf_seg_24.npy',
}

# Standardization files (for faces after remeshing) relative to repo root
_STD_FILES = {
    'voca':   'utils/voca/standardization.npy',
    'biwi':   None,   # BIWI uses face from template pkl directly
    'mf_SEN': 'utils/mf/standardization.npy',
    'mf_ROM': 'utils/mf/standardization.npy',
}


def _build_npy_path(entry, data_basedir):
    ds   = entry['dataset']
    splt = entry['split']
    idn  = entry['id_name']
    seq  = entry['sequence']
    frm  = int(entry['frame'])

    if ds == 'mf_SEN':
        subset = 'SEN'
        base = os.path.join(data_basedir, 'multiface_align', subset, splt, 'vertices_npy', idn, seq)
        files = sorted(glob.glob(os.path.join(base, '*.npy')))
    elif ds == 'mf_ROM':
        subset = 'ROM'
        base = os.path.join(data_basedir, 'multiface_align', subset, splt, 'vertices_npy', idn, seq)
        files = sorted(glob.glob(os.path.join(base, '*.npy')))
    elif ds == 'biwi':
        # BIWI: {biwi_base}/{split}/vertices_npy/{id_name}_{sequence}/*.npy
        base = os.path.join(data_basedir, 'BIWI_align_deci', splt, 'vertices_npy', f'{idn}_{seq}')
        files = sorted(glob.glob(os.path.join(base, '*.npy')))
    elif ds == 'voca':
        base = os.path.join(data_basedir, 'VOCA-COMA', 'VOCASET', splt, idn, 'vertices_npy', seq)
        files = sorted(glob.glob(os.path.join(base, '*.npy')))
    else:
        raise ValueError(f'Unknown dataset: {ds}')

    if not files:
        return None
    if frm >= len(files):
        frm = len(files) - 1
    return files[frm]


class CheckpointVisLoader:
    """
    Loads raw evaluation frames and visualizes reconstruction at checkpoint saves.
    Templates / faces / segmentations are loaded once at init.
    Raw .npy files are loaded fresh each time visualize() is called.
    """

    def __init__(self, opts, device='cpu'):
        self.opts = opts
        self.device = device
        self.frames = []       # list of dicts with loaded static data + path
        self._enabled = False

        vis_yml = getattr(opts, 'vis_frames', None)
        if not vis_yml or not os.path.exists(vis_yml):
            print(f'[VisLoader] vis_frames not set or not found ({vis_yml}), checkpoint vis disabled.')
            return

        with open(vis_yml, 'r') as f:
            cfg = yaml.safe_load(f)

        raw_entries = cfg.get('frames', [])
        if not raw_entries:
            print('[VisLoader] No frames defined in vis_frames.yml, checkpoint vis disabled.')
            return

        # Resolve data_basedir (same logic as CBDDataset with data_toggle)
        data_basedir = getattr(opts, 'data_basedir', '/data/sihun')
        toggle = getattr(opts, 'data_toggle', False)
        template_basedir = data_basedir if toggle else data_basedir + '/pca'

        # Cache: template pkl, faces, seg per dataset (loaded once)
        _template_cache = {}
        _faces_cache = {}
        _seg_cache = {}
        _std_cache = {}

        def _get_static(ds):
            if ds in _template_cache:
                return _template_cache[ds], _faces_cache[ds], _seg_cache[ds]

            seg = torch.tensor(np.load(_SEG_FILES[ds])).float()

            std_file = _STD_FILES[ds]
            if std_file and os.path.exists(std_file):
                std = np.load(std_file, allow_pickle=True).item()
                faces_np = np.array(std['new_f'])
            else:
                faces_np = None   # will fill from template pkl

            if ds in ('mf_SEN', 'mf_ROM'):
                pkl_path = os.path.join(template_basedir, 'multiface_align', 'mf_templates.pkl')
            elif ds == 'voca':
                pkl_path = os.path.join(template_basedir, 'VOCA-COMA', 'voca_templates.pkl')
            elif ds == 'biwi':
                pkl_path = os.path.join(template_basedir, 'BIWI_align_deci', 'templates_align_deci.pkl')
            else:
                raise ValueError(f'Unknown dataset: {ds}')

            with open(pkl_path, 'rb') as f:
                tmpl_dict = pickle.load(f)

            if faces_np is None:
                faces_np = np.array(tmpl_dict['face'])

            _template_cache[ds] = tmpl_dict
            _faces_cache[ds] = torch.tensor(faces_np).long()
            _seg_cache[ds] = seg
            return tmpl_dict, _faces_cache[ds], seg

        for entry in raw_entries:
            ds  = entry.get('dataset', '')
            idn = entry.get('id_name', '')
            exp = entry.get('expression', 'unknown')
            label = f"{ds}/{idn[:8]}/{exp}"

            if 'FILL' in idn or 'FILL' in entry.get('sequence', ''):
                continue   # skip unfilled placeholders

            if ds not in _MESH_DATA_INT:
                print(f'[VisLoader] Unknown dataset "{ds}", skipping.')
                continue

            npy_path = _build_npy_path(entry, data_basedir)
            if npy_path is None or not os.path.exists(npy_path):
                print(f'[VisLoader] Frame not found for {label} ({npy_path}), skipping.')
                continue

            tmpl_dict, faces, seg = _get_static(ds)
            if idn not in tmpl_dict:
                print(f'[VisLoader] id_name "{idn}" not found in template pkl for {ds}, skipping.')
                continue

            template_np = np.array(tmpl_dict[idn]).astype(np.float32)
            mesh_data_int = _MESH_DATA_INT[ds]

            self.frames.append({
                'label':        label,
                'dataset':      ds,
                'npy_path':     npy_path,
                'template_np':  template_np,
                'faces':        faces,
                'seg':          seg,
                'mesh_data':    torch.tensor(mesh_data_int),
            })

        if self.frames:
            self._enabled = True
            print(f'[VisLoader] Loaded {len(self.frames)} vis frames.')
        else:
            print('[VisLoader] No valid vis frames found after filtering, checkpoint vis disabled.')

    # ------------------------------------------------------------------
    def _load_frame(self, entry):
        """Load one raw .npy frame, apply procrustes alignment, compute normals."""
        from utils.remesh_utils import procrustes_LDM

        vertices_np = np.load(entry['npy_path']).astype(np.float32)
        template_np = entry['template_np']
        faces_np = entry['faces'].numpy()

        # Procrustes alignment (same as CBDDataset eval getters)
        try:
            R, t, _ = procrustes_LDM(vertices_np, template_np)
            vertices_np = vertices_np @ R.T + t
        except Exception:
            pass  # skip alignment if it fails (shape mismatch etc.)

        template_n = igl.per_vertex_normals(template_np, faces_np)
        vertices_n = igl.per_vertex_normals(vertices_np, faces_np)

        return {
            'template':          torch.tensor(template_np).float(),
            'vertices':          torch.tensor(vertices_np).float(),
            'template_normal':   torch.tensor(template_n).float(),
            'vertices_normal':   torch.tensor(vertices_n).float(),
            'faces':             entry['faces'],
            'seg':               entry['seg'],
            'mesh_data':         entry['mesh_data'],
            'label':             entry['label'],
        }

    # ------------------------------------------------------------------
    @torch.no_grad()
    def visualize(self, model_lbs, model_disp, epoch, save_dir, use_strain=False,
                  strain_dim=1, strain_full_grad=False, no_t_mask=False,
                  smooth_n_iter=0):
        """
        Run forward on all vis frames and save a grid image.
        Called only at checkpoint save — loading is done fresh here, no persistent GPU tensors.
        """
        if not self._enabled:
            return

        from utils.mesh_utils import compute_vertex_strain

        os.makedirs(save_dir, exist_ok=True)

        v_list, f_list, labels = [], [], []

        for entry in self.frames:
            try:
                sample = self._load_frame(entry)
            except Exception as e:
                print(f'[VisLoader] Error loading {entry["label"]}: {e}')
                continue

            template_v = sample['template'].unsqueeze(0).to(self.device)      # [1,V,3]
            vertices_v = sample['vertices'].unsqueeze(0).to(self.device)
            template_n = sample['template_normal'].unsqueeze(0).to(self.device)
            vertices_n = sample['vertices_normal'].unsqueeze(0).to(self.device)
            faces      = sample['faces'].to(self.device)
            mesh_data  = sample['mesh_data'].to(self.device)

            # ── LBS forward ──────────────────────────────────────────────
            pred_lbs, _, _, _, _, t_mask, _, _, _, _ = model_lbs(
                template_v, vertices_v, template_n, vertices_n,
                mesh_data, epoch=epoch
            )

            # ── Strain (optional) ─────────────────────────────────────────
            strain = None
            if use_strain:
                lbs_for_strain = pred_lbs.detach() if not strain_full_grad else pred_lbs
                strain = compute_vertex_strain(
                    lbs_for_strain, template_v, faces, return_trace=(strain_dim == 2)
                )
                if strain_dim == 2:
                    strain = torch.cat(list(strain), dim=-1)

            # ── LBS normals ───────────────────────────────────────────────
            lbs_norm = calc_norm_torch(pred_lbs, faces, at='verts')

            # ── DispNet forward ───────────────────────────────────────────
            displacement, _ = model_disp(pred_lbs, lbs_norm, strain=strain)

            # ── Composition ───────────────────────────────────────────────
            if not no_t_mask:
                displacement = displacement * t_mask
            pred_full = pred_lbs + displacement

            faces_cpu = faces.cpu()
            v_list += [
                vertices_v[0].cpu(),   # GT
                pred_lbs[0].cpu(),     # LBS output
                pred_full[0].cpu(),    # LBS + displacement
            ]
            f_list += [faces_cpu, faces_cpu, faces_cpu]
            labels += [f'{entry["label"]}/GT',
                       f'{entry["label"]}/LBS',
                       f'{entry["label"]}/pred']

        if not v_list:
            return

        name = f'vis_ckpt_{epoch:03d}'
        plot_image_array(
            v_list, f_list,
            rot_list=[[0, 0, 0]] * len(v_list),
            size=1, bg_black=False, mode='shade',
            logdir=save_dir, name=name, save=True
        )
        print(f'[VisLoader] Saved checkpoint vis: {save_dir}/{name}.png')
