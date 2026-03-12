"""
CheckpointVisLoader
===================
Loads fixed evaluation frames from raw .npy files (no PCA augmentation) and
runs model forward + visualization at eval_iter intervals.

Produces stitched panel images (like eval_CBD) with strain/displacement heatmaps.

Usage (in train loop):
    vis_loader = CheckpointVisLoader(opts, device)
    ...
    if epoch % opts.eval_iter == 0:
        vis_loader.visualize(model_lbs, model_disp, epoch, save_dir, ...)
"""
import os
import glob
import pickle
import numpy as np
import torch
import igl
import yaml

from utils.mesh_utils import calc_norm_torch, compute_strain_signal, taubin_smooth_np
from utils.matplotlib_rnd import vis_mesh_key_weight


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
    Loads raw evaluation frames and visualizes reconstruction at eval_iter.
    Templates / faces / segmentations are loaded once at init.
    Raw .npy files are loaded fresh each time visualize() is called.
    """

    def __init__(self, opts, device='cpu'):
        self.opts = opts
        self.device = device
        self.frames = []       # list of dicts with loaded static data + path
        self._enabled = False

        vis_yml = getattr(opts, 'vis_frames', 'config/vis_frames.yml')
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

        # Cache: template pkl, faces, seg per dataset (loaded once)
        _template_cache = {}
        _faces_cache = {}
        _seg_cache = {}

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
                pkl_path = os.path.join(data_basedir, 'multiface_align', 'mf_templates.pkl')
            elif ds == 'voca':
                pkl_path = os.path.join(data_basedir, 'VOCA-COMA', 'voca_templates.pkl')
            elif ds == 'biwi':
                pkl_path = os.path.join(data_basedir, 'BIWI_align_deci', 'templates_align_deci.pkl')
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
            seq = entry.get('sequence', 'unknown')
            frm = entry.get('frame', 0)
            label = f"{seq}/f{frm}"

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
                'faces_np':     faces.numpy().astype(np.int32),
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
        faces_np = entry['faces_np']

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
            'faces_np':          faces_np,
            'seg':               entry['seg'],
            'mesh_data':         entry['mesh_data'],
            'label':             entry['label'],
        }

    # ------------------------------------------------------------------
    @torch.no_grad()
    def visualize(self, model_lbs, model_disp, epoch, save_dir,
                  mode='stage_disp', stage=1,
                  strain_mode='norm', smooth_n_iter=16,
                  no_t_mask=False, use_source_template=False):
        """
        Run forward on all vis frames and save stitched panel images.

        Args:
            mode: 'stage_disp' or 'disp_only'
            stage: 1 or 2 (for stage_disp mode)
            strain_mode: strain signal mode
            smooth_n_iter: Taubin smoothing iterations for smooth_GT
        """
        if not self._enabled:
            return

        from PIL import Image

        os.makedirs(save_dir, exist_ok=True)
        tmp_dir = os.path.join(save_dir, '_tmp')
        os.makedirs(tmp_dir, exist_ok=True)

        model_lbs.eval()
        model_disp.eval()

        for fi, entry in enumerate(self.frames):
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
            faces_np   = sample['faces_np']

            # Compute smooth_GT via Taubin smoothing on real frame
            gt_np = sample['vertices'].numpy()
            smooth_np = taubin_smooth_np(gt_np, faces_np, n_iter=smooth_n_iter)
            smooth_v = torch.tensor(smooth_np).float().unsqueeze(0).to(self.device)

            # ── LBS forward ──
            pred_lbs, _, _, _, _, t_mask, _, _, _, _ = model_lbs(
                template_v, vertices_v, template_n, vertices_n,
                mesh_data, epoch=epoch
            )

            # ── Strain signals ──
            gt_strain = compute_strain_signal(
                smooth_v, template_v, faces, mode=strain_mode)
            pred_strain = compute_strain_signal(
                pred_lbs, template_v, faces, mode=strain_mode)
            gt_snorm = gt_strain.norm(dim=-1)[0].cpu().numpy()    # [V]
            pred_snorm = pred_strain.norm(dim=-1)[0].cpu().numpy()

            # ── DispNet forward ──
            lbs_norm = calc_norm_torch(pred_lbs, faces, at='verts')
            displacement, _ = model_disp(
                pred_lbs, lbs_norm,
                source_vert=template_v if use_source_template else None,
                source_norm=template_n if use_source_template else None,
                strain=pred_strain if mode != 'disp_only' else gt_strain,
            )
            if not no_t_mask:
                displacement = displacement * t_mask
            pred_full = pred_lbs + displacement

            # ── Numpy for visualization ──
            gt_v_np = gt_np
            sv_np = smooth_np
            pl_np = pred_lbs[0].cpu().numpy()
            pf_np = pred_full[0].cpu().numpy()
            disp_mag = displacement[0].norm(dim=-1).cpu().numpy()

            # ── Build panel specs ──
            if mode == 'disp_only':
                # GT | smoothGT | smoothGT+GT strain | GT pred | GT pred+disp
                panel_specs = [
                    (gt_v_np, np.zeros(gt_v_np.shape[0])[:, None], 'YlOrRd', 'GT'),
                    (sv_np,   np.zeros(sv_np.shape[0])[:, None],   'YlOrRd', 'smooth_GT'),
                    (sv_np,   gt_snorm[:, None],                   'coolwarm', 'sGT+GT_strain'),
                    (pf_np,   np.zeros(pf_np.shape[0])[:, None],   'YlOrRd', 'GT_pred'),
                    (pf_np,   disp_mag[:, None],                   'YlOrRd', 'GT_pred+disp'),
                ]
            elif stage == 1:
                # GT | smoothGT | smoothGT+GT strain | pred smoothGT | pred+pred strain
                panel_specs = [
                    (gt_v_np, np.zeros(gt_v_np.shape[0])[:, None], 'YlOrRd', 'GT'),
                    (sv_np, np.zeros(sv_np.shape[0])[:, None], 'YlOrRd', 'smooth_GT'),
                    (sv_np, gt_snorm[:, None],                 'coolwarm', 'sGT+GT_strain'),
                    (pl_np, np.zeros(pl_np.shape[0])[:, None], 'YlOrRd', 'pred_sGT'),
                    (pl_np, pred_snorm[:, None],               'coolwarm', 'pred+pred_strain'),
                ]
            else:
                # Stage 2: GT | smoothGT | smoothGT+GT strain | pred+pred strain | GT pred | GT pred+disp
                panel_specs = [
                    (gt_v_np, np.zeros(gt_v_np.shape[0])[:, None], 'YlOrRd', 'GT'),
                    (sv_np,   np.zeros(sv_np.shape[0])[:, None],   'YlOrRd', 'smooth_GT'),
                    (sv_np,   gt_snorm[:, None],                   'coolwarm', 'sGT+GT_strain'),
                    (pl_np,   pred_snorm[:, None],                 'coolwarm', 'pred+pred_strain'),
                    (pf_np,   np.zeros(pf_np.shape[0])[:, None],   'YlOrRd', 'GT_pred'),
                    (pf_np,   disp_mag[:, None],                   'YlOrRd', 'GT_pred+disp'),
                ]

            # ── Render & stitch ──
            panels = []
            for verts, weights, cmap, title in panel_specs:
                tmp_path = os.path.join(tmp_dir, f'{epoch:03d}_{fi:03d}_{title}.png')
                vmax = max(float(np.percentile(weights, 95)), 1e-6)
                vis_mesh_key_weight(
                    verts, faces_np, weights, cage_idx=0,
                    cmap=cmap, vmin=0, vmax=vmax,
                    view_yrots=(0,),
                    save_path=tmp_path, close=True, title=title,
                    shade=True,
                )
                panels.append(Image.open(tmp_path))
                os.remove(tmp_path)

            total_w = sum(p.width for p in panels)
            max_h = max(p.height for p in panels)
            stitched = Image.new('RGB', (total_w, max_h), (255, 255, 255))
            x_off = 0
            for p in panels:
                stitched.paste(p, (x_off, 0))
                x_off += p.width

            safe_label = entry['label'].replace('/', '_')
            stitched.save(os.path.join(save_dir, f"{epoch:03d}_{safe_label}.png"))

        print(f'[VisLoader] Saved {len(self.frames)} eval vis images at epoch {epoch}')
