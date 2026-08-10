"""
viser_debug.py — Live viser debug viewer for HLBS networks.

Three modes, one app:
  - bind_pose : per-id GT vs Pred joint position + error arrows
  - weight    : per-vertex skin weight heatmap (single / argmax / entropy)
  - anim      : skinned mesh animation driven by an ICT blendshape sequence,
                with joint axes triads (joint transform net output)

Usage:
    python tools/vis/viser_debug.py \
        --ckpt ckpts_hlbs/2026-05-14-14-02-03-HLBS-FullPred-ict-jTrans-nrm0.1-Wsm0.01 \
        --port 8080

Loads opts from ckpt/opts.json + train_opts.yml to mirror the trainer config
(active_joints_json, sigma_targets, helper_joint_idx, nfs_concat, etc.).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import igl
import matplotlib.cm as cm
import numpy as np
import torch
import yaml

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent.parent  # NeuralFacialAnimation/
sys.path.insert(0, str(_REPO))


def _ensure_viser():
    """Make `import viser` succeed on first run anywhere.

    Order: (1) already installed pip pkg, (2) bundled checkout at
    third_party/viser/src, (3) pip install latest from PyPI.
    """
    try:
        import viser  # noqa: F401
        return
    except ImportError:
        pass
    bundled_src = _REPO / "third_party" / "viser" / "src"
    if bundled_src.exists():
        sys.path.insert(0, str(bundled_src))
        try:
            import viser  # noqa: F401
            print(f"[viser] using bundled checkout at {bundled_src}")
            return
        except ImportError:
            # bundled checkout missing transitive deps (msgspec, websockets...).
            # Fall through to pip install.
            sys.path.pop(0)
    import subprocess
    print("[viser] not installed and bundled checkout unusable — pip install viser")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "viser"])
    import viser  # noqa: F401


_ensure_viser()
import viser

from models.hierarchical_lbs import HierarchicalLBS_FullPred
from utils.remesh_utils import ICT_face_model
from utils.rig_loader import load_rig


# ───────────────────────── ckpt / opts ──────────────────────────────────────


def _load_opts(ckpt_dir: Path) -> argparse.Namespace:
    """Mirror the eval pattern: peek opts.json + train_opts.yml."""
    opts = {}
    yml = ckpt_dir / "train_opts.yml"
    if yml.exists():
        with open(yml) as f:
            opts.update(yaml.safe_load(f) or {})
    js = ckpt_dir / "opts.json"
    if js.exists():
        with open(js) as f:
            opts.update(json.load(f) or {})
    return argparse.Namespace(**opts)


def _peek_ckpt_buffers(ckpt_path: Path):
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    face_joint_idx = sd["face_joint_idx"].tolist() if "face_joint_idx" in sd else None
    helper_joint_idx = (
        sd["helper_joint_idx_buf"].tolist() if "helper_joint_idx_buf" in sd else None
    )
    return sd, face_joint_idx, helper_joint_idx


def _build_model(ckpt_dir: Path, device: torch.device):
    opts = _load_opts(ckpt_dir)
    pth = ckpt_dir / "model_hlbs_best.pth"
    if not pth.exists():
        # pick highest-epoch pth
        cands = sorted(ckpt_dir.glob("model_hlbs_*.pth"))
        if not cands:
            raise FileNotFoundError(f"no model_hlbs_*.pth in {ckpt_dir}")
        pth = cands[-1]
    sd, face_joint_idx, helper_joint_idx = _peek_ckpt_buffers(pth)

    rig_path = getattr(opts, "rig_path", "maya_rig/hybrid")
    if not os.path.isabs(rig_path):
        rig_path = str(_REPO / rig_path)
    rig = load_rig(rig_path)

    base_joint_idx = None
    aj_json = getattr(opts, "active_joints_json", None)
    if aj_json:
        if not os.path.isabs(aj_json):
            aj_json = str(_REPO / aj_json)
        with open(aj_json) as f:
            aj = json.load(f)
        base_joint_idx = aj["base_joint_idx"]
        if face_joint_idx is None:
            face_joint_idx = aj["face_joint_idx"]

    sigma_targets = None
    sigma_npy = getattr(opts, "sigma_targets_npy", None)
    if sigma_npy:
        if not os.path.isabs(sigma_npy):
            sigma_npy = str(_REPO / sigma_npy)
        if os.path.exists(sigma_npy):
            sigma_targets = np.load(sigma_npy).astype(np.float32)

    model = HierarchicalLBS_FullPred(
        rig=rig,
        topology=getattr(opts, "topo_key", "ict"),
        in_dim_exp=12,
        hid_dim=getattr(opts, "hid_dim", 128),
        num_layers=getattr(opts, "num_layers", 4),
        device=str(device),
        use_joint_trans=getattr(opts, "use_joint_trans", True),
        smooth_W=getattr(opts, "smooth_delta_W", 0),
        smooth_W_alpha=getattr(opts, "smooth_delta_W_alpha", 0.5),
        dfn_skin=getattr(opts, "dfn_skin", False),
        dfn_bind=getattr(opts, "dfn_bind", False),
        dfn_exp=getattr(opts, "dfn_exp", False),
        nfs_feat_dim=int(getattr(opts, "nfs_feat_dim", 256)) if getattr(opts, "nfs_feat_dir", None) else 0,
        nfs_proj_dim=int(getattr(opts, "nfs_proj_dim", 0)),
        use_corrective=int(getattr(opts, "use_corrective", 0) or 0),
        nfs_concat=getattr(opts, "nfs_concat", False),
        adain_pos_norm=getattr(opts, "adain_pos_norm", False),
        freeze_bind_pose=getattr(opts, "freeze_bind_pose", False),
        use_gmm_hybrid=getattr(opts, "use_gmm_hybrid", False),
        init_log_sigma=getattr(opts, "init_log_sigma", -1.2),
        gmm_mode=getattr(opts, "gmm_mode", "additive"),
        residual_scale=getattr(opts, "residual_scale", 2.0),
        sigma_targets=sigma_targets,
        bind_pose_mode=getattr(opts, "bind_pose_mode", "net"),
        face_joint_idx=face_joint_idx,
        base_joint_idx=base_joint_idx,
        face_mask_r0=getattr(opts, "face_mask_r0", 1.0),
        face_mask_r1=getattr(opts, "face_mask_r1", 2.25),
        helper_joint_idx=helper_joint_idx,
        bind_pose_base_residual=bool(getattr(opts, "bind_pose_base_residual", 0)),
    ).to(device)

    msg = model.load_state_dict(sd, strict=False)
    if msg.missing_keys or msg.unexpected_keys:
        print(f"[load] missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}")
    model.eval()
    print(f"[load] {pth}")

    nfs_feat_dir = getattr(opts, "nfs_feat_dir", None)
    if nfs_feat_dir and not os.path.isabs(nfs_feat_dir):
        nfs_feat_dir = str(_REPO / nfs_feat_dir)

    geo_dist_per_topo = {}
    if getattr(opts, "use_geodesic_gauss", False):
        gd_dir = getattr(opts, "geo_dist_dir", None) or rig_path
        for topo in ("ict", "mf", "biwi", "coma"):
            p = os.path.join(gd_dir, f"geo_dist_{topo}.npy")
            if os.path.exists(p):
                geo_dist_per_topo[topo] = torch.from_numpy(
                    np.load(p)
                ).to(device).float()

    return model, rig, opts, nfs_feat_dir, helper_joint_idx, geo_dist_per_topo


# ───────────────────────── data helpers ─────────────────────────────────────


_ANIM_SEQS = {
    "922": _REPO / "_cap" / "20240318_MySlate_922_exp_coeffs.npy",
    "924": _REPO / "_cap" / "20240325_MySlate_924_exp_coeffs.npy",
}


def _load_exp_coeffs(seq: str) -> np.ndarray:
    p = _ANIM_SEQS[seq]
    if not p.exists():
        raise FileNotFoundError(f"ict-cap exp_coeffs not found: {p}")
    return np.load(p).astype(np.float32)


# ───────────────────────── topology data ────────────────────────────────────

import pickle
from dataclasses import dataclass, field


@dataclass
class TopoData:
    """All info needed to render one (dataset, topology) tuple.

    Identities are addressed by string `id_name` (the same key the model's
    cache files use, e.g. 'ict_007', 'm--20180226...', 'F1', 'FaceTalk_...').
    """
    name: str                              # 'ict_train', 'mf', 'biwi', 'coma'
    topo: str                              # 'ict' | 'mf' | 'biwi' | 'coma'
    id_names: list[str]
    faces: np.ndarray                      # [F, 3] uint32
    nfs_dir: str | None                    # cache base path or None
    geo_dist: torch.Tensor | None          # [J, V] or None
    supports_anim: bool = False            # True for ICT (blendshape-driven)
    # 'ict_blendshape' uses ict-cap exp coeffs; 'pca_mode' uses per-id PCA basis.
    exp_driver: str = "none"
    # Per-id neutral vertex provider — receives id_name, returns [V,3] float32.
    _verts_provider: callable = None
    # Optional: per-id 100-d identity vec (ICT only; used for blendshape exp).
    _ict_id_vecs: np.ndarray | None = None
    _ict_model: "ICT_face_model" = None
    # Optional: PCA exp driver per id (used by MF). {id_name: (mean[V,3], comps[K,V,3], std[K])}
    _pca_per_id: dict | None = None

    def neutral_verts(self, id_name: str) -> np.ndarray:
        return self._verts_provider(id_name)

    def ict_id_coeff(self, id_name: str) -> np.ndarray | None:
        """Return [100] id_coeff for this id, or None if unavailable.
        _ict_id_vecs may be an array (ict_train/val, indexed by trailing int) or
        a dict {id_name: coeff} (ict_real — only the ids with a known coeff)."""
        if self._ict_id_vecs is None:
            return None
        if isinstance(self._ict_id_vecs, dict):
            return self._ict_id_vecs.get(id_name)
        if id_name.startswith("ict_"):
            i = int(id_name.split("_")[-1])
            return self._ict_id_vecs[i]
        return None

    # Real per-id frame catalog: {id_name: {clip_name: [path1.npy, ...]}}
    _real_clips: dict | None = None
    _align_real_clips: bool = True  # off for pre-canonical clips (e.g. UniLS): per-frame procrustes injects jitter

    def apply_exp(self, id_name: str, exp_coeff) -> np.ndarray | None:
        """Apply expression driver to this id, return [V, 3] float32.

        For ICT blendshape:   exp_coeff is [53] blendshape coeffs.
        For MF PCA:           exp_coeff is [2] = (mode_idx, amp in [-1,1]).
        For MF real:          exp_coeff is (clip_name:str, frame_idx:int).
        """
        if self.exp_driver == "ict_blendshape":
            if self._ict_model is None: return None
            id_c = self.ict_id_coeff(id_name)
            if id_c is None: return None
            exp_T = exp_coeff[None, :] if exp_coeff.ndim == 1 else exp_coeff
            return self._ict_model.apply_coeffs(id_c, exp_T)[0].astype(np.float32)
        elif self.exp_driver == "ict_real_blend":
            # Real ICT capture: keep the captured neutral (_mesh.obj, matches the
            # validated precompute) and add the ICT-blendshape expression delta.
            # Only the ids with a known id_coeff can be a source; others are
            # target-only (returns None -> caller treats as no source frame).
            if self._ict_model is None:
                return None
            id_c = self.ict_id_coeff(id_name)
            if id_c is None:
                return None
            neu = self._verts_provider(id_name).astype(np.float32)
            exp_T = exp_coeff[None, :] if exp_coeff.ndim == 1 else exp_coeff
            full = self._ict_model.apply_coeffs(id_c, exp_T)[0].astype(np.float32)
            full0 = self._ict_model.apply_coeffs(id_c, None)[0].astype(np.float32)
            return (neu + (full - full0)).astype(np.float32)
        elif self.exp_driver == "pca_mode":
            if self._pca_per_id is None or id_name not in self._pca_per_id:
                return None
            mean, comps, std = self._pca_per_id[id_name]
            mode = int(exp_coeff[0]); amp = float(exp_coeff[1])
            mode = max(0, min(mode, comps.shape[0] - 1))
            disp = comps[mode] * (amp * 3.0 * std[mode])
            return (mean + disp).astype(np.float32)
        elif self.exp_driver == "mf_real":
            if self._real_clips is None or id_name not in self._real_clips:
                return None
            clip_name, fidx = exp_coeff
            if clip_name not in self._real_clips[id_name]:
                return None
            frames = self._real_clips[id_name][clip_name]
            if not frames: return None
            fidx = int(fidx) % len(frames)
            v = np.load(frames[fidx]).astype(np.float32)
            if not self._align_real_clips:
                return v
            # Procrustes-align to neutral template (matches dataloader pipeline).
            from utils.remesh_utils import procrustes_LDM
            template = self._verts_provider(id_name).astype(np.float32)
            R, t, _ = procrustes_LDM(v, template)
            return (v @ R.T + t).astype(np.float32)
        return None

    def clips_for(self, id_name: str) -> list[str]:
        """For mf_real driver: list available clip names for this id."""
        if self._real_clips is None or id_name not in self._real_clips:
            return []
        return sorted(self._real_clips[id_name].keys())

    def n_frames(self, id_name: str, clip: str) -> int:
        if self._real_clips and id_name in self._real_clips and clip in self._real_clips[id_name]:
            return len(self._real_clips[id_name][clip])
        return 0


def _build_topos(ict: "ICT_face_model", nfs_dir: str | None,
                 geo_per_topo: dict) -> dict[str, TopoData]:
    """Discover and build TopoData for every dataset present on disk."""
    topos: dict[str, TopoData] = {}

    # ── ICT (train + val variants share the same mesh / faces) ─────────
    ict_faces = ict.faces.astype(np.uint32)
    ict_train_vecs_path = _REPO / "ict_face_pt" / "random_identity_vecs.npy"
    if ict_train_vecs_path.exists():
        vecs = np.load(ict_train_vecs_path).astype(np.float32)[:111]
        ids = [f"ict_{i:03d}" for i in range(len(vecs))]
        topos["ict_train"] = TopoData(
            name="ict_train", topo="ict", id_names=ids, faces=ict_faces,
            nfs_dir=nfs_dir, geo_dist=geo_per_topo.get("ict"),
            supports_anim=True, exp_driver="ict_blendshape",
            _verts_provider=lambda nm, v=vecs, ic=ict:
                ic.apply_coeffs(v[int(nm.split("_")[-1])], exp_coeffs=None)[0]
                .astype(np.float32),
            _ict_id_vecs=vecs, _ict_model=ict,
        )
    ict_val_vecs_path = _REPO / "data" / "ICT_live_100" / "iden_vecs.npy"
    if ict_val_vecs_path.exists():
        vecs = np.load(ict_val_vecs_path).astype(np.float32)
        ids = [f"ict_val_{i:03d}" for i in range(len(vecs))]
        topos["ict_val"] = TopoData(
            name="ict_val", topo="ict", id_names=ids, faces=ict_faces,
            nfs_dir=None,  # no cache for val ids
            geo_dist=geo_per_topo.get("ict"),
            supports_anim=True, exp_driver="ict_blendshape",
            _verts_provider=lambda nm, v=vecs, ic=ict:
                ic.apply_coeffs(v[int(nm.split("_")[-1])], exp_coeffs=None)[0]
                .astype(np.float32),
            _ict_id_vecs=vecs, _ict_model=ict,
        )

    # ── MF / BIWI / COMA from bundled pkl templates ────────────────────
    tpl_root = _REPO / "utils" / "templates"
    for ds_name, pkl_name, topo_key in [
        ("mf",   "mf_templates.pkl",   "mf"),
        ("biwi", "biwi_templates.pkl", "biwi"),
        ("coma", "voca_templates.pkl", "coma"),
    ]:
        p = tpl_root / pkl_name
        if not p.exists():
            continue
        with open(p, "rb") as f:
            tpl = pickle.load(f)
        if "face" not in tpl:
            continue
        faces = np.asarray(tpl["face"], dtype=np.uint32)
        id_names = [k for k in tpl.keys() if k != "face"]
        verts_map = {k: np.asarray(tpl[k], dtype=np.float32) for k in id_names}

        # Prefer real per-id frame catalog; fall back to PCA basis (MF only).
        real_clips = _discover_real_clips(ds_name, id_names)
        pca_per_id = _try_load_pca(ds_name, id_names) if real_clips is None else None
        if real_clips is not None:
            exp_driver = "mf_real"  # same loader code path for all 3 topos
            supports_anim = True
        elif pca_per_id is not None:
            exp_driver = "pca_mode"; supports_anim = True
        else:
            exp_driver = "none"; supports_anim = False

        td = TopoData(
            name=ds_name, topo=topo_key, id_names=id_names, faces=faces,
            nfs_dir=nfs_dir, geo_dist=geo_per_topo.get(topo_key),
            supports_anim=supports_anim, exp_driver=exp_driver,
            _verts_provider=lambda nm, vm=verts_map: vm[nm],
            _pca_per_id=pca_per_id,
        )
        td._real_clips = real_clips
        topos[ds_name] = td

    # ── ICT real captures (m00..w09) ───────────────────────────────────
    # Neutral from the validated precompute dir's *_mesh.obj (so the on-disk
    # NFR precompute auto-matches -> no jitter). Same topology as synth ICT, so
    # faces = ict.faces. Target-capable for ALL ids; source-capable only for the
    # ids whose 100-d id_coeff is known (split_set/id_vecs, == ict_data_split
    # 'train' order) -> ict_real_blend driver. Built only if found on this host.
    import glob as _glob
    _real_dir = None
    for _r in ("/data/sihun", "/data/inyup", "/data2/inyup"):
        _d = os.path.join(_r, "ICT-audio2face/ICT/precompute-real-fullhead")
        if os.path.isdir(_d) and _glob.glob(os.path.join(_d, "*_mesh.obj")):
            _real_dir = _d
            break
    if _real_dir is not None:
        import trimesh as _tm
        real_ids = sorted(os.path.basename(p)[:-len("_mesh.obj")]
                          for p in _glob.glob(os.path.join(_real_dir, "*_mesh.obj")))
        # id_coeff for the 10 real-split ids (id_vecs row order == ict_data_split['train'])
        id_coeff_map: dict = {}
        try:
            from utils.keys import ict_data_split
            _ict_base = os.path.dirname(os.path.dirname(_real_dir))  # .../ICT-audio2face
            _ivp = os.path.join(_ict_base, "split_set", "id_vecs.npy")
            if os.path.exists(_ivp):
                _iv = np.load(_ivp).astype(np.float32)
                for _i, _nm in enumerate(ict_data_split["train"]):
                    if _i < len(_iv):
                        id_coeff_map[_nm] = _iv[_i]
        except Exception as _e:
            print(f"[ict_real] id_coeff load skipped ({_e}) — target-only")
        _neu_cache: dict = {}

        def _real_neu(nm, _d=_real_dir, _c=_neu_cache, _tmm=_tm):
            if nm not in _c:
                m = _tmm.load(os.path.join(_d, f"{nm}_mesh.obj"),
                              process=False, maintain_order=True)
                _c[nm] = np.asarray(m.vertices, dtype=np.float32)
            return _c[nm]

        topos["ict_real"] = TopoData(
            name="ict_real", topo="ict", id_names=real_ids,
            faces=ict.faces.astype(np.uint32), nfs_dir=nfs_dir,
            geo_dist=geo_per_topo.get("ict"), supports_anim=True,
            exp_driver="ict_real_blend",
            _verts_provider=_real_neu, _ict_id_vecs=id_coeff_map, _ict_model=ict,
        )
        print(f"[ict_real] {len(real_ids)} ids (source-capable: {len(id_coeff_map)}) "
              f"<- {_real_dir}")

    # ── Stylized test meshes (piers, malcolm, mary, morphy, girl, ...) ──
    # Each is its OWN topology (unique V / faces) -> one TopoData per mesh.
    # Target-only: retargeted FROM an anim source (mf/ict). Our model runs
    # given Diff3F nfs_feat ({slug}_nfs_feat.npy in <test-mesh>/diff3f); NFR/NFS
    # baselines run via on-the-fly precompute (topo key absent from
    # _PRECOMPUTE_SUBDIRS -> auto on-the-fly). Built only if found on this host.
    import glob as _gsty
    _sty_dir = None
    for _r in ("/data/sihun", "/data/inyup", "/data2/inyup", "/source/inyup"):
        _d = os.path.join(_r, "NFR_data/test-mesh")
        if os.path.isdir(_d) and _gsty.glob(os.path.join(_d, "*-align.obj")):
            _sty_dir = _d
            break
    if _sty_dir is not None:
        import trimesh as _tmsty
        _sty_feat_dir = os.path.join(_sty_dir, "diff3f")
        # curated stylized character set (skip FLAME/biwi/coma generic refs)
        _STY_WANT = ["piers", "malcolm", "mary", "morphy", "morphy-bald",
                     "girl", "bonnie-bald"]
        _n_sty = 0
        for _slug in _STY_WANT:
            _objp = os.path.join(_sty_dir, f"{_slug}-align.obj")
            if not os.path.exists(_objp):
                continue
            try:
                _m = _tmsty.load(_objp, process=False, maintain_order=True)
            except Exception as _e:
                print(f"[stylized] {_slug} load failed: {_e}")
                continue
            _vv = np.asarray(_m.vertices, dtype=np.float32)
            _ff = np.asarray(_m.faces, dtype=np.uint32)
            _name = "sty_" + _slug.replace("-", "_")
            _has_feat = os.path.exists(os.path.join(_sty_feat_dir,
                                                    f"{_slug}_nfs_feat.npy"))
            topos[_name] = TopoData(
                name=_name, topo=_name, id_names=[_slug], faces=_ff,
                nfs_dir=_sty_feat_dir, geo_dist=None,
                supports_anim=False, exp_driver="none",
                _verts_provider=(lambda nm, _v=_vv: _v),
            )
            _n_sty += 1
            print(f"[stylized] {_name}: V={len(_vv)} F={len(_ff)} "
                  f"diff3f={'yes' if _has_feat else 'MISSING'}")
        print(f"[stylized] {_n_sty} meshes <- {_sty_dir} (feat dir {_sty_feat_dir})")

    # ── extra zero-shot target topologies (GNM head / FLAME mean etc.) ──
    # Any .obj dropped into new_topo_meshes/ becomes a target topo "new_<stem>";
    # Diff3F feat expected at diff3f_feat_raw/<stem>_nfs_feat.npy.
    _ntm_dir = "/source/inyup/NeuralFacialAnimation/new_topo_meshes"
    if os.path.isdir(_ntm_dir):
        import trimesh as _tmnt, glob as _gnt
        _nt_feat_dir = "/source/inyup/NeuralFacialAnimation/diff3f_feat_raw"
        _n_new = 0
        for _objp in sorted(_gnt.glob(os.path.join(_ntm_dir, "*.obj"))):
            _slug = os.path.splitext(os.path.basename(_objp))[0]
            try:
                _m = _tmnt.load(_objp, process=False, maintain_order=True)
            except Exception as _e:
                print(f"[newtopo] {_slug} load failed: {_e}")
                continue
            _vv = np.asarray(_m.vertices, dtype=np.float32)
            _ff = np.asarray(_m.faces, dtype=np.uint32)
            _name = "new_" + _slug
            _has_feat = os.path.exists(os.path.join(_nt_feat_dir, f"{_slug}_nfs_feat.npy"))
            # optional clip source: new_topo_meshes/<slug>_clips/<clip>/NNNN.npy
            # -> makes this topo SOURCE-capable via the mf_real frame driver
            # (e.g. UniLS in-the-wild FLAME sequences).
            _clips_dir = os.path.join(_ntm_dir, f"{_slug}_clips")
            _real_clips = None
            if os.path.isdir(_clips_dir):
                _cat = {}
                for _cd in sorted(_gnt.glob(os.path.join(_clips_dir, "*"))):
                    if os.path.isdir(_cd):
                        _fr = sorted(_gnt.glob(os.path.join(_cd, "*.npy")))
                        if _fr:
                            _cat[os.path.basename(_cd)] = _fr
                if _cat:
                    _real_clips = {_slug: _cat}
            topos[_name] = TopoData(
                name=_name, topo=_name, id_names=[_slug], faces=_ff,
                nfs_dir=_nt_feat_dir, geo_dist=None,
                supports_anim=(_real_clips is not None),
                exp_driver=("mf_real" if _real_clips is not None else "none"),
                _verts_provider=(lambda nm, _v=_vv: _v),
                _real_clips=_real_clips,
                _align_real_clips=False,
            )
            _n_new += 1
            _nclip = len(_real_clips[_slug]) if _real_clips else 0
            print(f"[newtopo] {_name}: V={len(_vv)} F={len(_ff)} "
                  f"diff3f={'yes' if _has_feat else 'MISSING'}"
                  + (f" clips={_nclip} (source-capable)" if _nclip else ""))
        if _n_new:
            print(f"[newtopo] {_n_new} meshes <- {_ntm_dir}")

    return topos



_REAL_DATA_BASEDIRS: list[str] = []   # filled from CLI in main()


def _real_data_basedirs() -> list[Path]:
    """Candidate base dirs to scan for real animation .npy frames. Built from:
      1. --data_basedirs CLI args (highest priority)
      2. Environment variable HLBS_DATA_BASEDIRS (colon-sep)
      3. Hardcoded defaults that cover both servers we've seen + dataloader's
         --data_toggle pca variant ({base} and {base}/pca)
    Order matters — first hit wins per (id, clip) pair (we still aggregate
    across all hits to maximize coverage).
    """
    out = list(_REAL_DATA_BASEDIRS)
    env = os.environ.get("HLBS_DATA_BASEDIRS", "")
    if env:
        out.extend([s for s in env.split(":") if s])
    out.extend([
        # dataloader_CBD.py default + this-server discovery + likely siblings
        "/data/sihun", "/data2/sihun", "/data/inyup", "/data2/inyup",
        # dataloader's --data_toggle off variant (template_data_basedir = base+/pca)
        "/data/sihun/pca", "/data2/sihun/pca", "/data/inyup/pca", "/data2/inyup/pca",
        # observed on this server: data nested under a user dir
        "/data/inyup/data/sihun", "/data2/inyup/data/sihun",
    ])
    seen, dedup = set(), []
    for b in out:
        p = Path(b).resolve()
        if p in seen or not p.is_dir():
            continue
        seen.add(p); dedup.append(p)
    return dedup


def _discover_real_clips(ds_name: str, id_names: list[str]) -> dict | None:
    """Scan all candidate base dirs for per-id real animation clips.
    Returns {id_name: {clip_name: [sorted frame paths]}} or None if nothing found.

    Sub-paths per topology (relative to a basedir):
      mf:   multiface_align/{ROM,SEN}/{train,test}/vertices_npy/{id}/{clip}/*.npy
      biwi: BIWI_align_deci/{train,test}/vertices_npy/{id}_{clip}/*.npy
      coma: VOCA-COMA/COMA/{train,test}/{id}/vertices_npy/{clip}/*.npy
    """
    catalog: dict[str, dict[str, list]] = {nm: {} for nm in id_names}
    bases = _real_data_basedirs()

    def _add(catalog_id: str, key: str, frames: list[Path]):
        if frames and key not in catalog[catalog_id]:
            catalog[catalog_id][key] = frames

    for base in bases:
        if ds_name == "mf":
            for split in ("ROM/test", "ROM/train", "SEN/test", "SEN/train"):
                rp = base / "multiface_align" / split / "vertices_npy"
                if not rp.is_dir():
                    continue
                for id_dir in rp.iterdir():
                    nm = id_dir.name
                    if nm not in catalog or not id_dir.is_dir():
                        continue
                    for clip_dir in id_dir.iterdir():
                        if not clip_dir.is_dir():
                            continue
                        frames = sorted(clip_dir.glob("*.npy"))
                        frames = [f for f in frames if f.stat().st_size > 1024]
                        _add(nm, f"{split}/{clip_dir.name}", frames)
        elif ds_name == "biwi":
            for split in ("test", "train"):
                rp = base / "BIWI_align_deci" / split / "vertices_npy"
                if not rp.is_dir():
                    continue
                for sub in rp.iterdir():
                    if not sub.is_dir():
                        continue
                    nm, _, clip = sub.name.partition("_")
                    if nm not in catalog:
                        continue
                    frames = sorted(sub.glob("*.npy"))
                    frames = [f for f in frames if f.stat().st_size > 1024]
                    _add(nm, f"{split}/{clip}", frames)
        elif ds_name == "coma":
            for split in ("test", "train"):
                rp = base / "VOCA-COMA" / "COMA" / split
                if not rp.is_dir():
                    continue
                for id_dir in rp.iterdir():
                    if not id_dir.is_dir() or id_dir.name not in catalog:
                        continue
                    vnp = id_dir / "vertices_npy"
                    if not vnp.is_dir():
                        continue
                    for clip_dir in vnp.iterdir():
                        if not clip_dir.is_dir():
                            continue
                        frames = sorted(clip_dir.glob("*.npy"))
                        frames = [f for f in frames if f.stat().st_size > 1024]
                        _add(id_dir.name, f"{split}/{clip_dir.name}", frames)

    has_any = any(clips for clips in catalog.values())
    if has_any:
        total = sum(len(c) for c in catalog.values())
        with_data = sum(1 for c in catalog.values() if c)
        print(f"[real-clips] {ds_name}: {total} clips across {with_data}/{len(id_names)} ids "
              f"(scanned {len(bases)} basedirs)")
    return catalog if has_any else None


def _try_load_pca(ds_name: str, id_names: list[str]) -> dict | None:
    """Search disk for per-id PCA basis (mean + components + std).

    For MF: /data/sihun/pca/multiface_align/{SEN,ROM}/{train,test}/{id}_pca.npz
    Returns dict id_name → (mean[V,3], comps[K,V,3], std[K]) — or None if no
    id has a hit. PCA basis is shared across SEN/ROM splits in practice;
    we take whichever we find first per id.
    """
    if ds_name != "mf":
        return None
    # Scan every candidate basedir × {pca, plain} × {SEN, ROM} × {train, test}.
    bases = [b / "multiface_align" for b in _real_data_basedirs()]
    splits = ["SEN/train", "SEN/test", "ROM/train", "ROM/test"]
    out = {}
    for nm in id_names:
        found = None
        for base in bases:
            for sp in splits:
                p = base / sp / f"{nm}_pca.npz"
                if p.exists():
                    found = p; break
            if found: break
        if found is None:
            continue
        d = np.load(found)
        V_flat = d["mean_"]
        V = V_flat.shape[0] // 3
        mean = V_flat.reshape(V, 3).astype(np.float32)
        comps = d["components_"].reshape(-1, V, 3).astype(np.float32)
        std = np.sqrt(d["explained_variance_"].astype(np.float32))
        out[nm] = (mean, comps, std)
    return out if out else None


def _per_vertex_normal(v: np.ndarray, f: np.ndarray) -> np.ndarray:
    return igl.per_vertex_normals(
        v.astype(np.float64), f.astype(np.int64)
    ).astype(np.float32)


def _load_nfs_feat(nfs_dir: str, id_idx: int) -> np.ndarray | None:
    if nfs_dir is None:
        return None
    p = Path(nfs_dir) / f"ict_{id_idx:03d}_nfs_feat.npy"
    if not p.exists():
        return None
    return np.load(p).astype(np.float32)


def _load_gt_bind_pos(nfs_dir: str, id_idx: int) -> np.ndarray | None:
    if nfs_dir is None:
        return None
    p = Path(nfs_dir) / f"ict_{id_idx:03d}_bind_pos_landmark.npy"
    if not p.exists():
        return None
    return np.load(p).astype(np.float32)


def _fancy_skel_mesh(jp, parent, helper_set=None, bone_col=(200, 205, 214),
                     joint_col=(150, 155, 165), helper_col=(255, 80, 200),
                     ball_r=0.02, bone_r=0.012):
    """Maya-style skeleton as ONE merged mesh.
    - joint = icosphere (radius ball_r, gray)
    - bone  = 4-sided pyramid: base ring tangent to the PARENT ball
      (offset ball_r along the bone axis), apex AT the child joint.
    Sizes auto-shrink for very short bones so geometry never overlaps."""
    import trimesh as _tm
    jp = np.asarray(jp, dtype=np.float32)
    Jn = jp.shape[0]
    Vs, Fs, Cs = [], [], []
    off = 0
    for j in range(Jn):
        p = int(parent[j])
        if p < 0:
            continue
        a, b = jp[p], jp[j]
        d = b - a
        L = float(np.linalg.norm(d))
        if L < 1e-9:
            continue
        z = d / L
        up = np.array([0.0, 1.0, 0.0]) if abs(z[1]) < 0.9 else np.array([1.0, 0.0, 0.0])
        x = np.cross(up, z); x /= max(np.linalg.norm(x), 1e-9)
        y = np.cross(z, x)
        base_off = min(ball_r, 0.35 * L)          # ring sits on the parent ball
        r = min(bone_r, 0.30 * L)                 # base half-width
        w = a + z * base_off
        ring = [w + x * r, w + y * r, w - x * r, w - y * r]
        verts = np.array([*ring, b], dtype=np.float32)
        faces = np.array([[0, 1, 2], [0, 2, 3],          # base cap (faces parent)
                          [4, 1, 0], [4, 2, 1], [4, 3, 2], [4, 0, 3]])  # sides -> apex
        Vs.append(verts); Fs.append(faces + off); off += 5
        Cs.append(np.tile(bone_col, (5, 1)))
    ico = _tm.creation.icosphere(subdivisions=2, radius=1.0)
    iv = np.asarray(ico.vertices, dtype=np.float32)
    ifc = np.asarray(ico.faces)
    for j in range(Jn):
        Vs.append(iv * ball_r + jp[j])
        Fs.append(ifc + off); off += iv.shape[0]
        col = helper_col if (helper_set and j in helper_set) else joint_col
        Cs.append(np.tile(col, (iv.shape[0], 1)))
    V = np.concatenate(Vs, 0)
    F = np.concatenate(Fs, 0).astype(np.int64)
    C = np.concatenate(Cs, 0).astype(np.uint8)
    return V, F, C


_FEAT_CACHE: dict = {}

def _feat_raw(td, id_idx: int):
    """Raw per-vertex Diff3F feature [V, 2048] for topo/id (cached)."""
    id_name = td.id_names[id_idx]
    key = (td.name, id_name)
    if key in _FEAT_CACHE:
        return _FEAT_CACHE[key]
    f = None
    if td.nfs_dir is not None:
        p = Path(td.nfs_dir) / f"{id_name}_nfs_feat.npy"
        if p.exists():
            f = np.load(p).astype(np.float32)
    _FEAT_CACHE[key] = f
    return f


def _feat_pca_rgb(feats: list) -> list:
    """Joint 3-PC PCA -> RGB in a SHARED color space across the given meshes
    (same semantic region -> same color across topologies)."""
    rs = np.random.RandomState(0)
    X = np.concatenate(feats, 0)
    sub = X[rs.choice(X.shape[0], min(8000, X.shape[0]), replace=False)]
    mu = sub.mean(0)
    _, _, Vt = np.linalg.svd(sub - mu, full_matrices=False)
    P = Vt[:3].T
    Ys = [(F - mu) @ P for F in feats]
    Yall = np.concatenate(Ys, 0)
    lo = np.percentile(Yall, 2, axis=0); hi = np.percentile(Yall, 98, axis=0)
    return [(np.clip((Y - lo) / np.maximum(hi - lo, 1e-8), 0, 1) * 255).astype(np.uint8)
            for Y in Ys]


# ───────────────────────── color helpers ────────────────────────────────────


def _viridis_rgb(vals: np.ndarray) -> np.ndarray:
    """[N] in [0,1] → [N,3] uint8."""
    vals = np.clip(vals, 0.0, 1.0)
    return (cm.viridis(vals)[:, :3] * 255).astype(np.uint8)


def _err_rgb(vals: np.ndarray, name: str = "YlOrRd") -> np.ndarray:
    """[N] in [0,1] → [N,3] uint8 using matplotlib colormap (default YlOrRd —
    matches utils/matplotlib_rnd.plot_image_array_diff)."""
    vals = np.clip(vals, 0.0, 1.0)
    return (cm.get_cmap(name)(vals)[:, :3] * 255).astype(np.uint8)


def _tab20_rgb(idx: np.ndarray, J: int) -> np.ndarray:
    """[N] int in [0,J) → [N,3] uint8 (categorical)."""
    cmap = cm.get_cmap("tab20", max(J, 20))
    return (cmap(idx % cmap.N)[:, :3] * 255).astype(np.uint8)


# ───────────────────────── inference cache ──────────────────────────────────


class IdentityCache:
    """Memoize per-id model outputs (template / W / joint_pos), topology-aware.

    Cache keyed by (topo_name, id_name) — switching datasets / topologies is safe.
    """

    def __init__(self, model, rig, device, model_needs_nfs: bool):
        self.model = model
        self.rig = rig
        self.device = device
        self.model_needs_nfs = model_needs_nfs
        self._cache: dict[tuple, dict] = {}
        self.active: TopoData | None = None

    def set_active(self, topo_data: TopoData):
        self.active = topo_data

    @property
    def faces(self):
        return self.active.faces if self.active else None

    @torch.no_grad()
    def get(self, id_idx: int, topo: TopoData | None = None) -> dict:
        """Get cached model output for an id (by index into topo.id_names)."""
        td = topo if topo is not None else self.active
        id_name = td.id_names[id_idx]
        key = (td.name, id_name)
        if key in self._cache:
            return self._cache[key]
        neu_v = td.neutral_verts(id_name).astype(np.float32)
        if neu_v.ndim == 3:
            neu_v = neu_v[0]
        neu_n = _per_vertex_normal(neu_v, td.faces.astype(np.int64))

        # nfs cache + gt bind: file naming is `{id_name}_nfs_feat.npy`.
        nfs = None; gt_bind = None
        if td.nfs_dir is not None:
            p_nfs = Path(td.nfs_dir) / f"{id_name}_nfs_feat.npy"
            if p_nfs.exists():
                nfs = np.load(p_nfs).astype(np.float32)
            p_bp = Path(td.nfs_dir) / f"{id_name}_bind_pos_landmark.npy"
            if p_bp.exists():
                gt_bind = np.load(p_bp).astype(np.float32)

        # If model needs nfs cond but we have none, skip forward (mesh-only output).
        can_run = (not self.model_needs_nfs) or (nfs is not None)

        out = {
            "neu_v": neu_v, "neu_n": neu_n,
            "W": None, "joint_pos_pred": None, "gt_bind": gt_bind,
            "nfs": None, "dist_sq_geo": None, "model_ran": False,
        }

        if not can_run:
            self._cache[key] = out
            return out

        v_t = torch.from_numpy(neu_v).unsqueeze(0).to(self.device)
        n_t = torch.from_numpy(neu_n).unsqueeze(0).to(self.device)
        nfs_t = (
            torch.from_numpy(nfs).unsqueeze(0).to(self.device) if nfs is not None else None
        )

        dist_sq_geo = None
        if td.geo_dist is not None:
            gd = td.geo_dist.t()  # [V, J]
            if gd.shape[0] != v_t.shape[1]:
                gd = gd[: v_t.shape[1]]
            dist_sq_geo = (gd.unsqueeze(0)) ** 2

        delta = torch.zeros_like(v_t)
        src_in = torch.cat([v_t, n_t], dim=-1)
        deform_in = torch.cat([delta, n_t, src_in], dim=-1)

        _, extras = self.model(
            v_t, deform_in,
            source_normal=n_t, nfs_feat=nfs_t, dist_sq_geo=dist_sq_geo,
            return_extras=True,
        )
        out.update({
            "W": extras["W"][0].cpu().numpy(),
            "joint_pos_pred": extras["joint_pos"][0].cpu().numpy(),
            "nfs": nfs_t, "dist_sq_geo": dist_sq_geo, "model_ran": True,
        })
        self._cache[key] = out
        return out

    @torch.no_grad()
    def forward_frame(self, id_idx: int, exp_coeff: np.ndarray,
                      topo: TopoData | None = None) -> dict:
        """Run model on (neutral, deformed = blendshape-applied) for one frame.

        Only works for ICT topo (where blendshape exp is well-defined)."""
        td = topo if topo is not None else self.active
        base = self.get(id_idx, td)
        id_name = td.id_names[id_idx]

        gt_v = td.apply_exp(id_name, exp_coeff)
        if gt_v is None:
            return {"gt_v": None, "pred_v": None, "joint_pos": None,
                    "T_world": None, "local_R": None, "W": None,
                    "model_ran": False}

        out = {"gt_v": gt_v, "pred_v": None, "joint_pos": None,
               "T_world": None, "local_R": None, "W": None,
               "model_ran": base["model_ran"]}
        if not base["model_ran"]:
            return out

        gt_n = _per_vertex_normal(gt_v, td.faces.astype(np.int64))
        neu_v = base["neu_v"]; neu_n = base["neu_n"]
        v_t = torch.from_numpy(neu_v).unsqueeze(0).to(self.device)
        n_t = torch.from_numpy(neu_n).unsqueeze(0).to(self.device)
        gv_t = torch.from_numpy(gt_v).unsqueeze(0).to(self.device)
        gn_t = torch.from_numpy(gt_n).unsqueeze(0).to(self.device)

        delta = gv_t - v_t
        src_in = torch.cat([v_t, n_t], dim=-1)
        deform_in = torch.cat([delta, gn_t, src_in], dim=-1)

        pred_v, extras = self.model(
            v_t, deform_in,
            source_normal=n_t, nfs_feat=base["nfs"],
            dist_sq_geo=base["dist_sq_geo"],
            return_extras=True,
        )
        out.update({
            "pred_v": pred_v[0].cpu().numpy(),
            "joint_pos": extras["joint_pos"][0].cpu().numpy(),
            "T_world": extras["T_world"][0].cpu().numpy(),
            "local_R": extras["local_R"][0].cpu().numpy(),
            "W": extras["W"][0].cpu().numpy(),
        })
        return out

    @torch.no_grad()
    def retarget(self, src_topo: TopoData, src_id_idx: int, exp_coeff: np.ndarray,
                 tgt_topo: TopoData, tgt_id_idx: int) -> dict:
        """Apply src expression to tgt identity. Returns retargeted mesh + metrics.

        Source must support blendshape exp (ICT only currently). Target can be
        any topology with neutral mesh + (cached or runnable) nfs feature.
        """
        src_base = self.get(src_id_idx, src_topo)
        tgt_base = self.get(tgt_id_idx, tgt_topo)

        out = {"src_neu_v": src_base["neu_v"], "src_def_v": None,
               "tgt_neu_v": tgt_base["neu_v"], "tgt_pred_v": None,
               "src_joint_pos": None, "tgt_joint_pos": None, "tgt_T_world": None,
               "metrics": None, "model_ran": False}

        src_id_name = src_topo.id_names[src_id_idx]
        src_def_v = src_topo.apply_exp(src_id_name, exp_coeff)
        if src_def_v is None:
            return out
        out["src_def_v"] = src_def_v

        if not (src_base["model_ran"] and tgt_base["model_ran"]):
            return out

        src_neu_v = src_base["neu_v"]
        src_neu_n = src_base["neu_n"]
        src_def_n = _per_vertex_normal(src_def_v, src_topo.faces.astype(np.int64))
        tgt_neu_v = tgt_base["neu_v"]
        tgt_neu_n = tgt_base["neu_n"]

        def _b(a):
            return torch.from_numpy(a).unsqueeze(0).to(self.device)

        rigid_v, _tgt_jp, _tgt_Tw = self.model.retarget(
            src_neu_vert=_b(src_neu_v), src_neu_norm=_b(src_neu_n),
            src_def_vert=_b(src_def_v), src_def_norm=_b(src_def_n),
            tgt_neu_vert=_b(tgt_neu_v), tgt_neu_norm=_b(tgt_neu_n),
            tgt_nfs_feat=tgt_base["nfs"],
            tgt_dist_sq_geo=tgt_base["dist_sq_geo"],
            return_joints=True,
        )
        out["tgt_pred_v"] = rigid_v[0].cpu().numpy()
        out["tgt_joint_pos"] = _tgt_jp[0].cpu().numpy()
        out["tgt_T_world"] = _tgt_Tw[0].cpu().numpy()
        out["src_joint_pos"] = src_base.get("joint_pos_pred")
        out["model_ran"] = True

        # Metrics: if src == tgt (same id, same topo), self-retarget — compare to
        # blendshape GT applied to target. Otherwise, no GT available.
        if (src_topo.name == tgt_topo.name) and (src_id_idx == tgt_id_idx):
            gt_v = src_def_v  # GT for src==tgt is just the source deformed mesh
            out["metrics"] = _compute_metrics(out["tgt_pred_v"], gt_v, tgt_topo.faces)
        return out


_CUPY_CUSPARSE_SHIM_DONE = False


def _install_cupy_cusparse_shim():
    """Legacy operators.pkl pickled the Poisson LU solver (cupyx SuperLU) and a
    MatDescriptor under the OLD cupy module path `cupy.cusparse`. Modern cupy
    moved these to cupyx.* so unpickling raises ModuleNotFoundError /
    AttributeError and the NFR/NFS baselines silently fall back to on-the-fly
    operators (jitter + per-frame recompute). Register a lazy shim module that
    resolves any legacy `cupy.cusparse.<name>` from where it now lives. Idempotent;
    no-op if cupy isn't importable (e.g. CPU-only viser node)."""
    global _CUPY_CUSPARSE_SHIM_DONE
    if _CUPY_CUSPARSE_SHIM_DONE:
        return
    import sys as _sys, types as _types, importlib as _il
    _cands = []
    for _n in ('cupyx.cusparse', 'cupyx.scipy.sparse.linalg',
               'cupyx.scipy.sparse', 'cupy'):
        try:
            _cands.append(_il.import_module(_n))
        except Exception:
            pass
    if not _cands:
        _CUPY_CUSPARSE_SHIM_DONE = True  # no cupy here; nothing to shim
        return

    class _Lazy(_types.ModuleType):
        def __getattr__(self, k):
            for _m in _cands:
                if hasattr(_m, k):
                    return getattr(_m, k)
            raise AttributeError(k)

    _sys.modules.setdefault('cupy.cusparse', _Lazy('cupy.cusparse'))
    _CUPY_CUSPARSE_SHIM_DONE = True


_BASELINE_ENV_CHECKED = False
_BASELINE_ENV_MSG = ""


def _ensure_baseline_env():
    """One-time env guard for NFR/NFS baselines: need GPU + cupy + GPU-compiled
    pytorch3d. Workspace reopens reset /opt/conda (pytorch3d -> editable CPU
    build, cupy/tensorboard gone), so this self-heals like training's
    _ensure_pytorch3d_gpu. Returns (ok, msg). cupy auto-install can take effect
    mid-session; a pytorch3d rebuild needs a viser RESTART (C-ext already loaded)."""
    global _BASELINE_ENV_CHECKED, _BASELINE_ENV_MSG
    if _BASELINE_ENV_CHECKED:
        return (_BASELINE_ENV_MSG == "OK"), _BASELINE_ENV_MSG
    import sys, os, subprocess
    import torch
    if not torch.cuda.is_available():
        _BASELINE_ENV_MSG = "no GPU on this node -> NFR/NFS disabled (cupy operators are CUDA). Use a GPU viser node."
        _BASELINE_ENV_CHECKED = True
        print(f"[baseline env] {_BASELINE_ENV_MSG}")
        return False, _BASELINE_ENV_MSG
    try:
        import cupy, cupyx  # noqa: F401
    except Exception:
        print("[baseline env] cupy missing -> installing cupy-cuda12x (one-time)...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "cupy-cuda12x"], check=False)
        import importlib
        for _m in [k for k in list(sys.modules)
                   if k == "cupy" or k.startswith("cupy.") or k == "cupyx" or k.startswith("cupyx.")]:
            del sys.modules[_m]
        importlib.invalidate_caches()
        try:
            import cupy, cupyx  # noqa: F401
        except Exception as e:
            _BASELINE_ENV_MSG = ("cupy is installed but could not load in this running viser "
                                 f"-> RESTART viser to enable NFR/NFS (import error: {e}).")
            _BASELINE_ENV_CHECKED = True
            print(f"[baseline env] {_BASELINE_ENV_MSG}")
            return False, _BASELINE_ENV_MSG

    def _p3d_gpu_ok():
        try:
            import torch as _t
            from pytorch3d.ops import knn_points
            a = _t.rand(1, 16, 3, device="cuda")
            knn_points(a, a, K=3)
            return True
        except Exception:
            return False

    if not _p3d_gpu_ok():
        print("[baseline env] pytorch3d not GPU-compiled -> rebuilding (one-time ~5-10min)...")
        env = os.environ.copy()
        env["FORCE_CUDA"] = "1"
        try:
            cc = torch.cuda.get_device_capability(0)
            env["TORCH_CUDA_ARCH_LIST"] = f"{cc[0]}.{cc[1]}"
        except Exception:
            pass
        for ch in ("/usr/local/cuda-12.4", "/usr/local/cuda"):
            if os.path.isdir(ch):
                env["CUDA_HOME"] = ch
                env["PATH"] = ch + "/bin:" + env.get("PATH", "")
                env["LD_LIBRARY_PATH"] = ch + "/lib64:" + env.get("LD_LIBRARY_PATH", "")
                break
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "pytorch3d"], check=False, env=env)
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-cache-dir", "fvcore", "iopath"], check=False, env=env)
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-build-isolation",
                        "--no-cache-dir", "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.9"],
                       check=False, env=env)
        _BASELINE_ENV_MSG = ("pytorch3d rebuilt to GPU -> RESTART viser to use it "
                             "(this process still has the CPU build loaded).")
        _BASELINE_ENV_CHECKED = True
        print(f"[baseline env] {_BASELINE_ENV_MSG}")
        return False, _BASELINE_ENV_MSG

    _BASELINE_ENV_MSG = "OK"
    _BASELINE_ENV_CHECKED = True
    print("[baseline env] GPU + cupy + pytorch3d(GPU) ready.")
    return True, _BASELINE_ENV_MSG


class BaselineRunner:
    """Lazy NFR / NFS loaders + per-(src,tgt) precompute cache for cross-
    retargeting baselines.

    Both NFR and NFS expose `inference(gt_vertices[T,V,3], src_mesh, tgt_mesh)`
    returning `[T, V, 3]` predicted target vertices. We wrap them so the cross
    mode can call either side-by-side with HLBS.
    """

    def __init__(self, device, repo_root: Path):
        self.device = device
        self.repo = repo_root
        self._nfr = None       # NFR_helper
        self._nfs = None       # NFS module
        # precomp[(method, src_key, tgt_key)] = (src_mesh, tgt_mesh, src_precomp,
        #                                         tgt_precomp) for NFR; just
        # (src_mesh, tgt_mesh) for NFS (it builds operators internally).
        self._precomp: dict = {}

    def _ensure_legacy_symlinks(self):
        """NFR_helper resolves data/utils/ckpt paths relative to legacy/.
        Symlink the repo-root dirs into legacy/ so paths resolve (idempotent)."""
        legacy = self.repo / "legacy"
        for nm in ("data", "utils", "ckpts_comparison", "experiments"):
            dst = legacy / nm
            if dst.exists() or dst.is_symlink():
                continue
            src = self.repo / nm
            if src.exists():
                try:
                    dst.symlink_to(src.absolute())
                    print(f"[baseline] symlinked {dst} → {src}")
                except Exception as e:
                    print(f"[baseline] symlink {dst} failed: {e}")

    def _ensure_cupy(self):
        """NFR's deformation_transfer hard-requires cupy (no CPU fallback).
        Auto-install cupy-cuda12x via pip if not present."""
        try:
            import cupy  # noqa: F401
            return
        except ImportError:
            import subprocess
            print("[baseline] cupy missing — pip install cupy-cuda12x (one-time)")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "cupy-cuda12x"])
            import cupy  # noqa: F401

    def _ensure_tensorboard(self):
        """legacy/evaluation.py imports torch.utils.tensorboard.SummaryWriter at
        module load, which needs the `tensorboard` package. Auto-install if
        missing (one-time) so the NFR/NFS baselines can be loaded."""
        try:
            import tensorboard  # noqa: F401
            return
        except ImportError:
            import subprocess
            print("[baseline] tensorboard missing — pip install (one-time)")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorboard"])
            import tensorboard  # noqa: F401

    def _load_nfr(self):
        if self._nfr is not None:
            return self._nfr
        # cupy is a hard dep of deformation_transfer.py — install if missing.
        self._ensure_cupy()
        # legacy.evaluation imports tensorboard at module level.
        self._ensure_tensorboard()
        # NFR_helper uses Path(__file__).parents[0] as its base, which resolves
        # to legacy/ — but the actual data lives at repo root. Bridge with
        # symlinks (idempotent, one-time setup on first NFR use).
        self._ensure_legacy_symlinks()
        sys.path.insert(0, str(self.repo / "legacy"))
        try:
            from legacy.evaluation import NFR_helper
            class _Opts: pass
            opts = _Opts(); opts.ict_face_only = False
            self._nfr = NFR_helper(opts=opts, device=str(self.device))
            print("[baseline] NFR loaded")
        finally:
            try: sys.path.remove(str(self.repo / "legacy"))
            except ValueError: pass
        return self._nfr

    def _load_nfs(self):
        if self._nfs is not None:
            return self._nfs
        self._ensure_tensorboard()
        import yaml as _yaml
        from models.NFS import NFS
        ckpt_dir = self.repo / "ckpts_comparison" / "NFS-best"
        with open(ckpt_dir / "train_opts.yml") as f:
            opts_dict = _yaml.safe_load(f)

        class _Opts: pass
        opts = _Opts()
        for k, v in opts_dict.items():
            setattr(opts, k, v)
        opts.device = str(self.device); opts.is_train = False
        self._nfs = NFS(opts=opts).to(self.device)
        sd = torch.load(ckpt_dir / "model_best.pth",
                        map_location=self.device, weights_only=False)
        self._nfs.load_state_dict(sd, strict=False)
        self._nfs.eval()
        print(f"[baseline] NFS loaded: {ckpt_dir}/model_best.pth")
        return self._nfs

    @torch.no_grad()
    def _nfs_forward_precompute(self, nfs, gt_vertices, src_faces, src_dfn,
                                src_img, tgt_template, tgt_faces, tgt_dfn,
                                tgt_img, tgt_ops):
        """NFS forward using validated disk precompute — mirrors legacy
        eval_comp._forward_nfs_precompute. Sets precomputes explicitly
        (update_precomputes) so the DiffusionNet encoders don't hit the
        'no precomputes' path, and uses the validated dfn/operators (no jitter)."""
        dev = self.device
        tgt_img_feat = nfs.get_img_feat(tgt_img)
        tgt_verts_t = torch.tensor(tgt_template).float().unsqueeze(0).to(dev)
        tgt_faces_t = torch.tensor(tgt_faces).long().to(dev)
        tgt_vert_feat = nfs.get_local_feature(tgt_verts_t, tgt_faces_t, tgt_img_feat).float()
        nfs.mesh_id_encoder.update_precomputes(tgt_dfn)
        pred_id = nfs.encode_id(tgt_vert_feat, tgt_dfn)
        pred_seg = nfs.encode_seg(tgt_vert_feat, tgt_dfn)
        src_img_feat = nfs.get_img_feat(src_img)
        src_faces_t = torch.tensor(src_faces).long().to(dev)
        gt_v = gt_vertices.to(dev).float()
        vfe = [nfs.get_local_feature(gt_v[t:t+1], src_faces_t, src_img_feat).float()
               for t in range(gt_v.shape[0])]
        vert_feat_exp = torch.cat(vfe, dim=0)
        pred_exp = nfs.encode_exp(vert_feat_exp, src_dfn, batch_process=True)
        inputs = (tgt_vert_feat, pred_exp, pred_id, pred_seg, None,
                  tgt_verts_t, tgt_faces_t, tgt_ops)
        pred_outputs, _ = nfs.decode(inputs, batch_process=True)
        return pred_outputs

    def _meshes(self, src_neu_v, src_faces, tgt_neu_v, tgt_faces):
        import trimesh
        src = trimesh.Trimesh(vertices=src_neu_v, faces=src_faces,
                              process=False, maintain_order=True)
        tgt = trimesh.Trimesh(vertices=tgt_neu_v, faces=tgt_faces,
                              process=False, maintain_order=True)
        return src, tgt

    # ── Validated NFR precompute on disk ────────────────────────────────
    # Matches legacy/eval_comp.py: load the SAME dfn_info / img / operators
    # the comparison eval used. Computing these on-the-fly under the current
    # numpy/igl stack yields slightly different DFN basis / Poisson operators
    # → frame-to-frame render jitter. Loading the validated set removes it.
    _PRECOMPUTE_SUBDIRS = {
        'ict':  'ICT-audio2face/ICT/precompute-real-fullhead',
        'mf':   'multiface_align/precomputes',
        'biwi': 'BIWI_align_deci/precomputes',
        'coma': 'VOCA-COMA/precomputes',
        'voca': 'VOCA-COMA/precomputes',
    }
    _PRECOMPUTE_ROOTS = ('/data/sihun', '/data/inyup', '/data2/inyup')

    def _load_disk_precompute(self, topo, id_name, want_ops):
        """Cached wrapper: load the validated precompute ONCE per
        (topo, id, want_ops) and memoize (None included), so per-frame baseline
        inference doesn't re-read ~30MB from disk every frame."""
        ck = ("disk_pc", topo, id_name, want_ops)
        if ck not in self._precomp:
            self._precomp[ck] = self._load_disk_precompute_uncached(
                topo, id_name, want_ops)
        return self._precomp[ck]

    def _load_disk_precompute_uncached(self, topo, id_name, want_ops):
        """Load (dfn_info, img[, operators]) from disk, searching multiple data
        roots (sihun → inyup → data2). Returns the tuple or None (caller then
        falls back to on-the-fly, which can introduce jitter). id_name must
        match the precompute file prefix — true for mf/biwi/coma; ICT random
        identities have no validated precompute so this returns None there."""
        import os, pickle
        sub = self._PRECOMPUTE_SUBDIRS.get(topo)
        if sub is None or not id_name:
            return None
        for root in self._PRECOMPUTE_ROOTS:
            prefix = os.path.join(root, sub, id_name)
            dfn_p, img_p = prefix + '_dfn_info.pkl', prefix + '_img.npy'
            if not (os.path.exists(dfn_p) and os.path.exists(img_p)):
                continue
            with open(dfn_p, 'rb') as f:
                dfn = pickle.load(f)
            dfn = [x.to(self.device).float() if isinstance(x, torch.Tensor) else x
                   for x in dfn]
            img = torch.tensor(np.load(img_p)).float().to(self.device)
            if not want_ops:
                print(f"[NFR precompute] disk(src) {topo}/{id_name} <- {root}")
                return (dfn, img)
            ops_p = prefix + '_operators.pkl'
            if not os.path.exists(ops_p):
                return None
            _install_cupy_cusparse_shim()
            try:
                with open(ops_p, 'rb') as f:
                    lu_solver, idxs, vals, rhs = pickle.load(f)
                if isinstance(idxs, torch.Tensor): idxs = idxs.to(self.device)
                if isinstance(vals, torch.Tensor): vals = vals.to(self.device)
            except (ModuleNotFoundError, ImportError) as e:
                print(f"[NFR precompute] operators load failed ({e}) — on-the-fly")
                return None
            print(f"[NFR precompute] disk(tgt) {topo}/{id_name} <- {root}")
            return (dfn, img, (lu_solver, idxs, vals, rhs))
        return None

    def _nfr_src_precomp(self, key, src_mesh, topo=None, id_name=None):
        """Cache (dfn_info, img) per src mesh. Prefers validated disk precompute
        (multi-root); else on-the-fly (~30-60s on ICT; may cause jitter)."""
        ck = ("nfr_src", *key)
        if ck in self._precomp:
            return self._precomp[ck]
        disk = self._load_disk_precompute_uncached(topo, id_name, want_ops=False)
        if disk is not None:
            self._precomp[ck] = disk
            return disk
        nfr = self._load_nfr()
        import utils.nfr_utils as nfr_utils
        dfn = nfr_utils.get_dfn_info(src_mesh, map_location=self.device)
        img = nfr.renderer.render_img(src_mesh).float().to(self.device)
        print(f"[NFR precompute] ON-THE-FLY(src) {topo}/{id_name} — may jitter")
        self._precomp[ck] = (dfn, img)
        return self._precomp[ck]

    def _nfr_tgt_precomp(self, key, tgt_mesh, topo=None, id_name=None):
        """Cache (dfn_info, img, operators) per tgt mesh. Prefers validated disk
        precompute (multi-root); else on-the-fly (Poisson LU 1-3 min; may jitter).
        operators are the expensive bit and the most jitter-sensitive."""
        ck = ("nfr_tgt", *key)
        if ck in self._precomp:
            return self._precomp[ck]
        disk = self._load_disk_precompute_uncached(topo, id_name, want_ops=True)
        if disk is not None:
            self._precomp[ck] = disk
            return disk
        nfr = self._load_nfr()
        import utils.nfr_utils as nfr_utils
        dfn = nfr_utils.get_dfn_info(tgt_mesh, map_location=self.device)
        img = nfr.renderer.render_img(tgt_mesh).float().to(self.device)
        ops = nfr.get_mesh_operators(tgt_mesh)
        print(f"[NFR precompute] ON-THE-FLY(tgt) {topo}/{id_name} — may jitter")
        self._precomp[ck] = (dfn, img, ops)
        return self._precomp[ck]

    @torch.no_grad()
    def _nfs_src_precomp(self, key, src_mesh):
        """On-the-fly NFS src precompute (dfn_info, img) cached per src id, so
        dfn is computed ONCE and reused across all frames (compute-once =
        official design; eliminates per-frame recompute + jitter)."""
        ck = ("nfs_src", *key)
        if ck in self._precomp:
            return self._precomp[ck]
        import utils.nfr_utils as nfr_utils
        dfn = nfr_utils.get_dfn_info(src_mesh, map_location=self.device)
        img = self._nfs.renderer.render_img(src_mesh).float().to(self.device)
        self._precomp[ck] = (dfn, img)
        return self._precomp[ck]

    def _nfs_tgt_precomp(self, key, tgt_mesh):
        """On-the-fly NFS tgt precompute (dfn_info, img, operators) cached per
        tgt id. operators=get_mesh_operators -> cupy SuperLU (needs cupy+GPU)."""
        ck = ("nfs_tgt", *key)
        if ck in self._precomp:
            return self._precomp[ck]
        import utils.nfr_utils as nfr_utils
        try:
            from utils.mesh_utils import get_mesh_operators
        except Exception as e:
            raise RuntimeError(f"NFS baseline needs cupy+GPU (get_mesh_operators import: {e})")
        dfn = nfr_utils.get_dfn_info(tgt_mesh, map_location=self.device)
        img = self._nfs.renderer.render_img(tgt_mesh).float().to(self.device)
        ops = get_mesh_operators(tgt_mesh)
        self._precomp[ck] = (dfn, img, ops)
        return self._precomp[ck]

    def retarget(self, method: str, src_td, src_idx, src_neu_v, src_def_v,
                 tgt_td, tgt_idx, tgt_neu_v,
                 progress_cb=None) -> np.ndarray | None:
        """Run the chosen baseline. Returns tgt_pred_v [V_tgt, 3] or None.

        Precomputes (mesh operators, DFN info, rendered img) are cached per
        (src_topo, src_id) and (tgt_topo, tgt_id) so first call is slow
        (~1-3 min for ICT-sized meshes) and subsequent calls are fast.
        progress_cb(msg) called with elapsed-time ticker every ~1.5s so the
        UI shows live progress instead of a stuck status.
        """
        import threading as _th, time as _time

        def _msg(s):
            if progress_cb is not None:
                try: progress_cb(s)
                except Exception: pass

        def _with_ticker(prefix, fn):
            """Run fn() while a background thread updates progress_cb every
            1.5s with elapsed time, so the status bar doesn't appear frozen."""
            _msg(f"{prefix}  [0.0s]")
            stop = _th.Event()
            def _tick():
                t0 = _time.time()
                while not stop.wait(1.5):
                    _msg(f"{prefix}  [{_time.time()-t0:.1f}s elapsed]")
            th = _th.Thread(target=_tick, daemon=True)
            th.start()
            try:
                t0 = _time.time()
                out = fn()
                _msg(f"{prefix}  ✓ done in {_time.time()-t0:.1f}s")
                return out
            finally:
                stop.set()
                th.join(timeout=0.2)

        if method in ("nfr", "nfs"):
            _ok, _envmsg = _ensure_baseline_env()
            if not _ok:
                _envmsg2 = f"[{method}] baseline unavailable: {_envmsg}"
                print(_envmsg2)
                if progress_cb is not None:
                    try: progress_cb(_envmsg2)
                    except Exception: pass
                return None

        if method == "nfr":
            if self._nfr is None:
                _with_ticker("NFR: loading model + assets", self._load_nfr)
            nfr = self._nfr
            src_mesh, tgt_mesh = self._meshes(
                src_neu_v, src_td.faces, tgt_neu_v, tgt_td.faces)
            src_key = (src_td.name, src_idx)
            tgt_key = (tgt_td.name, tgt_idx)
            src_idn = src_td.id_names[src_idx]
            tgt_idn = tgt_td.id_names[tgt_idx]
            if ("nfr_src", *src_key) not in self._precomp:
                _with_ticker(
                    f"NFR: building src precomp [{src_td.name}#{src_idx}] V={src_neu_v.shape[0]}",
                    lambda: self._nfr_src_precomp(src_key, src_mesh, src_td.topo, src_idn),
                )
            src_pc = self._nfr_src_precomp(src_key, src_mesh, src_td.topo, src_idn)
            if ("nfr_tgt", *tgt_key) not in self._precomp:
                _with_ticker(
                    f"NFR: building tgt precomp [{tgt_td.name}#{tgt_idx}] V={tgt_neu_v.shape[0]} (Poisson LU)",
                    lambda: self._nfr_tgt_precomp(tgt_key, tgt_mesh, tgt_td.topo, tgt_idn),
                )
            tgt_pc = self._nfr_tgt_precomp(tgt_key, tgt_mesh, tgt_td.topo, tgt_idn)
            verts = torch.from_numpy(src_def_v).unsqueeze(0).to(self.device).float()
            pred = _with_ticker(
                f"NFR: inferring f={src_idx}→{tgt_idx}",
                lambda: nfr.inference(verts, src_mesh, tgt_mesh,
                                      src_precompute=src_pc, tgt_precompute=tgt_pc),
            )
            return pred[0].cpu().numpy().astype(np.float32)
        elif method == "nfs":
            if self._nfs is None:
                _with_ticker("NFS: loading model", self._load_nfs)
            nfs = self._nfs
            src_idn = src_td.id_names[src_idx]
            tgt_idn = tgt_td.id_names[tgt_idx]
            src_pc = self._load_disk_precompute(src_td.topo, src_idn, want_ops=False)
            tgt_pc = self._load_disk_precompute(tgt_td.topo, tgt_idn, want_ops=True)
            verts = torch.from_numpy(src_def_v).unsqueeze(0).to(self.device).float()
            if src_pc is not None and tgt_pc is not None:
                src_dfn, src_img = src_pc
                tgt_dfn, tgt_img, tgt_ops = tgt_pc
                pred = _with_ticker(
                    f"NFS: inferring (disk precompute) {src_td.name}#{src_idx}->{tgt_td.name}#{tgt_idx}",
                    lambda: self._nfs_forward_precompute(
                        nfs, verts, src_td.faces, src_dfn, src_img,
                        tgt_neu_v, tgt_td.faces, tgt_dfn, tgt_img, tgt_ops),
                )
            else:
                # No disk precompute: compute dfn/img/operators ON-THE-FLY but
                # ONCE per id (cached in self._precomp) and reuse across all
                # frames via the same precompute path. compute-once = official
                # design -> no per-frame recompute, no jitter. (operators need
                # cupy+GPU; on a CPU node this raises -> NFS needs a GPU node.)
                src_mesh, tgt_mesh = self._meshes(
                    src_neu_v, src_td.faces, tgt_neu_v, tgt_td.faces)
                src_key = (src_td.name, src_idx)
                tgt_key = (tgt_td.name, tgt_idx)
                if ("nfs_tgt", *tgt_key) not in self._precomp:
                    print(f"[NFS precompute] ON-THE-FLY(once) src={src_td.topo}/{src_idn} "
                          f"tgt={tgt_td.topo}/{tgt_idn} — computing dfn/img/operators (cached)")
                s_dfn, s_img       = self._nfs_src_precomp(src_key, src_mesh)
                t_dfn, t_img, t_ops = self._nfs_tgt_precomp(tgt_key, tgt_mesh)
                pred = _with_ticker(
                    f"NFS: inferring (on-the-fly precompute, cached) {src_td.name}#{src_idx}->{tgt_td.name}#{tgt_idx}",
                    lambda: self._nfs_forward_precompute(
                        nfs, verts, src_td.faces, s_dfn, s_img,
                        tgt_neu_v, tgt_td.faces, t_dfn, t_img, t_ops),
                )
            if pred.dim() == 4:
                pred = pred[0]
            return pred[0].cpu().numpy().astype(np.float32)
        return None


def _compute_metrics(pred_v: np.ndarray, gt_v: np.ndarray, faces: np.ndarray) -> dict:
    """Per-frame eval metrics: MSE, mean L2, max L2, normal cosine, Laplacian err."""
    diff = pred_v - gt_v
    sq = (diff ** 2).sum(-1)
    mse = float(sq.mean())
    l2 = np.sqrt(sq + 1e-12)
    mean_l2 = float(l2.mean()); max_l2 = float(l2.max()); med_l2 = float(np.median(l2))
    # Normal consistency
    f_int = faces.astype(np.int64)
    n_pred = igl.per_vertex_normals(pred_v.astype(np.float64), f_int)
    n_gt = igl.per_vertex_normals(gt_v.astype(np.float64), f_int)
    n_pred /= (np.linalg.norm(n_pred, axis=-1, keepdims=True) + 1e-8)
    n_gt /= (np.linalg.norm(n_gt, axis=-1, keepdims=True) + 1e-8)
    norm_cos = float((n_pred * n_gt).sum(-1).mean())
    # Laplacian smoothness (uniform): mean of ||L pred - L gt||
    L = igl.cotmatrix(gt_v.astype(np.float64), f_int)
    Lp = L @ pred_v.astype(np.float64)
    Lg = L @ gt_v.astype(np.float64)
    lap_err = float(np.linalg.norm(Lp - Lg, axis=-1).mean())
    return {"mse": mse, "mean_l2_mm": mean_l2 * 1000,
            "med_l2_mm": med_l2 * 1000, "max_l2_mm": max_l2 * 1000,
            "norm_cos": norm_cos, "lap_err": lap_err}


# ───────────────────────── viser app ────────────────────────────────────────


def _boost_saturation(rgb_uint8, factor):
    """HSV S boost on a [N, 3] uint8 array. factor=1.0 → noop."""
    if factor is None or abs(factor - 1.0) < 1e-3:
        return rgb_uint8
    from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
    hsv = rgb_to_hsv(np.clip(rgb_uint8.astype(np.float32) / 255.0, 0.0, 1.0))
    hsv[..., 1] = np.clip(hsv[..., 1] * float(factor), 0.0, 1.0)
    return np.clip(hsv_to_rgb(hsv) * 255.0, 0, 255).astype(np.uint8)


_VIS_FLAGS = {"src": True, "tgt": True, "neu": True}
# mesh group routing by node name (src / neutral / predicted)
def _mesh_group(name):
    if name.startswith(("/anim/gt", "/cross/src", "/feat/src")): return "src"
    if name.startswith(("/anim/neu", "/cross/tgt_neu")): return "neu"
    if name == "/mesh" or name.startswith(("/anim/pred", "/feat/tgt")): return "pred"
    if name.startswith("/cross/tgt"): return "pred"
    return None
_MAT = {"roughness": 0.4, "trim_rings": 0, "err_absmax": 0.02,
        "clay_key_int": 1.3, "clay_dx": 0.0, "clay_dy": 0.7, "clay_dz": 0.9,
        "clay_ambient": 0.3}

_ERR_STATS = {}
_CLAY_PLIGHTS = {}
_GROUND = {"h": None, "y": None}

def _sync_ground(server):
    on = bool(_MAT.get("ground_shadow", False))
    y = float(_MAT.get("ground_y", -1.75))
    if not on:
        if _GROUND["h"] is not None:
            try: _GROUND["h"].visible = False
            except Exception: pass
        return
    if _GROUND["h"] is None or _GROUND["y"] != y:
        if _GROUND["h"] is not None:
            try: _GROUND["h"].remove()
            except Exception: pass
        S = 40.0
        v = np.array([[-S, y, -S], [S, y, -S], [S, y, S], [-S, y, S]], dtype=np.float32)
        f = np.array([[0, 2, 1], [0, 3, 2]], dtype=np.uint32)
        try:
            _GROUND["h"] = server.scene.add_mesh_simple(
                "/ground", vertices=v, faces=f, color=(245, 245, 247),
                side="double", cast_shadow=False, receive_shadow=True)
            _GROUND["y"] = y
        except Exception:
            return
    try: _GROUND["h"].visible = True
    except Exception: pass

def _hide_clay_lights():
    for _e in _CLAY_PLIGHTS.values():
        try: _e[0].visible = False
        except Exception: pass

def _sync_clay_light(server, name, center):
    """Clay mode: one SPOT key-light per mesh — positioned at the mesh center
    plus a shared offset, aimed at the mesh center (identical relative angle
    for every mesh). Cone/penumbra/distance bound the beam so neighbor meshes
    get little to no spill. Handles lack live params -> recreate on change."""
    if _MAT.get("light_mode") != "clay":
        return
    _dx = float(_MAT.get("clay_dx", 0.0)); _dy = float(_MAT.get("clay_dy", 0.7)); _dz = float(_MAT.get("clay_dz", 0.9))
    params = (
        float(_MAT.get("clay_key_int", 4.0)),
        float(_MAT.get("clay_angle", 50.0)),
        float(_MAT.get("clay_penumbra", 0.4)),
        float(_MAT.get("clay_distance", 3.5)),
        round(_dx, 4), round(_dy, 4), round(_dz, 4),
        bool(_MAT.get("ground_shadow", False)),
    )
    pos = (float(center[0]) + _dx, float(center[1]) + _dy, float(center[2]) + _dz)
    dvec = np.array([-_dx, -_dy, -_dz], dtype=np.float64)   # aim back at mesh center
    dvec = dvec / max(np.linalg.norm(dvec), 1e-9)
    # mesh-group visibility: hidden mesh -> its light off too (no spill)
    _grp = _mesh_group(name)
    _mesh_visible = True
    if _grp == "src":
        _mesh_visible = bool(_VIS_FLAGS.get("src", True))
    elif _grp == "neu":
        _mesh_visible = bool(_VIS_FLAGS.get("neu", True))
    elif _grp == "pred":
        _mesh_visible = bool(_VIS_FLAGS.get("tgt", True))
    ent = _CLAY_PLIGHTS.get(name)
    if ent is not None and ent[1] != params:
        try: ent[0].remove()
        except Exception: pass
        ent = None; _CLAY_PLIGHTS.pop(name, None)
    if ent is None:
        try:
            h = server.scene.add_light_spot(
                "/lights/clay" + name, color=(255, 250, 244),
                intensity=params[0],
                angle=float(np.radians(params[1])),
                penumbra=params[2],
                distance=params[3],
                direction=tuple(dvec),
                cast_shadow=bool(_MAT.get("ground_shadow", False)),
            )
            _CLAY_PLIGHTS[name] = (h, params)
            ent = (h, params)
        except Exception:
            return
    try:
        ent[0].position = pos
        ent[0].visible = _mesh_visible
    except Exception:
        pass
_BMASK_CACHE = {}

def _boundary_vert_mask(faces, n_rings, n_verts):
    """True for vertices within n_rings of an open boundary (excluded from
    error stats/coloring — scan-crop rim junk)."""
    key = (int(n_verts), int(len(faces)), int(n_rings))
    if key in _BMASK_CACHE:
        return _BMASK_CACHE[key]
    f = np.asarray(faces)
    excl = np.zeros(n_verts, dtype=bool)
    for _ in range(int(n_rings)):
        e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], 0)
        e = np.sort(e, 1)
        _, idx, cnt = np.unique(e, axis=0, return_index=True, return_counts=True)
        bverts = np.unique(e[idx[cnt == 1]])
        if len(bverts) == 0:
            break
        excl[bverts[bverts < n_verts]] = True
        keep = ~excl[f].any(1)
        f = f[keep]
    _BMASK_CACHE[key] = excl
    return excl

_FMASK_CACHE = {}

def _face_vert_mask(verts):
    """Model-identical face mask: plateau_hat_points(r0=1.0, r1=2.25) >= 0.5.
    True = inside face region. Cached per mesh (shape + centroid)."""
    k = (verts.shape[0], tuple(np.round(verts.mean(0), 3)))
    if k in _FMASK_CACHE:
        return _FMASK_CACHE[k]
    try:
        import torch as _t
        from utils.exp_utils import plateau_hat_points
        v = _t.from_numpy(np.asarray(verts, dtype=np.float32))[None]
        tm = plateau_hat_points(v, r0=1.0, r1=2.25).squeeze(-1)[0].cpu().numpy()
        m = tm >= 0.5
    except Exception:
        m = np.ones(verts.shape[0], dtype=bool)
    _FMASK_CACHE[k] = m
    return m

def _err_norm2(err, key=None, faces=None, verts=None):
    """Sequence-adaptive, frame-independent error normalization.
    modes: 'seq p95' (default) / 'seq max' — running stats per (target, clip),
    stabilize after one pass; 'fixed' — absolute cap; 'per-frame' — legacy."""
    mode = str(_MAT.get("err_mode", "seq p95"))
    mask = None
    if faces is not None and int(_MAT.get("err_face_rings", 3)) > 0:
        mask = _boundary_vert_mask(faces, int(_MAT.get("err_face_rings", 3)), err.shape[0])
    if verts is not None and bool(_MAT.get("err_face_only", True)):
        fm = _face_vert_mask(verts)          # True = face region
        mask = (~fm) if mask is None else (mask | (~fm))
    core = err[~mask] if (mask is not None and (~mask).any()) else err
    if mode == "fixed":
        den = float(_MAT.get("err_absmax", 0.02))
    elif mode == "per-frame":
        den = float(np.max(core))
    else:
        st = _ERR_STATS.setdefault(key, {"max": 1e-8, "p95": 1e-8})
        st["max"] = max(st["max"], float(np.max(core)))
        st["p95"] = max(st["p95"], float(np.percentile(core, 95)))
        den = st["max"] if mode == "seq max" else st["p95"]
    n = np.clip(err / max(den, 1e-8), 0.0, 1.0)
    if mask is not None:
        n[mask] = 0.0
    return n

def _err_norm(err):
    """Frame-independent error normalization: fixed absolute cap when
    err_absmax>0 (same error = same color on every frame), else per-frame max."""
    cap = float(_MAT.get("err_absmax", 0.0))
    den = cap if cap > 0 else max(float(np.max(err)), 1e-8)
    return np.clip(err / den, 0.0, 1.0)

def _trim_boundary_faces(faces, n_rings, verts=None):
    """Drop faces touching BOTTOM open-boundary vertices, n_rings times.
    Only the neck-bottom rim is trimmed (boundary verts in the lowest 30%
    y-band) — eye/lip/nostril hole boundaries are left intact."""
    f = np.asarray(faces)
    ythr = None
    if verts is not None:
        v = np.asarray(verts)
        ythr = float(v[:, 1].min() + 0.30 * (v[:, 1].max() - v[:, 1].min()))
    for _ in range(int(n_rings)):
        e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], 0)
        e = np.sort(e, 1)
        _, idx, cnt = np.unique(e, axis=0, return_index=True, return_counts=True)
        bverts = np.unique(e[idx[cnt == 1]])
        if ythr is not None and len(bverts):
            bverts = bverts[np.asarray(verts)[bverts, 1] < ythr]
        if len(bverts) == 0:
            break
        bset = np.zeros(int(f.max()) + 1, dtype=bool)
        bset[bverts] = True
        keep = ~bset[f].any(1)
        if keep.all():
            break
        f = f[keep]
    return f

def _add_per_vertex_color_mesh(server, name, verts, faces, rgb_uint8,
                               opacity=1.0, shading="smooth", double_sided=False,
                               sat=1.0):
    """Per-vertex colored mesh with TRUE alpha blending via PBR alphaMode=BLEND.

    shading:
      'smooth' — embed area-weighted vertex normals (igl) in GLB → three.js
                 uses smooth normals (Gouraud-like).
      'flat'   — omit normals → three.js computes per-face normals → flat
                 shading exposes mesh edges.
    """
    import trimesh
    from trimesh.visual.material import PBRMaterial

    # Uniform tints (color pickers) bypass the saturation boost — WYSIWYG;
    # the boost only applies to per-vertex colors (weight/error heatmaps).
    _uniform = rgb_uint8.shape[0] > 1 and bool(np.all(rgb_uint8[:1] == rgb_uint8))
    if not _uniform and rgb_uint8.shape[1] >= 3:
        rgb_uint8 = _boost_saturation(rgb_uint8[:, :3], sat)
    _grp = _mesh_group(name)
    if _grp == "src":
        opacity = float(_MAT.get("op_src", opacity))
    elif _grp == "pred":
        opacity = float(_MAT.get("op_pred", opacity))
    # neutrals keep the passed-in (legacy) opacity untouched
    a_val = int(np.clip(opacity, 0.05, 1.0) * 255)
    if rgb_uint8.shape[1] == 3:
        a = np.full((rgb_uint8.shape[0], 1), a_val, dtype=np.uint8)
        rgba = np.concatenate([rgb_uint8, a], axis=-1)
    else:
        rgba = rgb_uint8.copy()
        rgba[:, 3] = a_val
    if _MAT.get("trim_rings", 0) and name.startswith(("/mesh", "/anim/", "/cross/", "/feat/")):
        faces = _trim_boundary_faces(faces, _MAT["trim_rings"], verts=verts)
    mesh = trimesh.Trimesh(
        vertices=verts.astype(np.float32),
        faces=faces.astype(np.uint32),
        vertex_colors=rgba,
        process=False,
        maintain_order=True,
    )
    if shading == "smooth":
        # Compute area-weighted vertex normals so GLB encodes them and three.js
        # interpolates per pixel — no visible facet edges.
        vn = igl.per_vertex_normals(
            verts.astype(np.float64), faces.astype(np.int64)
        ).astype(np.float32)
        vn = np.nan_to_num(vn, nan=0.0)   # orphaned verts (boundary trim) -> zero normal
        mesh.vertex_normals = vn
    # else: leave normals unset → trimesh GLB exports no normals → flat shading.

    # Force matte plastic PBR so vertex_colors aren't blown out by viser's HDRI
    # (default 'warehouse' env-map is bright and the glTF metallic default is
    # 1.0). roughness=1 + metallic=0 = pure Lambert-like; baseColor white so
    # vertex_colors aren't tinted; alphaMode=BLEND only when needed.
    mesh.visual.material = PBRMaterial(
        alphaMode="BLEND" if opacity < 1.0 else "OPAQUE",
        baseColorFactor=[1.0, 1.0, 1.0, 1.0],
        metallicFactor=0.0,
        roughnessFactor=float(_MAT.get("roughness", 1.0)),
        doubleSided=bool(double_sided),
    )
    _h = server.scene.add_mesh_trimesh(name, mesh)
    _sync_clay_light(server, name, np.asarray(verts).mean(0))
    try:
        if _grp == "src":
            _h.visible = bool(_VIS_FLAGS.get("src", True))
        elif _grp == "neu":
            _h.visible = bool(_VIS_FLAGS.get("neu", True))
        elif _grp == "pred":
            _h.visible = bool(_VIS_FLAGS.get("tgt", True))
    except Exception:
        pass
    return _h


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        type=str,
        default="ckpts_hlbs/2026-05-14-14-02-03-HLBS-FullPred-ict-jTrans-nrm0.1-Wsm0.01",
    )
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--data_basedirs", type=str, nargs="*", default=[],
        help="Extra base dirs to scan for animation data, in addition to "
             "auto-discovery of /data/{sihun,inyup}, /data2/..., and their "
             "/pca variants. Each should contain multiface_align/, "
             "BIWI_align_deci/, and/or VOCA-COMA/ subdirs. Also honors env "
             "var HLBS_DATA_BASEDIRS (colon-separated).",
    )
    args = parser.parse_args()
    # Make CLI basedirs visible to the discovery functions.
    global _REAL_DATA_BASEDIRS
    _REAL_DATA_BASEDIRS = list(args.data_basedirs or [])
    _bases = _real_data_basedirs()
    print(f"[data] real-clip basedirs ({len(_bases)} present): "
          + ", ".join(str(b) for b in _bases))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(args.ckpt)
    if not ckpt_dir.is_absolute():
        ckpt_dir = _REPO / ckpt_dir

    model, rig, opts, nfs_dir, helper_idx, geo_per_topo = _build_model(ckpt_dir, device)
    model_needs_nfs = getattr(opts, "nfs_feat_dir", None) is not None

    ict = ICT_face_model(base_dir=str(_REPO))
    J = len(rig.joint_names)
    parent_idx = rig.parent_idx.cpu().numpy()
    helper_set = set(helper_idx or [])
    # Helper text labels: "<parent>#<n>" (replaces cryptic joint name in 3D view).
    helper_label_map = {}
    try:
        import json as _hjson
        _hjp = getattr(opts, "helper_joints_json", None)
        if _hjp and not os.path.isabs(_hjp):
            _hjp = str(_REPO / _hjp)
        if _hjp and os.path.exists(_hjp):
            _hj = _hjson.load(open(_hjp))
            _hlist = _hj.get("helpers", [])
            _hidx = helper_idx or []
            _pc = {}
            for _i, _hh in enumerate(_hlist):
                if _i >= len(_hidx):
                    break
                _pn = _hh.get("parent", "?")
                _pc[_pn] = _pc.get(_pn, 0) + 1
                helper_label_map[_hidx[_i]] = f"{_pn}#{_pc[_pn]}"
    except Exception as _e:
        print(f"[helper labels] json skipped: {_e}")
    for _j in (helper_idx or []):
        if _j not in helper_label_map:
            _p = int(parent_idx[_j]) if 0 <= _j < len(parent_idx) else -1
            _pn = rig.joint_names[_p] if 0 <= _p < len(rig.joint_names) else "root"
            helper_label_map[_j] = f"{_pn}#{_j}"

    topos = _build_topos(ict, nfs_dir, geo_per_topo)
    if not topos:
        raise RuntimeError("no topologies found")
    avail_ds = list(topos.keys())
    avail_seqs = [k for k, p in _ANIM_SEQS.items() if p.exists()]
    if not avail_seqs:
        raise RuntimeError("no animation sequences found in _cap/")
    print(f"[setup] J={J} helpers={len(helper_set)} "
          f"datasets={[(k, len(t.id_names)) for k,t in topos.items()]} "
          f"anim_seqs={avail_seqs} model_needs_nfs={model_needs_nfs}")

    cache = IdentityCache(model, rig, device, model_needs_nfs=model_needs_nfs)

    # ── TTA rig injection ──────────────────────────────────────────────
    # Loads per-target optimized rigs (W logits->softmax'd W, bind pose, and
    # per-joint transform scales) produced by tta_selfrecon_v2/v3 into
    # <repo>/tta_proto*/{kind}_{id}/optimized.npz. A GUI dropdown swaps them
    # in at inference time; networks themselves are untouched.
    import glob as _tta_glob
    from pytorch3d.transforms import matrix_to_axis_angle as _tta_m2aa, \
        axis_angle_to_matrix as _tta_aa2m
    _tta_data = {}
    _tta = {"active": None, "scales_on": False}
    for _p in sorted(_tta_glob.glob(str(_REPO / "tta_proto*" / "*" / "optimized.npz"))):
        _nm = Path(_p).parent.name
        try:
            _kind, _idn = _nm.split("_", 1)
        except ValueError:
            continue
        _ver = os.path.basename(os.path.dirname(os.path.dirname(_p))).replace("tta_proto_", "").replace("tta_", "")
        _z = np.load(_p)
        _J = model.num_joints
        _tta_data[f"{_ver}:{_nm}"] = {
            "ds": _kind, "id": _idn, "V": int(_z["W"].shape[1]),
            "W": torch.tensor(np.asarray(_z["W"]), dtype=torch.float32, device=device),
            "bind": torch.tensor(np.asarray(_z["bind"]), dtype=torch.float32, device=device),
            "s_rot": (torch.tensor(np.asarray(_z["s_rot"]), dtype=torch.float32, device=device)
                      if "s_rot" in _z else torch.ones(_J, device=device)),
            "s_trn": (torch.tensor(np.asarray(_z["s_trn"]), dtype=torch.float32, device=device)
                      if "s_trn" in _z else torch.ones(_J, device=device)),
        }
    if _tta_data:
        print(f"[TTA] {len(_tta_data)} optimized rigs: {sorted(_tta_data.keys())}")
        _tta_orig_gsw = model._get_skinning_weights
        _tta_orig_gbp = model._get_bind_pose
        _tta_orig_rot6d = model._rot6d
        _tta_orig_pose = model.lbs_pose_model
        _tta_orig_ret = model.retarget

        def _tta_gsw(*a, **k):
            W, dW = _tta_orig_gsw(*a, **k)
            d = _tta_data.get(_tta["active"])
            if d is not None and W.shape[1] == d["V"]:
                W = d["W"]
            return W, dW

        def _tta_gbp(*a, **k):
            Bi, jp = _tta_orig_gbp(*a, **k)
            d = _tta_data.get(_tta["active"])
            if d is not None and a and getattr(a[0], "ndim", 0) >= 2 and a[0].shape[1] == d["V"]:
                jp = d["bind"]
                Bi = model._build_B_inv(jp)
            return Bi, jp

        def _tta_rot6d(x):
            Rm = _tta_orig_rot6d(x)
            d = _tta_data.get(_tta["active"])
            if d is None or not _tta["scales_on"]:
                return Rm
            _J = model.num_joints
            aa = _tta_m2aa(Rm).reshape(-1, _J, 3) * d["s_rot"][None, :, None]
            return _tta_aa2m(aa.reshape(-1, 3))

        class _TTAPose(torch.nn.Module):
            def __init__(self, orig):
                super().__init__(); self.orig = orig
            def forward(self, z, **kw):
                out = self.orig(z, **kw)
                d = _tta_data.get(_tta["active"])
                if d is None or not _tta["scales_on"]:
                    return out
                _J = model.num_joints; B_ = out.shape[0]
                rot = out[..., :_J * 6]
                trn = out[..., _J * 6:].reshape(B_, 1, _J, 3) * d["s_trn"][None, None, :, None]
                return torch.cat([rot, trn.reshape(B_, 1, _J * 3)], dim=-1)

        def _tta_ret(*a, **k):
            # transform scales are target-rig calibration: engage only inside
            # retarget so other topologies' self-recon renders stay untouched.
            _tta["scales_on"] = True
            try:
                return _tta_orig_ret(*a, **k)
            finally:
                _tta["scales_on"] = False

        model._get_skinning_weights = _tta_gsw
        model._get_bind_pose = _tta_gbp
        model._rot6d = _tta_rot6d
        model.lbs_pose_model = _TTAPose(_tta_orig_pose)
        model.retarget = _tta_ret
    baseline = BaselineRunner(device, _REPO)
    init_ds = "ict_train" if "ict_train" in topos else avail_ds[0]
    cache.set_active(topos[init_ds])
    init_seq = avail_seqs[0]
    state = {"exp_coeffs": _load_exp_coeffs(init_seq)}
    n_frames_init = state["exp_coeffs"].shape[0]

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+y")  # ICT face is y-up

    # ───── GUI ──────────────────────────────────────────────────────────
    g_mode = server.gui.add_dropdown(
        "Mode", options=["bind_pose", "weight", "anim", "cross", "feat"], initial_value="bind_pose"
    )
    def _list_epoch_tags(_ckpt_dir):
        """List of selectable epoch tags: 'best' + each numbered epoch desc."""
        out = []
        if (_ckpt_dir / "model_hlbs_best.pth").exists():
            out.append("best")
        epochs = []
        for p in _ckpt_dir.glob("model_hlbs_*.pth"):
            stem = p.stem.replace("model_hlbs_", "")
            if stem == "best":
                continue
            try:
                epochs.append(int(stem))
            except ValueError:
                pass
        for e in sorted(epochs, reverse=True):
            out.append(f"{e:03d}")
        return out or ["best"]

    _epoch_tags = _list_epoch_tags(ckpt_dir)
    g_epoch = server.gui.add_dropdown(
        "epoch", options=_epoch_tags, initial_value=_epoch_tags[0],
    )
    g_reload = server.gui.add_button("Reload ckpt (re-read selected epoch)")
    g_refresh_epochs = server.gui.add_button("↻ refresh epoch list")
    g_ckpt_info = server.gui.add_text(
        "ckpt", str(ckpt_dir.name), disabled=True,
    )
    g_dataset = server.gui.add_dropdown(
        "Dataset", options=avail_ds, initial_value=init_ds,
    )
    g_anim_seq = server.gui.add_dropdown(
        "Anim seq", options=avail_seqs, initial_value=init_seq,
    )
    g_id = server.gui.add_slider(
        "Identity", min=0, max=len(topos[init_ds].id_names) - 1, step=1, initial_value=0,
    )
    # global — applies to mesh in every mode (simulated via color-toward-white blend)
    g_mesh_opacity = server.gui.add_slider(
        "mesh opacity", min=0.05, max=1.0, step=0.05, initial_value=1.0,
    )
    g_shading = server.gui.add_dropdown(
        "shading", options=["smooth", "flat"], initial_value="smooth",
    )
    g_double_sided = server.gui.add_checkbox(
        "double-sided (kills dark fringe at open cuts; may distort normals)",
        False,
    )
    g_view = server.gui.add_dropdown(
        "view preset",
        options=["free", "front", "back", "left", "right", "top", "bottom"],
        initial_value="free",
    )
    g_ortho = server.gui.add_checkbox(
        "orthographic (small-FOV fake)", False,
    )
    # ── Exact camera control: live readout + numeric set (reproducible renders)
    with server.gui.add_folder("Camera (exact)"):
        g_cam_live = server.gui.add_text("live pos|look", "-", disabled=True)
        g_cam_px = server.gui.add_number("cam x", initial_value=-1.35, step=0.01)
        g_cam_py = server.gui.add_number("cam y", initial_value=-0.02, step=0.01)
        g_cam_pz = server.gui.add_number("cam z", initial_value=52.47, step=0.01)
        g_cam_lx = server.gui.add_number("look x", initial_value=-1.35, step=0.01)
        g_cam_ly = server.gui.add_number("look y", initial_value=-0.02, step=0.01)
        g_cam_lz = server.gui.add_number("look z", initial_value=0.0, step=0.01)
        g_cam_read = server.gui.add_button("read current camera")
        g_cam_apply = server.gui.add_button("apply to camera")

    def _cam_fill_from(cam):
        try:
            p = cam.position; l = cam.look_at
            g_cam_px.value = round(float(p[0]), 4); g_cam_py.value = round(float(p[1]), 4); g_cam_pz.value = round(float(p[2]), 4)
            g_cam_lx.value = round(float(l[0]), 4); g_cam_ly.value = round(float(l[1]), 4); g_cam_lz.value = round(float(l[2]), 4)
        except Exception:
            pass

    def _cam_read(_e=None):
        for c in server.get_clients().values():
            _cam_fill_from(c.camera); break

    def _cam_apply(_e=None):
        for c in server.get_clients().values():
            try:
                c.camera.position = (float(g_cam_px.value), float(g_cam_py.value), float(g_cam_pz.value))
                c.camera.look_at = (float(g_cam_lx.value), float(g_cam_ly.value), float(g_cam_lz.value))
            except Exception:
                pass
    g_cam_read.on_click(_cam_read)
    g_cam_apply.on_click(_cam_apply)

    @server.on_client_connect
    def _cam_on_connect(client):
        _first = {"done": False}

        @client.camera.on_update
        def _cam_live_update(_cam):
            try:
                p = client.camera.position; l = client.camera.look_at
                g_cam_live.value = (f"p({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f}) "
                                    f"l({l[0]:+.3f},{l[1]:+.3f},{l[2]:+.3f})")
                if not _first["done"]:
                    _first["done"] = True
                    _cam_fill_from(client.camera)
            except Exception:
                pass
    with server.gui.add_folder("Lighting"):
        g_lighting = server.gui.add_dropdown(
            "lighting mode",
            options=["hdri", "front only", "matplotlib (mild flat)", "6-axis studio", "clay (figure)", "clay v2 (reference)", "flat (no shadows)"],
            initial_value="clay v2 (reference)",
        )
        g_env_intensity = server.gui.add_slider(
            "env light intensity", min=0.0, max=2.0, step=0.05, initial_value=0.1,
        )
        g_env_map = server.gui.add_dropdown(
            "env HDRI",
            options=["none", "warehouse", "studio", "apartment", "city",
                     "dawn", "forest", "lobby", "night", "park", "sunset"],
            initial_value="studio",
        )
        g_clay_key = server.gui.add_slider(
            "clay key intensity", min=0.5, max=30.0, step=0.5, initial_value=1.3,
        )
        # key offset relative to each mesh center; closer = stronger key +
        # less spill onto neighbor meshes (inverse-square falloff)
        g_clay_dx = server.gui.add_number("clay key dx", initial_value=0.0, step=0.05)
        g_clay_dy = server.gui.add_number("clay key dy", initial_value=0.7, step=0.05)
        g_clay_dz = server.gui.add_number("clay key dz", initial_value=0.9, step=0.05)
        g_clay_angle = server.gui.add_slider(
            "clay key cone angle (deg)", min=10, max=90, step=2, initial_value=50,
        )
        g_clay_penumbra = server.gui.add_slider(
            "clay key penumbra (soft edge)", min=0.0, max=1.0, step=0.05, initial_value=0.4,
        )
        g_clay_distance = server.gui.add_slider(
            "clay key distance cutoff (0=inf)", min=0.0, max=10.0, step=0.25, initial_value=3.5,
        )
        g_clay_ambient = server.gui.add_slider(
            "clay ambient (global fill)", min=0.0, max=2.0, step=0.05, initial_value=0.3,
        )
        g_ground_shadow = server.gui.add_checkbox("ground + shadows", False)
        g_ground_y = server.gui.add_number("ground height (y)", initial_value=-1.75, step=0.05)

    with server.gui.add_folder("Mesh display"):
        g_mesh_color = server.gui.add_rgb("mesh color (default)", (89, 128, 212))
        g_src_color  = server.gui.add_rgb("cross src color",      (89, 128, 212))
        g_tgt_color  = server.gui.add_rgb("cross tgt color",      (89, 128, 212))
        g_mat_rough = server.gui.add_slider(
            "mesh roughness (1=matte)", min=0.2, max=1.0, step=0.05, initial_value=0.4,
        )
        g_trim_rings = server.gui.add_slider(
            "trim open boundary (rings)", min=0, max=6, step=1, initial_value=0,
        )
        g_global_sat = server.gui.add_slider(
            "global saturation", min=0.5, max=3.0, step=0.1, initial_value=1.5,
        )
        g_show_src = server.gui.add_checkbox("show source/GT mesh", True)
        g_show_tgt = server.gui.add_checkbox("show predicted (deformed) mesh", True)
        g_show_neu = server.gui.add_checkbox("show neutral mesh(es)", True)
        g_op_src = server.gui.add_slider(
            "source mesh opacity", min=0.05, max=1.0, step=0.05, initial_value=1.0,
        )
        g_op_pred = server.gui.add_slider(
            "predicted mesh opacity", min=0.05, max=1.0, step=0.05, initial_value=1.0,
        )

    with server.gui.add_folder("Skeleton"):
        g_skel_style = server.gui.add_dropdown(
            "skeleton style", options=["classic (lines)", "fancy (maya)"],
            initial_value="classic (lines)",
        )
        g_fancy_joint = server.gui.add_slider(
            "fancy joint size", min=0.002, max=0.08, step=0.002, initial_value=0.02,
        )
        g_fancy_bone = server.gui.add_slider(
            "fancy bone thickness", min=0.002, max=0.05, step=0.002, initial_value=0.012,
        )
        g_show_jlabels = server.gui.add_checkbox("show joint name labels", True)

    with server.gui.add_folder("Error vis"):
        g_err_scale_mode = server.gui.add_dropdown(
            "error scale", options=["seq p95", "seq max", "fixed", "per-frame"],
            initial_value="seq p95",
        )
        g_err_absmax = server.gui.add_number(
            "error fixed max (for 'fixed')", initial_value=0.02, step=0.005,
        )
        g_err_face_rings = server.gui.add_slider(
            "error: exclude boundary rings", min=0, max=8, step=1, initial_value=3,
        )
        g_err_face_only = server.gui.add_checkbox("error: face mask only", True)
        g_err_reset = server.gui.add_button("reset seq error scale")

    with server.gui.add_folder("Bind pose"):
        g_show_gt = server.gui.add_checkbox("show GT joints", True)
        g_show_pred = server.gui.add_checkbox("show Pred joints", True)
        g_show_err = server.gui.add_checkbox("show error arrows", True)
        g_show_helpers = server.gui.add_checkbox("highlight helpers", True)
        g_bind_bones = server.gui.add_checkbox("skeleton bones (anim-style)", False)

    with server.gui.add_folder("Weight"):
        g_w_mode = server.gui.add_dropdown(
            "weight mode",
            options=["single", "argmax", "soft", "entropy"],
            initial_value="soft",
        )
        g_w_show_joints = server.gui.add_checkbox("show skeleton (bind-pose style)", False)
        g_w_mark_joint = server.gui.add_checkbox("mark selected joint (single mode)", False)
        g_joint = server.gui.add_dropdown(
            "joint (single mode)",
            options=[f"{j:02d} {n}" for j, n in enumerate(rig.joint_names)],
            initial_value=f"00 {rig.joint_names[0]}",
        )
        g_soft_topk = server.gui.add_slider(
            "soft top-K joints", min=1, max=8, step=1, initial_value=3,
        )
        # Palette: nipy_spectral / turbo / gist_rainbow evenly sampled give the
        # best contrast across J=66 joints. tab20 cycles (only 20 unique).
        g_palette = server.gui.add_dropdown(
            "palette", options=[
                "nipy_spectral", "turbo", "gist_rainbow", "gist_ncar",
                "rainbow", "hsv", "tab20",
            ],
            initial_value="nipy_spectral",
        )
        # Post-blend HSV saturation. NOTE: with the new HSV circular-hue blend
        # below, opposite-hue cancellation no longer makes white patches; this
        # slider mostly tunes vividness for argmax/single modes.
        g_soft_sat = server.gui.add_slider(
            "soft saturation", min=1.0, max=3.0, step=0.1, initial_value=2.0,
        )

    # Real-data clip selector — lives at top-level so it's visible in BOTH
    # anim (cache.active.clips_for(g_id)) and cross-retarget (src_td.clips_for
    # (g_src_id)) modes. Options + frame-count auto-refresh per current context.
    _init_src_td = topos[init_ds]
    _init_clips = _init_src_td.clips_for(_init_src_td.id_names[0])
    g_clip = server.gui.add_dropdown(
        "clip (real-data driver)",
        options=_init_clips if _init_clips else ["<none>"],
        initial_value=_init_clips[0] if _init_clips else "<none>",
    )

    with server.gui.add_folder("Anim"):
        g_frame = server.gui.add_slider("frame", min=0, max=n_frames_init - 1, step=1, initial_value=0)
        g_prev_frame = server.gui.add_button("◀ prev frame")
        g_next_frame = server.gui.add_button("next frame ▶")
        g_playing = server.gui.add_checkbox("play", False)
        g_fps = server.gui.add_slider("fps", min=1, max=60, step=1, initial_value=15)
        g_show_axes = server.gui.add_checkbox("joint axes triads", True)
        g_show_bones = server.gui.add_checkbox("bones", True)
        g_show_joints = server.gui.add_checkbox("joint points", True)
        g_show_gt_anim = server.gui.add_checkbox("GT mesh side-by-side", True)
        g_err_color = server.gui.add_checkbox("color pred by L2 error", True)
        g_err_cmap = server.gui.add_dropdown(
            "err cmap",
            options=["YlOrRd", "OrRd", "Reds", "hot", "afmhot", "inferno", "magma", "viridis"],
            initial_value="YlOrRd",
        )
        g_nrm_vis = server.gui.add_dropdown(
            "normal vis (pred)", options=["off", "rgb", "needles"], initial_value="off",
        )
        # Joint position source. Each option lives in a different coord frame:
        # - T_world (animated): rig reference frame (Maya rig positions), where
        #   pred_v actually ends up after skinning. Aligned with pred mesh but
        #   may sit off-surface depending on rig design (e.g. eye joints inside
        #   the eyeball, jaw deep below chin).
        # - bind_pred (per-id): bind_pose_net output for THIS id. Lives on the
        #   per-id neutral mesh surface (where bind GT landmarks were). NOT the
        #   pose the pred mesh is in — overlay is for reference only.
        # - rig_ref: Maya rig bind pose (joint locations from rig_info_ict.json).
        #   Same frame as T_world at neutral; differs as T_delta diverges.
        g_jpos_src = server.gui.add_dropdown(
            "joint position source",
            options=["T_world (animated)", "bind_pred (per-id)", "rig_ref (Maya)"],
            initial_value="T_world (animated)",
        )

    # Cross-retarget: source must support blendshape exp (ICT topos).
    cross_src_options = [k for k, t in topos.items() if t.supports_anim]
    if not cross_src_options:
        cross_src_options = avail_ds  # fallback (degraded UX)
    with server.gui.add_folder("Cross-retarget"):
        g_src_ds = server.gui.add_dropdown(
            "src dataset", options=cross_src_options,
            initial_value=cross_src_options[0],
        )
        g_src_id = server.gui.add_slider(
            "src identity", min=0,
            max=len(topos[cross_src_options[0]].id_names) - 1,
            step=1, initial_value=0,
        )
        g_tgt_ds = server.gui.add_dropdown(
            "tgt dataset", options=avail_ds, initial_value=init_ds,
        )
        g_tgt_id = server.gui.add_slider(
            "tgt identity", min=0,
            max=len(topos[init_ds].id_names) - 1,
            step=1, initial_value=0,
        )
        # PCA mode index — only meaningful when src driver is pca_mode.
        g_pca_mode = server.gui.add_slider(
            "PCA mode idx (PCA src only)", min=0, max=20, step=1, initial_value=0,
        )
        g_show_src = server.gui.add_checkbox("show source meshes (neu+def)", True)
        g_show_tgt_neu = server.gui.add_checkbox("show target neutral", True)
        g_cross_err = server.gui.add_checkbox("self-retarget error overlay", True)
        g_cross_nrm = server.gui.add_dropdown(
            "normal vis (tgt pred)", options=["off", "rgb", "needles"], initial_value="off",
        )
        g_cross_joints = server.gui.add_dropdown("cross joints", options=["off", "bind (neutrals)", "posed (tgt anim)"], initial_value="off")
        g_compare_method = server.gui.add_dropdown(
            "method",
            options=["hlbs (ours)", "nfr", "nfs"],
            initial_value="hlbs (ours)",
        )
        _tta_opts = ["off"] + sorted(_tta_data.keys())
        g_tta = server.gui.add_dropdown(
            "TTA rig (optimized)", options=_tta_opts, initial_value="off",
        )
        # Optional datasets that depend on per-server data (precompute / meshes).
        # Shown so it's clear when one is unavailable on the current host.
        _opt_avail = "  ".join(
            f"{_nm}: {'있음' if _nm in topos else '없음(이 서버)'}"
            for _nm in ("ict_real",))
        server.gui.add_text("optional datasets", _opt_avail, disabled=True)

    with server.gui.add_folder("Feature (Diff3F)"):
        g_feat_mode = server.gui.add_dropdown(
            "feat mode", options=["pca-rgb", "correspondence"], initial_value="pca-rgb",
        )
        _feat_ds0 = "mf" if "mf" in topos else list(topos.keys())[0]
        g_feat_tgt_ds = server.gui.add_dropdown(
            "feat tgt dataset", options=list(topos.keys()), initial_value=_feat_ds0,
        )
        g_feat_tgt_id = server.gui.add_slider(
            "feat tgt id", min=0, max=max(len(topos[_feat_ds0].id_names) - 1, 1),
            step=1, initial_value=0,
        )
        g_feat_shared = server.gui.add_checkbox("shared PCA (src+tgt)", True)
        g_anchor = server.gui.add_slider(
            "anchor vertex (src, corr mode)", min=0, max=20000, step=1, initial_value=0,
        )

    with server.gui.add_folder("Sequence stats"):
        g_seq_stats_md = server.gui.add_markdown(
            "**seq stats**: idle — click Compute"
        )
        g_seq_stats_btn = server.gui.add_button("Compute (current seq)")
        g_seq_stats_force = server.gui.add_button(
            "Force recompute (ignore cache)"
        )

    with server.gui.add_folder("Render to video"):
        g_render_dir = server.gui.add_text(
            "render BASE dir",
            str(_REPO / "_diag" / "render"),
        )
        g_render_subpath = server.gui.add_text(
            "auto subpath (computed)", "(updates on mode/id/clip change)",
            disabled=True,
        )
        g_render_name = server.gui.add_text("video filename", "out.mp4")
        g_render_fps = server.gui.add_slider(
            "video fps", min=1, max=60, step=1, initial_value=30,
        )
        g_render_w = server.gui.add_slider(
            "render width", min=256, max=1920, step=64, initial_value=1200,
        )
        g_render_h = server.gui.add_slider(
            "render height", min=256, max=1080, step=64, initial_value=720,
        )
        g_render_keep_pngs = server.gui.add_checkbox(
            "keep per-frame PNGs (uncheck = video only)", False,
        )
        g_render_headless = server.gui.add_checkbox(
            "headless capture client (server-side, faster over remote)", False,
        )
        g_preview_btn = server.gui.add_button(
            "Preview (capture current frame @ chosen W/H)"
        )
        g_render_btn = server.gui.add_button(
            "Render video (sweeps all frames @ current view)"
        )
        g_render_progress = server.gui.add_text("render progress", "idle", disabled=True)
        # Pre-create the preview image inside this folder so it's always visible
        # in the Render section. Initialized to a tiny placeholder; .image
        # setter swaps in the real capture on preview-click.
        _placeholder = np.full((8, 8, 3), 220, dtype=np.uint8)
        _preview_handle = server.gui.add_image(
            image=_placeholder, label="preview — click button above to capture",
        )
        _preview_state = {"handle": _preview_handle}

    g_status = server.gui.add_text("status", "ready", disabled=True)

    # ───── scene-node registry (so we can clear between mode switches) ──
    nodes: list = []

    def _tile(c, n):
        """(r,g,b) tuple or [3] array → [n, 3] uint8."""
        return np.tile(np.asarray(c, dtype=np.uint8).reshape(1, 3), (n, 1))

    def _clear():
        nonlocal nodes
        for h in nodes:
            try:
                h.remove()
            except Exception:
                pass
        nodes = []

    # ───── render passes ────────────────────────────────────────────────

    def _render_bind_pose():
        id_idx = int(g_id.value)
        c = cache.get(id_idx)
        verts = c["neu_v"]
        # NOTE: We deliberately route bind_pose through the GLB path (with our
        # igl-computed vertex normals embedded) instead of add_mesh_simple.
        # add_mesh_simple lets three.js auto-compute normals from faces; for
        # ICT's quad-triangulated mesh that produces visible diagonal contour
        # artifacts. igl per_vertex_normals (area-weighted) avoids this.
        rgb = _tile(g_mesh_color.value, verts.shape[0])
        m = _add_per_vertex_color_mesh(
            server, "/mesh", verts, cache.faces, rgb,
            opacity=float(g_mesh_opacity.value), shading=g_shading.value,
        )
        nodes.append(m)

        pred = c["joint_pos_pred"]      # [J, 3] or None (model didn't run)
        gt = c["gt_bind"]                # [J, 3] or None

        if pred is None:
            g_status.value = (f"id={id_idx} ds={cache.active.name} | model skipped "
                              f"(no nfs cache) — mesh only")
            return

        # Helper coloring: filled-circle big for non-helper, ring for helper.
        sizes_pred = np.full(J, 0.012, dtype=np.float32)
        sizes_gt = np.full(J, 0.012, dtype=np.float32)
        for j in helper_set:
            if 0 <= j < J:
                sizes_pred[j] = 0.018 if g_show_helpers.value else 0.012
                sizes_gt[j] = 0.018 if g_show_helpers.value else 0.012

        if g_show_pred.value:
            cols_pred = np.tile(np.array([[255, 140, 30]], dtype=np.uint8), (J, 1))
            if g_show_helpers.value:
                for j in helper_set:
                    if 0 <= j < J:
                        cols_pred[j] = [255, 80, 200]
            h = server.scene.add_point_cloud(
                "/joints/pred", points=pred, colors=cols_pred, point_size=0.012
            )
            nodes.append(h)
            if g_show_helpers.value and g_show_jlabels.value:
                for _hj in helper_set:
                    if 0 <= _hj < J and _hj in helper_label_map:
                        try:
                            nodes.append(server.scene.add_label(
                                f"/joints/hlbl/{_hj}", text=helper_label_map[_hj],
                                position=tuple(float(_x) for _x in pred[_hj])))
                        except Exception:
                            pass
            if g_bind_bones.value:
                _bsegs = [[pred[int(parent_idx[j])], pred[j]] for j in range(J) if int(parent_idx[j]) >= 0]
                if _bsegs:
                    _bpts = np.array(_bsegs, dtype=np.float32)
                    _bcols = np.broadcast_to(np.array([80, 220, 180], dtype=np.uint8), (_bpts.shape[0], 2, 3)).copy()
                    nodes.append(server.scene.add_line_segments("/joints/pred_bones", points=_bpts, colors=_bcols, line_width=2.0))

        if g_show_gt.value and gt is not None:
            cols_gt = np.tile(np.array([[60, 130, 255]], dtype=np.uint8), (J, 1))
            if g_show_helpers.value:
                for j in helper_set:
                    if 0 <= j < J:
                        cols_gt[j] = [60, 200, 255]
            h = server.scene.add_point_cloud(
                "/joints/gt", points=gt, colors=cols_gt, point_size=0.014
            )
            nodes.append(h)

        if g_show_err.value and gt is not None:
            points = np.stack([gt, pred], axis=1).astype(np.float32)  # [J, 2, 3]
            err = np.linalg.norm(pred - gt, axis=-1)                  # [J]
            err_n = _err_norm(err)
            cols = _viridis_rgb(err_n)
            h = server.scene.add_arrows(
                "/joints/err_arrows", points=points, colors=cols,
                shaft_radius=0.0015, head_radius=0.004, head_length=0.008,
            )
            nodes.append(h)
            g_status.value = (
                f"id={id_idx} | bind err: mean={err.mean()*1000:.2f}mm "
                f"med={np.median(err)*1000:.2f}mm max={err.max()*1000:.2f}mm"
            )
        else:
            g_status.value = f"id={id_idx} | bind pose (no GT cache)" if gt is None else f"id={id_idx}"

    def _render_weight():
        id_idx = int(g_id.value)
        c = cache.get(id_idx)
        verts = c["neu_v"]
        W = c["W"]                                 # [V, J] or None
        if W is None:
            # mesh-only fallback
            rgb = _tile(g_mesh_color.value, verts.shape[0])
            h = _add_per_vertex_color_mesh(server, "/mesh", verts, cache.faces, rgb, opacity=float(g_mesh_opacity.value), shading=g_shading.value, double_sided=g_double_sided.value, sat=float(g_global_sat.value))
            nodes.append(h)
            g_status.value = (f"id={id_idx} ds={cache.active.name} | model skipped "
                              f"(no nfs cache) — mesh only")
            return
        mode = g_w_mode.value

        if mode == "single":
            j = int(g_joint.value.split()[0])
            vals = W[:, j]
            vals = vals / max(vals.max(), 1e-8)
            rgb = _viridis_rgb(vals)
            g_status.value = (
                f"id={id_idx} j={j}({rig.joint_names[j]}) | "
                f"W max={W[:, j].max():.3f} mean={W[:, j].mean():.3f}"
            )
            if g_w_mark_joint.value and c["joint_pos_pred"] is not None:
                _jp1 = c["joint_pos_pred"][j]
                _mw = float(verts[:, 0].max() - verts[:, 0].min())
                nodes.append(server.scene.add_icosphere(
                    "/joints/selected", radius=_mw * 0.012, color=(255, 40, 40),
                    position=tuple(float(x) for x in _jp1)))
        elif mode in ("argmax", "soft"):
            # Build evenly-sampled palette across J joints. Wide-spectrum cmaps
            # (nipy_spectral / turbo / gist_rainbow) give maximal contrast for
            # large J — proven approach from utils/matplotlib_rnd.py.
            pname = g_palette.value
            cmap = cm.get_cmap(pname)
            palette_f = np.asarray(
                [cmap(i / max(J - 1, 1))[:3] for i in range(J)], dtype=np.float32
            )  # [J, 3] in [0, 1]
            Wf = W.astype(np.float32)
            if mode == "argmax":
                idx = np.argmax(Wf, axis=-1)
                rgb_f = palette_f[idx]
            else:  # soft
                # Top-K mask zeros out long-tail noise before blending.
                K = max(int(g_soft_topk.value), 1)
                if K < J:
                    topk_idx = np.argpartition(-Wf, K - 1, axis=-1)[:, :K]
                    mask = np.zeros_like(Wf)
                    np.put_along_axis(mask, topk_idx, 1.0, axis=-1)
                    Wm = Wf * mask
                    Wm = Wm / (Wm.sum(-1, keepdims=True) + 1e-8)
                else:
                    Wm = Wf
                rgb_f = Wm @ palette_f                                        # [V, 3]
                # Per-channel stretch — pushes blended colors out of mid-gray
                # band so cheek/forehead/etc. stay distinct.
                ch_min = rgb_f.min(0, keepdims=True)
                ch_max = rgb_f.max(0, keepdims=True)
                rgb_f = (rgb_f - ch_min) / (ch_max - ch_min + 1e-8)
            rgb = np.clip(rgb_f * 255, 0, 255).astype(np.uint8)
            uniq = len(np.unique(np.argmax(Wf, axis=-1)))
            tag = mode if mode == "argmax" else f"soft top-{int(g_soft_topk.value)}"
            g_status.value = (
                f"id={id_idx} | {tag} | palette={pname} | {uniq}/{J} dominant joints"
            )
        else:  # entropy
            eps = 1e-8
            ent = -np.sum(W * np.log(W + eps), axis=-1)   # [V]
            max_ent = np.log(J)
            rgb = _viridis_rgb(ent / max_ent)
            g_status.value = (
                f"id={id_idx} | entropy | mean={ent.mean():.3f} "
                f"max={ent.max():.3f} (log J = {max_ent:.3f})"
            )

        h = _add_per_vertex_color_mesh(server, "/mesh", verts, cache.faces, rgb, opacity=float(g_mesh_opacity.value), shading=g_shading.value, double_sided=g_double_sided.value, sat=float(g_global_sat.value))
        nodes.append(h)
        # Skeleton overlay: hidden by default; toggle renders it bind-pose
        # style (pred joints orange, helpers pink, teal bones) or fancy (maya).
        if g_w_show_joints.value and c["joint_pos_pred"] is not None:
            _jp = c["joint_pos_pred"]
            if g_skel_style.value.startswith("fancy"):
                _fv, _ff, _fc = _fancy_skel_mesh(_jp, parent_idx, helper_set, ball_r=float(g_fancy_joint.value), bone_r=float(g_fancy_bone.value))
                nodes.append(_add_per_vertex_color_mesh(
                    server, "/joints/weight_fancy", _fv, _ff, _fc,
                    opacity=1.0, shading=g_shading.value, double_sided=False, sat=1.0))
            else:
                _cols = np.tile(np.array([[255, 140, 0]], dtype=np.uint8), (J, 1))
                for _j2 in helper_set:
                    if 0 <= _j2 < J:
                        _cols[_j2] = [255, 80, 200]
                nodes.append(server.scene.add_point_cloud(
                    "/joints/pred", points=_jp, colors=_cols, point_size=0.010))
                _bs = []
                for _j2 in range(J):
                    _p2 = int(parent_idx[_j2])
                    if _p2 >= 0:
                        _bs.append([_jp[_p2], _jp[_j2]])
                if _bs:
                    _bp = np.array(_bs, dtype=np.float32)
                    _bc = np.broadcast_to(np.array([80, 220, 180], dtype=np.uint8),
                                          (_bp.shape[0], 2, 3)).copy()
                    nodes.append(server.scene.add_line_segments(
                        "/joints/pred_bones", points=_bp, colors=_bc, line_width=2.0))

    def _build_exp(td, id_idx, frame):
        """Build exp_coeff per driver. Returns (exp, n_frames_for_this_clip) or
        (None, 0) if exp can't be built for the current GUI state."""
        if td.exp_driver == "ict_blendshape":
            ec = state["exp_coeffs"]
            return ec[frame % ec.shape[0]], ec.shape[0]
        if td.exp_driver == "mf_real":
            clip = g_clip.value
            nm = td.id_names[id_idx]
            nf = td.n_frames(nm, clip)
            if nf == 0:
                return None, 0
            return (clip, frame % nf), nf
        if td.exp_driver == "pca_mode":
            period = max(int(state["exp_coeffs"].shape[0]), 60)
            amp = float(np.sin(2 * np.pi * (frame % period) / period))
            return np.array([int(g_pca_mode.value), amp], dtype=np.float32), period
        return None, 0

    def _render_anim():
        td = cache.active
        id_idx = int(g_id.value)
        if not td.supports_anim:
            c = cache.get(id_idx)
            rgb = _tile(g_mesh_color.value, c["neu_v"].shape[0])
            h = _add_per_vertex_color_mesh(server, "/anim/neu", c["neu_v"], cache.faces, rgb, opacity=float(g_mesh_opacity.value), shading=g_shading.value, double_sided=g_double_sided.value, sat=float(g_global_sat.value))
            nodes.append(h)
            g_status.value = (f"ds={td.name}: anim mode unsupported "
                              f"(no exp driver) — neutral mesh only")
            return
        frame = int(g_frame.value)
        exp, nf = _build_exp(td, id_idx, frame)
        if exp is None:
            g_status.value = (f"ds={td.name} id={id_idx} driver={td.exp_driver}: "
                              f"no usable exp source (clip={g_clip.value!r}) — pick a clip")
            return

        out = cache.forward_frame(id_idx, exp)
        gt_v = out["gt_v"]; pred_v = out["pred_v"]
        T_world = out["T_world"]
        # Joint position source — user-selectable for diagnosing why joints
        # appear off-mesh. Defaults to T_world (the animated, FK-resolved pos).
        src = g_jpos_src.value
        if src.startswith("bind_pred"):
            jp = out["joint_pos"] if out["joint_pos"] is not None else None
        elif src.startswith("rig_ref"):
            jp = rig.bind_pos.cpu().numpy()
        else:  # T_world
            jp = T_world[:, :3, 3] if T_world is not None else out["joint_pos"]

        if gt_v is None:
            g_status.value = f"ds={td.name} id={id_idx} f={frame}: apply_exp returned None"
            return
        if pred_v is None:
            # mesh-only animation: show GT only
            rgb_gt = _tile(g_mesh_color.value, gt_v.shape[0])
            h = _add_per_vertex_color_mesh(server, "/anim/gt", gt_v, cache.faces, rgb_gt, opacity=float(g_mesh_opacity.value), shading=g_shading.value, double_sided=g_double_sided.value, sat=float(g_global_sat.value))
            nodes.append(h)
            g_status.value = (f"id={id_idx} f={frame:03d}/{nf-1} ds={td.name} | "
                              f"model skipped (no nfs cache) — GT mesh only")
            return

        if g_err_color.value:
            err = np.linalg.norm(pred_v - gt_v, axis=-1)             # [V]
            err_n = _err_norm2(err, key=("anim", g_dataset.value, g_anim_seq.value, err.size), faces=cache.faces, verts=gt_v)
            # hot: black (err≈0) → red → yellow → white (err=max). At low err the
            # heat is black, alpha is also low → mesh_color shows through. At
            # high err alpha=1 → full white/yellow highlight.
            heat = _err_rgb(err_n, g_err_cmap.value).astype(np.float32)
            base = np.tile(
                np.asarray(g_mesh_color.value, dtype=np.float32),
                (pred_v.shape[0], 1),
            )
            a = err_n[:, None]   # 0 → keep mesh color; 1 → full hot color
            rgb_pred = np.clip((1 - a) * base + a * heat, 0, 255).astype(np.uint8)
            err_mm = err.mean() * 1000.0
        else:
            rgb_pred = _tile(g_mesh_color.value, pred_v.shape[0])
            err_mm = float("nan")

        if g_nrm_vis.value == "rgb":
            _prn = _per_vertex_normal(pred_v, cache.faces.astype(np.int64))
            rgb_pred = ((_prn + 1.0) * 0.5 * 255).astype(np.uint8)

        # Layout: pred at origin, GT shifted +X by mesh width.
        x_off = float(pred_v[:, 0].max() - pred_v[:, 0].min()) * 1.15

        h = _add_per_vertex_color_mesh(server, "/anim/pred", pred_v, cache.faces, rgb_pred, opacity=float(g_mesh_opacity.value), shading=g_shading.value, double_sided=g_double_sided.value, sat=float(g_global_sat.value))
        nodes.append(h)
        if g_nrm_vis.value == "needles":
            _prn = _per_vertex_normal(pred_v, cache.faces.astype(np.int64))
            _w0 = float(pred_v[:, 0].max() - pred_v[:, 0].min())
            _st = max(1, pred_v.shape[0] // 4000)
            _p0 = pred_v[::_st]; _p1 = _p0 + _prn[::_st] * (_w0 * 0.02)
            _pts = np.stack([_p0, _p1], axis=1).astype(np.float32)
            _cols = np.broadcast_to(np.array([255, 200, 60], dtype=np.uint8),
                                    (_pts.shape[0], 2, 3)).copy()
            nodes.append(server.scene.add_line_segments(
                "/anim/normals", points=_pts, colors=_cols, line_width=1.0))
        if g_show_gt_anim.value:
            shifted = gt_v.copy()
            shifted[:, 0] += x_off
            rgb_gt = _tile(g_mesh_color.value, gt_v.shape[0])
            h2 = _add_per_vertex_color_mesh(server, "/anim/gt", shifted, cache.faces, rgb_gt, opacity=float(g_mesh_opacity.value), shading=g_shading.value, double_sided=g_double_sided.value, sat=float(g_global_sat.value))
            nodes.append(h2)

        # Skeleton on pred side: joint points + bones
        _fancy = g_skel_style.value.startswith("fancy")
        if _fancy and (g_show_joints.value or g_show_bones.value) and jp is not None:
            _fv, _ff, _fc = _fancy_skel_mesh(jp, parent_idx, helper_set, ball_r=float(g_fancy_joint.value), bone_r=float(g_fancy_bone.value))
            nodes.append(_add_per_vertex_color_mesh(
                server, "/anim/skel_fancy", _fv, _ff, _fc,
                opacity=1.0, shading=g_shading.value, double_sided=False, sat=1.0))
        if (not _fancy) and g_show_joints.value:
            h3 = server.scene.add_point_cloud(
                "/anim/joints",
                points=jp,
                colors=np.full((J, 3), 255, dtype=np.uint8),
                point_size=0.008,
            )
            nodes.append(h3)

        if (not _fancy) and g_show_bones.value:
            segs = []
            for j in range(J):
                p = int(parent_idx[j])
                if p >= 0:
                    segs.append([jp[p], jp[j]])
            if segs:
                pts = np.array(segs, dtype=np.float32)            # [E, 2, 3]
                # add_line_segments wants per-endpoint colors: [E, 2, 3]
                cols = np.broadcast_to(
                    np.array([80, 220, 180], dtype=np.uint8),
                    (pts.shape[0], 2, 3),
                ).copy()
                h4 = server.scene.add_line_segments(
                    "/anim/bones", points=pts, colors=cols, line_width=2.0,
                )
                nodes.append(h4)

        if g_show_axes.value:
            # Plant a small frame at each joint, rotation = T_world[j, :3, :3]
            # T_world is column-major in this codebase. Build wxyz quat.
            from scipy.spatial.transform import Rotation as R

            R_mats = T_world[:, :3, :3]  # [J, 3, 3]
            # quaternions: scipy returns xyzw → reorder to wxyz
            quats_xyzw = R.from_matrix(R_mats).as_quat()
            quats_wxyz = np.concatenate(
                [quats_xyzw[:, 3:4], quats_xyzw[:, :3]], axis=-1
            )
            for j in range(J):
                hf = server.scene.add_frame(
                    f"/anim/axes/{j:02d}",
                    wxyz=tuple(quats_wxyz[j]),
                    position=tuple(jp[j]),
                    axes_length=0.025,
                    axes_radius=0.0015,
                    show_axes=True,
                    origin_radius=0.0,
                )
                nodes.append(hf)

        g_status.value = (
            f"ds={td.name}[{id_idx}] f={frame:03d}/{nf-1} drv={td.exp_driver} | "
            f"mean pred err {err_mm:.2f}mm" if g_err_color.value
            else f"ds={td.name}[{id_idx}] f={frame:03d}/{nf-1}"
        )

    def _render_cross():
        src_td = topos[g_src_ds.value]
        tgt_td = topos[g_tgt_ds.value]
        src_idx = int(g_src_id.value); tgt_idx = int(g_tgt_id.value)
        ec = state["exp_coeffs"]
        nf = ec.shape[0]
        frame = int(g_frame.value)

        # Build exp_coeff per src driver.
        src_id_name = src_td.id_names[src_idx]
        if src_td.exp_driver in ("ict_blendshape", "ict_real_blend"):
            exp = ec[frame % nf]
        elif src_td.exp_driver == "pca_mode":
            amp = float(np.sin(2 * np.pi * (frame % nf) / nf))
            exp = np.array([int(g_pca_mode.value), amp], dtype=np.float32)
        elif src_td.exp_driver == "mf_real":
            clip = g_clip.value
            n_real = src_td.n_frames(src_id_name, clip)
            if n_real == 0:
                g_status.value = (f"cross: clip {clip!r} has no frames for "
                                  f"{src_td.name}[{src_id_name}]")
                return
            exp = (clip, frame % n_real)
        else:
            g_status.value = f"cross: src {src_td.name} has no exp driver"
            return

        out = cache.retarget(src_td, src_idx, exp, tgt_td, tgt_idx)
        # If the user picked a baseline method (nfr / nfs), override tgt_pred_v
        # with that baseline's prediction. src_neu/src_def/tgt_neu stay as
        # they are (computed from topologies, method-independent).
        method = g_compare_method.value
        if method != "hlbs (ours)" and out["src_def_v"] is not None and out["tgt_neu_v"] is not None:
            try:
                def _set_status(s):
                    g_status.value = s
                baseline_pred = baseline.retarget(
                    method=method.split()[0],
                    src_td=src_td, src_idx=src_idx,
                    src_neu_v=out["src_neu_v"], src_def_v=out["src_def_v"],
                    tgt_td=tgt_td, tgt_idx=tgt_idx, tgt_neu_v=out["tgt_neu_v"],
                    progress_cb=_set_status,
                )
                if baseline_pred is not None:
                    out["tgt_pred_v"] = baseline_pred
                    out["model_ran"] = True
                    # Recompute self-retarget metrics with this baseline's pred.
                    if (src_td.name == tgt_td.name) and (src_idx == tgt_idx):
                        out["metrics"] = _compute_metrics(
                            baseline_pred, out["src_def_v"], tgt_td.faces)
                    else:
                        out["metrics"] = None
            except Exception:
                import traceback as _tb
                print(f"[baseline {method}] EXC:", _tb.format_exc())
                g_status.value = f"baseline {method} failed — see console"
                return
        if not out["model_ran"]:
            g_status.value = (f"cross: src={src_td.name}[{src_idx}] "
                              f"tgt={tgt_td.name}[{tgt_idx}] — "
                              f"model skipped (missing nfs cache on one side)")
            # still show source mesh if we have it
            if out["src_def_v"] is not None:
                rgb = _tile(g_src_color.value, out["src_def_v"].shape[0])
                h = _add_per_vertex_color_mesh(
                    server, "/cross/src_def", out["src_def_v"], src_td.faces, rgb,
                    opacity=float(g_mesh_opacity.value), shading=g_shading.value, double_sided=g_double_sided.value, sat=float(g_global_sat.value))
                nodes.append(h)
            # ... and the target neutral, so a feat-less target (e.g. stylized
            # mesh without Diff3F cache) still shows up when clicked.
            if out["tgt_neu_v"] is not None and g_show_tgt_neu.value:
                _sw = float(out["src_def_v"][:, 0].max() - out["src_def_v"][:, 0].min()) if out["src_def_v"] is not None else 0.0
                _tv = out["tgt_neu_v"].copy(); _tv[:, 0] += _sw * 1.2
                rgb = _tile(g_tgt_color.value, _tv.shape[0])
                h = _add_per_vertex_color_mesh(
                    server, "/cross/tgt_neu", _tv, tgt_td.faces, rgb,
                    opacity=float(g_mesh_opacity.value), shading=g_shading.value, double_sided=g_double_sided.value, sat=float(g_global_sat.value))
                nodes.append(h)
            return

        # Layout: src_neu (left -2), src_def (left -1), tgt_neu (mid 0), tgt_pred (right +1)
        # Use mesh widths per topology so meshes don't overlap.
        def _w(v):
            return float(v[:, 0].max() - v[:, 0].min())
        src_w = max(_w(out["src_neu_v"]), _w(out["src_def_v"]))
        tgt_w = max(_w(out["tgt_neu_v"]), _w(out["tgt_pred_v"]))
        gap = max(src_w, tgt_w) * 0.20
        slot_w = max(src_w, tgt_w) + gap

        positions = {
            "src_neu":  -2 * slot_w if g_show_src.value else None,
            "src_def":  -1 * slot_w if g_show_src.value else None,
            "tgt_neu":   0.0 if g_show_tgt_neu.value else None,
            "tgt_pred":  1.0 * slot_w,
        }

        def _put(name, verts, faces, rgb):
            x = positions[name]
            if x is None:
                return
            v = verts.copy(); v[:, 0] += x
            h = _add_per_vertex_color_mesh(server, f"/cross/{name}", v, faces, rgb,
                                           opacity=float(g_mesh_opacity.value),
                                           shading=g_shading.value, double_sided=g_double_sided.value, sat=float(g_global_sat.value))
            nodes.append(h)

        # src_neu = lighter src tint; src_def = src tint as-is (saturation already
        # boosted globally). tgt_neu uses tgt color. tgt_pred uses tgt color if
        # there are no metrics, else viridis error heatmap.
        src_c = np.asarray(g_src_color.value, dtype=np.int16)
        src_neu_c = np.clip(src_c + 25, 0, 255).astype(np.uint8)  # slightly lighter
        tgt_c = np.asarray(g_tgt_color.value, dtype=np.uint8)
        _put("src_neu", out["src_neu_v"], src_td.faces, _tile(src_neu_c, out["src_neu_v"].shape[0]))
        _put("src_def", out["src_def_v"], src_td.faces, _tile(g_src_color.value, out["src_def_v"].shape[0]))
        _put("tgt_neu", out["tgt_neu_v"], tgt_td.faces, _tile(tgt_c, out["tgt_neu_v"].shape[0]))

        m = out["metrics"]
        if g_cross_err.value and m is not None:
            err = np.linalg.norm(out["tgt_pred_v"] - out["src_def_v"], axis=-1)
            err_n = _err_norm2(err, key=("cross", err.size), faces=tgt_td.faces, verts=out["tgt_neu_v"])
            heat = _err_rgb(err_n, g_err_cmap.value).astype(np.float32)
            base = np.tile(np.asarray(tgt_c, dtype=np.float32),
                           (out["tgt_pred_v"].shape[0], 1))
            a = err_n[:, None]
            rgb_pred = np.clip((1 - a) * base + a * heat, 0, 255).astype(np.uint8)
        else:
            rgb_pred = _tile(tgt_c, out["tgt_pred_v"].shape[0])
        if g_cross_nrm.value == "rgb":
            _tn = _per_vertex_normal(out["tgt_pred_v"], tgt_td.faces.astype(np.int64))
            rgb_pred = ((_tn + 1.0) * 0.5 * 255).astype(np.uint8)
        _put("tgt_pred", out["tgt_pred_v"], tgt_td.faces, rgb_pred)
        if g_cross_nrm.value == "needles":
            _tn = _per_vertex_normal(out["tgt_pred_v"], tgt_td.faces.astype(np.int64))
            _tv2 = out["tgt_pred_v"].copy(); _tv2[:, 0] += positions["tgt_pred"]
            _st = max(1, _tv2.shape[0] // 4000)
            _p0 = _tv2[::_st]; _p1 = _p0 + _tn[::_st] * (tgt_w * 0.02)
            _pts = np.stack([_p0, _p1], axis=1).astype(np.float32)
            _cols = np.broadcast_to(np.array([255, 200, 60], dtype=np.uint8),
                                    (_pts.shape[0], 2, 3)).copy()
            nodes.append(server.scene.add_line_segments(
                "/cross/normals", points=_pts, colors=_cols, line_width=1.0))

        # ── optional joint overlays (#5) ──────────────────────────────
        _cj = g_cross_joints.value
        if _cj != "off":
            def _draw_j(prefix, jp, x_off, bones=True, axes=False, Tw=None):
                if jp is None or x_off is None:
                    return
                jp = jp.copy(); jp[:, 0] += x_off
                nodes.append(server.scene.add_point_cloud(
                    prefix + "/pts", points=jp.astype(np.float32),
                    colors=np.tile(np.array([255, 255, 255], dtype=np.uint8), (len(jp), 1)),
                    point_size=0.009))
                if bones:
                    if g_skel_style.value.startswith("fancy"):
                        _fv, _ff, _fc = _fancy_skel_mesh(jp, parent_idx, helper_set=helper_set, ball_r=float(g_fancy_joint.value), bone_r=float(g_fancy_bone.value))
                        nodes.append(_add_per_vertex_color_mesh(
                            server, prefix + "/bones_fancy", _fv, _ff, _fc,
                            shading="smooth", sat=1.0))
                    else:
                        _seg = [[jp[int(parent_idx[j])], jp[j]] for j in range(len(jp)) if int(parent_idx[j]) >= 0]
                        if _seg:
                            _p = np.array(_seg, dtype=np.float32)
                            _c = np.broadcast_to(np.array([80, 220, 180], dtype=np.uint8), (_p.shape[0], 2, 3)).copy()
                            nodes.append(server.scene.add_line_segments(prefix + "/bones", points=_p, colors=_c, line_width=2.0))
                if axes and Tw is not None:
                    from scipy.spatial.transform import Rotation as _R
                    _q = _R.from_matrix(Tw[:, :3, :3]).as_quat()
                    _q = np.concatenate([_q[:, 3:4], _q[:, :3]], axis=-1)
                    for j in range(len(jp)):
                        nodes.append(server.scene.add_frame(
                            prefix + f"/ax/{j:02d}", wxyz=tuple(_q[j]), position=tuple(jp[j]),
                            axes_length=0.02, axes_radius=0.0012, show_axes=True, origin_radius=0.0))
            if _cj == "bind (neutrals)":
                _draw_j("/cross/sj", out.get("src_joint_pos"), positions["src_neu"], bones=True)
                _draw_j("/cross/tj", out.get("tgt_joint_pos"), positions["tgt_neu"], bones=True)
            elif _cj == "posed (tgt anim)":
                _Tw = out.get("tgt_T_world")
                _jp = _Tw[:, :3, 3] if _Tw is not None else None
                _draw_j("/cross/tjp", _jp, positions["tgt_pred"], bones=True, axes=True, Tw=_Tw)

        method_tag = method.split()[0] if method != "hlbs (ours)" else "hlbs"
        if m is not None:
            g_status.value = (
                f"cross[self/{method_tag}] src={src_td.name}[{src_idx}] f={frame:03d} | "
                f"L2 mean={m['mean_l2_mm']:.2f}mm "
                f"max={m['max_l2_mm']:.2f}mm | "
                f"norm_cos={m['norm_cos']:.4f} lap_err={m['lap_err']:.4f}"
            )
        else:
            g_status.value = (
                f"cross[{method_tag}] src={src_td.name}[{src_idx}] → "
                f"tgt={tgt_td.name}[{tgt_idx}] f={frame:03d} | no GT"
            )

    def _mesh_kw():
        return dict(opacity=float(g_mesh_opacity.value), shading=g_shading.value,
                    double_sided=g_double_sided.value, sat=float(g_global_sat.value))

    def _render_feat():
        td = cache.active
        src_idx = min(int(g_id.value), len(td.id_names) - 1)
        src_v = cache.get(src_idx)["neu_v"]
        Fs = _feat_raw(td, src_idx)
        if Fs is None or Fs.shape[0] != src_v.shape[0]:
            rgb = _tile(g_mesh_color.value, src_v.shape[0])
            nodes.append(_add_per_vertex_color_mesh(server, "/feat/src", src_v, td.faces, rgb, **_mesh_kw()))
            g_status.value = (f"feat: Diff3F cache missing/mismatch for {td.name}[{src_idx}]"
                              f" (feat={None if Fs is None else Fs.shape})")
            return
        tgt_td = topos[g_feat_tgt_ds.value]
        tgt_idx = min(int(g_feat_tgt_id.value), len(tgt_td.id_names) - 1)
        same = (tgt_td.name == td.name and tgt_idx == src_idx)
        slot = float(src_v[:, 0].max() - src_v[:, 0].min()) * 1.2

        if g_feat_mode.value == "pca-rgb":
            if g_feat_shared.value and not same:
                Ft = _feat_raw(tgt_td, tgt_idx)
                tgt_v = cache.get(tgt_idx, tgt_td)["neu_v"]
                if Ft is not None and Ft.shape[0] == tgt_v.shape[0]:
                    rgb_s, rgb_t = _feat_pca_rgb([Fs, Ft])
                    nodes.append(_add_per_vertex_color_mesh(server, "/feat/src", src_v, td.faces, rgb_s, **_mesh_kw()))
                    tv = tgt_v.copy(); tv[:, 0] += slot
                    nodes.append(_add_per_vertex_color_mesh(server, "/feat/tgt", tv, tgt_td.faces, rgb_t, **_mesh_kw()))
                    g_status.value = (f"feat pca-rgb SHARED: {td.name}[{src_idx}] <-> "
                                      f"{tgt_td.name}[{tgt_idx}] (same color = same semantic region)")
                    return
            rgb_s = _feat_pca_rgb([Fs])[0]
            nodes.append(_add_per_vertex_color_mesh(server, "/feat/src", src_v, td.faces, rgb_s, **_mesh_kw()))
            g_status.value = f"feat pca-rgb: {td.name}[{src_idx}] solo"
            return

        # correspondence: anchor-vertex cossim heatmap (src self-sim + tgt sim)
        Ft = _feat_raw(tgt_td, tgt_idx)
        tgt_v = cache.get(tgt_idx, tgt_td)["neu_v"]
        if Ft is None or Ft.shape[0] != tgt_v.shape[0]:
            g_status.value = f"feat corr: tgt Diff3F cache missing for {tgt_td.name}[{tgt_idx}]"
            return
        a = int(g_anchor.value) % Fs.shape[0]
        fs = Fs / np.clip(np.linalg.norm(Fs, axis=1, keepdims=True), 1e-8, None)
        ft = Ft / np.clip(np.linalg.norm(Ft, axis=1, keepdims=True), 1e-8, None)
        sim_s = fs @ fs[a]; sim_t = ft @ fs[a]
        def _n01(x):
            lo, hi = float(x.min()), float(x.max())
            return (x - lo) / max(hi - lo, 1e-8)
        nodes.append(_add_per_vertex_color_mesh(server, "/feat/src", src_v, td.faces,
                                                _viridis_rgb(_n01(sim_s)), **_mesh_kw()))
        tv = tgt_v.copy(); tv[:, 0] += slot
        nodes.append(_add_per_vertex_color_mesh(server, "/feat/tgt", tv, tgt_td.faces,
                                                _viridis_rgb(_n01(sim_t)), **_mesh_kw()))
        bi = int(np.argmax(sim_t))
        nodes.append(server.scene.add_icosphere(
            "/feat/anchor", radius=slot * 0.012, color=(255, 40, 40),
            position=tuple(float(x) for x in src_v[a])))
        nodes.append(server.scene.add_icosphere(
            "/feat/best", radius=slot * 0.012, color=(40, 255, 60),
            position=tuple(float(x) for x in tv[bi])))
        g_status.value = (f"corr: {td.name}[{src_idx}] v{a} -> {tgt_td.name}[{tgt_idx}] "
                          f"best v{bi} sim {sim_t[bi]:.3f} (mean {sim_t.mean():.3f}, min {sim_t.min():.3f})")

    # ── render orchestration + locking ─────────────────────────────────
    import threading, time, traceback
    render_lock = threading.Lock()

    def render():
        with render_lock:
            try:
                _clear()
                _hide_clay_lights()   # re-lit only for meshes drawn this frame
                mode = g_mode.value
                if mode == "bind_pose":
                    _render_bind_pose()
                elif mode == "weight":
                    _render_weight()
                elif mode == "anim":
                    _render_anim()
                elif mode == "feat":
                    _render_feat()
                else:  # cross
                    _render_cross()
            except Exception:
                print("[render] EXC:", traceback.format_exc())
        try:
            _refresh_render_subpath()
        except NameError:
            pass  # called before _refresh_render_subpath defined (startup only)

    # ── dataset / seq change handlers ──────────────────────────────────
    def _on_dataset_change(_e=None):
        ds = g_dataset.value
        if ds not in topos:
            g_status.value = f"dataset {ds} not registered"
            return
        cache.set_active(topos[ds])
        g_id.max = len(topos[ds].id_names) - 1
        if int(g_id.value) > g_id.max:
            g_id.value = 0
        _refresh_clip_options()
        render()

    def _on_id_change(_e=None):
        _refresh_clip_options()
        render()

    def _refresh_clip_options():
        """Re-populate g_clip + adjust g_frame.max based on the currently driven
        context (anim mode → cache.active; cross mode → src_td)."""
        if g_mode.value == "cross":
            td = topos.get(g_src_ds.value)
            idx = int(g_src_id.value)
        else:
            td = cache.active
            idx = int(g_id.value)
        if td is None or not td.id_names:
            return
        idx = min(idx, len(td.id_names) - 1)
        clips = td.clips_for(td.id_names[idx])
        if not clips:
            clips = ["<none>"]
        g_clip.options = tuple(clips)
        if g_clip.value not in clips:
            g_clip.value = clips[0]
        # Sync frame slider max to current driver.
        if td.exp_driver == "mf_real":
            nf = td.n_frames(td.id_names[idx], g_clip.value)
            if nf > 0:
                g_frame.max = nf - 1
                if int(g_frame.value) > g_frame.max:
                    g_frame.value = 0
        elif td.exp_driver == "ict_blendshape":
            g_frame.max = state["exp_coeffs"].shape[0] - 1
            if int(g_frame.value) > g_frame.max:
                g_frame.value = 0

    def _on_src_ds_change(_e=None):
        if g_src_ds.value in topos:
            g_src_id.max = len(topos[g_src_ds.value].id_names) - 1
            if int(g_src_id.value) > g_src_id.max:
                g_src_id.value = 0
        _refresh_clip_options()
        render()

    def _on_src_id_change(_e=None):
        _refresh_clip_options()
        render()

    def _on_mode_change(_e=None):
        _refresh_clip_options()
        render()

    def _on_clip_change(_e=None):
        # User picked a different clip — frame max may change.
        _refresh_clip_options()
        render()

    def _on_tgt_ds_change(_e=None):
        if g_tgt_ds.value in topos:
            g_tgt_id.max = len(topos[g_tgt_ds.value].id_names) - 1
            if int(g_tgt_id.value) > g_tgt_id.max:
                g_tgt_id.value = 0
        render()

    def _on_seq_change(_e=None):
        seq = g_anim_seq.value
        try:
            state["exp_coeffs"] = _load_exp_coeffs(seq)
        except FileNotFoundError as ex:
            g_status.value = f"seq missing: {ex}"
            return
        g_frame.max = state["exp_coeffs"].shape[0] - 1
        if int(g_frame.value) > g_frame.max:
            g_frame.value = 0
        render()

    # Persistent custom lights used by non-hdri modes. Created lazily; toggled
    # via .visible so we don't accumulate handles on every mode switch.
    _light_state = {"ambient": None, "front": None, "axis6": []}

    def _hide_axis6():
        for h in _light_state["axis6"]:
            try: h.visible = False
            except Exception: pass

    def _apply_lighting(_e=None):
        try:
            mode = g_lighting.value
            _entering = (_MAT.get("_last_mode") != mode)
            _MAT["_last_mode"] = mode
            _MAT["light_mode"] = "clay" if mode == "clay (figure)" else "other"
            if mode == "clay v2 (reference)":
                _MAT["light_mode"] = "clayv2"
            if _MAT["light_mode"] != "clay":
                _hide_clay_lights()
            if mode != "clay v2 (reference)" and _light_state.get("hemi") is not None:
                try: _light_state["hemi"].visible = False
                except Exception: pass
            # Always sync env from dropdowns first; modes may override.
            if mode == "hdri":
                hdri = g_env_map.value
                server.scene.configure_environment_map(
                    hdri=None if hdri == "none" else hdri,
                    environment_intensity=float(g_env_intensity.value),
                )
                server.scene.configure_default_lights(enabled=True, cast_shadow=True)
                if _light_state["ambient"] is not None:
                    _light_state["ambient"].visible = False
                if _light_state["front"] is not None:
                    _light_state["front"].visible = False
                _hide_axis6()
            elif mode == "front only":
                # One directional light pointing along +Z (toward face front),
                # default lights & HDRI off, mild ambient so back side not pitch.
                server.scene.configure_environment_map(hdri=None, environment_intensity=0.0)
                server.scene.configure_default_lights(enabled=False, cast_shadow=False)
                _hide_axis6()
                if _light_state["front"] is None:
                    _light_state["front"] = server.scene.add_light_directional(
                        "/lights/front", color=(255, 255, 255), intensity=2.0,
                        cast_shadow=False,
                    )
                    _light_state["front"].position = (0.0, 0.5, 3.0)
                _light_state["front"].visible = True
                _light_state["front"].intensity = 2.0
                if _light_state["ambient"] is None:
                    _light_state["ambient"] = server.scene.add_light_ambient(
                        "/lights/ambient", color=(255, 255, 255), intensity=0.6,
                    )
                _light_state["ambient"].visible = True
                _light_state["ambient"].intensity = 0.6
            elif mode == "matplotlib (mild flat)":
                # Mimic utils/matplotlib_rnd.py shading: shade = 0.7*max(N.L,0)+0.2
                # with L = +Z headlight. One soft frontal directional (no shadow)
                # + ambient floor at the matplotlib 0.2/0.7 ratio -> mild, even,
                # never blown out, back side still readable.
                server.scene.configure_environment_map(hdri=None, environment_intensity=0.0)
                server.scene.configure_default_lights(enabled=False, cast_shadow=False)
                _hide_axis6()
                if _light_state["front"] is None:
                    _light_state["front"] = server.scene.add_light_directional(
                        "/lights/front", color=(255, 255, 255), intensity=2.0,
                        cast_shadow=False,
                    )
                    _light_state["front"].position = (0.0, 0.5, 3.0)
                _light_state["front"].visible = True
                _light_state["front"].intensity = 1.15   # ~0.7 diffuse
                if _light_state["ambient"] is None:
                    _light_state["ambient"] = server.scene.add_light_ambient(
                        "/lights/ambient", color=(255, 255, 255), intensity=0.35,
                    )
                _light_state["ambient"].visible = True
                _light_state["ambient"].intensity = 0.35  # ~0.2 ambient floor
            elif mode == "clay v2 (reference)":
                # Disney-figure reference look: hemisphere dome (soft top-down
                # gradient, no hard edges) + gentle PARALLEL directional key
                # (identical shading on every side-by-side mesh) + low ambient
                # floor. Powder-blue tint + mild sheen. Per-mesh spots off.
                # env HDRI/intensity act as an EXTRA image-based fill here.
                _hdri = g_env_map.value
                server.scene.configure_environment_map(
                    hdri=None if _hdri == "none" else _hdri,
                    environment_intensity=float(g_env_intensity.value))
                server.scene.configure_default_lights(enabled=False, cast_shadow=False)
                _hide_axis6()
                _hide_clay_lights()
                _shadow = bool(_MAT.get("ground_shadow", False))
                if _light_state.get("hemi") is None:
                    _light_state["hemi"] = server.scene.add_light_hemisphere(
                        "/lights/hemi", sky_color=(255, 255, 255),
                        ground_color=(178, 184, 200), intensity=0.9,
                    )
                _light_state["hemi"].visible = True
                try: _light_state["hemi"].intensity = float(_MAT.get("clay_ambient", 0.3))
                except Exception: pass
                # directional key: recreate when shadow flag changes
                if _light_state.get("front_shadow_flag") != _shadow and _light_state.get("front") is not None:
                    try: _light_state["front"].remove()
                    except Exception: pass
                    _light_state["front"] = None
                if _light_state.get("front") is None:
                    _light_state["front"] = server.scene.add_light_directional(
                        "/lights/front", color=(255, 250, 244),
                        intensity=float(_MAT.get("clay_key_int", 1.3)),
                        cast_shadow=_shadow,
                    )
                    _light_state["front_shadow_flag"] = _shadow
                _light_state["front"].position = (
                    float(_MAT.get("clay_dx", 0.0)) * 2.4,
                    float(_MAT.get("clay_dy", 0.7)) * 2.4,
                    float(_MAT.get("clay_dz", 0.9)) * 2.4,
                )
                _light_state["front"].visible = True
                try: _light_state["front"].intensity = float(_MAT.get("clay_key_int", 1.3))
                except Exception: pass
                if _light_state["ambient"] is None:
                    _light_state["ambient"] = server.scene.add_light_ambient(
                        "/lights/ambient", color=(240, 244, 255), intensity=0.25,
                    )
                _light_state["ambient"].visible = True
                _light_state["ambient"].intensity = 0.25
                if _entering:
                    try:
                        g_mat_rough.value = 0.4
                        g_clay_key.value = 1.3
                        _clay2 = (89, 128, 212)
                        g_mesh_color.value = _clay2
                        g_src_color.value = _clay2
                        g_tgt_color.value = _clay2
                    except Exception:
                        pass
            elif mode == "clay (figure)":
                # Disney-figure clay look: soft key from upper-front-left +
                # strong ambient fill (shadows lifted, never black), zero
                # specular (meshes are already roughness=1/metallic=0), and a
                # matte slate-blue tint on all mesh color pickers.
                server.scene.configure_environment_map(hdri=None, environment_intensity=0.0)
                server.scene.configure_default_lights(enabled=False, cast_shadow=False)
                _hide_axis6()
                # key light is PER-MESH point lights (see _sync_clay_light);
                # the global directional stays off in clay mode.
                if _light_state["front"] is not None:
                    _light_state["front"].visible = False
                if _light_state["ambient"] is None:
                    _light_state["ambient"] = server.scene.add_light_ambient(
                        "/lights/ambient", color=(235, 240, 255), intensity=0.5,
                    )
                _light_state["ambient"].visible = True
                _light_state["ambient"].intensity = float(_MAT.get("clay_ambient", 0.5))
                if _entering:
                    try:
                        g_mat_rough.value = 0.42
                        _clay = (122, 138, 170)
                        g_mesh_color.value = _clay
                        g_src_color.value = _clay
                        g_tgt_color.value = _clay
                    except Exception:
                        pass
            elif mode == "6-axis studio":
                # Soft surround: 6 directional lights (±X, ±Y, ±Z), each lower
                # intensity so combined ≈ ambient but with shape cues from each
                # axis. No cast_shadow → no hard shadows.
                server.scene.configure_environment_map(hdri=None, environment_intensity=0.0)
                server.scene.configure_default_lights(enabled=False, cast_shadow=False)
                if _light_state["front"] is not None:
                    _light_state["front"].visible = False
                # tiny offset for "+Y" / "-Y" so look_at-origin direction is unambiguous.
                dirs = [(0, 0.5, 3, "front"), (0, 0.5, -3, "back"),
                        (3, 0.5, 0, "right"), (-3, 0.5, 0, "left"),
                        (0, 3, 0.01, "top"), (0, -3, 0.01, "bottom")]
                if not _light_state["axis6"]:
                    for x, y, z, nm in dirs:
                        h = server.scene.add_light_directional(
                            f"/lights/axis6/{nm}", color=(255, 255, 255),
                            intensity=0.65, cast_shadow=False,
                        )
                        h.position = (float(x), float(y), float(z))
                        _light_state["axis6"].append(h)
                else:
                    for h in _light_state["axis6"]:
                        h.visible = True
                # very mild ambient fills concave creases that all 6 lights miss.
                if _light_state["ambient"] is None:
                    _light_state["ambient"] = server.scene.add_light_ambient(
                        "/lights/ambient", color=(255, 255, 255), intensity=0.3,
                    )
                _light_state["ambient"].visible = True
                _light_state["ambient"].intensity = 0.3
            else:  # flat (no shadows)
                # Truly unlit-looking: kill HDRI, kill default lights (these
                # cast shadows!), keep only strong ambient → mesh = vertex
                # color regardless of normal.
                server.scene.configure_environment_map(hdri=None, environment_intensity=0.0)
                server.scene.configure_default_lights(enabled=False, cast_shadow=False)
                _hide_axis6()
                if _light_state["ambient"] is None:
                    _light_state["ambient"] = server.scene.add_light_ambient(
                        "/lights/ambient", color=(255, 255, 255), intensity=3.0,
                    )
                _light_state["ambient"].visible = True
                _light_state["ambient"].intensity = 3.0
                if _light_state["front"] is not None:
                    _light_state["front"].visible = False
        except Exception:
            import traceback as _tb
            print("[lighting] EXC:", _tb.format_exc())

    def _apply_view(_e=None):
        """Snap all current clients to a preset view + optional fake-ortho FOV."""
        name = g_view.value
        ortho = bool(g_ortho.value)
        if name == "free" and not ortho:
            for c in server.get_clients().values():
                try: c.camera.fov = math.radians(60.0)
                except Exception: pass
            return
        # ICT face is roughly centered at y≈0.05, z≈0; extent ~0.5.
        center = np.array([0.0, 0.05, 0.0])
        # Distance: large when ortho (small FOV needs long throw) else moderate.
        dist = 8.0 if ortho else 1.8
        fov = math.radians(8.0) if ortho else math.radians(60.0)
        presets = {
            "free":   None,
            "front":  (center + np.array([0,   0,   dist]),  (0,  1,  0)),
            "back":   (center + np.array([0,   0,  -dist]),  (0,  1,  0)),
            "left":   (center + np.array([-dist, 0, 0]),     (0,  1,  0)),
            "right":  (center + np.array([dist,  0, 0]),     (0,  1,  0)),
            "top":    (center + np.array([0,   dist, 0.001]),(0,  0, -1)),
            "bottom": (center + np.array([0,  -dist, 0.001]),(0,  0,  1)),
        }
        p = presets.get(name)
        for c in server.get_clients().values():
            try:
                c.camera.fov = fov
                if p is not None:
                    pos, up = p
                    c.camera.position = tuple(pos)
                    c.camera.look_at = tuple(center)
                    c.camera.up_direction = tuple(up)
            except Exception:
                pass

    def _on_reload(_e=None):
        import time as _t
        t0 = _t.time()
        try:
            tag = g_epoch.value
            if tag == "best":
                pth = ckpt_dir / "model_hlbs_best.pth"
            else:
                pth = ckpt_dir / f"model_hlbs_{tag}.pth"
            if not pth.exists():
                g_ckpt_info.value = f"reload: {pth.name} not found"
                return
            sd = torch.load(pth, map_location=device, weights_only=False)
            msg = model.load_state_dict(sd, strict=False)
            cache._cache.clear()
            _seq_stats_cache.clear()
            g_ckpt_info.value = (
                f"{pth.name} | reloaded in {_t.time()-t0:.1f}s "
                f"(missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)})"
            )
            render()
        except Exception:
            import traceback as _tb
            print("[reload] EXC:", _tb.format_exc())
            g_status.value = "reload FAILED — see console"

    def _on_refresh_epochs(_e=None):
        tags = _list_epoch_tags(ckpt_dir)
        g_epoch.options = tuple(tags)
        if g_epoch.value not in tags:
            g_epoch.value = tags[0]
        g_ckpt_info.value = f"epoch list refreshed: {len(tags)} ckpts available"

    def _step_frame(delta):
        nxt = int(g_frame.value) + delta
        nxt = max(0, min(g_frame.max, nxt))
        g_frame.value = nxt

    g_prev_frame.on_click(lambda _e: _step_frame(-1))
    g_next_frame.on_click(lambda _e: _step_frame(+1))

    def _sanitize_path_part(s: str) -> str:
        return s.replace("/", "_").replace("\\", "_").replace(" ", "_")

    def _clip_label_for(td) -> str:
        """Return a path-safe label for the current animation source on td."""
        if td is None:
            return "clip0"
        if td.exp_driver in ("ict_blendshape", "ict_real_blend"):
            return _sanitize_path_part(g_anim_seq.value)
        if td.exp_driver == "mf_real":
            v = g_clip.value
            if not v or v == "<none>":
                return "clip0"
            return _sanitize_path_part(v)
        if td.exp_driver == "pca_mode":
            return f"pca_mode{int(g_pca_mode.value)}"
        return "clip0"

    def _ckpt_prefix() -> str:
        """{ckpt_dir_name}/e{epoch_tag} — namespaces every render under the
        currently-loaded checkpoint AND epoch, so switching epochs doesn't
        overwrite previous renders."""
        return f"{ckpt_dir.name}/e{g_epoch.value}"

    def _compute_render_subpath() -> str:
        """Auto subpath under g_render_dir base. Encodes ckpt + epoch + mode
        + src/tgt + clip so every render lives in its own dir.
        For baseline methods (nfr / nfs) a '/{method}' subfolder is appended so
        baseline renders never overwrite the hlbs (ours) outputs."""
        pfx = _ckpt_prefix()
        # baseline nickname as deepest subfolder; empty for "hlbs (ours)"
        _method = g_compare_method.value
        mtag = "" if _method.startswith("hlbs") else f"/{_method}"
        mode = g_mode.value
        if mode == "anim":
            td = cache.active
            ds = td.name if td else "unknown"
            return f"{pfx}/anim_self/{ds}_id{int(g_id.value)}/{_clip_label_for(td)}{mtag}"
        if mode == "cross":
            src_ds = g_src_ds.value; src_id = int(g_src_id.value)
            tgt_ds = g_tgt_ds.value; tgt_id = int(g_tgt_id.value)
            kind = "cross_self" if (src_ds == tgt_ds and src_id == tgt_id) else "cross_cross"
            src_td = topos.get(src_ds)
            return (f"{pfx}/{kind}/{src_ds}_id{src_id}_to_{tgt_ds}_id{tgt_id}/"
                    f"{_clip_label_for(src_td)}{mtag}")
        # bind_pose / weight — single-frame, group by mode + ds + id
        td = cache.active
        ds = td.name if td else "unknown"
        return f"{pfx}/{mode}/{ds}_id{int(g_id.value)}{mtag}"

    def _refresh_render_subpath(_e=None):
        try:
            g_render_subpath.value = _compute_render_subpath()
        except Exception:
            pass

    def _on_render_video(_e=None):
        """Sweep all frames at current camera/lighting/mode, dump PNGs, ffmpeg."""
        import threading as _th
        def _worker():
            import time as _t, subprocess as _sp
            from imageio.v3 import imwrite as _imwrite
            clients = list(server.get_clients().values())
            if not clients:
                g_render_progress.value = "no client connected — open the page first"
                return
            client = clients[0]
            _hbrowser = None; _hpw = None
            if bool(g_render_headless.value):
                try:
                    from playwright.sync_api import sync_playwright
                    g_render_progress.value = "starting headless capture client..."
                    _pre = set(server.get_clients().keys())
                    _hpw = sync_playwright().start()
                    _hbrowser = _hpw.chromium.launch(headless=True, args=[
                        "--no-sandbox", "--use-angle=swiftshader",
                        "--enable-unsafe-swiftshader", "--disable-dev-shm-usage"])
                    _hpage = _hbrowser.new_page(
                        viewport={"width": int(g_render_w.value), "height": int(g_render_h.value)})
                    _hpage.goto(f"http://localhost:{args.port}",
                                wait_until="load", timeout=30000)
                    _hcli = None
                    for _ in range(120):
                        _new = set(server.get_clients().keys()) - _pre
                        if _new:
                            _hcli = server.get_clients()[sorted(_new)[0]]
                            break
                        _t.sleep(0.25)
                    if _hcli is not None:
                        try:   # clone the user's camera pose onto the capture client
                            _sc = client.camera
                            _hcli.camera.position = tuple(_sc.position)
                            _hcli.camera.look_at = tuple(_sc.look_at)
                            _hcli.camera.fov = float(_sc.fov)
                            _hcli.camera.up_direction = tuple(_sc.up_direction)
                        except Exception:
                            pass
                        _t.sleep(1.0)
                        client = _hcli
                        g_render_progress.value = "headless capture client ready"
                    else:
                        g_render_progress.value = "headless client timeout — using your browser"
                except Exception as _hex:
                    g_render_progress.value = f"headless unavailable ({_hex}) — using your browser"
            out_dir = Path(g_render_dir.value).expanduser() / _compute_render_subpath()
            out_dir.mkdir(parents=True, exist_ok=True)
            g_render_subpath.value = _compute_render_subpath()
            nf = int(g_frame.max) + 1
            H = int(g_render_h.value); W = int(g_render_w.value)
            orig = int(g_frame.value)
            saved = 0
            t0 = _t.time()
            vid_path = out_dir / g_render_name.value
            fps_int = int(g_render_fps.value)
            encode_ok = False
            _writer = None
            try:
                import imageio
                _writer = imageio.get_writer(
                    str(vid_path), fps=fps_int, codec="libx264",
                    macro_block_size=1, quality=8,
                )
            except Exception:
                _writer = None   # fall back to PNG+ffmpeg path below
            _keep = bool(g_render_keep_pngs.value)
            for f in range(nf):
                if g_mode.value in ("anim", "cross"):
                    g_frame.value = f
                    render()
                _t.sleep(0.02)
                try:
                    # JPEG transport: 2-3x faster encode+transfer than PNG,
                    # visually lossless after H.264 anyway.
                    img = client.get_render(
                        height=H, width=W, transport_format="jpeg",
                    )
                    if _writer is not None:
                        _writer.append_data(img)     # stream straight to video
                    if _keep or _writer is None:
                        _imwrite(out_dir / f"frame_{f:04d}.png", img)
                    saved += 1
                except Exception as ex:
                    g_render_progress.value = f"frame {f} failed: {ex}"
                    break
                if f % 5 == 0:
                    _fps_now = (f + 1) / max(_t.time() - t0, 1e-6)
                    g_render_progress.value = (
                        f"rendering {f+1}/{nf} ({(f+1)/nf*100:.0f}%) "
                        f"elapsed {_t.time()-t0:.1f}s ({_fps_now:.1f} fps)"
                    )
            if _writer is not None:
                try:
                    _writer.close()
                    encode_ok = saved > 0
                    g_render_progress.value = (
                        f"DONE  {saved} frames → {vid_path}  ({_t.time()-t0:.1f}s)"
                    )
                except Exception:
                    encode_ok = False
            ex_io = "streamed-writer unavailable"
            if not encode_ok:
              try:
                import imageio
                with imageio.get_writer(
                    str(vid_path), fps=fps_int, codec="libx264",
                    macro_block_size=1, quality=8,
                ) as w:
                    for f in range(saved):
                        w.append_data(imageio.imread(out_dir / f"frame_{f:04d}.png"))
                encode_ok = True
                g_render_progress.value = (
                    f"DONE  {saved} frames → {vid_path}  ({_t.time()-t0:.1f}s)"
                )
              except Exception as ex_io:
                try:
                    cmd = ["ffmpeg", "-y", "-framerate", str(fps_int),
                           "-i", str(out_dir / "frame_%04d.png"),
                           "-c:v", "libx264", "-pix_fmt", "yuv420p", str(vid_path)]
                    _sp.run(cmd, check=True, capture_output=True)
                    encode_ok = True
                    g_render_progress.value = (
                        f"DONE (system ffmpeg) {saved} frames → {vid_path}"
                    )
                except Exception as ex_ff:
                    g_render_progress.value = (
                        f"video encode failed (PNGs saved at {out_dir}): "
                        f"imageio: {ex_io} / ffmpeg: {ex_ff}"
                    )
            # PNG cleanup — default delete frames if video encoded OK and user
            # didn't ask to keep them. Encoding failure → always keep PNGs.
            if encode_ok and not bool(g_render_keep_pngs.value):
                deleted = 0
                for p in sorted(out_dir.glob("frame_*.png")):
                    try: p.unlink(); deleted += 1
                    except Exception: pass
                g_render_progress.value = (
                    g_render_progress.value + f"  (cleaned {deleted} PNGs)"
                )
            if _hbrowser is not None:
                try: _hbrowser.close()
                except Exception: pass
            if _hpw is not None:
                try: _hpw.stop()
                except Exception: pass
            # Restore original frame (always — even on encode failure)
            if g_mode.value in ("anim", "cross"):
                g_frame.value = orig
                render()
        _th.Thread(target=_worker, daemon=True).start()

    g_render_btn.on_click(_on_render_video)

    # ── Sequence stats: aggregate self-retarget metrics over current clip/seq.
    # Mirrors eval_hlbs.py.evaluate_self() — MSE / L2 (mean/med/p95/p99/max/
    # max-mean) / normal-cos / Laplacian-err — for the (cache.active, g_id,
    # current sequence) tuple. Runs in a background thread; clip changes mid-
    # run cancel the worker.
    _seq_stats_cancel = threading.Event()
    _seq_stats_thread: list = [None]
    _seq_stats_cache: dict = {}   # (ds_name, id_idx, driver, seq_label) → markdown

    def _seq_stats_key():
        td = cache.active
        if td is None or not td.supports_anim:
            return None
        id_idx = int(g_id.value)
        if td.exp_driver in ("ict_blendshape", "ict_real_blend"):
            return (td.name, id_idx, "ict", g_anim_seq.value)
        if td.exp_driver == "mf_real":
            return (td.name, id_idx, "real", g_clip.value)
        if td.exp_driver == "pca_mode":
            return (td.name, id_idx, "pca", int(g_pca_mode.value))
        return None

    def _seq_stats_worker(force: bool = False):
        try:
            td = cache.active
            if not td.supports_anim:
                g_seq_stats_md.content = f"**seq stats**: `{td.name}` has no anim driver"
                return
            id_idx = int(g_id.value)
            nf_full = int(g_frame.max) + 1
            if nf_full <= 0:
                g_seq_stats_md.content = "**seq stats**: empty sequence"
                return
            key = _seq_stats_key()
            if (not force) and key is not None and key in _seq_stats_cache:
                g_seq_stats_md.content = (
                    _seq_stats_cache[key] + "\n\n_(cached — click Force recompute to refresh)_"
                )
                return
            all_l2_mm, mses, lap_errs, norm_coss, l2_max_per_frame = [], [], [], [], []
            t0 = time.time()
            with torch.no_grad():
                for f in range(nf_full):
                    if _seq_stats_cancel.is_set():
                        g_seq_stats_md.content = "**seq stats**: cancelled"
                        return
                    exp, _ = _build_exp(td, id_idx, f)
                    if exp is None:
                        continue
                    out = cache.forward_frame(id_idx, exp)
                    if out["pred_v"] is None or out["gt_v"] is None:
                        continue
                    pred = out["pred_v"]; gt = out["gt_v"]
                    l2 = np.linalg.norm(pred - gt, axis=-1) * 1000.0  # [V] in mm
                    all_l2_mm.append(l2)
                    l2_max_per_frame.append(float(l2.max()))
                    m = _compute_metrics(pred, gt, td.faces)
                    mses.append(m["mse"])
                    lap_errs.append(m["lap_err"])
                    norm_coss.append(m["norm_cos"])
                    if (f + 1) % 5 == 0:
                        run_mean = float(np.mean([a.mean() for a in all_l2_mm]))
                        g_seq_stats_md.content = (
                            f"**seq stats** (computing {f+1}/{nf_full}) "
                            f"mean L2 so far ≈ {run_mean:.2f} mm"
                        )
            if not all_l2_mm:
                g_seq_stats_md.content = (
                    f"**seq stats** `{td.name}[{id_idx}]`: no usable frames "
                    "(model_ran=False — nfs cache?)"
                )
                return
            pv = np.concatenate(all_l2_mm)
            elapsed = time.time() - t0
            # Seq label — what clip / seq is currently driving frames?
            if td.exp_driver == "ict_blendshape":
                seq_lbl = g_anim_seq.value
            elif td.exp_driver in ("mf_real", "biwi_real", "coma_real"):
                seq_lbl = g_clip.value
            elif td.exp_driver == "pca_mode":
                seq_lbl = f"PCA mode {int(g_pca_mode.value)}"
            else:
                seq_lbl = "?"
            md = (
                f"**ds**=`{td.name}[{id_idx}]`  **seq**=`{seq_lbl}`  "
                f"**frames**=`{len(mses)}/{nf_full}`  "
                f"**elapsed**=`{elapsed:.1f}s`\n\n"
                f"| metric | value |\n|---|---|\n"
                f"| MSE | {np.mean(mses):.6f} |\n"
                f"| L2 mean (mm) | {pv.mean():.3f} |\n"
                f"| L2 median (mm) | {np.median(pv):.3f} |\n"
                f"| L2 p95 (mm) | {np.percentile(pv, 95):.3f} |\n"
                f"| L2 p99 (mm) | {np.percentile(pv, 99):.3f} |\n"
                f"| L2 max (mm) | {pv.max():.3f} |\n"
                f"| L2 max-mean (mm) | {np.mean(l2_max_per_frame):.3f} |\n"
                f"| normal cos | {np.mean(norm_coss):.4f} |\n"
                f"| Laplacian err | {np.mean(lap_errs):.4f} |\n"
            )
            g_seq_stats_md.content = md
            if key is not None:
                _seq_stats_cache[key] = md
        except Exception:
            import traceback as _tb
            g_seq_stats_md.content = "**seq stats**: FAILED — see console"
            print("[seq stats] EXC:", _tb.format_exc())

    def _on_compute_seq_stats(_e=None, force: bool = False):
        # Cancel any in-flight worker, then launch a fresh one.
        _seq_stats_cancel.set()
        old = _seq_stats_thread[0]
        if old is not None and old.is_alive():
            old.join(timeout=0.2)
        _seq_stats_cancel.clear()
        new = threading.Thread(
            target=_seq_stats_worker, args=(force,), daemon=True,
        )
        _seq_stats_thread[0] = new
        new.start()

    g_seq_stats_btn.on_click(lambda _e: _on_compute_seq_stats(force=False))
    g_seq_stats_force.on_click(lambda _e: _on_compute_seq_stats(force=True))

    def _on_preview(_e=None):
        """Capture one frame at the chosen W/H, save to disk AND show in GUI
        (image widget appears at the bottom of the Render folder)."""
        try:
            clients = list(server.get_clients().values())
            if not clients:
                g_render_progress.value = "preview: no client connected"
                return
            client = clients[0]
            H = int(g_render_h.value); W = int(g_render_w.value)
            img = client.get_render(height=H, width=W, transport_format="png")
            out_dir = Path(g_render_dir.value).expanduser() / _compute_render_subpath()
            out_dir.mkdir(parents=True, exist_ok=True)
            g_render_subpath.value = _compute_render_subpath()
            preview_path = out_dir / "preview.png"
            from imageio.v3 import imwrite as _imwrite
            _imwrite(preview_path, img)
            _preview_state["handle"].image = img
            _preview_state["handle"].label = f"preview {W}x{H} (saved → {preview_path})"
            g_render_progress.value = f"preview saved → {preview_path}  ({W}x{H})"
        except Exception:
            import traceback as _tb
            g_render_progress.value = "preview failed — see console"
            print("[preview] EXC:", _tb.format_exc())

    g_preview_btn.on_click(_on_preview)

    g_reload.on_click(_on_reload)
    g_refresh_epochs.on_click(_on_refresh_epochs)
    g_epoch.on_update(_on_reload)   # auto-reload on epoch change
    g_view.on_update(_apply_view)
    g_ortho.on_update(_apply_view)
    g_shading.on_update(lambda _e: render())
    g_double_sided.on_update(lambda _e: render())
    g_mesh_color.on_update(lambda _e: render())
    g_src_color.on_update(lambda _e: render())
    g_tgt_color.on_update(lambda _e: render())
    g_global_sat.on_update(lambda _e: render())
    g_nrm_vis.on_update(lambda _e: render())
    g_cross_nrm.on_update(lambda _e: render())
    g_feat_mode.on_update(lambda _e: render())
    g_feat_shared.on_update(lambda _e: render())
    g_anchor.on_update(lambda _e: render())
    g_feat_tgt_id.on_update(lambda _e: render())
    def _on_feat_tgt_ds(_e=None):
        _td = topos.get(g_feat_tgt_ds.value)
        if _td is not None:
            g_feat_tgt_id.max = max(len(_td.id_names) - 1, 1)
            g_feat_tgt_id.value = min(int(g_feat_tgt_id.value), g_feat_tgt_id.max)
        render()
    g_feat_tgt_ds.on_update(_on_feat_tgt_ds)
    g_skel_style.on_update(lambda _e: render())
    def _vis_upd(_e=None):
        _VIS_FLAGS["src"] = bool(g_show_src.value)
        _VIS_FLAGS["tgt"] = bool(g_show_tgt.value)
        _VIS_FLAGS["neu"] = bool(g_show_neu.value)
        _MAT["op_src"] = float(g_op_src.value)
        _MAT["op_pred"] = float(g_op_pred.value)
        render()
    g_show_src.on_update(_vis_upd)
    g_show_jlabels.on_update(lambda _e: render())
    g_fancy_joint.on_update(lambda _e: render())
    g_fancy_bone.on_update(lambda _e: render())
    def _rough_upd(_e=None):
        _MAT["roughness"] = float(g_mat_rough.value)
        render()
    g_mat_rough.on_update(_rough_upd)
    def _trim_upd(_e=None):
        _MAT["trim_rings"] = int(g_trim_rings.value)
        render()
    g_trim_rings.on_update(_trim_upd)
    def _errmax_upd(_e=None):
        _MAT["err_absmax"] = float(g_err_absmax.value)
        render()
    g_err_absmax.on_update(_errmax_upd)
    def _errmode_upd(_e=None):
        _MAT["err_mode"] = str(g_err_scale_mode.value)
        _MAT["err_face_rings"] = int(g_err_face_rings.value)
        render()
    g_err_scale_mode.on_update(_errmode_upd)
    def _errface_upd(_e=None):
        _MAT["err_face_only"] = bool(g_err_face_only.value)
        _ERR_STATS.clear()
        render()
    g_err_face_only.on_update(_errface_upd)
    g_err_face_rings.on_update(_errmode_upd)
    def _err_reset(_e=None):
        _ERR_STATS.clear()
        render()
    g_err_reset.on_click(_err_reset)
    g_show_tgt.on_update(_vis_upd)
    g_show_neu.on_update(_vis_upd)
    g_op_src.on_update(_vis_upd)
    g_op_pred.on_update(_vis_upd)
    def _clay_key_upd(_e=None):
        _MAT["clay_key_int"] = float(g_clay_key.value)
        if _MAT.get("light_mode") == "clayv2" and _light_state.get("front") is not None:
            try:
                _light_state["front"].intensity = float(g_clay_key.value)
                _light_state["front"].position = (
                    float(g_clay_dx.value) * 2.4,
                    float(g_clay_dy.value) * 2.4,
                    float(g_clay_dz.value) * 2.4,
                )
            except Exception: pass
        _MAT["clay_dx"] = float(g_clay_dx.value)
        _MAT["clay_dy"] = float(g_clay_dy.value)
        _MAT["clay_dz"] = float(g_clay_dz.value)
        _MAT["clay_angle"] = float(g_clay_angle.value)
        _MAT["clay_penumbra"] = float(g_clay_penumbra.value)
        _MAT["clay_distance"] = float(g_clay_distance.value)
        render()   # lights are recreated/repositioned during render
    g_clay_key.on_update(_clay_key_upd)
    g_clay_dx.on_update(_clay_key_upd)
    g_clay_dy.on_update(_clay_key_upd)
    g_clay_dz.on_update(_clay_key_upd)
    g_clay_angle.on_update(_clay_key_upd)
    def _clay_amb_upd(_e=None):
        _MAT["clay_ambient"] = float(g_clay_ambient.value)
        if _MAT.get("light_mode") == "clayv2":
            if _light_state.get("hemi") is not None:
                try: _light_state["hemi"].intensity = float(g_clay_ambient.value)
                except Exception: pass
        elif _light_state.get("ambient") is not None:
            try: _light_state["ambient"].intensity = float(g_clay_ambient.value)
            except Exception: pass
    g_clay_ambient.on_update(_clay_amb_upd)
    def _ground_upd(_e=None):
        _MAT["ground_shadow"] = bool(g_ground_shadow.value)
        _MAT["ground_y"] = float(g_ground_y.value)
        _sync_ground(server)
        _apply_lighting()   # v2 directional recreated with/without cast_shadow
        render()            # clay spots likewise
    g_ground_shadow.on_update(_ground_upd)
    g_ground_y.on_update(_ground_upd)
    g_clay_penumbra.on_update(_clay_key_upd)
    g_clay_distance.on_update(_clay_key_upd)
    g_w_show_joints.on_update(lambda _e: render())
    g_w_mark_joint.on_update(lambda _e: render())
    g_env_intensity.on_update(_apply_lighting)
    g_env_map.on_update(_apply_lighting)
    g_lighting.on_update(_apply_lighting)
    g_dataset.on_update(_on_dataset_change)
    g_anim_seq.on_update(_on_seq_change)
    g_src_ds.on_update(_on_src_ds_change)
    g_tgt_ds.on_update(_on_tgt_ds_change)

    def _on_tta_change(_e):
        sel = g_tta.value
        _tta["active"] = None if sel == "off" else sel
        d = _tta_data.get(sel)
        if d is not None and d["ds"] in topos:
            g_tgt_ds.value = d["ds"]
            g_tgt_id.max = len(topos[d["ds"]].id_names) - 1
            try:
                g_tgt_id.value = topos[d["ds"]].id_names.index(d["id"])
            except ValueError:
                pass
        render()
    g_tta.on_update(_on_tta_change)

    # bind primary controls — render on any change.
    g_mode.on_update(_on_mode_change)
    g_id.on_update(_on_id_change)
    g_src_id.on_update(_on_src_id_change)
    g_clip.on_update(_on_clip_change)

    def _on_method_change(_e=None):
        method = g_compare_method.value
        if method != "hlbs (ours)":
            g_status.value = (
                f"method → {method} | first call ~1-3 min on ICT "
                "(Poisson LU), then cached. Rendering..."
            )
        else:
            g_status.value = f"method → hlbs (ours) | rendering..."
        render()
    g_compare_method.on_update(_on_method_change)
    for h in [
        g_show_gt, g_show_pred, g_show_err, g_show_helpers, g_mesh_opacity,
        g_w_mode, g_joint, g_soft_topk, g_soft_sat, g_palette,
        g_frame, g_show_axes, g_show_bones, g_show_gt_anim, g_err_color, g_err_cmap,
        g_jpos_src, g_show_joints, g_bind_bones, g_cross_err, g_cross_joints,
        g_tgt_id, g_pca_mode, g_show_src, g_show_tgt_neu,
    ]:
        h.on_update(lambda _e: render())

    # ── play loop: drives g_frame.value AND calls render directly
    # (don't rely on on_update firing for programmatic value changes).
    def _play_loop():
        while True:
            try:
                if g_playing.value and g_mode.value in ("anim", "cross"):
                    nf = state["exp_coeffs"].shape[0]
                    nxt = (int(g_frame.value) + 1) % nf
                    g_frame.value = nxt   # updates UI slider
                    render()              # explicit render — robust
                    time.sleep(1.0 / max(int(g_fps.value), 1))
                else:
                    time.sleep(0.1)
            except Exception:
                print("[play] EXC:", traceback.format_exc())
                time.sleep(0.5)

    threading.Thread(target=_play_loop, daemon=True).start()

    _refresh_clip_options()
    _apply_lighting()   # set initial env intensity so first paint isn't blown out
    render()
    print(f"[viser] http://localhost:{args.port}")
    print("[viser] (use ssh -L if remote: ssh -L {p}:localhost:{p} <host>)".format(p=args.port))
    while True:
        time.sleep(60)


if __name__ == "__main__":
    main()
