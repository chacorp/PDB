---
name: viser HLBS interactive debugger
description: tools/vis/viser_debug.py — web GUI for bind_pose / weight / anim / cross-retarget visualization with hot-reload, render-to-video, sequence stats, multi-topology real-data support
type: project
---

# viser_debug — interactive HLBS debugger

Single-file web app at [tools/vis/viser_debug.py](../../tools/vis/viser_debug.py).
Built over a multi-day session for inspecting bind_pose_net / skin_weight_net /
joint_transform_net outputs and cross-retarget behavior. ~1900 LOC.

Branch `liy`. Four commits, in order:

| hash | summary |
|---|---|
| `1ebfd05` | initial viser debugger (bind_pose / weight / anim / cross modes, GLB per-vertex color, hot reload, opts auto-load mirroring eval_hlbs.py) |
| `fa11af8` | lighting modes / render-to-video / color pickers / multi-topology real data discovery / FK chain vis fix / view presets |
| `b10a73e` | auto-discover animation data across all candidate basedirs (`/data/sihun`, `/data/inyup`, `/data2/...`, `/pca` variants, nested) + `--data_basedirs` CLI + `HLBS_DATA_BASEDIRS` env var |
| `00b3dc2` | sequence stats panel (eval_hlbs-style metrics aggregated per clip, cached, button-only) |

`requirements.txt`: `viser>=1.0.29` added. `_ensure_viser()` at script top
auto-installs via pip if missing — clone-and-run on any server.

---

## CLI

```bash
python tools/vis/viser_debug.py \
    --ckpt ckpts_hlbs/<ckpt-dir> \
    --port 8080 \
    --device cuda \
    --data_basedirs /extra/path1 /extra/path2     # optional
```

Defaults: `--ckpt ckpts_hlbs/2026-05-14-14-02-03-HLBS-FullPred-ict-jTrans-nrm0.1-Wsm0.01`,
`--port 8080`, `--device cuda`.

`HLBS_DATA_BASEDIRS=/path:/other:/...` env var also adds basedirs.

---

## Architecture

### Top-level

- `_ensure_viser()`: try-import viser → bundled `third_party/viser/src` → pip install
- `_build_model(ckpt_dir, device)`: mirrors `eval_hlbs.py:347-410` pattern.
  Peeks `face_joint_idx` / `helper_joint_idx_buf` from state_dict so model
  __init__ buffer sizes match the saved weights. Auto-picks `model_hlbs_best.pth`
  → highest-epoch `model_hlbs_NNN.pth` fallback.
- `_build_topos(ict, nfs_dir, geo_per_topo)`: builds 5 `TopoData` entries
  (ict_train, ict_val, mf, biwi, coma). Per-topo prefers real data → PCA fallback.

### Core dataclass `TopoData`

```python
@dataclass
class TopoData:
    name: str                  # 'ict_train', 'ict_val', 'mf', 'biwi', 'coma'
    topo: str                  # 'ict', 'mf', 'biwi', 'coma'
    id_names: list[str]        # e.g. ['ict_000', ..., 'ict_110'] for ict_train
    faces: np.ndarray          # [F, 3] uint32, shared across ids per topo
    nfs_dir: str | None        # cache base path (for nfs_feat + bind_pos_landmark)
    geo_dist: torch.Tensor|None
    supports_anim: bool        # True if real clips OR pca OR ict blendshape
    exp_driver: str            # 'ict_blendshape' | 'mf_real' | 'pca_mode' | 'none'
    _verts_provider: callable  # id_name → neutral verts [V, 3] float32
    _ict_id_vecs: np.ndarray|None     # [N, 100] for ICT only (blendshape exp)
    _ict_model: ICT_face_model|None
    _pca_per_id: dict|None     # {id_name: (mean[V,3], comps[K,V,3], std[K])}
    _real_clips: dict|None     # {id_name: {clip_name: [sorted frame paths]}}

    def neutral_verts(id_name) → [V, 3]
    def apply_exp(id_name, exp_coeff) → [V, 3]  # driver-dependent
    def clips_for(id_name) → list[str]          # real-data only
    def n_frames(id_name, clip) → int
```

### `IdentityCache`

Memoizes per-id model output. Key = `(td.name, id_name)`. Each entry:
```python
{
    "neu_v":          [V, 3] float32,      # per-id neutral mesh
    "neu_n":          [V, 3] float32,      # vertex normals (igl)
    "W":              [V, J] float32|None, # skin weights (None if model didn't run)
    "joint_pos_pred": [J, 3] float32|None, # bind_pose_net output
    "gt_bind":        [J, 3] float32|None, # landmark cache (None if missing)
    "nfs":            tensor|None,
    "dist_sq_geo":    tensor|None,
    "model_ran":      bool,                # False = mesh-only fallback
}
```

`model_ran=False` when model_needs_nfs but the id has no cached `*_nfs_feat.npy`.
Renderers check this and show mesh-only with a status note.

Methods:
- `get(id_idx)` → cached dict above
- `forward_frame(id_idx, exp_coeff)` → adds `gt_v`, `pred_v`, `T_world`,
  `local_R`, `joint_pos`, `W` for the animated frame
- `retarget(src_td, src_idx, exp, tgt_td, tgt_idx)` → cross-retarget call
  with self-retarget metrics if `(src_td.name, src_idx) == (tgt_td.name, tgt_idx)`

### Helpers
- `_per_vertex_normal(v, f)`: igl area-weighted vertex normals
- `_add_per_vertex_color_mesh(server, name, verts, faces, rgb_uint8, opacity, shading, double_sided, sat)`:
  trimesh + GLB with PBR `(metallic=0, roughness=1, baseColor=white)`
  matte material, igl normals embedded, alphaMode BLEND when opacity<1, optional
  doubleSided, post-blend HSV saturation boost.
- `_viridis_rgb`, `_err_rgb(vals, name)`, `_hot_rgb`, `_tab20_rgb`,
  `_boost_saturation(rgb, factor)`.
- `_compute_metrics(pred_v, gt_v, faces)` → MSE / mean·med·max L2(mm) / norm_cos /
  lap_err. Uses `igl.per_vertex_normals`, `igl.cotmatrix`. Same definitions as
  `eval_hlbs.py:_compute_*`.

---

## Render modes (g_mode dropdown)

### `bind_pose`
- Per-id NEUTRAL mesh via `add_mesh_simple → _add_per_vertex_color_mesh` (single
  color = `g_mesh_color`). Igl normals embedded — avoids three.js auto-normal
  artifacts on ICT's quad-triangulated mesh.
- Pred joints (orange) at `joint_pos_pred` (bind_pose_net output, in per-id
  mesh frame).
- GT joints (blue) at `gt_bind` (from `nfs_features_seg/{id}_bind_pos_landmark.npy`).
- Error arrows GT→Pred, color = viridis(err_norm). Status: mean/med/max bind
  err in mm.
- Helper joints (per `helper_joint_idx_buf` from ckpt) optionally highlighted
  pink/cyan.

### `weight`
- Per-vertex skin weight visualization. Modes:
  - `single`: viridis heatmap of W[:, j] for joint selected via `g_joint`
  - `argmax`: per-vertex top-1 joint colored via palette
  - `soft`: top-K joint mix (`g_soft_topk` slider, default K=3). Algorithm:
    - `Wm = Wf × top_k_mask`, renormalize to sum 1 over the K
    - `rgb = Wm @ palette[J]`
    - per-channel rescale `(rgb − rgb.min(0)) / (rgb.max(0) − rgb.min(0))`
    - HSV saturation × `g_soft_sat` (default 2.0)
  - `entropy`: viridis of `H(W[v]) / log(J)`
- Palette dropdown (`g_palette`): `nipy_spectral` (default, evenly sampled
  across J=66) / `turbo` / `gist_rainbow` / `gist_ncar` / `rainbow` / `hsv` /
  `tab20`. Adopted from `utils/matplotlib_rnd.py:vis_mesh_all_cage_weights`.

### `anim` — SELF-RETARGETING
Equivalent to `model.forward(neu, deform_in)` where `deform_in = [V_def-V_neu,
N_def, V_neu, N_neu]` (12-ch). Path:
1. `V_neu = td.neutral_verts(id_name)` (per-id neutral)
2. `V_def = td.apply_exp(id_name, exp_coeff)` (per-id deformed)
3. `cache.forward_frame(id_idx, exp_coeff)` → `pred_v`, `T_world`, `W`, ...
4. Vis: pred mesh + GT side-by-side (X-offset), skeleton, axes triads

Joint position source toggle (`g_jpos_src`, **critical**):
- **`T_world (animated)`** (default): `T_world[:, :3, 3]` — animated FK-resolved
  world position. THIS IS WHAT FOLLOWS THE FK CHAIN.
- `bind_pred (per-id)`: `out["joint_pos"]` — bind_pose_net output (static).
  Originally hardcoded — was the bug that made joints appear stuck during
  animation.
- `rig_ref (Maya)`: `rig.bind_pos` — trainer rig bind, no animation.

Bones via `add_line_segments(points[E,2,3], colors[E,2,3])` — note `(E, 2, 3)`
color shape (one per endpoint, not per segment).

Axes triads at each joint from `T_world[:, :3, :3]` → scipy `as_quat()` →
wxyz reorder → `add_frame(wxyz, position)`.

Error coloring (`g_err_color` + `g_err_cmap`, default `YlOrRd`):
- `err_n = err / err.max()`
- `heat = err_cmap(err_n)`
- `rgb = (1 − err_n) · g_mesh_color + err_n · heat`
- → low err = mesh color, high err = bright heat (matches
  `utils/matplotlib_rnd.py:plot_image_array_diff` blend formula)

`g_show_axes` / `g_show_bones` / `g_show_gt_anim` toggles.

### `cross` — CROSS-RETARGETING
`model.retarget(src_neu/def/norm, tgt_neu/norm, tgt_nfs_feat, tgt_dist_sq_geo)`.

GUI: `g_src_ds` / `g_src_id` / `g_tgt_ds` / `g_tgt_id` / `g_pca_mode` /
`g_clip` (per src), `g_show_src` / `g_show_tgt_neu` toggles.

Only `supports_anim=True` topos appear in src dropdown.

Layout: 4 meshes in X-row — `src_neu | src_def | tgt_neu | tgt_pred`. Auto
spacing from mesh widths.

If `(src_td.name, src_idx) == (tgt_td.name, tgt_idx)` → "self-retarget"
detected → compute metrics: MSE / mean·med·max·max-mean L2(mm) / norm_cos /
lap_err displayed in status bar AND tgt_pred colored with hot/YlOrRd err map.

Frame slider drives source expression (ict blendshape index or mf real frame
index or pca sin sweep).

---

## Animation data sources

### `apply_exp(id_name, exp_coeff)` dispatch by `td.exp_driver`:

| driver | exp_coeff format | source |
|---|---|---|
| `ict_blendshape` | `[53]` blendshape coeffs | `ict_face_pt/random_identity_vecs.npy` (train, 111 ids) or `data/ICT_live_100/iden_vecs.npy` (val, 100) + `_cap/*_exp_coeffs.npy` (922/924) |
| `mf_real` | `(clip_name: str, frame_idx: int)` | `.npy` frame files (one per frame) |
| `pca_mode` | `[mode_idx, amp ∈ [-1,1]]` | per-id `_pca.npz` basis (mean/components/std) |

For `mf_real`: loads `np.load(path)`, runs `procrustes_LDM(v, template)` then
`v @ R.T + t` to align to the per-id template (matches dataloader pipeline).

### Real animation discovery (`_discover_real_clips`)

Scans `_real_data_basedirs()` × per-topology subpath patterns:

```
mf:   {base}/multiface_align/{ROM,SEN}/{train,test}/vertices_npy/{id}/{clip}/*.npy
biwi: {base}/BIWI_align_deci/{train,test}/vertices_npy/{id}_{clip}/*.npy
coma: {base}/VOCA-COMA/COMA/{train,test}/{id}/vertices_npy/{clip}/*.npy
```

### Candidate basedirs (`_real_data_basedirs()`)

Priority order (deduped, only existing dirs):
1. `--data_basedirs` CLI args
2. `HLBS_DATA_BASEDIRS` env var (colon-separated)
3. Hardcoded defaults:
   - `/data/sihun`, `/data2/sihun`, `/data/inyup`, `/data2/inyup`
   - `+ /pca` variant of each (dataloader_CBD's `data_toggle=False` path)
   - `/data/inyup/data/sihun`, `/data2/inyup/data/sihun` (nested case seen on
     this server)

Logs at startup:
```
[data] real-clip basedirs (N present): /path1, /path2, ...
[real-clips] mf: 746 clips across 12/13 ids (scanned 4 basedirs)
[real-clips] biwi: 24 clips across 6/14 ids (scanned 4 basedirs)
[real-clips] coma: 24 clips across 2/12 ids (scanned 4 basedirs)
```

### PCA fallback (`_try_load_pca`) — MF only

If no real frames found for MF, looks for `{base}/multiface_align/{SEN,ROM}/
{train,test}/{id}_pca.npz` — `mean_`, `components_`, `explained_variance_`.
Drives synthetic animation: `verts = mean + comps[mode] · (amp · 3 · std[mode])`
with `amp = sin(2π · frame / period)`. Uses same basedir list as real.

### ICT identities

Template meshes synthesized via `ICT_face_model.apply_coeffs(id_coeff, exp)`
(scale=0.1, V=11248, F=22288). `id_coeff[100]` comes from:
- `ict_train`: `ict_face_pt/random_identity_vecs.npy[:111]` (matches
  `train_hlbs.py` ICT learning ids per CLAUDE.md)
- `ict_val`: `data/ICT_live_100/iden_vecs.npy` (100 different ids)

### Non-ICT templates

From bundled pkls in repo: `utils/templates/{mf,biwi,voca}_templates.pkl`.
Each has per-id verts [V, 3] + a shared `'face'` key [F, 3].

| topo | V | F | id count |
|---|---|---|---|
| mf | 5223 | 10278 | 13 |
| biwi | 2560 | 4833 | 14 |
| coma (voca) | 3525 | 6910 | 12 |

---

## Lighting (`g_lighting` dropdown)

Critical discovery: viser has **default directional+shadow lights** separate
from HDRI/ambient. Must turn off via `configure_default_lights(enabled=False,
cast_shadow=False)` to actually kill shadows.

| mode | HDRI | default lights | ambient | other |
|---|---|---|---|---|
| `hdri` (default) | studio + intensity (`g_env_intensity` 0.4 default) | ON shadows | OFF | — |
| `front only` | OFF | OFF | 0.6 (back not pitch) | front directional at (0, 0.5, 3) intensity 2.0 |
| `6-axis studio` | OFF | OFF | 0.3 (concave fill) | 6 directional lights at ±X/±Y/±Z, each intensity 0.65 |
| `flat (no shadows)` | OFF | OFF | 3.0 (full) | none — unlit-like |

HDRI dropdown (`g_env_map`): `none / warehouse / studio / apartment / city /
dawn / forest / lobby / night / park / sunset`. Default `studio` (gentler than
warehouse).

`g_env_intensity` slider 0.0~2.0, default 0.4.

---

## Material setup (`_add_per_vertex_color_mesh`)

```python
mesh.visual.material = PBRMaterial(
    alphaMode="BLEND" if opacity < 1.0 else "OPAQUE",
    baseColorFactor=[1.0, 1.0, 1.0, 1.0],
    metallicFactor=0.0,
    roughnessFactor=1.0,
    doubleSided=g_double_sided.value,
)
mesh.vertex_normals = igl.per_vertex_normals(v, f)  # smooth shading
mesh.vertex_colors = rgba   # alpha = opacity*255
```

**Why matte explicitly**: glTF 2.0 default `metallicFactor=1.0` (per spec).
Without override, three.js renders meshes as metallic → blown out by HDRI →
washed white. Setting metallic=0 + roughness=1 = Lambert-like.

**Why igl normals**: trimesh `process=False` skips normal compute → GLB has no
normals → three.js falls back to per-face normals from quad-triangulation →
visible diagonal contour artifacts on ICT mesh. Pre-computed area-weighted
vertex normals via `igl.per_vertex_normals` eliminate this.

**doubleSided toggle**: ON kills dark fringe at open boundaries (MF neck cut,
ICT eye holes) but may distort normals at silhouette via three.js's
`gl_FrontFacing` flip. Default OFF.

---

## Colors

Default RGB values for single-color mesh tints:
- `g_mesh_color` = `rgb(105, 105, 105)` (dark gray — bind_pose/anim neutral/
  weight fallback/anim pred without err color)
- `g_src_color` = `rgb(209, 159, 130)` (warm beige — cross src_def. src_neu
  auto +25 lighter)
- `g_tgt_color` = `rgb(127, 174, 201)` (light blue — cross tgt_neu, tgt_pred
  when no metrics)

`g_global_sat` slider 0.5~3.0, default 1.5. HSV S boost applied to ALL output
colors via `_boost_saturation`.

`g_mesh_opacity` slider 0.05~1.0, default 1.0. Universal across all modes.

---

## View presets (`g_view`)

`free / front / back / left / right / top / bottom`. Center fixed at
`(0, 0.05, 0)` (approx ICT face center). Distance 1.8m (perspective) or 8m
(ortho fake). FOV 60° normal / 8° ortho-fake.

`g_ortho` checkbox: switches to small-FOV + far-distance (no actual
orthographic camera — viser doesn't expose it).

**Panning**: viser uses three.js OrbitControls default — right-click drag.
Alt+Middle remapping not configurable from Python (would need frontend patch).

---

## Render to video (`Render to video` folder)

Background-thread sweep of all frames at current camera/lighting/mode,
PNG-per-frame to disk, then encode mp4.

GUI:
- `out dir` (text input, default `{repo}/_diag/render`)
- `video filename` (default `out.mp4`)
- `video fps` (slider 1~60, default 30)
- `render width / height` (sliders 256~1920 / 256~1080, default 720/720)
- `Preview` button — capture 1 frame @ chosen W/H, save `preview.png` to out
  dir AND show in GUI image widget (pre-created placeholder inside folder so
  it's always visible)
- `Render video` button — main sweep
- `render progress` (text, live update)

Encoding fallback: `imageio.get_writer(..., codec='libx264')` first (uses
imageio-ffmpeg plugin, auto-installs binary). Falls back to system ffmpeg
subprocess. If both fail, PNGs remain.

In anim/cross modes the sweep advances `g_frame.value` and calls `render()`
per frame. For bind_pose/weight, same view repeated (no semantic frame
change).

---

## Sequence stats (`Sequence stats` folder)

Aggregate `eval_hlbs.py:evaluate_self()` metrics over the current animation
sequence — button-driven only (full ict-cap sweep takes ~5-15s).

GUI:
- `seq stats` markdown widget (results table)
- `Compute (current seq)` button
- `Force recompute (ignore cache)` button

Worker (background thread, cancellable via `_seq_stats_cancel`):
- Iterates frames via `_build_exp(td, id_idx, f)` → `cache.forward_frame`
- Accumulates per-vertex L2 (mm), MSE, normal cos, Laplacian err, frame max L2
- After all frames: concat L2 → compute mean / median / p95 / p99 / max /
  max-mean (frame-max averaged)

Cache: `_seq_stats_cache[(ds_name, id_idx, driver_kind, seq_label)] = markdown
result`. Re-clicking same key = instant cache return. Hot reload of ckpt
clears cache (weights changed → stats stale).

Result markdown table:
```
ds=ict_train[5]  seq=922  frames=183/183  elapsed=12.3s

| metric           | value |
| MSE              | ...   |
| L2 mean (mm)     | ...   |
| L2 median (mm)   | ...   |
| L2 p95 (mm)      | ...   |
| L2 p99 (mm)      | ...   |
| L2 max (mm)      | ...   |
| L2 max-mean (mm) | ...   |
| normal cos       | ...   |
| Laplacian err    | ...   |
```

---

## Hot reload (`Reload ckpt (hot)` button)

Re-reads `model_hlbs_best.pth` (or highest-epoch fallback) from same ckpt dir,
runs `model.load_state_dict(sd, strict=False)`, clears `cache._cache` AND
`_seq_stats_cache`. Takes ~0.2s. Status text shows
`{ckpt_name} | reloaded in 0.2s (missing=X unexpected=Y)`.

For watching live training: click whenever you want fresh weights. No
auto-polling.

---

## GUI structure (top→bottom)

```
Mode dropdown (bind_pose / weight / anim / cross)
Reload ckpt (hot) button
ckpt info (text, current ckpt name)
Dataset dropdown (ict_train / ict_val / mf / biwi / coma)
Anim seq dropdown (922 / 924) [used by ict_blendshape only]
Identity slider (0..N_id-1)
mesh opacity slider (0.05~1.0, default 1.0)
shading dropdown (smooth / flat)
double-sided checkbox
view preset dropdown (free / front / back / left / right / top / bottom)
orthographic checkbox
env light intensity slider (0~2, default 0.4)
env HDRI dropdown (none / studio / apartment / ...)
lighting mode dropdown (hdri / front only / 6-axis studio / flat)
mesh color rgb picker
cross src color rgb picker
cross tgt color rgb picker
global saturation slider (0.5~3, default 1.5)

[Bind pose folder]
  show GT joints / show Pred joints / show error arrows / highlight helpers

[Weight folder]
  weight mode (single / argmax / soft / entropy)
  joint dropdown (single mode)
  soft top-K slider (1~8, default 3)
  soft saturation slider (1~3, default 2.0)
  palette dropdown (nipy_spectral default)

[Anim folder]
  frame slider
  ◀ prev frame / next frame ▶
  play checkbox
  fps slider (1~60, default 15)
  joint axes triads / bones / GT mesh side-by-side / color pred by L2 error
  err cmap dropdown (YlOrRd default)
  joint position source (T_world / bind_pred / rig_ref)

[Cross-retarget folder]
  src dataset / src identity / tgt dataset / tgt identity
  PCA mode idx
  clip dropdown (dynamic per src_ds × src_id)
  show source meshes / show target neutral

[Sequence stats folder]
  seq stats (markdown table)
  Compute (current seq) button
  Force recompute (ignore cache) button

[Render to video folder]
  out dir / video filename / video fps / render width / render height
  Preview button
  Render video button
  render progress (text)
  preview image (always present in folder)

status (text — bottom, live render summary)
```

---

## Critical bug fixes (chronological)

1. **`add_line_segments` colors shape**: requires `(E, 2, 3)` (one per
   endpoint), not `(E, 3)`. Broadcast tile of single color.

2. **`add_mesh_simple` opacity worked, GLB didn't**: switched bind_pose to GLB
   path too (so igl normals fix applies), GLB now uses PBR `alphaMode=BLEND`
   with alpha in vertex_colors[:, 3].

3. **Play loop relying on `g_frame.value` setter triggering `on_update`**:
   not reliable. Explicit `render()` call in play loop + render_lock for
   thread safety.

4. **FK chain not visualized**: was `jp = out["joint_pos"]` (bind_pose_net
   output, static). Fixed to `jp = T_world[:, :3, 3]` — animated joint world
   position. **Crucial — was making joints appear stuck during animation.**

5. **igl `per_vertex_normals` API**: requires `faces.astype(np.int64)` (not
   uint32) and `verts.astype(np.float64)`.

6. **`apply_coeffs` returns `[1, V, 3]`** not `[V, 3]` — must index `[0]`.

7. **Quad-triangulation flat-shading artifacts**: ICT mesh splits each quad
   into `(0,1,2)+(0,2,3)` with uniform diagonal. three.js auto-normal compute
   picks up diagonals as contours. Fix: pre-compute area-weighted vertex
   normals with igl and embed in GLB.

8. **glTF default `metallicFactor=1.0`**: makes mesh act metallic + reflect
   HDRI → washed white. Fix: explicit `PBRMaterial(metallic=0, roughness=1)`.

9. **viser default lights weren't disabled**: `full_lit` mode kept showing
   shadows. Fix: `configure_default_lights(enabled=False)` for non-hdri modes.

10. **Anim mode hardcoded ict_blendshape**: MF/BIWI/COMA exp_coeff format is
    different. Fix: `_build_exp(td, id_idx, frame)` driver dispatch shared by
    anim + cross modes.

11. **Hardcoded `/data/inyup/...` paths**: other servers have data at
    `/data/sihun/...` etc. Fix: `_real_data_basedirs()` scans 8 default
    candidates + CLI + env var.

12. **`gt_bind` requires `td.nfs_dir`**: ckpts with `nfs_feat_dir=null` have
    `td.nfs_dir=None` → gt_bind also None. **NOT YET FIXED** — gt_bind cache
    files exist in `nfs_features_seg/` independent of whether model uses NFS.
    Should decouple gt_bind path from nfs_dir.

---

## Design decisions

- **Single-file**: kept ~1900 LOC in one file for grep-ability. Helper
  functions / TopoData / IdentityCache all in module scope. main() does GUI
  build + handler binding.
- **All in-memory cache**: `cache._cache` and `_seq_stats_cache` are plain
  dicts. No persistence. Reload button clears them.
- **Background threads for slow ops**: render-video, seq-stats. Each
  uses a cancel Event + thread join with timeout 0.2s.
- **Renderer architecture**: `render()` clears all scene nodes via tracked
  `nodes` list, dispatches by `g_mode.value`, locked with `render_lock`. Per-
  mode renderers are `_render_bind_pose / _render_weight / _render_anim /
  _render_cross`.
- **No autosave**: user must click Preview / Render video / Compute. Earlier
  attempt at auto-compute on every dataset/id/seq change was reverted because
  full seq sweep takes 5-15s (would lock UI on every selector twiddle).

---

## Diagnostic findings (CKPT comparison)

### Compared 3 ICT ckpts on bind_pose_net per-id behavior:

| ckpt | per_id_bind_pose_dir | lambda_bind_reg | lambda_bind_residual | bind_pose_base_residual | nfs_feat_dir | nfs_concat |
|---|---|---|---|---|---|---|
| 5/14 nrm0.1-Wsm0.01 | **nfs_features_seg** ✓ | **1.0** ✓ | n/a | False | nfs_features_seg ✓ | True ✓ |
| 5/26 FullPred-jTrans | None | 0.0 | 0.1 | True | None | False |
| 5/27 FullPred-jTrans | None | 0.0 | 0.0 | True | None | False |

### bind_pose_net per-id spread (all 111 train ids):

| ckpt | pred spread mean (mm) | pred std mean (mm) | % of GT variance |
|---|---|---|---|
| **GT reference** | 250.95 | 48.18 | 100% |
| 5/14 (epoch 200) | 49.46 | 11.48 | ~20% spread / ~24% std |
| 5/26 (best) | 0.00 | 0.00 | 0% |
| 5/27 (best) | 0.33 | 0.07 | 0.13% |

### Epoch progression (5/26 + 5/27 both have bind_pose_base_residual=True):

| epoch | 5/26 spread mean (mm) | 5/27 spread mean (mm) |
|---|---|---|
| 100 | 0.11 | 0.36 |
| 200 | 0.02 | 0.29 |
| 300 | 0.01 | 0.28 |
| 350-400 | 0.00 | 0.31 |

### Interpretation

- 5/14 was supervised by per-id GT (`per_id_bind_pose_dir` set,
  `lambda_bind_reg=1.0`) — but per CLAUDE.md, with **wrong identity mapping**
  GT (cache identity mismatch bug from before 5/19 fix). Still: it had a
  bind-pose supervision signal AND NFS feature input (256-d id-conditional
  embedding). Result: bind_pose differentiates ids by ~20% of GT variance.
- 5/26 and 5/27 dropped per-id bind supervision entirely
  (`per_id_bind_pose_dir=None`, `lambda_bind_reg=0`), use only reconstruction
  loss to drive bind, and removed NFS feature input. With `bind_pose_base_
  residual=True` the network only needs to output a residual on the ICT mean
  bind — which collapses to ~0 because:
  - 5/26 has `lambda_bind_residual=0.1` directly pulling residual to 0
  - 5/27 has no penalty but no id signal either (no NFS) → still uniform
- The viser_debug tool correctly displays this — the issue is **training
  setup**, not visualization.
- Implication: if joint movement in anim mode looks bad with 5/26/5/27,
  it's because bind pose collapsed → B_inv_id ≈ B_inv_mean for all ids →
  vertices end up in rig-mean frame regardless of input identity →
  joint_transform_net is the only thing differentiating identities, but it
  also only sees 6-ch pos+norm.

---

## Known limitations / TODOs

1. **gt_bind decoupling**: `gt_bind` cache lives in `nfs_features_seg/`
   regardless of whether model uses NFS features. Currently `_load_gt_bind_
   pos` is gated by `td.nfs_dir`. Should be separate variable so non-NFS
   ckpts (5/26 / 5/27) can also show bind_pose error arrows. ~10 line fix
   in `TopoData` + `IdentityCache.get`.

2. **ict_val no nfs cache**: model_ran=False for ict_val even with the 5/14
   ckpt. Could optionally compute NFS features on-the-fly via NFS model
   (eval_hlbs.py:_extract_seg_feat_online has the code). Not implemented —
   would add NFS model load.

3. **VOCASET data format mismatch**: COMA test loader expects per-frame
   .npy. VOCASET stores [T, V*3] arrays per clip. Currently VOCASET clips
   skipped (not in COMA discovery). Could add separate driver if needed.

4. **Pan via Alt+Middle**: viser frontend doesn't expose OrbitControls mouse
   mapping config. Only right-click pan available. Would need viser fork.

5. **Real ortho camera**: not in viser API. Fake via FOV=8° + far distance.

6. **Render-to-video sweeps once per frame**: no double-buffering; relies on
   ~40ms `time.sleep` to let scene update propagate. Could miss frames if
   transmission is slow. Workaround: increase sleep or render smaller W/H.

---

## Run on another server

```bash
# Clone
git pull origin liy

# Run (viser auto-installs)
python tools/vis/viser_debug.py \
    --ckpt ckpts_hlbs/<ckpt-dir> \
    --port 8080

# Or with extra data paths
python tools/vis/viser_debug.py \
    --ckpt ckpts_hlbs/<ckpt-dir> \
    --data_basedirs /some/extra/path

# Or via env var
export HLBS_DATA_BASEDIRS=/path1:/path2
python tools/vis/viser_debug.py --ckpt ...

# SSH port forward if remote
ssh -L 8080:localhost:8080 <host>
# Then browse http://localhost:8080
```

Startup log to verify data discovery:
```
[data] real-clip basedirs (N present): /...
[real-clips] mf: X clips across Y/13 ids
[real-clips] biwi: X clips across Y/14 ids
[real-clips] coma: X clips across Y/12 ids
[setup] J=66 helpers=N datasets=[...] anim_seqs=['922', '924'] model_needs_nfs=True
```

If 0 clips for a topo: data not on disk at any default path. Add the path
via `--data_basedirs`.

---

## Cross-references

- [eval_hlbs.py](../../eval_hlbs.py:347-410) — model load + opts pattern that
  `_build_model` mirrors.
- [eval_hlbs.py:evaluate_self](../../eval_hlbs.py:680) — metric definitions
  that `_compute_metrics` + sequence-stats reproduce.
- [utils/matplotlib_rnd.py:vis_mesh_all_cage_weights](../../utils/matplotlib_rnd.py:553) —
  weight viz palette + per-channel rescale approach.
- [utils/matplotlib_rnd.py:plot_image_array_diff](../../utils/matplotlib_rnd.py:1330) —
  err viz YlOrRd cmap + alpha-blend formula.
- [models/hierarchical_lbs.py:HierarchicalLBS_FullPred](../../models/hierarchical_lbs.py:493) —
  the model class being inspected.
- [models/hierarchical_lbs.py:_chain_hierarchy](../../models/hierarchical_lbs.py:272) —
  FK chain math — `T_world[j] = T_world[parent] @ T_bind_local[j] @ T_delta[j]`.
- [maya_rig/hybrid/active_joints_manual.json](../../maya_rig/hybrid/active_joints_manual.json) —
  face_joint_idx / helper_joint_idx the renderer overlays.
- [utils/templates/{mf,biwi,voca}_templates.pkl](../../utils/templates/) —
  bundled mesh templates for non-ICT topos.
- See [project_hlbs_dev_plan.md](project_hlbs_dev_plan.md) for the broader
  HLBS development context.
