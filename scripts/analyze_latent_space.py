"""
Analyze NFS/NFR expression latent space:
  Exp 1: t-SNE visualization — mild/moderate/intense clustering
  Exp 2: Expression pair cosine similarity — collapse detection

Usage:
    python scripts/analyze_latent_space.py
"""
import os, sys, json, pickle, yaml
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ============================================================
# ICT blendshape categorization by displacement magnitude
# ============================================================

# Verified order: alphabetical, matched to exp_basis.pt via displacement correlation
BLENDSHAPE_NAMES = [
    'browDown_L','browDown_R','browInnerUp_L','browInnerUp_R',
    'browOuterUp_L','browOuterUp_R',
    'cheekPuff_L','cheekPuff_R','cheekSquint_L','cheekSquint_R',
    'eyeBlink_L','eyeBlink_R',
    'eyeLookDown_L','eyeLookDown_R','eyeLookIn_L','eyeLookIn_R',
    'eyeLookOut_L','eyeLookOut_R','eyeLookUp_L','eyeLookUp_R',
    'eyeSquint_L','eyeSquint_R','eyeWide_L','eyeWide_R',
    'jawForward','jawLeft','jawOpen','jawRight',
    'mouthClose','mouthDimple_L','mouthDimple_R',
    'mouthFrown_L','mouthFrown_R','mouthFunnel',
    'mouthLeft','mouthLowerDown_L','mouthLowerDown_R',
    'mouthPress_L','mouthPress_R','mouthPucker','mouthRight',
    'mouthRollLower','mouthRollUpper',
    'mouthShrugLower','mouthShrugUpper',
    'mouthSmile_L','mouthSmile_R','mouthStretch_L','mouthStretch_R',
    'mouthUpperUp_L','mouthUpperUp_R',
    'noseSneer_L','noseSneer_R',
]


def categorize_blendshapes():
    """Categorize 53 ICT blendshapes into mild/moderate/intense by displacement magnitude."""
    exp_basis = torch.load('ict_face_pt/exp_basis.pt', weights_only=True).numpy()
    v_idx = 11248

    magnitudes = []
    for i in range(53):
        disp = exp_basis[i, :v_idx, :]
        mag = np.sqrt((disp ** 2).sum(axis=-1)).sum()
        magnitudes.append(mag)
    magnitudes = np.array(magnitudes)

    p33 = np.percentile(magnitudes, 33.3)
    p66 = np.percentile(magnitudes, 66.6)

    categories = {}
    for i, (name, mag) in enumerate(zip(BLENDSHAPE_NAMES[:53], magnitudes)):
        if mag < p33:
            categories[i] = 'mild'
        elif mag < p66:
            categories[i] = 'moderate'
        else:
            categories[i] = 'intense'

    return categories, magnitudes


# ============================================================
# NFS Encoder — standalone loading (no full NFS model)
# ============================================================

def load_nfs_encoder(ckpt_path='ckpts_hlbs/NFS-best/model_best.pth', device='cuda'):
    """Load NFS expression encoder standalone."""
    from models.encoder import BaseDiffusionNetEncoder
    from models.CNN import TextureEncoder
    from utils.nfr_utils import get_dfn_info
    import trimesh

    nfs_dir = os.path.dirname(ckpt_path)
    with open(os.path.join(nfs_dir, 'train_opts.yml')) as f:
        nfs_cfg = yaml.safe_load(f)

    nfs_rig_dim = nfs_cfg.get('rig_dim', 128)
    nfs_img_feat_dim = nfs_cfg.get('img_feat_dim', 128)
    in_shape = 6 + nfs_img_feat_dim

    exp_encoder = BaseDiffusionNetEncoder(
        in_shape=in_shape, pre_computes=None, out_shape=nfs_rig_dim,
    ).to(device)
    img_encoder = TextureEncoder().to(device)
    img_fc = torch.nn.Linear(128, nfs_img_feat_dim).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    skip_keys = {'mass', 'L_ind', 'L_val', 'evals', 'evecs', 'grad_X', 'grad_Y', 'faces'}

    def _extract(prefix):
        out = {}
        for k, v in ckpt.items():
            if k.startswith(prefix):
                short = k[len(prefix):]
                if not any(s in short for s in skip_keys):
                    out[short] = v
        return out

    exp_encoder.load_state_dict(_extract('mesh_exp_encoder.'), strict=False)
    img_encoder.load_state_dict(_extract('img_encoder.'), strict=False)
    img_fc.load_state_dict(_extract('img_fc.'), strict=False)

    for m in [exp_encoder, img_encoder, img_fc]:
        for p in m.parameters():
            p.requires_grad_(False)
        m.eval()

    # Precompute DiffusionNet operators on ICT mean face
    from utils.remesh_utils import ICT_face_model
    ict = ICT_face_model(face_only=False)
    verts_np = ict.neutral_verts.astype(np.float32)
    faces_np = ict.faces.astype(np.int32)
    mesh = trimesh.Trimesh(vertices=verts_np, faces=faces_np, process=False)
    dfn_info = get_dfn_info(mesh, map_location=device)

    # Precompute image feature
    img_np = np.load('data/MF_all_v5/m--20180426--0000--002643814--GHS_neutral_img.npy')
    img_t = torch.tensor(img_np, dtype=torch.float32, device=device)
    img_t = img_t.unsqueeze(0).permute(0, 3, 1, 2)
    with torch.no_grad():
        img_feat = img_fc(img_encoder(img_t))  # [1, 128]

    return exp_encoder, dfn_info, img_feat, nfs_rig_dim


@torch.no_grad()
def encode_expression(exp_encoder, dfn_info, img_feat, vertices, normals, device='cuda'):
    """Encode expression vertices to z_exp via NFS encoder."""
    B, V, _ = vertices.shape
    vert_feat = torch.cat([vertices, normals], dim=-1)  # [B, V, 6]
    img_exp = img_feat.expand(B, -1).unsqueeze(1).expand(-1, V, -1)  # [B, V, 128]
    nfs_input = torch.cat([vert_feat, img_exp], dim=-1)  # [B, V, 134]
    exp_encoder.update_precomputes(dfn_info)
    z_exp = exp_encoder(nfs_input)  # [B, 128]
    return z_exp


# ============================================================
# Generate ICT expressions and encode
# ============================================================

def generate_and_encode(exp_encoder, dfn_info, img_feat, device='cuda'):
    """Generate single-AU ICT expressions, encode to z_exp."""
    from utils.remesh_utils import ICT_face_model
    import igl

    ict = ICT_face_model(face_only=False)
    id_coeff = np.zeros(100)  # neutral identity
    categories, magnitudes = categorize_blendshapes()

    z_exps = []
    labels = []
    bs_names = []

    for au_idx in tqdm(range(53), desc='Encoding single-AU expressions'):
        exp_coeff = np.zeros(53)
        exp_coeff[au_idx] = 1.0

        deformed, template, _ = ict.apply_coeffs(id_coeff, exp_coeff, return_all=True)
        deformed = deformed[0]  # [V, 3]
        template = template[0]

        faces = ict.faces
        normals = igl.per_vertex_normals(
            np.asarray(deformed, dtype=np.float64),
            np.asarray(faces, dtype=np.int64)
        ).astype(np.float32)

        verts_t = torch.tensor(deformed, dtype=torch.float32, device=device).unsqueeze(0)
        norms_t = torch.tensor(normals, dtype=torch.float32, device=device).unsqueeze(0)

        z = encode_expression(exp_encoder, dfn_info, img_feat, verts_t, norms_t, device)
        z_exps.append(z.cpu().numpy())
        labels.append(categories[au_idx])
        bs_names.append(BLENDSHAPE_NAMES[au_idx])

    # Also encode neutral
    exp_coeff_neutral = np.zeros(53)
    deformed_n, template_n, _ = ict.apply_coeffs(id_coeff, exp_coeff_neutral, return_all=True)
    deformed_n = deformed_n[0]
    normals_n = igl.per_vertex_normals(
        np.asarray(deformed_n, dtype=np.float64),
        np.asarray(ict.faces, dtype=np.int64)
    ).astype(np.float32)
    verts_nt = torch.tensor(deformed_n, dtype=torch.float32, device=device).unsqueeze(0)
    norms_nt = torch.tensor(normals_n, dtype=torch.float32, device=device).unsqueeze(0)
    z_neutral = encode_expression(exp_encoder, dfn_info, img_feat, verts_nt, norms_nt, device)
    z_exps.append(z_neutral.cpu().numpy())
    labels.append('neutral')
    bs_names.append('neutral')

    z_exps = np.concatenate(z_exps, axis=0)  # [54, 128]
    return z_exps, labels, bs_names, magnitudes


# ============================================================
# Experiment 1: t-SNE visualization
# ============================================================

def exp1_tsne(z_exps, labels, bs_names, save_dir='analysis_output'):
    """t-SNE visualization of expression latent space."""
    os.makedirs(save_dir, exist_ok=True)

    color_map = {'mild': 'blue', 'moderate': 'orange', 'intense': 'red', 'neutral': 'black'}

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=15, random_state=42)
    z_2d = tsne.fit_transform(z_exps)

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    for cat in ['mild', 'moderate', 'intense', 'neutral']:
        mask = [i for i, l in enumerate(labels) if l == cat]
        if not mask:
            continue
        ax.scatter(z_2d[mask, 0], z_2d[mask, 1],
                   c=color_map[cat], label=cat, s=30 if len(labels) > 100 else 80, alpha=0.6)
        # Only annotate if few points (ICT mode)
        if len(labels) <= 60:
            for i in mask:
                ax.annotate(bs_names[i], (z_2d[i, 0], z_2d[i, 1]),
                            fontsize=6, alpha=0.7)

    ax.legend(fontsize=14)
    ax.set_title('NFS Expression Latent Space (t-SNE)\nExpressions colored by intensity', fontsize=14)
    ax.set_xlabel('t-SNE dim 1')
    ax.set_ylabel('t-SNE dim 2')

    save_path = os.path.join(save_dir, 'exp1_tsne_latent_space.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {save_path}')

    # PCA version too
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    z_pca = pca.fit_transform(z_exps)

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    for cat in ['mild', 'moderate', 'intense', 'neutral']:
        mask = [i for i, l in enumerate(labels) if l == cat]
        if not mask:
            continue
        ax.scatter(z_pca[mask, 0], z_pca[mask, 1],
                   c=color_map[cat], label=cat, s=30 if len(labels) > 100 else 80, alpha=0.6)
        if len(labels) <= 60:
            for i in mask:
                ax.annotate(bs_names[i], (z_pca[i, 0], z_pca[i, 1]),
                            fontsize=6, alpha=0.7)

    ax.legend(fontsize=14)
    ax.set_title('NFS Expression Latent Space (PCA)\nExpressions colored by intensity', fontsize=14)
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')

    save_path = os.path.join(save_dir, 'exp1_pca_latent_space.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {save_path}')


# ============================================================
# Experiment 2: Expression pair cosine similarity
# ============================================================

def exp2_cosine_similarity(z_exps, labels, bs_names, save_dir='analysis_output'):
    """Cosine similarity between expression pairs."""
    os.makedirs(save_dir, exist_ok=True)

    z_t = torch.tensor(z_exps, dtype=torch.float32)
    N = z_t.shape[0]

    # Sort by category for cleaner visualization
    cat_order = {'mild': 0, 'moderate': 1, 'intense': 2, 'neutral': 3}
    sorted_idx = sorted(range(N), key=lambda i: (cat_order.get(labels[i], 3), i))
    sorted_labels = [labels[i] for i in sorted_idx]

    z_sorted = z_t[sorted_idx]
    cos_sim = F.cosine_similarity(z_sorted.unsqueeze(0), z_sorted.unsqueeze(1), dim=-1).numpy()

    # Category boundaries
    boundaries = []
    for i in range(1, len(sorted_labels)):
        if sorted_labels[i] != sorted_labels[i-1]:
            boundaries.append(i)

    # Heatmap
    fig, ax = plt.subplots(1, 1, figsize=(10, 9))
    im = ax.imshow(cos_sim, cmap='RdYlBu_r', vmin=cos_sim.min(), vmax=1.0)

    for b in boundaries:
        ax.axhline(y=b-0.5, color='white', linewidth=2)
        ax.axvline(x=b-0.5, color='white', linewidth=2)

    cat_centers = []
    prev = 0
    unique_cats = [sorted_labels[0]]
    for b in boundaries:
        cat_centers.append((prev + b) / 2)
        prev = b
        unique_cats.append(sorted_labels[b])
    cat_centers.append((prev + N) / 2)

    cat_counts = [sorted_labels.count(c) for c in unique_cats]
    cat_labels = [f'{c}\n({n})' for c, n in zip(unique_cats, cat_counts)]

    ax.set_xticks(cat_centers)
    ax.set_xticklabels(cat_labels, fontsize=12)
    ax.set_yticks(cat_centers)
    ax.set_yticklabels(cat_labels, fontsize=12)

    plt.colorbar(im, ax=ax, label='Cosine Similarity')
    ax.set_title(f'NFS Latent Space: Pairwise Cosine Similarity\n(sorted by intensity, N={N})', fontsize=14)

    save_path = os.path.join(save_dir, 'exp2_cosine_sim_matrix.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {save_path}')

    # Category-level analysis
    print('\n=== Cosine Similarity Statistics by Category ===')
    neutral_indices = [i for i, l in enumerate(labels) if l == 'neutral']
    # If no neutral, use mild as reference (lowest magnitude)
    if neutral_indices:
        ref_idx = neutral_indices[0]
        ref_name = 'neutral'
    else:
        mild_indices = [i for i, l in enumerate(labels) if l == 'mild']
        ref_idx = mild_indices[0] if mild_indices else 0
        ref_name = 'mildest'

    for cat in ['mild', 'moderate', 'intense']:
        cat_idx = [i for i, l in enumerate(labels) if l == cat]
        if not cat_idx:
            continue

        # Similarity to reference
        sims_to_ref = [cos_sim[i, ref_idx] for i in cat_idx]
        print(f'\n{cat} → {ref_name}:')
        print(f'  mean={np.mean(sims_to_ref):.4f}, std={np.std(sims_to_ref):.4f}')
        print(f'  min={np.min(sims_to_ref):.4f}, max={np.max(sims_to_ref):.4f}')

    # Inter-category similarity
    for cat_a, cat_b in [('mild', 'intense'), ('mild', 'moderate'), ('moderate', 'intense')]:
        idx_a = [i for i, l in enumerate(labels) if l == cat_a]
        idx_b = [i for i, l in enumerate(labels) if l == cat_b]
        cross_sims = [cos_sim[i, j] for i in idx_a for j in idx_b]
        print(f'\n{cat_a} ↔ {cat_b}:')
        print(f'  mean={np.mean(cross_sims):.4f}, std={np.std(cross_sims):.4f}')

    # Specific pairs (only for ICT mode with named blendshapes)
    if len(bs_names) <= 60:
        collapse_pairs = [
            ('jawOpen', 'mouthSmile_L'),
            ('jawLeft', 'mouthLeft'),
            ('mouthFunnel', 'mouthPucker'),
            ('eyeBlink_L', 'eyeSquint_L'),
            ('browDown_L', 'browInnerUp_L'),
        ]
        print('\n=== Collapse Risk Pairs ===')
        name_to_idx = {n: i for i, n in enumerate(bs_names)}
        for a, b in collapse_pairs:
            if a in name_to_idx and b in name_to_idx:
                sim = cos_sim[name_to_idx[a], name_to_idx[b]]
                print(f'  {a:25s} vs {b:25s}: cos_sim = {sim:.4f}')

    # Save stats
    stats_path = os.path.join(save_dir, 'exp2_cosine_sim_stats.txt')
    with open(stats_path, 'w') as f:
        f.write(f'=== Cosine similarity to {ref_name} (idx={ref_idx}) ===\n')
        for i, (name, label) in enumerate(zip(bs_names, labels)):
            sim = cos_sim[i, ref_idx]
            f.write(f'{i:4d} [{label:8s}]  sim = {sim:.4f}\n')
    print(f'Saved: {stats_path}')


# ============================================================
# MF Real Data: load, categorize, encode
# ============================================================

def load_mf_real_and_encode(exp_encoder, dfn_info, img_feat,
                            data_basedir='/data/sihun', split='test',
                            max_frames=500, device='cuda'):
    """Load MF ROM real npy frames, categorize by displacement magnitude, encode."""
    import igl, glob, pickle
    from utils.remesh_utils import procrustes_LDM

    # Load template
    pkl_path = os.path.join(data_basedir, 'multiface_align', 'mf_templates.pkl')
    if not os.path.exists(pkl_path):
        pkl_path = os.path.join(data_basedir, 'pca', 'multiface_align', 'mf_templates.pkl')
    with open(pkl_path, 'rb') as f:
        templates = pickle.load(f)

    std = np.load('utils/mf/standardization.npy', allow_pickle=True).item()
    faces_np = np.array(std['new_f'], dtype=np.int32)

    # Precompute DiffusionNet operators for MF
    import trimesh
    from utils.nfr_utils import get_dfn_info as _get_dfn_info
    first_id = [k for k in templates if k != 'face'][0]
    mf_mesh = trimesh.Trimesh(vertices=templates[first_id].astype(np.float32),
                               faces=faces_np, process=False)
    mf_dfn_info = _get_dfn_info(mf_mesh, map_location=device)

    # Find all npy frames
    base = os.path.join(data_basedir, 'multiface_align', 'ROM', split, 'vertices_npy')
    id_dirs = sorted([d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))])

    all_verts = []
    all_templates = []
    all_names = []

    for id_name in id_dirs:
        template_np = templates.get(id_name)
        if template_np is None:
            continue
        template_np = template_np.astype(np.float32)
        id_dir = os.path.join(base, id_name)
        for seq_dir in sorted(os.listdir(id_dir)):
            seq_path = os.path.join(id_dir, seq_dir)
            if not os.path.isdir(seq_path):
                continue
            frame_files = sorted(glob.glob(os.path.join(seq_path, '*.npy')))
            # Sample evenly
            step = max(1, len(frame_files) // (max_frames // max(len(id_dirs), 1) // 5))
            for ff in frame_files[::step]:
                v = np.load(ff, allow_pickle=True).astype(np.float32)
                # Procrustes align
                R, t, _ = procrustes_LDM(v, template_np)
                v = (v @ R.T + t).astype(np.float32)
                all_verts.append(v)
                all_templates.append(template_np)
                all_names.append(f'{id_name}/{seq_dir}/{os.path.basename(ff)}')
                if len(all_verts) >= max_frames:
                    break
            if len(all_verts) >= max_frames:
                break
        if len(all_verts) >= max_frames:
            break

    print(f'Loaded {len(all_verts)} MF ROM frames')

    # Compute displacement magnitudes
    magnitudes = []
    for v, t in zip(all_verts, all_templates):
        mag = np.sqrt(((v - t) ** 2).sum(axis=-1)).sum()
        magnitudes.append(mag)
    magnitudes = np.array(magnitudes)

    # Percentile categorization
    p33 = np.percentile(magnitudes, 33.3)
    p66 = np.percentile(magnitudes, 66.6)
    labels = []
    for mag in magnitudes:
        if mag < p33:
            labels.append('mild')
        elif mag < p66:
            labels.append('moderate')
        else:
            labels.append('intense')

    print(f'  mild: {labels.count("mild")}, moderate: {labels.count("moderate")}, intense: {labels.count("intense")}')
    print(f'  mag range: [{magnitudes.min():.1f}, {magnitudes.max():.1f}], thresholds: {p33:.1f}, {p66:.1f}')

    # Encode
    z_exps = []
    for i, (v, name) in enumerate(tqdm(zip(all_verts, all_names), total=len(all_verts), desc='Encoding MF frames')):
        normals = igl.per_vertex_normals(
            np.asarray(v, dtype=np.float64),
            np.asarray(faces_np, dtype=np.int64)
        ).astype(np.float32)

        verts_t = torch.tensor(v, dtype=torch.float32, device=device).unsqueeze(0)
        norms_t = torch.tensor(normals, dtype=torch.float32, device=device).unsqueeze(0)

        z = encode_expression(exp_encoder, mf_dfn_info, img_feat, verts_t, norms_t, device)
        z_exps.append(z.cpu().numpy())

    z_exps = np.concatenate(z_exps, axis=0)
    return z_exps, labels, all_names, magnitudes


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='ict', choices=['ict', 'mf_real'],
                        help='ict: single-AU ICT analysis, mf_real: MF ROM real data')
    parser.add_argument('--data_basedir', type=str, default='/data/sihun')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--max_frames', type=int, default=500)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('Loading NFS encoder...')
    exp_encoder, dfn_info, img_feat, z_dim = load_nfs_encoder(device=device)
    print(f'Encoder loaded. z_dim={z_dim}')

    if args.mode == 'ict':
        save_dir = 'analysis_output/latent_analysis_ict'
        print('\nGenerating and encoding ICT expressions...')
        z_exps, labels, names, magnitudes = generate_and_encode(
            exp_encoder, dfn_info, img_feat, device=device)

    elif args.mode == 'mf_real':
        save_dir = 'analysis_output/latent_analysis_mf_real'
        print('\nLoading and encoding MF ROM real data...')
        z_exps, labels, names, magnitudes = load_mf_real_and_encode(
            exp_encoder, dfn_info, img_feat,
            data_basedir=args.data_basedir, split=args.split,
            max_frames=args.max_frames, device=device)

    print(f'Encoded {z_exps.shape[0]} expressions, z shape={z_exps.shape}')

    print('\n=== Experiment 1: t-SNE / PCA Visualization ===')
    exp1_tsne(z_exps, labels, names, save_dir=save_dir)

    print('\n=== Experiment 2: Cosine Similarity Analysis ===')
    exp2_cosine_similarity(z_exps, labels, names, save_dir=save_dir)

    print('\nDone!')
