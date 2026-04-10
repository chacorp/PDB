"""
Generate caricaturized ICT neutral meshes for data augmentation.

For each ICT identity, generates multiple caricature levels using
computational caricaturization (Sela et al. 2015).

The caricaturized neutrals can be used as augmented targets during training:
  - Target neutral: caricaturized mesh
  - Target GT: caricaturized neutral + original expression displacement

Usage:
    python scripts/generate_caricatures.py \
        --out_dir caricature_data \
        --gammas 0.05 0.1 0.15 0.2
"""
import os, sys, argparse, subprocess
import numpy as np
import trimesh

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

CARI_BIN = os.path.join(
    os.path.dirname(__file__), '..',
    'third_party', 'coupe.computational-caricaturization', 'build', 'main'
)


def caricaturize(input_obj, output_obj, gamma, ref_obj=None):
    """Run caricaturization binary."""
    if ref_obj:
        cmd = [CARI_BIN, 'ref', ref_obj, input_obj, str(gamma), output_obj]
    else:
        cmd = [CARI_BIN, 'noref', input_obj, str(gamma), output_obj]
    subprocess.run(cmd, check=True, capture_output=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="caricature_data")
    parser.add_argument("--gammas", type=float, nargs='+', default=[0.05, 0.1, 0.15, 0.2])
    parser.add_argument("--ref_obj", type=str, default=None,
                        help="Reference mesh for 'ref' mode (default: noref mode)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load ICT identity vectors
    from utils.remesh_utils import ICT_face_model
    ict_model = ICT_face_model()

    iden_vecs = np.load('data/ICT_live_100/iden_vecs.npy')
    ict_faces = ict_model.faces.astype(np.int32)
    print(f"ICT: {len(iden_vecs)} identities, {len(ict_faces)} faces")

    for i in range(len(iden_vecs)):
        id_coeff = iden_vecs[i]
        id_disps = ict_model.get_id_disp(id_coeff).squeeze()
        neutral_verts = (ict_model.neutral_verts + id_disps).astype(np.float32)

        # Save original neutral as obj (temp)
        orig_obj = os.path.join(args.out_dir, f"ict_{i:03d}_orig.obj")
        mesh = trimesh.Trimesh(vertices=neutral_verts, faces=ict_faces, process=False)
        mesh.export(orig_obj)

        for gamma in args.gammas:
            cari_obj = os.path.join(args.out_dir, f"ict_{i:03d}_g{gamma:.2f}.obj")
            try:
                caricaturize(orig_obj, cari_obj, gamma, ref_obj=args.ref_obj)
                cari_mesh = trimesh.load(cari_obj, process=False)
                cari_verts = np.array(cari_mesh.vertices, dtype=np.float32)

                # Save displacement from original neutral (for applying expression later)
                disp = cari_verts - neutral_verts
                disp_path = os.path.join(args.out_dir, f"ict_{i:03d}_g{gamma:.2f}_disp.npy")
                np.save(disp_path, disp)

                print(f"  ict_{i:03d} gamma={gamma:.2f}: disp_mean={np.abs(disp).mean():.6f}")
            except Exception as e:
                print(f"  ict_{i:03d} gamma={gamma:.2f}: FAILED ({e})")

        # Clean up original obj
        os.remove(orig_obj)

    print(f"\nDone. Saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
