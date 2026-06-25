# Neural coordinates for facial animation retargeting

<a href="https://chacorp.github.io/nfs-page/"><img src="https://img.shields.io/static/v1?label=Project&message=Page&color=red" height=22.5></a>

Official implementation of **"Neural Face Skinning for Mesh-agnostic Facial Expression Cloning"**.

<img src="assets/teaser.png" alt="teaser"/>

---

## TODOs
- [x] Installation
- [x] Training code
- [x] Inference script (`inference.py`)
- [x] Evaluation (self / cyclic consistency)
- [x] Custom data inference (`--src_custom`, `--tgt_custom`)
- [ ] Clean up utils
- [ ] Pretrained model release

---

## 1. Installation

### Requirements
- Ubuntu 20.04 / 22.04
- CUDA 11.8 (tested on NVIDIA RTX A5000)

### Docker
```bash
docker pull chacorp/audio2face:1.0   # CUDA 11.8
```

### Python environment
```bash
bash setup.sh -m 1   # CUDA 11.8
```

This installs all packages in `requirements.txt`, including:
- PyTorch, torch-geometric (scatter / sparse / cluster)
- libigl, trimesh, open3d
- pytorch3d
- mitsuba 3.6.1 (rendering)
- diffusion-net dependencies

### Third-party repos
```bash
cd third_party
git clone https://github.com/USC-ICT/ICT-FaceKit
git clone https://github.com/vsitzmann/siren
git clone https://github.com/wimmerth/back-to-3d-few-shot-keypoints
git clone https://github.com/yanx27/Pointnet_Pointnet2_pytorch.git
# diffusion-net is already included
```

### matplotrender (visualization)
```bash
pip install git+https://github.com/chacorp/matplotrender.git
```

---

## 2. Downloads

Download and place the following archives in the project root.

| File | Contents | Link |
|------|----------|------|
| `ict_face_pt.tar` | ICT parametric face model | [Google Drive](https://drive.google.com/file/d/1NeSJyVgybzZS-p6uHafv6e3Tv8jTxbCy/view?usp=sharing) |
| NFR files | `data/`, `experiments/`, `test-mesh/` | [Google Drive](https://drive.google.com/file/d/1cXXeU3AtpoGEVz2mhlWTSG1dEbAtCmD1/view?usp=sharing) |

After extraction the directory should look like:

```
NeuralFacialAnimation/
├── ict_face_pt/
│   ├── exp_basis.pt
│   ├── id_basis.pt
│   ├── neutral_verts.pt
│   ├── quad_faces.pt
│   ├── ict_id_vecs_test.pt
│   └── random_expression_vecs.npy
├── data/              # NFR files
├── experiments/       # NFR files
├── test-mesh/         # reference meshes (BIWI, FLAME, CoMA, …)
└── third_party/
    ├── diffusion-net
    ├── ICT-FaceKit
    └── …
```

---

## 3. Data preparation

### Supported datasets
| Key | Dataset |
|-----|---------|
| `voca` | [VOCA](https://voca.is.tue.mpg.de/) |
| `biwi` | [BIWI](https://data.vision.ee.ethz.ch/cvl/datasets/b3dac2.en.html) |
| `coma` | [CoMA](https://coma.is.tue.mpg.de/) |
| `mf_SEN` | [Multiface](https://github.com/facebookresearch/multiface) (sentence) |
| `mf_ROM` | [Multiface](https://github.com/facebookresearch/multiface) (range-of-motion) |
| `ict` | [ICT-3DRFE](https://github.com/USC-ICT/ICT-FaceKit) |
| `ict-cap` | ICT captured sequences |

### Pre-processing a custom neutral mesh

Align your mesh to the coordinate system using `align.blend` (from [NFR](https://github.com/dafei-qin/NFR_pytorch)), then run:

```bash
python utils/data_prepare.py \
  -i /path/to/neutral.obj \
  -o /path/to/processed/
```

Output:
```
/path/to/processed/
├── mesh_dfn_info.pkl   # DiffusionNet precomputes
├── mesh_img.npy        # rendered image feature
├── mesh_mesh.obj       # processed mesh
└── mesh_operators.pkl  # gradient operators
```

> Only the **neutral (rest-pose) mesh** needs to be processed — not every expression frame.

---

## 4. Training

Edit `train_CBD.sh` and run:

```bash
bash train_CBD.sh
```

### Key options

```bash
python train_CBD.py \
  --max_epoch 200 \
  --lr 1E-4 \
  --sc_step 20 \
  --version 5 \            # model version (5 = NGBCv5, our method)
  --batch_size 8 \
  --num_cage_v 512 \       # number of cage / control vertices
  --in_type 1 \            # input: 1 = position + normal
  --out_type 1 \           # output: 1 = absolute vertex position
  --last_activation relu \ # weight activation: relu | elu | softmax | softplus | sqrelu | none
  --data_toggle \          # randomly toggle between src/tgt roles during training
  --use_data1              # include additional dataset combinations
```

Checkpoints are saved under `ckpts_CBD/<timestamp>-NGBCv5/`.

### Continuing from a checkpoint

```bash
python train_CBD.py \
  --ckpt ckpts_CBD/<checkpoint> \
  --continue_ckpt \
  --start_epoch <epoch> \
  [other options …]
```

---

## 5. Inference

`inference.py` runs retargeting from a source expression sequence to a target identity and saves per-frame `.npy` meshes. Visualization images can be generated with `--vis`.

### Dataset mode

```bash
# cross-retarget: mf_ROM → ICT identity 2
python inference.py \
  --ckpt ckpts_CBD/<checkpoint> \
  --src mf_ROM --tgt ict --tgt_id 2

# self-retarget with rendered images
python inference.py \
  --ckpt ckpts_CBD/<checkpoint> \
  --src ict --src_id 0 --tgt ict --tgt_id 0 --vis
```

### Custom (in-the-wild) mode

```bash
# Custom source OBJ sequence → dataset target
python inference.py \
  --ckpt ckpts_CBD/<checkpoint> \
  --src_custom /path/to/obj_sequence/ \
  --tgt ict --tgt_id 2 --vis

# Dataset source → custom target mesh (.obj / .ply)
python inference.py \
  --ckpt ckpts_CBD/<checkpoint> \
  --src mf_ROM \
  --tgt_custom /path/to/target.ply --vis

# Both custom (optionally provide explicit neutral for the source)
python inference.py \
  --ckpt ckpts_CBD/<checkpoint> \
  --src_custom /path/to/obj_sequence/ --src_neutral /path/to/neutral.obj \
  --tgt_custom /path/to/target.obj --vis
```

### All options

| Option | Default | Description |
|--------|---------|-------------|
| `--ckpt` | required | checkpoint directory |
| `--src` | — | source dataset (`voca` / `biwi` / `mf_SEN` / `mf_ROM` / `coma` / `ict` / `ict-cap`) |
| `--src_custom` | — | source OBJ sequence folder (mutually exclusive with `--src`) |
| `--tgt` | — | target dataset (same choices as `--src`) |
| `--tgt_custom` | — | target neutral mesh file (.obj / .ply, mutually exclusive with `--tgt`) |
| `--src_id` / `--tgt_id` | 0 | identity index (dataset mode) |
| `--src_neutral` | first frame | explicit neutral mesh for custom source |
| `--max_frames` | -1 (all) | limit number of frames |
| `--output_dir` | `eval_CBD/` | root directory for `.npy` output |
| `--save_gt` | off | also save ground-truth source frames |
| `--vis` | off | save 4-panel rendered images |
| `--vis_dir` | `<output>/vis/` | image output directory |
| `--size` | 4 | panel size in inches |
| `--device` | `cuda:0` | compute device |

Output `.npy` files have shape `[V, 3]` (vertex positions) and are saved under:
```
<output_dir>/<ckpt_name>/<src>-to-<tgt>/
```

---

## 6. Evaluation

### Self-retargeting (reconstruction) MSE

```bash
python eval_CBD.py \
  --version 5 \
  --ckpt ckpts_CBD/<checkpoint> \
  --data_selection 4   # 0=voca 1=biwi 2=mf_SEN 3=coma 4=mf_ROM 5=ict 6=ict-cap
```

### Cyclic consistency evaluation

```bash
bash eval_CBD_cyc.sh
```

Or manually:

```bash
python eval_CBD_cyc.py \
  --version 5 \
  --ckpt ckpts_CBD/<checkpoint> \
  --src_data ict --tgt_data mf_SEN \
  --log_dir eval_cyc \
  --batch_size 1
```

Results are written to `eval_cyc/`.

---

## 7. Rendering

### Matplotlib-based (fast, headless)

Used by `inference.py --vis`. Renders 4-panel comparison:
`src neutral | src expression | tgt neutral | predicted expression`

```python
from matplotrender import plot_mesh_gouraud
plot_mesh_gouraud([v1, v2], [f1, f2], rot_list=[[0,-10,0]]*2, save=True, logdir='out/')
```

### Mitsuba 3 (high quality)

Path-traced renderer with soft shadows and physically-based skin material.

```bash
python render/render_mitsuba.py
```

```python
from render.render_mitsuba import render_figure, PAPER_CFG
render_figure([(src_v, f), (pred_v, f), (tgt_v, f)], PAPER_CFG, 'output.png')
```

### Trimesh-based

```bash
python render/render_trimesh.py
```

---

## 8. Notebooks

Interactive notebooks for visualization and analysis:

| Notebook | Description |
|----------|-------------|
| `notebook/NBCv2_visualize.ipynb` | Main visualization: retargeting, cage weights, animation |
| `notebook/NBC_inference.ipynb` | Quick inference and result inspection |
| `notebook/NFS_inference.ipynb` | NFS baseline inference |
| `notebook/NFR_inference.ipynb` | NFR baseline inference |
| `notebook/log_reader.ipynb` | Training log / TensorBoard reader |
| `notebook/visualize_meshes.ipynb` | Mesh comparison viewer |

---

## Acknowledgements

We thank the contributors of [NFR](https://github.com/dafei-qin/NFR_pytorch), [CodeTalker](https://github.com/Doubiiu/CodeTalker), [FaceFormer](https://github.com/EvelynFan/FaceFormer), and [Diffusion-Net](https://github.com/nmwsharp/diffusion-net) for their open research.

We also thank the contributors of [ICT-FaceKit](https://github.com/USC-ICT/ICT-FaceKit), [Multiface](https://github.com/facebookresearch/multiface), [VOCA](https://voca.is.tue.mpg.de/), [BIWI](https://data.vision.ee.ethz.ch/cvl/datasets/b3dac2.en.html), and [CoMA](https://coma.is.tue.mpg.de/) for making their data publicly available.

We appreciate the creators of [Mery](https://www.meryproject.com), [Malcolm](https://www.animschool.com), [Piers](https://www.cgtrader.com/free-3d-models/character/man/maya-character-rig-piers-3d-rig), [Morphy](http://www.joshburton.com/projects/morpheus.asp), and [Bonnie](https://www.joshsobelrigs.com/).
