---
name: Codebase key paths and versions
description: Key checkpoint paths, version numbering (v9 = original 13dim, v10 = 7dim decomposed), eval script conventions
type: reference
---

- v9 (original): `--use_source_template --smooth_n_iter 0`, 13dim exp_z input, single final recon loss
- v10 (current): no source template, 7dim, decomposed loss (recon-lbs + recon-wrinkle)
- Key checkpoint: `ckpts_CBD/2026-03-09-18-11-33-NGBC++v9` (v9 original, epochs 0-450+)
- Eval scripts: `eval_v9_self.sh`, `eval_v9_cross.sh` — take ckpt path as $1 arg
- Eval results go to: `eval_CBD/<ckpt_name>-eval/{self,cross}_e{epoch}/`
- Dataset vis: `dataset_vis/{mf_ROM, mf_SEN, coma}/` (mp4 videos, gitignored)
- Data basedir: `/data/sihun/`
