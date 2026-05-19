# HLBS — Handover Note (temporary)

> 인계용 임시 문서. 작업 마무리되면 삭제 또는 git rm.
> 작성일: 2026-05-19 / 사용자: leeinyup123@gmail.com

---

## 지금 상태 (가장 중요)

**Per-id bind pose GT cache identity mismatch 버그를 방금 잡았다.**

- `train_hlbs.py` ICT 학습은 `ict_face_pt/random_identity_vecs.npy[:111]` 을 id로 씀.
- 그런데 `nfs_features_seg/{name}_bind_pos_landmark.npy` 캐시는 `data/ICT_live_100/iden_vecs.npy` 로 만들어져 있었음.
- `id_name` 포맷이 둘 다 `ict_NNN` 이라 충돌은 안 났는데, **다른 사람의 GT로 supervise 하고 있었다.**
- → ICT bind_pose 학습이 안 됐던 추정 근본 원인.
- → `tools/precompute/precompute_per_id_bind_pos_v2.py` 에 `--ict_iden_vecs`, `--n_ict_ids` 추가. **train cache는 재생성 완료**. (`nfs_features_seg/ict_000_bind_pos_landmark.npy` ~ `ict_110_*` 확인됨.)
- val/test cache 는 별도 디렉토리 (`data/ICT_live_100/iden_vecs.npy` 용) 가 필요하면 다시 돌릴 것.

**다음 할 일: char-s03 settings 그대로 재학습 (fresh, 처음부터).**

매칭 기준 ckpt: `ckpts_hlbs/2026-05-14-14-02-03-HLBS-FullPred-ict-jTrans-nrm0.1-Wsm0.01/train_opts.yml`

재시작 커맨드 (verified, --tb 포함):
```bash
cd /source/inyup/NeuralFacialAnimation && \
python train_hlbs.py \
    --rig_path maya_rig/hybrid \
    --topo_key ict \
    --hid_dim 128 \
    --use_joint_trans \
    --full_prediction \
    --lambda_normal 0.1 \
    --lambda_W_smooth 0.01 \
    --lambda_init 0.0 \
    --lambda_bind_reg 1.0 \
    --lambda_helper_residual 0.01 \
    --lambda_mirror 0.1 \
    --gmm_mode residual \
    --residual_scale 4.0 \
    --sigma_targets_npy maya_rig/hybrid/sigma_targets.npy \
    --lambda_sigma_reg 0.1 \
    --lambda_net_center 0.05 \
    --use_helpers 1 \
    --active_joints_json maya_rig/hybrid/active_joints_manual.json \
    --use_gmm_hybrid \
    --use_geodesic_gauss \
    --per_id_bind_pose_dir nfs_features_seg \
    --subsample_mode mix4 \
    --subsample_contour_r0 0.05 \
    --subsample_contour_r1 0.2 \
    --subsample_contour_kind point \
    --nfs_feat_dir nfs_features_seg \
    --nfs_concat \
    --adain_pos_norm \
    --batch_size 8 \
    --max_epoch 500 \
    --val_every 10 \
    --use_data2 \
    --tb
```

**중요 주의:**
- `train_hlbs.py:2566` `--config` 머지 로직은 사실상 죽어있다 (argparse defaults가 yaml을 다 덮어씀). 그러니 위처럼 모든 flag CLI로 명시할 것. `--config <yml>` 만 믿지 말 것.
- `--continue_ckpt` / `--ckpt` / `--start_epoch` 안 줌 = **bottom-up 새로 학습**. ckpt 이어받기 절대 하지 말 것 (per-id GT가 바뀌었으므로 의미 없음).

---

## 최근 큰 변경 요약 (이 session에서 들어간 것)

1. **Phase B v3 per-id bind pose**: `tools/precompute/precompute_per_id_bind_pos_v2.py`
   - 공식: `bind_pose_id[j] = V_id[home_vidx[j]] + (bind_pos_mean_GT[j] - V_mean[home_vidx[j]])`
   - `home_vidx`: `maya_rig/hybrid/joint_home_vidx_{topo}.npy` (J,)
   - **helper joint override**: helper의 per-id bind = per-id parent bind (mesh surface 아님). `active_joints_manual.json::helper_joint_idx` 로 인식.

2. **Helpers (Option A)**: reparented 19 joints
   - `maya_rig/hybrid/active_joints_manual.json` (current)
   - `maya_rig/hybrid/helper_joints_v1.json`
   - 백업: `*.bak_pre_helpers` (helpers 끄고 싶을 때)
   - 학습 루프에 `L_helper_residual` (λ=0.01), `L_mirror` (λ=0.1), bind_pose_net의 helper residual reparam 들어감.

3. **Subsample mix4**: `tools/.../train_hlbs.py::_build_subsample_perm`
   - per-batch dynamic: full / FPS / random / importance-strict 중 하나 (4-way 추첨).
   - `--subsample_mode mix4`, `--subsample_ratio_min 0.1 --subsample_ratio_max 0.5`.
   - FPS: PyTorch3D `sample_farthest_points` (GPU 권장, CPU fallback 동작은 함).

4. **kNN-PCA normal estimation**: 서브샘플 시 face index가 깨지므로 vertex normal을 kNN-PCA로 estimate. `--normal_knn_k 16`.

5. **Cross-retarget loss**: `--lambda_cross_retarget`, cyclic 옵션 `--lambda_cross_cyclic`. 둘 다 현재 0 (속도 이슈로 보류).

6. **Caricaturization aug**: `tools/precompute/precompute_caricaturized_aug.py`
   - third_party Sela CVIU 2015 코드 기반.
   - γ=0.125~0.15, similarity align, landmark blend mask (mouth/nose 0.3, eye 0.5).
   - **아직 학습 cache로 들어가지 않음** (Stage 1 bind_pose_only 학습 후 검토 예정).

7. **bind_pose_only Stage 1 학습**: `tools/train/train_bind_pose_only.py`
   - 별도 standalone, HLBS model arch 로드하지만 bind_pose_net만 학습 (helper loss 없음).
   - 목적: 본 학습 전에 bind_pose_net만 미리 잘 학습시켜 state_dict 이어쓰기.
   - **현재 보류** — char-s03 재학습 결과 보고 결정.

---

## 파일 포인터 (자주 건드리는 곳)

| 용도 | 경로 |
|---|---|
| 메인 학습 | `train_hlbs.py` |
| 평가 | `eval_hlbs.py` |
| Per-id GT precompute | `tools/precompute/precompute_per_id_bind_pos_v2.py` |
| Caricaturization aug | `tools/precompute/precompute_caricaturized_aug.py` |
| bind_pose_only 학습 | `tools/train/train_bind_pose_only.py` |
| Cross-retarget 분석 | `tools/analyze/cross_retarget_internals.py` |
| Maya 시각화 | `tools/vis/vis_aug_bind_pose_maya.py` |
| 핸드오버 슬라이드 | `handover/hlbs_handover.md` (Marp) |
| Rig 설정 | `maya_rig/hybrid/{active_joints_manual.json, helper_joints_v1.json, sigma_targets.npy, joint_home_vidx_{topo}.npy}` |
| 매칭 기준 ckpt | `ckpts_hlbs/2026-05-14-14-02-03-HLBS-FullPred-ict-jTrans-nrm0.1-Wsm0.01/train_opts.yml` |

---

## 데이터 / 학습 셋업 메모

- ICT 학습 id: `ict_face_pt/random_identity_vecs.npy[:111]` (111명)
- ICT val/test id: `data/ICT_live_100/iden_vecs.npy` (100명, **다른 사람**)
- MF: 13명 (11 train / 1 val / 1 test)
- expression도 online 추출 아님 — 다 미리 뽑아둠.
- `--use_data2` 만 켜고 학습 (data0/1/3 안 씀).
- `data_basedir = /data/sihun` (다른 서버에서는 경로 확인 필요).

---

## 미해결 / 보류

1. **Unseen-topology generalization** (BIWI, COMA): landmark 없으니 `home_vidx` 못 만듦. 학습 시에만 필요하게 하고 inference 시 우회 방법 미정. GMM mean 모호성 (Top_Lip/Bottom_Lip 같은 부위) 도 같은 맥락.
2. **Cross-retarget loss 속도**: 너무 느려서 학습 사이클에 통합 보류. 최적화 필요.
3. **`lambda_dist` = 0** 으로 사용 안 함 (확인 완료).
4. ICT bind_pose 학습 품질 — **char-s03 재학습 결과 보고 판단**.

---

## 사용자 협업 메모 (이 사용자가 강하게 요구하는 것)

- **임의 결정 금지**: char-s03 / 원본 세팅 그대로 맞추라고 하면 진짜 그대로 맞춰야 함. "내 맘대로 정하지말고" 였음.
- **응답은 짧게**: 길게 설명하지 말 것.
- **변경한 거 모두 명시**: 한 거 / 안 한 거 정확히 표시.
- 한국어로 대화.

---

## 다른 서버에서 작업 시작할 때 체크리스트

1. `git pull` 후 이 파일 읽었나 ✓
2. `/data/sihun/...` 경로 마운트 / 데이터 위치 확인
3. `ict_face_pt/random_identity_vecs.npy` 있나 (학습용 id)
4. `nfs_features_seg/ict_000_bind_pos_landmark.npy` 있나 — **없으면 train cache 재생성 필요**:
   ```bash
   python tools/precompute/precompute_per_id_bind_pos_v2.py \
       --datasets ict --feat_dir nfs_features_seg \
       --ict_iden_vecs ict_face_pt/random_identity_vecs.npy --n_ict_ids 111
   ```
5. `maya_rig/hybrid/sigma_targets.npy`, `active_joints_manual.json`, `helper_joints_v1.json`, `joint_home_vidx_{ict,mf}.npy` 있나
6. 위 char-s03 매칭 커맨드 실행
7. tensorboard: `tensorboard --logdir ckpts_hlbs/`
