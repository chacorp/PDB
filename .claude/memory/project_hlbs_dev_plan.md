---
name: HLBS development plan
description: Phased plan for improving HierarchicalLBS — weight diagnosis, DiffusionNet encoder swap, EDD integration, NFS comparison
type: project
---

## HLBS Development Plan (2026-03-28)

### Phase 0: Weight Visualization + Diagnosis (현재)
- Maya init vs predicted W 비교 (vis_hlbs_weights.py)
- smooth한지? joint boundary 깔끔한지?
- 눈썹 좌/우 joint weight가 분리되어 있는지
- delta_W가 어디서 크게 변하는지

**Why:** bumpy surface 원인이 weight인지 encoder인지 판별
**How to apply:** Phase 1 방향 결정

### Phase 1: 진단 결과에 따른 조치
- Case A: Weight smooth + 표정 안 됨 → exp encoder 문제 → DiffusionNet encoder로 교체
- Case B: Weight bumpy/noisy → skin_weight_net을 DiffusionNet으로 교체
- Case C: Both OK, 여전히 bumpy → LBS 한계 → EDD로 보정

### Phase 2: Encoder backbone 교체 (NFS 방식)
- skin_weight_net: LinearEncoder → DiffusionNet (per-vertex output)
- bind_pose_net: LinearEncoder → DiffusionNet (global pool)
- lbs_exp_z_model: LinearEncoder → DiffusionNet (global pool)
- operators precompute 추가 (topology별 1회)

**Why:** pointwise MLP는 spatial context 없이 weight 예측 → bumpy. DiffusionNet은 표면 topology 인식
**How to apply:** NFS와 동일 encoder backbone, LBS decoder 유지 → fair comparison 가능

### Phase 3: EDD 통합
- 개선된 HLBS (frozen) + DiffusionNetEDD
- target: GT - HLBS_output
- 최종: HLBS coarse + EDD fine detail

### Phase 4: NFS 비교 (checkpoint 확보 후)
- 정량: 같은 test set에서 MSE, per-segment error
- 정성: side-by-side 렌더링/비디오
- HLBS+EDD > NFS 확인 필요

### Key Observation
- HLBS 200ep이 NFS 500ep보다 expression dynamics가 나아보임 (정량 검증 필요)
- 원인 추정: joint hierarchy의 anatomical prior (jaw→lip coupling 등)
- joint translation 추가로 smoothness 개선됨
