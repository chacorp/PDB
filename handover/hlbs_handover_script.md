# HLBS 인수인계 — 발표 스크립트

각 슬라이드별 짧은 발표 멘트.

---

## Slide 1 — 표지 (HLBS 인수인계)

> "안녕하세요. 오늘은 저희가 진행 중인 HLBS, Hierarchical Linear Blend Skinning 연구의 수학적 배경과 코드 구조를 인수인계 차원에서 정리해보려고 합니다."

---

## Slide 2 — 0. Big Picture

> "한 줄로 요약하면, **HLBS는 기존 LBS 식에 학습 가능한 bind pose와 skin weight를 얹은 것**입니다. 일반 LBS는 joint의 transform만 움직이지만, 저희는 identity마다 다른 bind pose, vertex마다 다른 skin weight도 신경망으로 예측합니다.
> 목적은 ICT, MF, BIWI, COMA처럼 토폴로지가 모두 다른 face mesh 4종에 같은 face joint set으로 표정을 학습하고 전이하는 것입니다.
> 핵심 구성요소는 **(1) joint hierarchy 기반 FK chain**, **(2) per-id bind pose 예측**, **(3) GMM prior와 net 출력 hybrid의 per-vertex skin weight**, 이 세 가지입니다."

---

## Slide 3 — 1. LBS 기초

> "LBS의 핵심 식 한 줄: **각 vertex의 변형된 위치는, 그 vertex에 영향을 주는 joint들의 transform 행렬의 가중 평균**입니다.
> 가중치 `W[v, j]`는 vertex가 joint에 얼마나 묶여있는지를 의미하고, sum-to-1 제약이 있습니다. `G_j`는 joint의 현재 world transform과 bind transform inverse의 곱이고요.
> LBS는 선형 가중 평균이라 빠르지만, 팔꿈치나 무릎 같은 곳에서 candy-wrapper artifact가 잘 생기는 단점이 있습니다. 저희 face mesh는 큰 회전이 없어서 LBS로도 충분합니다.
> 코드는 `models/hierarchical_lbs.py`의 `HierarchicalLBS_FullPred.forward` 마지막 부분에 있습니다."

---

## Slide 4 — 2. Bind Pose

> "Bind pose는 **skeleton과 mesh가 처음 정렬되어 있는 reference 자세**입니다. neutral expression, T-pose라고 보면 됩니다.
> 모든 LBS 계산은 이 자세 기준이라, joint의 bind world transform `B_j`와 그 inverse `B_j⁻¹`가 핵심 양입니다.
> 저희 task에서 특이한 점은, ICT identity마다 얼굴 크기와 구조가 달라서 joint의 bind 위치도 달라야 한다는 거예요. 그래서 `bind_pose_net`이라는 신경망이 **identity마다 다른 bind pose를 예측**합니다."

---

## Slide 5 — 3. FK Chain (Forward Kinematics)

> "FK는 **각 joint의 world transform이 부모의 world transform과 자신의 local transform의 곱**이라는, 거의 정의 그 자체입니다.
> 식은 단순하지만 '부모 먼저 계산'이라는 순서 제약이 있습니다. 보통은 joint index 순으로 그냥 돌면 되는데, 저희가 helper joint들을 reparent하면서 일부 joint의 parent index가 자기보다 커지는 상황이 생겼고, 그래서 `process_order`라는 topological sort 결과 buffer를 따로 만들어 사용합니다.
> Rotation 표현은 **6D continuous representation**을 씁니다. Zhou et al. CVPR 2019 논문에서 제안한 것으로, Euler angle의 gimbal lock이나 quaternion의 antipodal 문제를 회피합니다."

---

## Slide 6 — 4. Skin Weights W

> "Skin weight는 vertex별로 어떤 joint에 얼마나 묶여있는지를 나타내는 값, [0, 1] 범위에 J개 joint에 대해 sum-to-1 제약이 있습니다.
> 저희는 매 forward마다 신경망이 weight를 예측하는데, 신경망 출력만 쓰면 학습이 불안정해서 **GMM Gaussian prior와 hybrid**로 씁니다. 식은 `softmax(logit_net + logit_gauss)`이고요.
> `logit_gauss`는 vertex가 joint center에 가까울수록 큰 값을 주는 Gaussian이고, 각 joint마다 학습 가능한 σ를 갖습니다.
> 추가로 face_joint_idx mask가 있어서, face 영역 vertex는 face joint들의 softmax로만 weight 받고, 외곽 (목, 뒤통수)은 base joint 하나로 routing해서 face motion이 외곽으로 새지 않게 합니다."

---

## Slide 7 — 5. 학습 가능한 것 / 고정된 것

> "정리하면, **고정**은 rig의 joint hierarchy, parent index, helper reparenting 결과, 그리고 home vertex 매핑입니다.
> **학습**은 per-id bind pose, per-vertex skin weight, expression latent z_exp, per-joint transform delta T, 그리고 GMM의 σ까지. 즉 식의 거의 모든 unknown은 학습 대상입니다.
> 학습 신호는 reconstruction MSE + 여러 regularization으로 들어오고, 뒤 슬라이드에서 다룹니다."

---

## Slide 8 — 6. Helper Joints (Option A)

> "Helper joint는 **원래 frozen이던 19개 joint를 face 영역으로 reparent해서 face motion 표현력을 늘린 것**입니다.
> 예를 들어 Cranium 같은 머리 윗부분 joint를 Left_Eye_Socket의 자식으로 옮기는 식이에요.
> 직접 학습하면 위치가 발산할 수 있어서, **parent + small residual로 reparameterize**합니다. 즉 bind_pose_net의 helper 출력을 residual로 해석하고, L2 penalty를 줘서 helper가 parent에서 너무 멀리 가지 않게 합니다.
> 추가로 좌우 helper 8쌍에 대해 `L_mirror`로 x-축 대칭을 강제하고, helper의 per-id GT는 parent의 per-id 위치 그대로라서 `L_bind_reg`에서는 helper를 제외합니다."

---

## Slide 9 — 7. Per-id Bind Pose Supervision (Phase B v3)

> "각 identity에 대한 bind pose의 GT를 어떻게 만드느냐의 문제입니다.
> 핵심 아이디어: **각 joint마다 anatomical home vertex를 mesh vertex 인덱스로 미리 매핑해둡니다**. 예를 들어 'Top_Lip' joint는 'DTU3D landmark 50번 위치의 가장 가까운 mesh vertex'로 정의됩니다.
> mean rig에서 'joint 위치 - home vertex 위치' 차이를 offset으로 저장해두고, 각 identity에서는 그 id의 home vertex 위치에 offset만 더해주면 per-id bind pose가 됩니다.
> 결과는 `nfs_features_seg/*_bind_pos_landmark.npy` cache로 저장되어 학습 시 `L_bind_reg`의 target으로 쓰입니다. 다른 서버에서는 `precompute_per_id_bind_pos_v2.py`로 재생성합니다."

---

## Slide 10 — 8. Loss Functions 요약

> "Loss는 크게 세 그룹으로 나뉩니다.
> **Reconstruction**: `recon-lbs`가 vertex MSE, `recon-normal`이 normal cosine similarity, `recon-neu`가 neutral 출력 supervision.
> **Bind pose 학습**: `L_bind_reg`가 per-id GT MSE, `L_helper_residual`이 helper residual의 L2, `L_mirror`가 L/R helper 대칭.
> **Skin weight regularization**: `L_dist`가 vertex-joint locality, `L_net_center`가 softmax weight 분포 응집, `L_sigma`가 GMM σ 발산 방지.
> 마지막으로 Track B의 `L_cross_retarget`과 `L_cross_cyclic`은 identity-expression disentanglement를 위한 cross-id supervision입니다."

---

## Slide 11 — 9. Subsample Augmentation (mix4)

> "학습 augmentation은 매 batch마다 4가지 sampling mode 중 1/4 확률로 random pick하는 `mix4`를 씁니다.
> 모드는 **full mesh, FPS, random, importance-strict** 네 가지인데, importance-strict는 brow/eye/nose/mouth landmark 주변에서만 sampling합니다.
> 각 모드는 per-batch dynamic ratio를 가져서, 학습에 mesh resolution 다양성과 영역별 학습 강도 다양성을 모두 줍니다.
> recon-normal은 subsample 시 face connectivity가 깨지지만 GT 쪽은 pre-computed normal을 그대로 사용하고, pred 쪽은 PyTorch3D의 PCA-on-kNN normal estimation으로 해결했습니다."

---

## Slide 12 — 10. Cross-retarget (Track B)

> "Cross-retarget의 핵심 가정: **id A의 표정을 id B의 mesh에 적용한 결과가 id B가 그 표정을 지었을 때의 ground truth와 일치해야 한다**는 거예요. 이걸 직접 supervision으로 줘서 identity와 expression을 disentangle합니다.
> 데이터는 같은 expression z_exp를 두 identity에 적용한 4-tuple, ICT face model로 합성합니다.
> Forward는 source에서 expression을 뽑고 target에서 identity (W, bind_pose)를 가져와서 결합한 결과를 GT와 MSE.
> Cyclic 확장은 reverse retarget까지 돌려서 'B로 갔다가 다시 A로 복원'했을 때 원래 src deformed와 일치하는지 보는 것. bijective encoding을 강제합니다."

---

## Slide 13 — 11. 학습 파이프라인 한 번에

> "전체 파이프라인을 한 화면에 담은 그림입니다.
> 입력은 source neutral mesh와 deformed mesh, NFS feature. bind_pose_net이 per-id joint 위치를 예측하고 그것으로 B_inv를 만듭니다. expression encoder가 z_exp를 추출하면 pose model이 6D rotation과 translation을 joint별로 출력하고, FK chain이 process_order에 따라 world transform을 누적합니다.
> 그 후 G_j = T_world × B_inv로 LBS skinning matrix를 만들고, skin_weight_net의 logit과 GMM prior를 더한 softmax로 W를 구해 최종 LBS 식에 넣으면 prediction이 나옵니다."

---

## Slide 14 — 12. 코드 위치 참고

> "필요할 때 어디를 봐야 하는지 정리한 표입니다.
> Rig 로딩과 process_order는 `utils/rig_loader.py`, 모델 본체는 `models/hierarchical_lbs.py`, 학습 루프는 `train_hlbs.py`. 그 외 precompute와 vis script는 `tools/` 아래에 분류되어 있습니다.
> 새로운 분이 코드 read-through 할 때 model → trainer → precompute 순서로 보시면 좋습니다."

---

## Slide 15 — 끝

> "여기까지가 현재까지의 핵심 background입니다. 질문 받겠습니다."
