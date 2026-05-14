# Dynamic 6-Face Supplementary Video — 구현 계획

## 요청 요약

1920×1080, 단일 Mitsuba 씬에 6개 얼굴 메시를 X축으로 배치.  
처음 200프레임은 ict_neutral 클로즈업으로 얼굴 애니메이션 재생,  
200프레임 이후 카메라가 distance 증가 + target pan으로 6개 얼굴을 공개,  
이후 고정 wide view에서 나머지 애니메이션 재생.

---

## 타임라인

| 구간 | 프레임 | 카메라 상태 |
|------|--------|-------------|
| 클로즈업 | 0 ~ 199 | dist=2.5, target=(-1.5, -0.02, 0) — ict_neutral 정면 고정 |
| 트랜지션 | 200 ~ 289 | dist 2.5→6.5, target.x -1.5→0.0 (cosine easing, ~90프레임) |
| 와이드 | 290 ~ 1146 | dist=6.5, target=(0.0, -0.02, 0) — 6개 전부 고정 |

총 1147프레임 (소스 시퀀스 길이 기준), 30fps

---

## 데이터

- **VIS_DIR**: `vis_CBD/2026-04-27-15-36-49-NGBCv5/`
- **소스**: `ict-cap-ID_000_test`

| 순서 | 시퀀스 디렉토리 | mesh_type | X offset |
|------|----------------|-----------|----------|
| 0 | `ict-cap-ID_000_test-to-ict-neutral_01-masked` | ict | -1.5 |
| 1 | `ict-cap-ID_000_test-to-ict-cap-ID_000_test_01-masked` | ict | -0.9 |
| 2 | `ict-cap-ID_000_test-to-biwi-ID_012_test_01-masked` | biwi | -0.3 |
| 3 | `ict-cap-ID_000_test-to-coma-ID_006_test_01-masked` | coma | 0.3 |
| 4 | `ict-cap-ID_000_test-to-coma-ID_008_test_01-masked` | coma | 0.9 |
| 5 | `ict-cap-ID_000_test-to-mf_ROM-ID_012_test_01-masked` | mf_ROM | 1.5 |

---

## 구현 파일

**`render/render_dynamic_6faces.py`** (신규)

### 주요 함수

#### `_build_multi_scene(obj_x_list, cam_dist, cam_target, cfg, W, H, spp)`
- `obj_x_list`: `[(obj_path, x_offset), ...]` 6개
- 씬 dict에 `face_0`~`face_5` 각각 `to_world: _translate([x_offset, 0, 0])` 적용
- FOV 고정 (`fov=20`), `cam_distance`·`cam_target` 만 프레임마다 변화
- 카메라·조명·ground는 기존 `_build_scene()` 세팅과 동일

#### `camera_interp(frame_idx) -> (dist, target)`
```
frame_idx < 200  → dist=2.5, target_x=-1.5  (고정)
200 ≤ i < 290    → t=(i-200)/90, cosine easing
                   dist = 2.5 + (6.5-2.5) * (1-cos(π*t))/2
                   target_x = -1.5 + (0.0-(-1.5)) * (1-cos(π*t))/2
frame_idx ≥ 290  → dist=6.5, target_x=0.0  (고정)
```

#### `render_frame_multi(verts_list, faces_list, x_offsets, dist, target, cfg)`
- 6개 메시 temp OBJ 저장
- `_build_multi_scene()` 호출 → `mi.render()` → tonemap → uint8 반환

#### `render_all(debug=False)`
- debug 시 frame [0, 100, 200, 245, 290, 291] 총 6프레임만 렌더
- 각 프레임: `camera_interp(i)` → 6 npy 로드 → `render_frame_multi()` → PNG
- ffmpeg → `video_mi/dynamic_6faces.mp4`

---

## 렌더 세팅

| 항목 | 값 |
|------|----|
| 해상도 | 1920 × 1080 |
| FOV | 20° (전 구간 고정) |
| SPP | 256 |
| max_depth | 6 |
| FPS | 30 |
| mesh_scale | 0.225 |
| X spacing | 0.6 units |
| bg_color | (0.7, 0.7, 0.7) |
| ground_y | -0.5 |

---

## 검증 계획

- **실행**: `cd render && python render_dynamic_6faces.py --debug`
- frame 0 → dist=2.5 (클로즈업 확인)
- frame 245 → dist 중간값 (트랜지션 확인)
- frame 290 → dist=6.5 (와이드 확인)
- 출력 shape: `(1080, 1920, 3)`
- MP4 생성 확인

**최대 검증 루프: N = 3**

---

## 출력

- 프레임: `video_mi/dynamic_6faces_frames/`
- 최종: `video_mi/dynamic_6faces.mp4`
