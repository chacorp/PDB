# pip uninstall opencv-python opencv-contrib-python opencv-python-headless numpy -y && pip install cupy h5py tensorboard numpy==2.3.2 opencv-python==4.10.0.84 

## if docker by docker.io/jeolpyeoni0/gltorch:cu124-vessl
# pip install tensorboard cupy && python -m pip uninstall -y numpy && python -m pip install "numpy<2"
# 1) TensorBoard/protobuf 지뢰 제거
#   - 너가 봤던 'Descriptors cannot be created directly' 고정 해결
python -m pip uninstall -y protobuf || true
python -m pip install "protobuf==3.20.3"

# 2) OpenCV 지뢰 제거
#   - opencv-python + headless 같이 있으면 DictValue/typing mismatch로 터짐
# python -m pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python || true
# # GUI 필요 없으면 headless가 안전
# python -m pip install --no-cache-dir "opencv-python-headless==4.8.1.78"
## char-s03에서는 GUI 필요해서 일반 버전 설치
python -m pip uninstall -y numpy opencv-python opencv-python-headless opencv-contrib-python
python -m pip install "numpy<2"
python -m pip install --no-cache-dir opencv-python-headless

# 3) CuPy 지뢰 제거 + 단일 설치 (너가 방금 해결한 루트)
## if char-s03 (cuda 12.x)
python -m pip uninstall -y cupy cupy-cuda11x cupy-cuda12x
python -m pip install --no-cache-dir --index-url https://pypi.org/simple "cupy-cuda12x"
## else
python -m pip uninstall -y cupy cupy-cuda11x cupy-cuda12x cupy-cuda118 || true
python -m pip install --no-cache-dir --index-url https://pypi.org/simple "cupy-cuda11x"

# 4) PyTorch3D (너가 말한대로 필요)
#    - 설치 방식은 환경마다 달라서, "pip install pytorch3d"가 성공했던 그 방식 그대로 둠.
python -m pip install --no-cache-dir pytorch3d

## if docker by docker.io/chacorp/audio2face:1.0
## not yet

