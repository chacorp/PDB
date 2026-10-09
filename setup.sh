MODE="1"
while [[ $# -gt 0 ]];
do
    case $1 in
        -m|--mode)
            MODE=$2
            shift
            shift
            ;;
        -*|--*)
            echo "Unknown option "$1
            exit 1
            ;;
            *)
    esac
done

# ln -s /data/sihun/NFR_data/data ./data
# ln -s /data/sihun/NFR_data/test-mesh/ ./test-mesh
# ln -s /data/sihun/NFR_data/experiments/ ./experiments
# ln -s /data/sihun/NFR_data/ict_face_pt/ ./ict_face_pt
# cd third_party && git clone https://github.com/vsitzmann/siren

if [[ $MODE == "0" ]]; then
    pip install git+https://github.com/chacorp/matplotrender.git # for visualization
    pip install easydict h5py protobuf==3.20.0
elif [[ $MODE == "1" ]]; then
    pip install -r requirements.txt
    pip install git+https://github.com/chacorp/matplotrender.git # for visualization
    
    # # git clone https://github.com/MPI-IS/mesh.git
    # ## change Makefile:L7 -> @pip install --no-deps --verbose --no-cache-dir .
    # ## comment out requirements.txt -> # numpy pyopengl opencv-python
    # cd mesh && make all && cd ..
    ## ------------------ For the case when you have error installing pytorch3d ... ------------------
    cp cpp_extension.py /usr/local/lib/python3.8/dist-packages/torch/utils/cpp_extension.py
    pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.6"
    ## ------------------------------------------------------------------------------------------
    pip install easydict h5py protobuf==3.20.0
elif [[ $MODE == "3" ]]; then
    pip install h5py
    pip install git+https://github.com/chacorp/matplotrender.git # for visualization
    pip install --upgrade setuptools wheel
    ## ------------------ For the case when you have error installing pytorch3d ... ------------------
    # cp cpp_extension.py /opt/conda/lib/python3.10/site-packages/torch/utils/cpp_extension.py
    # pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.8" --no-build-isolation
    #pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
    
    curl -sL -o _tmp/pytorch3d_wheel/pytorch3d-0.7.5-cp310-cp310-linux_x86_64.whl \
    "https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt210/pytorch3d-0.7.5-cp310-cp310-linux_x86_64.whl"
    pip install --force-reinstall --no-deps _tmp/pytorch3d_wheel/pytorch3d-0.7.5-cp310-cp310-linux_x86_64.whl
    ## ------------------------------------------------------------------------------------------
    pip install cython gdist torch_geometric
elif [[ $MODE == "4" ]]; then
    # Same target stack as default (Python 3.10, torch 2.1.0+cu121), but with
    # cupy pinned to 11.3 -- matching the official NFR reference repo
    # (github.com/dafei-qin/NFR_pytorch)'s tested combo:
    #   mamba install pytorch=1.12.1 cudatoolkit=11.3 pytorch-sparse=0.6.15 \
    #     pytorch3d=0.7.1 cupy=11.3 numpy=1.23.5 ...
    # requirements-cuda12.1.txt pins cupy==13.4.0 (a cupy-cuda12x build), which
    # intermittently hits CUDA_ERROR_ILLEGAL_ADDRESS inside
    # utils/deformation_transfer.py's torch<->cupy DLPack handoff (a race
    # between PyTorch's and CuPy's CUDA streams) when running NFR/NFS eval at
    # scale. cupy-cuda12x has no 11.x builds below 11.5, so this installs
    # cupy-cuda11x==11.3.0 instead -- it bundles its own CUDA 11.3 runtime, so
    # it runs fine under a CUDA 12.1 driver (NVIDIA drivers are backward
    # compatible) alongside the rest of the CUDA-12.1-built stack.
    pip install git+https://github.com/chacorp/matplotrender.git # for visualization
    pip install torch_cluster -f https://data.pyg.org/whl/torch-2.1.0+cu121.html --no-cache-dir
    pip install torch_scatter -f https://data.pyg.org/whl/torch-2.1.0+cu121.html --no-cache-dir
    pip install torch_sparse -f https://data.pyg.org/whl/torch-2.1.0+cu121.html --no-cache-dir
    pip install -r requirements-cuda12.1.txt
    pip uninstall -y cupy cupy-cuda12x cupy-cuda11x
    pip install --no-deps cupy-cuda11x==11.3.0
    ## pytorch3d, same wheel as MODE 3 (needed for NFR/NFS's on-the-fly rendering)
    curl -sL -o _tmp/pytorch3d_wheel/pytorch3d-0.7.5-cp310-cp310-linux_x86_64.whl \
    "https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt210/pytorch3d-0.7.5-cp310-cp310-linux_x86_64.whl"
    pip install --force-reinstall --no-deps _tmp/pytorch3d_wheel/pytorch3d-0.7.5-cp310-cp310-linux_x86_64.whl
else
    pip install git+https://github.com/chacorp/matplotrender.git # for visualization
    pip install torch_cluster -f https://data.pyg.org/whl/torch-2.1.0+cu121.html --no-cache-dir
    pip install torch_scatter -f https://data.pyg.org/whl/torch-2.1.0+cu121.html --no-cache-dir
    pip install torch_sparse -f https://data.pyg.org/whl/torch-2.1.0+cu121.html --no-cache-dir
    pip install -r requirements-cuda12.1.txt
fi