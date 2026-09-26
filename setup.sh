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
    pip install easydict h5py protobuf==3.20.0
elif [[ $MODE == "1" ]]; then
    pip install -r requirements.txt
    
    # # git clone https://github.com/MPI-IS/mesh.git
    # ## change Makefile:L7 -> @pip install --no-deps --verbose --no-cache-dir .
    # ## comment out requirements.txt -> # numpy pyopengl opencv-python
    # cd mesh && make all && cd ..
    ## ------------------ For the case when you have error installing pytorch3d ... ------------------
    # cp cpp_extension.py /usr/local/lib/python3.8/dist-packages/torch/utils/cpp_extension.py
    # pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.6"

    curl -sL -o _tmp/pytorch3d_wheel/pytorch3d-0.7.5-cp310-cp310-linux_x86_64.whl \
    "https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt210/pytorch3d-0.7.5-cp310-cp310-linux_x86_64.whl"
    pip install --force-reinstall --no-deps _tmp/pytorch3d_wheel/pytorch3d-0.7.5-cp310-cp310-linux_x86_64.whl
    ## ------------------------------------------------------------------------------------------
    pip install easydict h5py protobuf==3.20.0
elif [[ $MODE == "3" ]]; then
    pip install h5py
    pip install --upgrade setuptools wheel
    cp cpp_extension.py /opt/conda/lib/python3.10/site-packages/torch/utils/cpp_extension.py
    pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.8" --no-build-isolation
    #pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
    pip install cython gdist torch_geometric
else
    pip install torch_cluster -f https://data.pyg.org/whl/torch-2.1.0+cu121.html --no-cache-dir
    pip install torch_scatter -f https://data.pyg.org/whl/torch-2.1.0+cu121.html --no-cache-dir
    pip install torch_sparse -f https://data.pyg.org/whl/torch-2.1.0+cu121.html --no-cache-dir
    pip install -r requirements-cuda12.1.txt
fi