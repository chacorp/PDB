# apt update && apt upgrade ffmpeg -y

# wget https://github.com/mmatl/travis_debs/raw/master/xenial/mesa_18.3.3-0.deb
# dpkg -i ./mesa_18.3.3-0.deb || true
# apt install -f

apt-get install -y llvm-6.0 freeglut3 freeglut3-dev libosmesa6-dev
pip install pyrender ffmpeg-python
pip install pyopengl==3.1.4

## for mitsuba renderer
# apt-get update -y && apt-get install -y libnvidia-gl-535 ffmpeg && pip install ffmpeg-python mitsuba numpy==1.26.4