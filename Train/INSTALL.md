# Installation

## Requirements
- Python >= 3.7
- Numpy
- PyTorch >= 1.9 (Acceleration for 3D depth-wise convolution)
- [fvcore](https://github.com/facebookresearch/fvcore/): `pip install 'git+https://github.com/facebookresearch/fvcore'`
- [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  You can install them together at [pytorch.org](https://pytorch.org) to make sure of this.
- simplejson: `pip install simplejson`
- GCC >= 4.9
- PyAV: `conda install av -c conda-forge`
- ffmpeg (4.0 is prefereed, will be installed along with PyAV)
- PyYaml: (will be installed along with fvcore)
- tqdm: (will be installed along with fvcore)
- iopath: `pip install -U iopath` or `conda install -c iopath iopath`
- psutil: `pip install psutil`
- OpenCV: `pip install opencv-python`
- torchvision: `pip install torchvision` or `conda install torchvision -c pytorch`
- tensorboard: `pip install tensorboard`
- moviepy: (optional, for visualizing video on tensorboard) `conda install -c conda-forge moviepy` or `pip install moviepy`
- PyTorchVideo: `pip install pytorchvideo`
- Decord: `pip install decord`
- detectron2
```
 git clone https://github.com/facebookresearch/detectron2 detectron2_repo
 pip install -e detectron2_repo
```

## Install
After having the above dependencies, run:
```
git clone https://github.com/RongchangLi/AICity2023_DrivingAction.git
cd AICity2023_DrivingAction/
cd Train/
```
You can run `python setup.py build develop` to build slowfast 

or run `export PYTHONPATH=./slowfast:$PYTHONPATH`to add this repository to $PYTHONPATH.


**Now you can use the environment to run the training and inference codes.**