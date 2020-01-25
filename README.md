# segment_deeplab
This project describes the use of TensorFlow's 
[DeepLab v3](https://github.com/tensorflow/models/tree/master/research/deeplab) 
for semantic segmentation.

### Prerequisites
We assume a Linux development environment running on Ubuntu 18.04. We assume that 
the training of the model will be performed on a system with an NVIDIA GPU, and 
as such we need to have CUDA and cuDNN installed. 

1. Install [CUDA](https://developer.nvidia.com/cuda-toolkit):
    ```bash
    $ wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
    $ sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
    $ sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
    $ sudo add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
    $ sudo apt-get update
    $ sudo apt-get -y install cuda
    ```
2. Install [cuDNN](https://developer.nvidia.com/rdp/cudnn-download):
    1. Login to the [NVIDIA Developer Network](https://developer.nvidia.com)
    2. Download the [cuDNN Runtime Library for Ubuntu18.04](https://developer.nvidia.com/compute/machine-learning/cudnn/secure/7.6.5.32/Production/10.2_20191118/Ubuntu18_04-x64/libcudnn7_7.6.5.32-1%2Bcuda10.2_amd64.deb)
    ```
    $ sudo apt install ./libcudnn7_7.6.5.32-1+cuda10.2_amd64.deb
    ```
3. Install [OpenCV](https://opencv.org/):
    ```
    $ sudo apt-get install libopencv-dev python3-opencv
    ```   
### Python Environment
1. Create a new Python virtual environment:
    ```bash
    $ conda config --add channels conda-forge
    $ conda create -n deeplab python=3 --yes
    $ conda activate deeplab
    ```
2. Get the TensorFlow DeepLab API:
    ```bash
    $ git clone https://github.com/tensorflow/models.git
    $ cd models/research/deeplab
    $ export DEEPLAB=`pwd`
    ```
3. Install additional libraries we'll use in our project (assumes that `conda-forge` 
is the primary channel):
    ```bash
    $ for pkg in opencv imutils imgaug tensorflow-gpu
    > do
    > conda install $pkg --yes
    > done
    $ pip install cvdata
    ```
4. Verify the installation:
    ```bash
    $ python
    >>> import cv2
    >>> import cvdata
    >>> import imutils
    >>> import imgaug
    >>> import tensorflow
    >>>
    ```

### Training Dataset
Acquire a dataset of images and corresponding object segmentation masks. This project 
assumes a dataset with a directory of image files in JPG format and a corresponding 
directory of mask image files in PNG format matching to each image file.

A good example dataset that includes image mask files is the 
[ISIC 2018 Skin Lesion Analysis Dataset](https://challenge2018.isic-archive.com/).

In order to convert the dataset of images and masks into TFRecords, which is the 
data format used for training data, we'll use the [cvdata](https://pypi.org/project/cvdata/) 
package's `cvdata_mask` entry point:
```bash
$ cvdata_mask --images /data/images --masks /data/masks \
>       --in_format png --out_format tfrecord \
>       --tfrecords /data/tfrecords \
>       --shards 4 -- train_pct 0.8
```

### Training

