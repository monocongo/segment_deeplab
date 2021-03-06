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
    $ cd models/research
    $ export TFMODELS=`pwd`
    $ cd deeplab
    $ export DEEPLAB=`pwd`
    ```
3. Install additional libraries we'll use in our project (assumes that `conda-forge` 
is the primary channel):
    ```bash
    $ for pkg in opencv imutils tensorflow-gpu
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
    >>> import tensorflow
    >>>
    ```
5. Set the `models/research` directory into the `PYTHONPATH` environment variable:
    ```bash
    $ export PYTHONPATH=/home/ubuntu/git/models/research
    ```
### Training Dataset
Acquire a dataset of images and corresponding object segmentation masks. This project 
assumes a dataset with a directory of image files in JPG format and a corresponding 
directory of mask image files in PNG format matching to each image file. The mask 
files are expected to contain mask values corresponding to the class ID of the objects 
being masked, with unmasked/background having value 0. For example if we have two 
classes, dog and cat with class IDs 1 and 2 respectively, then all dog masks in a 
mask image will be denoted with value 1 and all cat masks will be denoted with value 2.

An example dataset that includes image mask files is the 
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
The above example execution will result in eight TFRecord files -- four TFRecords 
for the training set, comprising of 80% of the images/masks, and four TFRecords 
for the validation set, comprising of 20% of the images/masks.

Once we have the TFRecords for training and validation we then modify the file `$DEEPLAB/segmentation_dataset.py`
### Training

##### Pretrained model
Download a pretrained model checkpoint from the 
[DeepLab Model Zoo](https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md):
```bash
$ wget http://download.tensorflow.org/models/xception_65_coco_pretrained_2018_10_02.tar.gz
$ tar -zxf xception_65_coco_pretrained_2018_10_02.tar.gz
``` 

##### Training script
Run the [DeepLab training script](https://github.com/tensorflow/models/blob/master/research/deeplab/train.py) 
referencing the pretrained model checkpoint, local dataset directory, training 
log directory, etc. For example:
```bash
$ cd ${TFMODELS}
$ python deeplab/train.py \
    --logtostderr \
    --training_number_of_steps=30000 \
    --train_split="train" \
    --model_variant="xception_65" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --train_crop_size="513,513" \
    --train_batch_size=1 \
    --dataset="basins" \
    --tf_initial_checkpoint=/home/ubuntu/deeplab/pretrained/x65-b2u1s2p-d48-2-3x256-sc-cr300k_init.ckpt.data-00000-of-00001 \
    --train_logdir=${DEEPLAB}/datasets/basins/exp/train_on_train_set/train \
    --dataset_dir=${DEEPLAB}/datasets/basins
```

##### Evaluation script
Once the model has started training and has written some checkpoints then an evaluation 
can be preformed using the [DeepLab evaluation script](https://github.com/tensorflow/models/blob/master/research/deeplab/eval.py). 
For example:
```bash
$ cd ${TFMODELS}
$ python deeplab/eval.py \
    --logtostderr \
    --eval_split="val" \
    --model_variant="xception_65" \
    --atrous_rates=6 \
    --atrous_rates=12 \
    --atrous_rates=18 \
    --output_stride=16 \
    --decoder_output_stride=4 \
    --eval_crop_size="513,513" \
    --dataset="basins" \
    --checkpoint_dir=${DEEPLAB}/datasets/basins/exp/train_on_train_set/train \
    --eval_logdir=${DEEPLAB}/datasets/basins/exp/train_on_train_set/eval \
    --dataset_dir=${DEEPLAB}/datasets/basins
```
