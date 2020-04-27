# Overview 

I have implemented scene action recognitions from UCF101 dataset which can be obtained from [here](https://www.crcv.ucf.edu/data/UCF101.php). 
The dataset being too big. You can download a smaller version of it like [UCF50](https://www.crcv.ucf.edu/data/UCF50.php) or [UCF11](https://www.crcv.ucf.edu/data/UCF_YouTube_Action.php).

# Data Preprocessing 
I converted videos into frames and took only 32 frames from every video for the training of model. 
These 16 frames were selected from complete video sequence by skipping frames according to video length. 

# Model used

[convlstm.py](https://github.com/iamrishab/Video-Classification-PyTorch/blob/master/models/convlstm.py)

[densenet.py](https://github.com/iamrishab/Video-Classification-PyTorch/blob/master/models/densenet.py)

[resnet.py](https://github.com/iamrishab/Video-Classification-PyTorch/blob/master/models/resnet.py)

[resnext.py](https://github.com/iamrishab/Video-Classification-PyTorch/blob/master/models/resnext.py)

[slowfast.py](https://github.com/iamrishab/Video-Classification-PyTorch/blob/master/models/slowfast.py)

[wide_resnet.py](https://github.com/iamrishab/Video-Classification-PyTorch/blob/master/models/wide_resnet.py)

One of the best model for action recognition [Slow Fast Networks for Video Recognition](https://arxiv.org/pdf/1812.03982.pdf) worked best. 
The implementation of this network in pytorch can be found [here](https://github.com/Guocode/SlowFast-Networks). 

# Training
$ `python train.py`
