# Object Detection for Graphical User Interface: Old Fashioned or Deep Learning or a Combination?

**Accepted to FSE2020**

*This repository includes all code/models in our paper, namely Faster RCNN, YOLO v3, CenterNet, Xianyu, REMAUI and our model*

- Paper: Coming soon


## Environment Setup

### FASTER RCNN

``
cd FASTER_RCNN
pip install -r requirements.txt
rm lib/build/*
cd lib & python setup.py build develop
``

### YOLOv3

Coming soon


### CenterNet

```
cd CenterNet-master
pip install -r requirements.txt
rm models/py_utils/_cpools/build/*
cd models/py_utils/_cpools & python setup.py install --user
```

### Xianyu

Coming soon


### REMAUI

Coming soon


### Our model

Coming soon


## Testing




## ACKNOWNLEDGES

The implementations of Faster RCNN, YOLO v3, CenterNet and REMAUI are based on the following GitHub Repositories. Thank for the works.

- Faster RCNN: https://github.com/jwyang/faster-rcnn.pytorch/tree/pytorch-1.0

- YOLOv3: https://github.com/eriklindernoren/PyTorch-YOLOv3

- CenterNet: https://github.com/Duankaiwen/CenterNet

- REMAUI: https://github.com/soumikmohianuta/pixtoapp

We implement Xianyu based on their technical blog

- XianYu: https://laptrinhx.com/ui2code-how-to-fine-tune-background-and-foreground-analysis-2293652041/
