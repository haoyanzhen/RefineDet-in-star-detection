# RefineDet-in-star-detection
> RefineDet reference from [luuuyi's RefineDet.PyTorch](https://github.com/luuuyi/RefineDet.PyTorch)

## RefineDet
Very grateful to luuuyi for providing RefineDet code in [PyTorch](https://pytorch.org/). [RefineDet](https://arxiv.org/abs/1711.06897) is an one-stage method for object detection. Here I hope it could work well in star detection of crowded field. The result shows it may work badly in crowded field detection, which means a wide range of sizes, simple features and high degree of overlap. Main changes to the code are focused on prior boxes design. A full-process visible [test code](./RefineDet/check_matchloss.py) is added to help realize and adjust network model. And an evalution code is designed for this task.

Dataset is uploading...

## cnn_starcount.ipynb
A test sample of using CNN to count amount of stars in one frame.

## dataset_generater.ipynb
This contains several dataset generater, which split one frame to several pieces, and generate other neccesary data based on catalog. (An ANN fitting is used.)

## statistic.ipynb
This contains some statistic for understanding crowded field features and prior boxes design.
