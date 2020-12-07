# FCN_for_Semantic_Segmentation
Fully Convolutional Neural Network based Semantic Segmentation for CityScape Dateset with Pytorch implementation
The model is based on CVPR '15 best paper honorable mentioned [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/pdf/1411.4038.pdf)

**Pre-requirements**
These dependencies can be installed using pip or conda (or anyother depending on your system)
1. Pytorch (with Torchvision)
2. matplotlib
3. Pandas
4. Scipy (=1.1.0)

This script achieves nearly 95% accuracy with GPU for CityScapes dataset.


**How to run**
1. Data setup:
  Download CityScapes dataset from the original website [here](https://www.cityscapes-dataset.com/downloads/). Then create a directory named "CityScapes", and put data into the directory
  Now run below to prepare and preprocess the data for the model.
  ```
  python3 Cityscapes_utils.py
  ```
 2. Then run Train.py script to run the training (one can change the hyperparameters inside the script).
 ```
 python3 train.py
 ```
