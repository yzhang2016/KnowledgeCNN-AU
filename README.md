# KnowledgeCNN-AU
Implementation of the multi-label version of [Weakly-Supervised CNN Learning for Facial Action Unit Intensity Estimation](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_Weakly-Supervised_Deep_Convolutional_CVPR_2018_paper.pdf), CVPR 2018.

# Environments
* python 
* tensorflow

# Usage
1. Collect data 
* images: images are collected from the videos.
* tuples: detect the intensity peaks and valleys first and then sample tuples from each segment.
2. Training 
```
python ShollowCNN_AUModel_weak_train.py
```
3. Testing
```
python ShollowCNN_AUModel_eval_weak.py
```
