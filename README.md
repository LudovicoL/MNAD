# MNAD
An extended [MNAD](https://github.com/cvlab-yonsei/MNAD) implementation for AITEX dataset.

Original paper: [**Learning Memory-guided Normality for Anomaly Detection**](https://openaccess.thecvf.com/content_CVPR_2020/papers/Park_Learning_Memory-Guided_Normality_for_Anomaly_Detection_CVPR_2020_paper.pdf)

This implementation include a validation phase to calculate the best threshold for the anomaly score detection.

## How to use
To train with AITEX dataset:
```
python train.py -d aitex
```
To evaluate:
```
python evaluate.py
```

## Minimum requirements
Python 3.9 with PyTorch 1.9.0. Use the file ```environment.yml``` for the conda environment.

## Reference
[1] Hyunjong Park, Jongyoun Noh, Bumsub Ham. *Learning Memory-guided Normality for Anomaly Detection*. https://cvlab.yonsei.ac.kr/projects/MNAD/

[2] https://github.com/cvlab-yonsei/MNAD
