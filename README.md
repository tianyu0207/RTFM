# RTFM
This repo contains the Pytorch implementation of our paper:
> [**Weakly-supervised Video Anomaly Detection with Robust Temporal Feature Magnitude Learning**](https://arxiv.org/pdf/2101.10030.pdf)
---


## Training

### Setup

Please download the extracted I3d features for ShanghaiTech dataset from these two links:

> [**train i3d onedirve**](https://uao365-my.sharepoint.com/:f:/g/personal/a1697106_adelaide_edu_au/EiLi_oBQnAFCq3UG184p_akB2sV7szCWvOV9PtaKJ6lxtQ?e=MeM3TE)

> [**test i3d onedrive**](https://uao365-my.sharepoint.com/:f:/g/personal/a1697106_adelaide_edu_au/EvUUrWqpWqVHrXBzxbzAdD8BGiZBiumWWOaZmQ_AMAkAdg?e=P1rwCg)

> [**ShanghaiTech feature on Google dirve**](https://drive.google.com/drive/folders/1L71Qa0gao6aLVhSjL0H-u2khmTRKcmQs?usp=sharing, https://drive.google.com/drive/folders/1z-CQPpVtTyfZyPKZdv2hZ-h2oMF6s8ep?usp=sharing)

Extracted I3d features for Ucf-Crime: 

> [**train i3d onedirve**](https://uao365-my.sharepoint.com/:f:/g/personal/a1697106_adelaide_edu_au/ErCr6bjDzzZPstgposv1ttYBudL8UVnap6eHS46fFbooAQ?e=RZsMtA)

> [**test i3d onedrive**](https://uao365-my.sharepoint.com/:f:/g/personal/a1697106_adelaide_edu_au/EsmBEpklrShEjTFOWTd5FooBVXbeoDHTTqPZn60Vj3Guhg?e=hvv46w)

> [**checkpoint for Ucf-crime**](https://uao365-my.sharepoint.com/:u:/g/personal/a1697106_adelaide_edu_au/Ed0gS0RZ5hFMqVa8LxcO3sYBqFEmzMU5IsvvLWxioTatKw?e=qHEl5Z)


The following files need to be adapted in order to run the code on your own machine:
- Change the file paths to the download datasets above in `list/shanghai-i3d-test-10crop.list` and `list/shanghai-i3d-train-10crop.list` for ShanghaiTech, or `list/ucf-i3d.list` and `list/ucf-i3d-test.list` for Ucf-Crime. 
- Feel free to change the hyperparameters in `option.py`
### Train and test the RTFM
After the setup, simply run the following command: 
```shell
python main.py
```
---
