# RTFM
This repo contains the Pytorch implementation of our paper:
> [**Weakly-supervised Video Anomaly Detection with Robust Temporal Feature Magnitude Learning**](https://arxiv.org/pdf/2101.10030.pdf)
>
> [Yu Tian](https://yutianyt.com/), [Guansong Pang](https://sites.google.com/site/gspangsite/home?authuser=0), Yuanhong Chen, Rajvinder Singh, Johan W. Verjans, [Gustavo Carneiro](https://cs.adelaide.edu.au/~carneiro/).

- **Accepted at ICCV 2021.**  

- **SOTA on 4 benchmarks.** Check out [**Papers With Code**](https://paperswithcode.com/paper/weakly-supervised-video-anomaly-detection) for [**Video Anomaly Detection**](https://paperswithcode.com/task/anomaly-detection-in-surveillance-videos). 


## Training

### Setup

**Please download the extracted I3d features for ShanghaiTech and UCF-Crime dataset from links below:**

> [**ShanghaiTech train i3d onedirve**](https://uao365-my.sharepoint.com/:f:/g/personal/a1697106_adelaide_edu_au/EiLi_oBQnAFCq3UG184p_akBLDBVdCqRNCzSDhbqpjFQXw?e=hBAexc)
> 
> [**ShanghaiTech test i3d onedrive**](https://uao365-my.sharepoint.com/:f:/g/personal/a1697106_adelaide_edu_au/EvUUrWqpWqVHrXBzxbzAdD8BlgH1SICKQbmdVu7K5nR9xA?e=oWTk8G)
> 
> [**ShanghaiTech features on Google dirve**](https://drive.google.com/file/d/1-w9xsx2FbwFf96A1y1GFcZ3odzdEBves/view?usp=sharing)
> 
> [**checkpoint for ShanghaiTech**](https://drive.google.com/file/d/1epISwbTZ_LXKfJzfYVIVwnxQ6q49lj5B/view?usp=sharing)

**Extracted I3d features for UCF-Crime dataset**

> [**UCF-Crime train i3d onedirve**](https://uao365-my.sharepoint.com/:f:/g/personal/a1697106_adelaide_edu_au/ErCr6bjDzzZPstgposv1ttYBjv_ZBsAbNTbwyl3yX8QCHA?e=BzNuJ2)
> 
> [**UCF-Crime test i3d onedrive**](https://uao365-my.sharepoint.com/:f:/g/personal/a1697106_adelaide_edu_au/EsmBEpklrShEjTFOWTd5FooBkJR3DPxp3cIZN-R8b2hhLA?e=hlcZFO)
> 
> [**UCF-Crime train I3d features on Google drive**](https://drive.google.com/file/d/16LumirTnWOOu8_Uh7fcC7RWpSBFobDUA/view?usp=sharing)
> 
> [**UCF-Crime test I3d features on Google drive**](https://drive.google.com/drive/folders/1QCBTDUMBXYU9PonPh1TWnRtpTKOX-fxr?usp=sharing)
> 
> [**checkpoint for Ucf-crime**](https://uao365-my.sharepoint.com/:u:/g/personal/a1697106_adelaide_edu_au/Ed0gS0RZ5hFMqVa8LxcO3sYBqFEmzMU5IsvvLWxioTatKw?e=qHEl5Z)

The above features use the resnet50 I3D to extract from this [**repo**](https://github.com/Tushar-N/pytorch-resnet3d).

Follow previous works, we also apply 10-crop augmentations. 

The following files need to be adapted in order to run the code on your own machine:
- Change the file paths to the download datasets above in `list/shanghai-i3d-test-10crop.list` and `list/shanghai-i3d-train-10crop.list`.
- Feel free to change the hyperparameters in `option.py`
### Train and test the RTFM
After the setup, simply run the following commands: 
```shell
python -m visdom.server
python main.py
```


## Citation

If you find this repo useful for your research, please consider citing our paper:

```bibtex
@article{tian2021weakly,
  title={Weakly-supervised Video Anomaly Detection with Robust Temporal Feature Magnitude Learning},
  author={Tian, Yu and Pang, Guansong and Chen, Yuanhong and Singh, Rajvinder and Verjans, Johan W and Carneiro, Gustavo},
  journal={arXiv preprint arXiv:2101.10030},
  year={2021}
}
```
---
