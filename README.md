![avatar](framework.jpg)

## Event-based Monocular Dense Depth Estimation with Recurrent Transformers

#### Dependencies

PyTorch >= 1.0
NumPy
OpenCV
Matplotlib

### Setup
This code has been tested with Python 3.7.10, Torch 1.9.0, CUDA 10.2 on Ubuntu 16.04.

- Setup python environment
```
conda create -n EReFormer python=3.7.10
source activate EReFormer 
pip install -r requirements.txt
conda install -c pytorch pytorch=1.9.0 torchvision cudatoolkit=10.2
conda install -c conda-forge opencv
conda install -c conda-forge matplotlib
```
### Public Datasets

The Multi Vehicle Stereo Event Camera Dataset - MVSEC Dataset:
https://daniilidis-group.github.io/mvsec/

Depth Estimation oN Synthetic Events - DENSE Dataset: 
https://rpg.ifi.uzh.ch/E2DEPTH.html
  
### Start training

The model only supports fp32 and does not support low precision (e.g., fp16) acceleration.

```bash
cd EReFormer-main
python train.py --config configs/EReFormer_dense_epoch200_000032.json
```
  
## Publication
If you find our project helpful in your research, please cite with:
```
@article{liu2022event,
  title={Event-based Monocular Dense Depth Estimation with Recurrent Transformers},
  author={Liu, Xu and Li, Jianing and Fan, Xiaopeng and Tian, Yonghong},
  journal={arXiv preprint arXiv:2212.02791},
  year={2022}
}
```
