# torchxrayvision

A library for chest X-ray datasets and models. Including pre-trainined models.

This code is still under development

## Getting started

```
pip install torchxrayvision

import torchxrayvision as xrv
```

These are default pathologies:
```
xrv.datasets.default_pathologies 

['Atelectasis',
 'Consolidation',
 'Infiltration',
 'Pneumothorax',
 'Edema',
 'Emphysema',
 'Fibrosis',
 'Effusion',
 'Pneumonia',
 'Pleural_Thickening',
 'Cardiomegaly',
 'Nodule',
 'Mass',
 'Hernia',
 'Lung Lesion',
 'Fracture',
 'Lung Opacity',
 'Enlarged Cardiomediastinum']
```

## models

Specify weights for pretrained models (currently all DenseNet121)
Note: Each pretrained model has 18 outputs. The `all` model has every output trained. However, for the other weights some targets are not trained and will predict randomly becuase they do not exist in the training dataset. The only valid outputs are listed in the field `{dataset}.pathologies` on the dataset that corresponds to the weights. 

```
model = xrv.models.DenseNet(weights="all")
model = xrv.models.DenseNet(weights="kaggle")
model = xrv.models.DenseNet(weights="nih")
model = xrv.models.DenseNet(weights="chex")
model = xrv.models.DenseNet(weights="minix_nb")
model = xrv.models.DenseNet(weights="minix_ch")

```


## datasets

```
transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),
                                            xrv.datasets.XRayResizer(224)])

d_kaggle = xrv.datasets.Kaggle_Dataset(imgpath="path to stage_2_train_images_jpg",
                                       transform=transform)
                
d_chex = xrv.datasets.CheX_Dataset(imgpath="path to CheXpert-v1.0-small",
                                   csvpath="path to CheXpert-v1.0-small/train.csv",
                                   transform=transform)

d_nih = xrv.datasets.NIH_Dataset(imgpath="path to NIH images")

d_nih2 = xrv.datasets.NIH_Google_Dataset(imgpath="path to NIH images")

d_pc = xrv.datasets.PC_Dataset(imgpath="path to image folder")


d_covid19 = xrv.datasets.COVID19_Dataset() # specify imgpath and csvpath for the dataset
```

## inference

Download one of the following datasets for evaluation:

NIH: https://academictorrents.com/details/e615d3aebce373f1dc8bd9d11064da55bdadede0

Kaggle: https://academictorrents.com/details/95588a735c9ae4d123f3ca408e56570409bcf2a9

NIH_Google: https://pubs.rsna.org/doi/10.1148/radiol.2019191293

PC: https://academictorrents.com/details/e0aeda79626589f31e8bf016660da801f5add88e

CheX: https://stanfordmlgroup.github.io/competitions/chexpert/

MIMIC: https://physionet.org/content/mimic-cxr-jpg/2.0.0/

Open_i: https://academictorrents.com/details/5a3a439df24931f410fac269b87b050203d9467d

COVID-19: https://github.com/ieee8023/covid-chestxray-dataset

Inference Notebook: test.ipynb

Example:

To infer on the PadChest dataset:

`d_pc = xrv.datasets.PC_Dataset(imgpath="data/PC/images-224")`



## dataset tools

relabel_dataset will align labels to have the same order as the pathologies argument.
```
xrv.datasets.relabel_dataset(xrv.datasets.default_pathologies , d_nih) # has side effects
```

## Citation

```
Joseph Paul Cohen, Joseph Viviano, Mohammad Hashir, and Hadrien Bertrand. 
TorchXrayVision: A library of chest X-ray datasets and models. 
https://github.com/mlmed/torchxrayvision, 2020
```
and
```
Cohen, J. P., Hashir, M., Brooks, R., & Bertrand, H. 
On the limits of cross-domain generalization in automated X-ray prediction. 2020 
arXiv preprint [https://arxiv.org/abs/2002.02497](https://arxiv.org/abs/2002.02497)

@article{cohen2020limits,
  title={On the limits of cross-domain generalization in automated X-ray prediction},
  author={Cohen, Joseph Paul and Hashir, Mohammad and Brooks, Rupert and Bertrand, Hadrien},
  journal={arXiv preprint arXiv:2002.02497},
  year={2020}
}
```
