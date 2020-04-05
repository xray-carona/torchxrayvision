# %%

import random
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

# %%

import torchxrayvision as xrv

# %%

"""
    PadChest dataset
    Hospital San Juan de Alicante – University of Alicante

    Dataset website:
    http://bimcv.cipf.es/bimcv-projects/padchest/

    Download full size images here:
    https://academictorrents.com/details/dec12db21d57e158f78621f06dcbe78248d14850

    Download resized (224x224) images here:
    https://academictorrents.com/details/e0aeda79626589f31e8bf016660da801f5add88e
"""
# d_pc = xrv.datasets.PC_Dataset(imgpath="/home/ronald/xray_corona/flask_backend/data/test/")

# %%

# Load model
model = xrv.models.DenseNet(num_classes=18, weights_dir="model_path/")

# %%

# Dataset info
# d_pc

# %%

# # Sample output
# IMGIDX = random.randint(0, len(d_pc))
#
# print("Showing X-Ray of IDX: ", IMGIDX, )
# ximage = d_pc[IMGIDX]["PA"][0]
# plt.imshow(ximage, cmap="Greys_r")
# pcpathologies = d_pc.pathologies.copy()
#
# print("Labels:")
# for idx, pathology in enumerate(pcpathologies):
#     print(pathology, d_pc.labels[IMGIDX][idx])
#
# # %%
#
# pcpathologies = sorted(d_pc.pathologies.copy())
# default_pathologies = sorted(xrv.datasets.default_pathologies)
#
# for i in range(10):
#     imidx = random.randint(0, len(d_pc))
#     print(imidx)
#     inferimage = torch.from_numpy(np.expand_dims(d_pc[imidx]["PA"], axis=0))
#
#     conf = torch.nn.Softmax()(model(inferimage)).detach().numpy()[0]
#     pclabels = d_pc[imidx]["lab"].copy()
#
#     print("Pathology Prob GT")
#     for idx, pathology in enumerate(default_pathologies):
#         try:
#             print(pathology, "%.2f%%" % (conf[idx] * 100), pclabels[pcpathologies.index(pathology)])
#         except ValueError:
#             print(pathology, "%.2f%%" % (conf[idx] * 100), "NoLabel")
#
#     print("\n")
#
# # %%
import requests
from torchxrayvision.datasets import normalize
image_loc='https://raw.githubusercontent.com/xray-carona/data-modeling/master/data/test/person1949_bacteria_4880.jpeg'
img_resp = requests.get(image_loc, stream=True).raw

image = np.asarray(bytearray(img_resp.read()), dtype="uint8")
image=normalize(image,65535)

out=model(torch.from_numpy(image))
print(out)