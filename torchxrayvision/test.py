import random
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision

import time
import torchxrayvision as xrv

from torchxrayvision.datasets import normalize


def normalize(sample, maxval=255): # maxval=65536
    """Scales images to be roughly [-1024 1024]."""
    sample = (2 * (sample.astype(np.float32) / maxval) - 1.) * 1024
    #sample = sample / np.std(sample)
    return sample

# First save image using requests. this can be from flask.
# check the flask image save directory, read the image here using plt.imread
ximage = plt.imread('img.jpeg')

model = xrv.models.DenseNet(num_classes=18, weights="all")

pathologies = ["Atelectasis", "Consolidation", "Infiltration",
                            "Pneumothorax", "Edema", "Emphysema", "Fibrosis",
                            "Effusion", "Pneumonia", "Pleural_Thickening",
                            "Cardiomegaly", "Nodule", "Mass", "Hernia","Fracture"]
#d_pc = xrv.datasets.PC_Dataset(imgpath="data/PC/images-224")

pcpathologies = sorted(pathologies.copy())

print("Labels:")
for idx, pathology in enumerate(pcpathologies):
    print(idx + 1, pathology)

#plt.imshow(ximage)
img = normalize(ximage)
disp = img

# Check that images are 2D arrays
if len(img.shape) > 2:
    img = img[:, :, 0]
if len(img.shape) < 2:
    print("error, dimension lower than 2 for image")

img = img[None, :, :]

inferimage = torch.from_numpy(np.expand_dims(img, axis=0))

conf = torch.nn.Softmax()(model(inferimage)).detach().numpy()[0]    
print(conf)


#plt.imshow(disp, cmap="Greys_r")
#plt.show()


