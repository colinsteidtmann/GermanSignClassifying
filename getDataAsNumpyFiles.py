import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import matplotlib.pyplot as plt
import cv2
from PIL import Image

# Save Train images
data=[]
labels=[]

for i in range(43) :
    path = "data/Train/{0}/".format(i)
    Class=os.listdir(path)
    for a in Class:
        try:
            image=cv2.imread(path+a)
            image_from_array = Image.fromarray(image, 'RGB')
            # The images are of different size
            # this quick hack resizes them to the same size
            size_image = image_from_array.resize((30, 30))
            data.append(np.array(size_image))
            labels.append(i)
        except AttributeError:
            print(" ")

Cells=np.array(data)
labels=np.array(labels)

np.save("trainCells",Cells)
np.save("trainLabels", labels)

