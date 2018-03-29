import cv2
import os
import pandas as pd
import glob
import numpy as np
from keras.utils import to_categorical

  
def load_data(path,breed,imgid):
    
    breedIndex = []
    for b in breed:
        if b not in breedIndex:
            breedIndex.append(b)
    breedIndex = sorted(breedIndex)

    encodedLabels = []
    for b in breed:
        encodedLabels.append(breedIndex.index(b))
    blabels = np.array(encodedLabels)
    blabels = to_categorical(blabels)

    features=[]
    for i in imgid:
        
        #Resizing images
        img=cv2.imread(path+"train/"+i+".jpg")
        img=cv2.resize(img,(256,256),0,0, cv2.INTER_LINEAR)
        img = img.astype(np.float32)
        img = np.multiply(img, 1.0 / 255.0)
        features.append(img)
    features = np.array(features)        
            
    return features,blabels,encodedLabels

  
        
    
    
