# -*- coding: utf-8 -*-
import cv2
import glob
import numpy as np
from keras.models import load_model
import pandas as pd
import re

labels=pd.read_csv('labels/labels.csv')

breed=labels['breed']
imgid=labels['id']

breedIndex = []
for b in breed:
    if b not in breedIndex:
        breedIndex.append(b)
breedIndex = sorted(breedIndex)


model=load_model("vgg_dogmodel-30-2.42.hdf5")
path="C:/Users/Raj Shah/Downloads/Dog Prediction/"

## To observe on a small test set
#j=0
#featurestest=[]
#for i in glob.glob(path+"small test/*.jpg"):
#
#    img=cv2.resize(cv2.imread(i),(256,256),0,0,cv2.INTER_LINEAR)
#    image=img
#    img = img.astype(np.float32)
#    img = np.multiply(img, 1.0 / 255.0)
#    featurestest.append(img)
#    imge = np.expand_dims(img, axis=0)
#    predictions = model.predict(imge)
#    bestPrediction = np.argmax(predictions)
#    prediction = breedIndex[bestPrediction]
#    #print(prediction)
#    cv2.putText(image,str(prediction), (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1) 
#    cv2.imshow(str(prediction),img)
#    image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
#    cv2.imwrite(path+"output/"+str(prediction)+"_dog_"+str(j)+".jpg",image)
#    j+=1
#    cv2.waitKey()
#    cv2.destroyAllWindows()   
#        
#featurestest=np.array(featurestest)

featurestest=[]
for i in glob.glob(path+"test/*.jpg"):

    img=cv2.resize(cv2.imread(i),(256,256),0,0,cv2.INTER_LINEAR)
    img = img.astype(np.float32)
    img = np.multiply(img, 1.0 / 255.0)
    featurestest.append(img)
    img = np.expand_dims(img, axis=0)
    #predictions = model.predict(img)
    #bestPrediction = np.argmax(predictions)
    #prediction = breedIndex[bestPrediction]
    
featurestest=np.array(featurestest)
predictions=model.predict(featurestest)


ids=[]
for i in glob.glob("test/*.jpg"):
    j=re.findall(r"[\w']+",i)
    ids.append(j[1])

columns = ["id"] + breedIndex
df = pd.DataFrame(columns=columns)
idDF = pd.DataFrame(ids, columns=["id"])

predictionsList = predictions.tolist()
predDF = pd.DataFrame(predictionsList,columns=breedIndex)

predDF['id'] = ids
predDF = predDF[['id']+breedIndex]

predDF.to_csv(path+"/submissions.csv",float_format='%.6f',index=False)