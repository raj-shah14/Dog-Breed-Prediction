
# coding: utf-8

# In[1]:


import cv2
import os
import pandas as pd
import glob
import numpy as np
from keras.utils import to_categorical


# In[3]:


labels=pd.read_csv('labels/labels.csv')

breed=labels['breed']
imgid=labels['id']


# In[4]:


def load_data(breed,imgid):

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
        img=cv2.imread("train/"+i+".jpg")
        img=cv2.resize(img,(256,256),0,0, cv2.INTER_LINEAR)
        img = img.astype(np.float32)
        img = np.multiply(img, 1.0 / 255.0)
        features.append(img)
    features = np.array(features)        
            
    return features,blabels,encodedLabels


# In[5]:


features,labels,encodedLabels=load_data(breed,imgid) 


# In[9]:


features.shape


# In[8]:


from sklearn import cross_validation
featureTrain, featureTest, labelTrain, labelTest = cross_validation.train_test_split(features,labels, test_size= 0.20)


# In[11]:


featureTrain.shape


# In[12]:


from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping


# In[25]:


img_width, img_height = 256, 256
batch_size = 32
epochs = 30


# In[14]:


model = applications.VGG19(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))


# In[15]:


# Freeze the layers which you don't want to train. Here I am freezing the first 5 layers.
for layer in model.layers[:5]:
    layer.trainable = False


# In[16]:


#Adding custom Layers 
x = model.output
x = Flatten()(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.2)(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.2)(x)
predictions = Dense(120, activation="softmax")(x)


# In[17]:


# creating the final model 
model_final = Model(input = model.input, output = predictions)


# In[28]:


# compile the model 
model_final.compile(loss = "categorical_crossentropy", optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), metrics=["accuracy"])


# In[19]:


model_final.summary()


# In[23]:


directory='weights'
filepath = '{0}/vgg_dogmodel-{{epoch:02d}}-{{val_loss:.2f}}.hdf5'.format(directory)                                              


# In[27]:


# Save the model according to the conditions  
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=2)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=1, mode='auto')


# In[ ]:


model_final.fit(featureTrain,labelTrain,
          batch_size=batch_size,epochs=epochs,
           validation_data=(featureTest,labelTest),
           callbacks=[checkpoint,early])

