# Dog-Breed-Prediction
Using VGG19 pre trained model to train on a dataset using transfer learning approach to predict different dog breeds.

[Dataset](https://www.kaggle.com/c/dog-breed-identification/data)

## Training
Use `train.py` to start the training process. It imports `load_data` function from the `datapreprocess.py` file.
Once the data is uploaded, split into training set and validation set.

The model is trained on a pre trained VGG19 model. The `input_shape=(256,256,3)` . I am freezing the first 5 layers of the model just so it doesn't waste time on low level features like edges etc. Here is a very well described link on [transfer learning](https://towardsdatascience.com/transfer-learning-using-keras-d804b2e04ef8) using keras.

I also trained the model on pre trained DenseNet model and accuracy improved from VGG19 model. The `input_shape = (224,224,3)` for Densenet model.

## Predictions

The accuracy of the VGG19 model on validation set was ~64%. The multiclass loss was 2.524
The accuracy of the DenseNet model on validation set was ~73%. The multiclass loss was 1.537
After training the model I am predicting on small test set. 

#### Actual: Yorkshire Terrier 
#### Predicted: Yorkshire Terrier

![alt text](https://github.com/raj-shah14/Dog-Breed-Prediction/blob/master/test%20set/00bbfaa5b2bff32a3dc8ce1563e484a3.jpg)
![alt text](https://github.com/raj-shah14/Dog-Breed-Prediction/blob/master/output/yorkshire_terrier_dog_7.jpg)

#### Actual: Cardigan
#### Predicted: Cardigan

![alt text](https://github.com/raj-shah14/Dog-Breed-Prediction/blob/master/test%20set/0b549d7b0a972428bfca5bec213ce494.jpg)
![alt text](https://github.com/raj-shah14/Dog-Breed-Prediction/blob/master/output/cardigan_dog_28.jpg)

#### Actual: Standard Poodle
#### Predicted: Standard Poodle

![alt text](https://github.com/raj-shah14/Dog-Breed-Prediction/blob/master/test%20set/00c6e480ca61e3d2da272d7b6bee0a9e.jpg)
![alt text](https://github.com/raj-shah14/Dog-Breed-Prediction/blob/master/output/standard_poodle_dog_10.jpg)

#### Actual: Japanese Spaniel
#### Predicted: Japanese Spaniel

![alt text](https://github.com/raj-shah14/Dog-Breed-Prediction/blob/master/test%20set/00bbbcb2bf285af6304bd4da0c10299e.jpg)
![alt text](https://github.com/raj-shah14/Dog-Breed-Prediction/blob/master/output/japanese_spaniel_dog_6.jpg)

#### Actual: Brabancon Griffon
#### Predicted: Brabancon Griffon

![alt text](https://github.com/raj-shah14/Dog-Breed-Prediction/blob/master/test%20set/00d6e16493e6af2886292fa8823bdfe0.jpg)
![alt text](https://github.com/raj-shah14/Dog-Breed-Prediction/blob/master/output/brabancon_griffon_dog_12.jpg)
