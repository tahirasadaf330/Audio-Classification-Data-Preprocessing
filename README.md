# Audio-Classification-Data-Preprocessing
### Let's read a sample audio using librosa
import librosa
audio_file_path='UrbanSound8K/100263-2-0-3.wav'
librosa_audio_data,librosa_sample_rate=librosa.load(audio_file_path)
print(librosa_audio_data)
[ 0.00330576  0.00468387  0.00360455 ... -0.0037562  -0.00348641
 -0.00356705]
### Lets plot the librosa audio data
import matplotlib.pyplot as plt
# Original audio with 1 channel 
plt.figure(figsize=(12, 4))
plt.plot(librosa_audio_data)
[<matplotlib.lines.Line2D at 0x1dac2ffc250>]

Observation
Here Librosa converts the signal to mono, meaning the channel will alays be 1

### Lets read with scipy
from scipy.io import wavfile as wav
wave_sample_rate, wave_audio = wav.read(audio_file_path) 
wave_audio
array([[ 194,  100],
       [ 179,  113],
       [ 160,  124],
       ...,
       [-143,  -87],
       [-134,  -91],
       [-110,  -98]], dtype=int16)
import matplotlib.pyplot as plt

# Original audio with 2 channels 
plt.figure(figsize=(12, 4))
plt.plot(wave_audio)
[<matplotlib.lines.Line2D at 0x1db383b0310>,
 <matplotlib.lines.Line2D at 0x1db383a31c0>]

Extract Features
Here we will be using Mel-Frequency Cepstral Coefficients(MFCC) from the audio samples. The MFCC summarises the frequency distribution across the window size, so it is possible to analyse both the frequency and time characteristics of the sound. These audio representations will allow us to identify features for classification.

mfccs = librosa.feature.mfcc(y=librosa_audio_data, sr=librosa_sample_rate, n_mfcc=40)
print(mfccs.shape)
(40, 173)
mfccs
array([[-4.45197296e+02, -4.47219299e+02, -4.49755127e+02, ...,
        -4.77412781e+02, -4.74241730e+02, -4.82704987e+02],
       [ 1.12513969e+02,  1.11970383e+02,  1.12244164e+02, ...,
         1.12045395e+02,  1.12248581e+02,  1.05560913e+02],
       [-1.58260956e+01, -2.30021858e+01, -3.12500191e+01, ...,
        -9.15441895e+00, -1.03232269e+01, -7.39410734e+00],
       ...,
       [-7.82766485e+00, -5.03880405e+00, -4.48165369e+00, ...,
        -1.90692782e-01,  4.34143972e+00,  1.00339069e+01],
       [-1.91763425e+00, -8.02737713e-01, -1.20930457e+00, ...,
        -1.23640239e-01,  2.90504694e-02,  9.22017097e-01],
       [-3.88130605e-01,  3.09317827e-01,  6.72155714e+00, ...,
        -2.33736587e+00, -4.25179911e+00, -2.31322765e+00]], dtype=float32)
#### Extracting MFCC's For every audio file
import pandas as pd
import os
import librosa

audio_dataset_path='UrbanSound8K/audio/'
metadata=pd.read_csv('UrbanSound8K/metadata/UrbanSound8K.csv')
metadata.head()
slice_file_name	fsID	start	end	salience	fold	classID	class
0	100032-3-0-0.wav	100032	0.0	0.317551	1	5	3	dog_bark
1	100263-2-0-117.wav	100263	58.5	62.500000	1	5	2	children_playing
2	100263-2-0-121.wav	100263	60.5	64.500000	1	5	2	children_playing
3	100263-2-0-126.wav	100263	63.0	67.000000	1	5	2	children_playing
4	100263-2-0-137.wav	100263	68.5	72.500000	1	5	2	children_playing
def features_extractor(file):
    audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast') 
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    
    return mfccs_scaled_features
    
import numpy as np
from tqdm import tqdm
### Now we iterate through every audio file and extract features 
### using Mel-Frequency Cepstral Coefficients
extracted_features=[]
for index_num,row in tqdm(metadata.iterrows()):
    file_name = os.path.join(os.path.abspath(audio_dataset_path),'fold'+str(row["fold"])+'/',str(row["slice_file_name"]))
    final_class_labels=row["class"]
    data=features_extractor(file_name)
    extracted_features.append([data,final_class_labels])
3554it [02:08, 27.26it/s]C:\Users\win10\anaconda3\envs\tensorflow\lib\site-packages\librosa\core\spectrum.py:222: UserWarning: n_fft=2048 is too small for input signal of length=1323
  warnings.warn(
8323it [04:54, 38.45it/s]C:\Users\win10\anaconda3\envs\tensorflow\lib\site-packages\librosa\core\spectrum.py:222: UserWarning: n_fft=2048 is too small for input signal of length=1103
  warnings.warn(
C:\Users\win10\anaconda3\envs\tensorflow\lib\site-packages\librosa\core\spectrum.py:222: UserWarning: n_fft=2048 is too small for input signal of length=1523
  warnings.warn(
8732it [05:08, 28.33it/s]
### converting extracted_features to Pandas dataframe
extracted_features_df=pd.DataFrame(extracted_features,columns=['feature','class'])
extracted_features_df.head()
feature	class
0	[-215.79301, 71.66612, -131.81377, -52.091335,...	dog_bark
1	[-424.68677, 110.56227, -54.148235, 62.01074, ...	children_playing
2	[-459.56467, 122.800354, -47.92471, 53.265697,...	children_playing
3	[-414.55377, 102.896904, -36.66495, 54.18041, ...	children_playing
4	[-447.397, 115.0954, -53.809113, 61.60859, 1.6...	children_playing
### Split the dataset into independent and dependent dataset
X=np.array(extracted_features_df['feature'].tolist())
y=np.array(extracted_features_df['class'].tolist())
X.shape
(8732, 40)
y
array(['dog_bark', 'children_playing', 'children_playing', ...,
       'car_horn', 'car_horn', 'car_horn'], dtype='<U16')
### Label Encoding
###y=np.array(pd.get_dummies(y))
### Label Encoder
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
y=to_categorical(labelencoder.fit_transform(y))
y
array([[0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 1., ..., 0., 0., 0.],
       [0., 0., 1., ..., 0., 0., 0.],
       ...,
       [0., 1., 0., ..., 0., 0., 0.],
       [0., 1., 0., ..., 0., 0., 0.],
       [0., 1., 0., ..., 0., 0., 0.]], dtype=float32)
### Train Test Split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
X_train
array([[[-1.3183614e+02,  1.1397464e+02, -2.3956861e+01, ...,
          3.3314774e+00, -1.4786109e+00,  2.8736601e+00]],

       [[-1.4074220e+01,  9.1916939e+01, -8.6787214e+00, ...,
         -3.3844023e+00, -5.2119045e+00, -1.5936136e+00]],

       [[-4.9532028e+01,  1.5521857e-01, -2.0369110e+01, ...,
          2.0491767e+00, -8.0537486e-01,  2.7793026e+00]],

       ...,

       [[-4.2699332e+02,  9.2890656e+01,  3.0233388e+00, ...,
          8.6335975e-01,  6.4766806e-01,  7.8490508e-01]],

       [[-1.4607024e+02,  1.3709459e+02, -3.4298344e+01, ...,
          1.3777871e+00, -1.9530845e+00, -8.9652127e-01]],

       [[-4.2167450e+02,  2.1169032e+02,  2.6820304e+00, ...,
         -5.1484952e+00, -3.6400862e+00, -1.3321606e+00]]], dtype=float32)
y
array([[0., 0., 0., ..., 0., 0., 0.],
       [0., 0., 1., ..., 0., 0., 0.],
       [0., 0., 1., ..., 0., 0., 0.],
       ...,
       [0., 1., 0., ..., 0., 0., 0.],
       [0., 1., 0., ..., 0., 0., 0.],
       [0., 1., 0., ..., 0., 0., 0.]], dtype=float32)
X_train.shape
(6985, 1, 40)
X_test.shape
(1747, 1, 40)
y_train.shape
(6985, 10)
y_test.shape
(1747, 10)
Model Creation
import tensorflow as tf
print(tf.__version__)
2.3.1
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten
from tensorflow.keras.optimizers import Adam
from sklearn import metrics
### No of classes
num_labels=y.shape[1]
 
model=Sequential()
###first layer
model.add(Dense(100,input_shape=(40,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
###second layer
model.add(Dense(200))
model.add(Activation('relu'))
model.add(Dropout(0.5))
###third layer
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dropout(0.5))

###final layer
model.add(Dense(num_labels))
model.add(Activation('softmax'))
model.summary()
Model: "sequential_6"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
dense_8 (Dense)              (None, 100)               4100      
_________________________________________________________________
activation_8 (Activation)    (None, 100)               0         
_________________________________________________________________
dropout_10 (Dropout)         (None, 100)               0         
_________________________________________________________________
dense_9 (Dense)              (None, 200)               20200     
_________________________________________________________________
activation_9 (Activation)    (None, 200)               0         
_________________________________________________________________
dropout_11 (Dropout)         (None, 200)               0         
_________________________________________________________________
dense_10 (Dense)             (None, 100)               20100     
_________________________________________________________________
activation_10 (Activation)   (None, 100)               0         
_________________________________________________________________
dropout_12 (Dropout)         (None, 100)               0         
_________________________________________________________________
dense_11 (Dense)             (None, 10)                1010      
_________________________________________________________________
activation_11 (Activation)   (None, 10)                0         
=================================================================
Total params: 45,410
Trainable params: 45,410
Non-trainable params: 0
_________________________________________________________________
model.compile(loss='categorical_crossentropy',metrics=['accuracy'],optimizer='adam')
## Trianing my model
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime 

num_epochs = 100
num_batch_size = 32

checkpointer = ModelCheckpoint(filepath='saved_models/audio_classification.hdf5', 
                               verbose=1, save_best_only=True)
start = datetime.now()

model.fit(X_train, y_train, batch_size=num_batch_size, epochs=num_epochs, validation_data=(X_test, y_test), callbacks=[checkpointer], verbose=1)


duration = datetime.now() - start
print("Training completed in time: ", duration)
Epoch 1/100
195/219 [=========================>....] - ETA: 0s - loss: 0.8612 - accuracy: 0.7104
Epoch 00001: val_loss improved from inf to 0.65317, saving model to saved_models\audio_classification.hdf5
219/219 [==============================] - 1s 3ms/step - loss: 0.8681 - accuracy: 0.7111 - val_loss: 0.6532 - val_accuracy: 0.7985
Epoch 2/100
204/219 [==========================>...] - ETA: 0s - loss: 0.8822 - accuracy: 0.7145
Epoch 00002: val_loss did not improve from 0.65317
219/219 [==============================] - 1s 2ms/step - loss: 0.8762 - accuracy: 0.7151 - val_loss: 0.6665 - val_accuracy: 0.7951
Epoch 3/100
200/219 [==========================>...] - ETA: 0s - loss: 0.8449 - accuracy: 0.7205
Epoch 00003: val_loss did not improve from 0.65317
219/219 [==============================] - 1s 2ms/step - loss: 0.8508 - accuracy: 0.7187 - val_loss: 0.6554 - val_accuracy: 0.7939
Epoch 4/100
217/219 [============================>.] - ETA: 0s - loss: 0.8584 - accuracy: 0.7154
Epoch 00004: val_loss did not improve from 0.65317
219/219 [==============================] - 1s 3ms/step - loss: 0.8576 - accuracy: 0.7158 - val_loss: 0.6656 - val_accuracy: 0.7939
Epoch 5/100
199/219 [==========================>...] - ETA: 0s - loss: 0.8835 - accuracy: 0.7112
Epoch 00005: val_loss improved from 0.65317 to 0.64595, saving model to saved_models\audio_classification.hdf5
219/219 [==============================] - 1s 2ms/step - loss: 0.8786 - accuracy: 0.7110 - val_loss: 0.6459 - val_accuracy: 0.8025
Epoch 6/100
200/219 [==========================>...] - ETA: 0s - loss: 0.8576 - accuracy: 0.7125
Epoch 00006: val_loss did not improve from 0.64595
219/219 [==============================] - 1s 2ms/step - loss: 0.8576 - accuracy: 0.7137 - val_loss: 0.6783 - val_accuracy: 0.7825
Epoch 7/100
209/219 [===========================>..] - ETA: 0s - loss: 0.8470 - accuracy: 0.7134
Epoch 00007: val_loss did not improve from 0.64595
219/219 [==============================] - 1s 2ms/step - loss: 0.8490 - accuracy: 0.7145 - val_loss: 0.6630 - val_accuracy: 0.7974
Epoch 8/100
197/219 [=========================>....] - ETA: 0s - loss: 0.8379 - accuracy: 0.7216
Epoch 00008: val_loss did not improve from 0.64595
219/219 [==============================] - 1s 2ms/step - loss: 0.8455 - accuracy: 0.7188 - val_loss: 0.6497 - val_accuracy: 0.7956
Epoch 9/100
198/219 [==========================>...] - ETA: 0s - loss: 0.8697 - accuracy: 0.7154
Epoch 00009: val_loss did not improve from 0.64595
219/219 [==============================] - 1s 2ms/step - loss: 0.8667 - accuracy: 0.7187 - val_loss: 0.6869 - val_accuracy: 0.7808
Epoch 10/100
201/219 [==========================>...] - ETA: 0s - loss: 0.8670 - accuracy: 0.7161
Epoch 00010: val_loss did not improve from 0.64595
219/219 [==============================] - 1s 2ms/step - loss: 0.8746 - accuracy: 0.7142 - val_loss: 0.6925 - val_accuracy: 0.7859
Epoch 11/100
203/219 [==========================>...] - ETA: 0s - loss: 0.8604 - accuracy: 0.7127
Epoch 00011: val_loss did not improve from 0.64595
219/219 [==============================] - 1s 2ms/step - loss: 0.8606 - accuracy: 0.7131 - val_loss: 0.6616 - val_accuracy: 0.7945
Epoch 12/100
202/219 [==========================>...] - ETA: 0s - loss: 0.8768 - accuracy: 0.7092
Epoch 00012: val_loss did not improve from 0.64595
219/219 [==============================] - 1s 2ms/step - loss: 0.8711 - accuracy: 0.7118 - val_loss: 0.6737 - val_accuracy: 0.7859
Epoch 13/100
212/219 [============================>.] - ETA: 0s - loss: 0.8552 - accuracy: 0.7165
Epoch 00013: val_loss did not improve from 0.64595
219/219 [==============================] - 1s 3ms/step - loss: 0.8502 - accuracy: 0.7180 - val_loss: 0.6771 - val_accuracy: 0.7865
Epoch 14/100
202/219 [==========================>...] - ETA: 0s - loss: 0.8634 - accuracy: 0.7157 ETA: 0s - loss: 0.9098 - 
Epoch 00014: val_loss did not improve from 0.64595
219/219 [==============================] - 1s 3ms/step - loss: 0.8627 - accuracy: 0.7161 - val_loss: 0.6584 - val_accuracy: 0.7997
Epoch 15/100
217/219 [============================>.] - ETA: 0s - loss: 0.8316 - accuracy: 0.7288
Epoch 00015: val_loss did not improve from 0.64595
219/219 [==============================] - 1s 3ms/step - loss: 0.8324 - accuracy: 0.7283 - val_loss: 0.6641 - val_accuracy: 0.7985
Epoch 16/100
218/219 [============================>.] - ETA: 0s - loss: 0.8472 - accuracy: 0.7130
Epoch 00016: val_loss did not improve from 0.64595
219/219 [==============================] - 1s 3ms/step - loss: 0.8475 - accuracy: 0.7130 - val_loss: 0.6584 - val_accuracy: 0.7956
Epoch 17/100
206/219 [===========================>..] - ETA: 0s - loss: 0.8656 - accuracy: 0.7113
Epoch 00017: val_loss did not improve from 0.64595
219/219 [==============================] - 1s 3ms/step - loss: 0.8611 - accuracy: 0.7128 - val_loss: 0.6759 - val_accuracy: 0.7899
Epoch 18/100
210/219 [===========================>..] - ETA: 0s - loss: 0.8740 - accuracy: 0.7103
Epoch 00018: val_loss did not improve from 0.64595
219/219 [==============================] - 1s 3ms/step - loss: 0.8751 - accuracy: 0.7101 - val_loss: 0.6775 - val_accuracy: 0.7808
Epoch 19/100
201/219 [==========================>...] - ETA: 0s - loss: 0.8594 - accuracy: 0.7096
Epoch 00019: val_loss did not improve from 0.64595
219/219 [==============================] - 1s 3ms/step - loss: 0.8608 - accuracy: 0.7095 - val_loss: 0.6550 - val_accuracy: 0.7979
Epoch 20/100
208/219 [===========================>..] - ETA: 0s - loss: 0.8610 - accuracy: 0.7163
Epoch 00020: val_loss did not improve from 0.64595
219/219 [==============================] - 1s 3ms/step - loss: 0.8633 - accuracy: 0.7157 - val_loss: 0.6733 - val_accuracy: 0.7831
Epoch 21/100
205/219 [===========================>..] - ETA: 0s - loss: 0.8408 - accuracy: 0.7166
Epoch 00021: val_loss improved from 0.64595 to 0.63746, saving model to saved_models\audio_classification.hdf5
219/219 [==============================] - 1s 3ms/step - loss: 0.8351 - accuracy: 0.7185 - val_loss: 0.6375 - val_accuracy: 0.7997
Epoch 22/100
205/219 [===========================>..] - ETA: 0s - loss: 0.8541 - accuracy: 0.7160
Epoch 00022: val_loss did not improve from 0.63746
219/219 [==============================] - 1s 3ms/step - loss: 0.8561 - accuracy: 0.7177 - val_loss: 0.6606 - val_accuracy: 0.7928
Epoch 23/100
203/219 [==========================>...] - ETA: 0s - loss: 0.8313 - accuracy: 0.7211
Epoch 00023: val_loss did not improve from 0.63746
219/219 [==============================] - 1s 3ms/step - loss: 0.8325 - accuracy: 0.7207 - val_loss: 0.6728 - val_accuracy: 0.7876
Epoch 24/100
208/219 [===========================>..] - ETA: 0s - loss: 0.8581 - accuracy: 0.7127
Epoch 00024: val_loss did not improve from 0.63746
219/219 [==============================] - 1s 3ms/step - loss: 0.8604 - accuracy: 0.7122 - val_loss: 0.6573 - val_accuracy: 0.7939
Epoch 25/100
203/219 [==========================>...] - ETA: 0s - loss: 0.8498 - accuracy: 0.7171
Epoch 00025: val_loss did not improve from 0.63746
219/219 [==============================] - 1s 3ms/step - loss: 0.8547 - accuracy: 0.7161 - val_loss: 0.6677 - val_accuracy: 0.7945
Epoch 26/100
204/219 [==========================>...] - ETA: 0s - loss: 0.8525 - accuracy: 0.7137
Epoch 00026: val_loss did not improve from 0.63746
219/219 [==============================] - 1s 3ms/step - loss: 0.8574 - accuracy: 0.7121 - val_loss: 0.6515 - val_accuracy: 0.7968
Epoch 27/100
193/219 [=========================>....] - ETA: 0s - loss: 0.8123 - accuracy: 0.7275
Epoch 00027: val_loss did not improve from 0.63746
219/219 [==============================] - 1s 2ms/step - loss: 0.8133 - accuracy: 0.7278 - val_loss: 0.6505 - val_accuracy: 0.7985
Epoch 28/100
192/219 [=========================>....] - ETA: 0s - loss: 0.8330 - accuracy: 0.7217
Epoch 00028: val_loss did not improve from 0.63746
219/219 [==============================] - 0s 2ms/step - loss: 0.8352 - accuracy: 0.7241 - val_loss: 0.6416 - val_accuracy: 0.7916
Epoch 29/100
184/219 [========================>.....] - ETA: 0s - loss: 0.8469 - accuracy: 0.7227
Epoch 00029: val_loss did not improve from 0.63746
219/219 [==============================] - 0s 2ms/step - loss: 0.8559 - accuracy: 0.7208 - val_loss: 0.6691 - val_accuracy: 0.7956
Epoch 30/100
195/219 [=========================>....] - ETA: 0s - loss: 0.8480 - accuracy: 0.7220
Epoch 00030: val_loss did not improve from 0.63746
219/219 [==============================] - 0s 2ms/step - loss: 0.8494 - accuracy: 0.7205 - val_loss: 0.6511 - val_accuracy: 0.7985
Epoch 31/100
218/219 [============================>.] - ETA: 0s - loss: 0.8415 - accuracy: 0.7241
Epoch 00031: val_loss did not improve from 0.63746
219/219 [==============================] - 0s 2ms/step - loss: 0.8423 - accuracy: 0.7238 - val_loss: 0.6773 - val_accuracy: 0.7865
Epoch 32/100
198/219 [==========================>...] - ETA: 0s - loss: 0.8698 - accuracy: 0.7159
Epoch 00032: val_loss did not improve from 0.63746
219/219 [==============================] - 0s 2ms/step - loss: 0.8672 - accuracy: 0.7157 - val_loss: 0.6709 - val_accuracy: 0.8014
Epoch 33/100
209/219 [===========================>..] - ETA: 0s - loss: 0.8221 - accuracy: 0.7267
Epoch 00033: val_loss did not improve from 0.63746
219/219 [==============================] - 0s 2ms/step - loss: 0.8202 - accuracy: 0.7278 - val_loss: 0.6516 - val_accuracy: 0.7934
Epoch 34/100
209/219 [===========================>..] - ETA: 0s - loss: 0.8538 - accuracy: 0.7129
Epoch 00034: val_loss did not improve from 0.63746
219/219 [==============================] - 0s 2ms/step - loss: 0.8513 - accuracy: 0.7134 - val_loss: 0.6676 - val_accuracy: 0.7831
Epoch 35/100
207/219 [===========================>..] - ETA: 0s - loss: 0.8232 - accuracy: 0.7261
Epoch 00035: val_loss did not improve from 0.63746
219/219 [==============================] - 0s 2ms/step - loss: 0.8249 - accuracy: 0.7241 - val_loss: 0.6485 - val_accuracy: 0.7934
Epoch 36/100
201/219 [==========================>...] - ETA: 0s - loss: 0.8576 - accuracy: 0.7256
Epoch 00036: val_loss did not improve from 0.63746
219/219 [==============================] - 0s 2ms/step - loss: 0.8573 - accuracy: 0.7256 - val_loss: 0.6669 - val_accuracy: 0.7991
Epoch 37/100
205/219 [===========================>..] - ETA: 0s - loss: 0.8619 - accuracy: 0.7159
Epoch 00037: val_loss did not improve from 0.63746
219/219 [==============================] - 0s 2ms/step - loss: 0.8619 - accuracy: 0.7152 - val_loss: 0.6549 - val_accuracy: 0.7962
Epoch 38/100
204/219 [==========================>...] - ETA: 0s - loss: 0.8647 - accuracy: 0.7149
Epoch 00038: val_loss did not improve from 0.63746
219/219 [==============================] - 0s 2ms/step - loss: 0.8648 - accuracy: 0.7138 - val_loss: 0.6578 - val_accuracy: 0.7951
Epoch 39/100
203/219 [==========================>...] - ETA: 0s - loss: 0.8547 - accuracy: 0.7191
Epoch 00039: val_loss did not improve from 0.63746
219/219 [==============================] - 0s 2ms/step - loss: 0.8483 - accuracy: 0.7211 - val_loss: 0.6557 - val_accuracy: 0.7899
Epoch 40/100
206/219 [===========================>..] - ETA: 0s - loss: 0.8328 - accuracy: 0.7260
Epoch 00040: val_loss did not improve from 0.63746
219/219 [==============================] - 0s 2ms/step - loss: 0.8334 - accuracy: 0.7254 - val_loss: 0.6708 - val_accuracy: 0.7939
Epoch 41/100
205/219 [===========================>..] - ETA: 0s - loss: 0.8501 - accuracy: 0.7305
Epoch 00041: val_loss did not improve from 0.63746
219/219 [==============================] - 0s 2ms/step - loss: 0.8539 - accuracy: 0.7303 - val_loss: 0.6781 - val_accuracy: 0.7876
Epoch 42/100
203/219 [==========================>...] - ETA: 0s - loss: 0.8375 - accuracy: 0.7212
Epoch 00042: val_loss did not improve from 0.63746
219/219 [==============================] - 0s 2ms/step - loss: 0.8341 - accuracy: 0.7225 - val_loss: 0.6541 - val_accuracy: 0.7922
Epoch 43/100
203/219 [==========================>...] - ETA: 0s - loss: 0.8805 - accuracy: 0.7135
Epoch 00043: val_loss did not improve from 0.63746
219/219 [==============================] - 0s 2ms/step - loss: 0.8812 - accuracy: 0.7130 - val_loss: 0.6730 - val_accuracy: 0.7871
Epoch 44/100
203/219 [==========================>...] - ETA: 0s - loss: 0.8272 - accuracy: 0.7244
Epoch 00044: val_loss did not improve from 0.63746
219/219 [==============================] - 0s 2ms/step - loss: 0.8360 - accuracy: 0.7213 - val_loss: 0.6572 - val_accuracy: 0.7997
Epoch 45/100
203/219 [==========================>...] - ETA: 0s - loss: 0.8382 - accuracy: 0.7275
Epoch 00045: val_loss did not improve from 0.63746
219/219 [==============================] - 0s 2ms/step - loss: 0.8393 - accuracy: 0.7258 - val_loss: 0.6731 - val_accuracy: 0.7831
Epoch 46/100
204/219 [==========================>...] - ETA: 0s - loss: 0.8668 - accuracy: 0.7206
Epoch 00046: val_loss did not improve from 0.63746
219/219 [==============================] - 0s 2ms/step - loss: 0.8627 - accuracy: 0.7207 - val_loss: 0.6580 - val_accuracy: 0.7974
Epoch 47/100
203/219 [==========================>...] - ETA: 0s - loss: 0.8457 - accuracy: 0.7231
Epoch 00047: val_loss did not improve from 0.63746
219/219 [==============================] - 0s 2ms/step - loss: 0.8417 - accuracy: 0.7247 - val_loss: 0.6430 - val_accuracy: 0.8031
Epoch 48/100
203/219 [==========================>...] - ETA: 0s - loss: 0.8513 - accuracy: 0.7197
Epoch 00048: val_loss did not improve from 0.63746
219/219 [==============================] - 0s 2ms/step - loss: 0.8518 - accuracy: 0.7200 - val_loss: 0.6690 - val_accuracy: 0.7911
Epoch 49/100
202/219 [==========================>...] - ETA: 0s - loss: 0.8445 - accuracy: 0.7170
Epoch 00049: val_loss did not improve from 0.63746
219/219 [==============================] - 0s 2ms/step - loss: 0.8441 - accuracy: 0.7155 - val_loss: 0.6702 - val_accuracy: 0.7962
Epoch 50/100
205/219 [===========================>..] - ETA: 0s - loss: 0.8285 - accuracy: 0.7238
Epoch 00050: val_loss did not improve from 0.63746
219/219 [==============================] - 0s 2ms/step - loss: 0.8301 - accuracy: 0.7238 - val_loss: 0.6537 - val_accuracy: 0.8002
Epoch 51/100
195/219 [=========================>....] - ETA: 0s - loss: 0.8354 - accuracy: 0.7216
Epoch 00051: val_loss did not improve from 0.63746
219/219 [==============================] - 0s 2ms/step - loss: 0.8293 - accuracy: 0.7237 - val_loss: 0.6630 - val_accuracy: 0.7853
Epoch 52/100
198/219 [==========================>...] - ETA: 0s - loss: 0.8501 - accuracy: 0.7169
Epoch 00052: val_loss did not improve from 0.63746
219/219 [==============================] - 0s 2ms/step - loss: 0.8453 - accuracy: 0.7173 - val_loss: 0.6826 - val_accuracy: 0.7991
Epoch 53/100
202/219 [==========================>...] - ETA: 0s - loss: 0.8609 - accuracy: 0.7203
Epoch 00053: val_loss did not improve from 0.63746
219/219 [==============================] - 0s 2ms/step - loss: 0.8599 - accuracy: 0.7198 - val_loss: 0.6525 - val_accuracy: 0.8037
Epoch 54/100
203/219 [==========================>...] - ETA: 0s - loss: 0.8557 - accuracy: 0.7197
Epoch 00054: val_loss did not improve from 0.63746
219/219 [==============================] - 0s 2ms/step - loss: 0.8534 - accuracy: 0.7203 - val_loss: 0.6805 - val_accuracy: 0.7888
Epoch 55/100
202/219 [==========================>...] - ETA: 0s - loss: 0.8395 - accuracy: 0.7189
Epoch 00055: val_loss did not improve from 0.63746
219/219 [==============================] - 0s 2ms/step - loss: 0.8408 - accuracy: 0.7181 - val_loss: 0.6568 - val_accuracy: 0.7899
Epoch 56/100
205/219 [===========================>..] - ETA: 0s - loss: 0.8526 - accuracy: 0.7155
Epoch 00056: val_loss did not improve from 0.63746
219/219 [==============================] - 0s 2ms/step - loss: 0.8530 - accuracy: 0.7152 - val_loss: 0.6866 - val_accuracy: 0.7802
Epoch 57/100
195/219 [=========================>....] - ETA: 0s - loss: 0.8334 - accuracy: 0.7266
Epoch 00057: val_loss did not improve from 0.63746
219/219 [==============================] - 0s 2ms/step - loss: 0.8468 - accuracy: 0.7234 - val_loss: 0.6768 - val_accuracy: 0.7934
Epoch 58/100
204/219 [==========================>...] - ETA: 0s - loss: 0.8520 - accuracy: 0.7108
Epoch 00058: val_loss did not improve from 0.63746
219/219 [==============================] - 0s 2ms/step - loss: 0.8527 - accuracy: 0.7114 - val_loss: 0.6717 - val_accuracy: 0.7796
Epoch 59/100
202/219 [==========================>...] - ETA: 0s - loss: 0.8256 - accuracy: 0.7200
Epoch 00059: val_loss did not improve from 0.63746
219/219 [==============================] - 0s 2ms/step - loss: 0.8240 - accuracy: 0.7215 - val_loss: 0.6396 - val_accuracy: 0.7979
Epoch 60/100
202/219 [==========================>...] - ETA: 0s - loss: 0.8444 - accuracy: 0.7186
Epoch 00060: val_loss did not improve from 0.63746
219/219 [==============================] - 0s 2ms/step - loss: 0.8435 - accuracy: 0.7185 - val_loss: 0.6671 - val_accuracy: 0.7853
Epoch 61/100
198/219 [==========================>...] - ETA: 0s - loss: 0.8411 - accuracy: 0.7208
Epoch 00061: val_loss did not improve from 0.63746
219/219 [==============================] - 0s 2ms/step - loss: 0.8385 - accuracy: 0.7214 - val_loss: 0.6507 - val_accuracy: 0.7956
Epoch 62/100
201/219 [==========================>...] - ETA: 0s - loss: 0.8107 - accuracy: 0.7253
Epoch 00062: val_loss did not improve from 0.63746
219/219 [==============================] - 0s 2ms/step - loss: 0.8144 - accuracy: 0.7234 - val_loss: 0.6449 - val_accuracy: 0.7934
Epoch 63/100
201/219 [==========================>...] - ETA: 0s - loss: 0.8268 - accuracy: 0.7236
Epoch 00063: val_loss did not improve from 0.63746
219/219 [==============================] - 0s 2ms/step - loss: 0.8257 - accuracy: 0.7223 - val_loss: 0.6439 - val_accuracy: 0.7928
Epoch 64/100
203/219 [==========================>...] - ETA: 0s - loss: 0.8484 - accuracy: 0.7226
Epoch 00064: val_loss did not improve from 0.63746
219/219 [==============================] - 0s 2ms/step - loss: 0.8507 - accuracy: 0.7221 - val_loss: 0.6560 - val_accuracy: 0.7865
Epoch 65/100
200/219 [==========================>...] - ETA: 0s - loss: 0.8181 - accuracy: 0.7273
Epoch 00065: val_loss did not improve from 0.63746
219/219 [==============================] - 0s 2ms/step - loss: 0.8217 - accuracy: 0.7253 - val_loss: 0.6398 - val_accuracy: 0.8042
Epoch 66/100
203/219 [==========================>...] - ETA: 0s - loss: 0.8513 - accuracy: 0.7163
Epoch 00066: val_loss did not improve from 0.63746
219/219 [==============================] - 0s 2ms/step - loss: 0.8534 - accuracy: 0.7165 - val_loss: 0.6594 - val_accuracy: 0.7928
Epoch 67/100
201/219 [==========================>...] - ETA: 0s - loss: 0.8443 - accuracy: 0.7215
Epoch 00067: val_loss did not improve from 0.63746
219/219 [==============================] - 0s 2ms/step - loss: 0.8491 - accuracy: 0.7223 - val_loss: 0.6499 - val_accuracy: 0.7979
Epoch 68/100
204/219 [==========================>...] - ETA: 0s - loss: 0.8429 - accuracy: 0.7221
Epoch 00068: val_loss did not improve from 0.63746
219/219 [==============================] - 0s 2ms/step - loss: 0.8416 - accuracy: 0.7220 - val_loss: 0.6479 - val_accuracy: 0.8014
Epoch 69/100
201/219 [==========================>...] - ETA: 0s - loss: 0.8248 - accuracy: 0.7282
Epoch 00069: val_loss did not improve from 0.63746
219/219 [==============================] - 0s 2ms/step - loss: 0.8232 - accuracy: 0.7284 - val_loss: 0.6405 - val_accuracy: 0.7951
Epoch 70/100
204/219 [==========================>...] - ETA: 0s - loss: 0.8146 - accuracy: 0.7319
Epoch 00070: val_loss did not improve from 0.63746
219/219 [==============================] - 0s 2ms/step - loss: 0.8194 - accuracy: 0.7307 - val_loss: 0.6506 - val_accuracy: 0.7899
Epoch 71/100
206/219 [===========================>..] - ETA: 0s - loss: 0.8531 - accuracy: 0.7163
Epoch 00071: val_loss did not improve from 0.63746
219/219 [==============================] - 0s 2ms/step - loss: 0.8517 - accuracy: 0.7173 - val_loss: 0.6490 - val_accuracy: 0.7825
Epoch 72/100
202/219 [==========================>...] - ETA: 0s - loss: 0.8405 - accuracy: 0.7280
Epoch 00072: val_loss did not improve from 0.63746
219/219 [==============================] - 0s 2ms/step - loss: 0.8348 - accuracy: 0.7283 - val_loss: 0.6536 - val_accuracy: 0.7911
Epoch 73/100
206/219 [===========================>..] - ETA: 0s - loss: 0.8441 - accuracy: 0.7191
Epoch 00073: val_loss did not improve from 0.63746
219/219 [==============================] - 0s 2ms/step - loss: 0.8466 - accuracy: 0.7181 - val_loss: 0.6672 - val_accuracy: 0.7939
Epoch 74/100
194/219 [=========================>....] - ETA: 0s - loss: 0.8330 - accuracy: 0.7202
Epoch 00074: val_loss improved from 0.63746 to 0.63103, saving model to saved_models\audio_classification.hdf5
219/219 [==============================] - 0s 2ms/step - loss: 0.8300 - accuracy: 0.7210 - val_loss: 0.6310 - val_accuracy: 0.8014
Epoch 75/100
203/219 [==========================>...] - ETA: 0s - loss: 0.8187 - accuracy: 0.7284
Epoch 00075: val_loss did not improve from 0.63103
219/219 [==============================] - 0s 2ms/step - loss: 0.8256 - accuracy: 0.7280 - val_loss: 0.6533 - val_accuracy: 0.7956
Epoch 76/100
203/219 [==========================>...] - ETA: 0s - loss: 0.8171 - accuracy: 0.7238
Epoch 00076: val_loss did not improve from 0.63103
219/219 [==============================] - 0s 2ms/step - loss: 0.8164 - accuracy: 0.7251 - val_loss: 0.6613 - val_accuracy: 0.7865
Epoch 77/100
205/219 [===========================>..] - ETA: 0s - loss: 0.8135 - accuracy: 0.7255
Epoch 00077: val_loss did not improve from 0.63103
219/219 [==============================] - 0s 2ms/step - loss: 0.8131 - accuracy: 0.7268 - val_loss: 0.6385 - val_accuracy: 0.7974
Epoch 78/100
205/219 [===========================>..] - ETA: 0s - loss: 0.8600 - accuracy: 0.7194
Epoch 00078: val_loss did not improve from 0.63103
219/219 [==============================] - 0s 2ms/step - loss: 0.8575 - accuracy: 0.7204 - val_loss: 0.6542 - val_accuracy: 0.7905
Epoch 79/100
208/219 [===========================>..] - ETA: 0s - loss: 0.8590 - accuracy: 0.7174
Epoch 00079: val_loss did not improve from 0.63103
219/219 [==============================] - 0s 2ms/step - loss: 0.8559 - accuracy: 0.7193 - val_loss: 0.6639 - val_accuracy: 0.7979
Epoch 80/100
207/219 [===========================>..] - ETA: 0s - loss: 0.8260 - accuracy: 0.7304
Epoch 00080: val_loss did not improve from 0.63103
219/219 [==============================] - 0s 2ms/step - loss: 0.8238 - accuracy: 0.7306 - val_loss: 0.6398 - val_accuracy: 0.8037
Epoch 81/100
202/219 [==========================>...] - ETA: 0s - loss: 0.8149 - accuracy: 0.7277
Epoch 00081: val_loss did not improve from 0.63103
219/219 [==============================] - 0s 2ms/step - loss: 0.8144 - accuracy: 0.7276 - val_loss: 0.6517 - val_accuracy: 0.7939
Epoch 82/100
204/219 [==========================>...] - ETA: 0s - loss: 0.8272 - accuracy: 0.7276
Epoch 00082: val_loss did not improve from 0.63103
219/219 [==============================] - 0s 2ms/step - loss: 0.8228 - accuracy: 0.7301 - val_loss: 0.6603 - val_accuracy: 0.8002
Epoch 83/100
205/219 [===========================>..] - ETA: 0s - loss: 0.8486 - accuracy: 0.7232
Epoch 00083: val_loss did not improve from 0.63103
219/219 [==============================] - 0s 2ms/step - loss: 0.8469 - accuracy: 0.7237 - val_loss: 0.6476 - val_accuracy: 0.7876
Epoch 84/100
198/219 [==========================>...] - ETA: 0s - loss: 0.8148 - accuracy: 0.7301
Epoch 00084: val_loss did not improve from 0.63103
219/219 [==============================] - 0s 2ms/step - loss: 0.8140 - accuracy: 0.7294 - val_loss: 0.6360 - val_accuracy: 0.8054
Epoch 85/100
217/219 [============================>.] - ETA: 0s - loss: 0.8342 - accuracy: 0.7229
Epoch 00085: val_loss did not improve from 0.63103
219/219 [==============================] - 0s 2ms/step - loss: 0.8345 - accuracy: 0.7228 - val_loss: 0.6597 - val_accuracy: 0.7911
Epoch 86/100
189/219 [========================>.....] - ETA: 0s - loss: 0.8554 - accuracy: 0.7174
Epoch 00086: val_loss did not improve from 0.63103
219/219 [==============================] - 0s 2ms/step - loss: 0.8502 - accuracy: 0.7208 - val_loss: 0.6347 - val_accuracy: 0.7951
Epoch 87/100
205/219 [===========================>..] - ETA: 0s - loss: 0.8410 - accuracy: 0.7239
Epoch 00087: val_loss did not improve from 0.63103
219/219 [==============================] - 0s 2ms/step - loss: 0.8375 - accuracy: 0.7238 - val_loss: 0.6399 - val_accuracy: 0.7968
Epoch 88/100
206/219 [===========================>..] - ETA: 0s - loss: 0.8426 - accuracy: 0.7221
Epoch 00088: val_loss did not improve from 0.63103
219/219 [==============================] - 0s 2ms/step - loss: 0.8396 - accuracy: 0.7231 - val_loss: 0.6475 - val_accuracy: 0.7939
Epoch 89/100
201/219 [==========================>...] - ETA: 0s - loss: 0.8247 - accuracy: 0.7284
Epoch 00089: val_loss did not improve from 0.63103
219/219 [==============================] - 0s 2ms/step - loss: 0.8270 - accuracy: 0.7248 - val_loss: 0.6569 - val_accuracy: 0.7934
Epoch 90/100
202/219 [==========================>...] - ETA: 0s - loss: 0.8377 - accuracy: 0.7232
Epoch 00090: val_loss did not improve from 0.63103
219/219 [==============================] - 0s 2ms/step - loss: 0.8452 - accuracy: 0.7213 - val_loss: 0.6453 - val_accuracy: 0.7985
Epoch 91/100
203/219 [==========================>...] - ETA: 0s - loss: 0.8140 - accuracy: 0.7306
Epoch 00091: val_loss did not improve from 0.63103
219/219 [==============================] - 0s 2ms/step - loss: 0.8089 - accuracy: 0.7321 - val_loss: 0.6563 - val_accuracy: 0.7882
Epoch 92/100
212/219 [============================>.] - ETA: 0s - loss: 0.8402 - accuracy: 0.7214
Epoch 00092: val_loss did not improve from 0.63103
219/219 [==============================] - 0s 2ms/step - loss: 0.8373 - accuracy: 0.7218 - val_loss: 0.6331 - val_accuracy: 0.7956
Epoch 93/100
191/219 [=========================>....] - ETA: 0s - loss: 0.8389 - accuracy: 0.7214
Epoch 00093: val_loss did not improve from 0.63103
219/219 [==============================] - 0s 2ms/step - loss: 0.8300 - accuracy: 0.7237 - val_loss: 0.6370 - val_accuracy: 0.8042
Epoch 94/100
219/219 [==============================] - ETA: 0s - loss: 0.7966 - accuracy: 0.7344
Epoch 00094: val_loss did not improve from 0.63103
219/219 [==============================] - 0s 2ms/step - loss: 0.7966 - accuracy: 0.7344 - val_loss: 0.6461 - val_accuracy: 0.7894
Epoch 95/100
193/219 [=========================>....] - ETA: 0s - loss: 0.8113 - accuracy: 0.7268
Epoch 00095: val_loss did not improve from 0.63103
219/219 [==============================] - 0s 2ms/step - loss: 0.8056 - accuracy: 0.7287 - val_loss: 0.6393 - val_accuracy: 0.7991
Epoch 96/100
187/219 [========================>.....] - ETA: 0s - loss: 0.8098 - accuracy: 0.7296
Epoch 00096: val_loss did not improve from 0.63103
219/219 [==============================] - 0s 2ms/step - loss: 0.8195 - accuracy: 0.7281 - val_loss: 0.6522 - val_accuracy: 0.7916
Epoch 97/100
192/219 [=========================>....] - ETA: 0s - loss: 0.8216 - accuracy: 0.7297
Epoch 00097: val_loss did not improve from 0.63103
219/219 [==============================] - 0s 2ms/step - loss: 0.8288 - accuracy: 0.7288 - val_loss: 0.6529 - val_accuracy: 0.7985
Epoch 98/100
191/219 [=========================>....] - ETA: 0s - loss: 0.8121 - accuracy: 0.7281
Epoch 00098: val_loss did not improve from 0.63103
219/219 [==============================] - 0s 2ms/step - loss: 0.8153 - accuracy: 0.7273 - val_loss: 0.6450 - val_accuracy: 0.7934
Epoch 99/100
191/219 [=========================>....] - ETA: 0s - loss: 0.8151 - accuracy: 0.7307 ETA: 0s - loss: 0.8208 - accura
Epoch 00099: val_loss did not improve from 0.63103
219/219 [==============================] - 0s 2ms/step - loss: 0.8212 - accuracy: 0.7274 - val_loss: 0.6649 - val_accuracy: 0.7997
Epoch 100/100
202/219 [==========================>...] - ETA: 0s - loss: 0.8116 - accuracy: 0.7293
Epoch 00100: val_loss did not improve from 0.63103
219/219 [==============================] - 0s 2ms/step - loss: 0.8071 - accuracy: 0.7303 - val_loss: 0.6363 - val_accuracy: 0.8002
Training completed in time:  0:00:45.200242
test_accuracy=model.evaluate(X_test,y_test,verbose=0)
print(test_accuracy[1])
0.8002289533615112
prediction_feature.shape
(1, 40)
X_test[1]
array([-466.1843    ,    1.5388278 ,  -34.397358  ,   35.715336  ,
        -15.166929  ,  -18.850813  ,   -0.7415805 ,  -15.99989   ,
        -21.354332  ,    7.6506834 ,  -29.031452  ,  -19.142824  ,
         -2.6798913 ,   -8.466884  ,  -14.7660475 ,   -7.004778  ,
         -7.103754  ,    8.887754  ,   14.911873  ,   21.47102   ,
         21.336624  ,    0.9169518 ,  -18.795404  ,   -5.001721  ,
         -0.70152664,    2.91399   ,   -6.7105994 ,  -16.638536  ,
         -9.821647  ,   12.8619585 ,    0.6552978 ,  -23.953394  ,
        -15.200551  ,    9.21079   ,   10.419799  ,   -0.57916117,
         -1.2440346 ,   17.722294  ,   13.837573  ,   -5.164349  ],
      dtype=float32)
model.predict_classes(X_test)
array([5, 3, 4, ..., 1, 2, 2], dtype=int64)
Testing Some Test Audio Data
Steps

Preprocess the new audio data
predict the classes
Invere transform your Predicted Label
filename="UrbanSound8K/drilling_1.wav"
audio, sample_rate = librosa.load(filename, res_type='kaiser_fast') 
mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)

print(mfccs_scaled_features)
mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)
print(mfccs_scaled_features)
print(mfccs_scaled_features.shape)
predicted_label=model.predict_classes(mfccs_scaled_features)
print(predicted_label)
prediction_class = labelencoder.inverse_transform(predicted_label) 
prediction_class
[-146.34639      52.85859       6.0391283    46.972637      0.48288426
   31.756617     -6.395756     36.949165     -2.2981966     9.0149975
   -8.056831     24.668858    -14.41076       7.5845594    -3.089655
   17.423319    -10.068965      9.606158     -1.4731672     7.745292
   -1.9399884    -1.5998945     3.373213      1.6671567    -4.9514785
    4.8195934    -6.1473813     3.8730834   -10.502274      1.3417107
   -5.616546      4.815169     -6.152183      2.0756485    -1.8508396
   -0.45990178   -4.9980536     2.528911     -0.7446382    -6.4779253 ]
[[-146.34639      52.85859       6.0391283    46.972637      0.48288426
    31.756617     -6.395756     36.949165     -2.2981966     9.0149975
    -8.056831     24.668858    -14.41076       7.5845594    -3.089655
    17.423319    -10.068965      9.606158     -1.4731672     7.745292
    -1.9399884    -1.5998945     3.373213      1.6671567    -4.9514785
     4.8195934    -6.1473813     3.8730834   -10.502274      1.3417107
    -5.616546      4.815169     -6.152183      2.0756485    -1.8508396
    -0.45990178   -4.9980536     2.528911     -0.7446382    -6.4779253 ]]
(1, 40)
[4]
array(['drilling'], dtype='<U16')
