#!/usr/bin/env python
# coding: utf-8

''' 
Machine Learning Speech Classification Task by Ebiere 
''' 

#import modules
import numpy as np 
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.callbacks import EarlyStopping
from keras import layers
from keras import models
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing

# load file
df_train = pd.read_csv('/train.csv')
df_test = pd.read_csv('/test.csv')
feat = np.load('/feat.npy',allow_pickle=True)
path = np.load('/path.npy')


# find the six unused features and their paths 
t = feat
ind =[]
for i in range(len(t)):
    if t[i].shape[0] > 99:
        ind.append(i)
ind
#[67391, 67392, 67393, 67394, 67395, 67396] 

#delete the unused features and paths 
feat2 =list(feat) 
del feat2[67391:67396+1]
len(feat2)

path2 = list(path)
del path2[67391:67396+1]
len(path2)


## to align feature/path/train/test files 
# merge train and test path
path_merge = list(df_train['path'])+list(df_test['path'])
print(len(path_merge)) #105829
#sort paths 
path_merge3 = sorted(path_merge)
print(len(path_merge3))

# reorganize data in train and test csv files to line up with merged paths
word_train =[] 
path_train =[] 
for i in range(len(path_merge3)):
    if path_merge3[i] in list(df_train['path']):
        index = list(df_train['path']).index(path_merge3[i])
        word_train.append(df_train['word'][index])
        path_train.append(path_merge3[i] )

# to line features with the new merged path 
def feature_sorting(path):
    sorted_path = []
    for i in range(len(path)):
        if path[i] in path2:
            index = path2.index(path[i])
            sorted_path.append(feat2[index])
    return sorted_path
 
#train data
feature_train = feature_sorting(path_train)
#test data
feature_test = feature_sorting(df_test['path'])

## feature engineer

# to zero pad to reshape features into fixed size vectors of (99,13)
def padded(feature):
    padded = [] 
    for i in range(len(feature)):
        result = np.zeros((99,13))
        result[:feature[i].shape[0],:feature[i].shape[1]]=feature[i]
        padded.append(result)
    return padded
    
padded_train = np.array(padded(feature_train))
padded_test = np.array(padded(feature_test))

# to rescale data 
def standardization(data):
    scaled_features = [] 
    scaler = StandardScaler( )
    for i in data:
        scaled_feature = scaler.fit_transform(i)
        scaled_features.append(scaled_feature)
    return scaled_features
X= np.array(standardization(padded_train))
x_test = standardization(padded_test)


#dummie coding
dummies = pd.get_dummies(word_train)
y = dummies.values
classes = dummies.columns
y.shape

# to reshape data for the purpose of oversampling 
temp = X
temp_list = [i.reshape(-1)for i in temp]
temp_list = np.array(temp_list)

# oversampling to deal with unbalanced classes 
X_resampled, y_resampled = SMOTE().fit_resample(temp_list, y)

# split data 
x_train, x_val, y_train, y_val = train_test_split(    
    X_resampled, y_resampled, test_size=0.25,random_state=44)

# reshape data to put into CNN
x_train=x_train.reshape((95392, 99, 13,1))
x_val=x_val.reshape((31798, 99, 13,1))
x_test = np.array(x_test).reshape((11005, 99, 13,1))


# Create a sequential model
model = models.Sequential()

# Add convolutional and pooling layers
model.add(layers.Conv2D(128, kernel_size=(2,2),activation='relu', input_shape=(99,13,1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, kernel_size=(2,2),activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(32, kernel_size=(2,2),activation='relu'))
model.add(layers.Dense(32,activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(35,activation='softmax'))
model.summary()

#compile 
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

#early stop 
monitor = EarlyStopping(monitor='val_loss',verbose=0,patience=5,restore_best_weights = True,mode='auto')

# to fit the model
model.fit(x_train,y_train,validation_data=(x_val,y_val),callbacks=[monitor],verbose=2,epochs=30)

# predict 
y_pred = model.predict(x_test)

# turn predicted output into labels
predict_classes = np.argmax(y_pred,axis=1)
words = classes[predict_classes]


# to write into csv
df = pd.DataFrame({'path':df_test['path'],'result':words})
df.to_csv("result.csv",index=False,sep=',')
