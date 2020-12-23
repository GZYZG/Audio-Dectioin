#!/usr/bin/env python
# coding: utf-8

# In[1]:


import keras
from keras import backend as K
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50
import keras.models as models
import keras.layers as layers
import tensorflow as tf
import _pickle as pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
import os
import json
import datetime
import pandas as pd

# ### Specify the model name and training data paths below
# 
# Separate directories for positive and negative examples of each class should be specified in train_dir_tp, and train_dir_fp, respectively

# In[2]:


model_path = "/"
model_file_name = 'ResNet50_test'  # name of trained model file
model_dir = '../../data'  # path to directory where model will be stored

train_dir_tp = 'data/train_tp'  # directory with examples of true-positive spectrograms of each class
train_dir_fp = 'data/train_fp'  # directory with examples of false-positive spectrograms of each class

num_classes = 24
input_shape = [224, 224, 3]
batch_size = 32
epochs = 5


# ### Run remaining cells to begin training

# In[4]:


files = []
target = []
class_dict = dict()

for c, i in enumerate(sorted(os.listdir(train_dir_tp))):
    class_dict[c] = i
    for j in os.listdir(train_dir_tp+'/'+i):
        if not j.endswith("png"):
            continue
        files.append(train_dir_tp+'/'+i+'/'+j)
        tmp = np.empty(num_classes)
        tmp[:] = np.nan
        tmp[c] = int(1)
        target.append(tmp)
        
for c, i in enumerate(sorted(os.listdir(train_dir_fp))):
    class_dict[c] = i
    for j in os.listdir(train_dir_fp+'/'+i):
        if not j.endswith("png"):
            continue
        files.append(train_dir_fp+'/'+i+'/'+j)
        tmp = np.empty(num_classes)
        tmp[:] = np.nan
        tmp[c] = int(0)
        target.append(tmp)
        
df_train = pd.concat([pd.DataFrame({'filename':files}),pd.DataFrame(np.asarray(target))],axis=1)

print(len(df_train))
validation_indices = np.random.choice(range(len(df_train)), size=int(len(df_train)*0.1), replace=False)
df_validation = df_train.iloc[validation_indices]
df_train.drop(df_train.index[validation_indices], inplace=True)
print(len(df_train)+len(df_validation))
df_validation.reset_index(drop=True, inplace=True)
df_train.reset_index(drop=True, inplace=True)
df_train.head()


# In[3]:


train_datagen = ImageDataGenerator(rescale=1/255.0)

test_datagen = ImageDataGenerator(rescale=1/255.0)

train_generator = train_datagen.flow_from_dataframe(df_train,
                                                    y_col=range(num_classes),
                                                    directory=None,
                                                    target_size=input_shape[:2],
                                                    batch_size=batch_size,
                                                    class_mode='raw')

validation_generator = test_datagen.flow_from_dataframe(df_validation,
                                                        y_col=range(num_classes),
                                                        directory=None,
                                                        target_size=input_shape[:2],
                                                        batch_size=batch_size,
                                                        class_mode='raw')


def generator_wrapper(generator):
    for batch_x, batch_y in generator:
        yield (batch_x, np.row_stack(batch_y))


# In[4]:


def masked_loss(y_true, y_pred):
    return K.mean(K.mean(K.binary_crossentropy(tf.where(tf.math.is_nan(y_true), tf.zeros_like(y_true), y_true),
                                        tf.multiply(y_pred, tf.cast(tf.logical_not(tf.math.is_nan(y_true)), tf.float32))), axis=-1))


# In[7]:


#Load the ResNet50 model
ResNet50_conv = ResNet50(weights='imagenet', 
                         include_top=False, 
                         input_shape=input_shape)

for layer in ResNet50_conv.layers:
    layer.trainable = True

# Create the model
model = models.Sequential()
# Add the convolutional base model
model.add(ResNet50_conv)

model.add(layers.AveragePooling2D((7, 7)))

# Add new layers
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(num_classes, activation='sigmoid'))

# Compile the model
optimizer = keras.optimizers.Adam(lr=0.0001, decay=1e-7)
model.compile(loss=masked_loss, optimizer=optimizer, metrics=['accuracy'])

model.summary()


# In[ ]:


model_json = model.to_json()
with open(model_path+'/'+model_file_name+'.json', "w") as json_file:
    json_file.write(model_json)
with open(model_path+'/'+model_file_name+'_classes.json', 'w') as f:
    json.dump(class_dict, f)
print('Saved model architecture')
    
model_history = model.fit_generator(train_generator,
                                steps_per_epoch = len(train_generator),
                                epochs = epochs,
                                validation_data = validation_generator,
                                validation_steps = len(validation_generator),
                                verbose = 1)
print('Saving model...')
model.save_weights(model_path+'/'+model_file_name+'.h5')
            


# In[ ]:




