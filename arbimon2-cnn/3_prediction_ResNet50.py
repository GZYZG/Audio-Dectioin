#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from keras.models import model_from_json
import json
import os
import matplotlib.pyplot as pyplot
import pickle
import numpy as np
import librosa
import librosa.display
import pandas as pd
# ### Specify the directory for test recordings, the path to the stored model, and the output file path/name

# In[23]:


# Recording directory
recording_dir = './data/test_recordings/'

# CNN model
model_path = './ResNet50_test'

# Path to output prediction CSV
output_path = './prediction_output.csv'


# ### Run remaining cells to generate prediction CSV

# In[24]:


# CNN input sample rate
model_sample_rate = 48000

test_recordings = os.listdir(recording_dir)


# In[1]:


# Load CNN model

model = model_from_json(open(model_path+'.json', 'r').read())
model.load_weights(model_path+'.h5')
class_dict = json.load(open(model_path+'_classes.json', 'r'))
class_dict_rev = {(str(v[0])): k for k, v in class_dict.items()}

print(model_path)
print('Loaded model ')

model_input_shape = model.get_layer(index=0).input_shape[1:]
n_classes = model.get_layer(index=-1).output_shape[1:][0]


# In[27]:


model_input_shape


# In[28]:


from keras.preprocessing.image import ImageDataGenerator
test_datagen = ImageDataGenerator(rescale=1./255)


# In[29]:


def fig2data ( fig ):
    """
    @brief Convert a Matplotlib figure to a 4D np array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a np 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw ( )
 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.frombuffer ( fig.canvas.tostring_rgb(), dtype=np.uint8 )
    buf.shape = ( w, h, 3 )
    
    return buf


# In[31]:


### Run detections

pixLen = 188  # 188 spectrogram pixels is ~2 seconds
shft = 93 # %50 overlap between 188-length windows

# Matrix of output predictions: rows are recordings, columns are species, 
prediction = np.zeros((len(uris), n_classes))


# Function to break image into frames
def divide_frames(im, w, s): 
    for i in range(0, im.shape[1], s):  
        yield im[:, i:i + w] 


for n, j in enumerate(test_recordings):  # loop over recordings
            
    print('Processing recording ' + str(j+1) + '/' + str(len(test_recordings)) + ' - ' + uris[j])
    
    audio_data, sampling_rate = librosa.load(recording_dir+j, sr=model_sample_rate)
    
    pxx = librosa.feature.melspectrogram(y = audio_data, 
                                           sr = sampling_rate,
                                           n_fft=2048, 
                                           hop_length=512, 
                                           win_length=1024)
    
    X = []
    for c, jj in enumerate(divide_frames(pxx, pixLen, shft)):  # loop over frames
        if jj.shape[1] != pixLen:
            continue
        dpi=100
        fig = pyplot.figure(num=None, figsize=(224/dpi, 224/dpi), dpi=dpi)
        pyplot.subplot(222)
        ax = pyplot.axes()
        ax.set_axis_off()
        librosa.display.specshow(librosa.power_to_db(jj, ref=np.max))
        img = fig2data(fig)
        pyplot.close()
        X.append(img/255.0)
    X = np.stack(X)
    
    p = model.predict(X)
            
    for i in range(n_classes):
        prediction[j, i] = max(p[:,i]) # Max-probability across 2s windows
#         prediction[j, i, 1] = np.mean(np.sort(p[:,i])[-2:]) # Mean probability of top 2 windows

        
            
            


# In[1]:


# Make dataframe of predictions
prediction = pd.DataFrame(prediction[:,:,0])
prediction.index = test_recordings
prediction.columns = [class_dict[str(i)][0] for i in range(n_classes)]
prediction.to_csv(output_path)


# In[ ]:




