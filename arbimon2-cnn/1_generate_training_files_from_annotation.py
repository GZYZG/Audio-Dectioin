#!/usr/bin/env python
# coding: utf-8

# ### Load dependencies

# In[5]:


import os
import shutil
import time
import pickle
import pandas as pd
import matplotlib.pyplot as pyplot
import librosa
import librosa.display
import gc
import numpy as np
import urllib
def download_rec(recording_id, rec_dir, filename):
    if not os.path.exists(rec_dir+'/'+recording_id.replace('/','_')):
        try:
            urllib.request.urlretrieve (recording_id, rec_dir+'/'+filename)
        except Exception as e:
            print(e)
            print('Error downloading '+recording_id)
        return
    

# ### Specify data paths:
# 
# 1. train_tp_set_dir - Folder where training data will be stored
# 2. recording_dir - Folder where audio recordings will be stored
# 3. sound_annotation_file - File storing template matching validation metadata
# 4. (Optional) sampling_rate - rate to resample training data recordings to
#     

# In[6]:


train_tp_set_dir = 'data/train_tp/'  # Folder where training data will be stored
train_fp_set_dir = 'data/train_fp/'
recording_dir = '../data/train/'  # Folder holding recordings

sound_annotation_files = ["../Audio Detection/data/train_all.csv"]# ['./example_annotations.csv']
# File storing ROIs of detected sounds (animal calls) 
#     Required columns:
#          species
#          x1 (start time of sound)
#          x2 (end time of sound)
#          recording_id (recording file path)

sampling_rate = 48000 # training data recording sample rate


# ### Run remaining cells to generate training data

# In[7]:


if not os.path.exists(recording_dir):
    os.mkdir(recording_dir)
if not os.path.exists(train_tp_set_dir):
    os.mkdir(train_tp_set_dir)


# In[7]:


if len(sound_annotation_files)==1:
    rois= pd.read_csv(sound_annotation_files[0])
elif len(sound_annotation_files)==0:
    print('Must provide an annotation file')
elif len(sound_annotation_files)>1:
    rois = pd.read_csv(sound_annotation_files[0])
    for i in sound_annotation_files[1:]:
        tmp = pd.read_csv(sound_annotation_files[i])
        rois = pd.concat([rois,tmp])
rois.head()


# In[8]:


# For using Arbimon 2 Pattern Matching results - convert recording_id to full download URL
# rois['recording_id'] = [i[1].recording_id.split('detections')[0]+
#                   'site_'+str(i[1].site_id)+'/'+
#                   str(i[1].year)+'/'+
#                   str(i[1].month)+'/'+
#                   i[1].recording for i in rois.iterrows()]


# In[9]:


print('Number of ROIs for each species\n')

for i in list(set(rois.species_id)):
    print(str(i)+'\t\t'+str(len(rois[rois.species_id == i])))


# In[14]:


window_length = 2 # sample time-window length in seconds
k = 0
t0 = time.time()
rec_loaded = False

g = rois.groupby("recording_id")
i = 0
for record, group in g:
    audio_file_path = recording_dir + record + ".flac"   # 音频文件路径
    audio_data, sampling_rate = librosa.load(audio_file_path, sr=sampling_rate)
    for idx in range(len(group)):
        item = group.iloc[idx]
        species_id = item["species_id"]
        sound_start, sound_end = item['t_min'], item['t_max']
        positive = item["positive"]
        save_dir = train_tp_set_dir if positive == 1 else train_fp_set_dir  # 保存频谱图的目录
        save_dir += f"{species_id}/"
        shft = ((sound_end - sound_start) - window_length) / 2
        start_sample = round(sampling_rate * (sound_start + shft))
        start_sample = max(start_sample, 0)
        filename = record + '_' + str(round(start_sample / sampling_rate, 2)) + '-' + str(
            round((start_sample / sampling_rate) + window_length, 2)) + '.png'  # 频谱图的文件名

        if os.path.exists(save_dir + filename):
            print(f"{save_dir + filename} already exists, skip generating it ... {idx} / {len(group)}")
            continue

        S = librosa.feature.melspectrogram(
            y=audio_data[int(start_sample): int(start_sample + round(sampling_rate * window_length))],
            sr=sampling_rate,
            n_fft=2048,
            hop_length=512,
            win_length=1024)
        dpi = 100
        fig = pyplot.figure(num=None, figsize=(300 / dpi, 300 / dpi), dpi=dpi)
        pyplot.subplot(222)
        ax = pyplot.axes()
        ax.set_axis_off()
        librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
        pyplot.savefig(save_dir + filename, bbox_inches='tight',
                       transparent=True, pad_inches=0.0)
        pyplot.close()
        print(f"generating {save_dir + filename} finished ... {idx} / {len(group)}")
    i += 1
    print(f"\n{'*'*10} extracting {record}.flac finished ... {i+1} / {len(g)} {'*'*10}\n")




"""
for i in list(set(rois.recording_id)):  # loop over recordings
    
    k = k+1
#     if k%200==0:
    print(k)
        
    tmp = rois[rois.recording_id==i]
    if 's3.amazonaws' in tmp.iloc[0].recording_id:
        audio_filename = tmp.iloc[0].recording_id.replace('/','_').split('arbimon2_')[1]

    for c in range(len(tmp)):  # loop over spectrogram ROIs
        
        try:

            sound_start, sound_end = [tmp.iloc[c]['t_min'], tmp.iloc[c]['t_max']]
            species = tmp.iloc[c].species.replace(' ','_')
            
            if not os.path.exists(train_tp_set_dir+'/'+str(species)):
                os.mkdir(train_tp_set_dir+'/'+str(species))

            shft = ((sound_end-sound_start)-window_length)/2
            start_sample = round(sampling_rate*(sound_start+shft))
            start_sample = max(start_sample, 0)
            filename = audio_filename.split('.')[0]+'_'+str(round(start_sample/sampling_rate,2))+'-'+str(round((start_sample/sampling_rate)+window_length,2))+'.png'

            if not os.path.exists(train_tp_set_dir+str(species)+'/'+filename):
                if not rec_loaded:
                    if not os.path.exists(recording_dir+'/'+audio_filename):
                        download_rec(tmp.iloc[0].recording_id, recording_dir, audio_filename)
                    try:
                        audio_data, sampling_rate = librosa.load(recording_dir+audio_filename, sr=sampling_rate)
                        rec_loaded = True
                    except Exception as e:
                        print(e)
                        continue
                S = librosa.feature.melspectrogram(y = audio_data[int(start_sample): int(start_sample+round(sampling_rate*window_length))], 
                                               sr = sampling_rate,
                                               n_fft=2048, 
                                               hop_length=512, 
                                               win_length=1024)
                dpi=100
                fig = pyplot.figure(num=None, figsize=(300/dpi, 300/dpi), dpi=dpi)
                pyplot.subplot(222)
                ax = pyplot.axes()
                ax.set_axis_off()
                librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
                pyplot.savefig(train_tp_set_dir+str(species.replace(' ','_'))+'/'+filename, bbox_inches='tight', transparent=True, pad_inches=0.0)
                pyplot.close()
                
        except Exception as e:
            print(e)
            continue
        
    rec_loaded = False    
"""

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




