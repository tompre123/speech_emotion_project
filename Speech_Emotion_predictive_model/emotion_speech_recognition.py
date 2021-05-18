#!/usr/bin/env python
# coding: utf-8

# In[1]:


import librosa
import soundfile
import os,glob,pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


# In[2]:


#Extracting feautures from soundfile
def extract_feature(file_name, mfcc, chroma, mel):
    with soundfile.SoundFile(file_name) as sound_file:
        audio_data = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma:
            stft = np.abs(librosa.stft(audio_data))
        result = np.array([])
        if mfcc:
            mfccs=np.mean(librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        if chroma:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        if mel:
            mel=np.mean(librosa.feature.melspectrogram(audio_data, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
    return result


# In[3]:


#emotions in RAVDESS dataset
emotions={
  '01':'neutral',
  '02':'calm',
  '03':'happy',
  '04':'sad',
  '05':'angry',
  '06':'fearful',
  '07':'disgust',
  '08':'surprised'
}

#emotions that we are gonna observe
observed_emotions = ['calm','happy','sad','angry']


# In[4]:


#loading the data and extracting features/emotion labels from files
def load_data(test_size = 0.2):
    x,y = [],[]
    for file in glob.glob("C:\\Users\\TESTER\\Desktop\\python_proj\\speech-emotion-recognition-ravdess-data\\Actor_*\\*.wav"):
        file_name = os.path.basename(file)
        emotion = emotions[file_name.split("-")[2]]
        if emotion not in observed_emotions:
            continue
        feature = extract_feature(file,mfcc=True,chroma=True,mel=True)
        x.append(feature)
        y.append(emotion)
    
    return train_test_split(np.array(x), y, test_size=test_size, random_state=9)


# In[5]:


#spliting dataset in trainnig and validation set
x_train, x_test, y_train, y_test = load_data(test_size = 0.25)

print((x_train.shape[0],x_test.shape[0]))

print(f'Features extracted: {x_train.shape[1]}')


# In[12]:


#using a multi layer perceptron 

model = MLPClassifier(alpha = 0.01, batch_size = 256, epsilon = 1e-09, hidden_layer_sizes = (400,), verbose = True, learning_rate = 'adaptive', max_iter = 10000, solver = 'lbfgs')

model.fit(x_train,y_train)


# In[15]:


y_pred=model.predict(x_test)
print(x_test)


# In[14]:



accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)

print("Accuracy: {:.2f}%".format(accuracy*100))


# In[18]:



#file = "C:\\Users\\TESTER\\Desktop\\python_proj\\speech-emotion-recognition-ravdess-data\\Actor_12\\03-01-02-01-02-01-12.wav"

#emotion = model.predict(extract_feature(file,mfcc=True,chroma=True,mel=True).reshape(1,-1))
#print(emotion)


# In[ ]:

with open('speech_model.pkl','wb') as file:
    pickle.dump(model,file)


