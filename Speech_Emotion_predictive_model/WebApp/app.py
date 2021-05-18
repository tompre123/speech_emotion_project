import flask
import pickle
import librosa
import soundfile
import numpy as np


# Load model from disk
with open(f'model/speech_model.pkl','rb') as file:
    model = pickle.load(file)

app = flask.Flask(__name__,template_folder='Templates')


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

@app.route('/', methods=['GET','POST'])
def main():
    if flask.request.method == 'GET':
        return (flask.render_template('main.html'))
    
    if flask.request.method == 'POST':
        print("hello")
        path = flask.request.form['path']
        print(path)
        prediction = model.predict(extract_feature(path,mfcc=True,chroma=True,mel=True).reshape(1,-1))[0]

        return flask.render_template('main.html',original_input={'path':path},result=prediction)

if __name__ == '__main__':
    app.run(debug=True)