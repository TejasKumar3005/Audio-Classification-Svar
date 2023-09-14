import speech_recognition as sr
import threading
import time
import pyttsx3
from scipy.io import wavfile
import math
import numpy as np
import json
import tensorflow as tf
from clean import downsample_mono, envelope
import os

def gaussian_pdf(x, mean, std):
    exponent = np.exp(-((x-mean)**2 / (2 * std**2 )))
    return (1 / (np.sqrt(2 * np.pi) * std)) * exponent

def classify(model_path, voice_path, values, sample_class):
    
    # sample_embedding: 44-dimensional embedding of the sample
    # values: dictionary of class means and standard deviations
    # sample_class: class of the sample
    
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    rate, wav = downsample_mono(voice_path, 16000)
    mask, env = envelope(wav, rate, threshold=20)
    clean_wav = wav[mask]
    step = int(16000*2.5)
    
    batch = []

    for i in range(0, clean_wav.shape[0], step):
        sample = clean_wav[i:i+step]
        sample = sample.reshape(-1, 1)
        if sample.shape[0] < step:
            tmp = np.zeros(shape=(step, 1), dtype=np.float32)
            tmp[:sample.shape[0],:] = sample.flatten().reshape(-1, 1)
            sample = tmp
        batch.append(sample)
    
    X_batch = np.array(batch, dtype=np.float32)
    y_pred = []
    for x in X_batch:
        interpreter.set_tensor(input_details[0]['index'], [x])
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        # print("output_data")
        
        # print(output_data)
        y_pred.append(output_data)

    y_mean = np.mean(y_pred, axis=0)
    
    # get position of sample_class in values
    i = values.keys().index(sample_class)
    
    prob = np.prod(gaussian_pdf(y_mean, values[sample_class][0][0], values[sample_class][1][0]))
    
    print(prob)


# import ./logs/tflite_pred_1.json
with open('./logs/tflite_pred_1.json', 'r') as f:
    values = json.load(f)

# model = "./model1.tflite"
# dataset_dir = "./wavfiles"
# for root, dirs, files in os.walk(dataset_dir):
#     print(dirs)
#     for dirs in dirs:
#         for root, dirs, files in os.walk(os.path.join(dataset_dir, dirs)):
#             for file in files:
#                 if file.endswith(".wav"):
#                     classify(model, os.path.join(dataset_dir, dirs, file), values, dirs)



np.set_printoptions(threshold=np.inf)

def st(command):
    # Initialize the engine
    engine = pyttsx3.init()
    engine.setProperty("rate", 150)
    engine.say(command)
    engine.runAndWait()


def recognizing(alphabet):
    r = sr.Recognizer()
    r.energy_threshold = 5000
    with sr.Microphone() as source:
        ask_1 = '''Try speaking {} '''.format(alphabet)

        def speak():
            time.sleep(.55)
            st(ask_1)

        def listen():
            global sampling_rate, signal,audio
            audio = r.listen(source)
            

        t2 = threading.Thread(target=listen)
        t3 = threading.Thread(target=speak)
        t3.start()
        t2.start()
        t3.join()
        t2.join()
        st("Recognizing")
     
        x = audio.get_wav_data()
        open("test.wav", "wb").write(x)
        sampling_rate, signal = wavfile.read("test.wav")
        v1 = extract_st_features(signal, sampling_rate)
        sampling_rate, signal = wavfile.read("C:/IITD/python learning projects/ELL 205/Svar-App-1-master/Svar-App-1-master/"+ alphabet + ".wav")
        v2 = extract_st_features(signal, sampling_rate)
        
        diff = abs((v1 - v2)/(abs(v1)+abs(v2)))
        weight=np.full((diff.shape[0],diff.shape[1]), 1)
        rms = 0
        sum = 0
        for i in range(0, diff.shape[0]):
            for j in range(0, diff.shape[1]):
                rms = rms + ((1-diff[i][j])**2)*weight[i][j]
                sum = sum + weight[i][j]
        rms = math.sqrt(rms/sum)
        return (rms*100//5)*5