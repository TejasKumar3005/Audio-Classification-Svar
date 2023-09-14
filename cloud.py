import numpy as np
import json
import tensorflow as tf
from clean import downsample_mono, envelope
import os
from scipy.spatial.distance import cosine

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def gaussian_pdf(x, mean, std):
    # x= np.array(x)
    # mean = np.array(mean)
    # std = np.array(std)
    exponent = np.exp(-((x-mean)**2 / (2 * std**2 )))
    # return (1 / (np.sqrt(2 * np.pi) * std)) * exponent #/ 100
    return exponent

def classify(model_path, voice_path, values, sample_class):
    
    # sample_embedding: 44-dimensional embedding of the sample
    # values: dictionary of class means and standard deviations
    # sample_class: class of the sample
    # sample_class = "sound 'a'"
    
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
    

    print(sample_class)
    
    # prob = (gaussian_pdf(y_mean, values[sample_class][0][0], values[sample_class][1][0]))[i]
    dist = cosine(y_mean,values[sample_class][0] )
    prob = gaussian_pdf(dist, 0 , values[sample_class][1]) **(0.5)
    print(prob)


# import ./logs/tflite_pred_1.json
with open('./logs/tflite_cloud.json', 'r') as f:
    values = json.load(f)

model = "./model1.tflite"
dataset_dir = "./wavfiles"
for root, dirs, files in os.walk(dataset_dir):
    print(dirs)
    for folder in dirs:
        for root, _, files in os.walk(os.path.join(dataset_dir, folder)):
            for file in files:
                if file.endswith(".wav"):
                    classify(model, os.path.join(dataset_dir, folder, file), values, folder)
