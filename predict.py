from tensorflow.keras.models import load_model
from clean import downsample_mono, envelope
from kapre.time_frequency import STFT, Magnitude, ApplyFilterbank, MagnitudeToDecibel
from sklearn.preprocessing import LabelEncoder
import numpy as np
from glob import glob
import argparse
import os
import pandas as pd
import json
from tqdm import tqdm
import tensorflow as tf
from scipy.spatial.distance import cosine
import faulthandler
faulthandler.enable()
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

with open("f.json", 'w') as f:
        # convert np.array to list
        values = {}
        values["a"] = ([1,2,3],[1,2,3])
        json.dump(values, f)

def make_prediction(args):

    model = load_model(args.model_fn,
        custom_objects={'STFT':STFT,
                        'Magnitude':Magnitude,
                        'ApplyFilterbank':ApplyFilterbank,
                        'MagnitudeToDecibel':MagnitudeToDecibel})
    wav_paths = glob('{}/**'.format(args.src_dir), recursive=True)
    wav_paths = sorted([x.replace(os.sep, '/') for x in wav_paths if '.wav' in x])
    classes = sorted(os.listdir(args.src_dir))
    labels = [os.path.split(x)[0].split('/')[-1] for x in wav_paths]
    le = LabelEncoder()
    y_true = le.fit_transform(labels)
    results = []

    for z, wav_fn in tqdm(enumerate(wav_paths), total=len(wav_paths)):
        rate, wav = downsample_mono(wav_fn, args.sr)
        mask, env = envelope(wav, rate, threshold=args.threshold)
        clean_wav = wav[mask]
        step = int(args.sr*args.dt)
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
        y_pred = model.predict(X_batch)
        y_mean = np.mean(y_pred, axis=0)
        y_pred = np.argmax(y_mean)
        real_class = os.path.dirname(wav_fn).split('/')[-1]
        print('Actual class: {}, Predicted class: {}'.format(real_class, classes[y_pred]))
        results.append(y_mean)

    np.save(os.path.join('logs', args.pred_fn), np.array(results))



def make_prediction_lite(args):

    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path=args.model_fn)
    interpreter.allocate_tensors()

    # Get input and output tensors information
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    wav_paths = glob('{}/**'.format(args.src_dir), recursive=True)
    wav_paths = sorted([x.replace(os.sep, '/') for x in wav_paths if '.wav' in x])
    classes = sorted(os.listdir(args.src_dir))
    labels = [os.path.split(x)[0].split('/')[-1] for x in wav_paths]
    le = LabelEncoder()
    y_true = le.fit_transform(labels)
    results = []
    values = {}

    for z, wav_fn in tqdm(enumerate(wav_paths), total=len(wav_paths)):
        rate, wav = downsample_mono(wav_fn, args.sr)
        mask, env = envelope(wav, rate, threshold=args.threshold)
        clean_wav = wav[mask]
        step = int(args.sr*args.dt)
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

        # Use the interpreter to predict
        y_pred = []
        for x in X_batch:
            interpreter.set_tensor(input_details[0]['index'], [x])
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            # print("output_data")
            
            # print(output_data)
            y_pred.append(output_data)

        y_mean = np.mean(y_pred, axis=0)
        # print("y_mean")
        
        # print(y_mean)
        y_pred = np.argmax(y_mean)
        real_class = os.path.dirname(wav_fn).split('/')[-1]
        print('Actual class: {}, Predicted class: {}'.format(real_class, classes[y_pred]))
        if real_class not in values:
            values[real_class] = []
        # take softmax of y_mean
        # y_mean = np.exp(y_mean) / np.sum(np.exp(y_mean))
        values[real_class].append(y_mean)
        print(y_mean)
        results.append(y_mean)

    for k, v in values.items():
        mean_vector = np.mean(v, axis=0)[0]
        cosine_distances = [cosine(mean_vector, vec) for vec in v]
        values[k] = (mean_vector, np.std(cosine_distances))

        
    # save values as josn
    with open(os.path.join('logs', args.pred_fn + '.json'), 'w') as f:
        # convert np.array to list
        values = {k: (v[0].tolist(), v[1].tolist()) for k, v in values.items()}
        json.dump(values, f)
    np.save(os.path.join('logs', args.pred_fn), np.array(results))
    
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Audio Classification Training')
    parser.add_argument('--model_fn', type=str, default='model1.tflite',
                        help='model file to make predictions')
    parser.add_argument('--pred_fn', type=str, default='tflite_cloud',
                        help='fn to write predictions in logs dir')
    parser.add_argument('--src_dir', type=str, default='wavfiles',
                        help='directory containing wavfiles to predict')
    parser.add_argument('--dt', type=float, default=2.5,
                        help='time in seconds to sample audio')
    parser.add_argument('--sr', type=int, default=16000,
                        help='sample rate of clean audio')
    parser.add_argument('--threshold', type=str, default=20,
                        help='threshold magnitude for np.int16 dtype')
    args, _ = parser.parse_known_args()

    make_prediction_lite(args)

