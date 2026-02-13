import os 
import sys 
import math 

import csv 
import pywt 

import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt 

from scipy import stats
from sklearn.utils import resample

def ReadFiles(path=None):
    records = []
    annotations = []
    filenames = os.listdir(path) 

    for f in filenames:
        filename, ext = os.path.splitext(f) 
        if ext == '.csv':
            records.append(os.path.join(path, f))
        else:
            annotations.append(os.path.join(path, f)) 
    
    return records, annotations 


def ReadSignal(record):
    emp_signal=[]
    with open(record,'r') as f:
        reader = csv.reader(f, delimiter=',',quotechar='|')
        row_index= -1
        for row in reader:
            if row_index >=0:
                emp_signal.insert(row_index, int(row[1]))
            row_index += 1
    return emp_signal 


#  Baseline Wander Removal
def dwt_baseline_removal(signal, wavelet='db6', level=4):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    coeffs[0] = np.zeros_like(coeffs[0])  # remove low-frequency baseline
    corrected_signal = pywt.waverec(coeffs, wavelet)

    # Adjust reconstructed length
    if corrected_signal.size > signal.size:
        corrected_signal = corrected_signal[:signal.size]
    elif corrected_signal.size < signal.size:
        corrected_signal = np.pad(corrected_signal, (0, signal.size - corrected_signal.size), mode='edge')
    return corrected_signal


def denoise(data):
    w = pywt.Wavelet('sym4')
    maxlev = pywt.dwt_max_level(len(data), w.dec_len)
    threshold = 0.04 # Threshold for filtering

    coeffs = pywt.wavedec(data, 'sym4', level=maxlev)
  #  print(len(coeffs))
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold*max(coeffs[i]))

    datarec = pywt.waverec(coeffs, 'sym4')
    return datarec

def DatasetGeneration(records, annotations, window_size=180):
    X = []
    y = []
    for r in range(0,len(records)):
        signals = [ ] 
    
        with open(records[r],'r') as f:
            reader = csv.reader(f, delimiter=',',quotechar='|')
            row_index= -1
            for row in reader:
                if row_index >=0:
                    signals.insert(row_index, int(row[1]))
                row_index += 1
    
        signals = dwt_baseline_removal(signals)
        signals = denoise(signals)
        signals = stats.zscore(signals)

        with open(annotations[r], 'r') as fileID:
            data = fileID.readlines()
            beat = list()

            for d in range(1, len(data)):
                splitted = data[d].split(' ')
                splitted = filter(None, splitted)
                next(splitted)
                pos = int(next(splitted))
                arrhythmia_type = next(splitted)

                if(arrhythmia_type in classes):
                    arrhythmia_index = classes.index(arrhythmia_type)
                    count_classes[arrhythmia_index] += 1
                    if(window_size <= pos and pos < (len(signals) - window_size)):
                        beat = signals[pos-window_size:pos+window_size]
                        X.append(beat)
                        y.append(arrhythmia_index)
    
    return np.array(X), np.array(y)

def databalancing(X_new_df):
    df_0=(X_new_df[X_new_df[X_new_df.shape[1]-1]==0]).sample(n=7000,random_state=42)
    df_1=X_new_df[X_new_df[X_new_df.shape[1]-1]==1]
    df_2=X_new_df[X_new_df[X_new_df.shape[1]-1]==2]
    df_3=X_new_df[X_new_df[X_new_df.shape[1]-1]==3]
    df_4=X_new_df[X_new_df[X_new_df.shape[1]-1]==4]

    df_1_upsample=resample(df_1,replace=True,n_samples=7000,random_state=125)
    df_2_upsample=resample(df_2,replace=True,n_samples=7000,random_state=77)
    df_3_upsample=resample(df_3,replace=True,n_samples=7000,random_state=103)
    df_4_upsample=resample(df_4,replace=True,n_samples=7000,random_state=59)

    X_new_df=pd.concat([df_0,df_1_upsample,df_2_upsample,df_3_upsample,df_4_upsample])
    return X_new_df

if __name__ == '__main__':

    os.system('cls' if os.name == 'nt' else 'clear') 

    path = 'Specify the data path here' 

    os.environ["CUDA_VISIBLE_DEVICES"]='0'
    plt.rcParams["figure.figsize"] = (10,6)
    classes = ['N', 'L', 'R', 'A', 'V']
    n_classes = len(classes)
    count_classes = [0]*n_classes

    records, annptation = ReadFiles(path=path) 

    X, y = DatasetGeneration(records=records, annotations=annptation) 

    X_reshaped = X.reshape(-1,360,)
    X_df = pd.DataFrame(X_reshaped) 
    y_df = pd.DataFrame(y)
    X_new_df = pd.concat([X_df,y_df],axis=1)

    ax=list(range(361))
    X_new_df = X_new_df.set_axis(ax, axis='columns')




