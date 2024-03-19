import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy import signal
import math
import pywt
import seaborn as sns
import evaluate_knn
import evaluate_mlp
import evaluate_rbf_svm
import Feature_extraction
import Preprocessing
import plot_Tasks

#Definir la ruta al archivo CSV
Sujeto1 = "/content/drive/MyDrive/TFG/DataEdwin/Dataset/EDW_mod.csv" #12000
Sujeto2 = "/content/drive/MyDrive/TFG/DataEdwin/Dataset/GIANLUCA_mod.csv" #10000
Sujeto3 = "/content/drive/MyDrive/TFG/DataEdwin/Dataset/LEO_mod.csv" #32000


#Leer el archivo CSV con el delimitador adecuado
df1 = pd.read_csv(Sujeto1)
df2 = pd.read_csv(Sujeto2)
df3 = pd.read_csv(Sujeto3)

filter_type = 'bandpass'    #lowpass para paso bajo y bandpass para paso banda

filtered_EMG1_Sj1 = apply_filter(df1.CH1, filter_type)
filtered_EMG2_Sj1 = apply_filter(df1.CH2, filter_type)
filtered_EMG3_Sj1 = apply_filter(df1.CH3, filter_type)
filtered_EMG4_Sj1 = apply_filter(df1.CH4, filter_type)
filtered_EMG5_Sj1 = apply_filter(df1.CH5, filter_type)
filtered_EMG6_Sj1 = apply_filter(df1.CH6, filter_type)
filtered_EMG7_Sj1 = apply_filter(df1.CH7, filter_type)
filtered_EMG8_Sj1 = apply_filter(df1.CH8, filter_type)

filtered_EMG1_Sj2 = apply_filter(df2.CH1, filter_type)
filtered_EMG2_Sj2 = apply_filter(df2.CH2, filter_type)
filtered_EMG3_Sj2 = apply_filter(df2.CH3, filter_type)
filtered_EMG4_Sj2 = apply_filter(df2.CH4, filter_type)
filtered_EMG5_Sj2 = apply_filter(df2.CH5, filter_type)
filtered_EMG6_Sj2 = apply_filter(df2.CH6, filter_type)
filtered_EMG7_Sj2 = apply_filter(df2.CH7, filter_type)
filtered_EMG8_Sj2 = apply_filter(df2.CH8, filter_type)

filtered_EMG1_Sj3 = apply_filter(df3.CH1, filter_type)
filtered_EMG2_Sj3 = apply_filter(df3.CH2, filter_type)
filtered_EMG3_Sj3 = apply_filter(df3.CH3, filter_type)
filtered_EMG4_Sj3 = apply_filter(df3.CH4, filter_type)
filtered_EMG5_Sj3 = apply_filter(df3.CH5, filter_type)
filtered_EMG6_Sj3 = apply_filter(df3.CH6, filter_type)
filtered_EMG7_Sj3 = apply_filter(df3.CH7, filter_type)
filtered_EMG8_Sj3 = apply_filter(df3.CH8, filter_type)

visualizar_task(filtered_EMG1_Sj1, df1.Task, start_index=0)

#Normalization and rectification

rectified_CH1_Sj1 = np.abs(filtered_EMG1_Sj1)
rectified_CH2_Sj1 = np.abs(filtered_EMG2_Sj1)
rectified_CH3_Sj1 = np.abs(filtered_EMG3_Sj1)
rectified_CH4_Sj1 = np.abs(filtered_EMG4_Sj1)
rectified_CH5_Sj1 = np.abs(filtered_EMG5_Sj1)
rectified_CH6_Sj1 = np.abs(filtered_EMG6_Sj1)
rectified_CH7_Sj1 = np.abs(filtered_EMG7_Sj1)
rectified_CH8_Sj1 = np.abs(filtered_EMG8_Sj1)

rectified_CH1_Sj2 = np.abs(filtered_EMG1_Sj2)
rectified_CH2_Sj2 = np.abs(filtered_EMG2_Sj2)
rectified_CH3_Sj2 = np.abs(filtered_EMG3_Sj2)
rectified_CH4_Sj2 = np.abs(filtered_EMG4_Sj2)
rectified_CH5_Sj2 = np.abs(filtered_EMG5_Sj2)
rectified_CH6_Sj2 = np.abs(filtered_EMG6_Sj2)
rectified_CH7_Sj2 = np.abs(filtered_EMG7_Sj2)
rectified_CH8_Sj2 = np.abs(filtered_EMG8_Sj2)

rectified_CH1_Sj3 = np.abs(filtered_EMG1_Sj3)
rectified_CH2_Sj3 = np.abs(filtered_EMG2_Sj3)
rectified_CH3_Sj3 = np.abs(filtered_EMG3_Sj3)
rectified_CH4_Sj3 = np.abs(filtered_EMG4_Sj3)
rectified_CH5_Sj3 = np.abs(filtered_EMG5_Sj3)
rectified_CH6_Sj3 = np.abs(filtered_EMG6_Sj3)
rectified_CH7_Sj3 = np.abs(filtered_EMG7_Sj3)
rectified_CH8_Sj3 = np.abs(filtered_EMG8_Sj3)

scaler = MinMaxScaler(feature_range=(0, 1))
normalized_EMG_CH1_Sj1 = scaler.fit_transform(rectified_CH1_Sj1.reshape(-1, 1)).ravel()
normalized_EMG_CH2_Sj1 = scaler.fit_transform(rectified_CH2_Sj1.reshape(-1, 1)).ravel()
normalized_EMG_CH3_Sj1 = scaler.fit_transform(rectified_CH3_Sj1.reshape(-1, 1)).ravel()
normalized_EMG_CH4_Sj1 = scaler.fit_transform(rectified_CH4_Sj1.reshape(-1, 1)).ravel()
normalized_EMG_CH5_Sj1 = scaler.fit_transform(rectified_CH5_Sj1.reshape(-1, 1)).ravel()
normalized_EMG_CH6_Sj1 = scaler.fit_transform(rectified_CH6_Sj1.reshape(-1, 1)).ravel()
normalized_EMG_CH7_Sj1 = scaler.fit_transform(rectified_CH7_Sj1.reshape(-1, 1)).ravel()
normalized_EMG_CH8_Sj1 = scaler.fit_transform(rectified_CH8_Sj1.reshape(-1, 1)).ravel()

normalized_EMG_CH1_Sj2 = scaler.fit_transform(rectified_CH1_Sj2.reshape(-1, 1)).ravel()
normalized_EMG_CH2_Sj2 = scaler.fit_transform(rectified_CH2_Sj2.reshape(-1, 1)).ravel()
normalized_EMG_CH3_Sj2 = scaler.fit_transform(rectified_CH3_Sj2.reshape(-1, 1)).ravel()
normalized_EMG_CH4_Sj2 = scaler.fit_transform(rectified_CH4_Sj2.reshape(-1, 1)).ravel()
normalized_EMG_CH5_Sj2 = scaler.fit_transform(rectified_CH5_Sj2.reshape(-1, 1)).ravel()
normalized_EMG_CH6_Sj2 = scaler.fit_transform(rectified_CH6_Sj2.reshape(-1, 1)).ravel()
normalized_EMG_CH7_Sj2 = scaler.fit_transform(rectified_CH7_Sj2.reshape(-1, 1)).ravel()
normalized_EMG_CH8_Sj2 = scaler.fit_transform(rectified_CH8_Sj2.reshape(-1, 1)).ravel()

normalized_EMG_CH1_Sj3 = scaler.fit_transform(rectified_CH1_Sj3.reshape(-1, 1)).ravel()
normalized_EMG_CH2_Sj3 = scaler.fit_transform(rectified_CH2_Sj3.reshape(-1, 1)).ravel()
normalized_EMG_CH3_Sj3 = scaler.fit_transform(rectified_CH3_Sj3.reshape(-1, 1)).ravel()
normalized_EMG_CH4_Sj3 = scaler.fit_transform(rectified_CH4_Sj3.reshape(-1, 1)).ravel()
normalized_EMG_CH5_Sj3 = scaler.fit_transform(rectified_CH5_Sj3.reshape(-1, 1)).ravel()
normalized_EMG_CH6_Sj3 = scaler.fit_transform(rectified_CH6_Sj3.reshape(-1, 1)).ravel()
normalized_EMG_CH7_Sj3 = scaler.fit_transform(rectified_CH7_Sj3.reshape(-1, 1)).ravel()
normalized_EMG_CH8_Sj3 = scaler.fit_transform(rectified_CH8_Sj3.reshape(-1, 1)).ravel()

FeaturesCH1_Sj1 = Feature_extraction(normalized_EMG_CH1_Sj1,df1.Task, 100, 500)
FeaturesCH2_Sj1 = Feature_extraction(normalized_EMG_CH2_Sj1,df1.Task, 100, 500)
FeaturesCH3_Sj1 = Feature_extraction(normalized_EMG_CH3_Sj1,df1.Task, 100, 500)
FeaturesCH4_Sj1 = Feature_extraction(normalized_EMG_CH4_Sj1,df1.Task, 100, 500)
FeaturesCH5_Sj1 = Feature_extraction(normalized_EMG_CH5_Sj1,df1.Task, 100, 500)
FeaturesCH6_Sj1 = Feature_extraction(normalized_EMG_CH6_Sj1,df1.Task, 100, 500)
FeaturesCH7_Sj1 = Feature_extraction(normalized_EMG_CH7_Sj1,df1.Task, 100, 500)
FeaturesCH8_Sj1 = Feature_extraction(normalized_EMG_CH8_Sj1,df1.Task, 100, 500)

FeaturesCH1_Sj2 = Feature_extraction(normalized_EMG_CH1_Sj2,df2.Task, 100, 500)
FeaturesCH2_Sj2 = Feature_extraction(normalized_EMG_CH2_Sj2,df2.Task, 100, 500)
FeaturesCH3_Sj2 = Feature_extraction(normalized_EMG_CH3_Sj2,df2.Task, 100, 500)
FeaturesCH4_Sj2 = Feature_extraction(normalized_EMG_CH4_Sj2,df2.Task, 100, 500)
FeaturesCH5_Sj2 = Feature_extraction(normalized_EMG_CH5_Sj2,df2.Task, 100, 500)
FeaturesCH6_Sj2 = Feature_extraction(normalized_EMG_CH6_Sj2,df2.Task, 100, 500)
FeaturesCH7_Sj2 = Feature_extraction(normalized_EMG_CH7_Sj2,df2.Task, 100, 500)
FeaturesCH8_Sj2 = Feature_extraction(normalized_EMG_CH8_Sj2,df2.Task, 100, 500)

FeaturesCH1_Sj3 = Feature_extraction(normalized_EMG_CH1_Sj3,df3.Task, 100, 500)
FeaturesCH2_Sj3 = Feature_extraction(normalized_EMG_CH2_Sj3,df3.Task, 100, 500)
FeaturesCH3_Sj3 = Feature_extraction(normalized_EMG_CH3_Sj3,df3.Task, 100, 500)
FeaturesCH4_Sj3 = Feature_extraction(normalized_EMG_CH4_Sj3,df3.Task, 100, 500)
FeaturesCH5_Sj3 = Feature_extraction(normalized_EMG_CH5_Sj3,df3.Task, 100, 500)
FeaturesCH6_Sj3 = Feature_extraction(normalized_EMG_CH6_Sj3,df3.Task, 100, 500)
FeaturesCH7_Sj3 = Feature_extraction(normalized_EMG_CH7_Sj3,df3.Task, 100, 500)
FeaturesCH8_Sj3 = Feature_extraction(normalized_EMG_CH8_Sj3,df3.Task, 100, 500)

# Creamos una lista vacía para almacenar las características de cada canal
X_features_Sj1 = []
X_features_Sj2 = []
X_features_Sj3 = []
# Iteramos sobre los canales del primer sujeto
for channel in range(1, 9):
    X_features_Sj1.extend([f'FeaturesCH{channel}_Sj1.MAV', f'FeaturesCH{channel}_Sj1.RMS'])
    X_features_Sj2.extend([f'FeaturesCH{channel}_Sj2.MAV', f'FeaturesCH{channel}_Sj2.RMS'])
    X_features_Sj3.extend([f'FeaturesCH{channel}_Sj3.MAV', f'FeaturesCH{channel}_Sj3.RMS'])

# Convertimos la lista de características en un arreglo de NumPy
X1 = np.column_stack([eval(feature) for feature in X_features_Sj1])
X2 = np.column_stack([eval(feature) for feature in X_features_Sj2])
X3 = np.column_stack([eval(feature) for feature in X_features_Sj3])

# Definimos la variable objetivo
y1 = np.array(FeaturesCH1_Sj1.Task)
y2 = np.array(FeaturesCH1_Sj2.Task)
y3 = np.array(FeaturesCH1_Sj3.Task)

X_train, X_test, y_train, y_test = train_test_split(X3, y3, test_size=0.2, random_state=55)

#Entrenamos y evaluamos los respectivos algoritmos.

evaluate_rbf_svm(X_train, y_train, X_test, y_test)
evaluate_mlp(X_train, y_train, X_test, y_test)
evaluate_knn(X_train, y_train, X_test, y_test)
