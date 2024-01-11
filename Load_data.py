import pandas as pd

def cargar_y_preprocesar_datos(ruta_archivo_tab):
    # Cargar datos desde el archivo
    data = pd.read_csv(ruta_archivo_tab, sep='\t')

    # Realizar el preprocesamiento
    columnas_emg = ['EMG1', 'EMG2', 'EMG3', 'EMG4', 'EMG5', 'EMG6', 'EMG7', 'EMG8']
    factor_conversion = 0.045

    for columna in columnas_emg:
        data[columna] = data[columna] * factor_conversion

    return data

# Ruta del archivo
ruta_archivo_tab = '/content/drive/MyDrive/TFG/Datasets/subject_3_EMG.tab'
