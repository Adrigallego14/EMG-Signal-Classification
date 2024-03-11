import pandas as pd
import numpy as np
from scipy.signal import welch

def Feature_extraction(emgval, taskval, window_size, sampling_rate):
    """
    Calcula el Valor Absoluto Medio (MAV), la Media Cuadrática (RMS),
    el Área Bajo la Curva (AUC) y la Frecuencia Dominante (DF)
    para cada ventana de muestras donde las muestras tienen la misma tarea.

    Args:
    - emgval (Series de Pandas): Serie que contiene los datos normalizados.
    - taskval (Series de Pandas): Serie que contiene las tareas asociadas con cada muestra de datos.
    - window_size (int): Tamaño de la ventana.
    - sampling_rate (float): Tasa de muestreo de la señal sEMG.

    Returns:
    - df_features (DataFrame de Pandas): DataFrame que contiene las características extraídas
      (MAV, RMS, AUC, DF) y su tarea asociada.
    """

    # Convertir las series de Pandas en DataFrames
    emgval_df = pd.DataFrame(emgval)
    taskval_df = pd.DataFrame(taskval)

    # Inicializar listas para almacenar las características extraídas y las tareas asociadas
    mav_values = []
    rms_values = []
    auc_values = []
    df_values = []
    associated_tasks = []

    # Calcular el número máximo de ventanas completas
    max_windows = len(emgval_df) // window_size

    # Iterar sobre el número máximo de ventanas completas
    for i in range(max_windows):
        # Obtener las tareas dentro de la ventana actual
        tasks_in_window = taskval_df.iloc[i * window_size: (i + 1) * window_size]

        # Verificar si todas las tareas son iguales dentro de la ventana
        if len(set(tasks_in_window)) == 1:
            # Seleccionar la ventana de datos
            window_data = emgval_df.iloc[i * window_size: (i + 1) * window_size]

            # Calcular el valor absoluto medio (MAV) de la ventana
            abs_mean = np.mean(np.abs(window_data.values))
            mav_values.append(abs_mean)

            # Calcular la media cuadrática (RMS) de la ventana
            rms_value = np.sqrt(np.mean(np.square(window_data.values)))
            rms_values.append(rms_value)

            # Calcular el área bajo la curva (AUC) de la ventana
            auc_value = np.trapz(window_data.values.flatten())
            auc_values.append(auc_value)

            # Calcular la frecuencia dominante (DF) de la ventana
            f, Pxx = welch(window_data.values.flatten(), fs=sampling_rate, nperseg=window_data.shape[0])
            df_value = f[np.argmax(Pxx)]
            df_values.append(df_value)

            # Almacenar la tarea asociada a las características extraídas
            associated_tasks.append(tasks_in_window.iloc[0].values[0])

    # Crear un DataFrame con las características extraídas y su tarea asociada
    df_features = pd.DataFrame({'MAV': mav_values, 'RMS': rms_values, 'AUC': auc_values, 'DF': df_values, 'Task': associated_tasks})

    return df_features
