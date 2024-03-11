import numpy as np
import matplotlib.pyplot as plt

def visualizar_task(emg, task_labels, start_index=0):
    """
    Visualiza señales EMG con colores según las tareas.

    Parámetros:
    - emg: Lista o array con los valores de amplitud de señal EMG.
    - task_labels: Lista o array con las etiquetas de tarea (Task1, Task2, ..., Task8, Rest).
    - start_index: Índice a partir del cual se inicia la visualización.

    """

    # Diccionario de mapeo de tareas a colores y nombres
    colores_nombres_tasks = {
        'Task1': {'color': 'blue', 'nombre': 'Fist'},
        'Task2': {'color': 'green', 'nombre': 'Open hand'},
        'Task3': {'color': 'orange', 'nombre': 'Wrist extension'},
        'Task4': {'color': 'red', 'nombre': 'Wrist flexion'},
        'Task5': {'color': 'purple', 'nombre': 'Pronation'},
        'Task6': {'color': 'brown', 'nombre': 'Supination'},
        'Task7': {'color': 'cyan', 'nombre': 'Radial deviation'},
        'Task8': {'color': 'magenta', 'nombre': 'Ulnar deviation'},
        'Rest': {'color': 'black', 'nombre': 'Rest'}
    }

    # Crear una figura (aumentar el tamaño)
    fig, ax = plt.subplots(figsize=(12, 6))

    for task in colores_nombres_tasks.keys():
        indices = [i for i, t in enumerate(task_labels) if t == task]

        # Graficar la señal EMG con colores según la etiqueta de la tarea
        ax.plot(np.arange(start_index, start_index+len(emg))[indices], np.array(emg[start_index:])[indices],
                color=colores_nombres_tasks[task]['color'], label=f'Señal EMG - {colores_nombres_tasks[task]["nombre"]}')

    # Añadir leyenda para colores y tareas en la parte superior derecha
    leyenda_tasks = [plt.Line2D([0], [0], color=colores_nombres_tasks[task]['color'],
                                label=f'{colores_nombres_tasks[task]["nombre"]}') for task in
                     colores_nombres_tasks.keys()]
    ax.legend(handles=leyenda_tasks, title="Tareas", loc='lower right', prop={'size': 7}, ncol=2)

    # Etiquetar los ejes
    ax.set_xlabel('Índice de muestra')
    ax.set_ylabel('Amplitud de señal EMG (µV)')

    # Mostrar la figura
    plt.show()
