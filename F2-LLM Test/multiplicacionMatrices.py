import sys  # Modulo sys, que proporciona acceso a algunas variables 
import numpy as np  # Biblioteca NumPy, que se utiliza para trabajar 
import tensorflow as tf  # Biblioteca de aprendizaje automático y cálculo
from datetime import datetime  # Datetime para trabajar con fechas y horas

def prueba_dispositivo(nombre_dispositivo, tamanio_matriz):
    ''' Calcula el tiempo consumido en hacer multiplicaciones 
    matriciales usando un dispositivo y un tamaño de filas y columnas de la matriz aleatoria.
    :param nombre_dispositivo: nombre del dispositivo reconocible para        
    TensorFlow (ej: "/cpu:0" o "/gpu:0").
    :param tamanio_matriz: número de filas y de columnas de la matriz aleatoria.
    Ejemplo:
    prueba_dispositivo("/cpu:0", 100)
    '''
    dimension = (int(tamanio_matriz), int(tamanio_matriz))  
    with tf.device(nombre_dispositivo):  # Especificar (CPU o GPU)
    # Genera una matriz aleatoria uniforme con valores entre 0 y 1
    random_matrix = tf.random_uniform(shape=dimension, minval=0, maxval=1, seed=1993)
   # Realiza la multiplicación de la matriz aleatoria por su transpuesta
   dot_operation = tf.matmul(random_matrix, tf.transpose(random_matrix))
        # Calcula la suma de todos los elementos del resultado
        sum_operation = tf.reduce_sum(dot_operation)
 
    inicio = datetime.now()  # Guarda el tiempo para medir la duracion

    # Inicia una sesión de TensorFlow 
    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session:
        result = session.run(sum_operation)  
    tiempo_consumido = datetime.now() - inicio  
    return tiempo_consumido

resultados = {'CPU': [], 'GPU': []} 
# Recorre un rango de tamaños de matriz en una escala geométrica desde 10 hasta 15000, dividiéndolo en 200 puntos.
for tamanio_matriz in np.geomspace(10, 15000, num=200):
    # Función de prueba en el dispositivo CPU y almacena el tiempo consumido.
    resultados['CPU'].append(prueba_dispositivo("/cpu:0", tamanio_matriz))
    # Función de prueba en el dispositivo GPU y almacena el tiempo consumido.
    resultados['GPU'].append(prueba_dispositivo("/gpu:0", tamanio_matriz))
