import matplotlib.pyplot as plt
import numpy as np

# Datos de la tabla
modelos = ["Dificil", "Normal", "Facil"]
escala_inflesz = [73.9425, 59.035, 27.1675]
escala_fernandez = [78.64, 64.1325, 33.145]
tiempo_ejecucion = [10.355, 13.125, 13.11]

# Configuración de las barras
x = np.arange(len(modelos))  # ubicaciones de los grupos
ancho = 0.25  # ancho de las barras

# Crear la figura y los ejes
fig, ax1 = plt.subplots()

# Gráfica de barras para el tiempo de ejecución
#rects1 = ax1.bar(x - ancho, tiempo_ejecucion, ancho, label='Tiempo de Ejecución (s)', color='red')

# Crear un segundo eje y para las escalas de legibilidad
ax2 = ax1.twinx()
rects2 = ax2.bar(x, escala_inflesz, ancho, label='Escala INFLESZ', color='blue')
rects3 = ax2.bar(x + ancho, escala_fernandez, ancho, label='Escala Fernández-Huerta', color='green')

# Añadir etiquetas y título
ax1.set_xlabel('Modelo de IA')
ax1.set_ylabel('Tiempo de Ejecución (s)', color='red')
ax2.set_ylabel('Escalas de Legibilidad')

ax1.set_xticks(x)
ax1.set_xticklabels(modelos)

# Añadir leyenda
fig.tight_layout()
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')

# Mostrar el gráfico
plt.show()
