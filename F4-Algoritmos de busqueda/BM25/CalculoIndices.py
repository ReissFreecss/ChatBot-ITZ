
from nltk.tokenize import sent_tokenize, RegexpTokenizer
import re
import logging

def calcular_silabas(palabra):
    try:
        palabra = palabra.lower().replace('ü', 'u')
        palabra = re.sub(r'(ue|ui|iu|ie|eu|ei|oi|io|ou|uo|au|ai|aí|aú|eí|eú|oí|oú)', 'a', palabra)
        return len(re.findall(r'[aeiouáéíóú]', palabra))
    except Exception as e:
            logging.error(f"Error en la función silabas: {e}")
            return 0

# Función con cálculos necesarios para índices
def calculos(texto):
    oraciones = sent_tokenize(texto, language='spanish')
    tokenizer = RegexpTokenizer(r'\w+')
    palabras = tokenizer.tokenize(texto)
    total_silabas = sum(calcular_silabas(palabra) for palabra in palabras)
    total_palabras = len(palabras)
    total_frases = len(oraciones)
    
    return total_silabas, total_palabras, total_frases

# Función para calcular índice FERNANDEZ-HUERTA
def calcular_fernandez_huerta(texto):
    try:
        total_silabas, total_palabras, total_frases = calculos(texto)

        P = (total_silabas / total_palabras) * 100
        F = (total_palabras / total_frases)

        huerta = 206.84 - (0.60 * P) - (1.02 * F)
        return max(0, min(huerta, 100))  # Asegura que el valor esté entre 0 y 100
    except Exception as e:
        logging.error(f"Error en la función calcular_fernandez_huerta: {e}")
        return 0
    
# Función para calcular índice INFLESZ
def calcular_inflesz(texto):
    try:
        total_silabas, total_palabras, total_frases = calculos(texto)

        inflesz = 206.835 - (62.3 * (total_silabas / total_palabras)) - (total_palabras / total_frases)
        return max(0, min(inflesz, 100))  # Asegura que el valor esté entre 0 y 100
    except Exception as e:
        logging.error(f"Error en la función calcular_inflesz: {e}")
        return 0
    
prompt = "Los errores de lectura en la secuenciación de ADN de una molécula única pueden afectar la interpretación de variantes genómicas complejas de varias maneras. 1. **Alteración de los límites de la secuencia:** Los errores de lectura pueden introducir mutaciones en los límites de la secuencia, lo que puede afectar la capacidad de la enzima de lectura de reconocer el sitio de interés. 2. **Cambios en el orden de las bases:** Los errores de lectura también pueden cambiar el orden de las bases en la secuencia, lo que puede afectar la lectura de los codones y las proteínas que se codifican. 3. **Inserciones y delecciones:** Los errores de lectura también pueden introducir o eliminar secuencias de ADN, lo que puede afectar el tamaño de la molécula y las proteínas que se codifican. 4. **Cambios en la lectura de los codones:** Los errores de lectura también pueden cambiar la lectura de los codones, lo que puede afectar la lectura de las proteínas y los funciones de las proteínas."
# Limpia el prompt para eliminar etiquetas especiales
prompt_cleaned = re.sub(r'<.*?>', '', prompt)
indice_fernandez_prompt = calcular_fernandez_huerta(prompt_cleaned)
indice_inflesz_prompt = calcular_inflesz(prompt_cleaned)
print(f"Fernández-Huerta prompt: {indice_fernandez_prompt:.2f}")
print(f"INFLESZ prompt: {indice_inflesz_prompt:.2f}")