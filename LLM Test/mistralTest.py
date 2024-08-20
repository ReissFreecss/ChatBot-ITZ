import torch
from transformers import AutoTokenizer, pipeline
import logging
import re
import time
from nltk.tokenize import sent_tokenize, RegexpTokenizer

# Iniciamos temporizador
start_time = time.time()

# Define el modelo que se va a utilizar
model = "mistralai/Mistral-7B-Instruct-v0.3"

# Carga el tokenizador asociado con el modelo especificado
tokenizer = AutoTokenizer.from_pretrained(model)

# Configura el pipeline para la generación de texto utilizando el modelo especificado
pipeline = pipeline(
    "text-generation",
    model=model,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",
)

# Define el prompt directamente (sin plantilla)
prompt = "Analiza cómo la teoría cuántica de campos puede reconciliarse con la teoría de la relatividad general."

# Genera texto utilizando el pipeline configurado
outputs = pipeline(
    prompt,
    max_new_tokens=256,
    add_special_tokens=True,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.95
)

# Extrae el texto generado por el modelo
generated_text = outputs[0]["generated_text"][len(prompt):]

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

# Calcula el índice de legibilidad para el prompt
indice_fernandez_prompt = calcular_fernandez_huerta(prompt)
indice_inflesz_prompt = calcular_inflesz(prompt)

# Calcula el índice de legibilidad para la respuesta generada
indice_fernandez_respuesta = calcular_fernandez_huerta(generated_text)
indice_inflesz_respuesta = calcular_inflesz(generated_text)

# Detén el cronómetro y calcula el tiempo de ejecución en minutos
end_time = time.time()
execution_time = end_time - start_time
execution_time_minutes = (end_time - start_time) / 60

print(f"Texto del prompt:--------------------------------------------------------\n{prompt}")
print(f"\nTexto generado por el chatbot:-----------------------------------------\n{generated_text}")

# Imprime los resultados
print(f"Escala de complegidad:--------------------------------------------------------")
print(f"Fernández-Huerta prompt: {indice_fernandez_prompt:.2f}")
print(f"INFLESZ prompt: {indice_inflesz_prompt:.2f}")
print(f"Fernández-Huerta chatbot: {indice_fernandez_respuesta:.2f}")
print(f"INFLESZ chatbot: {indice_inflesz_respuesta:.2f}")

print(f"Tiempo de ejecución:--------------------------------------------------------")
print(f"Minutos: {execution_time_minutes:.2f}")
print(f"Segundos: {execution_time:.2f}")
