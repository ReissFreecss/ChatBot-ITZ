import os
from rank_bm25 import BM25Okapi
import nltk
import string
from nltk.corpus import stopwords
from unidecode import unidecode
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, pipeline
import torch

# Descargar los recursos necesarios de NLTK
nltk.download('stopwords')
nltk.download('punkt')

# Función para normalizar texto
def normalize_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    text = unidecode(text)
    words = nltk.word_tokenize(text, language='spanish')
    stop_words = set(stopwords.words('spanish'))
    words = [word for word in words if word not in stop_words]
    return ' '.join(words) 

# Cargar y normalizar documentos de texto
directory = r'C:\Users\Carlos\OneDrive - Instituto Tecnológico de Zacatepec\Archivos Semestres\9no Semestre\Residencias\ChatBot-ITZ\ChatBot-ITZ\Data\TXT'
documents = []
filenames = []

for filename in os.listdir(directory):
    if filename.endswith('.txt'):
        filepath = os.path.join(directory, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as file:
                text = file.read()
        except UnicodeDecodeError:
            with open(filepath, 'r', encoding='latin-1') as file:
                text = file.read()

        normalized_text = normalize_text(text)
        documents.append(normalized_text)
        filenames.append(filename)  

print("-------------------------------------------------------")
print(f"Documentos leídos:  {len(documents)}")
print("-------------------------------------------------------")

# Tokenización de documentos
tokenized_documents = [doc.split() for doc in documents]
bm25 = BM25Okapi(tokenized_documents)

# Consulta BM25 utilizando la variable 'pregunta'
pregunta = "dame las competencias previas de la asignatura desarrollo de Aplicaciones .NET Core"
normalized_query = normalize_text(pregunta)
tokenized_query = normalized_query.split()
scores = bm25.get_scores(tokenized_query)

# Combinar nombres de archivos y puntajes
documents_with_scores = list(zip(filenames, scores))
documents_with_scores.sort(key=lambda x: x[1], reverse=True)
n_top_document = documents_with_scores[0]  # El documento con mayor puntaje

print(f"Documento seleccionado: {n_top_document[0]} con puntaje {n_top_document[1]:.3f}")
print("-------------------------------------------------------")

# Procesar el documento con mayor puntaje
filepath = os.path.join(directory, n_top_document[0])
with open(filepath, 'r', encoding='utf-8') as file:
    text = file.read()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=0, length_function=len)
chunks = text_splitter.split_text(text)

# Crear embeddings y base de conocimiento
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
knowledge_base = FAISS.from_texts(chunks, embeddings)

# Configurar el modelo GEMMA
model = "microsoft/Phi-3-mini-4k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model)
gemma_pipeline = pipeline(
    "text-generation",
    model=model,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",
)

# Definir el template del prompt
custom_template = """Usa para generar la respuesta del usuario unicamente la siguiente información.
Si no sabes la respuesta, o no se encuentra dentro del contexto simplemente di que no la sabes, no intentes inventar una respuesta.

Contexto: {context}

Pregunta: {question}

Respuesta útil:

"""

# Crear el prompt template
prompt_template = PromptTemplate(template=custom_template, input_variables=["context", "question"])

def generate_response(pregunta):
    # Buscar documentos relevantes
    docs = knowledge_base.similarity_search(pregunta, 1)
    
    # Extraer el contenido de los documentos y concatenarlo
    context = " ".join([doc.page_content for doc in docs])
    
    # Formatear el prompt usando el template
    prompt = prompt_template.format(context=context, question=pregunta)
    
    # Generar respuesta usando GEMMA
    outputs = gemma_pipeline(
        prompt,
        max_new_tokens=250,
        add_special_tokens=True,
        do_sample=True,
        temperature=0.1,
        top_k=50,
        top_p=0.95
    )
    # Devolver la respuesta generada
    return outputs[0]["generated_text"]

# Ejemplo de uso 
"""
Q? ->                       LANGCHAIN -> LMM GR -> R+U
     OCR (TXT) -> BM25 ->
     
    Test con tablas mal formadas (tabuladas)

Modelo Multimodales
Pruebas
    Set de datos
    Valores de perdidas (manejado como que tanto se puede equivocar, o salir de contexto)
    Valores ACCURACY 

Analisis de datos, Pre-Procesamiento, Procesamiento, Interaccion
"""
respuesta = generate_response(pregunta)
print(respuesta)