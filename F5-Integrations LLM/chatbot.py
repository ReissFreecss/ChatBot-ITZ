import os
from rank_bm25 import BM25Okapi
import nltk
import string
from nltk.corpus import stopwords
from unidecode import unidecode
from langchain.document_loaders import PyMuPDFLoader
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

# Cargar y normalizar documentos PDF
directory = './Data/PDF'
documents = []
filenames = []

for filename in os.listdir(directory):
    if filename.endswith('.pdf'):
        filepath = os.path.join(directory, filename)
        loader = PyMuPDFLoader(filepath)
        data_pdf = loader.load()
        text = " ".join([page.page_content for page in data_pdf])
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
pregunta = "Dame el nombre de las unidades de la asignatura biologia molecular"
normalized_query = normalize_text(pregunta)
tokenized_query = normalized_query.split()
scores = bm25.get_scores(tokenized_query)

# Combinar nombres de archivos y puntajes
documents_with_scores = list(zip(filenames, scores))
documents_with_scores.sort(key=lambda x: x[1], reverse=True)
n_top_document = documents_with_scores[0]  # El documento con mayor puntaje

print(f"Documento seleccionado: {n_top_document[0]} con puntaje {n_top_document[1]:.3f}")
print("-------------------------------------------------------")

# Cargar y procesar el documento con mayor puntaje
loader = PyMuPDFLoader(os.path.join(directory, n_top_document[0]))
data_pdf = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500, length_function=len)
chunks = text_splitter.split_documents(data_pdf)

# Crear embeddings y base de conocimiento
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
knowledge_base = FAISS.from_texts([chunk.page_content for chunk in chunks], embeddings)

# Configurar el modelo GEMMA
model = "google/gemma-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model)
gemma_pipeline = pipeline(
    "text-generation",
    model=model,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",
)

# Definir el template del prompt
custom_template = """Usa la siguiente información para responder la pregunta del usuario.
Si no sabes la respuesta, simplemente di que no la sabes, no intentes inventar una respuesta.

Contexto: {context}
Pregunta: {question}

Solo devuelve la respuesta útil a continuación y nada más.
Respuesta útil:

"""

# Crear el prompt template
prompt_template = PromptTemplate(template=custom_template, input_variables=["context", "question"])

def generate_response(pregunta):
    # Buscar documentos relevantes
    docs = knowledge_base.similarity_search(pregunta, 3)
    
    # Concatenar el texto de los documentos encontrados
    context = " ".join([doc.page_content for doc in docs])
    
    # Formatear el prompt usando el template
    prompt = prompt_template.format(context=context, question=pregunta)
    
    # Generar respuesta usando GEMMA
    outputs = gemma_pipeline(
        prompt,
        max_new_tokens=150,
        add_special_tokens=True,
        do_sample=True,
        temperature=0.3,
        top_k=50,
        top_p=0.95
    )
    
    # Devolver la respuesta generada
    return outputs[0]["generated_text"]

# Ejemplo de uso
respuesta = generate_response(pregunta)
print(respuesta)
