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

# Cargar y normalizar documentos de texto
directory = r'C:\Users\Carlos\OneDrive - Instituto Tecnológico de Zacatepec\Archivos Semestres\9no Semestre\Residencias\ChatBot-ITZ\ChatBot-ITZ\Data\TXT'
documents = []
filenames = []

# Función para normalizar texto
def normalize_text(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    text = unidecode(text)
    words = nltk.word_tokenize(text, language='spanish')
    stop_words = set(stopwords.words('spanish'))
    words = [word for word in words if word not in stop_words]
    print(words)
    return ' '.join(words)

def load_and_normalize_documents(directory):
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

    return documents, filenames

print("-------------------------------------------------------")
print(f"Documentos leídos:  {len(documents)}")
print("-------------------------------------------------------")

# Tokenización de documentos
tokenized_documents = [doc.split() for doc in documents]
bm25 = BM25Okapi(tokenized_documents)

# Ingreso del input
while True:
    pregunta = input("Pregunta (escribe 'salir' para terminar): ")
    if pregunta.lower() == "salir":
        break

    normalized_query = normalize_text(pregunta)
    tokenized_query = normalized_query.split()
    scores = bm25.get_scores(tokenized_query)
