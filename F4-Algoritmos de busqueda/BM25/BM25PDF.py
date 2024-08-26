import os
from rank_bm25 import BM25Okapi
import nltk
import string
from nltk.corpus import stopwords
from unidecode import unidecode
import torch
from transformers import AutoTokenizer, pipeline
import fitz  # PyMuPDF para leer PDFs

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

# Carga de documentos PDF
directory = './Data/PDF'
documents = []
filenames = []

# Leer y normalizar documentos PDF
for filename in os.listdir(directory):
    if filename.endswith('.pdf'):
        filepath = os.path.join(directory, filename)
        with fitz.open(filepath) as pdf_doc:
            text = ""
            for page in pdf_doc:
                text += page.get_text()
            normalized_text = normalize_text(text)
            documents.append(normalized_text)
            filenames.append(filename)

print("-------------------------------------------------------")
print(f"Documentos leídos:  {len(documents)}")
print("-------------------------------------------------------")

# Tokenización de documentos
tokenized_documents = [doc.split() for doc in documents]
bm25 = BM25Okapi(tokenized_documents)

# Consulta BM25
query = "Dime dónde se encuentra la universidad"
normalized_query = normalize_text(query)
tokenized_query = normalized_query.split()
scores = bm25.get_scores(tokenized_query)

# Combinar nombres de archivos y puntajes
documents_with_scores = list(zip(filenames, scores))
documents_with_scores.sort(key=lambda x: x[1], reverse=True)
n_top_documents = 3
top_documents = documents_with_scores[:n_top_documents]
print("Documentos clave:")
for filename, score in top_documents:
    print(f"{filename}: {score:.3f}")
print("-------------------------------------------------------")

# Carga de Gemma2
model = "google/gemma-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model)
gemma_pipeline = pipeline(
    "text-generation",
    model=model,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device="cuda",
)

# Generación de resúmenes con Gemma2
for filename, score in top_documents:
    with fitz.open(os.path.join(directory, filename)) as pdf_doc:
        doc_content = ""
        for page in pdf_doc:
            doc_content += page.get_text().strip()

    messages = [
        {"role": "user", "content": f"Resumen del documento sobre {query}: {doc_content}"}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = gemma_pipeline(
        prompt, max_new_tokens=256, 
        add_special_tokens=True, 
        do_sample=True, temperature=0.7, 
        top_k=50, top_p=0.95)
    
    summary = outputs[0]["generated_text"][len(prompt):]
    print("--------------------------------------------------------------------------------------------------------------")
    print(f"Resumen del documento {filename}")
    print("--------------------------------------------------------------------------------------------------------------")
    print(summary)
