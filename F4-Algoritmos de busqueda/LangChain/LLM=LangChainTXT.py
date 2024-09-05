import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, pipeline
import torch

# Función para leer y concatenar textos de múltiples archivos TXT
def read_txt_files(directory):
    text = ""
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                text += file.read() + "\n"
    return text

# Ruta del directorio donde están los archivos TXT
txt_directory = r'C:\Users\Carlos\OneDrive - Instituto Tecnológico de Zacatepec\Archivos Semestres\9no Semestre\Residencias\ChatBot-ITZ\ChatBot-ITZ\Data\TXT'

# Leer y procesar los archivos TXT
text = read_txt_files(txt_directory)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=10,
    length_function=len
)
chunks = text_splitter.split_text(text)

# Crear embeddings y base de conocimiento
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
knowledge_base = FAISS.from_texts(chunks, embeddings)

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

Respuesta útil:

"""


# Crear el prompt template
prompt_template = PromptTemplate(template=custom_template, input_variables=["context", "question"])


def generate_response(question):
    # Buscar documentos relevantes
    docs = knowledge_base.similarity_search(question, 1)
    #print(docs)
        
    # Concatenar el texto de los documentos encontrados
    context = " ".join([doc.page_content for doc in docs])
    
    # Formatear el prompt usando el template
    prompt = prompt_template.format(context=context, question=question)
    
    # Generar respuesta usando GEMMA
    outputs = gemma_pipeline(
        prompt,
        max_new_tokens=550,
        add_special_tokens=True,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95
    )
    
     # Devolver la respuesta generada
    return outputs[0]["generated_text"]

# Ejemplo de uso
while True:
    pregunta = input("Pregunta (escribe 'salir' para terminar): ")
    if pregunta.lower() == "salir":
        break
    respuesta = generate_response(pregunta)
    print(respuesta)
