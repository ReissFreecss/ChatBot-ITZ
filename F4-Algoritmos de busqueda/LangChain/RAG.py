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
txt_directory = r'C:\Users\darkd\OneDrive - Instituto Tecnológico de Zacatepec\Archivos Semestres\9no Semestre\Residencias\ChatBot-ITZ\ChatBot-ITZ\Data\TXT'

# Leer y procesar los archivos TXT
text = read_txt_files(txt_directory)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=20,
    length_function=len
)

chunks = text_splitter.split_text(text)
# Crear embeddings con el modelo Hugging Face
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Ruta para guardar el archivo FAISS
faiss_index_file = r'C:\Users\darkd\Desktop\a\test\index.faiss'

# Verificar si ya existe un archivo de índice FAISS guardado
if os.path.exists(faiss_index_file):
    # Si el archivo existe, cargar el índice FAISS
    knowledge_base = FAISS.load_local(faiss_index_file, embeddings)
else:
    # Si el archivo no existe, crearlo a partir de los chunks de texto y guardarlo
    knowledge_base = FAISS.from_texts(chunks, embeddings)
    
    # Crear el directorio si no existe
    os.makedirs(os.path.dirname(faiss_index_file), exist_ok=True)
    
    # Guardar el índice FAISS en un archivo
    knowledge_base.save_local(faiss_index_file)

# Configurar el modelo GEMMA
model = "google/gemma-2b-it"
tokenizer = AutoTokenizer.from_pretrained(model)
gemma_pipeline = pipeline(
    "text-generation",
    model=model,
    model_kwargs={"torch_dtype": torch.bfloat16, "hidden_activation": "gelu_pytorch_tanh"},
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
    docs = knowledge_base.similarity_search(question, 4)
    
    # Concatenar el texto de los documentos encontrados
    context = " ".join([doc.page_content for doc in docs])
    
    # Formatear el prompt usando el template
    prompt = prompt_template.format(context=context, question=question)
    
    # Generar respuesta usando GEMMA
    outputs = gemma_pipeline(
        prompt,
        max_new_tokens=650,
        add_special_tokens=True,
        do_sample=True,
        temperature=0.9,
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

    #TF y IDF