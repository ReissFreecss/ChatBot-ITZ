from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, pipeline
import torch

# Cargar el documento PDF
loader = PyMuPDFLoader("./Data/PDF/4-Gestion_Agil_de_Proyectos.pdf")
data_pdf = loader.load()

# Dividir el texto en fragmentos
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

def generate_response(question):
    # Buscar documentos relevantes
    docs = knowledge_base.similarity_search(question, 3)
    
    # Concatenar el texto de los documentos encontrados
    context = " ".join([doc.page_content for doc in docs])
    
    # Formatear el prompt usando el template
    prompt = prompt_template.format(context=context, question=question)
    
    # Generar respuesta usando GEMMA
    outputs = gemma_pipeline(
        prompt,
        max_new_tokens=350,
        add_special_tokens=True,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95
    )
    
    # Devolver la respuesta generada
    return outputs[0]["generated_text"]

# Ejemplo de uso
pregunta = "¿Que unidades lleva la gestion de proyectos?"
respuesta = generate_response(pregunta)
print(respuesta)
