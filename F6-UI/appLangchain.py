from flask import Flask, request, jsonify, render_template
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, pipeline
import markdown
import torch

app = Flask(__name__)

# Configuración de directorio de documentos
directory = 'C:/Users/darkd/OneDrive - Instituto Tecnológico de Zacatepec/Archivos Semestres/9no Semestre/Administracion de redes/A proyecto/t9-py-UIFlask/normalization'

# Función para leer y concatenar textos de múltiples archivos TXT
def read_txt_files(directory):
    text = ""
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                text += file.read() + "\n"
    return text

# Leer y procesar los archivos TXT
text = read_txt_files(directory)

# Dividir texto en fragmentos manejables
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=10, length_function=len)
chunks = text_splitter.split_text(text)

# Crear embeddings y el almacén de búsqueda FAISS
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
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
custom_template = """Responde la pregunta del usuario relacionada con las redes de computadoras, puedes usar esta informacion:.

**Contexto**: {context}
**Pregunta**: {question}

**Respuesta útil**:
"""
prompt_template = PromptTemplate(template=custom_template, input_variables=["context", "question"])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def query():
    data = request.json
    query = data.get('query')

    if not query:
        return jsonify({'error': 'No query provided'}), 400

    # Buscar documentos relevantes usando FAISS y LangChain
    docs = knowledge_base.similarity_search(query, k=2)
    if not docs:
        return jsonify({'error': 'No documents found'}), 404

    # Concatenar el texto de los documentos encontrados
    context = " ".join([doc.page_content for doc in docs])

    # Formatear el prompt usando el template
    prompt = prompt_template.format(context=context, question=query)

    # Generar respuesta usando GEMMA
    outputs = gemma_pipeline(
        prompt,
        max_new_tokens=550,
        add_special_tokens=True,
        do_sample=True,
        temperature=0.5,
        top_k=50,
        top_p=0.95
    )
    
    # Obtén la respuesta del modelo
    summary = outputs[0]["generated_text"]

    # Convierte el resumen a HTML para mejor legibilidad
    summary_html = markdown.markdown(summary)

    return jsonify({
        #'query': query,
        #'context': context,
        'summary': summary_html  # Devuelve el resumen en HTML
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)