# Integracion de langchain con LLM.
El proceso de instalación y configuración de un pipeline de procesamiento de lenguaje natural utilizando LangChain, FAISS, y el modelo GEMMA. El código permite cargar un documento PDF, dividir su contenido en fragmentos, crear una base de conocimiento mediante embeddings, y finalmente generar respuestas a preguntas basadas en ese contenido.

## Explicacion de codigo.
### Instalación de dependencias
Se requiere la instalación de las siguientes bibliotecas para ejecutar:
```python
pip -q install PyPDF2==3.0.1
pip -q install langchain==0.0.324
pip -q install faiss-cpu==1.7.4
pip -q install openai==0.28.1
pip -q install python-dotenv==1.0.0
pip -q install sentence_transformers==2.2.2
```
# Importación de dependencias
```python
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, pipeline
import torch
```
* `PyMuPDFLoader` permite leer el contenido de un archivo PDF y convertirlo en un formato que pueda ser procesado por otros componentes de LangChain
* `langchain.text_splitter` Esta clase divide el texto en fragmentos más pequeños o "chunks".
* `langchain.embeddings` Esta clase se encarga de convertir el texto en representaciones numéricas conocidas como embeddings. r búsquedas de similitud o para alimentar modelos de lenguaje.
* `from langchain.vectorstores import FAISS` es una biblioteca de Facebook AI que se utiliza para la búsqueda de similitud rápida en grandes colecciones de vectores.
* `promptTemplate` permite definir plantillas de prompts que se utilizan para interactuar con modelos de lenguaje.
### Cargar el Documento PDF
```python
loader = PyMuPDFLoader("C:/Users/luisv/OneDrive/Documentos/Verano Cientifico/Progrmacion/tprueba-py-model/test2.pdf")
data_pdf = loader.load()
```
El documento PDF se carga utilizando `PyMuPDFLoader`, que permite leer y procesar archivos PDF.

### Dividir el Texto en Fragmentos.
```python
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500, length_function=len)
chunks = text_splitter.split_documents(data_pdf)
```
El contenido del PDF se divide en fragmentos utilizando `RecursiveCharacterTextSplitter`, lo que facilita el procesamiento de grandes cantidades de texto. 
* `chunk_size=2000`: Este parámetro define el tamaño máximo de cada fragmento en términos de número de caracteres. Aquí, cada fragmento tendrá un máximo de 2000 caracteres. Esto significa que el texto se dividirá en partes manejables que pueden ser procesadas individualmente.
* `chunk_overlap=500`: Este parámetro define cuántos caracteres se solaparán entre un fragmento y el siguiente. En este caso, hay un solapamiento de 500 caracteres entre fragmentos consecutivos.
* `length_function=len`Este parámetro permite definir la función que se utilizará para medir la longitud del texto. En este caso, se está utilizando la función len, que cuenta el número de caracteres.
* `split_documents(data_pdf)`:  Este método aplica la lógica de división a los documentos cargados que contiene el PDF.

### Crear Embeddings y Base de Conocimiento
```python
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
knowledge_base = FAISS.from_texts([chunk.page_content for chunk in chunks], embeddings)
```
Generar representaciones numéricas (embeddings) de los fragmentos de texto (chunks) utilizando el modelo `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`. Estas representaciones capturan el significado semántico de cada fragmento, permitiendo comparaciones entre ellos. `knowledge_base` almacena los embeddings generados, utilizando `FAISS` para permitir la búsqueda eficiente de fragmentos de texto similares. Esto facilita encontrar los fragmentos más relevantes en respuesta a una consulta.
