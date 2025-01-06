import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Función para leer y concatenar textos de múltiples archivos TXT
def read_txt_files(directory):
    text = ""
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                text += file.read() + "\n"
    return text

# Configuración
txt_directory = r'C:\Users\darkd\OneDrive - Instituto Tecnológico de Zacatepec\Archivos Semestres\9no Semestre\Residencias\ChatBot-ITZ\ChatBot-ITZ\Data\TXT'
faiss_index_file = r'C:\Users\darkd\Desktop\a\test\index.faiss'

# Leer y procesar los archivos TXT
text = read_txt_files(txt_directory)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, length_function=len)
chunks = text_splitter.split_text(text)

# Crear embeddings con Hugging Face
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Cargar o crear índice FAISS
if os.path.exists(faiss_index_file):
    knowledge_base = FAISS.load_local(faiss_index_file, embeddings)
else:
    knowledge_base = FAISS.from_texts(chunks, embeddings)
    os.makedirs(os.path.dirname(faiss_index_file), exist_ok=True)
    knowledge_base.save_local(faiss_index_file)

# Cargar los embeddings desde FAISS
index_vectors = knowledge_base.index.reconstruct_n(0, knowledge_base.index.ntotal)

# Reducir dimensiones a 3D con t-SNE
reduced_vectors = TSNE(n_components=3, random_state=42).fit_transform(index_vectors)

# Visualización 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(
    reduced_vectors[:, 0],
    reduced_vectors[:, 1],
    reduced_vectors[:, 2],
    c='blue', marker='o'
)
ax.set_title("Visualización de Embeddings en 3D")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

# Asignar tokens (nombres) a los puntos
tokens = chunks[:len(index_vectors)]  # Asegurarnos de que coincidan las longitudes

def on_hover(event):
    if event.inaxes == ax:
        cont, ind = scatter.contains(event)
        if cont and "ind" in ind and len(ind["ind"]) > 0:
            # Obtener índice del punto
            index = ind["ind"][0]
            token = tokens[index] if index < len(tokens) else "Sin token"
            print(f"Token: {token}")  # Mostrar en la terminal
        elif cont:
            print("Sin token asociado.")  # Si no hay token asociado al punto

fig.canvas.mpl_connect("motion_notify_event", on_hover)

plt.show()
