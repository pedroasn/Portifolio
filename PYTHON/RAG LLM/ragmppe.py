import os

from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ==============================
# CONFIG
# ==============================
DEEPSEEK_API_KEY = ""
PASTA_PDFS = "./documents"
CAMINHO_INDICE = "./indice_faiss"

# ==============================
#CLIENTE DEEPSEEK
# ==============================
client = OpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com"
)

# ==============================
# 1. CARREGAR PDFs
# ==============================
def carregar_pdfs(pasta):
    documentos = []
    for arquivo in os.listdir(pasta):
        if arquivo.endswith(".pdf"):
            caminho = os.path.join(pasta, arquivo)
            loader = PyPDFLoader(caminho)
            documentos.extend(loader.load())
    return documentos

# ==============================
# 2. DIVIDIR TEXTO (CHUNKS)
# ==============================
def dividir_documentos(documentos):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_documents(documentos)

# ==============================
# 3. CRIAR OU CARREGAR ÍNDICE
# ==============================
def criar_ou_carregar_indice():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    if os.path.exists(CAMINHO_INDICE):
        print("📦 Carregando índice existente...")
        db = FAISS.load_local(CAMINHO_INDICE, embeddings, allow_dangerous_deserialization=True)
    else:
        print("📄 Processando PDFs...")
        documentos = carregar_pdfs(PASTA_PDFS)
        print(f"→ {len(documentos)} páginas carregadas")

        docs_divididos = dividir_documentos(documentos)
        print(f"→ {len(docs_divididos)} chunks gerados")

        db = FAISS.from_documents(docs_divididos, embeddings)
        db.save_local(CAMINHO_INDICE)
        print("💾 Índice salvo!")

    return db

# ==============================
# 4. GERAR RESPOSTA (DEEPSEEK)
# ==============================
def gerar_resposta(pergunta, contexto):
    contexto = contexto[:12000]  # evita limite de token

    prompt = f"""
Use o contexto abaixo para responder a pergunta.
Se não encontrar a resposta, diga claramente que não encontrou.

Contexto:
{contexto}

Pergunta: {pergunta}
"""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "Você responde perguntas com base em documentos."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content

# ==============================
# 5. FUNÇÃO RAG
# ==============================
def perguntar(db, pergunta):
    docs_relevantes = db.similarity_search(pergunta, k=5)

    contexto = "\n\n".join([doc.page_content for doc in docs_relevantes])

    resposta = gerar_resposta(pergunta, contexto)

    return resposta

# ==============================
# 6. LOOP INTERATIVO
# ==============================
def main():
    db = criar_ou_carregar_indice()

    print("\n🤖 RAG com DeepSeek pronta! (digite 'sair' para encerrar)\n")

    while True:
        pergunta = input("❓ Pergunta: ")

        if pergunta.lower() in ["sair", "exit", "quit"]:
            break

        resposta = perguntar(db, pergunta)

        print("\n🧠 Resposta:")
        print(resposta)
        print("\n" + "-"*50 + "\n")

# ==============================
# EXECUTAR
# ==============================
if __name__ == "__main__":
    main()