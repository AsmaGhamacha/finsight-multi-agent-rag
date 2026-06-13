import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

load_dotenv()

PDF_FOLDER = "data/raw"
FAISS_INDEX_PATH = "data/processed/faiss_index"

def load_and_split_pdfs(folder_path):
    all_documents = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            print(f"Loading {filename}...")
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()
            all_documents.extend(documents)
    print(f"\nTotal pages loaded: {len(all_documents)}")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(all_documents)
    print(f"Total chunks created: {len(chunks)}")
    return chunks

def build_vectorstore(chunks):
    print("\nBuilding vector store...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embeddings)
    os.makedirs(FAISS_INDEX_PATH, exist_ok=True)
    vectorstore.save_local(FAISS_INDEX_PATH)
    print(f"Vector store saved to {FAISS_INDEX_PATH}")
    return vectorstore

def get_retriever_chain():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True,
    )
    return chain

if __name__ == "__main__":
    if not os.path.exists(FAISS_INDEX_PATH):
        print("No existing vector store found. Building from PDFs...")
        chunks = load_and_split_pdfs(PDF_FOLDER)
        build_vectorstore(chunks)
    else:
        print("Existing vector store found. Skipping rebuild.")

    print("\nLoading retriever chain...")
    chain = get_retriever_chain()

    question = "What was the ECB's main monetary policy decision in 2023?"
    print(f"\nQuestion: {question}")
    result = chain.invoke({"query": question})
    print(f"\nAnswer: {result['result']}")