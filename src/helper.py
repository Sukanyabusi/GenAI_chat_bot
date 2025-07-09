from langchain.document_loaders import  PyPDFLoader,DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
# Data Loading
def load_pdf_data(data):
    loader=DirectoryLoader(data,glob='*.pdf',loader_cls=PyPDFLoader)
    documents=loader.load()
    return documents

# extracted_data
def text_split(extracted_data):
    text_spliter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=20)
    text_chunks=text_spliter.split_documents(extracted_data)
    return text_chunks
# Embeddings
def download_hugging_face_embeddings():
    embeddngs=HuggingFaceBgeEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    return embeddngs