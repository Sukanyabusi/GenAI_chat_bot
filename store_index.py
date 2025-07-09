from src.helper import load_pdf_data,text_split,download_hugging_face_embeddings
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
load_dotenv()
import os

PINECONE_API_KEY=os.environ.get('PINE_CONE_API_KEY')
# OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY')
Groq_API_KEY=os.environ.get('Groq_API_KEY')

os.environ['PINECONE_API_KEY']=PINECONE_API_KEY
# os.environ['OPENAI_API_KEY']=OPENAI_API_KEY
os.environ['Groq_API_KEY']=Groq_API_KEY


extracted_data=load_pdf_data(data='Data/')
text_chunks=text_split(extracted_data)
embeddngs=download_hugging_face_embeddings()

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medicalchatbot2"
pc.create_index(
    name=index_name,
    dimension=384,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)

doc_search=PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddngs
    

)