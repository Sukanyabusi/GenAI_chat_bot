from flask import Flask, render_template, jsonify, request
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
from langchain_pinecone import PineconeVectorStore
from src.helper import download_hugging_face_embeddings
import os
app=Flask(__name__)
load_dotenv()

PINECONE_API_KEY=os.environ.get('PINE_CONE_API_KEY')
# OPENAI_API_KEY=os.environ.get('OPENAI_API_KEY')
Groq_API_KEY=os.environ.get('Groq_API_KEY')

os.environ['PINECONE_API_KEY']=PINECONE_API_KEY
# os.environ['OPENAI_API_KEY']=OPENAI_API_KEY
os.environ['Groq_API_KEY']=Groq_API_KEY
embeddngs=download_hugging_face_embeddings()

index_name = "medicalchatbot2"
docsearch=PineconeVectorStore.from_existing_index(
    # documents=text_chunks,
    index_name=index_name,
    embedding=embeddngs)

retriever=docsearch.as_retriever(search_type='similarity',search_kwars={"k":3})

prompt=ChatPromptTemplate.from_messages([
    ('system',system_prompt),
    ('human',"{input}")
])
llm = ChatGroq(
    temperature=0.4,
    max_tokens=500,
    model_name="llama3-70b-8192"  # or another supported Groq model
)

question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


@app.route("/")
def index():
    return render_template('chatbot.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    response = rag_chain.invoke({"input": msg})
    print("Response : ", response["answer"])
    return str(response["answer"])



if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)

# response=rag_chain.invoke({"input":"what is Diabetes mellitus?"})
# print(response["answer"])