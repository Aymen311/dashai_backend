from flask import Flask, request, jsonify
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import CharacterTextSplitter
import os
import pinecone 
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain import OpenAI, LLMChain, PromptTemplate
from flask_cors import CORS

import traceback

from langchain.memory import VectorStoreRetrieverMemory

app = Flask(__name__)
cors = CORS(app, resources={r"*": {"origins": "*"}})
PINECONE_API_KEY = "YOUR API KEY"
PINECONE_ENV = "YOUR PINECONE_ENV"
OPENAI_API_KEY = "YOUR API KEY"
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

doc_db = None
file_path = 'files/file.csv'

def load_csv_and_embed(file_path):
    global doc_db
    
    loader = CSVLoader(file_path=file_path, encoding="utf-8")
    data = loader.load()

    text_splitter = CharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=0
    )
    docs_split = text_splitter.split_documents(data)

    embeddings = OpenAIEmbeddings()
    pinecone.init(
        api_key=PINECONE_API_KEY,
        environment=PINECONE_ENV
    )

    doc_db = Pinecone.from_documents(
        docs_split, 
        embeddings, 
        index_name='reviews-idx'
    )
    
load_csv_and_embed(file_path)

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files.get('file', '')
    print(file)
    if file:
        file_path = 'files/file.csv'  # Provide the appropriate path to save the file
        file.save(file_path)

        load_csv_and_embed(file_path)

        return jsonify({'message': 'CSV file uploaded and processed successfully.'})
    else:


        return jsonify({'message': 'No file was uploaded.'})

@app.route('/predict', methods=['POST'])
def predict():
    human_input = request.json['human_input']
    print(human_input)
    template = """
        You are an E-commerce AI assistant named Robby. The user gives you access to data about his client reviews, your job is to discuss with the client about the different insights he could get from those reviews,
        content is represented by the following pieces of context, use them to answer the question at the end.
        If you don't know the answer, just say you don't know. Do NOT try to make up an answer.
        If the question is not related to the context, politely respond that you are tuned to only answer questions that are related to the context.
        Use as much detail as possible when responding.

        context: {history}
        =========
        Human: {human_input}
        Assistant:"""

    prompt = PromptTemplate(input_variables=["history", "human_input"], template=template)

    llm = OpenAI(temperature=0)
    retriever = doc_db.as_retriever()
    memory = VectorStoreRetrieverMemory(retriever=retriever)

    chatgpt_chain = LLMChain(
        llm=llm,
        prompt=prompt,
        verbose=False,
        memory=memory,
    )

    output = chatgpt_chain.predict(human_input=human_input)

    return jsonify({'output': output})

@app.errorhandler(Exception)
def handle_error(error):
    trace = traceback.format_exc()
    print(trace)

    return jsonify({'error': 'An error occurred during processing.'}), 500



if __name__ == '__main__':
    app.run()
