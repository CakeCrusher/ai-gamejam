from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import chromadb
from chromadb.config import Settings
import openai
from chromadb.utils import embedding_functions
import uuid
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
CHROMA_API_IMPL = os.getenv('CHROMA_API_IMPL')
CHROMA_SERVER_HOST = os.getenv('CHROMA_SERVER_HOST')
CHROMA_SERVER_HTTP_PORT = int(os.getenv('CHROMA_SERVER_HTTP_PORT'))
PORT = int(os.getenv('PORT'))
DEVELOPMENT = os.getenv('DEVELOPMENT') == 'true'

app = Flask(__name__)
CORS(app)

client = chromadb.Client(Settings(chroma_api_impl=CHROMA_API_IMPL,
                            chroma_server_host=CHROMA_SERVER_HOST,
                            chroma_server_http_port=CHROMA_SERVER_HTTP_PORT))
openai.api_key = OPENAI_API_KEY
openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=OPENAI_API_KEY,
                model_name="text-embedding-ada-002"
            )
collection = client.get_or_create_collection(
    name="test",
    embedding_function=openai_ef
)


# create an /add POST endpoint that will recieve a json like: {documents: ["doc1", "doc2", "doc3"]}
@app.route('/add', methods=['POST'])
def add():
    data = request.get_json()
    documents = data['documents']
    random_ids = [str(uuid.uuid4()) for _ in range(len(documents))]
    collection.add(
        documents=documents,
        ids=random_ids
    )
    return jsonify({"success": True})


    
    
# now create a /query post endpoint that will recieve a json like: {query: "doc1"}
@app.route('/query', methods=['POST'])
def query():
    data = request.get_json()
    query = data['query']
    results = collection.query(
        query_texts=[query],
        n_results=1
    )
    return jsonify({"results": results})



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=PORT, debug=DEVELOPMENT)