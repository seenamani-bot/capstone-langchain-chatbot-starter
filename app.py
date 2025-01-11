from flask import Flask, render_template
from flask import request, jsonify, abort

from langchain.llms import Cohere
from langchain import PromptTemplate, LLMChain
from langchain.chains import SequentialChain
from langchain.chains import RetrievalQA
from langchain.embeddings import CohereEmbeddings
from langchain.vectorstores import Chroma


import os

app = Flask(__name__)

def load_db():
    try:
        embeddings = CohereEmbeddings(cohere_api_key=os.environ["COHERE_API_KEY"])
        vectordb = Chroma(persist_directory='db', embedding_function=embeddings)
        qa = RetrievalQA.from_chain_type(
            llm=Cohere(),
            chain_type="refine",
            retriever=vectordb.as_retriever(),
            return_source_documents=True
        )
        return qa
    except Exception as e:
        print("Error:", e)

qa = load_db()

def answer_from_knowledgebase(message):
    res = qa({"query": message})
    return res['result']

def search_knowledgebase(message):
   # Search the knowledgebase and generate a response
  res = qa({"query": message})
  sources = ""
  for count, source in enumerate(res['source_documents'],1):
    sources += "Source " + str(count) + "\n"
    sources += source.page_content + "\n"

def answer_as_chatbot(message):
    template = """Question: {question}

   Answer as if you are an expert athlete dietition"""

    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm = Cohere(cohere_api_key=os.environ["COHERE_API_KEY"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    res = llm_chain.run(message)
    return res 


   
@app.route('/kbanswer', methods=['POST'])
def kbanswer():
   message = request.json['message']
   response_message_knowlwdgebase = answer_from_knowledgebase(message)
    # Return the response as JSON
   return jsonify({'message': response_message_knowlwdgebase}), 200

@app.route('/search', methods=['POST'])
def search():    
    # Search the knowledgebase and generate a response
    message = request.json['message']
    search_response_message = search_knowledgebase(message)
    # Return the response as JSON
    return jsonify({'message': search_response_message}), 200
    # (call search_knowledgebase())
    
    # Return the response as JSON
    return

@app.route('/answer', methods=['POST'])
def answer():
    message = request.json['message']
    
    # Generate a response as an expert Python developer
    response_message = answer_as_chatbot(message)
    
    # Return the response as JSON
    return jsonify({'message': response_message}), 200

@app.route("/")
def index():
    return render_template("index.html", title="DietBot")

if __name__ == "__main__":
    app.run()