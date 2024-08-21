import os

from dotenv import load_dotenv
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sympy.physics.units import temperature

load_dotenv()

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

base_dir = './OneFlower'

documents = []
# 1 加载文件
for file in os.listdir(base_dir):
    file_path = os.path.join(base_dir, file)
    if file.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        documents.extend(loader.load())
    elif file.endswith(".docx"):
        print("docx:" + file)
        loader = Docx2txtLoader(file_path)
        documents.extend(loader.load())
    elif file.endswith(".txt"):
        loader = TextLoader(file_path)
        documents.extend(loader.load())

# print(documents)

# 2 将documents切分成块 以便进行嵌入和向量存储
text_spliter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=10)
chunked_documents = text_spliter.split_documents(documents)

# 3 将分割嵌入并存储在向量数据库Qdrant中

vector_store = Qdrant.from_documents(
    documents=chunked_documents,
    embedding=OpenAIEmbeddings(),
    location=":memory:",
    collection_name="my_documents"
)

# 4 准备模型 和Retrieval链

import logging
from langchain_openai import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chains import RetrievalQA

# 配置日志
logging.basicConfig()
logging.getLogger("langchain.retrievers.muti_query").setLevel(logging.INFO)

chat = ChatOpenAI(
    temperature=0,
)

# 实例化一个MultiQueryRetriever
retrievers_from_llm = MultiQueryRetriever.from_llm(retriever=vector_store.as_retriever(), llm=chat)

chain = RetrievalQA.from_chain_type(
    llm=chat,
    chain_type="stuff",
    retriever=retrievers_from_llm,
)

# 问答系统的UI实现。
from flask import Flask, render_template, request

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        question = request.form.get("question")

        result = chain({"query": question})

        return render_template('index.html', result=result)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
