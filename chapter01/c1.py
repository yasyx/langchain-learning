from dotenv import load_dotenv

load_dotenv()

from langchain_community.llms import Ollama

llm = Ollama(model="llama3.1")

res = llm.invoke("请给我的花店起个名字")

#
# from langchain_openai import ChatOpenAI
#
#
# llm = ChatOpenAI()
#
#
# res = llm.invoke("请给我的花店起个名字")

print(res)

