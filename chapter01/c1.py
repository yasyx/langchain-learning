from dotenv import load_dotenv

load_dotenv()

from langchain_openai import ChatOpenAI


llm = ChatOpenAI()


res = llm.invoke("你是谁")

print(res)

