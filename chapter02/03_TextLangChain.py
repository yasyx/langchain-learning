from dotenv import load_dotenv
load_dotenv()

# from langchain_openai import OpenAI
from  langchain_ollama import OllamaLLM

llm = OllamaLLM(
    model="llama3.1",
    temperature=0.8,
    max_tokens=60,
)

res = llm.invoke("请给我的花店起个名字")

print(res)
