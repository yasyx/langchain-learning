from dotenv import load_dotenv
load_dotenv()

from langchain_openai import OpenAI

llm = OpenAI(
    model="gpt-3.5-turbo-instruct",
    temperature=0.8,
    max_tokens=60,
)

res = llm.invoke("请给我的花店起个名字")

print(res)
